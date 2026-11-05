"""カテゴリを判定する head への、未知語の依存度を測る
（dependence-on-category-judging-heads measure-dependence --model）。

未知語の獲得の各 run（run-1〜run-8）の最後の重みで、語義の追加で「カテゴリを判定する」と特定した
head（test-attribute-directions の attribute_judging_heads の、属性が category で3つの基準を満たす
もの。同じモデル）をまとめて無効化し、課題2のカテゴリの問題（中立文脈の21条件）で、正解のカテゴリ
（その run で割り当てた名詞のカテゴリ）の対数確率がどれだけ下がるかを測る。
- 無効化（ablation が category-direction）: 文の中の語の最後のトークンでの各 head の出力の、
  カテゴリの方向の成分を、自然な語での平均の成分に置き換える（x − P x + P μ）
- 比較の基準: 同じ層の無作為な head（train-attribute-probes で選んだもの）のカテゴリの方向
  （random-head）と、同じ head の同じ次元数の無作為な方向（random-direction。
  test-attribute-directions と同じ方向）で、同じ無効化をする
- d_未知: 未知語での低下（21条件の平均）。d_既知: 同じ問題の語を、割り当てた名詞に替えたときの低下
- 依存度 = d_未知 / d_既知。d_既知 が MINIMUM_KNOWN_DROP 以下の組（その head が名詞にも効いていない
  組）は、依存度を出さない（usable が偽）
- 無効化する前の、未知語と名詞の課題2の正解率（21条件での割合）も記録する
カテゴリを判定する head がなければ、何も測らずに空の表を書く。
出力は outputs/results/dependence-on-category-judging-heads/<モデル>/dependence.parquet（run ×
未知語 × 無効化の種類ごと）。
"""

import argparse
import gc
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pyarrow as arrow
import pyarrow.parquet as parquet
import torch

from lexical_acquisition_mechanisms import (
    attribute_direction_patching,
    attribute_probes,
    run_settings,
    significant_heads_and_mlps,
    task_questions,
    unknown_word_assignments,
    word_finalization,
)
from lexical_acquisition_mechanisms.attribute_directions import attribute_basis, random_basis
from lexical_acquisition_mechanisms.attribute_judgments import AttributeJudgingHead
from lexical_acquisition_mechanisms.attribute_probes import (
    ATTRIBUTES,
    adopted_heads,
    baseline_heads,
    read_attribute_probes,
)
from lexical_acquisition_mechanisms.command_stopped import CommandStopped
from lexical_acquisition_mechanisms.component_patching import (
    SubspacePatch,
    subspace_patched_log_probabilities,
)
from lexical_acquisition_mechanisms.csv_tables import read_rows
from lexical_acquisition_mechanisms.language_model import LanguageModel
from lexical_acquisition_mechanisms.logistic_probes import LogisticProbe
from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.settings import ModelSource, Settings
from lexical_acquisition_mechanisms.task_prompts import (
    choice_token_ids,
    encode_prompt_with_word_spans,
)
from lexical_acquisition_mechanisms.training_checkpoints import FINAL_DIRECTORY
from lexical_acquisition_mechanisms.unknown_word_assignments import UnknownWordAssignment
from lexical_acquisition_mechanisms.unknown_word_indicators import checkpoint_states
from lexical_acquisition_mechanisms.word_selection import SelectedWord

EXPERIMENT = "dependence-on-category-judging-heads"
RESULTS_DIRECTORY = Path("outputs/results") / EXPERIMENT
ABLATIONS = ("category-direction", "random-head", "random-direction")
# 名詞での低下がこれ以下（自然対数の単位）なら、依存度を出さない
MINIMUM_KNOWN_DROP = 0.05
BATCH_SIZE = 16


def dependence_path(model: str) -> Path:
    return RESULTS_DIRECTORY / model / "dependence.parquet"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="settings.yaml の models の名前")


def measure_dependence(settings: Settings, arguments: argparse.Namespace) -> None:
    model = arguments.model
    command = f"{EXPERIMENT} measure-dependence --model {model}"
    directory = run_settings.CHECKPOINTS_DIRECTORY / unknown_word_assignments.EXPERIMENT / model
    checkpoints = {
        run: directory / run / FINAL_DIRECTORY for run in unknown_word_assignments.run_names()
    }
    missing = [str(path) for path in checkpoints.values() if not (path / "config.json").exists()]
    if missing:
        raise CommandStopped(
            f"学習を終えていない run があります（{missing}）。先に学習してください。",
            rerun_from=command,
        )
    _, _, judging_path, directions_path = attribute_direction_patching.output_paths(model)
    tested_by = f"sense-addition test-attribute-directions --model {model}"
    inputs = {
        judging_path: tested_by,
        directions_path: tested_by,
        significant_heads_and_mlps.output_paths(model)[2]: (
            f"sense-addition select-significant-heads-and-mlps --model {model}"
        ),
        attribute_probes.output_paths(model)[0]: (
            f"sense-addition train-attribute-probes --model {model}"
        ),
        task_questions.UNKNOWN_WORD_QUESTIONS_PATH: task_questions.COMMAND,
        unknown_word_assignments.ASSIGNMENTS_PATH: unknown_word_assignments.COMMAND,
        word_finalization.OUTPUT_PATH: word_finalization.COMMAND,
    }
    step = Step(
        command,
        inputs=inputs,
        outputs=[dependence_path(model)],
        settings={
            "checkpoints": checkpoint_states(list(checkpoints.values())),
            "seed": settings.words.seed,
            "minimum_known_drop": MINIMUM_KNOWN_DROP,
        },
    )
    run_unless_unchanged(
        step,
        lambda: write_dependence(
            checkpoints,
            lambda source: LanguageModel.load(source, batch_size=BATCH_SIZE),
            tuple(inputs),
            dependence_path(model),
            settings.words.seed,
        ),
        arguments.recompute,
    )


def write_dependence(
    checkpoints: dict[str, Path],
    load_model: Callable[[ModelSource], LanguageModel],
    input_paths: Sequence[Path],
    output: Path,
    seed: int,
) -> None:
    judging_path, directions_path, classes_path, probes_path, *rest = input_paths
    questions_path, assignments_path, words_path = rest
    heads = sorted(
        (row.layer, row.head)
        for row in read_rows(judging_path, AttributeJudgingHead)
        if row.attribute == "category" and row.judging
    )
    rows: list[dict[str, object]] = []
    if heads:
        directions = np.load(directions_path)
        probes = read_attribute_probes(probes_path)
        adopted = adopted_heads(classes_path)
        questions = parquet.read_table(questions_path).to_pylist()
        words = {word.word: word for word in read_rows(words_path, SelectedWord)}
        for run, checkpoint in checkpoints.items():
            model = load_model(ModelSource(repository=str(checkpoint)))
            baselines = baseline_heads(adopted, model.model.config.num_attention_heads, seed)
            ablations = _ablations(heads, baselines, directions, probes, seed)
            run_questions = [
                question
                for question in questions
                if question["run"] == run and question["task"] == "category"
            ]
            rows += _measure_run(model, run, run_questions, ablations, words)
            # 次の run の重みを読む前に、この重みを GPU から消す
            del model
            gc.collect()
            torch.cuda.empty_cache()
        assignments = {
            (row.run, row.unknown_word): row
            for row in read_rows(assignments_path, UnknownWordAssignment)
        }
        for row in rows:
            assignment = assignments[row["run"], row["unknown_word"]]
            row |= {
                "noun_category": assignment.noun_category,
                "token_count": assignment.token_count,
                "spelling_closeness": assignment.spelling_closeness,
                "exposure": assignment.exposure,
            }
    output.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_table(arrow.Table.from_pylist(rows), output)


def _ablations(
    heads: Sequence[tuple[int, int]],
    baselines: dict[tuple[int, int], tuple[int, int]],
    directions: Mapping[str, np.ndarray],
    probes: dict[tuple[int, int, str], LogisticProbe],
    seed: int,
) -> dict[str, tuple[SubspacePatch, ...]]:
    """無効化の種類ごとの、まとめて行う入れ替え（置き換える値は、自然な語での平均）。"""
    chosen: dict[str, list[SubspacePatch]] = {name: [] for name in ABLATIONS}
    for layer, head in heads:
        basis = directions[f"{layer}-{head}-category-basis"]
        mean = directions[f"{layer}-{head}-category-mean"][None, :]
        chosen[ABLATIONS[0]].append(SubspacePatch(layer, head, basis, mean))
        other = baselines.get((layer, head))
        if other is not None and (*other, "category") in probes:
            probe = probes[(*other, "category")]
            patch = SubspacePatch(*other, attribute_basis(probe), probe.mean[None, :])
            chosen[ABLATIONS[1]].append(patch)
        # test-attribute-directions と同じ無作為な方向
        random_numbers = np.random.default_rng([seed, layer, head, ATTRIBUTES.index("category")])
        directions_basis = random_basis(basis.shape[0], basis.shape[1], random_numbers)
        chosen[ABLATIONS[2]].append(SubspacePatch(layer, head, directions_basis, mean))
    return {name: tuple(patches) for name, patches in chosen.items()}


def _measure_run(
    model: LanguageModel,
    run: str,
    questions: Sequence[dict[str, object]],
    ablations: dict[str, tuple[SubspacePatch, ...]],
    words: dict[str, SelectedWord],
) -> list[dict[str, object]]:
    """1つの run の、未知語 × 無効化の種類ごとの低下（条件で平均）と依存度。"""
    drops: dict[tuple[str, str], dict[str, list[np.ndarray]]] = {}
    for question in questions:
        candidates = choice_token_ids(question["choices"], model.tokenizer)
        correct = list(question["choices"]).index(question["answer"])
        key = (question["unknown_word"], question["noun"])
        for version in ("unknown", "known"):
            word = words[key[0] if version == "unknown" else key[1]]
            token_ids, spans = encode_prompt_with_word_spans(
                question["prompt_template"],
                [int(token) for token in word.token_ids.split()],
                model.tokenizer,
            )
            values = subspace_patched_log_probabilities(
                model, token_ids, [spans[0][-1]], [(), *ablations.values()], candidates
            )
            measured = drops.setdefault(key, {}).setdefault(version, [])
            # 先頭は無効化する前に正解か、続く3つは無効化の種類ごとの正解の対数確率の低下
            measured.append(
                np.array(
                    [values[0].argmax() == correct, *(values[0, correct] - values[1:, correct])]
                )
            )
    rows = []
    for (unknown_word, noun), versions in drops.items():
        unknown, known = (np.mean(versions[version], axis=0) for version in ("unknown", "known"))
        for index, ablation in enumerate(ABLATIONS, start=1):
            usable = bool(known[index] > MINIMUM_KNOWN_DROP)
            rows.append(
                {
                    "run": run,
                    "unknown_word": unknown_word,
                    "noun": noun,
                    "ablation": ablation,
                    "unknown_drop": float(unknown[index]),
                    "known_drop": float(known[index]),
                    "dependence": float(unknown[index] / known[index]) if usable else float("nan"),
                    "usable": usable,
                    "unknown_accuracy": float(unknown[0]),
                    "known_accuracy": float(known[0]),
                    "conditions": len(versions["unknown"]),
                }
            )
    return rows
