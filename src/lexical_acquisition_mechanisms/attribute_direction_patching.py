"""属性の方向の成分を入れ替えて、語義と属性の判定への効果を測る
（sense-addition test-attribute-directions --model の測定）。

手順2で採用した各 head と属性（カテゴリ・品詞）ごとに、属性のプローブ（train-attribute-probes）の
重みが張る部分空間（属性の方向）の成分だけを入れ替える。
- run: 足す語義の数が最大（4）の、3つの種類の run の最後の重み
- 最小対: 手順1で2つの語義とも使えるとした最小対から、run ごとに無作為に TESTED_PAIRS 組（両方向）
- 入れ替え: 問題(s_X) を処理するとき、文の中のターゲット語の最後のトークンでの head の出力の、
  属性の方向の成分を、問題(s_Y) での成分に置き換える（patch が attribute）
- 比較の基準: 同じ層の無作為な head（train-attribute-probes で選んだもの）の属性の方向
  （random-head）と、同じ head の同じ次元数の無作為な方向（random-direction）で、同じ入れ替えをする
- 測るもの（metric_change は、s_Y の側への動き）
  - 課題1（sense）: m の変化 m − m_入れ替え後（m は、s_X と s_Y の語義の選択肢の対数確率の差）と、
    s_Y の語義を選ぶようになったか（switched。全ての選択肢のうち、s_Y の語義の対数確率が最も大きい）
  - 課題2（その run の種類の課題。名詞を足す条件はカテゴリ、形容詞は品詞）: s_Y の正解の対数確率の
    増え方と、答えが s_Y の正解になったか
出力は outputs/results/sense-addition/<モデル>/attribute_patching.parquet。続けて、属性を判定する
head を3つの基準で決めて保存する（attribute_judgments）。
"""

import argparse
import gc
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as arrow
import pyarrow.parquet as parquet
import torch

from lexical_acquisition_mechanisms import (
    attribute_probes,
    evaluation_finalization,
    pretrained_model_checks,
    sense_accuracy,
    significant_heads_and_mlps,
    word_finalization,
)
from lexical_acquisition_mechanisms.attribute_directions import attribute_basis, random_basis
from lexical_acquisition_mechanisms.attribute_judgments import write_attribute_judgments
from lexical_acquisition_mechanisms.attribute_probes import (
    ATTRIBUTES,
    adopted_heads,
    baseline_heads,
    read_attribute_probes,
)
from lexical_acquisition_mechanisms.component_adoption import LARGEST
from lexical_acquisition_mechanisms.component_patching import (
    SubspacePatch,
    subspace_patched_log_probabilities,
)
from lexical_acquisition_mechanisms.csv_tables import read_rows
from lexical_acquisition_mechanisms.head_and_mlp_effects import DIRECTIONS, pair_inputs
from lexical_acquisition_mechanisms.language_model import LanguageModel
from lexical_acquisition_mechanisms.logistic_probes import LogisticProbe
from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.sense_addition_assignments import KINDS
from lexical_acquisition_mechanisms.settings import ModelSource, Settings
from lexical_acquisition_mechanisms.task_prompts import choice_token_ids
from lexical_acquisition_mechanisms.unknown_word_indicators import checkpoint_states
from lexical_acquisition_mechanisms.word_selection import SelectedWord

TESTED_PAIRS = 100
BATCH_SIZE = 16
TESTED_RUNS = tuple(f"{kind}-{LARGEST}" for kind in KINDS)
PATCHES = ("attribute", "random-head", "random-direction")


@dataclass(frozen=True)
class PlannedPatch:
    """1つの入れ替え。調べる head と属性と、実際に入れ替える head と部分空間。"""

    head: tuple[int, int]
    attribute: str
    patch: str
    patched_head: tuple[int, int]
    basis: np.ndarray


def output_paths(model: str) -> tuple[Path, Path, Path, Path]:
    directory = sense_accuracy.RESULTS_DIRECTORY / model
    return (
        directory / "attribute_patching.parquet",
        directory / "attribute_direction_tests.csv",
        directory / "attribute_judging_heads.csv",
        directory / "attribute_judging_heads.npz",
    )


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="settings.yaml の models の名前")


def test_attribute_directions(settings: Settings, arguments: argparse.Namespace) -> None:
    model = arguments.model
    command = f"sense-addition test-attribute-directions --model {model}"
    checkpoints = sense_accuracy.final_checkpoints(model, command)
    tested = {run: checkpoints[run] for run in TESTED_RUNS}
    # 課題1は必ず使う。使えない課題2（カテゴリ・品詞）は、その測定だけを飛ばす
    pretrained_model_checks.require_usable_tasks(model, ["sense"], command)
    tasks = pretrained_model_checks.usable_tasks(model)
    skipped = sorted({"category", "part-of-speech"} - tasks)
    if skipped:
        print(f"モデル {model} では課題 {skipped} が使えないので、その課題2は測りません。")
    classes_path = significant_heads_and_mlps.output_paths(model)[2]
    probes_path, probe_accuracy_path, transfer_path = attribute_probes.output_paths(model)
    accuracy_path = sense_accuracy.result_paths(model)[1]
    probes_command = f"sense-addition train-attribute-probes --model {model}"
    # 入力のファイルと、それを書くコマンド（write_attribute_patching に、この順で渡す）
    inputs = {
        classes_path: f"sense-addition select-significant-heads-and-mlps --model {model}",
        probes_path: probes_command,
        evaluation_finalization.QUESTIONS_PATH: evaluation_finalization.COMMAND,
        accuracy_path: f"sense-addition measure-sense-accuracy --model {model}",
        word_finalization.OUTPUT_PATH: word_finalization.COMMAND,
    }
    patching_path, *judgment_paths = output_paths(model)
    step = Step(
        command,
        inputs={
            **inputs,
            probe_accuracy_path: probes_command,
            transfer_path: probes_command,
            pretrained_model_checks.TASK_ACCURACY_PATH: pretrained_model_checks.COMMAND,
        },
        outputs=list(output_paths(model)),
        settings={
            "checkpoints": checkpoint_states(list(tested.values())),
            "seed": settings.words.seed,
            "tested_pairs": TESTED_PAIRS,
        },
    )

    def compute() -> None:
        rows = write_attribute_patching(
            tested,
            lambda source: LanguageModel.load(source, batch_size=BATCH_SIZE),
            tuple(inputs),
            patching_path,
            settings.words.seed,
            tasks,
        )
        write_attribute_judgments(
            rows,
            (probe_accuracy_path, transfer_path, probes_path),
            tuple(judgment_paths),
            settings.words.seed,
        )

    run_unless_unchanged(step, compute, arguments.recompute)


def write_attribute_patching(
    checkpoints: dict[str, Path],
    load_model: Callable[[ModelSource], LanguageModel],
    input_paths: tuple[Path, Path, Path, Path, Path],
    output: Path,
    seed: int,
    tasks: set[str] | None = None,
) -> list[dict[str, object]]:
    """入れ替えの行を書き、返す（tasks は、測る課題。None なら全て）。"""
    classes_path, probes_path, questions_path, accuracy_path, words_path = input_paths
    adopted = adopted_heads(classes_path)
    rows: list[dict[str, object]] = []
    if adopted:
        probes = read_attribute_probes(probes_path)
        questions = parquet.read_table(questions_path).to_pylist()
        usable = sense_accuracy.usable_senses(accuracy_path)
        words = {word.word: word for word in read_rows(words_path, SelectedWord)}
        for run, checkpoint in checkpoints.items():
            model = load_model(ModelSource(repository=str(checkpoint)))
            baselines = baseline_heads(adopted, model.model.config.num_attention_heads, seed)
            plans = planned_patches(adopted, baselines, probes, seed)
            for pair in tested_pairs(run, questions, usable, seed):
                rows += measure_pair(model, run, pair, plans, questions, words, tasks)
            # 次の run の重みを読む前に、この重みを GPU から消す
            del model
            gc.collect()
            torch.cuda.empty_cache()
    output.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_table(arrow.Table.from_pylist(rows), output)
    return rows


def planned_patches(
    adopted: Sequence[tuple[int, int]],
    baselines: dict[tuple[int, int], tuple[int, int]],
    probes: dict[tuple[int, int, str], LogisticProbe],
    seed: int,
) -> list[PlannedPatch]:
    """採用した head × 属性ごとの、属性の方向と2つの比較の基準の入れ替え。"""
    plans = []
    for layer, head in adopted:
        for attribute in ATTRIBUTES:
            probe = probes.get((layer, head, attribute))
            if probe is None:
                continue
            basis = attribute_basis(probe)
            plans.append(PlannedPatch((layer, head), attribute, PATCHES[0], (layer, head), basis))
            other = baselines.get((layer, head))
            if other is not None and (*other, attribute) in probes:
                other_basis = attribute_basis(probes[(*other, attribute)])
                plans.append(PlannedPatch((layer, head), attribute, PATCHES[1], other, other_basis))
            random_numbers = np.random.default_rng([seed, layer, head, ATTRIBUTES.index(attribute)])
            directions = random_basis(basis.shape[0], basis.shape[1], random_numbers)
            plans.append(
                PlannedPatch((layer, head), attribute, PATCHES[2], (layer, head), directions)
            )
    return plans


def tested_pairs(
    run: str,
    questions: Sequence[dict[str, object]],
    usable: set[tuple[str, int, str, str]],
    seed: int,
) -> list[dict[tuple[str, str], int]]:
    """run の条件の、使える最小対（（手がかり, 課題）ごとの問題の番号）から、TESTED_PAIRS 組。"""
    found: dict[tuple, dict[tuple[str, str], int]] = {}
    for index, question in enumerate(questions):
        if f"{question['kind']}-{question['number_of_senses']}" != run:
            continue
        key = tuple(
            question[name]
            for name in ("kind", "number_of_senses", "target", "donor", "donor_sentence")
        )
        found.setdefault(key, {})[question["cue"], question["task"]] = index
    usable_pairs = [
        cues
        for (kind, number, target, donor, _), cues in found.items()
        if len(cues) == 4
        and all((kind, number, target, sense) in usable for sense in (donor, "original"))
    ]
    count = min(TESTED_PAIRS, len(usable_pairs))
    return random.Random(f"{seed}:{run}").sample(usable_pairs, count)


def measure_pair(
    model: LanguageModel,
    run: str,
    pair: dict[tuple[str, str], int],
    plans: Sequence[PlannedPatch],
    questions: Sequence[dict[str, object]],
    words: dict[str, SelectedWord],
    tasks: set[str] | None = None,
) -> list[dict[str, object]]:
    """1つの最小対の、両方向 × 2つの課題 × 入れ替えごとの行（tasks にない課題は測らない）。"""
    rows = []
    attribute_task = next(task for _, task in pair if task != "sense")
    measured = [task for task in ("sense", attribute_task) if tasks is None or task in tasks]
    heads = sorted({plan.patched_head for plan in plans})
    for direction, (source_cue, destination_cue) in zip(
        DIRECTIONS, (("donor", "target"), ("target", "donor")), strict=True
    ):
        for task in measured:
            source, destination = pair[source_cue, task], pair[destination_cue, task]
            question = questions[source]
            inputs = pair_inputs(model, questions, words, question["target"], source, destination)
            if inputs is None:
                continue
            source_ids, destination_ids, span, answers = inputs
            position = span[-1]
            candidates = choice_token_ids(question["choices"], model.tokenizer)
            outputs = model.head_outputs([destination_ids], [position], heads)[0]
            patches = [()] + [
                (
                    SubspacePatch(
                        *plan.patched_head,
                        plan.basis,
                        outputs[heads.index(plan.patched_head)][None, :],
                    ),
                )
                for plan in plans
            ]
            values = subspace_patched_log_probabilities(
                model, source_ids, [position], patches, candidates
            )
            moves, switched = movements(values, task, candidates, answers)
            rows += [
                {
                    "run": run,
                    "kind": question["kind"],
                    "target": question["target"],
                    "donor": question["donor"],
                    "donor_sentence": question["donor_sentence"],
                    "direction": direction,
                    "task": task,
                    "layer": plan.head[0],
                    "head": plan.head[1],
                    "attribute": plan.attribute,
                    "patch": plan.patch,
                    "patched_head": plan.patched_head[1],
                    "dimensions": plan.basis.shape[1],
                    "metric_change": float(moves[number + 1]),
                    "switched": bool(switched[number + 1]),
                    "clean_switched": bool(switched[0]),
                }
                for number, plan in enumerate(plans)
            ]
    return rows


def movements(
    values: np.ndarray, task: str, candidates: Sequence[int], answers: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """行ごとの、s_Y の側への動き（入れ替えない1行目との差）と、s_Y の答えを選ぶか。

    values は（入れ替え, 選択肢）の対数確率。answers は（処理する文の正解, s_Y の正解）のトークン。
    課題1は、m（処理する文の語義 − s_Y の語義の対数確率）の減り方。課題2は、s_Y の正解の対数確率の
    増え方。選ぶかは、全ての選択肢のうちで s_Y の正解の対数確率が最も大きいか。
    """
    destination = list(candidates).index(answers[1])
    switched = values.argmax(axis=1) == destination
    if task == "sense":
        metric = values[:, list(candidates).index(answers[0])] - values[:, destination]
        return metric[0] - metric, switched
    return values[:, destination] - values[0, destination], switched
