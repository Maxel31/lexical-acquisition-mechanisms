"""同じ名詞の意味を、未知語に付けた場合と既知語に足した場合とで、head ごとの効果の並びを比べる
（dependence-on-category-judging-heads compare-head-maps --model。設計書 3-3 の探索的な比較）。

- 新しい形の地図: 名詞ごとに、未知語の獲得で露出回数が最大の未知語（その run の最後の重み）で、
  課題1の問題（中立文脈の21条件）を処理するとき、文の中の未知語の位置（語の最後のトークン）の
  各 head の出力を、同じ run の、トークン数と綴りの近さが同じ露出0回の文字列（同じ問題文に
  入れたもの）での出力に入れ替えたときの、正解（割り当てた名詞）の対数確率の低下。勾配による近似
  （component_patching）で全ての head を出し、条件で平均する
- 既存の形の地図: 語義の追加の head の地図（measure-head-and-mlp-effects）のうち、その名詞が donor
  の最小対（名詞を足す種類の、対照でない run。全ての N）での、ターゲット語の位置の、対照を引いた
  効果の平均
- 比べるもの: 名詞ごとの、2つの地図の head の効果の順位相関（Spearman）と、効果が大きい TOP_HEADS
  個の head の重なりの割合（偶然の水準は TOP_HEADS ÷ head の数）
出力は outputs/results/dependence-on-category-judging-heads/<モデル>/ の head_maps.npz（名詞 ×
head の2つの地図）と head_map_comparison.csv（名詞ごと。名詞での平均と区間は make-report で出す）。
"""

import argparse
import gc
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as parquet
import torch

from lexical_acquisition_mechanisms import (
    head_and_mlp_effects,
    pretrained_model_checks,
    task_questions,
    unknown_word_assignments,
    word_finalization,
)
from lexical_acquisition_mechanisms.category_dependence import (
    EXPERIMENT,
    RESULTS_DIRECTORY,
    unknown_word_checkpoints,
)
from lexical_acquisition_mechanisms.component_patching import (
    MLP,
    components,
    estimated_effects,
    outputs_and_gradients,
)
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.language_model import LanguageModel
from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.settings import ModelSource, Settings
from lexical_acquisition_mechanisms.significant_heads_and_mlps import SCOPES, minimal_pair_units
from lexical_acquisition_mechanisms.task_prompts import (
    choice_token_ids,
    encode_prompt_with_word_spans,
)
from lexical_acquisition_mechanisms.unknown_word_assignments import UnknownWordAssignment
from lexical_acquisition_mechanisms.unknown_word_indicators import checkpoint_states
from lexical_acquisition_mechanisms.word_selection import SelectedWord

TOP_HEADS = 20
BATCH_SIZE = 16
NOUN_KINDS = ("same-category", "other-category")


@dataclass
class HeadMapComparison:
    noun: str
    # 新しい形の地図に使った run と未知語
    run: str
    unknown_word: str
    spearman: float
    top_overlap: float
    chance_overlap: float


def output_paths(model: str) -> tuple[Path, Path]:
    directory = RESULTS_DIRECTORY / model
    return directory / "head_maps.npz", directory / "head_map_comparison.csv"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="settings.yaml の models の名前")


def compare_head_maps(settings: Settings, arguments: argparse.Namespace) -> None:
    model = arguments.model
    command = f"{EXPERIMENT} compare-head-maps --model {model}"
    checkpoints = unknown_word_checkpoints(model, command)
    pretrained_model_checks.require_usable_tasks(model, ["sense"], command)
    pairs_path, effects_path, _, _ = head_and_mlp_effects.effect_paths(model)
    measured_by = f"sense-addition measure-head-and-mlp-effects --model {model}"
    inputs = {
        task_questions.UNKNOWN_WORD_QUESTIONS_PATH: task_questions.COMMAND,
        unknown_word_assignments.ASSIGNMENTS_PATH: unknown_word_assignments.COMMAND,
        word_finalization.OUTPUT_PATH: word_finalization.COMMAND,
        pairs_path: measured_by,
        effects_path: measured_by,
    }
    exposure = max(settings.unknown_word_acquisition.exposure_levels)
    step = Step(
        command,
        inputs={
            **inputs,
            pretrained_model_checks.TASK_ACCURACY_PATH: pretrained_model_checks.COMMAND,
        },
        outputs=list(output_paths(model)),
        settings={
            "checkpoints": checkpoint_states(list(checkpoints.values())),
            "exposure": exposure,
            "top_heads": TOP_HEADS,
        },
    )
    run_unless_unchanged(
        step,
        lambda: write_head_maps(
            checkpoints,
            lambda source: LanguageModel.load(source, batch_size=BATCH_SIZE),
            tuple(inputs),
            output_paths(model),
            exposure,
        ),
        arguments.recompute,
    )


def write_head_maps(
    checkpoints: dict[str, Path],
    load_model: Callable[[ModelSource], LanguageModel],
    input_paths: Sequence[Path],
    outputs: tuple[Path, Path],
    exposure: int,
) -> None:
    questions_path, assignments_path, words_path, pairs_path, effects_path = input_paths
    assignments = read_rows(assignments_path, UnknownWordAssignment)
    new_form = new_form_maps(
        checkpoints,
        load_model,
        parquet.read_table(questions_path).to_pylist(),
        assignments,
        {word.word: word for word in read_rows(words_path, SelectedWord)},
        exposure,
    )
    arrays = np.load(effects_path)
    existing = existing_form_maps(parquet.read_table(pairs_path).to_pylist(), arrays)
    nouns = sorted(set(new_form) & set(existing))
    heads = arrays["heads"] != MLP
    comparisons = []
    for noun in nouns:
        values, assignment = new_form[noun]
        spearman = _spearman(values, existing[noun])
        top = min(TOP_HEADS, len(values))
        overlap = len(set(np.argsort(-values)[:top]) & set(np.argsort(-existing[noun])[:top]))
        comparisons.append(
            HeadMapComparison(
                noun,
                assignment.run,
                assignment.unknown_word,
                spearman,
                overlap / top,
                top / len(values),
            )
        )
    maps_path, comparison_path = outputs
    maps_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        maps_path,
        nouns=np.array(nouns),
        new_form=np.array([new_form[noun][0] for noun in nouns]).reshape(len(nouns), -1),
        existing_form=np.array([existing[noun] for noun in nouns]).reshape(len(nouns), -1),
        layers=arrays["layers"][heads],
        heads=arrays["heads"][heads],
    )
    write_rows(comparison_path, HeadMapComparison, comparisons)


def new_form_maps(
    checkpoints: dict[str, Path],
    load_model: Callable[[ModelSource], LanguageModel],
    questions: Sequence[dict[str, object]],
    assignments: Sequence[UnknownWordAssignment],
    words: dict[str, SelectedWord],
    exposure: int,
) -> dict[str, tuple[np.ndarray, UnknownWordAssignment]]:
    """名詞ごとの、新しい形の地図（head ごとの効果）と、使った未知語の割り当て。"""
    targets = [row for row in assignments if row.exposure == exposure]
    zero_words: dict[tuple[str, int, str], str] = {}
    for row in sorted(assignments, key=lambda row: row.unknown_word):
        if row.exposure == 0:
            zero_words.setdefault(
                (row.run, row.token_count, row.spelling_closeness), row.unknown_word
            )
    maps = {}
    for run, checkpoint in checkpoints.items():
        run_targets = [row for row in targets if row.run == run]
        if not run_targets:
            continue
        model = load_model(ModelSource(repository=str(checkpoint)))
        head_indices = [index for index, item in enumerate(components(model)) if item.head != MLP]
        for assignment in run_targets:
            zero = zero_words.get((run, assignment.token_count, assignment.spelling_closeness))
            if zero is None:
                continue
            effects = [
                effect[head_indices]
                for question in questions
                if question["run"] == run
                and question["unknown_word"] == assignment.unknown_word
                and question["task"] == "sense"
                and (effect := _question_effects(model, question, words, assignment, zero))
                is not None
            ]
            if effects:
                maps[assignment.noun] = (np.mean(effects, axis=0), assignment)
        # 次の run の重みを読む前に、この重みを GPU から消す
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return maps


def existing_form_maps(rows: Sequence[dict[str, object]], arrays) -> dict[str, np.ndarray]:
    """donor の名詞ごとの、既存の形の地図（head ごとの、対照を引いた効果の平均）。"""
    values = arrays[SCOPES[0]].astype(np.float64)
    heads = arrays["heads"] != MLP
    by_donor: dict[str, list[np.ndarray]] = {}
    for (kind, _), units in minimal_pair_units(rows).items():
        if kind not in NOUN_KINDS:
            continue
        for unit in units:
            difference = values[unit.treated].mean(axis=0) - values[unit.control].mean(axis=0)
            by_donor.setdefault(unit.donor, []).append(difference[heads])
    return {donor: np.mean(maps, axis=0) for donor, maps in by_donor.items()}


def _question_effects(
    model: LanguageModel,
    question: dict[str, object],
    words: dict[str, SelectedWord],
    assignment: UnknownWordAssignment,
    zero: str,
) -> np.ndarray | None:
    """1つの問題の、全ての構成要素の効果（未知語の位置を、露出0回の文字列の出力に入れ替える）。"""
    encoded = [
        encode_prompt_with_word_spans(
            question["prompt_template"],
            [int(token) for token in words[word].token_ids.split()],
            model.tokenizer,
        )
        for word in (assignment.unknown_word, zero)
    ]
    (unknown_ids, unknown_spans), (zero_ids, zero_spans) = encoded
    if len(unknown_ids) != len(zero_ids) or unknown_spans[0] != zero_spans[0]:
        return None
    correct = choice_token_ids([question["answer"]], model.tokenizer)[0]
    # 未知語の位置は、語の最後のトークン（設計書の「語の位置」）
    _, outputs, gradients = outputs_and_gradients(
        model, [unknown_ids, zero_ids], [unknown_spans[0][-1]], [(correct, None)] * 2
    )
    return estimated_effects(outputs[0], gradients[0], outputs[1])


def _spearman(first: np.ndarray, second: np.ndarray) -> float:
    """順位相関（値の順位どうしの相関係数）。"""
    ranks = [np.argsort(np.argsort(values)) for values in (first, second)]
    if len(first) < 2 or ranks[0].std() == 0 or ranks[1].std() == 0:
        return float("nan")
    return float(np.corrcoef(*ranks)[0, 1])
