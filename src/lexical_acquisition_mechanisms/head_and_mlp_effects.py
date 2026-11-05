"""各 head と MLP の出力を入れ替えたときの、語義の選択への効果を測る
（sense-addition measure-head-and-mlp-effects --model）。

語義の追加の各 run の最後の重みで、課題1の最小対の問題の組（s_X: donor の語義を手がかりで示す文、
s_Y: 元の語義を示す文）ごとに、両方向の効果を、全ての構成要素について勾配による近似で出す
（component_patching）。
- 方向: donor-to-target は、s_X を処理するとき s_Y での出力に入れ替える。target-to-donor はその逆
- m: 処理する文の語義の選択肢と、もう一方の文の語義の選択肢の、ロジットの差
- 効果: e_c = m − m_c。全てを入れ替えたときの効果（正規化の分母）は、m(s_X) − m(s_Y)（両方向で同じ）
- 入れ替える位置: 文の中のターゲット語のトークン（本分析）と、そこから問題文の最後までの全ての位置
  （補足）
対照の run は、対応する種類（名詞の対照は同カテゴリと別カテゴリ、形容詞の対照は形容詞）の、
全ての数の問題で測る（手がかりの語そのものの影響を、後で引くため）。
出力は outputs/results/sense-addition/<モデル>/ の
- patching_pairs.parquet: 入れ替えの組ごとの、run・条件・最小対・方向・問題の番号・m・分母と、
  手順1で2つの語義とも使えるとした組か（usable）
- head_and_mlp_effects.npz: 組 × 構成要素の効果（target_position と after_target_positions）と、
  構成要素の層（layers）と head の番号（heads。MLP は −1）
- patching_checks.csv・patching_check_summary.csv: 近似を、上位と無作為に選んだ構成要素の実際の
  入れ替えで確かめた結果（patching_checks）
"""

import argparse
import gc
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as arrow
import pyarrow.parquet as parquet
import torch

from lexical_acquisition_mechanisms import (
    evaluation_finalization,
    sense_accuracy,
    word_finalization,
)
from lexical_acquisition_mechanisms.component_patching import (
    components,
    estimated_effects,
    outputs_and_gradients,
)
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.language_model import LanguageModel
from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.patching_checks import (
    PatchingCheck,
    PatchingCheckSummary,
    check_estimates,
    summarize_checks,
)
from lexical_acquisition_mechanisms.settings import ModelSource, Settings
from lexical_acquisition_mechanisms.task_prompts import (
    choice_token_ids,
    encode_prompt_with_word_spans,
)
from lexical_acquisition_mechanisms.unknown_word_indicators import checkpoint_states
from lexical_acquisition_mechanisms.word_selection import SelectedWord

BATCH_SIZE = 16
DIRECTIONS = ("donor-to-target", "target-to-donor")


@dataclass(frozen=True)
class MinimalPairQuestions:
    """同じ条件の、最小対の2文の課題1の問題（問題の番号）。"""

    kind: str
    number_of_senses: int
    target: str
    donor: str
    donor_sentence: str
    donor_question: int
    target_question: int


def effect_paths(model: str) -> tuple[Path, Path, Path, Path]:
    directory = sense_accuracy.RESULTS_DIRECTORY / model
    return (
        directory / "patching_pairs.parquet",
        directory / "head_and_mlp_effects.npz",
        directory / "patching_checks.csv",
        directory / "patching_check_summary.csv",
    )


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="settings.yaml の models の名前")


def measure_head_and_mlp_effects(settings: Settings, arguments: argparse.Namespace) -> None:
    model = arguments.model
    command = f"sense-addition measure-head-and-mlp-effects --model {model}"
    checkpoints = sense_accuracy.final_checkpoints(model, command)
    accuracy_path = sense_accuracy.result_paths(model)[1]
    input_paths = (
        word_finalization.OUTPUT_PATH,
        evaluation_finalization.QUESTIONS_PATH,
        accuracy_path,
    )
    step = Step(
        command,
        inputs={
            word_finalization.OUTPUT_PATH: word_finalization.COMMAND,
            evaluation_finalization.QUESTIONS_PATH: evaluation_finalization.COMMAND,
            accuracy_path: f"sense-addition measure-sense-accuracy --model {model}",
        },
        outputs=list(effect_paths(model)),
        settings={
            "checkpoints": checkpoint_states(list(checkpoints.values())),
            "seed": settings.words.seed,
        },
    )
    run_unless_unchanged(
        step,
        lambda: write_effects(
            checkpoints,
            lambda source: LanguageModel.load(source, batch_size=BATCH_SIZE),
            input_paths,
            effect_paths(model),
            settings.words.seed,
        ),
        arguments.recompute,
    )


def minimal_pair_questions(questions: Sequence[dict[str, object]]) -> list[MinimalPairQuestions]:
    """課題1の問題を、同じ条件の最小対の2文の組にする。"""
    found: dict[tuple, dict[str, int]] = {}
    for index, question in enumerate(questions):
        if question["task"] != "sense":
            continue
        key = tuple(
            question[name]
            for name in ("kind", "number_of_senses", "target", "donor", "donor_sentence")
        )
        found.setdefault(key, {})[question["cue"]] = index
    return [
        MinimalPairQuestions(*key, cues["donor"], cues["target"])
        for key, cues in found.items()
        if set(cues) == {"donor", "target"}
    ]


def write_effects(
    checkpoints: dict[str, Path],
    load_model: Callable[[ModelSource], LanguageModel],
    input_paths: tuple[Path, Path, Path],
    output_paths: tuple[Path, Path, Path, Path],
    seed: int,
) -> None:
    words_path, questions_path, accuracy_path = input_paths
    words = {word.word: word for word in read_rows(words_path, SelectedWord)}
    questions = parquet.read_table(questions_path).to_pylist()
    usable = sense_accuracy.usable_senses(accuracy_path)
    pairs = minimal_pair_questions(questions)
    rows: list[dict[str, object]] = []
    effects: dict[str, list[np.ndarray]] = {"target_position": [], "after_target_positions": []}
    order = []
    checks: list[PatchingCheck] = []
    # 位置がそろわない最小対（run によらないので、run をまたいで1回だけ数える）
    skipped: set[MinimalPairQuestions] = set()
    for run, checkpoint in checkpoints.items():
        run_pairs = [
            pair
            for pair in pairs
            if sense_accuracy.evaluated_in(run, questions[pair.donor_question])
        ]
        if not run_pairs:
            continue
        model = load_model(ModelSource(repository=str(checkpoint)))
        order = components(model)
        first_row = len(rows)
        for pair in run_pairs:
            measured = _measure_pair(model, pair, questions, words)
            if measured is None:
                skipped.add(pair)
                continue
            metrics, directions = measured
            both_usable = all(
                (pair.kind, pair.number_of_senses, pair.target, sense) in usable
                for sense in (pair.donor, "original")
            )
            question_orders = (
                (pair.donor_question, pair.target_question),
                (pair.target_question, pair.donor_question),
            )
            for direction, (source, destination), metric, (word, after) in zip(
                DIRECTIONS, question_orders, metrics, directions, strict=True
            ):
                rows.append(
                    {
                        "run": run,
                        "kind": pair.kind,
                        "number_of_senses": pair.number_of_senses,
                        "target": pair.target,
                        "donor": pair.donor,
                        "donor_sentence": pair.donor_sentence,
                        "direction": direction,
                        "question": source,
                        "partner": destination,
                        "metric": float(metric),
                        "total_effect": float(metrics[0] + metrics[1]),
                        "usable": both_usable,
                    }
                )
                effects["target_position"].append(word)
                effects["after_target_positions"].append(after)
        if not run.endswith("control") and len(rows) > first_row:
            checks += check_estimates(
                model,
                run,
                first_row,
                rows[first_row:],
                np.array(effects["target_position"][first_row:]),
                lambda row, model=model: _target_position_inputs(model, questions, words, row),
                seed,
            )
        del model
        gc.collect()
        torch.cuda.empty_cache()
    if skipped:
        print(
            f"2文の問題の位置がそろわない最小対を {len(skipped)} 組除きました"
            f"（例: {[(pair.target, pair.donor_sentence) for pair in list(skipped)[:3]]}）。"
        )
    pairs_path, effects_path, checks_path, summary_path = output_paths
    pairs_path.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_table(arrow.Table.from_pylist(rows), pairs_path)
    size = len(order)
    np.savez(
        effects_path,
        **{
            name: np.array(values, dtype=np.float32).reshape(len(values), size)
            for name, values in effects.items()
        },
        layers=np.array([component.layer for component in order], dtype=np.int16),
        heads=np.array([component.head for component in order], dtype=np.int16),
    )
    write_rows(checks_path, PatchingCheck, checks)
    write_rows(summary_path, PatchingCheckSummary, summarize_checks(checks))


def pair_inputs(
    model: LanguageModel,
    questions: Sequence[dict[str, object]],
    words: dict[str, SelectedWord],
    target: str,
    source: int,
    destination: int,
) -> tuple[list[int], list[int], range, tuple[int, int]] | None:
    """処理する問題と、入れ替えのもとの問題のトークン列と、文の中のターゲット語の位置の範囲と、
    選択肢（処理する問題の正解, もう一方の問題の正解）。

    2つの問題の、長さか、ターゲット語の位置がそろわなければ None。
    """
    word_ids = [int(token) for token in words[target].token_ids.split()]
    (source_ids, source_spans), (destination_ids, destination_spans) = (
        encode_prompt_with_word_spans(
            questions[index]["prompt_template"], word_ids, model.tokenizer
        )
        for index in (source, destination)
    )
    # 文の中のターゲット語は、問題文で最初に現れる語
    if len(source_ids) != len(destination_ids) or source_spans[0] != destination_spans[0]:
        return None
    answers = [questions[index]["answer"] for index in (source, destination)]
    preferred, other = choice_token_ids(answers, model.tokenizer)
    return source_ids, destination_ids, source_spans[0], (preferred, other)


def _target_position_inputs(
    model: LanguageModel,
    questions: Sequence[dict[str, object]],
    words: dict[str, SelectedWord],
    row: dict[str, object],
) -> tuple[list[int], list[int], list[int], tuple[int, int]]:
    """測った組を、ターゲット語の位置で入れ替えるための入力（patching_checks に渡す）。"""
    inputs = pair_inputs(model, questions, words, row["target"], row["question"], row["partner"])
    assert inputs is not None, "測った組なので、位置はそろっている"
    source_ids, destination_ids, span, choice = inputs
    return source_ids, destination_ids, list(span), choice


def _measure_pair(
    model: LanguageModel,
    pair: MinimalPairQuestions,
    questions: Sequence[dict[str, object]],
    words: dict[str, SelectedWord],
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]] | None:
    """1つの最小対の、2文の m と、方向ごとの（ターゲット語の位置, そこから後の全位置）の効果。

    2文の問題の位置がそろわなければ None。
    """
    inputs = pair_inputs(
        model, questions, words, pair.target, pair.donor_question, pair.target_question
    )
    if inputs is None:
        return None
    donor_ids, target_ids, span, choice = inputs
    positions = list(range(span.start, len(donor_ids)))
    metrics, outputs, gradients = outputs_and_gradients(
        model, [donor_ids, target_ids], positions, [choice, choice[::-1]]
    )
    word = range(len(span))
    directions = []
    for source, destination in ((0, 1), (1, 0)):
        directions.append(
            (
                estimated_effects(
                    outputs[source].at(word),
                    gradients[source].at(word),
                    outputs[destination].at(word),
                ),
                estimated_effects(outputs[source], gradients[source], outputs[destination]),
            )
        )
    return metrics, directions
