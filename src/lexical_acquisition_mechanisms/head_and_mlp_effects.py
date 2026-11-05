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
from lexical_acquisition_mechanisms.csv_tables import read_rows
from lexical_acquisition_mechanisms.language_model import LanguageModel
from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.sense_accuracy import SenseAccuracy
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


def effect_paths(model: str) -> tuple[Path, Path]:
    directory = sense_accuracy.RESULTS_DIRECTORY / model
    return directory / "patching_pairs.parquet", directory / "head_and_mlp_effects.npz"


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
        settings={"checkpoints": checkpoint_states(list(checkpoints.values()))},
    )
    run_unless_unchanged(
        step,
        lambda: write_effects(
            checkpoints,
            lambda source: LanguageModel.load(source, batch_size=BATCH_SIZE),
            input_paths,
            effect_paths(model),
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
    output_paths: tuple[Path, Path],
) -> None:
    words_path, questions_path, accuracy_path = input_paths
    words = {word.word: word for word in read_rows(words_path, SelectedWord)}
    questions = parquet.read_table(questions_path).to_pylist()
    usable = {
        (row.kind, row.number_of_senses, row.target, row.sense)
        for row in read_rows(accuracy_path, SenseAccuracy)
        if row.usable
    }
    pairs = minimal_pair_questions(questions)
    rows: list[dict[str, object]] = []
    effects: dict[str, list[np.ndarray]] = {"target_position": [], "after_target_positions": []}
    order = []
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
        del model
        gc.collect()
        torch.cuda.empty_cache()
    if skipped:
        print(
            f"2文の問題の位置がそろわない最小対を {len(skipped)} 組除きました"
            f"（例: {[(pair.target, pair.donor_sentence) for pair in list(skipped)[:3]]}）。"
        )
    pairs_path, effects_path = output_paths
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


def _measure_pair(
    model: LanguageModel,
    pair: MinimalPairQuestions,
    questions: Sequence[dict[str, object]],
    words: dict[str, SelectedWord],
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]] | None:
    """1つの最小対の、2文の m と、方向ごとの（ターゲット語の位置, そこから後の全位置）の効果。

    2文の問題の、長さか、ターゲット語の位置がそろわなければ None。
    """
    word_ids = [int(token) for token in words[pair.target].token_ids.split()]
    (donor_ids, donor_spans), (target_ids, target_spans) = (
        encode_prompt_with_word_spans(
            questions[index]["prompt_template"], word_ids, model.tokenizer
        )
        for index in (pair.donor_question, pair.target_question)
    )
    if len(donor_ids) != len(target_ids) or donor_spans[0] != target_spans[0]:
        return None
    # 文の中のターゲット語（問題文で最初に現れる語）から、問題文の最後まで
    span = donor_spans[0]
    positions = list(range(span.start, len(donor_ids)))
    donor_choice, original_choice = choice_token_ids(
        [pair.donor, questions[pair.target_question]["answer"]], model.tokenizer
    )
    metrics, outputs, gradients = outputs_and_gradients(
        model,
        [donor_ids, target_ids],
        positions,
        [(donor_choice, original_choice), (original_choice, donor_choice)],
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
