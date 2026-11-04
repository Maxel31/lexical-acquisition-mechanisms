"""属性のプローブを学習し、確かめる（sense-addition train-attribute-probes --model）。

手順2で採用した各 head（component_classes.csv の、MLP でないもの）について、カテゴリ（6クラス）と
品詞（2クラス。noun / adjective）を当てるロジスティック回帰を学習する。head の出力は、問題の文の中の
語の最後のトークンの位置で取り出す（出力の射影に入る前の値）。
1. 学習: 学習前のモデル（settings.yaml の版）で、自然な語の課題2の問題（カテゴリの問題と品詞の
   問題。evaluation-data build-task-questions が書いた natural_word_questions.parquet）を使う。
   語の単位で、クラスごとに HELD_OUT_SHARE の語を確認用に分ける。
2. (a) 確認用の語での正解率と、偶然の水準（1 ÷ クラスの数）を超えるかの片側の二項検定。
3. (b) 語義を足した各 run（対照を除く）の最後の重みで、その run の条件の最小対の課題2の問題（名詞を
   足す条件はカテゴリ、形容詞を足す条件は品詞）のターゲット語に当て、手がかりが示す語義の属性
   （問題の正解）を当てる正解率と、二項検定。手順1で使えるとした語義の問題だけを使う。
出力は outputs/probes/<モデル>/attribute_probes.npz（キーは
"{層}-{head}-{属性}-{weights|biases|mean|scale}"）と、outputs/results/sense-addition/<モデル>/ の
attribute_probe_accuracy.csv（(a)）・attribute_probe_transfer.csv（(b)）。
"""

import argparse
import gc
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as parquet
import torch

from lexical_acquisition_mechanisms import (
    evaluation_finalization,
    real_word_probes,
    sense_accuracy,
    significant_heads_and_mlps,
    task_questions,
    word_finalization,
)
from lexical_acquisition_mechanisms.component_adoption import ComponentClass
from lexical_acquisition_mechanisms.component_patching import MLP
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.language_model import LanguageModel
from lexical_acquisition_mechanisms.logistic_probes import LogisticProbe, train_logistic_probe
from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.sense_accuracy import SenseAccuracy
from lexical_acquisition_mechanisms.settings import ModelSource, Settings
from lexical_acquisition_mechanisms.task_accuracy import SIGNIFICANCE_LEVEL, binomial_p_value
from lexical_acquisition_mechanisms.task_prompts import (
    PARTS_OF_SPEECH,
    encode_prompt_with_word_spans,
)
from lexical_acquisition_mechanisms.unknown_word_indicators import checkpoint_states
from lexical_acquisition_mechanisms.word_selection import SelectedWord

HELD_OUT_SHARE = 0.2
BATCH_SIZE = 32
# 属性ごとの、課題2の問題の種類（task の列の値）
ATTRIBUTES = ("category", "part-of-speech")


@dataclass
class AttributeProbeAccuracy:
    layer: int
    head: int
    # 当てる属性（category か part-of-speech）
    attribute: str
    training_accuracy: float
    held_out_accuracy: float
    held_out_questions: int
    chance: float
    p_value: float
    significant: bool


@dataclass
class AttributeProbeTransfer:
    run: str
    layer: int
    head: int
    attribute: str
    questions: int
    accuracy: float
    chance: float
    p_value: float
    significant: bool


def output_paths(model: str) -> tuple[Path, Path, Path]:
    directory = sense_accuracy.RESULTS_DIRECTORY / model
    return (
        real_word_probes.PROBES_DIRECTORY / model / "attribute_probes.npz",
        directory / "attribute_probe_accuracy.csv",
        directory / "attribute_probe_transfer.csv",
    )


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="settings.yaml の models の名前")


def train_attribute_probes(settings: Settings, arguments: argparse.Namespace) -> None:
    model = arguments.model
    command = f"sense-addition train-attribute-probes --model {model}"
    checkpoints = sense_accuracy.final_checkpoints(model, command)
    classes_path = significant_heads_and_mlps.output_paths(model)[2]
    accuracy_path = sense_accuracy.result_paths(model)[1]
    input_paths = (classes_path, task_questions.NATURAL_QUESTIONS_PATH)
    transfer_paths = (
        evaluation_finalization.QUESTIONS_PATH,
        accuracy_path,
        word_finalization.OUTPUT_PATH,
    )
    categories = [category.name for category in settings.words.categories]
    step = Step(
        command,
        inputs={
            classes_path: f"sense-addition select-significant-heads-and-mlps --model {model}",
            task_questions.NATURAL_QUESTIONS_PATH: task_questions.COMMAND,
            evaluation_finalization.QUESTIONS_PATH: evaluation_finalization.COMMAND,
            accuracy_path: f"sense-addition measure-sense-accuracy --model {model}",
            word_finalization.OUTPUT_PATH: word_finalization.COMMAND,
        },
        outputs=list(output_paths(model)),
        settings={
            "model": settings.models[model].model_dump(),
            "checkpoints": checkpoint_states(list(checkpoints.values())),
            "categories": categories,
            "seed": settings.words.seed,
            "held_out_share": HELD_OUT_SHARE,
        },
    )

    def load_model(source: ModelSource) -> LanguageModel:
        return LanguageModel.load(source, batch_size=BATCH_SIZE)

    def compute() -> None:
        probes_path, accuracy_output, transfer_output = output_paths(model)
        probes = write_attribute_probes(
            settings.models[model],
            load_model,
            input_paths,
            (probes_path, accuracy_output),
            categories,
            settings.words.seed,
        )
        write_probe_transfer(
            checkpoints, load_model, probes, categories, transfer_paths, transfer_output
        )

    run_unless_unchanged(step, compute, arguments.recompute)


def write_attribute_probes(
    pretrained: ModelSource,
    load_model: Callable[[ModelSource], LanguageModel],
    input_paths: tuple[Path, Path],
    outputs: tuple[Path, Path],
    categories: Sequence[str],
    seed: int,
) -> dict[tuple[int, int, str], LogisticProbe]:
    """学習したプローブ（（層, head, 属性）ごと）を書き、返す。"""
    classes_path, natural_path = input_paths
    heads = sorted(
        {
            (row.layer, row.head)
            for row in read_rows(classes_path, ComponentClass)
            if row.head != MLP
        }
    )
    names = {"category": list(categories), "part-of-speech": list(PARTS_OF_SPEECH)}
    probes: dict[tuple[int, int, str], LogisticProbe] = {}
    accuracy = []
    if heads:
        natural = parquet.read_table(natural_path).to_pylist()
        model = load_model(pretrained)
        for attribute in ATTRIBUTES:
            rows = [row for row in natural if row["task"] == attribute]
            held_out_words = _held_out_words(rows, seed)
            held_out = np.array([row["word"] in held_out_words for row in rows])
            labels = np.array([names[attribute].index(row["answer"]) for row in rows])
            features = _head_outputs(model, rows, [row["word_token_ids"] for row in rows], heads)
            for number, (layer, head) in enumerate(heads):
                probe = train_logistic_probe(
                    features[~held_out, number], labels[~held_out], len(names[attribute]), seed
                )
                probes[layer, head, attribute] = probe
                accuracy.append(
                    _accuracy_row(
                        probe, features[:, number], labels, held_out, layer, head, attribute
                    )
                )
        del model
        _free_memory()
    probes_path, accuracy_output = outputs
    probes_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        probes_path,
        **{
            f"{layer}-{head}-{attribute}-{name}": getattr(probe, name)
            for (layer, head, attribute), probe in probes.items()
            for name in ("weights", "biases", "mean", "scale")
        },
    )
    write_rows(accuracy_output, AttributeProbeAccuracy, accuracy)
    return probes


def write_probe_transfer(
    checkpoints: dict[str, Path],
    load_model: Callable[[ModelSource], LanguageModel],
    probes: dict[tuple[int, int, str], LogisticProbe],
    categories: Sequence[str],
    input_paths: tuple[Path, Path, Path],
    output: Path,
) -> None:
    """(b) 語義を足した各 run で、最小対の課題2の問題のターゲット語に当てた正解率を書く。"""
    questions_path, accuracy_path, words_path = input_paths
    heads = sorted({(layer, head) for layer, head, _ in probes})
    names = {"category": list(categories), "part-of-speech": list(PARTS_OF_SPEECH)}
    if not heads:
        write_rows(output, AttributeProbeTransfer, [])
        return
    questions = parquet.read_table(questions_path).to_pylist()
    usable = {
        (row.kind, row.number_of_senses, row.target, row.sense)
        for row in read_rows(accuracy_path, SenseAccuracy)
        if row.usable
    }
    words = {word.word: word for word in read_rows(words_path, SelectedWord)}
    results = []
    for run, checkpoint in checkpoints.items():
        if run.endswith("control"):
            continue
        attribute = "part-of-speech" if run.startswith("adjective") else "category"
        chosen = [
            question
            for question in questions
            if sense_accuracy.evaluated_in(run, question)
            and question["task"] == attribute
            and (
                question["kind"],
                question["number_of_senses"],
                question["target"],
                question["donor"] if question["cue"] == "donor" else "original",
            )
            in usable
        ]
        if not chosen:
            continue
        model = load_model(ModelSource(repository=str(checkpoint)))
        word_ids = [
            [int(token) for token in words[question["target"]].token_ids.split()]
            for question in chosen
        ]
        features = _head_outputs(model, chosen, word_ids, heads)
        # 次の run の重みを読む前に、この重みを GPU から消す
        del model
        _free_memory()
        labels = np.array([names[attribute].index(question["answer"]) for question in chosen])
        chance = 1 / len(names[attribute])
        for number, (layer, head) in enumerate(heads):
            predicted = probes[layer, head, attribute].probabilities(features[:, number])
            correct = int((predicted.argmax(axis=1) == labels).sum())
            p_value = binomial_p_value(correct, len(labels), chance)
            results.append(
                AttributeProbeTransfer(
                    run,
                    layer,
                    head,
                    attribute,
                    len(labels),
                    correct / len(labels),
                    chance,
                    p_value,
                    p_value < SIGNIFICANCE_LEVEL,
                )
            )
    write_rows(output, AttributeProbeTransfer, results)


def _head_outputs(
    model: LanguageModel,
    rows: Sequence[dict[str, object]],
    word_ids: Sequence[Sequence[int]],
    heads: Sequence[tuple[int, int]],
) -> np.ndarray:
    """問題ごとの、文の中の語（問題文で最初に現れる語）の最後のトークンでの head の出力。"""
    inputs, positions = [], []
    for row, ids in zip(rows, word_ids, strict=True):
        token_ids, spans = encode_prompt_with_word_spans(
            row["prompt_template"], list(ids), model.tokenizer
        )
        inputs.append(token_ids)
        positions.append(spans[0][-1])
    return model.head_outputs(inputs, positions, heads)


def _held_out_words(rows: Sequence[dict[str, object]], seed: int) -> set[str]:
    """クラス（正解）ごとに、HELD_OUT_SHARE の語を確認用に選ぶ。"""
    held_out = set()
    for answer in sorted({row["answer"] for row in rows}):
        members = sorted({row["word"] for row in rows if row["answer"] == answer})
        count = round(HELD_OUT_SHARE * len(members))
        held_out |= set(random.Random(f"{seed}:{answer}").sample(members, count))
    return held_out


def _accuracy_row(
    probe: LogisticProbe,
    features: np.ndarray,
    labels: np.ndarray,
    held_out: np.ndarray,
    layer: int,
    head: int,
    attribute: str,
) -> AttributeProbeAccuracy:
    predicted = probe.probabilities(features).argmax(axis=1) == labels
    chance = 1 / len(probe.biases)
    correct = int(predicted[held_out].sum())
    p_value = binomial_p_value(correct, int(held_out.sum()), chance)
    return AttributeProbeAccuracy(
        layer,
        head,
        attribute,
        float(predicted[~held_out].mean()),
        float(predicted[held_out].mean()) if held_out.any() else float("nan"),
        int(held_out.sum()),
        chance,
        p_value,
        p_value < SIGNIFICANCE_LEVEL,
    )


def _free_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()
