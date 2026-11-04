"""最小対の文で、ターゲット語の語義を使い分けられているかを測る
（sense-addition measure-sense-accuracy --model）。

語義の追加の各 run の最後の重みで、その run の条件（足す語義の種類と数）の問題（evaluation-data
finalize が書いたもの）を解かせる。対照の run は、対応する条件（名詞の対照は同カテゴリと別カテゴリ、
形容詞の対照は形容詞）の問題を解かせる。
- 問題ごとの答え: outputs/results/sense-addition/<モデル>/answers.parquet
- 課題1の正解率（sense_accuracy.csv）: 条件 × ターゲット語 × 語義（donor か、元の語義）ごとに、
  正解率・対照の run の正解率・その差・偶然の水準（1 ÷ 選択肢の数）を超えるかの片側の二項検定。
  超えない組は、head の地図と属性の特定（手順2・3）で使わない（表の usable）。
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
    pretrained_model_checks,
    sense_addition_assignments,
    word_finalization,
)
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.language_model import LanguageModel
from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.settings import ModelSource, Settings
from lexical_acquisition_mechanisms.task_accuracy import (
    SIGNIFICANCE_LEVEL,
    answer_questions,
    binomial_p_value,
)
from lexical_acquisition_mechanisms.training_checkpoints import final_directories
from lexical_acquisition_mechanisms.unknown_word_indicators import checkpoint_states
from lexical_acquisition_mechanisms.word_selection import SelectedWord

EXPERIMENT = sense_addition_assignments.EXPERIMENT
RESULTS_DIRECTORY = Path("outputs/results") / EXPERIMENT
CONTROL_OF_KIND = {
    "same-category": "noun-control",
    "other-category": "noun-control",
    "adjective": "adjective-control",
}
BATCH_SIZE = 16


@dataclass
class SenseAccuracy:
    kind: str
    number_of_senses: int
    target: str
    # 手がかりが示す語義（donor か、元の語義なら original）
    sense: str
    questions: int
    accuracy: float
    control_accuracy: float
    difference: float
    chance: float
    p_value: float
    usable: bool


def result_paths(model: str) -> tuple[Path, Path]:
    directory = RESULTS_DIRECTORY / model
    return directory / "answers.parquet", directory / "sense_accuracy.csv"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="settings.yaml の models の名前")


def final_checkpoints(model: str, command: str) -> dict[str, Path]:
    """語義の追加の各 run の、最後の重みのディレクトリ（学習を終えていない run があれば止まる）。"""
    return final_directories(EXPERIMENT, model, sense_addition_assignments.run_names(), command)


def measure_sense_accuracy(settings: Settings, arguments: argparse.Namespace) -> None:
    model = arguments.model
    command = f"sense-addition measure-sense-accuracy --model {model}"
    checkpoints = final_checkpoints(model, command)
    pretrained_model_checks.require_usable_tasks(model, ["sense"], command)
    step = Step(
        command,
        inputs={
            word_finalization.OUTPUT_PATH: word_finalization.COMMAND,
            evaluation_finalization.QUESTIONS_PATH: evaluation_finalization.COMMAND,
            pretrained_model_checks.TASK_ACCURACY_PATH: pretrained_model_checks.COMMAND,
        },
        outputs=list(result_paths(model)),
        settings={"checkpoints": checkpoint_states(list(checkpoints.values()))},
    )
    run_unless_unchanged(
        step,
        lambda: write_sense_accuracy(
            checkpoints,
            lambda source: LanguageModel.load(source, batch_size=BATCH_SIZE),
            word_finalization.OUTPUT_PATH,
            evaluation_finalization.QUESTIONS_PATH,
            result_paths(model),
        ),
        arguments.recompute,
    )


def write_sense_accuracy(
    checkpoints: dict[str, Path],
    load_model: Callable[[ModelSource], LanguageModel],
    words_path: Path,
    questions_path: Path,
    output_paths: tuple[Path, Path],
) -> None:
    answers_path, accuracy_path = output_paths
    words = {word.word: word for word in read_rows(words_path, SelectedWord)}
    questions = parquet.read_table(questions_path).to_pylist()
    answers = []
    for run, checkpoint in checkpoints.items():
        chosen = [
            (index, question)
            for index, question in enumerate(questions)
            if evaluated_in(run, question)
        ]
        if not chosen:
            continue
        model = load_model(ModelSource(repository=str(checkpoint)))
        replies = answer_questions(
            model,
            [question["prompt_template"] for _, question in chosen],
            [
                [int(token) for token in words[question["target"]].token_ids.split()]
                for _, question in chosen
            ],
            [question["choices"] for _, question in chosen],
        )
        answers += [
            {"run": run, "question": index, "answer": reply, "correct": reply == question["answer"]}
            for (index, question), reply in zip(chosen, replies, strict=True)
        ]
        del model
        gc.collect()
        torch.cuda.empty_cache()
    answers_path.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_table(arrow.Table.from_pylist(answers), answers_path)
    write_rows(accuracy_path, SenseAccuracy, sense_accuracy(questions, answers))


def usable_senses(accuracy_path: Path) -> set[tuple[str, int, str, str]]:
    """手順2・3で使える（条件, N, ターゲット語, 語義）。語義は donor の語か original。"""
    return {
        (row.kind, row.number_of_senses, row.target, row.sense)
        for row in read_rows(accuracy_path, SenseAccuracy)
        if row.usable
    }


def evaluated_in(run: str, question: dict[str, object]) -> bool:
    """問題を、その run で解かせるか（条件の run と、対応する対照の run）。"""
    condition = f"{question['kind']}-{question['number_of_senses']}"
    return run == condition or run == CONTROL_OF_KIND[question["kind"]]


def sense_accuracy(
    questions: Sequence[dict[str, object]], answers: Sequence[dict[str, object]]
) -> list[SenseAccuracy]:
    """課題1の、条件 × ターゲット語 × 語義ごとの正解率と、対照との差・二項検定。"""
    correct: dict[tuple, list[bool]] = {}
    control_correct: dict[tuple, list[bool]] = {}
    chances: dict[tuple, list[float]] = {}
    for answer in answers:
        question = questions[answer["question"]]
        if question["task"] != "sense":
            continue
        sense = question["donor"] if question["cue"] == "donor" else "original"
        key = (question["kind"], question["number_of_senses"], question["target"], sense)
        if answer["run"].endswith("control"):
            control_correct.setdefault(key, []).append(bool(answer["correct"]))
        else:
            correct.setdefault(key, []).append(bool(answer["correct"]))
            chances.setdefault(key, []).append(1 / len(question["choices"]))
    rows = []
    for key, values in correct.items():
        kind, number, target, sense = key
        control = control_correct.get(key, [])
        chance = float(np.mean(chances[key]))
        p_value = binomial_p_value(sum(values), len(values), chance)
        accuracy = float(np.mean(values))
        control_accuracy = float(np.mean(control)) if control else float("nan")
        rows.append(
            SenseAccuracy(
                kind,
                number,
                target,
                sense,
                len(values),
                accuracy,
                control_accuracy,
                accuracy - control_accuracy,
                chance,
                p_value,
                p_value < SIGNIFICANCE_LEVEL,
            )
        )
    return rows
