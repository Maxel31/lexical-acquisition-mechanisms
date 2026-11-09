"""2つ目の語義の追加の各 run で、課題1の文脈の効果を、学習の前・途中・最後の重みで測る
（second-sense-addition measure-context-effects --model）。

run ごとに、2つ目の語義を足す前の重み（同じ名前の未知語の獲得の run の最後の重み）と、足した後の
学習の途中（run の設定の checkpoint_fractions。例 checkpoint-25・50・75）と最後の重みで、run の
全ての未知語に、その未知語の (X, Y) の組の課題1の問題（evaluation-data build-second-sense-questions
が書いたもの）を、空欄に未知語を入れて解かせる。
- 問題ごとの選択肢の対数確率: outputs/results/second-sense-addition/<モデル>/context_choices.parquet
- 文脈の効果の表（context_effects.csv）: 種類 × 2つ目の露出回数 × 重みごとの、次の値。
  - context_effect: 未知語ごとの文脈の効果（Y の手がかりの文での Y の割合 − X の手がかりの文での
    Y の割合。最小対で平均）の平均と、未知語を単位にした bootstrap の 95% 区間。Y の割合は、X と
    Y の確率の和を 1 にしたときの Y の確率（選択肢の偏りは、2 文の差で打ち消される）。
  - second_share_at_second_cue・second_share_at_first_cue: 2 つの文での Y の割合
  - words: 未知語の数
  2つ目の露出回数が 0 回の未知語（Y を足していない）が対照になる。
"""

import argparse
import gc
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as arrow
import pyarrow.parquet as parquet
import torch

from lexical_acquisition_mechanisms import (
    run_settings,
    second_sense_assignments,
    second_sense_evaluation_data,
    word_finalization,
)
from lexical_acquisition_mechanisms.bootstrap_statistics import BOOTSTRAP_SAMPLES
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.language_model import LanguageModel
from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.run_results import RunResults
from lexical_acquisition_mechanisms.second_sense_assignments import SecondSenseAssignment
from lexical_acquisition_mechanisms.settings import ModelSource, Settings
from lexical_acquisition_mechanisms.task_prompts import choice_token_ids, encode_prompt
from lexical_acquisition_mechanisms.training_checkpoints import (
    COMPLETED_FILE_NAME,
    FINAL_DIRECTORY,
    checkpoint_names,
    final_directories,
)
from lexical_acquisition_mechanisms.unknown_word_indicators import checkpoint_states
from lexical_acquisition_mechanisms.word_selection import SelectedWord

EXPERIMENT = second_sense_assignments.EXPERIMENT
RESULTS_DIRECTORY = Path("outputs/results") / EXPERIMENT
# 2つ目の語義を足す前の重みの名前（足した後の重みは、保存したディレクトリの名前）
BEFORE = "before"
BATCH_SIZE = 16


@dataclass
class ContextEffect:
    kind: str
    second_exposure: int
    weights: str
    context_effect: float
    lower: float
    upper: float
    second_share_at_second_cue: float
    second_share_at_first_cue: float
    words: int


def output_paths(model: str) -> tuple[Path, Path]:
    directory = RESULTS_DIRECTORY / model
    return directory / "context_choices.parquet", directory / "context_effects.csv"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="settings.yaml の models の名前")


def weight_directories(model: str, command: str) -> dict[tuple[str, str], Path]:
    """(run, 重みの名前) ごとの重みのディレクトリ（学習の順）。学習を終えていない run で止まる。"""
    runs = run_settings.EXPERIMENT_RUNS[EXPERIMENT]
    directories = {}
    for run, final in final_directories(EXPERIMENT, model, runs, command).items():
        directories[(run, BEFORE)] = run_settings.first_phase_weights(model, run)
        for name in saved_weights(final.parent):
            directories[(run, name)] = final.parent / name
    return directories


def saved_weights(directory: Path) -> list[str]:
    """学習を終えた run のディレクトリに保存した重みの名前（学習の順。最後は final）。

    step の少ない学習では、途中の重みのいくつかが同じ step になり、保存されない。
    """
    record = json.loads((directory / COMPLETED_FILE_NAME).read_text(encoding="utf-8"))
    fractions = record["fingerprint"]["settings"]["checkpoint_fractions"]
    names = checkpoint_names(record["steps"], fractions)
    return [names[step] for step in sorted(names)]


def weight_order(name: str) -> int:
    """重みの名前の並び順（before、学習の途中の順、final）。"""
    if name == BEFORE:
        return -1
    if name == FINAL_DIRECTORY:
        return 101
    return int(name.removeprefix("checkpoint-"))


def measure_context_effects(settings: Settings, arguments: argparse.Namespace) -> None:
    model = arguments.model
    command = f"second-sense-addition measure-context-effects --model {model}"
    directories = weight_directories(model, command)
    inputs = {
        word_finalization.OUTPUT_PATH: word_finalization.COMMAND,
        second_sense_assignments.ASSIGNMENTS_PATH: second_sense_assignments.COMMAND,
        second_sense_evaluation_data.SecondSensePaths().questions: (
            second_sense_evaluation_data.QUESTIONS_COMMAND
        ),
    }
    step = Step(
        command,
        inputs=inputs,
        outputs=list(output_paths(model)),
        # 学習した重みには処理の記録がないので、ファイルの大きさと更新時刻を設定として記録する
        settings={
            "checkpoints": checkpoint_states(directories.values()),
            "seed": settings.words.seed,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
        },
    )
    results = RunResults(step, arguments.recompute)
    run_unless_unchanged(
        step,
        lambda: write_context_effects(
            directories,
            lambda source: LanguageModel.load(source, batch_size=BATCH_SIZE),
            tuple(inputs),
            output_paths(model),
            settings.words.seed,
            results,
        ),
        arguments.recompute,
    )
    results.clear()


def write_context_effects(
    directories: dict[tuple[str, str], Path],
    load_model: Callable[[ModelSource], LanguageModel],
    input_paths: tuple[Path, Path, Path],
    output_paths: tuple[Path, Path],
    seed: int,
    results: RunResults | None = None,
) -> None:
    words_path, assignments_path, questions_path = input_paths
    words = {word.word: word for word in read_rows(words_path, SelectedWord)}
    assignments = read_rows(assignments_path, SecondSenseAssignment)
    by_pair: dict[tuple[str, str], list[dict]] = {}
    for question in parquet.read_table(questions_path).to_pylist():
        by_pair.setdefault((question["first_noun"], question["second_noun"]), []).append(question)
    rows = []
    for run in sorted({run for run, _ in directories}):
        done = results.get(run) if results is not None else None
        if done is None:
            chosen = [
                (assignment, question)
                for assignment in assignments
                if assignment.run == run
                for question in by_pair.get((assignment.first_noun, assignment.second_noun), [])
            ]
            done = []
            for name in [name for key, name in directories if key == run]:
                model = load_model(ModelSource(repository=str(directories[(run, name)])))
                values = choice_log_probabilities(model, chosen, words)
                done += [
                    _row(run, name, assignment, question, value)
                    for (assignment, question), value in zip(chosen, values, strict=True)
                ]
                del model
                gc.collect()
                torch.cuda.empty_cache()
            if results is not None:
                results.put(run, done)
        rows += done
    choices_path, effects_path = output_paths
    choices_path.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_table(arrow.Table.from_pylist(rows), choices_path)
    write_rows(effects_path, ContextEffect, context_effects(rows, seed))


def choice_log_probabilities(
    model: LanguageModel,
    chosen: Sequence[tuple[SecondSenseAssignment, dict]],
    words: dict[str, SelectedWord],
) -> list[list[float]]:
    """問題ごとに、"Answer:" の次に来る各選択肢（問題の choices の順）の対数確率。"""
    inputs = [
        encode_prompt(
            question["prompt_template"],
            [int(token) for token in words[assignment.unknown_word].token_ids.split()],
            model.tokenizer,
        )
        for assignment, question in chosen
    ]
    candidates = sorted({choice for _, question in chosen for choice in question["choices"]})
    column = {choice: index for index, choice in enumerate(candidates)}
    values = model.next_token_log_probabilities(
        inputs,
        [len(tokens) - 1 for tokens in inputs],
        choice_token_ids(candidates, model.tokenizer),
    )
    return [
        [float(row[column[choice]]) for choice in question["choices"]]
        for (_, question), row in zip(chosen, values, strict=True)
    ]


def _row(
    run: str, weights: str, assignment: SecondSenseAssignment, question: dict, value: list[float]
) -> dict[str, object]:
    return {
        "run": run,
        "weights": weights,
        "unknown_word": assignment.unknown_word,
        "kind": assignment.kind,
        "second_exposure": assignment.second_exposure,
        "first_noun": question["first_noun"],
        "second_noun": question["second_noun"],
        "pair": question["pair"],
        "cue": question["cue"],
        "choices": question["choices"],
        "log_probabilities": value,
    }


def context_effects(rows: Sequence[dict[str, object]], seed: int) -> list[ContextEffect]:
    """種類 × 2つ目の露出回数 × 重みごとの、未知語ごとの文脈の効果の平均と 95% 区間。"""
    shares: dict[tuple, dict[str, float]] = {}
    for row in rows:
        probabilities = dict(zip(row["choices"], np.exp(row["log_probabilities"]), strict=True))
        first, second = probabilities[row["first_noun"]], probabilities[row["second_noun"]]
        key = (row["kind"], row["second_exposure"], row["weights"], row["run"], row["unknown_word"])
        shares.setdefault(key + (row["pair"],), {})[row["cue"]] = second / (first + second)
    per_word: dict[tuple, list[tuple[float, float]]] = {}
    for key, cues in shares.items():
        per_word.setdefault(key[:5], []).append((cues["second"], cues["first"]))
    groups: dict[tuple, list[np.ndarray]] = {}
    for key, values in per_word.items():
        groups.setdefault(key[:3], []).append(np.mean(values, axis=0))
    random_numbers = np.random.default_rng(seed)
    table = []
    for (kind, exposure, weights), values in sorted(
        groups.items(), key=lambda item: (item[0][0], item[0][1], weight_order(item[0][2]))
    ):
        values = np.array(values)
        effects = values[:, 0] - values[:, 1]
        samples = random_numbers.choice(effects, (BOOTSTRAP_SAMPLES, len(effects))).mean(axis=1)
        lower, upper = np.quantile(samples, [0.025, 0.975])
        table.append(
            ContextEffect(
                kind,
                exposure,
                weights,
                float(effects.mean()),
                float(lower),
                float(upper),
                float(values[:, 0].mean()),
                float(values[:, 1].mean()),
                len(effects),
            )
        )
    return table
