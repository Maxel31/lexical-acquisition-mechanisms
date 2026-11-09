"""フェーズの学習で変わった重みを部品ごとに移し、課題1の値がどれだけ戻るかを測る
（second-sense-addition graft-weights --model --phase）。

移し方（sufficiency・necessity）と部品は weight_grafting のとおり。
- --phase second-sense-addition: 2つ目の語義を足す前（未知語の獲得の最後の重み）→ 足した後。
  2つ目の露出回数が 0 回（対照）と 1000 回の未知語で、文脈の効果（measure-context-effects と
  同じ）を測る。
- --phase unknown-word-acquisition: 学習前の公開モデル → 未知語の獲得の最後の重み。2つ目の語義の
  追加の割り当てにある未知語（統合まで済んだ未知語）で、X の手がかりの文で X を選ぶ割合を測る。
結果は outputs/results/second-sense-addition/<モデル>/ に書く。
- grafting-<フェーズ>.parquet: run・部品・移し方・問題ごとの選択肢の対数確率
- grafting-<フェーズ>.csv: 部品 × 移し方 × 種類 × 2つ目の露出回数 × 未知語のトークン数（all は全て）
  ごとの、値の平均（未知語ごとの平均の平均）と、θ1 から θ2 までの変化のうち戻した割合（recovered）
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
from transformers import AutoModelForCausalLM

from lexical_acquisition_mechanisms import (
    run_settings,
    second_sense_assignments,
    second_sense_evaluation_data,
    word_finalization,
)
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.language_model import LanguageModel
from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.run_results import RunResults
from lexical_acquisition_mechanisms.second_sense_assignments import SecondSenseAssignment
from lexical_acquisition_mechanisms.second_sense_context_effects import (
    RESULTS_DIRECTORY,
    choice_log_probabilities,
)
from lexical_acquisition_mechanisms.settings import ModelSource, Settings
from lexical_acquisition_mechanisms.training_checkpoints import final_directories
from lexical_acquisition_mechanisms.unknown_word_indicators import checkpoint_states
from lexical_acquisition_mechanisms.weight_grafting import grafted_weights
from lexical_acquisition_mechanisms.word_selection import SelectedWord
from lexical_acquisition_mechanisms.word_tokenizer import WordTokenizer

PHASES = ("unknown-word-acquisition", second_sense_assignments.EXPERIMENT)
# 2つ目の語義の追加で測る、2つ目の露出回数（0 回は対照）
SECOND_EXPOSURES = (0, 1000)
BATCH_SIZE = 16


@dataclass
class GraftingSummary:
    configuration: str
    mode: str
    kind: str
    second_exposure: int
    token_count: str
    value: float
    recovered: float
    words: int


def output_paths(model: str, phase: str) -> tuple[Path, Path]:
    directory = RESULTS_DIRECTORY / model
    return directory / f"grafting-{phase}.parquet", directory / f"grafting-{phase}.csv"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="settings.yaml の models の名前")
    parser.add_argument(
        "--phase", choices=PHASES, required=True, help="重みを移すフェーズ（その学習の前 → 後）"
    )


def phase_weights(
    settings: Settings, model: str, phase: str, command: str
) -> dict[str, tuple[ModelSource, Path]]:
    """run ごとの、フェーズの前の重みと後の重み。学習を終えていない run があれば止まる。"""
    runs = run_settings.EXPERIMENT_RUNS[second_sense_assignments.EXPERIMENT]
    first = final_directories("unknown-word-acquisition", model, runs, command)
    if phase == "unknown-word-acquisition":
        return {run: (settings.models[model], first[run]) for run in runs}
    second = final_directories(second_sense_assignments.EXPERIMENT, model, runs, command)
    return {run: (ModelSource(repository=str(first[run])), second[run]) for run in runs}


def graft_weights(settings: Settings, arguments: argparse.Namespace) -> None:
    model, phase = arguments.model, arguments.phase
    command = f"second-sense-addition graft-weights --model {model} --phase {phase}"
    weights = phase_weights(settings, model, phase, command)
    questions_path = second_sense_evaluation_data.SecondSensePaths().questions
    inputs = {
        word_finalization.OUTPUT_PATH: word_finalization.COMMAND,
        second_sense_assignments.ASSIGNMENTS_PATH: second_sense_assignments.COMMAND,
        questions_path: second_sense_evaluation_data.QUESTIONS_COMMAND,
    }
    input_paths = tuple(inputs)
    step = Step(
        command,
        inputs=inputs,
        outputs=list(output_paths(model, phase)),
        # 学習した重みには処理の記録がないので、ファイルの大きさと更新時刻を設定として記録する
        settings={
            "checkpoints": checkpoint_states([after for _, after in weights.values()]),
            "before": {run: before.model_dump() for run, (before, _) in weights.items()},
            "second_exposures": SECOND_EXPOSURES,
        },
    )
    results = RunResults(step, arguments.recompute)
    run_unless_unchanged(
        step,
        lambda: write_grafting(
            phase, weights, _load_network, input_paths, output_paths(model, phase), results
        ),
        arguments.recompute,
    )
    results.clear()


def _load_network(source: ModelSource) -> torch.nn.Module:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return AutoModelForCausalLM.from_pretrained(
        source.repository, revision=source.revision, dtype=torch.float32
    ).to(device)


def write_grafting(
    phase: str,
    weights: dict[str, tuple[ModelSource, Path]],
    load_network: Callable[[ModelSource], torch.nn.Module],
    input_paths: tuple[Path, Path, Path],
    output_paths: tuple[Path, Path],
    results: RunResults | None = None,
) -> None:
    words_path, assignments_path, questions_path = input_paths
    words = {word.word: word for word in read_rows(words_path, SelectedWord)}
    assignments = read_rows(assignments_path, SecondSenseAssignment)
    by_pair: dict[tuple[str, str], list[dict]] = {}
    for question in parquet.read_table(questions_path).to_pylist():
        by_pair.setdefault((question["first_noun"], question["second_noun"]), []).append(question)
    rows = []
    for run, (before, after) in weights.items():
        done = results.get(run) if results is not None else None
        if done is None:
            chosen = [
                (row, question)
                for row in assignments
                if row.run == run
                and (phase == "unknown-word-acquisition" or row.second_exposure in SECOND_EXPOSURES)
                for question in by_pair.get((row.first_noun, row.second_noun), [])
            ]
            done = _graft_run(run, before, after, load_network, chosen, words)
            if results is not None:
                results.put(run, done)
        rows += done
    grafting_path, summary_path = output_paths
    grafting_path.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_table(arrow.Table.from_pylist(rows), grafting_path)
    write_rows(summary_path, GraftingSummary, summarize(phase, rows))


def _graft_run(
    run: str,
    before: ModelSource,
    after: Path,
    load_network: Callable[[ModelSource], torch.nn.Module],
    chosen: Sequence[tuple[SecondSenseAssignment, dict]],
    words: dict[str, SelectedWord],
) -> list[dict[str, object]]:
    network = load_network(before)
    model = LanguageModel(network, WordTokenizer.from_model(before), BATCH_SIZE)
    rows = []
    for name, mode in grafted_weights(network, after / "model.safetensors"):
        values = choice_log_probabilities(model, chosen, words)
        rows += [
            {
                "run": run,
                "configuration": name,
                "mode": mode,
                "unknown_word": row.unknown_word,
                "token_count": row.token_count,
                "kind": row.kind,
                "second_exposure": row.second_exposure,
                "first_noun": question["first_noun"],
                "second_noun": question["second_noun"],
                "pair": question["pair"],
                "cue": question["cue"],
                "choices": question["choices"],
                "log_probabilities": value,
            }
            for (row, question), value in zip(chosen, values, strict=True)
        ]
    del network, model
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def summarize(phase: str, rows: Sequence[dict[str, object]]) -> list[GraftingSummary]:
    """部品 × 移し方 × 種類 × 2つ目の露出回数 × トークン数ごとの値と、戻した割合。

    値は、2つ目の語義の追加では文脈の効果、未知語の獲得では X の手がかりの文で X を選ぶ割合。
    """
    shares: dict[tuple, dict[str, float]] = {}
    for row in rows:
        probabilities = dict(zip(row["choices"], np.exp(row["log_probabilities"]), strict=True))
        first, second = probabilities[row["first_noun"]], probabilities[row["second_noun"]]
        key = (row["configuration"], row["mode"], row["kind"], row["second_exposure"])
        key += (row["token_count"], row["run"], row["unknown_word"], row["pair"])
        shares.setdefault(key, {})[row["cue"]] = second / (first + second)
    per_word: dict[tuple, list[float]] = {}
    for key, cues in shares.items():
        value = cues["second"] - cues["first"] if phase != PHASES[0] else 1 - cues["first"]
        per_word.setdefault(key[:7], []).append(value)
    groups: dict[tuple, list[float]] = {}
    for key, values in per_word.items():
        for token_count in (str(key[4]), "all"):
            groups.setdefault((*key[:4], token_count), []).append(float(np.mean(values)))
    means = {key: float(np.mean(values)) for key, values in groups.items()}
    table = []
    for (name, mode, kind, exposure, token_count), value in sorted(means.items()):
        start = means[("before", "baseline", kind, exposure, token_count)]
        end = means[("after", "baseline", kind, exposure, token_count)]
        recovered = (value - start) / (end - start) if end != start else float("nan")
        table.append(
            GraftingSummary(
                name,
                mode,
                kind,
                exposure,
                token_count,
                value,
                recovered,
                len(groups[(name, mode, kind, exposure, token_count)]),
            )
        )
    return table
