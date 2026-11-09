"""2つ目の語義の追加の各 run で、未知語の表現が X と Y のどちらに近いかを、層ごとに測る
（second-sense-addition measure-meaning --model）。

run ごとに、語義を足す前・学習の途中・最後の重み（measure-context-effects と同じ）で、run の
全ての未知語について、層ごとに次の値を測る。
- neutral_first・neutral_second: 中立文脈（例 "Do you know about the"）の直後の、未知語の表現と、
  1つ目の語義の名詞 X・2つ目の語義の名詞 Y の表現の cos 類似度（文脈で平均。未知語の獲得の
  意味の指標と同じ測り方）。Y に近づき、X から離れなければ多義、X から離れれば置き換え。
- cue_effect: 最小対の文の空欄の位置での、(未知語と Y の cos − 未知語と X の cos) の、Y の手がかりの
  文と X の手がかりの文の差（表現での文脈の効果。最小対で平均）。手がかりに応じて、未知語の表現が
  X か Y に寄るなら正になる。
結果は outputs/results/second-sense-addition/<モデル>/ に書く。
- meaning.parquet: run・重み・未知語・層・値の名前ごとの値
- meaning.csv: 種類 × 2つ目の露出回数 × 重み × 層ごとの、未知語の平均
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
    neutral_contexts,
    second_sense_assignments,
    second_sense_evaluation_data,
    word_finalization,
)
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.language_model import LanguageModel
from lexical_acquisition_mechanisms.minimal_pairs import BLANK
from lexical_acquisition_mechanisms.neutral_contexts import NeutralContext
from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.pretrained_model_checks import CheckedMinimalPair
from lexical_acquisition_mechanisms.run_results import RunResults
from lexical_acquisition_mechanisms.second_sense_assignments import SecondSenseAssignment
from lexical_acquisition_mechanisms.second_sense_context_effects import (
    RESULTS_DIRECTORY,
    weight_directories,
    weight_order,
)
from lexical_acquisition_mechanisms.settings import ModelSource, Settings
from lexical_acquisition_mechanisms.unknown_word_indicators import checkpoint_states
from lexical_acquisition_mechanisms.word_representations import word_representations
from lexical_acquisition_mechanisms.word_selection import SelectedWord

MEASURES = ("neutral_first", "neutral_second", "cue_effect")
BATCH_SIZE = 16


@dataclass
class MeaningSummary:
    kind: str
    second_exposure: int
    weights: str
    layer: int
    neutral_first: float
    neutral_second: float
    cue_effect: float
    words: int


def output_paths(model: str) -> tuple[Path, Path]:
    directory = RESULTS_DIRECTORY / model
    return directory / "meaning.parquet", directory / "meaning.csv"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="settings.yaml の models の名前")


def measure_meaning(settings: Settings, arguments: argparse.Namespace) -> None:
    model = arguments.model
    command = f"second-sense-addition measure-meaning --model {model}"
    directories = weight_directories(model, command)
    checked_path = second_sense_evaluation_data.SecondSensePaths().checked
    input_paths = (
        word_finalization.OUTPUT_PATH,
        second_sense_assignments.ASSIGNMENTS_PATH,
        neutral_contexts.CONTEXTS_PATH,
        checked_path,
    )
    step = Step(
        command,
        inputs=dict(
            zip(
                input_paths,
                (
                    word_finalization.COMMAND,
                    second_sense_assignments.COMMAND,
                    neutral_contexts.COMMAND,
                    second_sense_evaluation_data.QUESTIONS_COMMAND,
                ),
                strict=True,
            )
        ),
        outputs=list(output_paths(model)),
        # 学習した重みには処理の記録がないので、ファイルの大きさと更新時刻を設定として記録する
        settings={"checkpoints": checkpoint_states(directories.values())},
    )
    results = RunResults(step, arguments.recompute)
    run_unless_unchanged(
        step,
        lambda: write_meaning(
            directories,
            lambda source: LanguageModel.load(source, batch_size=BATCH_SIZE),
            input_paths,
            output_paths(model),
            results,
        ),
        arguments.recompute,
    )
    results.clear()


def write_meaning(
    directories: dict[tuple[str, str], Path],
    load_model: Callable[[ModelSource], LanguageModel],
    input_paths: tuple[Path, Path, Path, Path],
    output_paths: tuple[Path, Path],
    results: RunResults | None = None,
) -> None:
    words_path, assignments_path, contexts_path, checked_path = input_paths
    words = {word.word: word for word in read_rows(words_path, SelectedWord)}
    assignments = read_rows(assignments_path, SecondSenseAssignment)
    contexts = [row.context for row in read_rows(contexts_path, NeutralContext)]
    pairs: dict[tuple[str, str], list[CheckedMinimalPair]] = {}
    for pair in read_rows(checked_path, CheckedMinimalPair):
        pairs.setdefault((pair.target, pair.donor), []).append(pair)
    rows = []
    for run in sorted({run for run, _ in directories}):
        done = results.get(run) if results is not None else None
        if done is None:
            chosen = [row for row in assignments if row.run == run]
            done = []
            for name in [name for key, name in directories if key == run]:
                model = load_model(ModelSource(repository=str(directories[(run, name)])))
                done += _measure(model, run, name, chosen, words, contexts, pairs)
                del model
                gc.collect()
                torch.cuda.empty_cache()
            if results is not None:
                results.put(run, done)
        rows += done
    meaning_path, summary_path = output_paths
    meaning_path.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_table(arrow.Table.from_pylist(rows), meaning_path)
    write_rows(summary_path, MeaningSummary, summarize(rows))


def _ids(word: SelectedWord) -> list[int]:
    return [int(token) for token in word.token_ids.split()]


def _measure(
    model: LanguageModel,
    run: str,
    weights: str,
    assignments: Sequence[SecondSenseAssignment],
    words: dict[str, SelectedWord],
    contexts: Sequence[str],
    pairs: dict[tuple[str, str], list[CheckedMinimalPair]],
) -> list[dict[str, object]]:
    """1つの run・重みの、未知語ごと・層ごとの3つの値（縦長の行）。"""
    measured = sorted(
        {row.unknown_word for row in assignments}
        | {row.first_noun for row in assignments}
        | {row.second_noun for row in assignments}
    )
    neutral = word_representations(model, [_ids(words[word]) for word in measured], contexts)
    index = {word: number for number, word in enumerate(measured)}
    # 最小対の文ごとに、空欄の前までの書き出しの後に、未知語・X・Y を置いた表現
    inputs, positions, keys = [], [], []
    for row in assignments:
        for number, pair in enumerate(pairs.get((row.first_noun, row.second_noun), [])):
            for cue, sentence in (("first", pair.target_sentence), ("second", pair.donor_sentence)):
                prefix = sentence.split(BLANK)[0].removesuffix(" ")
                for role, word in (
                    ("unknown", row.unknown_word),
                    ("first", row.first_noun),
                    ("second", row.second_noun),
                ):
                    parts = [prefix, _ids(words[word])] if prefix else [_ids(words[word])]
                    token_ids, spans = model.tokenizer.encode_with_spans(parts)
                    inputs.append(token_ids)
                    positions.append(spans[-1][-1])
                    keys.append((row.unknown_word, number, cue, role))
    states = model.residual_streams(inputs, positions) if inputs else np.zeros((0, 0, 0))
    state = {key: states[position] for position, key in enumerate(keys)}
    rows = []
    for row in assignments:
        unknown = neutral[index[row.unknown_word]]
        values = {
            "neutral_first": _cosine(unknown, neutral[index[row.first_noun]]).mean(axis=0),
            "neutral_second": _cosine(unknown, neutral[index[row.second_noun]]).mean(axis=0),
        }
        effects = []
        for number in range(len(pairs.get((row.first_noun, row.second_noun), []))):
            lean = {}
            for cue in ("first", "second"):
                vector = state[(row.unknown_word, number, cue, "unknown")]
                lean[cue] = _cosine(
                    vector, state[(row.unknown_word, number, cue, "second")]
                ) - _cosine(vector, state[(row.unknown_word, number, cue, "first")])
            effects.append(lean["second"] - lean["first"])
        layers = len(values["neutral_first"])
        values["cue_effect"] = np.mean(effects, axis=0) if effects else np.full(layers, np.nan)
        rows += [
            {
                "run": run,
                "weights": weights,
                "unknown_word": row.unknown_word,
                "kind": row.kind,
                "second_exposure": row.second_exposure,
                "layer": layer,
                "measure": measure,
                "value": float(values[measure][layer]),
            }
            for measure in MEASURES
            for layer in range(layers)
        ]
    return rows


def _cosine(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return (first * second).sum(axis=-1) / (
        np.linalg.norm(first, axis=-1) * np.linalg.norm(second, axis=-1) + 1e-12
    )


def summarize(rows: Sequence[dict[str, object]]) -> list[MeaningSummary]:
    """種類 × 2つ目の露出回数 × 重み × 層ごとの、未知語の平均。

    最小対のない未知語は、cue_effect の平均から除く。
    """
    groups: dict[tuple, dict[str, list[float]]] = {}
    for row in rows:
        key = (row["kind"], row["second_exposure"], row["weights"], row["layer"])
        groups.setdefault(key, {measure: [] for measure in MEASURES})[row["measure"]].append(
            row["value"]
        )
    return [
        MeaningSummary(
            kind,
            exposure,
            weights,
            layer,
            *(float(np.nanmean(values[measure])) for measure in MEASURES),
            len(values["neutral_first"]),
        )
        for (kind, exposure, weights, layer), values in sorted(
            groups.items(),
            key=lambda item: (item[0][0], item[0][1], weight_order(item[0][2]), item[0][3]),
        )
    ]
