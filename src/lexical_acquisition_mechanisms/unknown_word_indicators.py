"""未知語の3つの指標を、各 run の最後の重みで層ごとに測る。

unknown-word-acquisition measure-indicators の処理。

語を中立文脈の各条件の直後に置き、条件で平均した値を出す。
- 形への慣れ（form）: 語の2つ目以降のトークンの驚き度 −log p(t_k | 文脈, t_1…t_{k−1}) の平均
  （3トークンの語は、2つ目（form-second）と3つ目（form-third）も出す）。1トークンの語には出さない。
- 語としての統合（integration）: 実在語らしさのプローブ（語のトークン数のもの）が、層ごとに実在語と
  判定する確率。
- 意味（meaning）: 層ごとの、語の内部表現と、その run で割り当てた名詞の内部表現（同じ文脈）の
  cos 類似度。
同じ run で、正規化の基準も測る: 既知の名詞（2・3トークン。1トークンは名詞72語）の形への慣れと
統合（m1）と、名詞を人工的に2・3トークンに分けた形と元の名詞の意味（m1）。
結果は outputs/results/unknown-word-acquisition/<モデル>/indicators.parquet（縦長の表）に書く。
"""

import argparse
import gc
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pyarrow as arrow
import pyarrow.parquet as parquet
import torch

from lexical_acquisition_mechanisms import (
    neutral_contexts,
    probe_and_normalization_words,
    real_word_probes,
    run_settings,
    unknown_word_assignments,
    word_finalization,
)
from lexical_acquisition_mechanisms.csv_tables import read_rows
from lexical_acquisition_mechanisms.language_model import LanguageModel
from lexical_acquisition_mechanisms.logistic_probes import LogisticProbe
from lexical_acquisition_mechanisms.neutral_contexts import NeutralContext
from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.probe_and_normalization_words import NormalizationWord
from lexical_acquisition_mechanisms.settings import ModelSource, Settings
from lexical_acquisition_mechanisms.training_checkpoints import final_directories
from lexical_acquisition_mechanisms.unknown_word_assignments import UnknownWordAssignment
from lexical_acquisition_mechanisms.word_representations import word_representations
from lexical_acquisition_mechanisms.word_selection import SelectedWord

EXPERIMENT = unknown_word_assignments.EXPERIMENT
RESULTS_DIRECTORY = Path("outputs/results") / EXPERIMENT
BATCH_SIZE = 32
# 形への慣れの行の層（層によらない値）
NO_LAYER = -1


def indicators_path(model: str) -> Path:
    return RESULTS_DIRECTORY / model / "indicators.parquet"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="settings.yaml の models の名前")


def measure_indicators(settings: Settings, arguments: argparse.Namespace) -> None:
    model = arguments.model
    runs = run_settings.EXPERIMENT_RUNS[EXPERIMENT]
    command = f"unknown-word-acquisition measure-indicators --model {model}"
    checkpoints = list(final_directories(EXPERIMENT, model, runs, command).values())
    probes_path, _ = real_word_probes.probe_paths(model)
    step = Step(
        f"unknown-word-acquisition measure-indicators --model {model}",
        inputs={
            word_finalization.OUTPUT_PATH: word_finalization.COMMAND,
            unknown_word_assignments.ASSIGNMENTS_PATH: unknown_word_assignments.COMMAND,
            neutral_contexts.CONTEXTS_PATH: neutral_contexts.COMMAND,
            probe_and_normalization_words.NORMALIZATION_WORDS_PATH: (
                probe_and_normalization_words.COMMAND
            ),
            probes_path: f"unknown-word-acquisition train-real-word-probes --model {model}",
        },
        outputs=[indicators_path(model)],
        # 学習した重みには処理の記録がないので、ファイルの大きさと更新時刻を設定として記録する
        settings={"checkpoints": checkpoint_states(checkpoints)},
    )
    run_unless_unchanged(
        step,
        lambda: write_indicators(
            settings,
            dict(zip(runs, checkpoints, strict=True)),
            lambda source: LanguageModel.load(source, batch_size=BATCH_SIZE),
            probes_path,
            indicators_path(model),
        ),
        arguments.recompute,
    )


def checkpoint_states(checkpoints: Sequence[Path]) -> dict[str, list[list[object]]]:
    """重みのディレクトリごとの、ファイルの名前・大きさ・更新時刻（学習し直したら変わる）。"""
    return {
        str(path): [
            [file.name, file.stat().st_size, file.stat().st_mtime_ns]
            for file in sorted(path.iterdir())
        ]
        for path in checkpoints
    }


def write_indicators(
    settings: Settings,
    checkpoints: dict[str, Path],
    load_model: Callable[[ModelSource], LanguageModel],
    probes_path: Path,
    output_path: Path,
    input_paths: Sequence[Path] = (
        word_finalization.OUTPUT_PATH,
        unknown_word_assignments.ASSIGNMENTS_PATH,
        neutral_contexts.CONTEXTS_PATH,
        probe_and_normalization_words.NORMALIZATION_WORDS_PATH,
    ),
) -> None:
    words_path, assignments_path, contexts_path, normalization_path = input_paths
    words = {word.word: word for word in read_rows(words_path, SelectedWord)}
    assignments = read_rows(assignments_path, UnknownWordAssignment)
    contexts = [row.context for row in read_rows(contexts_path, NeutralContext)]
    references = read_rows(normalization_path, NormalizationWord)
    probes = np.load(probes_path)
    rows = []
    for run, checkpoint in checkpoints.items():
        model = load_model(ModelSource(repository=str(checkpoint)))
        run_assignments = [row for row in assignments if row.run == run]
        rows += _measure_run(model, run, run_assignments, words, references, contexts, probes)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_table(arrow.Table.from_pylist(rows), output_path)


def _measure_run(
    model: LanguageModel,
    run: str,
    assignments: Sequence[UnknownWordAssignment],
    words: dict[str, SelectedWord],
    references: Sequence[NormalizationWord],
    contexts: Sequence[str],
    probes,
) -> list[dict[str, object]]:
    """1つの run の、未知語と正規化の基準の指標（縦長の行）。"""
    unknown = [word for word in words.values() if word.role == "unknown"]
    nouns = [word for word in words.values() if word.role == "noun"]
    assigned = {row.unknown_word: row for row in assignments}
    measured = [
        ("unknown", word.word, _ids(word.token_ids), word.spelling_closeness) for word in unknown
    ]
    measured += [("noun", noun.word, _ids(noun.token_ids), "") for noun in nouns]
    measured += [
        (reference.kind, reference.text, _ids(reference.token_ids), reference.source_noun)
        for reference in references
    ]
    states = word_representations(model, [ids for _, _, ids, _ in measured], contexts)
    surprisals = _surprisals(model, [ids for _, _, ids, _ in measured], contexts)
    index_of = {(kind, text): index for index, (kind, text, _, _) in enumerate(measured)}
    rows = []
    for index, (kind, text, ids, extra) in enumerate(measured):
        token_count = len(ids)
        row = {"run": run, "kind": kind, "word": text, "token_count": token_count}
        if kind == "unknown":
            assignment = assigned.get(text)
            row |= {
                "spelling_closeness": extra,
                "noun": assignment.noun if assignment else "",
                "exposure": assignment.exposure if assignment else 0,
            }
            partner = index_of[("noun", assignment.noun)] if assignment else None
        elif kind == "split-noun":
            row |= {"spelling_closeness": "", "noun": extra, "exposure": -1}
            partner = index_of[("noun", extra)]
        else:
            row |= {"spelling_closeness": "", "noun": "", "exposure": -1}
            partner = None
        # 形への慣れと統合は未知語と既知の語で、意味は未知語と分けた名詞で測る
        if token_count > 1 and kind != "split-noun":
            values = surprisals[index]
            form = {"form": values.mean()}
            if token_count == 3:
                form |= {"form-second": values[0], "form-third": values[1]}
            rows += [
                row | {"layer": NO_LAYER, "indicator": name, "value": float(value)}
                for name, value in form.items()
            ]
        for layer in range(states.shape[2]):
            key = f"{token_count}-{layer}"
            if kind != "split-noun" and f"{key}-weights" in probes:
                probe = LogisticProbe(
                    *(probes[f"{key}-{name}"] for name in ("weights", "biases", "mean", "scale"))
                )
                probability = probe.probabilities(states[index, :, layer])[:, 1].mean()
                rows.append(
                    row | {"layer": layer, "indicator": "integration", "value": float(probability)}
                )
            if partner is not None:
                cosine = _cosine(states[index, :, layer], states[partner, :, layer]).mean()
                rows.append(row | {"layer": layer, "indicator": "meaning", "value": float(cosine)})
    return rows


def _surprisals(
    model: LanguageModel, words: Sequence[list[int]], contexts: Sequence[str]
) -> list[np.ndarray]:
    """語ごとの、2つ目以降のトークンの驚き度（条件で平均。トークンの順）。"""
    inputs = []
    spans = []
    for token_ids in words:
        for context in contexts:
            parts = [context, token_ids] if context else [token_ids]
            ids, part_spans = model.tokenizer.encode_with_spans(parts)
            inputs.append(ids)
            spans.append(part_spans[-1])
    log_probabilities = model.token_log_probabilities(inputs)
    values = [
        -(values[span.start + 1 : span.stop])
        for values, span in zip(log_probabilities, spans, strict=True)
    ]
    per_word = len(contexts)
    return [
        np.mean(values[index * per_word : (index + 1) * per_word], axis=0)
        for index in range(len(words))
    ]


def _cosine(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """行ごとの cos 類似度。"""
    return (first * second).sum(axis=-1) / (
        np.linalg.norm(first, axis=-1) * np.linalg.norm(second, axis=-1) + 1e-12
    )


def _ids(text: str) -> list[int]:
    return [int(token) for token in text.split()]
