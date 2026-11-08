"""2つ目の語義の追加の試行で、未知語の表現が X と Y のどちらに近いかを測る（試行の作業。38|）。

- measure <run>（GPU 1枚）: その run の、2つ目の語義を足す前の重みと、足した後の学習の 25・50・75・
  100% の重みで、run の未知語ごとに、層ごとの次の値を測り、
  outputs/partial-results/second-sense-evaluation/meaning/<run>.parquet に書く。
  - neutral_first・neutral_second: 中立文脈（例 "Do you know about the"）の直後の、未知語の表現と
    X・Y の表現の cos 類似度（文脈で平均。未知語の獲得の意味の指標と同じ測り方）
  - cue_effect: 最小対の文の空欄の位置での、(未知語と Y の cos − 未知語と X の cos) の、
    Y の手がかりの文と X の手がかりの文の差（表現での文脈の効果。最小対で平均）
- summarize（CPU）: 層を選び（既定は 12 と 16。未知語の獲得で、意味と統合を測った層）、重み ×
  2つ目の露出回数 × 種類ごとに平均して meaning_summary.txt に書く。
"""

import gc
import sys
from pathlib import Path

import numpy as np
import pandas
import pyarrow as arrow
import pyarrow.parquet as parquet
import torch

from lexical_acquisition_mechanisms import (
    neutral_contexts,
    second_sense_assignments,
    word_finalization,
)
from lexical_acquisition_mechanisms.csv_tables import read_rows
from lexical_acquisition_mechanisms.language_model import LanguageModel
from lexical_acquisition_mechanisms.minimal_pairs import BLANK, MinimalPairCandidate
from lexical_acquisition_mechanisms.neutral_contexts import NeutralContext
from lexical_acquisition_mechanisms.run_settings import first_phase_weights
from lexical_acquisition_mechanisms.second_sense_assignments import SecondSenseAssignment
from lexical_acquisition_mechanisms.settings import ModelSource
from lexical_acquisition_mechanisms.word_representations import word_representations
from lexical_acquisition_mechanisms.word_selection import SelectedWord

DIRECTORY = Path("outputs/partial-results/second-sense-evaluation")
MODEL = "OLMo-2-0425-1B"
CHECKPOINTS = ["before", "checkpoint-25", "checkpoint-50", "checkpoint-75", "final"]
LAYERS = (12, 16)


def weights(run: str, checkpoint: str) -> Path:
    if checkpoint == "before":
        return first_phase_weights(MODEL, run)
    experiment = second_sense_assignments.EXPERIMENT
    return Path("outputs/checkpoints") / experiment / MODEL / run / checkpoint


def cosine(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return (first * second).sum(axis=-1) / (
        np.linalg.norm(first, axis=-1) * np.linalg.norm(second, axis=-1) + 1e-12
    )


def sentence_states(
    model: LanguageModel, prefixes: list[str], word_ids: list[list[int]]
) -> np.ndarray:
    """文の書き出しの直後に語を置いたときの、語の最後のトークンでの全層の表現。"""
    inputs, positions = [], []
    for prefix, ids in zip(prefixes, word_ids, strict=True):
        token_ids, spans = model.tokenizer.encode_with_spans([prefix, ids])
        inputs.append(token_ids)
        positions.append(spans[-1][-1])
    return model.residual_streams(inputs, positions)


def measure(run: str) -> None:
    words = {word.word: word for word in read_rows(word_finalization.OUTPUT_PATH, SelectedWord)}
    ids = {name: [int(token) for token in word.token_ids.split()] for name, word in words.items()}
    contexts = [row.context for row in read_rows(neutral_contexts.CONTEXTS_PATH, NeutralContext)]
    assignments = [
        row
        for row in read_rows(second_sense_assignments.ASSIGNMENTS_PATH, SecondSenseAssignment)
        if row.run == run
    ]
    pairs: dict[tuple[str, str], list[MinimalPairCandidate]] = {}
    for pair in read_rows(DIRECTORY / "checked_minimal_pairs.csv", MinimalPairCandidate):
        pairs.setdefault((pair.target, pair.donor), []).append(pair)
    neutral_words = sorted(
        {row.unknown_word for row in assignments}
        | {row.first_noun for row in assignments}
        | {row.second_noun for row in assignments}
    )
    # 最小対の文ごとに、未知語・X・Y を空欄に置く
    prefixes, placed, keys = [], [], []
    for row in assignments:
        for number, pair in enumerate(pairs.get((row.first_noun, row.second_noun), [])):
            for cue, sentence in (("first", pair.target_sentence), ("second", pair.donor_sentence)):
                prefix = sentence.split(BLANK)[0].removesuffix(" ")
                for role, word in (
                    ("unknown", row.unknown_word),
                    ("first", row.first_noun),
                    ("second", row.second_noun),
                ):
                    prefixes.append(prefix)
                    placed.append(ids[word])
                    keys.append((row.unknown_word, number, cue, role))
    rows = []
    for checkpoint in CHECKPOINTS:
        model = LanguageModel.load(ModelSource(repository=str(weights(run, checkpoint))))
        neutral = word_representations(model, [ids[word] for word in neutral_words], contexts)
        index = {word: position for position, word in enumerate(neutral_words)}
        in_sentences = sentence_states(model, prefixes, placed)
        state = {key: in_sentences[position] for position, key in enumerate(keys)}
        for row in assignments:
            base = {
                "run": run,
                "checkpoint": checkpoint,
                "unknown_word": row.unknown_word,
                "token_count": row.token_count,
                "second_exposure": row.second_exposure,
                "kind": row.kind,
            }
            unknown = neutral[index[row.unknown_word]]
            first = cosine(unknown, neutral[index[row.first_noun]]).mean(axis=0)
            second = cosine(unknown, neutral[index[row.second_noun]]).mean(axis=0)
            numbers = range(len(pairs.get((row.first_noun, row.second_noun), [])))
            effects = []
            for number in numbers:
                lean = {}
                for cue in ("first", "second"):
                    word = state[(row.unknown_word, number, cue, "unknown")]
                    lean[cue] = cosine(
                        word, state[(row.unknown_word, number, cue, "second")]
                    ) - cosine(word, state[(row.unknown_word, number, cue, "first")])
                effects.append(lean["second"] - lean["first"])
            cue_effect = np.mean(effects, axis=0) if effects else np.full(len(first), np.nan)
            for layer in range(len(first)):
                rows += [
                    base | {"layer": layer, "measure": name, "value": float(values[layer])}
                    for name, values in (
                        ("neutral_first", first),
                        ("neutral_second", second),
                        ("cue_effect", cue_effect),
                    )
                ]
        print(f"{run} {checkpoint}", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    (DIRECTORY / "meaning").mkdir(parents=True, exist_ok=True)
    parquet.write_table(arrow.Table.from_pylist(rows), DIRECTORY / "meaning" / f"{run}.parquet")


def summarize() -> None:
    table = pandas.concat(
        parquet.read_table(path).to_pandas()
        for path in sorted((DIRECTORY / "meaning").glob("*.parquet"))
    )
    table = table[table["layer"].isin(LAYERS)]
    table["checkpoint"] = pandas.Categorical(table["checkpoint"], CHECKPOINTS, ordered=True)
    summary = table.pivot_table(
        index=["layer", "kind", "second_exposure", "checkpoint"],
        columns="measure",
        values="value",
        aggfunc="mean",
        observed=True,
    )
    text = summary.round(3).to_string()
    (DIRECTORY / "meaning_summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    if sys.argv[1] == "measure":
        measure(sys.argv[2])
    else:
        summarize()
