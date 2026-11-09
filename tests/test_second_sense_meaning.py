"""2つ目の語義の追加の、未知語の表現が X と Y のどちらに近いかを測る処理のテスト。"""

import math
from pathlib import Path

import pyarrow.parquet as parquet
import pytest

from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.language_model import LanguageModel
from lexical_acquisition_mechanisms.neutral_contexts import NeutralContext
from lexical_acquisition_mechanisms.pretrained_model_checks import CheckedMinimalPair
from lexical_acquisition_mechanisms.second_sense_assignments import SecondSenseAssignment
from lexical_acquisition_mechanisms.second_sense_meaning import (
    MEASURES,
    MeaningSummary,
    summarize,
    write_meaning,
)
from lexical_acquisition_mechanisms.settings import ModelSource
from lexical_acquisition_mechanisms.word_selection import SelectedWord


def assignment(word: str, exposure: int) -> SecondSenseAssignment:
    return SecondSenseAssignment(
        "run-1", word, 1, "cola", "food", 3000, "other-category", "car", "vehicle", exposure
    )


def test_every_weight_measures_each_unknown_word_at_every_layer(
    tmp_path: Path, tiny_model_source: ModelSource
) -> None:
    words = [
        SelectedWord("cola", "noun", token_count=1, token_ids="1103"),
        SelectedWord("car", "noun", token_count=1, token_ids="1559"),
        SelectedWord("flaid", "unknown", token_count=1, token_ids="100300"),
        SelectedWord("blick", "unknown", token_count=1, token_ids="100301"),
    ]
    write_rows(tmp_path / "words.csv", SelectedWord, words)
    write_rows(
        tmp_path / "assignments.csv",
        SecondSenseAssignment,
        [assignment("flaid", 100), assignment("blick", 0)],
    )
    write_rows(
        tmp_path / "contexts.csv",
        NeutralContext,
        [NeutralContext(0, ""), NeutralContext(1, "Do you know about the")],
    )
    pair = CheckedMinimalPair(
        "cola",
        "food",
        1,
        "car",
        "noun",
        "other-category",
        "We drove the ___.",
        "We drank the ___.",
        "drove",
        "drank",
    )
    write_rows(tmp_path / "checked.csv", CheckedMinimalPair, [pair])
    names = ("before", "checkpoint-50", "final")
    directories = {("run-1", name): Path(tiny_model_source.repository) for name in names}
    outputs = (tmp_path / "out" / "meaning.parquet", tmp_path / "out" / "meaning.csv")

    write_meaning(
        directories,
        lambda source: LanguageModel.load(source, device="cpu"),
        (
            tmp_path / "words.csv",
            tmp_path / "assignments.csv",
            tmp_path / "contexts.csv",
            tmp_path / "checked.csv",
        ),
        outputs,
    )

    rows = parquet.read_table(outputs[0]).to_pylist()
    # 極小のモデルは 2 層（埋め込みを含めて 3 つの表現）
    assert len(rows) == len(names) * 2 * len(MEASURES) * 3
    assert all(math.isfinite(row["value"]) for row in rows)
    summary = read_rows(outputs[1], MeaningSummary)
    assert {(row.second_exposure, row.words) for row in summary} == {(0, 1), (100, 1)}
    # 同じ重みの未知語・X・Y の表現どうしなので、cos は -1〜1
    assert all(-1 <= row.neutral_first <= 1 for row in summary)


def test_summary_leaves_out_words_without_minimal_pairs_from_the_cue_effect() -> None:
    def row(word: str, measure: str, value: float) -> dict[str, object]:
        return {
            "run": "run-1",
            "weights": "final",
            "unknown_word": word,
            "kind": "other-category",
            "second_exposure": 1000,
            "layer": 12,
            "measure": measure,
            "value": value,
        }

    rows = [
        row("flaid", "neutral_first", 0.8),
        row("flaid", "neutral_second", 0.6),
        row("flaid", "cue_effect", 0.2),
        row("blick", "neutral_first", 0.6),
        row("blick", "neutral_second", 0.4),
        row("blick", "cue_effect", float("nan")),
    ]

    (summary,) = summarize(rows)

    assert (summary.neutral_first, summary.neutral_second) == (
        pytest.approx(0.7),
        pytest.approx(0.5),
    )
    assert summary.cue_effect == pytest.approx(0.2)
    assert summary.words == 2
