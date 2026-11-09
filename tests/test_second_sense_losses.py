"""2つ目の語義の追加の、1つ目と2つ目の語義の用例での未知語の損失を測る処理のテスト。"""

import math
from pathlib import Path

import pyarrow as arrow
import pyarrow.parquet as parquet
import pytest

from lexical_acquisition_mechanisms import second_sense_losses
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.language_model import LanguageModel
from lexical_acquisition_mechanisms.passage_replacement import PassageReplacer
from lexical_acquisition_mechanisms.second_sense_assignments import SecondSenseAssignment
from lexical_acquisition_mechanisms.second_sense_losses import LossSummary, summarize, write_losses
from lexical_acquisition_mechanisms.settings import ModelSource
from lexical_acquisition_mechanisms.word_selection import SelectedWord
from lexical_acquisition_mechanisms.word_tokenizer import WordTokenizer


def test_every_weight_reads_the_evaluation_passages_of_both_senses(
    tmp_path: Path, tiny_model_source: ModelSource, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(second_sense_losses, "PASSAGES_PER_NOUN", 2)
    tokenizer = WordTokenizer.from_model(tiny_model_source)
    nouns = {noun: tokenizer.word_token_ids(noun) for noun in ("cola", "car")}
    words = [
        *(
            SelectedWord(noun, "noun", token_count=len(ids), token_ids=" ".join(map(str, ids)))
            for noun, ids in nouns.items()
        ),
        SelectedWord("flaid", "unknown", token_count=1, token_ids="100300"),
    ]
    write_rows(tmp_path / "words.csv", SelectedWord, words)
    assignment = SecondSenseAssignment(
        "run-1", "flaid", 1, "cola", "food", 3000, "other-category", "car", "vehicle", 100
    )
    write_rows(tmp_path / "assignments.csv", SecondSenseAssignment, [assignment])
    prefix = tokenizer.word_token_ids("We saw the")
    rows = []
    for noun, ids in nouns.items():
        for key in range(3):
            # 評価用の用例 3 件（2 件だけ使う）と、学習用の用例 1 件（使わない）
            for split in ("evaluation", "training") if key == 0 else ("evaluation",):
                rows.append(
                    {
                        "word": noun,
                        "role": "noun",
                        "split": split,
                        "key": f"{key}-{split}",
                        "token_ids": [*prefix, *ids, *prefix],
                        "occurrence_positions": [len(prefix)],
                    }
                )
    parquet.write_table(arrow.Table.from_pylist(rows), tmp_path / "passages.parquet")
    length = len(prefix) * 2 + 1
    names = ("before", "checkpoint-50", "final")
    directories = {("run-1", name): Path(tiny_model_source.repository) for name in names}
    outputs = (tmp_path / "out" / "losses.parquet", tmp_path / "out" / "losses.csv")

    write_losses(
        directories,
        lambda source: LanguageModel.load(source, device="cpu"),
        PassageReplacer(tokenizer, length),
        (tmp_path / "words.csv", tmp_path / "assignments.csv", tmp_path / "passages.parquet"),
        outputs,
    )

    losses = parquet.read_table(outputs[0]).to_pylist()
    assert [(row["weights"], row["sense"], row["passage"]) for row in losses] == [
        (name, sense, number)
        for name in names
        for sense in ("first", "second")
        for number in (0, 1)
    ]
    assert all(math.isfinite(row["surprisal"]) and row["surprisal"] > 0 for row in losses)
    summary = read_rows(outputs[1], LossSummary)
    assert [(row.sense, row.weights) for row in summary] == [
        (sense, name) for sense in ("first", "second") for name in names
    ]


def test_summary_averages_passages_within_each_word_first() -> None:
    def row(word: str, passage: int, surprisal: float) -> dict[str, object]:
        return {
            "run": "run-1",
            "weights": "final",
            "unknown_word": word,
            "kind": "other-category",
            "second_exposure": 1000,
            "sense": "first",
            "passage": passage,
            "surprisal": surprisal,
        }

    rows = [row("flaid", 0, 2.0), row("flaid", 1, 4.0), row("flaid", 2, 6.0), row("blick", 0, 1.0)]

    (summary,) = summarize(rows)

    # 未知語ごとの平均（4.0 と 1.0）の平均
    assert summary.surprisal == pytest.approx(2.5)
    assert summary.words == 2
