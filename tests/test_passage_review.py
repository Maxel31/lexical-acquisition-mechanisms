"""確認用の用例の抜き出しのテスト。"""

from pathlib import Path

import pyarrow as arrow
import pyarrow.parquet as parquet
import pytest

from lexical_acquisition_mechanisms import passage_review
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.passage_collection import CANDIDATE_SCHEMA
from lexical_acquisition_mechanisms.passage_review import (
    PassageReviewRow,
    build_review,
    words_to_review_more,
)
from lexical_acquisition_mechanisms.settings import ModelSource, Settings, load_settings
from lexical_acquisition_mechanisms.word_selection import SelectedWord
from lexical_acquisition_mechanisms.word_tokenizer import WordTokenizer

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
JUDGMENT_SCHEMA = (
    CANDIDATE_SCHEMA.append(arrow.field("split", arrow.string()))
    .append(arrow.field("passed", arrow.bool_()))
    .append(arrow.field("reasons", arrow.list_(arrow.string())))
)


def judgment(tokenizer: WordTokenizer, word: str, key: float, text: str, passed: bool) -> dict:
    token_ids = tokenizer.encode([text])
    word_id = tokenizer.word_token_ids(word)[0]
    positions = [index for index, token in enumerate(token_ids) if token == word_id]
    return {
        "word": word,
        "role": "",
        "shard": 0,
        "row": int(key * 100),
        "start": 0,
        "key": key,
        "token_ids": token_ids,
        "occurrence_positions": positions,
        "split": "training",
        "passed": passed,
        "reasons": ["stub"],
    }


def test_passed_passages_with_the_smallest_keys_are_sampled(
    tmp_path: Path, tiny_model_source: ModelSource, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(passage_review, "REVIEWED_PASSAGES_PER_WORD", 2)
    settings: Settings = load_settings(REPOSITORY_ROOT / "settings.yaml")
    settings = settings.model_copy(update={"models": {"tiny": tiny_model_source}})
    tokenizer = WordTokenizer.from_model(tiny_model_source)
    words = [
        SelectedWord("pasta", "noun", token_count=1, definition="a dish"),
        SelectedWord("quiet", "natural-adjective", token_count=1, definition="calm"),
    ]
    write_rows(tmp_path / "words.csv", SelectedWord, words)
    judgments = [
        judgment(tokenizer, "pasta", 0.4, "Some pasta and\nmore pasta.", True),
        judgment(tokenizer, "pasta", 0.1, "A pasta brand.", False),
        judgment(tokenizer, "pasta", 0.2, "We ate pasta.", True),
        judgment(tokenizer, "pasta", 0.9, "Cold pasta.", True),
        judgment(tokenizer, "quiet", 0.3, "It was quiet.", True),
    ]
    table = arrow.Table.from_pylist(judgments, schema=JUDGMENT_SCHEMA)
    parquet.write_table(table, tmp_path / "judgments.parquet")

    build_review(
        settings, tmp_path / "words.csv", tmp_path / "judgments.parquet", tmp_path / "review.csv"
    )

    rows = read_rows(tmp_path / "review.csv", PassageReviewRow)
    # 合格した pasta の用例のうち、鍵の小さい2件。自然な語は確かめない
    assert [row.passage for row in rows] == [
        "We ate [[pasta]].",
        "Some [[pasta]] and more [[pasta]].",
    ]
    assert {(row.word, row.definition, row.approved) for row in rows} == {("pasta", "a dish", "")}


def test_more_passages_are_sampled_for_slightly_imprecise_words(
    tmp_path: Path, tiny_model_source: ModelSource, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(passage_review, "REVIEWED_PASSAGES_PER_WORD", 2)
    settings: Settings = load_settings(REPOSITORY_ROOT / "settings.yaml")
    passages = settings.passages.model_copy(update={"additional_reviewed_passages_per_word": 18})
    settings = settings.model_copy(
        update={"models": {"tiny": tiny_model_source}, "passages": passages}
    )
    tokenizer = WordTokenizer.from_model(tiny_model_source)
    write_rows(tmp_path / "words.csv", SelectedWord, [SelectedWord("pasta", "noun", token_count=1)])
    judgments = [
        judgment(tokenizer, "pasta", key, f"Pasta number {number}.", True)
        for number, key in enumerate([0.1, 0.2, 0.3, 0.4])
    ]
    table = arrow.Table.from_pylist(judgments, schema=JUDGMENT_SCHEMA)
    parquet.write_table(table, tmp_path / "judgments.parquet")
    paths = (tmp_path / "words.csv", tmp_path / "judgments.parquet", tmp_path / "review.csv")
    build_review(settings, *paths)
    rows = read_rows(paths[2], PassageReviewRow)
    rows[0].approved, rows[1].approved = "1", "0"
    write_rows(paths[2], PassageReviewRow, rows)

    build_review(settings, *paths, words_to_review_more(rows, settings))

    rows = read_rows(paths[2], PassageReviewRow)
    # 20 件まで足せば 0 が 1 件まで許される。記入した 2 件はそのままで、続きの鍵の用例が足される
    assert [(row.row, row.approved) for row in rows] == [(10, "1"), (20, "0"), (30, ""), (40, "")]


@pytest.mark.parametrize(
    ("zeros", "unfilled", "rows", "reviewed_more"),
    [
        (2, 0, 20, True),  # 60 件で 0 が 3 件まで許される
        (3, 0, 20, True),
        (4, 0, 20, False),  # 足しても届かない
        (1, 0, 20, False),  # すでに精度が足りている
        (2, 1, 20, False),  # 記入し終えていない
        (5, 0, 60, True),  # すでに足した語は、足したまま
    ],
)
def test_words_to_review_more(zeros: int, unfilled: int, rows: int, reviewed_more: bool) -> None:
    settings: Settings = load_settings(REPOSITORY_ROOT / "settings.yaml")
    values = ["0"] * zeros + [""] * unfilled
    values += ["1"] * (rows - len(values))
    review = [
        PassageReviewRow("pasta", "noun", "", 0, row, 0, "", value)
        for row, value in enumerate(values)
    ]

    assert (words_to_review_more(review, settings) == {"pasta"}) is reviewed_more
