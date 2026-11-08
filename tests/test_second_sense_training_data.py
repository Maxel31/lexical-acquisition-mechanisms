"""2つ目の語義の追加の学習データのテスト。"""

from pathlib import Path

import numpy as np
import pyarrow as arrow
import pyarrow.parquet as parquet
import pytest

from lexical_acquisition_mechanisms import second_sense_assignments
from lexical_acquisition_mechanisms.command_stopped import CommandStopped
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.passage_collection import CANDIDATE_SCHEMA
from lexical_acquisition_mechanisms.second_sense_assignments import SecondSenseAssignment
from lexical_acquisition_mechanisms.second_sense_training_data import (
    SecondSenseTrainingDataCheck,
    second_sense_base_tokens,
    write_second_sense_training_data,
)
from lexical_acquisition_mechanisms.settings import ModelSource, Settings, load_settings
from lexical_acquisition_mechanisms.word_selection import SelectedWord
from lexical_acquisition_mechanisms.word_tokenizer import WordTokenizer

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = "We ate {word} and then we had {word} again at home today."


def passage(tokenizer: WordTokenizer, word: str, row: int) -> dict:
    token_ids = tokenizer.encode([TEMPLATE.format(word=word)])
    word_id = tokenizer.word_token_ids(word)[0]
    return {
        "word": word,
        "role": "noun",
        "shard": 0,
        "row": row,
        "start": 0,
        "key": row / 100,
        "token_ids": token_ids,
        "occurrence_positions": [
            index for index, token in enumerate(token_ids) if token == word_id
        ],
        "split": "training",
    }


def word(tokenizer: WordTokenizer, text: str, role: str) -> SelectedWord:
    ids = tokenizer.word_token_ids(text)
    return SelectedWord(text, role, token_count=len(ids), token_ids=" ".join(map(str, ids)))


def test_second_and_first_sense_passages_are_inserted_into_the_next_base_sequences(
    tmp_path: Path, tiny_model_source: ModelSource, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(second_sense_assignments, "run_names", lambda: ["run-1", "run-2"])
    tokenizer = WordTokenizer.from_model(tiny_model_source)
    length = len(tokenizer.encode([TEMPLATE.format(word="pasta")]))
    settings: Settings = load_settings(REPOSITORY_ROOT / "settings.yaml")
    settings = settings.model_copy(
        update={
            "models": {"tiny": tiny_model_source},
            "passages": settings.passages.model_copy(update={"tokens_per_passage": length}),
            "training": settings.training.model_copy(
                update={"sequence_length": 3000, "number_of_base_sequences_used": 4}
            ),
        }
    )
    words = [
        word(tokenizer, "pasta", "noun"),
        word(tokenizer, "bread", "noun"),
        word(tokenizer, "zubbet", "unknown"),
        word(tokenizer, "blicket", "unknown"),
    ]
    write_rows(tmp_path / "words.csv", SelectedWord, words)
    rows = [passage(tokenizer, "pasta", row) for row in range(4)]
    rows += [passage(tokenizer, "bread", row) for row in range(10, 14)]
    schema = CANDIDATE_SCHEMA.append(arrow.field("split", arrow.string()))
    parquet.write_table(arrow.Table.from_pylist(rows, schema=schema), tmp_path / "passages.parquet")

    def assignment(run: str, unknown: str, first: str, second: str, exposure: int):
        return SecondSenseAssignment(
            run, unknown, 2, first, "food", 3, "same-category", second, "food", exposure
        )

    assignments = [
        assignment("run-1", "zubbet", "pasta", "bread", 2),
        assignment("run-1", "blicket", "bread", "pasta", 0),
        assignment("run-2", "zubbet", "pasta", "bread", 0),
        assignment("run-2", "blicket", "bread", "pasta", 2),
    ]
    write_rows(tmp_path / "assignments.csv", SecondSenseAssignment, assignments)
    end = tokenizer.end_of_text_id
    base = np.random.default_rng(0).integers(1000, 2000, 8 * 4096).astype(np.uint32)
    base[99::100] = end
    np.save(tmp_path / "base.npy", base.reshape(8, 4096))
    output = tmp_path / "training"

    write_second_sense_training_data(
        settings,
        tmp_path / "words.csv",
        tmp_path / "passages.parquet",
        tmp_path / "assignments.csv",
        tmp_path / "base.npy",
        output,
    )

    checks = read_rows(output / "checks.csv", SecondSenseTrainingDataCheck)
    assert [(check.run, check.sense, check.noun, check.inserted_passages) for check in checks] == [
        ("run-1", "second", "bread", 2),
        ("run-1", "first", "pasta", 2),
        ("run-1", "second", "pasta", 0),
        ("run-1", "first", "bread", 0),
        ("run-2", "second", "bread", 0),
        ("run-2", "first", "pasta", 0),
        ("run-2", "second", "pasta", 2),
        ("run-2", "first", "bread", 2),
    ]
    assert {check.remaining_occurrences for check in checks} == {0}
    inserted = parquet.read_table(output / "run-1" / "inserted.parquet").to_pylist()
    # 2つ目の語義は用例の後ろ（鍵の大きい順）から、1つ目の語義は前（鍵の小さい順）から取る
    assert sorted((row["sense"], row["row"]) for row in inserted) == [
        ("first", 0),
        ("first", 1),
        ("second", 12),
        ("second", 13),
    ]
    tokens = np.load(output / "run-1" / "tokens.npy")
    # 基データは、未知語の獲得で使った4つの系列の直後から
    assert np.array_equal(tokens[:99], base[4 * 4096 : 4 * 4096 + 99])
    starts = np.load(output / "run-1" / "passage_starts.npy")
    texts = [tokenizer.decode(tokens[start : start + length].tolist()) for start in starts]
    assert all("zubbet" in text and "bread" not in text for text in texts)


def test_stops_without_unused_base_sequences(tmp_path: Path) -> None:
    np.save(tmp_path / "base.npy", np.zeros((6, 8), dtype=np.uint32))

    for used in (None, 4):
        with pytest.raises(CommandStopped):
            second_sense_base_tokens(tmp_path / "base.npy", used)
    assert len(second_sense_base_tokens(tmp_path / "base.npy", 3)) == 3 * 8
