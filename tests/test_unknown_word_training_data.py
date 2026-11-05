"""未知語の獲得の学習データのテスト。"""

from pathlib import Path

import numpy as np
import pyarrow as arrow
import pyarrow.parquet as parquet
import pytest

from lexical_acquisition_mechanisms import unknown_word_assignments
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.passage_collection import CANDIDATE_SCHEMA
from lexical_acquisition_mechanisms.settings import ModelSource, Settings, load_settings
from lexical_acquisition_mechanisms.unknown_word_assignments import UnknownWordAssignment
from lexical_acquisition_mechanisms.unknown_word_training_data import (
    TrainingDataCheck,
    write_unknown_word_training_data,
)
from lexical_acquisition_mechanisms.word_selection import SelectedWord
from lexical_acquisition_mechanisms.word_tokenizer import WordTokenizer

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = "We ate {word} and then we had {word} again at home today."


@pytest.fixture
def tokenizer(tiny_model_source: ModelSource) -> WordTokenizer:
    return WordTokenizer.from_model(tiny_model_source)


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


def test_runs_insert_the_replaced_passages(
    tmp_path: Path,
    tiny_model_source: ModelSource,
    tokenizer: WordTokenizer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(unknown_word_assignments, "run_names", lambda: ["run-1", "run-2"])
    length = len(tokenizer.encode([TEMPLATE.format(word="pasta")]))
    settings: Settings = load_settings(REPOSITORY_ROOT / "settings.yaml")
    passages_settings = settings.passages.model_copy(update={"tokens_per_passage": length})
    # 学習で系列に分ける長さ（用例は、この長さの区切りをまたがないように置く）
    training = settings.training.model_copy(update={"sequence_length": 3000})
    settings = settings.model_copy(
        update={
            "models": {"tiny": tiny_model_source},
            "passages": passages_settings,
            "training": training,
        }
    )
    (new_token,) = tokenizer.unused_token_ids(1)
    words = [
        SelectedWord(
            "pasta", "noun", token_ids=" ".join(map(str, tokenizer.word_token_ids("pasta")))
        ),
        SelectedWord(
            "bread", "noun", token_ids=" ".join(map(str, tokenizer.word_token_ids("bread")))
        ),
        SelectedWord("<new_01>", "unknown", token_count=1, token_ids=str(new_token)),
        SelectedWord(
            "zubbet",
            "unknown",
            token_count=len(tokenizer.word_token_ids("zubbet")),
            token_ids=" ".join(map(str, tokenizer.word_token_ids("zubbet"))),
        ),
    ]
    write_rows(tmp_path / "words.csv", SelectedWord, words)
    rows = [passage(tokenizer, "pasta", row) for row in range(3)]
    rows += [passage(tokenizer, "bread", row) for row in range(10, 12)]
    schema = CANDIDATE_SCHEMA.append(arrow.field("split", arrow.string()))
    parquet.write_table(arrow.Table.from_pylist(rows, schema=schema), tmp_path / "passages.parquet")
    assignments = [
        UnknownWordAssignment("run-1", "<new_01>", 1, "", 1, 0, "pasta", "food", 2),
        UnknownWordAssignment("run-1", "zubbet", 2, "near", 1, 1, "bread", "food", 1),
        UnknownWordAssignment("run-2", "<new_01>", 1, "", 1, 0, "bread", "food", 1),
        UnknownWordAssignment("run-2", "zubbet", 2, "near", 1, 1, "pasta", "food", 2),
    ]
    write_rows(tmp_path / "assignments.csv", UnknownWordAssignment, assignments)
    end = tokenizer.end_of_text_id
    base = np.random.default_rng(0).integers(1000, 2000, 4 * 4096).astype(np.uint32)
    base[99::100] = end
    np.save(tmp_path / "base.npy", base.reshape(4, 4096))
    output = tmp_path / "training"

    write_unknown_word_training_data(
        settings,
        tmp_path / "words.csv",
        tmp_path / "passages.parquet",
        tmp_path / "assignments.csv",
        tmp_path / "base.npy",
        output,
    )

    checks = read_rows(output / "checks.csv", TrainingDataCheck)
    assert [
        (check.run, check.inserted_passages, check.replaced_occurrences) for check in checks
    ] == [
        ("run-1", 2, 4),
        ("run-1", 1, 2),
        ("run-2", 1, 2),
        ("run-2", 2, 4),
    ]
    assert len({check.total_tokens for check in checks}) == 1
    for run in ("run-1", "run-2"):
        tokens = np.load(output / run / "tokens.npy")
        starts = np.load(output / run / "passage_starts.npy")
        assert len(tokens) % 3000 == 0 and len(starts) == 3
        assert all(start // 3000 == (start + length) // 3000 for start in starts)
        texts = [tokenizer.decode(tokens[start : start + length].tolist()) for start in starts]
        assert all("pasta" not in text and "bread" not in text for text in texts)
        assert sum("zubbet" in text for text in texts) == (1 if run == "run-1" else 2)
        assert all(tokens[start + length] == end for start in starts)
        inserted = parquet.read_table(output / run / "inserted.parquet").to_pylist()
        assert {row["unknown_word"] for row in inserted} == {"<new_01>", "zubbet"}
    # 対照の run は、基データだけ（系列の長さの倍数に切る）
    assert np.array_equal(np.load(output / "control" / "tokens.npy"), base[:15000])
    assert "## run-1" in (output / "decoded_passages.txt").read_text(encoding="utf-8")
