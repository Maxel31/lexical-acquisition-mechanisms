"""語義の追加の学習データのテスト。"""

from pathlib import Path

import numpy as np
import pyarrow as arrow
import pyarrow.parquet as parquet
import pytest

from lexical_acquisition_mechanisms import sense_addition_assignments
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.passage_collection import CANDIDATE_SCHEMA
from lexical_acquisition_mechanisms.sense_addition_assignments import SenseAdditionAssignment
from lexical_acquisition_mechanisms.sense_addition_training_data import (
    SenseAdditionDataCheck,
    write_sense_addition_training_data,
)
from lexical_acquisition_mechanisms.settings import ModelSource, load_settings
from lexical_acquisition_mechanisms.word_selection import SelectedWord
from lexical_acquisition_mechanisms.word_tokenizer import WordTokenizer

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = "We saw {word} and then we had {word} again at home today."
RUNS = ["other-category-1", "adjective-1", "noun-control"]


def test_donors_are_replaced_only_in_their_runs(
    tmp_path: Path, tiny_model_source: ModelSource, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sense_addition_assignments, "run_names", lambda: RUNS)
    tokenizer = WordTokenizer.from_model(tiny_model_source)
    length = len(tokenizer.encode([TEMPLATE.format(word="pasta")]))
    settings = load_settings(REPOSITORY_ROOT / "settings.yaml")
    settings = settings.model_copy(
        update={
            "models": {"tiny": tiny_model_source},
            "passages": settings.passages.model_copy(update={"tokens_per_passage": length}),
            "sense_addition": settings.sense_addition.model_copy(
                update={"passages_per_added_sense": 2}
            ),
        }
    )
    roles = {
        "car": "target",
        "pasta": "noun",
        "bread": "noun",
        "quiet": "adjective",
        "happy": "adjective",
    }
    words = [
        SelectedWord(word, role, token_ids=" ".join(map(str, tokenizer.word_token_ids(word))))
        for word, role in roles.items()
    ]
    write_rows(tmp_path / "words.csv", SelectedWord, words)
    rows = []
    for word, role in roles.items():
        if role == "target":
            continue
        token_ids = tokenizer.encode([TEMPLATE.format(word=word)])
        word_id = tokenizer.word_token_ids(word)[0]
        for number in range(3):
            rows.append(
                {
                    "word": word,
                    "role": role,
                    "shard": 0,
                    "row": number,
                    "start": 0,
                    "key": number / 10,
                    "token_ids": token_ids,
                    "occurrence_positions": [
                        i for i, token in enumerate(token_ids) if token == word_id
                    ],
                    "split": "training",
                }
            )
    schema = CANDIDATE_SCHEMA.append(arrow.field("split", arrow.string()))
    parquet.write_table(arrow.Table.from_pylist(rows, schema=schema), tmp_path / "passages.parquet")
    assignments = [
        SenseAdditionAssignment(
            "other-category-1", "car", "vehicle", 1, "pasta", "noun", "food", 1
        ),
        SenseAdditionAssignment("adjective-1", "car", "vehicle", 1, "quiet", "adjective", "", 1),
    ]
    write_rows(tmp_path / "assignments.csv", SenseAdditionAssignment, assignments)
    base = np.random.default_rng(0).integers(1000, 2000, 4 * 4096).astype(np.uint32)
    base[99::100] = tokenizer.end_of_text_id
    np.save(tmp_path / "base.npy", base.reshape(4, 4096))
    output = tmp_path / "training"

    write_sense_addition_training_data(
        settings,
        (
            tmp_path / "words.csv",
            tmp_path / "passages.parquet",
            tmp_path / "assignments.csv",
            tmp_path / "base.npy",
        ),
        output,
    )

    texts = {}
    for run in RUNS:
        tokens = np.load(output / run / "tokens.npy")
        starts = np.load(output / run / "passage_starts.npy")
        texts[run] = [tokenizer.decode(tokens[start : start + length].tolist()) for start in starts]
    # 各プールは 2語 × 2件。割り当てのある donor だけを置き換える
    assert sorted(texts["other-category-1"]) == sorted(
        [TEMPLATE.format(word="car")] * 2 + [TEMPLATE.format(word="bread")] * 2
    )
    assert sorted(texts["noun-control"]) == sorted(
        [TEMPLATE.format(word="pasta")] * 2 + [TEMPLATE.format(word="bread")] * 2
    )
    assert sorted(texts["adjective-1"]) == sorted(
        [TEMPLATE.format(word="car")] * 2 + [TEMPLATE.format(word="happy")] * 2
    )
    checks = read_rows(output / "checks.csv", SenseAdditionDataCheck)
    replaced = {
        (check.run, check.donor): (check.target, check.replaced_occurrences) for check in checks
    }
    assert replaced[("other-category-1", "pasta")] == ("car", 4)
    assert replaced[("other-category-1", "bread")] == ("", 0)
    assert len({check.total_tokens for check in checks}) == 1
    # ターゲット語に置き換えた用例に donor は残らず、置き換えない用例には元のまま残る
    assert all(check.remaining_occurrences == 0 for check in checks if check.target)
    assert all(check.remaining_occurrences > 0 for check in checks if not check.target)
