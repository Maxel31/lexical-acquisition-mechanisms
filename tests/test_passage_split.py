"""ほぼ同じ候補の除去と、学習用・評価用への分け方のテスト。"""

from pathlib import Path

import numpy as np
import pyarrow as arrow
import pyarrow.parquet as parquet
import pytest

from lexical_acquisition_mechanisms.csv_tables import read_rows
from lexical_acquisition_mechanisms.passage_collection import CANDIDATE_SCHEMA
from lexical_acquisition_mechanisms.passage_split import SplitSummary, build_split
from lexical_acquisition_mechanisms.settings import Settings, load_settings

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LENGTH = 40


@pytest.fixture
def settings() -> Settings:
    """名詞は、評価用 1 件 + 学習用 9 件を必要とする（評価用の割合は 1/10）。"""
    settings = load_settings(REPOSITORY_ROOT / "settings.yaml")
    passages = settings.passages.model_copy(update={"evaluation_passages_per_word": 1})
    exposure = settings.unknown_word_acquisition.model_copy(update={"exposure_levels": [0, 9]})
    return settings.model_copy(update={"passages": passages, "unknown_word_acquisition": exposure})


def candidate(word: str, role: str, row: int, key: float, token_ids: np.ndarray) -> dict:
    return {
        "word": word,
        "role": role,
        "shard": 0,
        "row": row,
        "start": 0,
        "key": key,
        "token_ids": token_ids.tolist(),
        "occurrence_positions": [0],
    }


def test_duplicates_are_removed_and_documents_are_split(tmp_path: Path, settings: Settings) -> None:
    generator = np.random.default_rng(1)
    windows = generator.integers(1000, 50_000, (21, LENGTH))
    # 2番目の窓は、1番目の窓の最後の2トークンだけを替えたもの（ほぼ同じ）
    windows[1] = windows[0]
    windows[1, -2:] = [7, 8]
    rows = [
        candidate("pasta", "noun", row, row / 100, window) for row, window in enumerate(windows)
    ]
    bagel = generator.integers(1000, 50_000, LENGTH)
    rows.append(candidate("bagel", "target", 100, 0.5, bagel))
    candidates_path = tmp_path / "candidates.parquet"
    parquet.write_table(arrow.Table.from_pylist(rows, schema=CANDIDATE_SCHEMA), candidates_path)
    # 基データには、bagel の窓の先頭の33トークンが入っている
    base = np.zeros((2, 64), dtype=np.uint32)
    base[0, 10:43] = bagel[:33]
    np.save(tmp_path / "base.npy", base)

    build_split(
        settings,
        candidates_path,
        tmp_path / "base.npy",
        tmp_path / "split.parquet",
        tmp_path / "summary.csv",
    )

    summary = {row.word: row for row in read_rows(tmp_path / "summary.csv", SplitSummary)}
    assert (summary["pasta"].candidates, summary["pasta"].duplicates) == (21, 1)
    # 文書ごとの鍵が 1/10 より小さい文書を評価用にする（割合は、文書の多いテストで確かめる）
    assert summary["pasta"].training + summary["pasta"].evaluation == 20
    # ターゲット語は全て評価用にし、基データと重なる候補は除く
    assert (summary["bagel"].overlapping_base_data, summary["bagel"].evaluation) == (1, 0)
    split = parquet.read_table(tmp_path / "split.parquet").to_pylist()
    assert 1 not in {row["row"] for row in split}
    assert "training" in {row["split"] for row in split}


def test_near_duplicates_with_scattered_differences_are_removed(
    tmp_path: Path, settings: Settings
) -> None:
    generator = np.random.default_rng(2)
    windows = generator.integers(1000, 50_000, (4, 100))
    # 2番目の窓は、1番目の窓を20トークンに1つずつ替えたもの（一致率 95%）。3番目の窓は、2番目の
    # 窓を半分だけ替えたもの（一致率 50%。残す）
    windows[1] = windows[0]
    windows[1, ::20] = generator.integers(1000, 50_000, 5)
    windows[2] = windows[1]
    windows[2, 50:] = generator.integers(1000, 50_000, 50)
    rows = [
        candidate("pasta", "noun", row, row / 100, window) for row, window in enumerate(windows)
    ]
    candidates_path = tmp_path / "candidates.parquet"
    parquet.write_table(arrow.Table.from_pylist(rows, schema=CANDIDATE_SCHEMA), candidates_path)
    np.save(tmp_path / "base.npy", np.zeros((1, 64), dtype=np.uint32))

    build_split(
        settings,
        candidates_path,
        tmp_path / "base.npy",
        tmp_path / "split.parquet",
        tmp_path / "summary.csv",
    )

    (summary,) = read_rows(tmp_path / "summary.csv", SplitSummary)
    assert summary.duplicates == 1
    assert 1 not in {
        row["row"] for row in parquet.read_table(tmp_path / "split.parquet").to_pylist()
    }


def test_documents_are_split_in_the_same_way_for_every_word(
    tmp_path: Path, settings: Settings
) -> None:
    generator = np.random.default_rng(2)
    rows = []
    for row in range(300):
        # 同じ文書の、2つの名詞の候補（窓は別々）
        for word in ("pasta", "bread"):
            rows.append(
                candidate(word, "noun", row, row / 100, generator.integers(1000, 50_000, LENGTH))
            )
    # 文書 0 には、ターゲット語の候補もある（ターゲット語は全て評価用）
    rows.append(candidate("bagel", "target", 0, 0.5, generator.integers(1000, 50_000, LENGTH)))
    candidates_path = tmp_path / "candidates.parquet"
    parquet.write_table(arrow.Table.from_pylist(rows, schema=CANDIDATE_SCHEMA), candidates_path)
    np.save(tmp_path / "base.npy", np.zeros((1, 64), dtype=np.uint32))

    build_split(
        settings,
        candidates_path,
        tmp_path / "base.npy",
        tmp_path / "split.parquet",
        tmp_path / "summary.csv",
    )

    split = parquet.read_table(tmp_path / "split.parquet").to_pylist()
    by_word = {
        word: {row["row"]: row["split"] for row in split if row["word"] == word}
        for word in ("pasta", "bread")
    }
    assert by_word["pasta"] == by_word["bread"]
    assert by_word["pasta"][0] == "evaluation"
    # 評価用の文書は、ほぼ 1/10
    evaluation = sum(split == "evaluation" for split in by_word["pasta"].values())
    assert 15 <= evaluation <= 50
