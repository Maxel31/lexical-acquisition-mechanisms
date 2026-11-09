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
    # 20 件の 1/10 の、2つの文書を評価用にする
    assert (summary["pasta"].training, summary["pasta"].evaluation) == (18, 2)
    # ターゲット語は全て評価用にし、基データと重なる候補は除く
    assert (summary["bagel"].overlapping_base_data, summary["bagel"].evaluation) == (1, 0)
    split = parquet.read_table(tmp_path / "split.parquet").to_pylist()
    assert 1 not in {row["row"] for row in split}
    assert {row["split"] for row in split} == {"training", "evaluation"}
