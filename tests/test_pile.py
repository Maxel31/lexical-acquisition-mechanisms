"""Pile の読み込みのテスト。ダウンロードの代わりに、手元に作った parquet を使う。"""

from collections.abc import Iterator
from pathlib import Path

import pyarrow as arrow
import pyarrow.parquet as parquet
import pytest

from lexical_acquisition_mechanisms import pile
from lexical_acquisition_mechanisms.pile import PileDocument, documents_in_shard, scan_shards

TEXTS_BY_SHARD = {
    0: ["I ate pasta.", "The car is red."],
    1: ["A lamp.", "A sofa.", "A desk."],
}
FILE_NAMES = ["data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet"]


def write_local_shard(file_name: str, directory: Path) -> Path:
    """断片をダウンロードする代わりに、TEXTS_BY_SHARD の parquet を書く（別のプロセスでも動く）。"""
    shard = FILE_NAMES.index(file_name)
    directory.mkdir(parents=True)
    path = directory / "shard.parquet"
    parquet.write_table(arrow.table({"text": TEXTS_BY_SHARD[shard]}), path, row_group_size=1)
    return path


def texts_with_positions(documents: Iterator[PileDocument]) -> list[tuple[int, int, str]]:
    return [(document.shard, document.row, document.text) for document in documents]


def test_documents_come_in_row_order_and_the_shard_is_deleted(tmp_path: Path) -> None:
    documents = documents_in_shard(1, FILE_NAMES[1], tmp_path, write_local_shard)

    assert texts_with_positions(documents) == [
        (1, 0, "A lamp."),
        (1, 1, "A sofa."),
        (1, 2, "A desk."),
    ]
    assert list(tmp_path.iterdir()) == []


def test_shard_is_deleted_when_reading_stops_early(tmp_path: Path) -> None:
    documents = documents_in_shard(1, FILE_NAMES[1], tmp_path, write_local_shard)
    next(documents)
    documents.close()

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("number_of_processes", [1, 2])
def test_scan_results_come_in_shard_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, number_of_processes: int
) -> None:
    monkeypatch.setattr(pile, "shard_file_names", lambda: FILE_NAMES)

    results = list(
        scan_shards([1, 0], texts_with_positions, number_of_processes, tmp_path, write_local_shard)
    )

    assert [len(result) for result in results] == [3, 2]
    assert results[1][0] == (0, 0, "I ate pasta.")
