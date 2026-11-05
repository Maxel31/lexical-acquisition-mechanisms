"""未知語の獲得の図と表のテスト。"""

import csv
from pathlib import Path

import pyarrow as arrow
import pyarrow.parquet as parquet
from test_acquisition_order import rows

from lexical_acquisition_mechanisms.acquisition_order import write_acquisition_order
from lexical_acquisition_mechanisms.unknown_word_report import write_report


def test_figures_and_table_are_written(tmp_path: Path) -> None:
    parquet.write_table(arrow.Table.from_pylist(rows()), tmp_path / "indicators.parquet")
    outputs = (tmp_path / "normalized.parquet", tmp_path / "curves.csv", tmp_path / "order.csv")
    write_acquisition_order(tmp_path / "indicators.parquet", outputs, seed=0)

    write_report(["tiny"], tmp_path / "figures", lambda model: outputs)

    figures = tmp_path / "figures"
    assert (figures / "curves-tiny.png").stat().st_size > 0
    assert (figures / "layers-tiny-meaning.png").stat().st_size > 0
    with (figures / "acquisition_order.csv").open(encoding="utf-8") as file:
        table = list(csv.DictReader(file))
    assert {row["model"] for row in table} == {"tiny"}
    assert {row["indicator"] for row in table if row["group"] == "all"} == {
        "form",
        "meaning",
        "integration",
    }
