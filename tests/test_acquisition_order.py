"""未知語の指標の正規化と、獲得の順番の検定のテスト。"""

import math
from pathlib import Path

import pyarrow as arrow
import pyarrow.parquet as parquet
import pytest

from lexical_acquisition_mechanisms.acquisition_order import (
    AcquisitionOrder,
    normalize,
    write_acquisition_order,
)
from lexical_acquisition_mechanisms.csv_tables import read_rows

# 露出回数ごとの、正規化する前の値（m0 は露出0回、m1 は既知の名詞・分けた名詞）
RAW = {
    "meaning": {0: 0.0, 10: 0.3, 100: 0.7, 1000: 0.9},
    "integration": {0: 0.0, 10: 0.1, 100: 0.2, 1000: 0.6},
    "form": {0: 5.0, 10: 2.6, 100: 2.0, 1000: 1.5},
}
REFERENCES = {"meaning": 1.0, "integration": 1.0, "form": 1.0}
# 4つの run で、各未知語が4つの露出回数に1回ずつ当たる
LEVELS = [0, 10, 100, 1000]
SCHEDULE = {
    f"run-{run + 1}": {f"u{word + 1}": LEVELS[(run + word) % 4] for word in range(4)}
    for run in range(4)
}


def rows() -> list[dict]:
    result = []
    for run, exposures in SCHEDULE.items():
        for word, exposure in exposures.items():
            for indicator, values in RAW.items():
                layer = -1 if indicator == "form" else 0
                result.append(
                    {
                        "run": run,
                        "kind": "unknown",
                        "word": word,
                        "token_count": 2,
                        "spelling_closeness": "near",
                        "noun": "pasta",
                        "exposure": exposure,
                        "layer": layer,
                        "indicator": indicator,
                        "value": values[exposure],
                    }
                )
        for indicator in ("integration", "form"):
            result.append(
                {
                    "run": run,
                    "kind": "known-noun",
                    "word": "giraffe",
                    "token_count": 2,
                    "spelling_closeness": "",
                    "noun": "",
                    "exposure": -1,
                    "layer": -1 if indicator == "form" else 0,
                    "indicator": indicator,
                    "value": REFERENCES[indicator],
                }
            )
        result.append(
            {
                "run": run,
                "kind": "split-noun",
                "word": " pa|sta",
                "token_count": 2,
                "spelling_closeness": "",
                "noun": "pasta",
                "exposure": -1,
                "layer": 0,
                "indicator": "meaning",
                "value": 1.0,
            }
        )
    return result


def test_values_are_normalized_between_unexposed_and_known_words() -> None:
    normalized = normalize(rows())

    meaning = next(
        row for row in normalized if row["indicator"] == "meaning" and row["exposure"] == 100
    )
    assert meaning["normalized"] == pytest.approx(0.7)
    form = next(row for row in normalized if row["indicator"] == "form" and row["exposure"] == 10)
    assert (form["m0"], form["m1"], form["normalized"]) == pytest.approx((5.0, 1.0, 0.6))


def test_indicators_reach_half_in_order(tmp_path: Path) -> None:
    parquet.write_table(arrow.Table.from_pylist(rows()), tmp_path / "indicators.parquet")
    outputs = (tmp_path / "normalized.parquet", tmp_path / "curves.csv", tmp_path / "order.csv")

    write_acquisition_order(tmp_path / "indicators.parquet", outputs, seed=0)

    orders = {
        row.indicator: row for row in read_rows(outputs[2], AcquisitionOrder) if row.group == "all"
    }
    assert (
        orders["form"].half_exposure
        < orders["meaning"].half_exposure
        < orders["integration"].half_exposure
    )
    # 形への慣れは、露出 0 と 10 の間（log10(1 + 露出回数) の直線で 0.5 に達する所）
    assert orders["form"].half_exposure == pytest.approx(10 ** (math.log10(11) * 0.5 / 0.6) - 1)
    assert "meaning: 1.000" in orders["form"].earlier_than
