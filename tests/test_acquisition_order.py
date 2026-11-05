"""未知語の指標の正規化と、獲得の順番の検定のテスト。"""

import math
from pathlib import Path

import numpy as np
import pyarrow as arrow
import pyarrow.parquet as parquet
import pytest

from lexical_acquisition_mechanisms.acquisition_order import (
    AcquisitionCurvePoint,
    AcquisitionOrder,
    _half_exposure,
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
    assert "meaning: 0.000" in orders["form"].p_values
    assert orders["form"].reached_share == 1.0


def test_resamples_that_never_reach_half_widen_the_range(tmp_path: Path) -> None:
    # 統合は、u1 だけが最大の露出回数で大きく、u1 を含まない再抽出では 0.5 に達しない
    indicators = rows()
    for row in indicators:
        if row["indicator"] == "integration" and row["exposure"] == 1000:
            row["value"] = 2.0 if row["word"] == "u1" else 0.3
    parquet.write_table(arrow.Table.from_pylist(indicators), tmp_path / "indicators.parquet")
    outputs = (tmp_path / "normalized.parquet", tmp_path / "curves.csv", tmp_path / "order.csv")

    write_acquisition_order(tmp_path / "indicators.parquet", outputs, seed=0)

    (integration,) = [
        row
        for row in read_rows(outputs[2], AcquisitionOrder)
        if (row.group, row.indicator) == ("all", "integration")
    ]
    assert math.isfinite(integration.half_exposure)
    # u1 を1回も含まない再抽出（約 (3/4)^4 = 32%）は 0.5 に達しないので、上限は inf
    assert integration.upper == math.inf
    assert 0.5 < integration.reached_share < 0.8


def test_values_with_a_small_reference_gap_are_not_used(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # 統合の m1（既知の名詞）と m0（露出0回の未知語）の差が、下限（0.05）より小さい
    indicators = rows()
    for row in indicators:
        if row["kind"] == "known-noun" and row["indicator"] == "integration":
            row["value"] = 0.01

    normalized = normalize(indicators)

    integration = [row for row in normalized if row["indicator"] == "integration"]
    assert integration and all(math.isnan(row["normalized"]) for row in integration)
    assert "integration（層 0）で、m1 と m0 の差が下限" in capsys.readouterr().out


def test_groups_without_values_of_an_indicator_leave_it_out(tmp_path: Path) -> None:
    # 1トークンの未知語には形への慣れがないので、1トークンの語だけの集まりでは、形への慣れの曲線も
    # 順番も、それとの比べ合いも出さない
    indicators = rows()
    for row in rows():
        if row["kind"] == "unknown" and row["indicator"] != "form":
            indicators.append({**row, "word": row["word"].replace("u", "v"), "token_count": 1})
        if row["kind"] == "known-noun" and row["indicator"] == "integration":
            indicators.append({**row, "kind": "noun", "word": "pasta", "token_count": 1})
    parquet.write_table(arrow.Table.from_pylist(indicators), tmp_path / "indicators.parquet")
    outputs = (tmp_path / "normalized.parquet", tmp_path / "curves.csv", tmp_path / "order.csv")

    write_acquisition_order(tmp_path / "indicators.parquet", outputs, seed=0)

    orders = [
        row for row in read_rows(outputs[2], AcquisitionOrder) if row.group == "token-count-1"
    ]
    assert sorted(row.indicator for row in orders) == ["integration", "meaning"]
    assert all("form" not in row.earlier_than + row.p_values for row in orders)
    curves = read_rows(outputs[1], AcquisitionCurvePoint)
    assert {row.indicator for row in curves if row.group == "token-count-1"} == {
        "integration",
        "meaning",
    }
    # 2トークンの語の集まりと全体には、形への慣れがある
    assert {row.indicator for row in read_rows(outputs[2], AcquisitionOrder)} >= {"form"}


def test_half_exposure_crosses_levels_without_values() -> None:
    # 露出 10 回の値がない（nan）ときは、露出 0 回と 100 回の間で 0.5 に達する
    reached = _half_exposure(np.array([0.0, np.nan, 1.0]), [0, 10, 100])
    assert reached == pytest.approx(10 ** (math.log10(101) * 0.5) - 1)
    assert _half_exposure(np.array([0.0, np.nan, 0.4]), [0, 10, 100]) == math.inf
    # 最初の水準で達していれば、その露出回数（後で下がってまた上がっても）
    assert _half_exposure(np.array([np.nan, 0.6, 1.0]), [0, 10, 100]) == pytest.approx(10)
    assert _half_exposure(np.array([np.nan, 0.6, 0.3, 0.8]), [0, 10, 30, 100]) == pytest.approx(10)
    assert _half_exposure(np.array([0.6, 0.3, 0.8, 0.9]), [0, 10, 30, 100]) == 0
