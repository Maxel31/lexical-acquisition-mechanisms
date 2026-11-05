"""カテゴリ判定 head への依存の図と表のテスト（作った結果の表で）。"""

import csv
from pathlib import Path

import numpy as np
import pyarrow as arrow
import pyarrow.parquet as parquet
import pytest

from lexical_acquisition_mechanisms.csv_tables import write_rows
from lexical_acquisition_mechanisms.dependence_regression import RegressionTerm, StateComparison
from lexical_acquisition_mechanisms.dependence_report import write_report
from lexical_acquisition_mechanisms.head_map_comparison import HeadMapComparison


def test_figures_and_tables_are_written(tmp_path: Path) -> None:
    random_numbers = np.random.default_rng(0)
    data = [
        {
            "run": "run-1",
            "unknown_word": f"word-{index}",
            "ablation": ablation,
            "dependence": float(value),
            "meaning": float(value),
            "integration": float(value),
        }
        for ablation in ("category-direction", "random-head", "random-direction")
        for index, value in enumerate(random_numbers.uniform(0, 1, 10))
    ]
    parquet.write_table(arrow.Table.from_pylist(data), tmp_path / "data.parquet")
    terms = [RegressionTerm("category-direction", "integration", 0.8, 0.1, 0.6, 1.0, 0.001, 10)]
    write_rows(tmp_path / "regression.csv", RegressionTerm, terms)
    states = [
        StateComparison("category-direction", threshold, state, 5, 5, 0.5, 0.4, 0.6)
        for threshold in (0.3, 0.5, 0.7)
        for state in ("both", "meaning-only", "integration-only", "neither")
    ]
    write_rows(tmp_path / "states.csv", StateComparison, states)
    comparisons = [
        HeadMapComparison(noun, "run-1", "word-0", float(value), 0.1, 0.02)
        for noun, value in zip(("pasta", "car", "lamp"), (0.3, 0.4, 0.5), strict=True)
    ]
    write_rows(tmp_path / "comparison.csv", HeadMapComparison, comparisons)
    paths = tuple(
        tmp_path / name
        for name in ("data.parquet", "regression.csv", "states.csv", "comparison.csv")
    )

    write_report(["tiny"], tmp_path / "figures", 0, lambda model: paths)

    figures = tmp_path / "figures"
    assert (figures / "dependence-tiny.png").stat().st_size > 0
    assert (figures / "states-tiny.png").stat().st_size > 0
    with (figures / "dependence.csv").open(encoding="utf-8") as file:
        assert len(list(csv.DictReader(file))) == len(data)
    with (figures / "head_map_comparison_summary.csv").open(encoding="utf-8") as file:
        summary = {row["measure"]: row for row in csv.DictReader(file)}
    assert float(summary["spearman"]["mean"]) == pytest.approx(0.4)
    assert float(summary["spearman"]["lower"]) <= 0.4 <= float(summary["spearman"]["upper"])


    # 区間は、ほかのモデルの結果があるかどうかで変わらない
    write_report(["other", "tiny"], tmp_path / "both", 0, lambda model: paths)
    with (tmp_path / "both" / "head_map_comparison_summary.csv").open(encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if row["model"] == "tiny"]
    assert {row["measure"]: row for row in rows}["spearman"] == summary["spearman"]
