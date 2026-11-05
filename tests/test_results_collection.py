"""全結果の集計のテスト（作った結果の表で）。"""

from collections import Counter
from pathlib import Path

import pyarrow as arrow
import pyarrow.parquet as parquet

from lexical_acquisition_mechanisms.acquisition_order import AcquisitionOrder
from lexical_acquisition_mechanisms.component_adoption import ComponentClass
from lexical_acquisition_mechanisms.csv_tables import write_rows
from lexical_acquisition_mechanisms.dependence_regression import RegressionTerm
from lexical_acquisition_mechanisms.results_collection import write_results
from lexical_acquisition_mechanisms.sense_accuracy import SenseAccuracy
from lexical_acquisition_mechanisms.significant_heads_and_mlps import ComponentEffect

NAMES = (
    "indicators.parquet",
    "normalized.parquet",
    "order.csv",
    "accuracy.csv",
    "effects.csv",
    "classes.csv",
    "data.parquet",
    "terms.csv",
)


def write_inputs(directory: Path) -> None:
    measured = {"run": "run-1", "kind": "unknown", "word": "blicket", "layer": 3}
    indicators = [measured | {"indicator": "meaning", "value": 0.4}]
    parquet.write_table(arrow.Table.from_pylist(indicators), directory / NAMES[0])
    normalized = [row | {"normalized": 0.8} for row in indicators]
    parquet.write_table(arrow.Table.from_pylist(normalized), directory / NAMES[1])
    orders = [
        AcquisitionOrder("all", indicator, 3, 100.0, 50.0, 200.0, 1.0, "", "")
        for indicator in ("form", "meaning", "integration")
    ]
    write_rows(directory / NAMES[2], AcquisitionOrder, orders)
    accuracy = [SenseAccuracy("adjective", 4, "car", "quiet", 20, 0.9, 0.5, 0.4, 0.5, 0.01, True)]
    write_rows(directory / NAMES[3], SenseAccuracy, accuracy)
    effects = [
        ComponentEffect(
            "adjective",
            4,
            "target_position",
            0,
            head,
            0.2,
            0.1,
            0.1,
            0.05,
            0.0,
            0.2,
            0.01,
            0.02,
            True,
            20,
            3,
        )
        for head in (0, -1)
    ]
    write_rows(directory / NAMES[4], ComponentEffect, effects)
    classes = [ComponentClass(0, 0, "none", "none", "adopted", "part-of-speech")]
    write_rows(directory / NAMES[5], ComponentClass, classes)
    data = [
        {
            "run": "run-1",
            "unknown_word": "blicket",
            "ablation": "category-direction",
            "dependence": 0.7,
        }
    ]
    parquet.write_table(arrow.Table.from_pylist(data), directory / NAMES[6])
    terms = [
        RegressionTerm("category-direction", term, 0.5, 0.1, 0.3, 0.7, 0.01, 100)
        for term in ("meaning", "integration")
    ]
    write_rows(directory / NAMES[7], RegressionTerm, terms)


def test_results_of_all_experiments_are_collected(tmp_path: Path) -> None:
    write_inputs(tmp_path)
    missing = tmp_path / "missing"

    write_results(
        ["tiny", "absent"],
        (tmp_path / "all_results.parquet", tmp_path / "model_sizes.png"),
        lambda model: {(tmp_path if model == "tiny" else missing) / name: "" for name in NAMES},
    )

    rows = parquet.read_table(tmp_path / "all_results.parquet").to_pylist()
    counts = Counter((row["experiment"], row["indicator"]) for row in rows)
    assert counts == {
        ("unknown-word-acquisition", "meaning"): 1,
        ("unknown-word-acquisition", "meaning-normalized"): 1,
        ("sense-addition", "sense-accuracy"): 1,
        ("sense-addition", "sense-accuracy-difference"): 1,
        ("sense-addition", "head-effect"): 2,
        ("sense-addition", "head-effect-normalized"): 2,
        ("dependence-on-category-judging-heads", "dependence"): 1,
    }
    assert {row["model"] for row in rows} == {"tiny"}
    effect = next(row for row in rows if row["indicator"] == "head-effect" and row["head"] == -1)
    assert (effect["run"], effect["number_of_senses"], effect["layer"]) == ("adjective-4", 4, 0)
    assert (tmp_path / "model_sizes.png").stat().st_size > 0
