"""依存度の回帰と、状態ごとの比較のテスト（作った値で）。"""

from pathlib import Path

import numpy as np
import pyarrow as arrow
import pyarrow.parquet as parquet
import pytest

from lexical_acquisition_mechanisms.acquisition_order import AcquisitionOrder
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.dependence_regression import (
    RegressionTerm,
    StateComparison,
    write_regression,
)


def test_regression_recovers_the_effect_of_integration(tmp_path: Path) -> None:
    random_numbers = np.random.default_rng(0)
    dependence, normalized = [], []
    word_effects = random_numbers.normal(0, 0.1, 30)
    for run in range(8):
        for word in range(30):
            meaning, integration = random_numbers.uniform(0, 1, 2)
            value = 0.1 + 0.8 * integration + word_effects[word] + random_numbers.normal(0, 0.05)
            names = {"run": f"run-{run + 1}", "unknown_word": f"word-{word}"}
            dependence.append(
                {
                    **names,
                    "noun": f"noun-{(word + run) % 20}",
                    "ablation": "category-direction",
                    "dependence": value,
                    # 最初の語の行は使えない組とする
                    "usable": word != 0,
                    "token_count": 1 + word % 3,
                    "spelling_closeness": "near" if word % 2 else "far",
                    "exposure": 10,
                }
            )
            for indicator, measured in (("meaning", meaning), ("integration", integration)):
                for layer, shift in ((5, 0.0), (6, 9.0)):
                    normalized.append(
                        {
                            "run": names["run"],
                            "word": names["unknown_word"],
                            "indicator": indicator,
                            "layer": layer,
                            "normalized": measured + shift,
                        }
                    )
    parquet.write_table(arrow.Table.from_pylist(dependence), tmp_path / "dependence.parquet")
    parquet.write_table(arrow.Table.from_pylist(normalized), tmp_path / "normalized.parquet")
    # 順番の判定に使った層は 5（層 6 の値は使わない）
    orders = [
        AcquisitionOrder("all", indicator, 5, 10.0, 5.0, 20.0, "")
        for indicator in ("meaning", "integration")
    ]
    write_rows(tmp_path / "order.csv", AcquisitionOrder, orders)

    write_regression(
        (tmp_path / "dependence.parquet", tmp_path / "normalized.parquet", tmp_path / "order.csv"),
        (tmp_path / "data.parquet", tmp_path / "regression.csv", tmp_path / "states.csv"),
        0,
    )

    assert parquet.read_table(tmp_path / "data.parquet").num_rows == 8 * 29
    terms = {row.term: row for row in read_rows(tmp_path / "regression.csv", RegressionTerm)}
    integration, meaning = terms["integration"], terms["meaning"]
    assert integration.coefficient == pytest.approx(0.8, abs=0.05)
    assert integration.lower < 0.8 < integration.upper and integration.p_value < 0.001
    assert abs(meaning.coefficient) < 0.05
    assert {"variance: unknown_word", "variance: noun", "variance: run"} <= set(terms)
    states = read_rows(tmp_path / "states.csv", StateComparison)
    halves = {row.state: row for row in states if row.threshold == 0.5}
    assert sum(row.observations for row in halves.values()) == 8 * 29
    # 統合が進んだ状態ほど、依存度が大きい
    assert halves["both"].mean_dependence > halves["meaning-only"].mean_dependence
    assert halves["integration-only"].mean_dependence > halves["neither"].mean_dependence
