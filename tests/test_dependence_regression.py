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
    fit_mixed_model,
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
                    # 実際のデータと同じく、1トークンの未知語には綴りの近さがない
                    "spelling_closeness": "" if word % 3 == 0 else ("near" if word % 2 else "far"),
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
    # 綴りの近さとトークン数を 0・1 の列にしたので、計画行列が退化しない（真の near の効果は 0）
    assert abs(terms["near"].coefficient) < 0.1 and terms["near"].p_value > 0.01
    fixed = ("Intercept", "meaning", "integration", "two_tokens", "three_tokens", "near")
    assert all(np.isfinite(terms[name].standard_error) for name in fixed)
    assert {"variance: unknown_word", "variance: noun", "variance: run"} <= set(terms)
    states = read_rows(tmp_path / "states.csv", StateComparison)
    halves = {row.state: row for row in states if row.threshold == 0.5}
    assert sum(row.observations for row in halves.values()) == 8 * 29
    # 統合が進んだ状態ほど、依存度が大きい
    assert halves["both"].mean_dependence > halves["meaning-only"].mean_dependence
    assert halves["integration-only"].mean_dependence > halves["neither"].mean_dependence


def test_constant_columns_are_left_out_and_small_tables_are_skipped(
    capsys: pytest.CaptureFixture[str],
) -> None:
    random_numbers = np.random.default_rng(1)
    rows = [
        {
            "dependence": float(random_numbers.normal()),
            "meaning": float(random_numbers.uniform()),
            "integration": float(random_numbers.uniform()),
            # 使える組が全て2トークンなら、トークン数の列は式から外す
            "two_tokens": 1,
            "three_tokens": 0,
            "near": index % 2,
            "unknown_word": f"word-{index % 10}",
            "noun": f"noun-{index % 7}",
            "run": f"run-{index % 4}",
        }
        for index in range(40)
    ]

    terms = {row.term for row in fit_mixed_model("random-head", rows)}

    assert {"meaning", "integration", "near"} <= terms
    assert not {"two_tokens", "three_tokens"} & terms
    assert fit_mixed_model("random-direction", rows[:10]) == []
    assert "10 組しかない" in capsys.readouterr().out
