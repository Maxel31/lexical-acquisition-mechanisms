"""有意な head と MLP を選ぶ処理のテスト（作った効果の値で）。"""

from pathlib import Path

import numpy as np
import pyarrow as arrow
import pyarrow.parquet as parquet
import pytest

from lexical_acquisition_mechanisms.csv_tables import read_rows
from lexical_acquisition_mechanisms.significant_heads_and_mlps import (
    ComponentEffect,
    write_component_effects,
)


def test_control_effects_are_subtracted_and_tested(tmp_path: Path) -> None:
    random_numbers = np.random.default_rng(0)
    rows = []
    values = []
    pairs = [
        (target, f"sentence {number}") for target in ("car", "lamp", "cup") for number in (1, 2)
    ]
    # 使えない最小対と、対照の組がない最小対は除く
    pairs += [("car", "unusable"), ("car", "no control")]
    for target, sentence in pairs:
        for direction in ("donor-to-target", "target-to-donor"):
            for run in ("other-category-1", "noun-control"):
                if run == "noun-control" and sentence == "no control":
                    continue
                rows.append(
                    {
                        "run": run,
                        "kind": "other-category",
                        "number_of_senses": 1,
                        "target": target,
                        "donor": "pasta",
                        "donor_sentence": sentence,
                        "direction": direction,
                        "total_effect": 2.0,
                        "usable": sentence != "unusable",
                    }
                )
                # 構成要素0は対照より 0.8 大きい。1は差がない（ばらつきだけ）。2は対照と同じ値
                noise = random_numbers.normal(0, 0.01, 2)
                first = (1.0 if run == "other-category-1" else 0.2) + noise[0]
                values.append([first, noise[1], 0.5])
    parquet.write_table(arrow.Table.from_pylist(rows), tmp_path / "pairs.parquet")
    effects = np.array(values, dtype=np.float32)
    np.savez(
        tmp_path / "effects.npz",
        target_position=effects,
        after_target_positions=effects,
        layers=np.array([0, 0, 1]),
        heads=np.array([0, -1, 0]),
    )

    write_component_effects(
        (tmp_path / "pairs.parquet", tmp_path / "effects.npz"), tmp_path / "effects.csv", 0
    )

    results = [
        row
        for row in read_rows(tmp_path / "effects.csv", ComponentEffect)
        if row.scope == "target_position"
    ]
    assert [(row.layer, row.head) for row in results] == [(0, 0), (0, -1), (1, 0)]
    first, second, third = results
    assert (first.minimal_pairs, first.targets) == (6, 3)
    assert first.effect == pytest.approx(0.8, abs=0.02)
    assert first.normalized_effect == pytest.approx(0.4, abs=0.01)
    assert first.lower < first.effect < first.upper
    assert first.significant and not second.significant
    assert (third.effect, third.p_value, third.significant) == (0.0, 1.0, False)
