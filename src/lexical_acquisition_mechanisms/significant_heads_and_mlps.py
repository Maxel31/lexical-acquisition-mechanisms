"""各 head と MLP の効果の区間を出し、有意なものを選ぶ
（sense-addition select-significant-heads-and-mlps --model）。

measure-head-and-mlp-effects の組ごとの効果から、条件（足す語義の種類 × 数 N）× 入れ替えた位置 ×
構成要素ごとに、次を出す（効果の地図）。
1. 対照を引く: 組ごとに、対照の run の同じ組（同じ最小対・方向）の効果を引く
2. 最小対の両方向で平均し、最小対で平均する（手順1で2つの語義とも使えるとした最小対だけ）
3. ターゲット語と最小対を単位にした2段のブートストラップで、95% の区間と、標準誤差による正規近似の
   両側の p 値を出す
4. 条件と位置ごとに、構成要素の数について Benjamini–Hochberg 法で補正し、FDR を 5% に抑える
正規化した効果は、対照を引いた効果を、全てを入れ替えたときの効果（m(s_X) − m(s_Y)）の平均で割った値
（1 なら、その構成要素だけで語義が完全に切り替わる）。
結果は outputs/results/sense-addition/<モデル>/component_effects.csv に書く。
"""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as parquet

from lexical_acquisition_mechanisms import head_and_mlp_effects, sense_accuracy
from lexical_acquisition_mechanisms.bootstrap_statistics import (
    BOOTSTRAP_SAMPLES,
    benjamini_hochberg,
    bootstrap_weights,
    normal_p_values,
    weighted_means,
)
from lexical_acquisition_mechanisms.csv_tables import write_rows
from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.settings import Settings

SCOPES = ("target_position", "after_target_positions")
FALSE_DISCOVERY_RATE = 0.05


@dataclass
class ComponentEffect:
    kind: str
    number_of_senses: int
    # 入れ替えた位置（target_position: ターゲット語、after_target_positions: そこから後の全て）
    scope: str
    layer: int
    # head の番号（MLP は −1）
    head: int
    # 最小対での平均（両方向の平均の）。対照の run の値と、それを引いた値
    treated_effect: float
    control_effect: float
    effect: float
    normalized_effect: float
    # 対照を引いた効果の 95% の区間・両側の p 値・FDR の補正をした q 値
    lower: float
    upper: float
    p_value: float
    q_value: float
    significant: bool
    minimal_pairs: int
    targets: int


@dataclass
class MinimalPairUnit:
    """1つの条件の、1つの最小対の行の番号（patching_pairs.parquet の。両方向）。"""

    target: str
    donor: str
    donor_sentence: str
    treated: list[int]
    control: list[int]


def output_path(model: str) -> Path:
    return sense_accuracy.RESULTS_DIRECTORY / model / "component_effects.csv"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="settings.yaml の models の名前")


def select_significant_heads_and_mlps(settings: Settings, arguments: argparse.Namespace) -> None:
    model = arguments.model
    pairs_path, effects_path, _, _ = head_and_mlp_effects.effect_paths(model)
    measured_by = f"sense-addition measure-head-and-mlp-effects --model {model}"
    step = Step(
        f"sense-addition select-significant-heads-and-mlps --model {model}",
        inputs={pairs_path: measured_by, effects_path: measured_by},
        outputs=[output_path(model)],
        settings={
            "seed": settings.words.seed,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "false_discovery_rate": FALSE_DISCOVERY_RATE,
        },
    )
    run_unless_unchanged(
        step,
        lambda: write_component_effects(
            (pairs_path, effects_path), output_path(model), settings.words.seed
        ),
        arguments.recompute,
    )


def write_component_effects(
    input_paths: tuple[Path, Path], output: Path, seed: int
) -> list[ComponentEffect]:
    pairs_path, effects_path = input_paths
    rows = parquet.read_table(pairs_path).to_pylist()
    arrays = np.load(effects_path)
    random_numbers = np.random.default_rng(seed)
    results = []
    for (kind, number), units in minimal_pair_units(rows).items():
        weights = bootstrap_weights(
            [unit.target for unit in units], BOOTSTRAP_SAMPLES, random_numbers
        )
        total = np.mean([rows[index]["total_effect"] for unit in units for index in unit.treated])
        for scope in SCOPES:
            values = arrays[scope].astype(np.float64)
            treated = np.array([values[unit.treated].mean(axis=0) for unit in units])
            control = np.array([values[unit.control].mean(axis=0) for unit in units])
            difference = treated - control
            samples = weighted_means(weights, difference)
            effect = difference.mean(axis=0)
            lower, upper = np.percentile(samples, [2.5, 97.5], axis=0)
            p_values = normal_p_values(effect, samples.std(axis=0))
            q_values = benjamini_hochberg(p_values)
            results += [
                ComponentEffect(
                    kind,
                    number,
                    scope,
                    int(arrays["layers"][index]),
                    int(arrays["heads"][index]),
                    float(treated[:, index].mean()),
                    float(control[:, index].mean()),
                    float(effect[index]),
                    float(effect[index] / total) if total else float("nan"),
                    float(lower[index]),
                    float(upper[index]),
                    float(p_values[index]),
                    float(q_values[index]),
                    bool(q_values[index] < FALSE_DISCOVERY_RATE),
                    len(units),
                    len({unit.target for unit in units}),
                )
                for index in range(values.shape[1])
            ]
    write_rows(output, ComponentEffect, results)
    return results


def minimal_pair_units(
    rows: Sequence[dict[str, object]],
) -> dict[tuple[str, int], list[MinimalPairUnit]]:
    """条件ごとの、使える最小対（両方向とも、対照の run に同じ組があるもの）。"""

    def key(row: dict[str, object]) -> tuple:
        names = ("kind", "number_of_senses", "target", "donor", "donor_sentence", "direction")
        return tuple(row[name] for name in names)

    controls = {key(row): index for index, row in enumerate(rows) if row["run"].endswith("control")}
    grouped: dict[tuple, MinimalPairUnit] = {}
    for index, row in enumerate(rows):
        control = controls.get(key(row))
        if row["run"].endswith("control") or not row["usable"] or control is None:
            continue
        unit = grouped.setdefault(
            key(row)[:5],
            MinimalPairUnit(row["target"], row["donor"], row["donor_sentence"], [], []),
        )
        unit.treated.append(index)
        unit.control.append(control)
    units: dict[tuple[str, int], list[MinimalPairUnit]] = {}
    for (kind, number, *_), unit in grouped.items():
        if len(unit.treated) == len(head_and_mlp_effects.DIRECTIONS):
            units.setdefault((kind, number), []).append(unit)
    return units
