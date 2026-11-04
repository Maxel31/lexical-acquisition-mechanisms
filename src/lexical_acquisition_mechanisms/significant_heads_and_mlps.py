"""各 head と MLP の効果の区間を出し、有意なものを選ぶ
（sense-addition select-significant-heads-and-mlps --model）。

measure-head-and-mlp-effects の組ごとの効果から、条件（足す語義の種類 × 数 N）× 入れ替えた位置 ×
構成要素ごとに、次を出す（効果の地図）。
1. 対照を引く: 組ごとに、対照の run の同じ組（同じ最小対・方向）の効果を引く
2. 最小対の両方向で平均し、最小対で平均する（手順1で2つの語義とも使えるとした最小対だけ。
   measure-sense-accuracy の表から決める）
3. ターゲット語と最小対を単位にした2段のブートストラップで、95% の区間と、標準誤差による正規近似の
   両側の p 値を出す
4. 条件と位置ごとに、構成要素の数について Benjamini–Hochberg 法で補正し、FDR を 5% に抑える
正規化した効果は、対照を引いた効果を、全てを入れ替えたときの効果（m(s_X) − m(s_Y)）の平均で割った値
（1 なら、その構成要素だけで語義が完全に切り替わる）。
さらに、採用の基準（N に対する傾き・ターゲット語を半分ずつに分けた再現）を当て、2-2 の表に従って
分類する（component_adoption）。
結果は outputs/results/sense-addition/<モデル>/ の component_effects.csv（効果の地図）・
component_selection.csv（種類 × 構成要素ごとの採用の基準）・component_classes.csv（分類）に書く。
アブレーション（設計書 2-5「ターゲット語のトークン数で、効く head の層が変わるか」）のために、
ターゲット語の位置の効果の地図を、ターゲット語のトークン数ごとにも出す
（component_effects_by_token_count.csv。補正と区間は、トークン数ごとに同じ方法で出す）。
"""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as parquet

from lexical_acquisition_mechanisms import head_and_mlp_effects, sense_accuracy, word_finalization
from lexical_acquisition_mechanisms.bootstrap_statistics import (
    BOOTSTRAP_SAMPLES,
    benjamini_hochberg,
    bootstrap_weights,
    normal_p_values,
    weighted_means,
)
from lexical_acquisition_mechanisms.component_adoption import (
    ComponentClass,
    ComponentSelection,
    classify,
    select_components,
    target_halves,
)
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.settings import Settings
from lexical_acquisition_mechanisms.word_selection import SelectedWord

SCOPES = ("target_position", "after_target_positions")
FALSE_DISCOVERY_RATE = 0.05
TOKEN_COUNTS = (1, 2, 3)


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
class TokenCountEffect(ComponentEffect):
    """ターゲット語のトークン数ごとの、効果の地図の行。"""

    token_count: int = 0


@dataclass
class MinimalPairUnit:
    """1つの条件の、1つの最小対の行の番号（patching_pairs.parquet の。両方向）。"""

    target: str
    donor: str
    donor_sentence: str
    treated: list[int]
    control: list[int]


def output_paths(model: str) -> tuple[Path, Path, Path]:
    directory = sense_accuracy.RESULTS_DIRECTORY / model
    return (
        directory / "component_effects.csv",
        directory / "component_selection.csv",
        directory / "component_classes.csv",
    )


def token_count_effects_path(model: str) -> Path:
    return sense_accuracy.RESULTS_DIRECTORY / model / "component_effects_by_token_count.csv"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="settings.yaml の models の名前")


def select_significant_heads_and_mlps(settings: Settings, arguments: argparse.Namespace) -> None:
    model = arguments.model
    pairs_path, effects_path, _, _ = head_and_mlp_effects.effect_paths(model)
    accuracy_path = sense_accuracy.result_paths(model)[1]
    measured_by = f"sense-addition measure-head-and-mlp-effects --model {model}"
    categories = [category.name for category in settings.words.categories]
    step = Step(
        f"sense-addition select-significant-heads-and-mlps --model {model}",
        inputs={
            pairs_path: measured_by,
            effects_path: measured_by,
            accuracy_path: f"sense-addition measure-sense-accuracy --model {model}",
            word_finalization.OUTPUT_PATH: word_finalization.COMMAND,
        },
        outputs=[*output_paths(model), token_count_effects_path(model)],
        settings={
            "seed": settings.words.seed,
            "categories": categories,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "false_discovery_rate": FALSE_DISCOVERY_RATE,
        },
    )
    run_unless_unchanged(
        step,
        lambda: write_results(
            (pairs_path, effects_path, accuracy_path, word_finalization.OUTPUT_PATH),
            (*output_paths(model), token_count_effects_path(model)),
            categories,
            settings.words.seed,
        ),
        arguments.recompute,
    )


def write_results(
    input_paths: tuple[Path, Path, Path, Path],
    outputs: tuple[Path, Path, Path, Path],
    categories: Sequence[str],
    seed: int,
) -> None:
    pairs_path, effects_path, accuracy_path, words_path = input_paths
    effects_output, selection_output, classes_output, token_count_output = outputs
    rows = parquet.read_table(pairs_path).to_pylist()
    usable = sense_accuracy.usable_senses(accuracy_path)
    arrays = np.load(effects_path)
    results, differences = component_effects(rows, arrays, usable, seed)
    write_rows(effects_output, ComponentEffect, results)
    words = read_rows(words_path, SelectedWord)
    by_token_count = []
    for token_count in TOKEN_COUNTS:
        targets = {
            word.word for word in words if word.role == "target" and word.token_count == token_count
        }
        counted, _ = component_effects(rows, arrays, usable, seed, targets, SCOPES[:1])
        by_token_count += [
            TokenCountEffect(**vars(row), token_count=token_count) for row in counted
        ]
    write_rows(token_count_output, TokenCountEffect, by_token_count)
    # N の条件ごとの、ターゲット語の位置で、効果が正で有意か（構成要素の順）
    significant: dict[tuple[str, int], list[bool]] = {}
    for row in results:
        if row.scope == SCOPES[0]:
            key = (row.kind, row.number_of_senses)
            significant.setdefault(key, []).append(row.significant and row.effect > 0)
    halves = target_halves(words, categories)
    selections = select_components(
        differences,
        {key: np.array(values) for key, values in significant.items()},
        halves,
        arrays["layers"],
        arrays["heads"],
        seed,
    )
    write_rows(selection_output, ComponentSelection, selections)
    write_rows(classes_output, ComponentClass, classify(selections))


def component_effects(
    rows: Sequence[dict[str, object]],
    arrays,
    usable: set[tuple[str, int, str, str]],
    seed: int,
    targets: set[str] | None = None,
    scopes: Sequence[str] = SCOPES,
) -> tuple[list[ComponentEffect], dict[tuple[str, int], tuple[list[tuple], np.ndarray]]]:
    """効果の地図と、条件ごとの（最小対, 最小対 × 構成要素の効果）。usable は手順1で使える語義。

    後者は、ターゲット語の位置で入れ替えた、対照を引いた効果（採用の基準に使う）。targets があれば、
    そのターゲット語の最小対だけを使う。
    """
    random_numbers = np.random.default_rng(seed)
    results = []
    differences = {}
    for (kind, number), all_units in minimal_pair_units(rows, usable).items():
        units = [unit for unit in all_units if targets is None or unit.target in targets]
        if not units:
            continue
        weights = bootstrap_weights(
            [unit.target for unit in units], BOOTSTRAP_SAMPLES, random_numbers
        )
        total = np.mean([rows[index]["total_effect"] for unit in units for index in unit.treated])
        for scope in scopes:
            values = arrays[scope].astype(np.float64)
            treated = np.array([values[unit.treated].mean(axis=0) for unit in units])
            control = np.array([values[unit.control].mean(axis=0) for unit in units])
            difference = treated - control
            if scope == SCOPES[0]:
                keys = [(unit.target, unit.donor, unit.donor_sentence) for unit in units]
                differences[kind, number] = (keys, difference)
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
    return results, differences


def minimal_pair_units(
    rows: Sequence[dict[str, object]], usable: set[tuple[str, int, str, str]]
) -> dict[tuple[str, int], list[MinimalPairUnit]]:
    """条件ごとの、使える最小対（手順1で2つの語義とも使える組で、両方向とも、対照の run に同じ組が
    あるもの）。usable は sense_accuracy.usable_senses の（条件, N, ターゲット語, 語義）。
    """

    def key(row: dict[str, object]) -> tuple:
        names = ("kind", "number_of_senses", "target", "donor", "donor_sentence", "direction")
        return tuple(row[name] for name in names)

    controls = {key(row): index for index, row in enumerate(rows) if row["run"].endswith("control")}
    grouped: dict[tuple, MinimalPairUnit] = {}
    for index, row in enumerate(rows):
        control = controls.get(key(row))
        both_usable = all(
            (row["kind"], row["number_of_senses"], row["target"], sense) in usable
            for sense in (row["donor"], "original")
        )
        if row["run"].endswith("control") or not both_usable or control is None:
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
