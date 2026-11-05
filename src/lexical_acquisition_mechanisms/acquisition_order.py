"""未知語の指標を正規化し、3つの指標が獲得される順番を検定する
（unknown-word-acquisition analyze-acquisition-order）。

1. 正規化: m̂ = (m − m0) / (m1 − m0)。m0 は同じ run の露出0回の未知語（トークン数と綴りの近さが
   同じもの）の平均。m1 は、形への慣れと統合では同じ run の同じトークン数の既知の名詞（1トークンは
   名詞72語）の平均、意味では同じ run の、割り当てた名詞を同じトークン数に分けた形と元の名詞の値
   （1トークンの未知語は2トークンに分けた形）。|m1 − m0| が指標ごとの下限より小さい組は、値が何倍
   にも膨らみうるので、m̂ を nan にして（曲線と層の選び方から外して）、その数を表示する。
2. 曲線: 指標ごと（統合と意味は層ごと）に、露出回数の水準ごとの m̂ の、未知語での平均。露出回数は
   log10(1 + 露出回数) で並べ、m̂ が 0.5 に達する露出回数を、隣り合う水準の間の直線で求める。
   順番の判定に使う層は、指標ごとに、露出回数が最大の水準での平均が最も大きい層（表に書く）。
3. 検定: 未知語を単位に BOOTSTRAP_SAMPLES 回の再抽出をし、0.5 に達する露出回数の 95% の範囲
   （達しない再抽出も inf として含める）と、0.5 に達した再抽出の割合と、指標がほかの指標より先に
   達した再抽出の割合を出す（両側の p 値は、2 × min(割合, 1 − 割合)）。トークン数・綴りの近さ
   ごとにも出す。
出力は outputs/results/unknown-word-acquisition/<モデル>/ の normalized_indicators.parquet・
acquisition_curves.csv・acquisition_order.csv。
"""

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as arrow
import pyarrow.parquet as parquet

from lexical_acquisition_mechanisms.csv_tables import write_rows
from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.settings import Settings
from lexical_acquisition_mechanisms.unknown_word_indicators import (
    RESULTS_DIRECTORY,
    indicators_path,
)

INDICATORS = ("form", "meaning", "integration")
# |m1 − m0| がこれより小さければ、正規化した値が不安定なので使わない（指標の単位で）
MINIMUM_REFERENCE_GAPS = {
    "form": 0.1,
    "form-second": 0.1,
    "form-third": 0.1,
    "integration": 0.05,
    "meaning": 0.02,
}
BOOTSTRAP_SAMPLES = 1000
HALF = 0.5


@dataclass
class AcquisitionCurvePoint:
    group: str
    indicator: str
    layer: int
    exposure: int
    mean_normalized: float
    words: int


@dataclass
class AcquisitionOrder:
    # 比べた語の集まり（all、トークン数、綴りの近さ）と、指標・使った層
    group: str
    indicator: str
    layer: int
    # 0.5 に達する露出回数の推定と、再抽出による 95% の範囲（達しなければ inf）
    half_exposure: float
    lower: float
    upper: float
    # 0.5 に達した再抽出の割合
    reached_share: float
    # この指標が、ほかの2つの指標のそれぞれより先に 0.5 に達した再抽出の割合
    earlier_than: str


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="settings.yaml の models の名前")


def output_paths(model: str) -> tuple[Path, Path, Path]:
    directory = RESULTS_DIRECTORY / model
    return (
        directory / "normalized_indicators.parquet",
        directory / "acquisition_curves.csv",
        directory / "acquisition_order.csv",
    )


def analyze_acquisition_order(settings: Settings, arguments: argparse.Namespace) -> None:
    model = arguments.model
    step = Step(
        f"unknown-word-acquisition analyze-acquisition-order --model {model}",
        inputs={
            indicators_path(model): f"unknown-word-acquisition measure-indicators --model {model}"
        },
        outputs=list(output_paths(model)),
        settings={"seed": settings.words.seed, "bootstrap_samples": BOOTSTRAP_SAMPLES},
    )
    run_unless_unchanged(
        step,
        lambda: write_acquisition_order(
            indicators_path(model), output_paths(model), settings.words.seed
        ),
        arguments.recompute,
    )


def write_acquisition_order(
    input_path: Path, output_paths: tuple[Path, Path, Path], seed: int
) -> None:
    normalized_path, curves_path, order_path = output_paths
    rows = parquet.read_table(input_path).to_pylist()
    normalized = normalize(rows)
    normalized_path.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_table(arrow.Table.from_pylist(normalized), normalized_path)
    groups = {"all": lambda row: True}
    for token_count in (1, 2, 3):
        groups[f"token-count-{token_count}"] = lambda row, count=token_count: (
            row["token_count"] == count
        )
    for closeness in ("near", "far"):
        groups[closeness] = lambda row, value=closeness: row["spelling_closeness"] == value
    curves = []
    orders = []
    random_numbers = np.random.default_rng(seed)
    for group, selected in groups.items():
        chosen = [row for row in normalized if selected(row)]
        group_curves, group_orders = _analyze_group(group, chosen, random_numbers)
        curves += group_curves
        orders += group_orders
    write_rows(curves_path, AcquisitionCurvePoint, curves)
    write_rows(order_path, AcquisitionOrder, orders)


def normalize(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """未知語の行に、正規化した値（normalized）を足して返す。"""

    def mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else math.nan

    by_key: dict[tuple, list[float]] = {}
    for row in rows:
        if row["kind"] == "unknown" and row["exposure"] == 0:
            key = (
                "m0",
                row["run"],
                row["token_count"],
                row["spelling_closeness"],
                row["indicator"],
                row["layer"],
            )
            by_key.setdefault(key, []).append(row["value"])
        if row["kind"] in {"known-noun", "noun"} and row["indicator"] != "meaning":
            key = ("m1", row["run"], row["token_count"], row["indicator"], row["layer"])
            by_key.setdefault(key, []).append(row["value"])
        if row["kind"] == "split-noun":
            key = ("split", row["run"], row["noun"], row["token_count"], row["layer"])
            by_key.setdefault(key, []).append(row["value"])
    normalized = []
    unstable: dict[tuple[str, int], int] = {}
    for row in rows:
        if row["kind"] != "unknown" or not row["noun"]:
            continue
        indicator = row["indicator"]
        m0_key = (
            "m0",
            row["run"],
            row["token_count"],
            row["spelling_closeness"],
            indicator,
            row["layer"],
        )
        m0 = mean(by_key.get(m0_key, []))
        if indicator == "meaning":
            split_count = max(row["token_count"], 2)
            m1 = mean(by_key.get(("split", row["run"], row["noun"], split_count, row["layer"]), []))
        else:
            m1 = mean(
                by_key.get(("m1", row["run"], row["token_count"], indicator, row["layer"]), [])
            )
        gap = m1 - m0
        if abs(gap) < MINIMUM_REFERENCE_GAPS[indicator]:
            key = (indicator, row["layer"])
            unstable[key] = unstable.get(key, 0) + 1
            value = math.nan
        else:
            value = (row["value"] - m0) / gap
        normalized.append({**row, "m0": m0, "m1": m1, "normalized": value})
    for (indicator, layer), count in sorted(unstable.items()):
        print(
            f"{indicator}（層 {layer}）で、m1 と m0 の差が下限"
            f"（{MINIMUM_REFERENCE_GAPS[indicator]}）より小さい {count} 行の、正規化した値を"
            "使いません。"
        )
    return normalized


def _analyze_group(
    group: str, rows: Sequence[dict[str, object]], random_numbers: np.random.Generator
):
    """語の集まりの、曲線と、0.5 に達する露出回数と、指標の順番。"""
    words = sorted({row["word"] for row in rows})
    exposures = sorted({row["exposure"] for row in rows})
    curves = []
    chosen_layers = {}
    tables = {}
    for indicator in INDICATORS:
        layers = sorted({row["layer"] for row in rows if row["indicator"] == indicator})
        best = None
        for layer in layers:
            table = _word_table(rows, indicator, layer, words, exposures)
            means = np.nanmean(table, axis=0)
            for index, (exposure, value) in enumerate(zip(exposures, means, strict=True)):
                count = int(np.sum(~np.isnan(table[:, index])))
                curves.append(
                    AcquisitionCurvePoint(group, indicator, layer, exposure, float(value), count)
                )
            if not np.isnan(means[-1]) and (best is None or means[-1] > best[1]):
                best = (layer, means[-1])
        if best is not None:
            chosen_layers[indicator] = best[0]
            tables[indicator] = _word_table(rows, indicator, best[0], words, exposures)
    estimates = {
        name: _half_exposure(np.nanmean(table, axis=0), exposures) for name, table in tables.items()
    }
    samples = {name: [] for name in tables}
    for _ in range(BOOTSTRAP_SAMPLES):
        picked = random_numbers.integers(0, len(words), len(words))
        for name, table in tables.items():
            samples[name].append(_half_exposure(np.nanmean(table[picked], axis=0), exposures))
    orders = []
    for name in tables:
        values = np.array(samples[name])
        # 達しない再抽出（inf）も含めて、値の間を補間せずに分位点を取る
        lower = np.percentile(values, 2.5, method="lower")
        upper = np.percentile(values, 97.5, method="higher")
        earlier = [
            f"{other}: {np.mean(values < np.array(samples[other])):.3f}"
            for other in tables
            if other != name
        ]
        orders.append(
            AcquisitionOrder(
                group,
                name,
                chosen_layers[name],
                estimates[name],
                float(lower),
                float(upper),
                float(np.mean(np.isfinite(values))),
                ", ".join(earlier),
            )
        )
    return curves, orders


def _word_table(
    rows: Sequence[dict[str, object]],
    indicator: str,
    layer: int,
    words: Sequence[str],
    exposures: Sequence[int],
) -> np.ndarray:
    """形は（未知語, 露出回数の水準）。未知語と水準ごとの、正規化した値の平均（run をまたぐ）。"""
    table = np.full((len(words), len(exposures)), np.nan)
    word_index = {word: index for index, word in enumerate(words)}
    exposure_index = {exposure: index for index, exposure in enumerate(exposures)}
    for row in rows:
        if row["indicator"] == indicator and row["layer"] == layer:
            table[word_index[row["word"]], exposure_index[row["exposure"]]] = row["normalized"]
    return table


def _half_exposure(means: np.ndarray, exposures: Sequence[int]) -> float:
    """平均の曲線が 0.5 に初めて達する露出回数。

    水準の間は、log10(1 + 露出回数) に対して直線で結ぶ。達しなければ inf。
    """
    positions = np.log10(1 + np.array(exposures, dtype=float))
    for index in range(1, len(means)):
        before, after = means[index - 1], means[index]
        if np.isnan(before) or np.isnan(after):
            continue
        if before < HALF <= after:
            share = (HALF - before) / (after - before)
            position = positions[index - 1] + share * (positions[index] - positions[index - 1])
            return float(10**position - 1)
    if len(means) and means[0] >= HALF:
        return 0.0
    return math.inf
