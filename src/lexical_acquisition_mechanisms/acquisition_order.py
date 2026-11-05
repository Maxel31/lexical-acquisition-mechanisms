"""未知語の指標を正規化し、3つの指標が獲得される順番を検定する
（unknown-word-acquisition analyze-acquisition-order）。

1. 正規化: m̂ = (m − m0) / (m1 − m0)。m0 は同じ run の露出0回の未知語（トークン数と綴りの近さが
   同じもの）の平均。m1 は、形への慣れと統合では同じ run の同じトークン数の既知の名詞（1トークンは
   名詞72語）の平均、意味では同じ run の、割り当てた名詞を同じトークン数に分けた形と元の名詞の値
   （1トークンの未知語は2トークンに分けた形）。|m1 − m0| が指標ごとの下限より小さい組は、値が何倍
   にも膨らみうるので、m̂ を nan にして（曲線と層の選び方から外して）、その数を表示する。
2. 曲線: 指標ごと（統合と意味は層ごと）に、混合効果モデル
       m̂ ~ 0 + 露出回数の水準 + (1 | 未知語) + (1 | 置換元) + (1 | run)
   を REML で当てはめ（mixed_effects_curves。ランダム切片は交差させる）、水準ごとの固定効果を
   曲線とする。
   当てはめられないとき（値がばらつかないなど）は、水準ごとの平均にする。露出回数は
   log10(1 + 露出回数) で並べ、曲線が 0.5 に達する露出回数を、隣り合う水準の間の直線で求める。
   順番の判定に使う層は、指標ごとに、全ての未知語の曲線で、露出回数が最大の水準の値が最も大きい層
   （表に書く）。語の集まり（トークン数・綴りの近さ）ごとの曲線と検定にも、同じ層を使う。その集まりに
   値のない指標（1トークンの語だけの集まりの、形への慣れ）は、曲線も順番も出さない。
3. 検定: 未知語を単位に BOOTSTRAP_SAMPLES 回の再抽出をし、分散を全体の当てはめの値に固定した
   一般化最小二乗で曲線を求め直す。0.5 に達する露出回数の 95% の範囲（達しない再抽出も inf として
   含める）と、0.5 に達した再抽出の割合と、指標がほかの指標より先に達した再抽出の割合（同時は半分と
   数える）と、その両側の p 値（2 × min(割合, 1 − 割合)）を出す。語の集まりごとにも出す。
出力は outputs/results/unknown-word-acquisition/<モデル>/ の normalized_indicators.parquet・
acquisition_curves.csv（全体は全ての層、語の集まりは判定に使った層）・acquisition_order.csv。
"""

import argparse
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas
import pyarrow as arrow
import pyarrow.parquet as parquet

from lexical_acquisition_mechanisms.csv_tables import write_rows
from lexical_acquisition_mechanisms.mixed_effects_curves import (
    LevelFit,
    fit_levels,
    generalized_least_squares,
)
from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.settings import Settings
from lexical_acquisition_mechanisms.unknown_word_indicators import (
    RESULTS_DIRECTORY,
    indicators_path,
)

INDICATORS = ("form", "meaning", "integration")
# 語の集まりと、未知語の行がその集まりに入るか
GROUPS: dict[str, Callable[[dict[str, object]], bool]] = {
    "all": lambda row: True,
    **{
        f"token-count-{count}": lambda row, count=count: row["token_count"] == count
        for count in (1, 2, 3)
    },
    **{
        closeness: lambda row, closeness=closeness: row["spelling_closeness"] == closeness
        for closeness in ("near", "far")
    },
}
# 曲線の混合効果モデルの、交差したランダム切片（正規化した表の列）
RANDOM_INTERCEPTS = ("word", "noun", "run")
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
    # 露出回数の水準の固定効果（m̂ の推定。当てはめられなければ水準ごとの平均）と標準誤差
    normalized: float
    standard_error: float
    # 値のある未知語の数
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
    # この指標が、ほかの2つの指標のそれぞれより先に 0.5 に達した再抽出の割合（同時は半分と
    # 数える）と、その両側の p 値
    earlier_than: str
    p_values: str


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
    curves = []
    orders = []
    if normalized:
        frame = pandas.DataFrame(normalized)
        exposures = sorted(frame["exposure"].unique().tolist())
        curves, layer_of = _curves_of_all_layers(frame, exposures)
        random_numbers = np.random.default_rng(seed)
        for group, selected in GROUPS.items():
            members = frame[[selected(row) for row in normalized]]
            fits = {}
            for indicator, layer in layer_of.items():
                table = members[(members["indicator"] == indicator) & (members["layer"] == layer)]
                if not np.isfinite(table["normalized"]).any():
                    continue
                fit = fit_curve(table, exposures)
                fits[indicator] = (table, fit)
                if group != "all":
                    curves += _curve_points(group, indicator, layer, table, fit, exposures)
            orders += _orders(group, layer_of, fits, exposures, random_numbers)
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


def fit_curve(table: pandas.DataFrame, exposures: Sequence[int]) -> LevelFit:
    """m̂ ~ 0 + 露出回数の水準 + (1 | 未知語) + (1 | 置換元) + (1 | run) の、水準ごとの固定効果。

    値（m̂）のない行は使わない（mixed_effects_curves.fit_levels。当てはめられなければ、水準ごとの
    平均）。
    """
    data = table[np.isfinite(table["normalized"])]
    if data.empty:
        empty = np.full(len(exposures), np.nan)
        return LevelFit(empty, empty.copy(), None, None)
    levels = data["exposure"].map({exposure: index for index, exposure in enumerate(exposures)})
    intercepts = [pandas.factorize(data[name])[0] for name in RANDOM_INTERCEPTS]
    return fit_levels(
        levels.to_numpy(), data["normalized"].to_numpy(dtype=float), intercepts, len(exposures)
    )


def _curves_of_all_layers(
    frame: pandas.DataFrame, exposures: Sequence[int]
) -> tuple[list[AcquisitionCurvePoint], dict[str, int]]:
    """全ての未知語での、指標と層ごとの曲線と、指標ごとに順番の判定に使う層。

    層は、露出回数が最大の水準での値が最も大きい層。
    """
    curves = []
    layer_of = {}
    for indicator in INDICATORS:
        best = None
        rows = frame[frame["indicator"] == indicator]
        for layer in sorted(rows["layer"].unique().tolist()):
            table = rows[rows["layer"] == layer]
            fit = fit_curve(table, exposures)
            curves += _curve_points("all", indicator, layer, table, fit, exposures)
            if np.isfinite(fit.values[-1]) and (best is None or fit.values[-1] > best[1]):
                best = (layer, fit.values[-1])
        if best is not None:
            layer_of[indicator] = best[0]
    return curves, layer_of


def _curve_points(
    group: str,
    indicator: str,
    layer: int,
    table: pandas.DataFrame,
    fit: LevelFit,
    exposures: Sequence[int],
) -> list[AcquisitionCurvePoint]:
    """表に行のある水準ごとの、曲線の点。"""
    words = table[np.isfinite(table["normalized"])].groupby("exposure")["word"].nunique()
    present = set(table["exposure"])
    return [
        AcquisitionCurvePoint(
            group,
            indicator,
            layer,
            exposure,
            float(fit.values[index]),
            float(fit.standard_errors[index]),
            int(words.get(exposure, 0)),
        )
        for index, exposure in enumerate(exposures)
        if exposure in present
    ]


def _orders(
    group: str,
    layer_of: dict[str, int],
    fits: dict[str, tuple[pandas.DataFrame, LevelFit]],
    exposures: Sequence[int],
    random_numbers: np.random.Generator,
) -> list[AcquisitionOrder]:
    """語の集まりの、指標ごとの 0.5 に達する露出回数と、未知語を単位にした再抽出での順番。"""
    words = sorted({word for table, _ in fits.values() for word in table["word"]})
    if not words:
        return []
    estimates = {name: _half_exposure(fit.values, exposures) for name, (_, fit) in fits.items()}
    resampling = {
        name: _Resampling(table, fit, words, exposures) for name, (table, fit) in fits.items()
    }
    samples = {name: np.empty(BOOTSTRAP_SAMPLES) for name in fits}
    for sample in range(BOOTSTRAP_SAMPLES):
        picked = random_numbers.integers(0, len(words), len(words))
        for name, resampled in resampling.items():
            samples[name][sample] = _half_exposure(resampled.curve(picked), exposures)
    orders = []
    for name, values in samples.items():
        # 達しない再抽出（inf）も含めて、値の間を補間せずに分位点を取る
        lower = np.percentile(values, 2.5, method="lower")
        upper = np.percentile(values, 97.5, method="higher")
        shares = {
            other: float(np.mean(values < samples[other]) + 0.5 * np.mean(values == samples[other]))
            for other in samples
            if other != name
        }
        orders.append(
            AcquisitionOrder(
                group,
                name,
                layer_of[name],
                estimates[name],
                float(lower),
                float(upper),
                float(np.mean(np.isfinite(values))),
                ", ".join(f"{other}: {share:.3f}" for other, share in shares.items()),
                ", ".join(
                    f"{other}: {min(1.0, 2 * min(share, 1 - share)):.3f}"
                    for other, share in shares.items()
                ),
            )
        )
    return orders


class _Resampling:
    """未知語を再抽出した曲線を、分散を全体の当てはめの値に固定した一般化最小二乗で求める。

    同じ未知語を2回以上選んだときは、別の未知語として扱う。全体の当てはめが水準ごとの平均だった
    ときは、再抽出でも水準ごとの平均にする。
    """

    def __init__(
        self, table: pandas.DataFrame, fit: LevelFit, words: Sequence[str], exposures: Sequence[int]
    ) -> None:
        data = table[np.isfinite(table["normalized"])]
        self._number_of_levels = len(exposures)
        self._fit = fit
        levels = data["exposure"].map({exposure: index for index, exposure in enumerate(exposures)})
        columns = [
            levels.to_numpy(),
            data["normalized"].to_numpy(dtype=float),
            *(pandas.factorize(data[name])[0] for name in RANDOM_INTERCEPTS[1:]),
        ]
        self._rows = []
        for word in words:
            mask = (data["word"] == word).to_numpy()
            self._rows.append([column[mask] for column in columns])

    def curve(self, picked: np.ndarray) -> np.ndarray:
        parts = [self._rows[index] for index in picked]
        levels, targets, *codes = (np.concatenate(column) for column in zip(*parts, strict=True))
        if not len(levels):
            return np.full(self._number_of_levels, np.nan)
        copies = np.concatenate([np.full(len(part[0]), copy) for copy, part in enumerate(parts)])
        if self._fit.intercept_variances is None:
            return fit_levels(levels, targets, [], self._number_of_levels).values
        return generalized_least_squares(
            levels,
            targets,
            [copies, *codes],
            self._fit.intercept_variances,
            self._fit.residual_variance,
            self._number_of_levels,
        )


def _half_exposure(means: np.ndarray, exposures: Sequence[int]) -> float:
    """平均の曲線が 0.5 に初めて達する露出回数。

    値のない（nan の）水準は除き、残った水準の間を、log10(1 + 露出回数) に対して直線で結ぶ
    （再抽出で、ある水準の値が全て nan になっても、その水準をまたぐ到達を見落とさないように）。
    最初の水準で達していれば、その露出回数。達しなければ inf。
    """
    finite = np.isfinite(np.asarray(means, dtype=float))
    values = np.asarray(means, dtype=float)[finite]
    positions = np.log10(1 + np.array(exposures, dtype=float))[finite]
    for index in range(1, len(values)):
        before, after = values[index - 1], values[index]
        if before < HALF <= after:
            share = (HALF - before) / (after - before)
            position = positions[index - 1] + share * (positions[index] - positions[index - 1])
            return float(10**position - 1)
    if len(values) and values[0] >= HALF:
        return float(10 ** positions[0] - 1)
    return math.inf
