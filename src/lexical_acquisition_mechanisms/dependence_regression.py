"""依存度を、意味と語としての統合の指標などで説明する回帰を当てはめる
（dependence-on-category-judging-heads fit-regression --model）。

measure-dependence の低下から依存度（d_未知 / d_既知）を出し（d_既知 が MINIMUM_KNOWN_DROP 以下の
組、つまり head が名詞にも効いていない組は除く）、同じ重みでの未知語の獲得の正規化した指標
m̂（analyze-acquisition-order。意味と統合は、順番の判定に使った層の値）を結びつけ、無効化の種類ごとに、
次の混合効果モデルを REML で当てはめる（statsmodels。ランダム切片は交差させる）。
    依存度 ~ 意味の m̂ + 統合の m̂ + トークン数 + 綴りの近さ + (1 | 未知語) + (1 | 置換元) + (1 | run)
トークン数と綴りの近さは、0・1 の列にする: two_tokens・three_tokens（1トークンが基準）と near
（near なら 1。far と、綴りの近さのない1トークンの未知語は 0。2・3トークンの中での near と far
の差）。
値が1種類しかない列（例: 使える組が全て2トークンの語）は、式から外す。使える組に1トークンの語が
ないと、two_tokens + three_tokens が切片と同じ列になるので、two_tokens を外し、2トークンを基準に
する（3トークンの係数の名前は three_tokens_versus_two_tokens にする）。露出0回の未知語も含める。
当てはめに使える組が MINIMUM_ROWS に満たない種類と、当てはめられなかった種類は、表示して飛ばす。
頑健性: 意味と統合の m̂ を閾値（THRESHOLDS）で区切った状態（both・meaning-only・integration-only・
neither）ごとの依存度の平均と、未知語と run を単位にした2段のブートストラップの 95% の区間。
出力は outputs/results/dependence-on-category-judging-heads/<モデル>/ の regression_data.parquet
（当てはめに使った表）・regression.csv（係数・標準誤差・95% の区間・p 値と、ランダム切片の分散）・
state_comparison.csv。
"""

import argparse
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas
import pyarrow as arrow
import pyarrow.parquet as parquet
import statsmodels.formula.api as formulas

from lexical_acquisition_mechanisms import acquisition_order
from lexical_acquisition_mechanisms.acquisition_order import AcquisitionOrder
from lexical_acquisition_mechanisms.bootstrap_statistics import (
    BOOTSTRAP_SAMPLES,
    bootstrap_weights,
    weighted_means,
)
from lexical_acquisition_mechanisms.category_dependence import (
    EXPERIMENT,
    RESULTS_DIRECTORY,
    dependence_path,
)
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows
from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.settings import Settings

# 名詞での低下がこれ以下（自然対数の単位）なら、依存度を出さない
MINIMUM_KNOWN_DROP = 0.05
# 固定効果の列（値が1種類しかない列は、当てはめるときに外す）
FIXED_EFFECTS = ("meaning", "integration", "two_tokens", "three_tokens", "near")
# 1トークンの語の組がなく、2トークンの語を基準にしたときの、3トークンの列の名前
THREE_VERSUS_TWO_TOKENS = "three_tokens_versus_two_tokens"
# 固定効果6個（切片を含む）と分散3個に対して、当てはめに使う組の数の下限
MINIMUM_ROWS = 30
RANDOM_INTERCEPTS = {
    "unknown_word": "0 + C(unknown_word)",
    "noun": "0 + C(noun)",
    "run": "0 + C(run)",
}
THRESHOLDS = (0.3, 0.5, 0.7)


@dataclass
class RegressionTerm:
    ablation: str
    # 係数の名前（ランダム切片の分散は "variance: <単位>"）
    term: str
    coefficient: float
    standard_error: float
    lower: float
    upper: float
    p_value: float
    observations: int


@dataclass
class StateComparison:
    ablation: str
    threshold: float
    # both・meaning-only・integration-only・neither
    state: str
    observations: int
    unknown_words: int
    mean_dependence: float
    lower: float
    upper: float


def output_paths(model: str) -> tuple[Path, Path, Path]:
    directory = RESULTS_DIRECTORY / model
    return (
        directory / "regression_data.parquet",
        directory / "regression.csv",
        directory / "state_comparison.csv",
    )


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="settings.yaml の models の名前")


def fit_regression(settings: Settings, arguments: argparse.Namespace) -> None:
    model = arguments.model
    normalized_path, _, order_path = acquisition_order.output_paths(model)
    analyzed_by = f"unknown-word-acquisition analyze-acquisition-order --model {model}"
    step = Step(
        f"{EXPERIMENT} fit-regression --model {model}",
        inputs={
            dependence_path(model): f"{EXPERIMENT} measure-dependence --model {model}",
            normalized_path: analyzed_by,
            order_path: analyzed_by,
        },
        outputs=list(output_paths(model)),
        settings={
            "seed": settings.words.seed,
            "minimum_known_drop": MINIMUM_KNOWN_DROP,
            "thresholds": list(THRESHOLDS),
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
        },
    )
    run_unless_unchanged(
        step,
        lambda: write_regression(
            (dependence_path(model), normalized_path, order_path),
            output_paths(model),
            settings.words.seed,
        ),
        arguments.recompute,
    )


def write_regression(
    input_paths: tuple[Path, Path, Path], outputs: tuple[Path, Path, Path], seed: int
) -> None:
    dependence_input, normalized_path, order_path = input_paths
    data_path, regression_path, states_path = outputs
    data = regression_data(
        parquet.read_table(dependence_input).to_pylist(),
        parquet.read_table(normalized_path).to_pylist(),
        read_rows(order_path, AcquisitionOrder),
    )
    data_path.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_table(arrow.Table.from_pylist(data), data_path)
    terms, states = [], []
    random_numbers = np.random.default_rng(seed)
    for ablation in sorted({row["ablation"] for row in data}):
        rows = [row for row in data if row["ablation"] == ablation]
        terms += fit_mixed_model(ablation, rows)
        states += compare_states(ablation, rows, random_numbers)
    write_rows(regression_path, RegressionTerm, terms)
    write_rows(states_path, StateComparison, states)


def regression_data(
    dependence: Sequence[dict[str, object]],
    normalized: Sequence[dict[str, object]],
    orders: Sequence[AcquisitionOrder],
) -> list[dict[str, object]]:
    """依存度を出せる組に、依存度と、意味と統合の m̂ を足した表（m̂ がない組は除く）。"""
    layer_of = {order.indicator: order.layer for order in orders if order.group == "all"}
    indicators = {
        (row["run"], row["word"], row["indicator"]): row["normalized"]
        for row in normalized
        if row["indicator"] in {"meaning", "integration"}
        and row["layer"] == layer_of.get(row["indicator"])
    }
    data = []
    for row in dependence:
        values = {
            name: indicators.get((row["run"], row["unknown_word"], name))
            for name in ("meaning", "integration")
        }
        known = all(value is not None and np.isfinite(value) for value in values.values())
        if row["known_drop"] > MINIMUM_KNOWN_DROP and known:
            data.append(
                {
                    **row,
                    "dependence": row["unknown_drop"] / row["known_drop"],
                    **values,
                    "two_tokens": int(row["token_count"] == 2),
                    "three_tokens": int(row["token_count"] == 3),
                    "near": int(row["spelling_closeness"] == "near"),
                }
            )
    return data


def fit_mixed_model(ablation: str, rows: Sequence[dict[str, object]]) -> list[RegressionTerm]:
    """交差したランダム切片の混合効果モデルの係数と分散。当てはめられなければ空。"""
    if len(rows) < MINIMUM_ROWS:
        print(f"{ablation}: 依存度を出せた組が {len(rows)} 組しかないので、回帰を当てはめません。")
        return []
    frame = pandas.DataFrame(rows)
    terms = [name for name in FIXED_EFFECTS if frame[name].nunique() > 1]
    if {"two_tokens", "three_tokens"} <= set(terms) and not (
        frame["two_tokens"] + frame["three_tokens"] == 0
    ).any():
        print(f"{ablation}: 1トークンの語の組がないので、2トークンの語を基準にします。")
        # 係数の意味が「1トークンとの差」から変わるので、名前も変える（表や報告で混ざらないように）
        frame = frame.rename(columns={"three_tokens": THREE_VERSUS_TWO_TOKENS})
        terms = [THREE_VERSUS_TWO_TOKENS if name == "three_tokens" else name for name in terms]
        terms.remove("two_tokens")
    model = formulas.mixedlm(
        "dependence ~ " + " + ".join(terms),
        frame,
        groups=np.ones(len(frame)),
        re_formula="0",
        vc_formula=RANDOM_INTERCEPTS,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = model.fit(reml=True)
        except (np.linalg.LinAlgError, ValueError) as error:
            print(f"{ablation} の回帰を当てはめられませんでした（{error}）。")
            return []
    for warning in caught:
        print(f"{ablation} の回帰: {warning.message}")
    fixed = model.exog_names[: model.k_fe]
    bounds = np.asarray(result.conf_int())[: model.k_fe]
    terms = [
        RegressionTerm(
            ablation,
            name,
            float(result.fe_params[name]),
            float(result.bse_fe[name]),
            float(bounds[index, 0]),
            float(bounds[index, 1]),
            float(result.pvalues[name]),
            len(rows),
        )
        for index, name in enumerate(fixed)
    ]
    nan = float("nan")
    terms += [
        RegressionTerm(ablation, f"variance: {name}", float(value), nan, nan, nan, nan, len(rows))
        for name, value in zip(model.exog_vc.names, result.vcomp, strict=True)
    ]
    return terms


def compare_states(
    ablation: str, rows: Sequence[dict[str, object]], random_numbers: np.random.Generator
) -> list[StateComparison]:
    """閾値ごと・状態ごとの依存度の平均と、2段のブートストラップの区間。"""
    comparisons = []
    for threshold in THRESHOLDS:
        for state in ("both", "meaning-only", "integration-only", "neither"):
            members = [row for row in rows if _state(row, threshold) == state]
            if not members:
                comparisons.append(
                    StateComparison(ablation, threshold, state, 0, 0, *(float("nan"),) * 3)
                )
                continue
            values = np.array([[row["dependence"]] for row in members])
            words = [row["unknown_word"] for row in members]
            samples = weighted_means(
                bootstrap_weights(words, BOOTSTRAP_SAMPLES, random_numbers), values
            )[:, 0]
            lower, upper = np.percentile(samples, [2.5, 97.5])
            comparisons.append(
                StateComparison(
                    ablation,
                    threshold,
                    state,
                    len(members),
                    len(set(words)),
                    float(values.mean()),
                    float(lower),
                    float(upper),
                )
            )
    return comparisons


def _state(row: dict[str, object], threshold: float) -> str:
    meaning = row["meaning"] >= threshold
    integration = row["integration"] >= threshold
    if meaning and integration:
        return "both"
    if meaning:
        return "meaning-only"
    return "integration-only" if integration else "neither"
