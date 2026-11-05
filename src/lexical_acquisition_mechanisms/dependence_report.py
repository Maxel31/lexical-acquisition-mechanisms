"""カテゴリ判定 head への未知語の依存の図と表を作る
（dependence-on-category-judging-heads make-report）。

fit-regression まで終えたモデルごとに、outputs/figures/dependence-on-category-judging-heads/ に
次を書く。
- dependence-<モデル>.png: 意味と統合の m̂ に対する依存度（カテゴリの方向の無効化。比較の基準の
  無効化は薄い色）
- states-<モデル>.png: 閾値ごとの、状態（both・meaning-only・integration-only・neither）ごとの
  依存度の平均と 95% の区間
- 全てのモデルの表（モデルの列を足す）: dependence.csv（未知語 × run ごとの依存度と m̂）・
  regression.csv・state_comparison.csv・head_map_comparison_summary.csv（compare-head-maps を終えた
  モデルの、順位相関と重なりの、名詞での平均と、名詞を単位にしたブートストラップの 95% の区間）
"""

import argparse
import csv
import dataclasses
import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as pyplot
import numpy as np
import pyarrow.parquet as parquet

from lexical_acquisition_mechanisms import dependence_regression, head_map_comparison
from lexical_acquisition_mechanisms.bootstrap_statistics import (
    BOOTSTRAP_SAMPLES,
    bootstrap_weights,
    weighted_means,
)
from lexical_acquisition_mechanisms.category_dependence import ABLATIONS, EXPERIMENT
from lexical_acquisition_mechanisms.csv_tables import read_rows
from lexical_acquisition_mechanisms.dependence_regression import (
    THRESHOLDS,
    RegressionTerm,
    StateComparison,
)
from lexical_acquisition_mechanisms.head_map_comparison import HeadMapComparison
from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.settings import Settings

COMMAND = f"{EXPERIMENT} make-report"
FIGURES_DIRECTORY = Path("outputs/figures") / EXPERIMENT
STATES = ("both", "meaning-only", "integration-only", "neither")


@dataclass
class HeadMapComparisonSummary:
    measure: str
    nouns: int
    mean: float
    lower: float
    upper: float


def report_inputs(model: str) -> tuple[Path, Path, Path, Path]:
    """依存度と m̂ の表・回帰・状態ごとの比較・地図の比較（なくてもよい）。"""
    return (
        *dependence_regression.output_paths(model),
        head_map_comparison.output_paths(model)[1],
    )


def make_report(settings: Settings, arguments: argparse.Namespace) -> None:
    models = [model for model in settings.models if report_inputs(model)[1].exists()]
    inputs = {}
    for model in models:
        data, regression, states, comparison = report_inputs(model)
        for path in (data, regression, states):
            inputs[path] = f"{EXPERIMENT} fit-regression --model {model}"
        if comparison.exists():
            inputs[comparison] = f"{EXPERIMENT} compare-head-maps --model {model}"
    step = Step(
        COMMAND,
        inputs=inputs,
        outputs=[FIGURES_DIRECTORY],
        settings={"models": models, "seed": settings.words.seed},
    )
    run_unless_unchanged(
        step,
        lambda: write_report(models, FIGURES_DIRECTORY, settings.words.seed),
        arguments.recompute,
    )


def write_report(
    models: Sequence[str],
    directory: Path,
    seed: int,
    paths_of: Callable[[str], tuple[Path, Path, Path, Path]] = report_inputs,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    tables: dict[str, list[dict[str, object]]] = {
        name: [] for name in ("data", "terms", "states", "maps")
    }
    for model in models:
        # モデルごとに乱数を分ける（ほかのモデルの結果の有無で、区間が変わらないように）
        random_numbers = np.random.default_rng([seed, zlib.crc32(model.encode())])
        data_path, regression_path, states_path, comparison_path = paths_of(model)
        data = parquet.read_table(data_path).to_pylist()
        states = read_rows(states_path, StateComparison)
        _plot_dependence(data, directory / f"dependence-{model}.png", model)
        _plot_states(states, directory / f"states-{model}.png", model)
        tables["data"] += [{"model": model, **row} for row in data]
        tables["terms"] += [
            {"model": model, **vars(row)} for row in read_rows(regression_path, RegressionTerm)
        ]
        tables["states"] += [{"model": model, **vars(row)} for row in states]
        if comparison_path.exists():
            comparisons = read_rows(comparison_path, HeadMapComparison)
            tables["maps"] += [
                {"model": model, **vars(row)} for row in _summaries(comparisons, random_numbers)
            ]
    data_fields = sorted({key for row in tables["data"] for key in row} - {"model"})
    files = {
        "data": ("dependence.csv", data_fields),
        "terms": ("regression.csv", _fields(RegressionTerm)),
        "states": ("state_comparison.csv", _fields(StateComparison)),
        "maps": ("head_map_comparison_summary.csv", _fields(HeadMapComparisonSummary)),
    }
    for key, (file_name, fields) in files.items():
        with (directory / file_name).open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["model", *fields])
            writer.writeheader()
            writer.writerows(tables[key])


def _fields(row_type: type) -> list[str]:
    return [field.name for field in dataclasses.fields(row_type)]


def _summaries(
    comparisons: Sequence[HeadMapComparison], random_numbers: np.random.Generator
) -> list[HeadMapComparisonSummary]:
    """順位相関と重なりの、名詞での平均と、名詞を単位にしたブートストラップの 95% の区間。"""
    summaries = []
    for measure in ("spearman", "top_overlap", "chance_overlap"):
        values = np.array([[getattr(row, measure)] for row in comparisons])
        if not len(values):
            continue
        nouns = [row.noun for row in comparisons]
        samples = weighted_means(
            bootstrap_weights(nouns, BOOTSTRAP_SAMPLES, random_numbers), values
        )
        lower, upper = np.nanpercentile(samples[:, 0], [2.5, 97.5])
        summaries.append(
            HeadMapComparisonSummary(
                measure, len(values), float(np.nanmean(values)), float(lower), float(upper)
            )
        )
    return summaries


def _plot_dependence(data: Sequence[dict[str, object]], path: Path, model: str) -> None:
    figure, axes = pyplot.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, indicator in zip(axes, ("meaning", "integration"), strict=True):
        for ablation in reversed(ABLATIONS):
            rows = [row for row in data if row["ablation"] == ablation]
            main = ablation == ABLATIONS[0]
            axis.scatter(
                [row[indicator] for row in rows],
                [row["dependence"] for row in rows],
                s=10 if main else 6,
                alpha=0.8 if main else 0.3,
                label=ablation,
            )
        axis.axhline(0, color="gray", linestyle=":")
        axis.set_xlabel(f"normalized {indicator}")
    axes[0].set_ylabel("dependence")
    axes[0].legend(fontsize="small")
    figure.suptitle(model)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    pyplot.close(figure)


def _plot_states(states: Sequence[StateComparison], path: Path, model: str) -> None:
    figure, axes = pyplot.subplots(
        1, len(THRESHOLDS), figsize=(5 * len(THRESHOLDS), 4), sharey=True
    )
    for axis, threshold in zip(axes, THRESHOLDS, strict=True):
        rows = {
            row.state: row
            for row in states
            if row.threshold == threshold and row.ablation == ABLATIONS[0]
        }
        means = [rows[state].mean_dependence if state in rows else np.nan for state in STATES]
        errors = [
            [means[index] - rows[state].lower, rows[state].upper - means[index]]
            if state in rows
            else [np.nan, np.nan]
            for index, state in enumerate(STATES)
        ]
        axis.bar(range(len(STATES)), means, yerr=np.array(errors).T, capsize=4)
        axis.set_xticks(range(len(STATES)), STATES, rotation=30)
        axis.set_title(f"threshold {threshold}")
    axes[0].set_ylabel("mean dependence")
    figure.suptitle(model)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    pyplot.close(figure)
