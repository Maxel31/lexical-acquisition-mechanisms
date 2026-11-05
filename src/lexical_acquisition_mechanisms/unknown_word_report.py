"""未知語の獲得の図と表を作る（unknown-word-acquisition make-report）。

analyze-acquisition-order を終えたモデルごとに、outputs/figures/unknown-word-acquisition/ に
次を書く。
- curves-<モデル>.png: 3つの指標の、露出回数に対する正規化した値（混合効果モデルの曲線）。語の
  集まり（全体・トークン数ごと・綴りの近さごと）ごとの線。統合と意味は、順番の判定に使った層の値
- layers-<モデル>-<指標>.png: 統合と意味の、層 × 露出回数の正規化した値（全体）
- acquisition_order.csv: 全てのモデルの、0.5 に達する露出回数と順番の表（モデルの列を足す）
"""

import argparse
import csv
import dataclasses
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as pyplot
import numpy as np

from lexical_acquisition_mechanisms import acquisition_order
from lexical_acquisition_mechanisms.acquisition_order import (
    GROUPS,
    INDICATORS,
    AcquisitionCurvePoint,
    AcquisitionOrder,
)
from lexical_acquisition_mechanisms.command_stopped import CommandStopped
from lexical_acquisition_mechanisms.csv_tables import read_rows
from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.settings import Settings

COMMAND = "unknown-word-acquisition make-report"
FIGURES_DIRECTORY = Path("outputs/figures/unknown-word-acquisition")


def make_report(settings: Settings, arguments: argparse.Namespace) -> None:
    models = [
        model for model in settings.models if acquisition_order.output_paths(model)[2].exists()
    ]
    if not models:
        first = next(iter(settings.models))
        raise CommandStopped(
            "analyze-acquisition-order を終えたモデルがないので、図と表を作りません。",
            rerun_from=f"unknown-word-acquisition analyze-acquisition-order --model {first}",
        )
    inputs = {}
    for model in models:
        for path in acquisition_order.output_paths(model)[1:]:
            inputs[path] = f"unknown-word-acquisition analyze-acquisition-order --model {model}"
    step = Step(COMMAND, inputs=inputs, outputs=[FIGURES_DIRECTORY], settings={"models": models})
    run_unless_unchanged(step, lambda: write_report(models, FIGURES_DIRECTORY), arguments.recompute)


def write_report(
    models: Sequence[str], directory: Path, paths_of=acquisition_order.output_paths
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    table = []
    for model in models:
        _, curves_path, order_path = paths_of(model)
        curves = read_rows(curves_path, AcquisitionCurvePoint)
        orders = read_rows(order_path, AcquisitionOrder)
        layer_of = {order.indicator: order.layer for order in orders if order.group == "all"}
        _plot_curves(curves, layer_of, directory / f"curves-{model}.png", model)
        for indicator in ("meaning", "integration"):
            _plot_layers(curves, indicator, directory / f"layers-{model}-{indicator}.png", model)
        table += [{"model": model, **vars(order)} for order in orders]
    names = ["model", *(field.name for field in dataclasses.fields(AcquisitionOrder))]
    with (directory / "acquisition_order.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=names)
        writer.writeheader()
        writer.writerows(table)


def _positions(exposures: Sequence[int]) -> np.ndarray:
    return np.log10(1 + np.array(exposures, dtype=float))


def _plot_curves(
    curves: Sequence[AcquisitionCurvePoint], layer_of: dict[str, int], path: Path, model: str
) -> None:
    figure, axes = pyplot.subplots(1, len(INDICATORS), figsize=(5 * len(INDICATORS), 4))
    for axis, indicator in zip(axes, INDICATORS, strict=True):
        for group in GROUPS:
            points = sorted(
                (point.exposure, point.normalized)
                for point in curves
                if point.group == group
                and point.indicator == indicator
                and point.layer == layer_of.get(indicator)
            )
            if points:
                exposures, values = zip(*points, strict=True)
                axis.plot(_positions(exposures), values, marker="o", label=group)
        axis.axhline(0.5, color="gray", linestyle=":")
        axis.set_title(f"{indicator} (layer {layer_of.get(indicator)})")
        axis.set_xlabel("log10(1 + exposure)")
        axis.set_ylabel("normalized value")
    axes[0].legend(fontsize="small")
    figure.suptitle(model)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    pyplot.close(figure)


def _plot_layers(
    curves: Sequence[AcquisitionCurvePoint], indicator: str, path: Path, model: str
) -> None:
    points = [point for point in curves if point.group == "all" and point.indicator == indicator]
    if not points:
        return
    layers = sorted({point.layer for point in points})
    exposures = sorted({point.exposure for point in points})
    values = np.full((len(layers), len(exposures)), np.nan)
    for point in points:
        values[layers.index(point.layer), exposures.index(point.exposure)] = point.normalized
    figure, axis = pyplot.subplots(figsize=(6, 5))
    image = axis.imshow(values, aspect="auto", origin="lower", cmap="viridis")
    axis.set_xticks(range(len(exposures)), [str(exposure) for exposure in exposures], rotation=45)
    axis.set_yticks(range(0, len(layers), max(1, len(layers) // 8)))
    axis.set_xlabel("exposure")
    axis.set_ylabel("layer")
    axis.set_title(f"{model}: {indicator}")
    figure.colorbar(image, ax=axis, label="normalized value")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    pyplot.close(figure)
