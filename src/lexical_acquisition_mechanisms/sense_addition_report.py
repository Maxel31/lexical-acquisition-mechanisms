"""語義の追加の図と表を作る（sense-addition make-report）。

test-attribute-directions まで終えたモデルごとに、outputs/figures/sense-addition/ に次を書く。
- sense-accuracy-<モデル>.png: 種類ごとに、足す語義の数 N に対する、課題1の正解率の対照との差
  （ターゲット語で平均。足した語義と元の語義を別の線にする）
- head-maps-<モデル>.png: 種類 × N ごとの、ターゲット語の位置で入れ替えた正規化した効果の地図
  （層 × head。右端の列が MLP。採用の基準の前の、効果の地図で有意なものに点を打つ）
- 全てのモデルの表（モデルの列を足す）: sense_accuracy.csv（手順1）・component_classes.csv（手順2の
  分類）・attribute_judging_heads.csv（手順3の3つの基準。手順2の分類の列を足す）
"""

import argparse
import csv
import dataclasses
from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib.pyplot as pyplot
import numpy as np

from lexical_acquisition_mechanisms import (
    attribute_direction_patching,
    sense_accuracy,
    significant_heads_and_mlps,
)
from lexical_acquisition_mechanisms.attribute_judgments import AttributeJudgingHead
from lexical_acquisition_mechanisms.component_adoption import ComponentClass
from lexical_acquisition_mechanisms.component_patching import MLP
from lexical_acquisition_mechanisms.csv_tables import read_rows
from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.sense_accuracy import SenseAccuracy
from lexical_acquisition_mechanisms.sense_addition_assignments import KINDS, NUMBERS_OF_SENSES
from lexical_acquisition_mechanisms.settings import Settings
from lexical_acquisition_mechanisms.significant_heads_and_mlps import ComponentEffect

COMMAND = "sense-addition make-report"
FIGURES_DIRECTORY = Path("outputs/figures/sense-addition")


def report_inputs(model: str) -> tuple[Path, Path, Path, Path]:
    """手順1の正解率・効果の地図・分類・属性を判定する head の表。"""
    effects, _, classes = significant_heads_and_mlps.output_paths(model)
    return (
        sense_accuracy.result_paths(model)[1],
        effects,
        classes,
        attribute_direction_patching.output_paths(model)[2],
    )


def make_report(settings: Settings, arguments: argparse.Namespace) -> None:
    models = [model for model in settings.models if report_inputs(model)[3].exists()]
    inputs = {}
    for model in models:
        accuracy, effects, classes, judging = report_inputs(model)
        selected_by = f"sense-addition select-significant-heads-and-mlps --model {model}"
        inputs[accuracy] = f"sense-addition measure-sense-accuracy --model {model}"
        inputs[effects] = selected_by
        inputs[classes] = selected_by
        inputs[judging] = f"sense-addition test-attribute-directions --model {model}"
    step = Step(COMMAND, inputs=inputs, outputs=[FIGURES_DIRECTORY], settings={"models": models})
    run_unless_unchanged(step, lambda: write_report(models, FIGURES_DIRECTORY), arguments.recompute)


def write_report(
    models: Sequence[str],
    directory: Path,
    paths_of: Callable[[str], tuple[Path, Path, Path, Path]] = report_inputs,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    tables: dict[str, list[dict[str, object]]] = {"accuracy": [], "classes": [], "judging": []}
    for model in models:
        accuracy_path, effects_path, classes_path, judging_path = paths_of(model)
        accuracy = read_rows(accuracy_path, SenseAccuracy)
        classes = read_rows(classes_path, ComponentClass)
        _plot_accuracy(accuracy, directory / f"sense-accuracy-{model}.png", model)
        _plot_maps(
            read_rows(effects_path, ComponentEffect), directory / f"head-maps-{model}.png", model
        )
        class_of = {(row.layer, row.head): row.component_class for row in classes}
        tables["accuracy"] += [{"model": model, **vars(row)} for row in accuracy]
        tables["classes"] += [{"model": model, **vars(row)} for row in classes]
        tables["judging"] += [
            {
                "model": model,
                **vars(row),
                "component_class": class_of.get((row.layer, row.head), ""),
            }
            for row in read_rows(judging_path, AttributeJudgingHead)
        ]
    names = {
        "accuracy": ("sense_accuracy.csv", SenseAccuracy, []),
        "classes": ("component_classes.csv", ComponentClass, []),
        "judging": ("attribute_judging_heads.csv", AttributeJudgingHead, ["component_class"]),
    }
    for key, (file_name, row_type, extra) in names.items():
        fields = ["model", *(field.name for field in dataclasses.fields(row_type)), *extra]
        with (directory / file_name).open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(tables[key])


def _plot_accuracy(rows: Sequence[SenseAccuracy], path: Path, model: str) -> None:
    figure, axes = pyplot.subplots(1, len(KINDS), figsize=(5 * len(KINDS), 4), sharey=True)
    for axis, kind in zip(axes, KINDS, strict=True):
        for label, added in (("added senses", True), ("original sense", False)):
            means = [
                np.mean(
                    [
                        row.difference
                        for row in rows
                        if row.kind == kind
                        and row.number_of_senses == number
                        and (row.sense != "original") == added
                    ]
                    or [np.nan]
                )
                for number in NUMBERS_OF_SENSES
            ]
            axis.plot(NUMBERS_OF_SENSES, means, marker="o", label=label)
        axis.axhline(0, color="gray", linestyle=":")
        axis.set_title(kind)
        axis.set_xlabel("number of added senses")
        axis.set_xticks(NUMBERS_OF_SENSES)
    axes[0].set_ylabel("accuracy - control accuracy")
    axes[0].legend(fontsize="small")
    figure.suptitle(model)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    pyplot.close(figure)


def _plot_maps(rows: Sequence[ComponentEffect], path: Path, model: str) -> None:
    rows = [row for row in rows if row.scope == significant_heads_and_mlps.SCOPES[0]]
    if not rows:
        return
    layers = max(row.layer for row in rows) + 1
    heads = max(row.head for row in rows) + 1
    values = [abs(row.normalized_effect) for row in rows if np.isfinite(row.normalized_effect)]
    limit = max(values, default=1.0) or 1.0
    figure, axes = pyplot.subplots(
        len(KINDS), len(NUMBERS_OF_SENSES), figsize=(4 * len(NUMBERS_OF_SENSES), 4 * len(KINDS))
    )
    for row_axes, kind in zip(np.atleast_2d(axes), KINDS, strict=True):
        for axis, number in zip(row_axes, NUMBERS_OF_SENSES, strict=True):
            table = np.full((layers, heads + 1), np.nan)
            marks = []
            for row in rows:
                if row.kind == kind and row.number_of_senses == number:
                    column = heads if row.head == MLP else row.head
                    table[row.layer, column] = row.normalized_effect
                    if row.significant:
                        marks.append((column, row.layer))
            image = axis.imshow(
                table, aspect="auto", origin="lower", cmap="coolwarm", vmin=-limit, vmax=limit
            )
            if marks:
                axis.scatter(*zip(*marks, strict=True), s=4, color="black")
            axis.set_title(f"{kind}, N = {number}")
            axis.set_xlabel("head (last column: MLP)")
            axis.set_ylabel("layer")
    figure.colorbar(image, ax=axes, label="normalized effect")
    figure.suptitle(model)
    figure.savefig(path, dpi=120)
    pyplot.close(figure)
