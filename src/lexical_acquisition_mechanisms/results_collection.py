"""全ての結果を1つの表にまとめ、モデルサイズの比較の図を作る（results collect）。

分析を終えたモデルの結果を、縦長の1つの表 outputs/results/all_results.parquet にまとめる。列は
experiment・model・run・word・condition・number_of_senses・indicator・layer・head・value（使わない
列は空）。
- unknown-word-acquisition: 未知語の指標（indicator は form・form-second・form-third・integration・
  meaning。形への慣れの layer は −1）と、正規化した値（indicator の後に "-normalized"）
- sense-addition: 課題1の正解率（sense-accuracy。condition は語義）と対照との差
  （sense-accuracy-difference）、構成要素の効果（head-effect と head-effect-normalized。condition は
  入れ替えた位置。MLP の head は −1）、手順2で採用した構成要素（component-class。condition は
  分類、値は 1）、属性のプローブの確認用の語での正解率（attribute-probe-accuracy。condition は
  属性）、属性の方向の入れ替えの比べ（condition は属性、indicator は「課題-値の名前」。例:
  category-attribute-change・sense-random-head-p-value）、属性を判定する head の基準
  （attribute-judging-{encoding|causal|specificity|judging}。値は 0・1）
- unknown-word-acquisition の獲得の順番（half-exposure-{指標} とその lower・upper、
  reached-share-{指標}。condition は語の集まり）
- dependence-on-category-judging-heads: 依存度（dependence。condition は無効化の種類）と、回帰の
  係数と p 値（regression-{項} と regression-{項}-p-value）
- words: 語の共変量（base-data-count: 基データでの出現回数、token-count。condition は役割）
図 outputs/figures/model_sizes.png: モデルごとの、3つの指標が 0.5 に達する露出回数（全体）・
手順2の分類ごとの構成要素の数・依存度の回帰の、意味と統合の係数と 95% の区間。
"""

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib.pyplot as pyplot
import numpy as np
import pyarrow as arrow
import pyarrow.parquet as parquet

from lexical_acquisition_mechanisms import (
    acquisition_order,
    attribute_direction_patching,
    attribute_probes,
    dependence_regression,
    sense_accuracy,
    significant_heads_and_mlps,
    unknown_word_assignments,
    unknown_word_indicators,
    word_finalization,
)
from lexical_acquisition_mechanisms.acquisition_order import INDICATORS, AcquisitionOrder
from lexical_acquisition_mechanisms.attribute_judgments import (
    AttributeDirectionTest,
    AttributeJudgingHead,
)
from lexical_acquisition_mechanisms.attribute_probes import AttributeProbeAccuracy
from lexical_acquisition_mechanisms.category_dependence import ABLATIONS
from lexical_acquisition_mechanisms.category_dependence import EXPERIMENT as CONNECTION
from lexical_acquisition_mechanisms.component_adoption import ComponentClass
from lexical_acquisition_mechanisms.csv_tables import read_rows
from lexical_acquisition_mechanisms.dependence_regression import RegressionTerm
from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.sense_accuracy import SenseAccuracy
from lexical_acquisition_mechanisms.sense_addition_assignments import EXPERIMENT as SENSES
from lexical_acquisition_mechanisms.settings import Settings
from lexical_acquisition_mechanisms.significant_heads_and_mlps import ComponentEffect
from lexical_acquisition_mechanisms.word_selection import SelectedWord

COMMAND = "results collect"
RESULTS_PATH = Path("outputs/results/all_results.parquet")
FIGURE_PATH = Path("outputs/figures/model_sizes.png")
UNKNOWN_WORDS = unknown_word_assignments.EXPERIMENT
CLASSES = ("sense-selection", "category", "part-of-speech", "within-category", "other")


def result_files(model: str) -> dict[Path, str]:
    """モデルの結果のファイルと、それを書くコマンド（まだないファイルも含む）。

    並びは、未知語の指標・正規化した指標・獲得の順番・課題1の正解率・構成要素の効果・分類・
    回帰に使った表・回帰の係数・属性の方向の入れ替えの比べ・属性を判定する head・属性のプローブの
    正解率（write_results がこの順で読む）。
    """
    order = f"{UNKNOWN_WORDS} analyze-acquisition-order --model {model}"
    selected = f"{SENSES} select-significant-heads-and-mlps --model {model}"
    regression = f"{CONNECTION} fit-regression --model {model}"
    tested = f"{SENSES} test-attribute-directions --model {model}"
    normalized, _, order_path = acquisition_order.output_paths(model)
    effects, _, classes = significant_heads_and_mlps.output_paths(model)
    data, terms, _ = dependence_regression.output_paths(model)
    _, direction_tests, judging, _ = attribute_direction_patching.output_paths(model)
    return {
        unknown_word_indicators.indicators_path(model): (
            f"{UNKNOWN_WORDS} measure-indicators --model {model}"
        ),
        normalized: order,
        order_path: order,
        sense_accuracy.result_paths(model)[1]: f"{SENSES} measure-sense-accuracy --model {model}",
        effects: selected,
        classes: selected,
        data: regression,
        terms: regression,
        direction_tests: tested,
        judging: tested,
        attribute_probes.output_paths(model)[1]: (
            f"{SENSES} train-attribute-probes --model {model}"
        ),
    }


def collect_results(settings: Settings, arguments: argparse.Namespace) -> None:
    inputs = {
        path: command
        for model in settings.models
        for path, command in result_files(model).items()
        if path.exists()
    }
    inputs[word_finalization.OUTPUT_PATH] = word_finalization.COMMAND
    step = Step(COMMAND, inputs=inputs, outputs=[RESULTS_PATH, FIGURE_PATH])
    run_unless_unchanged(
        step,
        lambda: write_results(
            list(settings.models),
            (RESULTS_PATH, FIGURE_PATH),
            words_path=word_finalization.OUTPUT_PATH,
        ),
        arguments.recompute,
    )


def write_results(
    models: Sequence[str],
    outputs: tuple[Path, Path],
    files_of: Callable[[str], dict[Path, str]] = result_files,
    words_path: Path | None = None,
) -> None:
    results_path, figure_path = outputs
    rows: list[dict[str, object]] = []
    summaries = {}
    for model in models:
        paths = list(files_of(model))
        indicators, normalized, order, accuracy, effects, classes, data, terms = paths[:8]
        rows += _summary_rows(model, paths)
        if indicators.exists() and normalized.exists():
            rows += _unknown_word_rows(model, indicators, normalized)
        if accuracy.exists() and effects.exists():
            rows += _sense_addition_rows(model, accuracy, effects)
        if data.exists():
            rows += [
                _row(CONNECTION, model, row["run"], row["unknown_word"], row["ablation"])
                | {"indicator": "dependence", "value": row["dependence"]}
                for row in parquet.read_table(data).to_pylist()
            ]
        summaries[model] = [
            read_rows(path, row_type) if path.exists() else []
            for path, row_type in (
                (order, AcquisitionOrder),
                (classes, ComponentClass),
                (terms, RegressionTerm),
            )
        ]
    if words_path is not None:
        for word in read_rows(words_path, SelectedWord):
            base = _row("words", "", "", word.word, word.role)
            rows.append(base | {"indicator": "base-data-count", "value": word.base_data_count})
            rows.append(base | {"indicator": "token-count", "value": word.token_count})
    results_path.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_table(arrow.Table.from_pylist(rows), results_path)
    _plot_model_sizes(summaries, figure_path)


def _row(experiment: str, model: str, run: str, word: str, condition: str) -> dict[str, object]:
    return {
        "experiment": experiment,
        "model": model,
        "run": run,
        "word": word,
        "condition": condition,
        "number_of_senses": None,
        "indicator": "",
        "layer": None,
        "head": None,
        "value": None,
    }


def _unknown_word_rows(model: str, indicators: Path, normalized: Path) -> list[dict[str, object]]:
    rows = [
        _row(UNKNOWN_WORDS, model, row["run"], row["word"], row["kind"])
        | {"indicator": row["indicator"], "layer": row["layer"], "value": row["value"]}
        for row in parquet.read_table(indicators).to_pylist()
    ]
    rows += [
        _row(UNKNOWN_WORDS, model, row["run"], row["word"], row["kind"])
        | {
            "indicator": f"{row['indicator']}-normalized",
            "layer": row["layer"],
            "value": row["normalized"],
        }
        for row in parquet.read_table(normalized).to_pylist()
    ]
    return rows


def _sense_addition_rows(model: str, accuracy: Path, effects: Path) -> list[dict[str, object]]:
    rows = []
    for row in read_rows(accuracy, SenseAccuracy):
        base = _row(SENSES, model, f"{row.kind}-{row.number_of_senses}", row.target, row.sense)
        base["number_of_senses"] = row.number_of_senses
        rows.append(base | {"indicator": "sense-accuracy", "value": row.accuracy})
        rows.append(base | {"indicator": "sense-accuracy-difference", "value": row.difference})
    for row in read_rows(effects, ComponentEffect):
        base = _row(SENSES, model, f"{row.kind}-{row.number_of_senses}", "", row.scope)
        base |= {"number_of_senses": row.number_of_senses, "layer": row.layer, "head": row.head}
        rows.append(base | {"indicator": "head-effect", "value": row.effect})
        rows.append(base | {"indicator": "head-effect-normalized", "value": row.normalized_effect})
    return rows


def _summary_rows(model: str, paths: Sequence[Path]) -> list[dict[str, object]]:
    """獲得の順番・分類・回帰の係数・属性の方向の比べ・属性を判定する head・プローブの正解率の
    行。"""
    order, classes, terms = paths[2], paths[5], paths[7]
    direction_tests, judging, probe_accuracy = paths[8:11]
    rows = []
    if order.exists():
        for row in read_rows(order, AcquisitionOrder):
            base = _row(UNKNOWN_WORDS, model, "", "", row.group) | {"layer": row.layer}
            for name, value in (
                (f"half-exposure-{row.indicator}", row.half_exposure),
                (f"half-exposure-{row.indicator}-lower", row.lower),
                (f"half-exposure-{row.indicator}-upper", row.upper),
                (f"reached-share-{row.indicator}", row.reached_share),
            ):
                rows.append(base | {"indicator": name, "value": value})
    if classes.exists():
        rows += [
            _row(SENSES, model, "", "", row.component_class)
            | {"indicator": "component-class", "layer": row.layer, "head": row.head, "value": 1.0}
            for row in read_rows(classes, ComponentClass)
        ]
    if terms.exists():
        for row in read_rows(terms, RegressionTerm):
            base = _row(CONNECTION, model, "", "", row.ablation)
            rows.append(base | {"indicator": f"regression-{row.term}", "value": row.coefficient})
            rows.append(
                base | {"indicator": f"regression-{row.term}-p-value", "value": row.p_value}
            )
    if direction_tests.exists():
        measures = (
            "attribute_change",
            "random_head_change",
            "random_direction_change",
            "attribute_switch_rate",
            "random_head_p_value",
            "random_direction_p_value",
        )
        for row in read_rows(direction_tests, AttributeDirectionTest):
            base = _row(SENSES, model, row.run, "", row.attribute)
            base |= {"layer": row.layer, "head": row.head}
            for measure in measures:
                name = f"{row.task}-{measure.replace('_', '-')}"
                rows.append(base | {"indicator": name, "value": getattr(row, measure)})
    if judging.exists():
        for row in read_rows(judging, AttributeJudgingHead):
            base = _row(SENSES, model, "", "", row.attribute)
            base |= {"layer": row.layer, "head": row.head}
            for measure in ("encoding", "causal", "specificity", "judging"):
                value = float(getattr(row, measure))
                rows.append(base | {"indicator": f"attribute-judging-{measure}", "value": value})
    if probe_accuracy.exists():
        rows += [
            _row(SENSES, model, "", "", row.attribute)
            | {
                "indicator": "attribute-probe-accuracy",
                "layer": row.layer,
                "head": row.head,
                "value": row.held_out_accuracy,
            }
            for row in read_rows(probe_accuracy, AttributeProbeAccuracy)
            if not row.baseline
        ]
    return rows


def _plot_model_sizes(summaries: dict[str, list[list]], path: Path) -> None:
    models = list(summaries)
    positions = np.arange(len(models))
    figure, axes = pyplot.subplots(1, 3, figsize=(16, 4))
    width = 0.8 / len(INDICATORS)
    for number, indicator in enumerate(INDICATORS):
        values = [
            next(
                (
                    order.half_exposure
                    for order in summary[0]
                    if order.group == "all" and order.indicator == indicator
                ),
                np.nan,
            )
            for summary in summaries.values()
        ]
        axes[0].bar(positions + number * width, values, width, label=indicator)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("exposure to reach 0.5")
    width = 0.8 / len(CLASSES)
    for number, name in enumerate(CLASSES):
        counts = [
            sum(row.component_class == name for row in summary[1]) for summary in summaries.values()
        ]
        axes[1].bar(positions + number * width, counts, width, label=name)
    axes[1].set_ylabel("adopted components")
    for number, term in enumerate(("meaning", "integration")):
        rows = [
            next(
                (row for row in summary[2] if row.term == term and row.ablation == ABLATIONS[0]),
                None,
            )
            for summary in summaries.values()
        ]
        values = [row.coefficient if row else np.nan for row in rows]
        errors = [
            [row.coefficient - row.lower, row.upper - row.coefficient] if row else [np.nan] * 2
            for row in rows
        ]
        axes[2].bar(positions + number * 0.4, values, 0.4, yerr=np.array(errors).T, label=term)
    axes[2].axhline(0, color="gray", linestyle=":")
    axes[2].set_ylabel("coefficient for dependence")
    titles = ("unknown-word acquisition", "sense addition", "dependence on category heads")
    for axis, title in zip(axes, titles, strict=True):
        axis.set_xticks(positions + 0.3, models, rotation=15)
        axis.set_title(title)
        axis.legend(fontsize="small")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    pyplot.close(figure)
