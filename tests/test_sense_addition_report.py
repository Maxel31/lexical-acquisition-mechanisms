"""語義の追加の図と表のテスト（作った結果の表で）。"""

import csv
from pathlib import Path

from lexical_acquisition_mechanisms.attribute_judgments import AttributeJudgingHead
from lexical_acquisition_mechanisms.component_adoption import ComponentClass
from lexical_acquisition_mechanisms.csv_tables import write_rows
from lexical_acquisition_mechanisms.sense_accuracy import SenseAccuracy
from lexical_acquisition_mechanisms.sense_addition_report import write_report
from lexical_acquisition_mechanisms.significant_heads_and_mlps import ComponentEffect

KINDS = ("same-category", "other-category", "adjective")


def test_figures_and_tables_are_written(tmp_path: Path) -> None:
    accuracy = [
        SenseAccuracy(kind, number, "car", sense, 20, 0.9, 0.5, 0.4, 0.5, 0.001, True)
        for kind in KINDS
        for number in (1, 2, 4)
        for sense in ("pasta", "original")
    ]
    effects = [
        ComponentEffect(
            kind,
            number,
            scope,
            layer,
            head,
            0.2,
            0.1,
            0.1,
            0.05 * layer,
            0.0,
            0.2,
            0.01,
            0.02,
            head == 1,
            20,
            3,
        )
        for kind in KINDS
        for number in (1, 2, 4)
        for scope in ("target_position", "after_target_positions")
        for layer in (0, 1)
        for head in (0, 1, -1)
    ]
    classes = [ComponentClass(0, 1, "none", "adopted", "adopted", "category")]
    judging = [AttributeJudgingHead(0, 1, "category", 0.8, 0.7, 0.9, True, True, True, True)]
    paths = tuple(
        tmp_path / name for name in ("accuracy.csv", "effects.csv", "classes.csv", "judging.csv")
    )
    for path, row_type, rows in zip(
        paths,
        (SenseAccuracy, ComponentEffect, ComponentClass, AttributeJudgingHead),
        (accuracy, effects, classes, judging),
        strict=True,
    ):
        write_rows(path, row_type, rows)

    write_report(["tiny"], tmp_path / "figures", lambda model: paths)

    figures = tmp_path / "figures"
    assert (figures / "sense-accuracy-tiny.png").stat().st_size > 0
    assert (figures / "head-maps-tiny.png").stat().st_size > 0
    with (figures / "attribute_judging_heads.csv").open(encoding="utf-8") as file:
        table = list(csv.DictReader(file))
    assert [(row["model"], row["component_class"], row["judging"]) for row in table] == [
        ("tiny", "category", "True")
    ]
    with (figures / "sense_accuracy.csv").open(encoding="utf-8") as file:
        assert len(list(csv.DictReader(file))) == len(accuracy)
