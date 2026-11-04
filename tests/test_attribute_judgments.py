"""属性を判定する head を、3つの基準で決める処理のテスト（作った値で）。"""

from pathlib import Path

import numpy as np

from lexical_acquisition_mechanisms.attribute_judgments import (
    AttributeJudgingHead,
    write_attribute_judgments,
)
from lexical_acquisition_mechanisms.attribute_probes import (
    AttributeProbeAccuracy,
    AttributeProbeTransfer,
)
from lexical_acquisition_mechanisms.csv_tables import read_rows, write_rows


def patching_rows(
    head: int, run: str, task: str, effect: float, random_numbers, attribute: str = "category"
) -> list[dict]:
    """1つの head・run・課題の、3つの入れ替えの行。属性の方向だけ effect だけ大きく動かす。"""
    rows = []
    for target in ("car", "lamp", "cup"):
        for sentence in ("first", "second"):
            for direction in ("donor-to-target", "target-to-donor"):
                for patch, base in (
                    ("attribute", effect),
                    ("random-head", 0.0),
                    ("random-direction", 0.0),
                ):
                    rows.append(
                        {
                            "run": run,
                            "target": target,
                            "donor": "pasta",
                            "donor_sentence": sentence,
                            "direction": direction,
                            "task": task,
                            "layer": 0,
                            "head": head,
                            "attribute": attribute,
                            "patch": patch,
                            "metric_change": base + random_numbers.normal(0, 0.05),
                            "switched": base > 0.5,
                        }
                    )
    return rows


def test_heads_need_encoding_causal_effect_and_specificity(tmp_path: Path) -> None:
    random_numbers = np.random.default_rng(0)
    rows = []
    # head 1: 別カテゴリと形容詞の課題1で効き、同カテゴリでは効かない。head 2: 同カテゴリでも効く
    for head, same_category in ((1, 0.0), (2, 1.0)):
        rows += patching_rows(head, "other-category-4", "category", 1.0, random_numbers)
        for run, effect in (
            ("other-category-4", 1.0),
            ("adjective-4", 1.0),
            ("same-category-4", same_category),
        ):
            rows += patching_rows(head, run, "sense", effect, random_numbers)
    accuracy = [
        AttributeProbeAccuracy(0, head, "category", 0.9, 0.8, 40, 1 / 6, 0.001, True, False)
        for head in (1, 2)
    ]
    write_rows(tmp_path / "accuracy.csv", AttributeProbeAccuracy, accuracy)
    transfer = [
        AttributeProbeTransfer("other-category-4", 0, head, "category", 40, 0.7, 1 / 6, 0.001, True)
        for head in (1, 2)
    ]
    write_rows(tmp_path / "transfer.csv", AttributeProbeTransfer, transfer)
    probes = {}
    for head in (1, 2):
        probes |= {
            f"0-{head}-category-weights": random_numbers.normal(size=(6, 8)),
            f"0-{head}-category-biases": np.zeros(6),
            f"0-{head}-category-mean": np.full(8, 0.5),
            f"0-{head}-category-scale": np.ones(8),
        }
    np.savez(tmp_path / "probes.npz", **probes)

    write_attribute_judgments(
        rows,
        (tmp_path / "accuracy.csv", tmp_path / "transfer.csv", tmp_path / "probes.npz"),
        (tmp_path / "tests.csv", tmp_path / "heads.csv", tmp_path / "directions.npz"),
        0,
    )

    heads = read_rows(tmp_path / "heads.csv", AttributeJudgingHead)
    assert [(row.head, row.encoding, row.causal, row.specificity) for row in heads] == [
        (1, True, True, True),
        (2, True, True, False),
    ]
    assert [row.judging for row in heads] == [True, False]
    directions = np.load(tmp_path / "directions.npz")
    assert sorted(directions.files) == ["0-1-category-basis", "0-1-category-mean"]
    assert directions["0-1-category-basis"].shape == (8, 5)
    np.testing.assert_allclose(directions["0-1-category-mean"], 0.5)


def test_part_of_speech_heads_must_not_work_where_parts_of_speech_agree(tmp_path: Path) -> None:
    random_numbers = np.random.default_rng(0)
    rows = []
    # head 3: 形容詞の課題1でだけ効く。head 4: 別カテゴリの課題1でも効く
    for head, other_category in ((3, 0.0), (4, 1.0)):
        rows += patching_rows(
            head, "adjective-4", "part-of-speech", 1.0, random_numbers, "part-of-speech"
        )
        for run, effect in (
            ("adjective-4", 1.0),
            ("other-category-4", other_category),
            ("same-category-4", 0.0),
        ):
            rows += patching_rows(head, run, "sense", effect, random_numbers, "part-of-speech")
    accuracy = [
        AttributeProbeAccuracy(0, head, "part-of-speech", 0.9, 0.8, 40, 0.5, 0.001, True, False)
        for head in (3, 4)
    ]
    write_rows(tmp_path / "accuracy.csv", AttributeProbeAccuracy, accuracy)
    transfer = [
        AttributeProbeTransfer("adjective-4", 0, head, "part-of-speech", 40, 0.8, 0.5, 0.001, True)
        for head in (3, 4)
    ]
    write_rows(tmp_path / "transfer.csv", AttributeProbeTransfer, transfer)
    probes = {}
    for head in (3, 4):
        probes |= {
            f"0-{head}-part-of-speech-weights": random_numbers.normal(size=(2, 8)),
            f"0-{head}-part-of-speech-biases": np.zeros(2),
            f"0-{head}-part-of-speech-mean": np.zeros(8),
            f"0-{head}-part-of-speech-scale": np.ones(8),
        }
    np.savez(tmp_path / "probes.npz", **probes)

    write_attribute_judgments(
        rows,
        (tmp_path / "accuracy.csv", tmp_path / "transfer.csv", tmp_path / "probes.npz"),
        (tmp_path / "tests.csv", tmp_path / "heads.csv", tmp_path / "directions.npz"),
        0,
    )

    heads = read_rows(tmp_path / "heads.csv", AttributeJudgingHead)
    assert [(row.head, row.specificity, row.judging) for row in heads] == [
        (3, True, True),
        (4, False, False),
    ]
