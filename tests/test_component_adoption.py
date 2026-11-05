"""採用の基準と分類のテスト（作った効果の値で）。"""

import numpy as np

from lexical_acquisition_mechanisms.component_adoption import (
    ComponentSelection,
    classify,
    select_components,
    target_halves,
)
from lexical_acquisition_mechanisms.word_selection import SelectedWord


def test_targets_are_split_evenly_by_token_count() -> None:
    words = [
        SelectedWord(f"{category}-{count}", "target", category=category, token_count=count)
        for category in ("food", "animal")
        for count in (1, 2, 3)
    ]

    halves = target_halves(words, ["food", "animal"])

    assert halves == {
        "food-1": 1,
        "food-2": 0,
        "food-3": 1,
        "animal-1": 0,
        "animal-2": 1,
        "animal-3": 0,
    }


def test_components_need_a_positive_slope_and_both_halves() -> None:
    random_numbers = np.random.default_rng(0)
    halves = {f"target-{index}": int(index >= 3) for index in range(6)}
    base = {1: 0.2, 2: 0.4, 4: 0.8}
    differences = {}
    for number in (1, 2, 4):
        keys = [
            (target, f"donor-{donor}", f"sentence-{sentence}")
            for target in halves
            for donor in range(number)
            for sentence in range(2)
        ]
        values = []
        for target, _, _ in keys:
            # 0: N とともに強まる。1: N によらない。2: 前半のターゲット語でだけ効く
            first_half = base[number] if halves[target] == 0 else 0.0
            values.append([base[number], 0.5, first_half] + random_numbers.normal(0, 0.05, 3))
        differences["other-category", number] = (keys, np.array(values))
    significant = {("other-category", 4): np.array([True, True, True])}

    selections = select_components(
        differences, significant, halves, np.array([0, 0, 1]), np.array([0, -1, 0]), 0
    )

    assert [(row.kind, row.layer, row.head) for row in selections] == [
        ("other-category", 0, 0),
        ("other-category", 0, -1),
        ("other-category", 1, 0),
    ]
    increasing, constant, one_half = selections
    assert abs(increasing.slope - 0.2) < 0.02
    assert increasing.replicated and increasing.adopted
    assert constant.slope_p_value > 0.05 and not constant.adopted
    assert one_half.found_in_first_half and one_half.second_half_p_value > 0.05
    assert not one_half.replicated and not one_half.adopted


def test_components_are_classified_by_where_they_are_adopted() -> None:
    def row(kind: str, head: int, state: str) -> ComponentSelection:
        return ComponentSelection(
            kind,
            0,
            head,
            1.0,
            state != "none",
            0.1,
            0.0,
            0.0,
            0.0,
            True,
            True,
            True,
            state == "adopted",
        )

    states = {
        0: ("adopted", "adopted", "adopted"),
        1: ("none", "adopted", "adopted"),
        2: ("none", "none", "adopted"),
        3: ("adopted", "none", "none"),
        4: ("significant", "adopted", "none"),
        5: ("significant", "none", "none"),
    }
    kinds = ("same-category", "other-category", "adjective")
    selections = [
        row(kind, head, state)
        for head, values in states.items()
        for kind, state in zip(kinds, values, strict=True)
    ]

    classes = classify(selections)

    assert [(row.head, row.component_class) for row in classes] == [
        (0, "sense-selection"),
        (1, "category"),
        (2, "part-of-speech"),
        (3, "within-category"),
        (4, "other"),
    ]
    assert (classes[4].same_category, classes[4].other_category) == ("significant", "adopted")
