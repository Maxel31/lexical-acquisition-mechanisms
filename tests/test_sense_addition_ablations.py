"""語義の追加のアブレーションの表のテスト（作った効果の地図で）。"""

import math

import pytest

from lexical_acquisition_mechanisms.component_adoption import ComponentClass
from lexical_acquisition_mechanisms.sense_addition_ablations import (
    by_model_size,
    by_number_of_senses,
    by_token_count,
)
from lexical_acquisition_mechanisms.significant_heads_and_mlps import (
    ComponentEffect,
    TokenCountEffect,
)


def effect(number: int, layer: int, head: int, significant: bool, value: float = 0.2):
    """別カテゴリ・ターゲット語の位置の行（正規化した効果は、効果の2倍）。"""
    return ComponentEffect(
        "other-category",
        number,
        "target_position",
        layer,
        head,
        0.0,
        0.0,
        value,
        2 * value,
        0.0,
        0.0,
        0.0,
        0.0,
        significant,
        10,
        3,
    )


def test_effective_heads_are_compared_across_numbers_of_senses() -> None:
    # 5層。N = 1 で効く head は (1, 0)、N = 4 では (1, 0) と (4, 1)。MLP と負の効果と有意でない
    # head は数えない
    effects = [
        effect(1, 1, 0, True),
        effect(1, 2, 0, False),
        effect(4, 1, 0, True),
        effect(4, 4, 1, True, 0.4),
        effect(4, 3, -1, True),
        effect(4, 0, 0, True, -0.3),
    ]

    rows = {row.number_of_senses: row for row in by_number_of_senses(effects)}

    assert (rows[1].effective_heads, rows[4].effective_heads) == (1, 2)
    assert rows[4].mean_normalized_effect == pytest.approx(0.6)
    assert rows[4].mean_relative_depth == pytest.approx((1 / 4 + 4 / 4) / 2)
    assert (rows[1].overlap_with_one_sense, rows[4].overlap_with_one_sense) == (1.0, 0.5)


def test_layers_of_effective_heads_by_token_count() -> None:
    effects = [
        TokenCountEffect(**vars(effect(4, layer, 0, True)), token_count=count)
        for count, layer in ((1, 1), (1, 3), (3, 8))
    ]

    rows = {row.token_count: row for row in by_token_count(effects, 9)}

    assert (rows[1].effective_heads, rows[1].median_layer) == (2, 2.0)
    assert rows[1].mean_relative_depth == pytest.approx(0.25)
    assert rows[3].mean_relative_depth == 1.0


def test_relative_depth_of_classes_by_model() -> None:
    classes = [
        ComponentClass(2, 0, "none", "adopted", "adopted", "category"),
        ComponentClass(4, 1, "none", "adopted", "adopted", "category"),
        ComponentClass(4, -1, "none", "adopted", "adopted", "category"),
    ]

    (row,) = by_model_size(classes, 5)

    assert (row.component_class, row.heads, row.number_of_layers) == ("category", 2, 5)
    assert row.mean_relative_depth == pytest.approx(0.75)
    assert not math.isnan(row.median_relative_depth)
