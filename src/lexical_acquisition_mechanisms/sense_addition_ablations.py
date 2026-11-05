"""語義の追加のアブレーションの表（設計書 2-5。sense-addition make-report が書く）。

効く head は、ターゲット語の位置の効果の地図で、効果が正で有意な head（MLP を除く）。層の深さは、
相対的な深さ（層の番号 ÷ (層の数 − 1)。最初の層が 0、最後の層が 1）で比べる。
- 足す語義の数 N: 種類 × N ごとの、効く head の数・その正規化した効果の平均・相対的な深さの平均と、
  N = 1 の効く head との重なり（Jaccard 係数: 共通の数 ÷ 合わせた数）
- ターゲット語のトークン数: 種類 × N × トークン数ごとの、効く head の数・相対的な深さの平均・層の
  中央値（select-significant-heads-and-mlps の、トークン数ごとの地図から）
- モデルサイズ: モデル × 分類（手順2の component_classes の、MLP でないもの）ごとの、head の数と、
  相対的な深さの平均・中央値
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from lexical_acquisition_mechanisms.component_adoption import ComponentClass
from lexical_acquisition_mechanisms.component_patching import MLP
from lexical_acquisition_mechanisms.significant_heads_and_mlps import (
    SCOPES,
    ComponentEffect,
    TokenCountEffect,
)


@dataclass
class NumberOfSensesAblation:
    kind: str
    number_of_senses: int
    effective_heads: int
    mean_normalized_effect: float
    mean_relative_depth: float
    # N = 1 の効く head との重なり（Jaccard 係数。どちらにもなければ空）
    overlap_with_one_sense: float


@dataclass
class TokenCountAblation:
    kind: str
    number_of_senses: int
    token_count: int
    effective_heads: int
    mean_relative_depth: float
    median_layer: float


@dataclass
class ModelSizeAblation:
    component_class: str
    heads: int
    number_of_layers: int
    mean_relative_depth: float
    median_relative_depth: float


def number_of_layers(effects: Sequence[ComponentEffect]) -> int:
    return max((row.layer for row in effects), default=-1) + 1


def by_number_of_senses(effects: Sequence[ComponentEffect]) -> list[NumberOfSensesAblation]:
    layers = number_of_layers(effects)
    groups = _effective_heads(effects)
    rows = []
    for (kind, number), heads in sorted(groups.items()):
        one = {(row.layer, row.head) for row in groups.get((kind, 1), [])}
        mine = {(row.layer, row.head) for row in heads}
        union = one | mine
        rows.append(
            NumberOfSensesAblation(
                kind,
                number,
                len(heads),
                _mean([row.normalized_effect for row in heads]),
                _mean([_depth(row.layer, layers) for row in heads]),
                len(one & mine) / len(union) if union else float("nan"),
            )
        )
    return rows


def by_token_count(effects: Sequence[TokenCountEffect], layers: int) -> list[TokenCountAblation]:
    rows = []
    for token_count in sorted({row.token_count for row in effects}):
        mine = [row for row in effects if row.token_count == token_count]
        for (kind, number), heads in sorted(_effective_heads(mine).items()):
            rows.append(
                TokenCountAblation(
                    kind,
                    number,
                    token_count,
                    len(heads),
                    _mean([_depth(row.layer, layers) for row in heads]),
                    float(np.median([row.layer for row in heads])) if heads else float("nan"),
                )
            )
    return rows


def by_model_size(classes: Sequence[ComponentClass], layers: int) -> list[ModelSizeAblation]:
    rows = []
    for name in sorted({row.component_class for row in classes}):
        depths = [
            _depth(row.layer, layers)
            for row in classes
            if row.component_class == name and row.head != MLP
        ]
        if depths:
            rows.append(
                ModelSizeAblation(
                    name, len(depths), layers, float(np.mean(depths)), float(np.median(depths))
                )
            )
    return rows


def _effective_heads(
    effects: Sequence[ComponentEffect],
) -> dict[tuple[str, int], list[ComponentEffect]]:
    """種類 × N ごとの、ターゲット語の位置で効果が正で有意な head。

    効く head がない条件も、空の一覧で入れる。
    """
    groups: dict[tuple[str, int], list[ComponentEffect]] = {}
    for row in effects:
        if row.scope != SCOPES[0]:
            continue
        heads = groups.setdefault((row.kind, row.number_of_senses), [])
        if row.head != MLP and row.significant and row.effect > 0:
            heads.append(row)
    return groups


def _depth(layer: int, layers: int) -> float:
    return layer / (layers - 1) if layers > 1 else 0.0


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if len(values) else float("nan")
