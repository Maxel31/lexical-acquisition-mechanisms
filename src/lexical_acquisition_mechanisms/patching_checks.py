"""勾配による近似の効果を、実際の入れ替えで確かめる（measure-head-and-mlp-effects の一部）。

語義の追加の run（対照の run を除く）ごとに、ターゲット語の位置での近似の効果を組で平均した値
（手順1で使える組で。なければ全ての組で）の絶対値が大きい順に CHECKED_COMPONENTS 個と、残りから
無作為に CHECKED_COMPONENTS 個の構成要素を選ぶ。その run の組から無作為に CHECKED_PAIRS 組を選び、
選んだ構成要素ごとに、実際に入れ替えた効果と近似の効果を比べる。
"""

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from lexical_acquisition_mechanisms.component_patching import (
    components,
    outputs_and_gradients,
    patched_metrics,
)
from lexical_acquisition_mechanisms.language_model import LanguageModel

CHECKED_COMPONENTS = 50
CHECKED_PAIRS = 100


@dataclass
class PatchingCheck:
    run: str
    # patching_pairs.parquet の行の番号
    pair: int
    layer: int
    # head の番号（MLP は −1）
    head: int
    # 近似の効果が大きい順に選んだもの（largest）か、無作為に選んだもの（random）か
    chosen: str
    estimated: float
    actual: float


@dataclass
class PatchingCheckSummary:
    run: str
    chosen: str
    components: int
    pairs: int
    # 近似と実際の効果の相関係数（構成要素 × 組の全ての値で）
    correlation: float
    mean_estimated: float
    mean_actual: float


# 組の行から、（処理する問題, 入れ替えのもとの問題, ターゲット語の位置, 選択肢）を作る関数
PairInputs = Callable[[dict[str, object]], tuple[list[int], list[int], list[int], tuple[int, int]]]


def check_estimates(
    model: LanguageModel,
    run: str,
    first_row: int,
    rows: Sequence[dict[str, object]],
    estimates: np.ndarray,
    inputs_of: PairInputs,
    seed: int,
) -> list[PatchingCheck]:
    """1つの run の組（patching_pairs.parquet の first_row 行目からの rows）で確かめる。

    estimates は、rows の組 × 構成要素の、ターゲット語の位置での近似の効果。
    """
    order = components(model)
    # 手順1で使える組かによらず、run の全ての組での平均で上位を選ぶ（手順1の基準を変えても、
    # 確かめ直さないように）
    means = np.abs(estimates.mean(axis=0))
    largest = [int(index) for index in np.argsort(-means, kind="stable")[:CHECKED_COMPONENTS]]
    random_numbers = random.Random(f"{seed}:{run}")
    rest = [index for index in range(len(order)) if index not in set(largest)]
    chosen = largest + random_numbers.sample(rest, min(CHECKED_COMPONENTS, len(rest)))
    labels = ["largest"] * len(largest) + ["random"] * (len(chosen) - len(largest))
    checks = []
    for index in sorted(random_numbers.sample(range(len(rows)), min(CHECKED_PAIRS, len(rows)))):
        source_ids, destination_ids, positions, choice = inputs_of(rows[index])
        metrics, outputs, _ = outputs_and_gradients(
            model, [source_ids, destination_ids], positions, [choice, choice[::-1]]
        )
        patches = [(order[number], outputs[1].of(order[number])) for number in chosen]
        actual = metrics[0] - patched_metrics(model, source_ids, positions, choice, patches)
        checks += [
            PatchingCheck(
                run,
                first_row + index,
                order[number].layer,
                order[number].head,
                label,
                float(estimates[index, number]),
                float(value),
            )
            for number, label, value in zip(chosen, labels, actual, strict=True)
        ]
    return checks


def summarize_checks(checks: Sequence[PatchingCheck]) -> list[PatchingCheckSummary]:
    """run と選び方ごとの、近似と実際の効果の相関と平均。"""
    groups: dict[tuple[str, str], list[PatchingCheck]] = {}
    for check in checks:
        groups.setdefault((check.run, check.chosen), []).append(check)
    summaries = []
    for (run, chosen), members in groups.items():
        estimated = np.array([check.estimated for check in members])
        actual = np.array([check.actual for check in members])
        varies = len(members) > 1 and estimated.std() > 0 and actual.std() > 0
        summaries.append(
            PatchingCheckSummary(
                run,
                chosen,
                len({(check.layer, check.head) for check in members}),
                len({check.pair for check in members}),
                float(np.corrcoef(estimated, actual)[0, 1]) if varies else float("nan"),
                float(estimated.mean()),
                float(actual.mean()),
            )
        )
    return summaries
