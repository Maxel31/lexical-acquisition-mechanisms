"""ブートストラップによる区間と p 値と、多重比較の補正。

単位が入れ子になったデータ（例: ターゲット語の中の最小対）では、上の単位を復元抽出し、選んだ
上の単位の中で下の単位を復元抽出する（2段のブートストラップ）。
"""

import math
from collections.abc import Sequence

import numpy as np

BOOTSTRAP_SAMPLES = 1000


def bootstrap_weights(
    groups: Sequence[str], samples: int, random_numbers: np.random.Generator
) -> np.ndarray:
    """2段の復元抽出で、各単位が選ばれた回数（形は（再抽出, 単位））。

    groups は、単位ごとの上の単位の名前。
    """
    names = sorted(set(groups))
    members = {
        name: np.array([index for index, group in enumerate(groups) if group == name])
        for name in names
    }
    weights = np.zeros((samples, len(groups)))
    for sample in range(samples):
        for name in random_numbers.choice(names, len(names)):
            chosen = random_numbers.choice(members[name], len(members[name]))
            np.add.at(weights[sample], chosen, 1)
    return weights


def weighted_means(
    weights: np.ndarray, values: np.ndarray, present: np.ndarray | None = None
) -> np.ndarray:
    """再抽出ごとの、単位の値の平均（形は（再抽出, 値の次元））。

    present（単位ごとの真偽）があれば、値がある単位だけで平均する（なければ nan）。
    """
    if present is not None:
        weights = weights * present
        values = np.where(present[:, None], values, 0.0)
    totals = weights.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (weights @ values) / totals


def normal_p_values(means: np.ndarray, standard_errors: np.ndarray) -> np.ndarray:
    """平均が 0 という仮説の、標準誤差による正規近似の両側の p 値。

    再抽出の割合から出す p 値は、1,000 回の再抽出では 0.002 より小さくならず、数百以上の比較の補正に
    使えないため、こちらを使う。
    """
    means = np.asarray(means, dtype=float)
    errors = np.asarray(standard_errors, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        scores = np.where(errors > 0, np.abs(means) / errors, np.where(means == 0, 0.0, np.inf))
    return np.array([math.erfc(score / math.sqrt(2)) for score in scores.ravel()]).reshape(
        scores.shape
    )


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg 法で補正した q 値（q が基準より小さいものを選ぶと、FDR が基準以下）。"""
    values = np.nan_to_num(np.asarray(p_values, dtype=float), nan=1.0)
    order = np.argsort(values, kind="stable")
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    q_values = np.empty_like(values)
    q_values[order] = np.minimum(adjusted, 1.0)
    return q_values
