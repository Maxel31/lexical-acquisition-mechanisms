"""属性の方向（属性のプローブの重みが張る部分空間）と、比較の基準にする無作為な方向。"""

import numpy as np

from lexical_acquisition_mechanisms.logistic_probes import LogisticProbe


def attribute_basis(probe: LogisticProbe) -> np.ndarray:
    """プローブの重みが張る部分空間の正規直交基底（形は（特徴の次元, 部分空間の次元））。

    特徴は標準化してから重みをかけるので、元の空間での重みは weights / scale。判定に効くのは
    クラスの間の重みの差だけなので、クラスで平均を引いてから基底を求める（カテゴリの6クラスなら
    最大5次元、品詞の2クラスなら1次元）。
    """
    raw = probe.weights / probe.scale
    centered = raw - raw.mean(axis=0, keepdims=True)
    _, values, vectors = np.linalg.svd(centered, full_matrices=False)
    if not values.size or values.max() == 0:
        return np.zeros((raw.shape[1], 0))
    rank = int((values > values.max() * 1e-6).sum())
    return vectors[:rank].T


def random_basis(dimension: int, rank: int, random_numbers: np.random.Generator) -> np.ndarray:
    """無作為な部分空間の正規直交基底（形は（dimension, rank））。"""
    basis, _ = np.linalg.qr(random_numbers.normal(size=(dimension, rank)))
    return basis
