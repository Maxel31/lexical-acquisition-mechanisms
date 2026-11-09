"""ロジスティック回帰のプローブ（内部表現から語の性質を当てる分類器）。

特徴は学習データの平均と標準偏差で標準化し、L2 正則化した交差エントロピーを LBFGS で最小にする。
2クラス（例: 実在語か）と多クラス（例: カテゴリ）の両方に使う。
"""

from dataclasses import dataclass

import numpy as np
import torch

L2_WEIGHT = 1e-2
ITERATIONS = 200


@dataclass
class LogisticProbe:
    # 形は（クラスの数, 次元）と（クラスの数）。特徴を標準化する平均と標準偏差（次元）
    weights: np.ndarray
    biases: np.ndarray
    mean: np.ndarray
    scale: np.ndarray

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        """形は（例, クラスの数）。"""
        standardized = (features - self.mean) / self.scale
        logits = standardized @ self.weights.T + self.biases
        logits -= logits.max(axis=1, keepdims=True)
        exponentials = np.exp(logits)
        return exponentials / exponentials.sum(axis=1, keepdims=True)

    def accuracy(self, features: np.ndarray, labels: np.ndarray) -> float:
        return float((self.probabilities(features).argmax(axis=1) == labels).mean())


def train_logistic_probe(
    features: np.ndarray, labels: np.ndarray, number_of_classes: int, seed: int = 0
) -> LogisticProbe:
    """特徴（例 × 次元）とラベル（0 から数えるクラスの番号）から、プローブを学習する。"""
    mean = features.mean(axis=0)
    scale = features.std(axis=0) + 1e-6
    torch.manual_seed(seed)
    inputs = torch.from_numpy(((features - mean) / scale).astype(np.float32))
    targets = torch.from_numpy(labels.astype(np.int64))
    weights = torch.zeros((number_of_classes, features.shape[1]), requires_grad=True)
    biases = torch.zeros(number_of_classes, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [weights, biases], max_iter=ITERATIONS, line_search_fn="strong_wolfe"
    )

    def loss() -> torch.Tensor:
        optimizer.zero_grad()
        value = torch.nn.functional.cross_entropy(inputs @ weights.T + biases, targets)
        value = value + L2_WEIGHT * (weights**2).sum() / len(inputs)
        value.backward()
        return value

    optimizer.step(loss)
    return LogisticProbe(
        weights.detach().numpy(),
        biases.detach().numpy(),
        mean.astype(np.float32),
        scale.astype(np.float32),
    )
