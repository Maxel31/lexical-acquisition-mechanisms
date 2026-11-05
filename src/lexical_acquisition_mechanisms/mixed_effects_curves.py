"""水準ごとの値（曲線）を、交差したランダム切片の混合効果モデルで求める。

    値 ~ 0 + 水準 + (1 | 単位1) + (1 | 単位2) + ...

固定効果は水準ごとの値（切片なし）。分散（ランダム切片ごとと残差）は REML で求める。V⁻¹ と log|V| は
Woodbury の式と行列式の補題で、ランダム切片の水準の数の大きさの行列だけから求めるので、行の数が
多くても速い（未知語を再抽出して、何度も解き直すため）。
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import linalg, optimize

# REML で探す分散の範囲（値の分散を単位にした対数。下限は、その分散が実際上 0 のとき）
LOG_VARIANCE_BOUNDS = (math.log(1e-8), math.log(10.0))


@dataclass(frozen=True)
class LevelFit:
    """水準ごとの値と標準誤差（行のない水準は nan）と、当てはめた分散。"""

    values: np.ndarray
    standard_errors: np.ndarray
    # ランダム切片ごとの分散（外したランダム切片は 0）と、残差の分散。混合効果モデルを
    # 当てはめられず、水準ごとの平均にしたときは None
    intercept_variances: tuple[float, ...] | None
    residual_variance: float | None


def fit_levels(
    levels: np.ndarray,
    targets: np.ndarray,
    intercepts: Sequence[np.ndarray],
    number_of_levels: int,
) -> LevelFit:
    """REML で当てはめた、水準ごとの固定効果。

    levels は行ごとの水準の番号、intercepts はランダム切片ごとの、行ごとの単位の番号。単位が1つしか
    ないランダム切片は外す。当てはめられないとき（行が水準の数より多くない、値がばらつかない）は、
    水準ごとの平均と、その標準誤差にする。
    """
    used = [index for index, codes in enumerate(intercepts) if len(np.unique(codes)) > 1]
    present = np.unique(levels)
    if not used or len(targets) <= len(present) or np.var(targets) == 0:
        return _level_means(levels, targets, number_of_levels)
    products = _Products(levels, targets, [intercepts[index] for index in used], number_of_levels)
    scale = float(np.var(targets))

    def objective(logs: np.ndarray) -> float:
        variances = np.exp(logs) * scale
        solution = products.solve(variances[:-1], variances[-1])
        if solution is None:
            return math.inf
        return solution.log_determinant + solution.log_information + solution.quadratic

    start = np.full(len(used) + 1, -math.log(len(used) + 1))
    result = optimize.minimize(
        objective, start, method="L-BFGS-B", bounds=[LOG_VARIANCE_BOUNDS] * len(start)
    )
    variances = np.exp(result.x) * scale
    solution = products.solve(variances[:-1], variances[-1])
    if not np.isfinite(result.fun) or solution is None:
        return _level_means(levels, targets, number_of_levels)
    errors = np.full(number_of_levels, np.nan)
    errors[present] = np.sqrt(np.diag(np.linalg.inv(solution.information)))
    intercept_variances = [0.0] * len(intercepts)
    for index, variance in zip(used, variances[:-1], strict=True):
        intercept_variances[index] = float(variance)
    return LevelFit(solution.values, errors, tuple(intercept_variances), float(variances[-1]))


def generalized_least_squares(
    levels: np.ndarray,
    targets: np.ndarray,
    intercepts: Sequence[np.ndarray],
    intercept_variances: Sequence[float],
    residual_variance: float,
    number_of_levels: int,
) -> np.ndarray:
    """分散を固定した一般化最小二乗での、水準ごとの固定効果（行のない水準は nan）。

    分散が 0 のランダム切片は外す。
    """
    used = [
        (codes, variance)
        for codes, variance in zip(intercepts, intercept_variances, strict=True)
        if variance > 0
    ]
    if not used:
        return _level_means(levels, targets, number_of_levels).values
    products = _Products(levels, targets, [codes for codes, _ in used], number_of_levels)
    solution = products.solve(np.array([variance for _, variance in used]), residual_variance)
    if solution is None:
        return _level_means(levels, targets, number_of_levels).values
    return solution.values


@dataclass(frozen=True)
class _Solution:
    # 水準ごとの固定効果（行のない水準は nan）と、行のある水準での XᵀV⁻¹X
    values: np.ndarray
    information: np.ndarray
    # REML の目的関数の3つの項: log|V|・log|XᵀV⁻¹X|・(y − Xβ)ᵀV⁻¹(y − Xβ)
    log_determinant: float
    log_information: float
    quadratic: float


class _Products:
    """分散によらない積（ZᵀZ・ZᵀX・Zᵀy・XᵀX・Xᵀy・yᵀy）。Z はランダム切片の、X は水準の指示行列。"""

    def __init__(
        self,
        levels: np.ndarray,
        targets: np.ndarray,
        intercepts: Sequence[np.ndarray],
        number_of_levels: int,
    ) -> None:
        self._present = np.bincount(levels, minlength=number_of_levels) > 0
        index = np.cumsum(self._present) - 1
        rows = index[levels]
        width = int(self._present.sum())
        sizes = [int(codes.max()) + 1 for codes in intercepts]
        offsets = np.cumsum([0, *sizes])
        columns = [codes + offset for codes, offset in zip(intercepts, offsets[:-1], strict=True)]
        self._sizes = sizes
        self._number_of_rows = len(targets)
        self._number_of_levels = number_of_levels
        self.intercept_products = np.zeros((offsets[-1], offsets[-1]))
        self.cross_products = np.zeros((offsets[-1], width + 1))
        for first in columns:
            for second in columns:
                np.add.at(self.intercept_products, (first, second), 1.0)
            np.add.at(self.cross_products, (first, rows), 1.0)
            np.add.at(self.cross_products[:, -1], first, targets)
        self.level_counts = np.bincount(rows, minlength=width).astype(float)
        self.level_sums = np.bincount(rows, weights=targets, minlength=width)
        self.sum_of_squares = float(targets @ targets)

    def solve(self, intercept_variances: np.ndarray, residual_variance: float) -> _Solution | None:
        """分散を与えたときの解。M = σ_e² D⁻¹ + ZᵀZ が解けなければ None。"""
        inverse = np.repeat(residual_variance / np.asarray(intercept_variances), self._sizes)
        try:
            factor = linalg.cho_factor(self.intercept_products + np.diag(inverse))
        except linalg.LinAlgError:
            return None
        solved = linalg.cho_solve(factor, self.cross_products)
        level_part = self.cross_products[:, :-1]
        # σ_e² XᵀV⁻¹X と σ_e² XᵀV⁻¹y
        information = np.diag(self.level_counts) - level_part.T @ solved[:, :-1]
        right = self.level_sums - level_part.T @ solved[:, -1]
        try:
            coefficients = np.linalg.solve(information, right)
        except np.linalg.LinAlgError:
            return None
        # σ_e² (y − Xβ)ᵀV⁻¹(y − Xβ) = rᵀr − (Zᵀr)ᵀ M⁻¹ Zᵀr
        residual_sum = (
            self.sum_of_squares
            - 2 * coefficients @ self.level_sums
            + coefficients @ (self.level_counts * coefficients)
        )
        intercept_residuals = self.cross_products[:, -1] - level_part @ coefficients
        quadratic = residual_sum - intercept_residuals @ linalg.cho_solve(
            factor, intercept_residuals
        )
        size = len(inverse)
        # log|V| = (n − q) log σ_e² + log|D| + log|M|（行列式の補題）
        log_determinant = (
            (self._number_of_rows - size) * math.log(residual_variance)
            + float(np.sum(np.repeat(np.log(intercept_variances), self._sizes)))
            + 2 * float(np.sum(np.log(np.diag(factor[0]))))
        )
        log_information = float(np.linalg.slogdet(information)[1]) - len(right) * math.log(
            residual_variance
        )
        values = np.full(self._number_of_levels, np.nan)
        values[self._present] = coefficients
        return _Solution(
            values,
            information / residual_variance,
            log_determinant,
            log_information,
            float(quadratic / residual_variance),
        )


def _level_means(levels: np.ndarray, targets: np.ndarray, number_of_levels: int) -> LevelFit:
    values = np.full(number_of_levels, np.nan)
    errors = np.full(number_of_levels, np.nan)
    for level in np.unique(levels):
        chosen = targets[levels == level]
        values[level] = chosen.mean()
        if len(chosen) > 1:
            errors[level] = chosen.std(ddof=1) / math.sqrt(len(chosen))
    return LevelFit(values, errors, None, None)
