"""ブートストラップと多重比較の補正のテスト。"""

import numpy as np
import pytest

from lexical_acquisition_mechanisms.bootstrap_statistics import (
    benjamini_hochberg,
    bootstrap_weights,
    normal_p_values,
    weighted_means,
)


def test_groups_are_resampled_and_then_their_members() -> None:
    weights = bootstrap_weights(["a", "a", "b", "b", "b"], 200, np.random.default_rng(0))

    assert weights.shape == (200, 5)
    # 上の単位を2回選ぶので、下の単位の回数の和は、選んだ上の単位の大きさの和になる
    totals = set(zip(weights[:, :2].sum(axis=1), weights[:, 2:].sum(axis=1), strict=True))
    assert totals == {(4, 0), (2, 3), (0, 6)}


def test_weighted_means_use_only_present_units() -> None:
    weights = np.array([[1.0, 2.0, 1.0]])
    values = np.array([[1.0], [4.0], [np.nan]])

    means = weighted_means(weights, values, np.array([True, True, False]))

    assert means[0, 0] == pytest.approx(3.0)


def test_p_values_and_false_discovery_rate() -> None:
    p_values = normal_p_values(np.array([1.96, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))

    np.testing.assert_allclose(p_values, [0.05, 1.0, 0.0], atol=1e-3)
    q_values = benjamini_hochberg(np.array([0.01, 0.04, 0.03, 0.005]))
    np.testing.assert_allclose(q_values, [0.02, 0.04, 0.04, 0.02])
