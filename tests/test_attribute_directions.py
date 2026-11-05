"""属性の方向と、無作為な方向のテスト。"""

import numpy as np

from lexical_acquisition_mechanisms.attribute_directions import attribute_basis, random_basis
from lexical_acquisition_mechanisms.logistic_probes import LogisticProbe


def test_attribute_basis_spans_the_differences_between_classes() -> None:
    weights = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    scale = np.array([1.0, 2.0, 1.0, 1.0])
    probe = LogisticProbe(weights, np.zeros(3), np.zeros(4), scale)

    basis = attribute_basis(probe)

    # 3クラスなので2次元。標準化を戻した重みの、クラスの間の差を含む
    assert basis.shape == (4, 2)
    np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=1e-10)
    differences = (weights / scale)[:2] - (weights / scale)[2]
    np.testing.assert_allclose(differences @ basis @ basis.T, differences, atol=1e-10)


def test_two_classes_give_one_direction() -> None:
    probe = LogisticProbe(np.array([[1.0, 1.0], [-1.0, 1.0]]), np.zeros(2), np.zeros(2), np.ones(2))

    basis = attribute_basis(probe)

    assert basis.shape == (2, 1)
    np.testing.assert_allclose(np.abs(basis[:, 0]), [1.0, 0.0], atol=1e-10)


def test_random_basis_is_orthonormal() -> None:
    basis = random_basis(8, 3, np.random.default_rng(0))

    assert basis.shape == (8, 3)
    np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-10)
