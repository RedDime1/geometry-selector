import numpy as np
import pytest

from geomselect.metrics import pairwise_euclidean, stress1, stress1_from_vectors


def test_pairwise_euclidean_simple():
    X = np.array([
        [0.0, 0.0],
        [3.0, 4.0],
    ])

    D = pairwise_euclidean(X)

    assert D.shape == (2, 2)
    assert np.allclose(D, [[0.0, 5.0], [5.0, 0.0]])


def test_stress1_zero_for_equal_matrices():
    D = np.array([
        [0.0, 1.0, 2.0],
        [1.0, 0.0, 3.0],
        [2.0, 3.0, 0.0],
    ])

    assert stress1(D, D) == pytest.approx(0.0)


def test_stress1_positive_for_different_matrices():
    D_true = np.array([
        [0.0, 1.0],
        [1.0, 0.0],
    ])
    D_pred = np.array([
        [0.0, 2.0],
        [2.0, 0.0],
    ])

    assert stress1(D_true, D_pred) == pytest.approx(1.0)


def test_stress1_from_vectors_zero():
    d = np.array([1.0, 2.0, 3.0])

    assert stress1_from_vectors(d, d) == pytest.approx(0.0)