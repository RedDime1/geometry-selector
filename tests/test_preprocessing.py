import numpy as np
import pytest

from geomselect.preprocessing import check_distance_matrix, normalize_distance_matrix, all_pairs, sample_pairs


def test_check_distance_matrix_accepts_valid_matrix():
    D = np.array([
        [0.0, 1.0, 2.0],
        [1.0, 0.0, 3.0],
        [2.0, 3.0, 0.0],
    ])

    checked = check_distance_matrix(D)

    assert checked.shape == (3, 3)
    assert np.allclose(checked, checked.T)
    assert np.allclose(np.diag(checked), 0.0)


def test_check_distance_matrix_rejects_non_square():
    D = np.zeros((2, 3))

    with pytest.raises(ValueError):
        check_distance_matrix(D)


def test_check_distance_matrix_rejects_non_symmetric():
    D = np.array([
        [0.0, 1.0],
        [2.0, 0.0],
    ])

    with pytest.raises(ValueError):
        check_distance_matrix(D)


def test_normalize_distance_matrix_median():
    D = np.array([
        [0.0, 2.0, 4.0],
        [2.0, 0.0, 6.0],
        [4.0, 6.0, 0.0],
    ])

    D_norm, scale = normalize_distance_matrix(D, method="median")

    assert scale == 4.0
    assert np.allclose(D_norm, D / 4.0)


def test_all_pairs():
    rows, cols = all_pairs(4)

    assert list(zip(rows, cols)) == [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    ]


def test_sample_pairs_size():
    rows, cols = sample_pairs(10, max_pairs=7, random_state=42)

    assert len(rows) == 7
    assert len(cols) == 7
    assert np.all(rows < cols)