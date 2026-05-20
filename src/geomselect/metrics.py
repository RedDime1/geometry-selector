from __future__ import annotations

import numpy as np

from geomselect.preprocessing import check_distance_matrix


def pairwise_euclidean(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("Point matrix X must be two-dimensional.")

    sq_norms = np.sum(X * X, axis=1)
    D2 = sq_norms[:, None] + sq_norms[None, :] - 2.0 * X @ X.T
    D2 = np.maximum(D2, 0.0)

    D = np.sqrt(D2)
    np.fill_diagonal(D, 0.0)
    return D


def stress1(
    D_true: np.ndarray,
    D_pred: np.ndarray,
    *,
    eps: float = 1e-12,
) -> float:
    D_true = check_distance_matrix(D_true)
    D_pred = check_distance_matrix(D_pred)

    if D_true.shape != D_pred.shape:
        raise ValueError("Distance matrices must have the same shape.")

    rows, cols = np.triu_indices(D_true.shape[0], k=1)

    numerator = np.sum((D_true[rows, cols] - D_pred[rows, cols]) ** 2)
    denominator = np.sum(D_true[rows, cols] ** 2)

    if denominator <= eps:
        raise ValueError("True distance matrix has near-zero norm.")

    return float(np.sqrt(numerator / denominator))


def stress1_from_vectors(
    d_true: np.ndarray,
    d_pred: np.ndarray,
    *,
    eps: float = 1e-12,
) -> float:
    d_true = np.asarray(d_true, dtype=float)
    d_pred = np.asarray(d_pred, dtype=float)

    if d_true.shape != d_pred.shape:
        raise ValueError("Distance vectors must have the same shape.")

    numerator = np.sum((d_true - d_pred) ** 2)
    denominator = np.sum(d_true ** 2)

    if denominator <= eps:
        raise ValueError("True distance vector has near-zero norm.")

    return float(np.sqrt(numerator / denominator))