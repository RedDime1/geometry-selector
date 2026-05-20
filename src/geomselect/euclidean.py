from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import eigsh

from geomselect.metrics import pairwise_euclidean, stress1, stress1_from_vectors
from geomselect.preprocessing import check_distance_matrix
from geomselect.result import GeometryCandidate


def classical_mds(
    D: np.ndarray,
    d: int = 2,
    *,
    full_threshold: int = 300,
    eig_tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    D = check_distance_matrix(D)

    if d < 1:
        raise ValueError("Embedding dimension d must be positive.")

    n = D.shape[0]
    k = min(d, n - 1)

    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ (D ** 2) @ H

    if n <= full_threshold or k >= n - 1:
        evals, evecs = np.linalg.eigh(B)
    else:
        evals, evecs = eigsh(B, k=k, which="LA", tol=eig_tol)

    order = np.argsort(evals)[::-1]
    evals = np.asarray(evals[order], dtype=float)
    evecs = np.asarray(evecs[:, order], dtype=float)

    positive = evals > eig_tol
    evals_pos = evals[positive][:d]
    evecs_pos = evecs[:, positive][:, :d]

    if evals_pos.size == 0:
        X = np.zeros((n, d), dtype=float)
        return X, evals_pos

    X = evecs_pos * np.sqrt(evals_pos)

    if X.shape[1] < d:
        X = np.pad(X, ((0, 0), (0, d - X.shape[1])))

    return X, evals_pos


def euclidean_stress(
    D: np.ndarray,
    X: np.ndarray,
    *,
    pairs: tuple[np.ndarray, np.ndarray] | None = None,
) -> float:
    D = check_distance_matrix(D)
    X = np.asarray(X, dtype=float)

    if pairs is None:
        D_hat = pairwise_euclidean(X)
        return stress1(D, D_hat)

    rows, cols = pairs
    d_true = D[rows, cols]
    d_pred = np.linalg.norm(X[rows] - X[cols], axis=1)

    return stress1_from_vectors(d_true, d_pred)


def fit_euclidean(
    D: np.ndarray,
    d: int = 2,
    *,
    pairs: tuple[np.ndarray, np.ndarray] | None = None,
) -> GeometryCandidate:
    D = check_distance_matrix(D)
    X, evals = classical_mds(D, d=d)
    stress = euclidean_stress(D, X, pairs=pairs)

    return GeometryCandidate(
        geometry="euclidean",
        stress=float(stress),
        embedding=X,
        parameter_name=None,
        parameter_value=None,
        metadata={
            "d": int(d),
            "eigenvalues": evals,
        },
    )