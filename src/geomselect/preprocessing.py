from __future__ import annotations

import numpy as np


def check_distance_matrix(D: np.ndarray, *, atol: float = 1e-8) -> np.ndarray:
    """Validate and return a distance matrix as float ndarray."""
    D = np.asarray(D, dtype=float)

    if D.ndim != 2:
        raise ValueError("Distance matrix must be two-dimensional.")

    if D.shape[0] != D.shape[1]:
        raise ValueError("Distance matrix must be square.")

    if not np.all(np.isfinite(D)):
        raise ValueError("Distance matrix must contain only finite values.")

    if np.any(D < -atol):
        raise ValueError("Distance matrix must be non-negative.")

    if not np.allclose(D, D.T, atol=atol):
        raise ValueError("Distance matrix must be symmetric.")

    if not np.allclose(np.diag(D), 0.0, atol=atol):
        raise ValueError("Distance matrix diagonal must be zero.")

    D = D.copy()
    D[D < 0] = 0.0
    np.fill_diagonal(D, 0.0)
    return D


def normalize_distance_matrix(
    D: np.ndarray,
    method: str | None = "median",
    *,
    eps: float = 1e-12,
) -> tuple[np.ndarray, float]:
    """
    Normalize a distance matrix and return (normalized_matrix, scale).

    Supported methods:
    - None or "none": no normalization
    - "median": divide by median of positive distances
    - "mean": divide by mean of positive distances
    - "max": divide by maximum distance
    """
    D = check_distance_matrix(D)

    if method is None or method == "none":
        return D.copy(), 1.0

    positive = D[D > eps]
    if positive.size == 0:
        raise ValueError("Distance matrix has no positive distances.")

    if method == "median":
        scale = float(np.median(positive))
    elif method == "mean":
        scale = float(np.mean(positive))
    elif method == "max":
        scale = float(np.max(positive))
    else:
        raise ValueError(f"Unknown normalization method: {method!r}")

    if scale <= eps:
        raise ValueError("Normalization scale is too small.")

    return D / scale, scale


def all_pairs(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return upper-triangular index pairs i < j."""
    if n < 2:
        raise ValueError("At least two objects are required.")
    return np.triu_indices(n, k=1)


def sample_pairs(
    n: int,
    max_pairs: int | None,
    *,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return either all pairs or a random subset of upper-triangular pairs."""
    rows, cols = all_pairs(n)
    total = len(rows)

    if max_pairs is None or max_pairs >= total:
        return rows, cols

    if max_pairs <= 0:
        raise ValueError("max_pairs must be positive.")

    rng = np.random.default_rng(random_state)
    idx = rng.choice(total, size=max_pairs, replace=False)
    return rows[idx], cols[idx]