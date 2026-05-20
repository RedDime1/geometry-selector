from __future__ import annotations

import numpy as np

from geomselect import (
    pairwise_euclidean,
    poincare_distance_matrix,
    spherical_distance_matrix,
    select_geometry,
)


def make_euclidean_data(n: int = 80, dim: int = 2, seed: int = 1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, dim))
    D = pairwise_euclidean(X)
    return D


def make_spherical_data(n: int = 80, R: float = 3.0, seed: int = 2):
    rng = np.random.default_rng(seed)

    Z = rng.normal(size=(n, 3))
    U = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-15)

    D = spherical_distance_matrix(U, R=R)
    return D


def make_hyperbolic_data(n: int = 80, kappa: float = 2.5, max_norm: float = 0.9, seed: int = 3):
    rng = np.random.default_rng(seed)

    angles = rng.uniform(0.0, 2.0 * np.pi, size=n)
    radii = max_norm * np.sqrt(rng.uniform(0.0, 1.0, size=n))

    Y = np.column_stack(
        [
            radii * np.cos(angles),
            radii * np.sin(angles),
        ]
    )

    D = poincare_distance_matrix(Y, kappa=kappa)
    return D


def run_case(name: str, D: np.ndarray, expected: str):
    result = select_geometry(
        D,
        d=2,
        do_plus=True,
        hyper_grid_num=31,
        sphere_grid_num=31,
        n_refine=3,
        refine_num=25,
        random_state=42,
    )

    print()
    print("=" * 80)
    print(name)
    print("expected:", expected)
    print("selected:", result.selected_geometry)
    print("recommendation:", result.recommendation)
    print()
    print(result.candidate_table.to_string(index=False))

    return result


if __name__ == "__main__":
    cases = [
        ("euclidean_clean", make_euclidean_data(), "euclidean"),
        ("spherical_clean", make_spherical_data(), "spherical"),
        ("hyperbolic_clean", make_hyperbolic_data(), "hyperbolic"),
    ]

    for name, D, expected in cases:
        run_case(name, D, expected)