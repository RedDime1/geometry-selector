import numpy as np

from geomselect.hyperbolic import (
    fit_hyperbolic,
    hyperbolic_stress,
    poincare_distance_matrix,
)


def make_poincare_test_data(n=50, kappa=2.5, max_norm=0.75, seed=0):
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

    return D, Y, kappa


def test_poincare_distance_matrix_is_valid():
    D, _, _ = make_poincare_test_data(n=30, kappa=2.5, seed=1)

    assert D.shape == (30, 30)
    assert np.all(np.isfinite(D))
    assert np.allclose(D, D.T)
    assert np.allclose(np.diag(D), 0.0)
    assert np.min(D) >= 0.0


def test_hyperbolic_stress_is_zero_on_original_points():
    D, Y, kappa = make_poincare_test_data(n=40, kappa=2.5, seed=2)

    s = hyperbolic_stress(D, Y, kappa=kappa)

    assert s < 1e-10


def test_fit_hyperbolic_on_clean_hyperbolic_data():
    D, _, _ = make_poincare_test_data(
        n=60,
        kappa=2.5,
        max_norm=0.8,
        seed=3,
    )

    candidate = fit_hyperbolic(
        D,
        d=2,
        grid_num=21,
        n_refine=2,
        refine_num=15,
    )

    assert candidate.geometry == "hyperbolic"
    assert candidate.parameter_name == "kappa"
    assert candidate.parameter_value > 0
    assert np.isfinite(candidate.stress)
    assert candidate.stress < 0.1
    assert candidate.embedding.shape == (60, 2)