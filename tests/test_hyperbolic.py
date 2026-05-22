import numpy as np

from geomselect.hyperbolic import (
    fit_hyperbolic,
    hyperbolic_stress,
    poincare_distance_matrix,
    hydra_fixed_kappa_eigsh
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


def test_hyperbolic_stress_with_pairs():
    rng = np.random.default_rng(12)

    n = 50
    d = 2
    kappa = 2.5

    directions = rng.normal(size=(n, d))
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-15)

    radii = rng.uniform(0.05, 0.70, size=(n, 1))
    Y = directions * radii

    D = poincare_distance_matrix(Y, kappa=kappa)

    rows = np.array([0, 1, 2, 3, 4, 10, 11])
    cols = np.array([5, 6, 7, 8, 9, 20, 21])

    value = hyperbolic_stress(
        D,
        Y,
        kappa=kappa,
        pairs=(rows, cols),
    )

    assert np.isfinite(value)
    assert value < 1e-10

def test_hydra_fixed_kappa_eigsh_basic():
    rng = np.random.default_rng(21)

    n = 60
    d = 2
    kappa = 2.5

    directions = rng.normal(size=(n, d))
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-15)

    radii = rng.uniform(0.05, 0.65, size=(n, 1))
    Y = directions * radii

    D = poincare_distance_matrix(Y, kappa=kappa)

    Y_hat, X, lam_top, lam_bottom, A = hydra_fixed_kappa_eigsh(
        D,
        d=d,
        kappa=kappa,
    )

    assert Y_hat.shape == (n, d)
    assert X.shape == (n, d + 1)
    assert A.shape == (n, n)
    assert lam_bottom.shape == (d,)

    assert np.all(np.isfinite(Y_hat))
    assert np.all(np.isfinite(X))
    assert np.isfinite(lam_top)
    assert np.all(np.isfinite(lam_bottom))
    assert np.all(np.isfinite(A))

    assert np.all(np.linalg.norm(Y_hat, axis=1) < 1.0)

    value = hyperbolic_stress(D, Y_hat, kappa=kappa)
    assert np.isfinite(value)
    assert value < 0.1