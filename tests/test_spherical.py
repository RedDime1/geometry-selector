import numpy as np

from geomselect.spherical import (
    fit_spherical,
    spherical_distance_matrix,
    spherical_stress, spherical_fixed_R_eigsh,
)


def make_sphere_test_data(n=80, d=2, R=3.0, seed=0, noise=0.0):
    rng = np.random.default_rng(seed)

    Z = rng.normal(size=(n, d + 1))
    U = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-15)

    D = spherical_distance_matrix(U, R=R)

    if noise > 0:
        positive = D[D > 0]
        scale = np.median(positive) if positive.size else 1.0

        E = rng.normal(scale=noise * scale, size=D.shape)
        E = 0.5 * (E + E.T)

        D = np.maximum(D + E, 0.0)
        np.fill_diagonal(D, 0.0)

    return D, U, R



def test_spherical_distance_matrix_is_symmetric():
    D, _, _ = make_sphere_test_data(
        n=30,
        d=2,
        R=3.0,
        seed=1,
    )

    assert D.shape == (30, 30)
    assert np.allclose(D, D.T)
    assert np.allclose(np.diag(D), 0.0)
    assert np.all(D >= 0.0)


def test_fit_spherical_on_clean_sphere_data():
    D, _, R_true = make_sphere_test_data(
        n=80,
        d=2,
        R=3.0,
        seed=2,
        noise=0.0,
    )

    candidate = fit_spherical(
        D,
        d=2,
        grid_num=31,
        n_refine=3,
        refine_num=25,
    )

    assert candidate.geometry == "spherical"
    assert candidate.parameter_name == "R"
    assert candidate.parameter_value > 0
    assert np.isfinite(candidate.stress)
    assert candidate.stress < 0.03
    assert abs(candidate.parameter_value - R_true) / R_true < 0.15
    assert candidate.embedding.shape == (80, 3)


def test_fit_spherical_on_noisy_sphere_data():
    D, _, _ = make_sphere_test_data(
        n=80,
        d=2,
        R=3.0,
        seed=3,
        noise=0.02,
    )

    candidate = fit_spherical(
        D,
        d=2,
        grid_num=31,
        n_refine=3,
        refine_num=25,
    )

    assert candidate.geometry == "spherical"
    assert candidate.parameter_name == "R"
    assert candidate.parameter_value > 0
    assert np.isfinite(candidate.stress)
    assert candidate.stress < 0.08


def test_spherical_stress_is_small_for_true_points():
    D, U, R = make_sphere_test_data(
        n=50,
        d=2,
        R=3.0,
        seed=4,
        noise=0.0,
    )

    value = spherical_stress(D, U, R=R)

    assert np.isfinite(value)
    assert value < 1e-10


