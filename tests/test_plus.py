import numpy as np

from geomselect.hyperbolic import fit_hyperbolic, poincare_distance_matrix
from geomselect.metrics import pairwise_euclidean
from geomselect.selector import select_geometry
from geomselect.spherical import fit_spherical, spherical_distance_matrix


def make_poincare_test_data(n=70, kappa=2.5, max_norm=0.86, seed=0):
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


def make_sphere_test_data(n=70, d=2, R=3.0, seed=0):
    rng = np.random.default_rng(seed)

    Z = rng.normal(size=(n, d + 1))
    U = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-15)

    D = spherical_distance_matrix(U, R=R)

    return D, U, R


def make_euclidean_test_data(n=70, dim=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, dim))
    D = pairwise_euclidean(X)

    return D, X


def test_hyperbolic_plus_does_not_increase_stress_with_rollback():
    D, _, _ = make_poincare_test_data(
        n=70,
        kappa=2.5,
        max_norm=0.86,
        seed=1,
    )

    base = fit_hyperbolic(
        D,
        d=2,
        grid_num=25,
        n_refine=2,
        refine_num=15,
        do_plus=False,
    )

    refined = fit_hyperbolic(
        D,
        d=2,
        grid_num=25,
        n_refine=2,
        refine_num=15,
        do_plus=True,
        plus_maxiter=20,
        rollback_plus=True,
        random_state=1,
    )

    assert np.isfinite(refined.stress)
    assert refined.stress <= base.stress + 1e-10
    assert "stress_before_plus" in refined.metadata
    assert "stress_after_plus" in refined.metadata
    assert "used_plus" in refined.metadata


def test_spherical_plus_does_not_increase_stress_with_rollback():
    D, _, _ = make_sphere_test_data(
        n=70,
        d=2,
        R=3.0,
        seed=2,
    )

    base = fit_spherical(
        D,
        d=2,
        grid_num=25,
        n_refine=2,
        refine_num=15,
        do_plus=False,
    )

    refined = fit_spherical(
        D,
        d=2,
        grid_num=25,
        n_refine=2,
        refine_num=15,
        do_plus=True,
        plus_maxiter=20,
        rollback_plus=True,
        random_state=2,
    )

    assert np.isfinite(refined.stress)
    assert refined.stress <= base.stress + 1e-10
    assert "stress_before_plus" in refined.metadata
    assert "stress_after_plus" in refined.metadata
    assert "used_plus" in refined.metadata


def test_select_geometry_with_plus_runs_on_hyperbolic_data():
    D, _, _ = make_poincare_test_data(
        n=70,
        kappa=2.5,
        max_norm=0.86,
        seed=3,
    )

    result = select_geometry(
        D,
        d=2,
        do_plus=True,
        hyper_grid_num=25,
        sphere_grid_num=21,
        n_refine=2,
        refine_num=15,
        plus_maxiter_hyper=20,
        plus_maxiter_sphere=20,
        random_state=3,
    )

    assert result.selected_geometry == "hyperbolic"
    assert result.metadata["do_plus"] is True
    assert "used_plus" in result.candidate_table.columns
    assert np.isfinite(result.selected.stress)


def test_select_geometry_with_plus_runs_on_spherical_data():
    D, _, _ = make_sphere_test_data(
        n=70,
        d=2,
        R=3.0,
        seed=4,
    )

    result = select_geometry(
        D,
        d=2,
        do_plus=True,
        hyper_grid_num=21,
        sphere_grid_num=25,
        n_refine=2,
        refine_num=15,
        plus_maxiter_hyper=20,
        plus_maxiter_sphere=20,
        random_state=4,
    )

    assert result.selected_geometry == "spherical"
    assert result.metadata["do_plus"] is True
    assert "used_plus" in result.candidate_table.columns
    assert np.isfinite(result.selected.stress)


def test_select_geometry_with_plus_keeps_euclidean_choice():
    D, _ = make_euclidean_test_data(
        n=70,
        dim=2,
        seed=5,
    )

    result = select_geometry(
        D,
        d=2,
        do_plus=True,
        hyper_grid_num=15,
        sphere_grid_num=15,
        n_refine=1,
        refine_num=9,
        plus_maxiter_hyper=10,
        plus_maxiter_sphere=10,
        random_state=5,
    )

    assert result.selected_geometry == "euclidean"
    assert result.metadata["do_plus"] is True
    assert np.isfinite(result.selected.stress)