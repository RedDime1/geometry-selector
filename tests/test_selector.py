import numpy as np
import pytest

from geomselect.metrics import pairwise_euclidean
from geomselect.selector import select_geometry
from geomselect.hyperbolic import poincare_distance_matrix
from geomselect.spherical import spherical_distance_matrix


def make_euclidean_test_data(n=70, dim=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, dim))
    D = pairwise_euclidean(X)
    return D, X


def make_sphere_test_data(n=80, d=2, R=3.0, seed=0):
    rng = np.random.default_rng(seed)

    Z = rng.normal(size=(n, d + 1))
    U = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-15)

    D = spherical_distance_matrix(U, R=R)

    return D, U, R


def make_poincare_test_data(n=80, kappa=2.5, max_norm=0.82, seed=0):
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


def test_select_geometry_on_clean_euclidean_data():
    D, _ = make_euclidean_test_data(n=70, dim=2, seed=1)

    result = select_geometry(
        D,
        d=2,
        hyper_grid_num=21,
        sphere_grid_num=21,
        n_refine=2,
        refine_num=15,
    )

    assert result.selected_geometry == "euclidean"
    assert result.selected.geometry == "euclidean"
    assert result.selected.selected is True
    assert np.isfinite(result.selected.stress)
    assert result.selected.stress < 1e-10
    assert len(result.candidates) == 3
    assert set(result.candidate_table["geometry"]) == {
        "euclidean",
        "hyperbolic",
        "spherical",
    }


def test_select_geometry_on_clean_spherical_data():
    D, _, _ = make_sphere_test_data(
        n=80,
        d=2,
        R=3.0,
        seed=2,
    )

    result = select_geometry(
        D,
        d=2,
        hyper_grid_num=21,
        sphere_grid_num=31,
        n_refine=3,
        refine_num=25,
    )

    assert result.selected_geometry == "spherical"
    assert result.selected.geometry == "spherical"
    assert result.selected.parameter_name == "R"
    assert result.selected.parameter_value > 0
    assert np.isfinite(result.selected.stress)
    assert result.selected.stress < 1e-4


def test_select_geometry_on_clean_hyperbolic_data():
    D, _, _ = make_poincare_test_data(
        n=80,
        kappa=2.5,
        max_norm=0.86,
        seed=3,
    )

    result = select_geometry(
        D,
        d=2,
        hyper_grid_num=31,
        sphere_grid_num=21,
        n_refine=3,
        refine_num=25,
    )

    assert result.selected_geometry == "hyperbolic"
    assert result.selected.geometry == "hyperbolic"
    assert result.selected.parameter_name == "kappa"
    assert result.selected.parameter_value > 0
    assert np.isfinite(result.selected.stress)
    assert result.selected.stress < 0.1


def test_select_geometry_with_pair_sample():
    D, _ = make_euclidean_test_data(n=90, dim=2, seed=4)

    result = select_geometry(
        D,
        d=2,
        pair_sample=500,
        random_state=123,
        hyper_grid_num=15,
        sphere_grid_num=15,
        n_refine=1,
        refine_num=9,
    )

    assert result.metadata["used_pair_sample"] is True
    assert result.metadata["pair_sample"] == 500
    assert result.candidate_table.shape[0] == 3
    assert result.candidate_table.iloc[0]["selected"] is True or bool(result.candidate_table.iloc[0]["selected"])


def test_select_geometry_can_run_single_geometry():
    D, _ = make_euclidean_test_data(n=50, dim=2, seed=5)

    result = select_geometry(
        D,
        d=2,
        geometries=("euclidean",),
    )

    assert result.selected_geometry == "euclidean"
    assert len(result.candidates) == 1
    assert result.metadata["second_geometry"] is None
    assert np.isnan(result.metadata["second_stress"])


def test_select_geometry_rejects_unknown_geometry():
    D, _ = make_euclidean_test_data(n=40, dim=2, seed=6)

    with pytest.raises(NotImplementedError):
        select_geometry(
            D,
            d=2,
            geometries=("euclidean", "unknown"),
        )


def test_select_geometry_rejects_duplicate_geometries():
    D, _ = make_euclidean_test_data(n=40, dim=2, seed=7)

    with pytest.raises(ValueError):
        select_geometry(
            D,
            d=2,
            geometries=("euclidean", "euclidean"),
        )