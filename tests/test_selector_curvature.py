import numpy as np

import geomselect.selector as selector_module
from geomselect.hyperbolic import poincare_distance_matrix
from geomselect.metrics import pairwise_euclidean
from geomselect.result import GeometryCandidate
from geomselect.selector import select_geometry
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


def make_poincare_test_data(n=80, kappa=2.5, max_norm=0.9, seed=0):
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


def test_candidate_table_contains_curvature_columns():
    D, _ = make_euclidean_test_data(n=60, dim=2, seed=1)

    result = select_geometry(
        D,
        d=2,
        hyper_grid_num=15,
        sphere_grid_num=15,
        n_refine=1,
        refine_num=9,
    )

    assert "signed_curvature" in result.candidate_table.columns
    assert "curvature_signal" in result.candidate_table.columns
    assert "curvature_regime" in result.candidate_table.columns


def test_euclidean_candidate_has_flat_curvature_regime():
    D, _ = make_euclidean_test_data(n=60, dim=2, seed=2)

    result = select_geometry(
        D,
        d=2,
        geometries=("euclidean",),
    )

    row = result.candidate_table.iloc[0]

    assert result.selected_geometry == "euclidean"
    assert row["geometry"] == "euclidean"
    assert row["signed_curvature"] == 0.0
    assert row["curvature_signal"] == 0.0
    assert row["curvature_regime"] == "flat"


def test_clean_spherical_data_has_curved_regime():
    D, _, _ = make_sphere_test_data(
        n=80,
        d=2,
        R=3.0,
        seed=3,
    )

    result = select_geometry(
        D,
        d=2,
        hyper_grid_num=21,
        sphere_grid_num=31,
        n_refine=3,
        refine_num=25,
    )

    spherical_row = result.candidate_table[
        result.candidate_table["geometry"] == "spherical"
    ].iloc[0]

    assert result.selected_geometry == "spherical"
    assert spherical_row["curvature_regime"] == "curved"
    assert spherical_row["curvature_signal"] > 0.35
    assert spherical_row["signed_curvature"] > 0.0


def test_clean_hyperbolic_data_has_curved_regime():
    D, _, _ = make_poincare_test_data(
        n=80,
        kappa=2.5,
        max_norm=0.9,
        seed=4,
    )

    result = select_geometry(
        D,
        d=2,
        hyper_grid_num=31,
        sphere_grid_num=21,
        n_refine=3,
        refine_num=25,
    )

    hyperbolic_row = result.candidate_table[
        result.candidate_table["geometry"] == "hyperbolic"
    ].iloc[0]

    assert result.selected_geometry == "hyperbolic"
    assert hyperbolic_row["curvature_regime"] == "curved"
    assert hyperbolic_row["curvature_signal"] > 0.35
    assert hyperbolic_row["signed_curvature"] < 0.0


def test_near_flat_curved_candidate_is_overridden_by_close_euclidean(monkeypatch):
    D, _ = make_euclidean_test_data(n=20, dim=2, seed=5)

    def fake_fit_euclidean(D, d=2, pairs=None):
        return GeometryCandidate(
            geometry="euclidean",
            stress=0.1005,
            embedding=np.zeros((D.shape[0], d)),
            parameter_name=None,
            parameter_value=np.nan,
            metadata={},
        )

    def fake_fit_spherical(D, d=2, **kwargs):
        return GeometryCandidate(
            geometry="spherical",
            stress=0.1000,
            embedding=np.zeros((D.shape[0], d + 1)),
            parameter_name="R",
            parameter_value=1000.0,
            metadata={
                "R_scaled": 1000.0,
                "scale_s": 1.0,
            },
        )

    monkeypatch.setattr(selector_module, "fit_euclidean", fake_fit_euclidean)
    monkeypatch.setattr(selector_module, "fit_spherical", fake_fit_spherical)

    result = select_geometry(
        D,
        d=2,
        geometries=("euclidean", "spherical"),
        euclidean_flat_close_ratio=1.10,
        euclidean_flat_close_abs=1e-3,
    )

    assert result.selected_geometry == "euclidean"
    assert result.metadata["best_geometry_by_stress"] == "spherical"
    assert result.metadata["used_near_flat_override"] is True
    assert result.metadata["selected_curvature_regime"] == "flat"