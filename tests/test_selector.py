import numpy as np

from geomselect.hyperbolic import poincare_distance_matrix
from geomselect.metrics import pairwise_euclidean
from geomselect.selector import select_geometry
from geomselect.spherical import spherical_distance_matrix


def make_sphere_test_data(n=70, R=3.0, seed=0):
    rng = np.random.default_rng(seed)

    Z = rng.normal(size=(n, 3))
    U = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-15)

    return spherical_distance_matrix(U, R=R)


def make_hyperbolic_test_data(n=70, kappa=2.5, max_norm=0.85, seed=0):
    rng = np.random.default_rng(seed)

    angles = rng.uniform(0.0, 2.0 * np.pi, size=n)
    radii = max_norm * np.sqrt(rng.uniform(0.0, 1.0, size=n))

    Y = np.column_stack(
        [
            radii * np.cos(angles),
            radii * np.sin(angles),
        ]
    )

    return poincare_distance_matrix(Y, kappa=kappa)


def test_select_geometry_detects_euclidean_clean_data():
    rng = np.random.default_rng(0)

    X = rng.normal(size=(60, 2))
    D = pairwise_euclidean(X)

    result = select_geometry(D, d=2)

    assert result.selected_geometry == "euclidean"
    assert result.selected.geometry == "euclidean"
    assert result.selected.stress < 1e-10
    assert result.selected.embedding.shape == (60, 2)
    assert result.candidate_table.shape[0] == 3
    assert bool(result.candidate_table.loc[0, "selected"])


def test_select_geometry_with_pair_sample():
    rng = np.random.default_rng(1)

    X = rng.normal(size=(80, 3))
    D = pairwise_euclidean(X)

    result = select_geometry(
        D,
        d=3,
        pair_sample=400,
        random_state=42,
    )

    assert result.selected_geometry == "euclidean"
    assert np.isfinite(result.selected.stress)
    assert result.selected.stress < 1e-10
    assert result.metadata["used_pair_sample"] is True


def test_selector_can_choose_hyperbolic_candidate():
    D = make_hyperbolic_test_data(
        n=80,
        kappa=2.5,
        max_norm=0.9,
        seed=10,
    )

    result = select_geometry(
        D,
        d=2,
        geometries=("euclidean", "hyperbolic", "spherical"),
    )

    assert result.selected_geometry == "hyperbolic"
    assert result.selected.stress < 0.08


def test_selector_can_choose_spherical_candidate():
    D = make_sphere_test_data(
        n=80,
        R=3.0,
        seed=11,
    )

    result = select_geometry(
        D,
        d=2,
        geometries=("euclidean", "hyperbolic", "spherical"),
    )

    assert result.selected_geometry == "spherical"
    assert result.selected.stress < 0.05