import numpy as np
import pytest

from geomselect.metrics import pairwise_euclidean
from geomselect.selector import select_geometry
from geomselect.hyperbolic import poincare_distance_matrix


def test_select_geometry_detects_euclidean_clean_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 2))
    D = pairwise_euclidean(X)

    result = select_geometry(D, d=2)

    assert result.selected_geometry == "euclidean"
    assert result.selected.geometry == "euclidean"
    assert result.selected.stress < 1e-10
    assert result.selected.embedding.shape == (40, 2)
    assert result.candidate_table.shape[0] == 1
    assert result.candidate_table.loc[0, "selected"] is True or result.candidate_table.loc[0, "selected"] == True


def test_select_geometry_with_pair_sample():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 3))
    D = pairwise_euclidean(X)

    result = select_geometry(
        D,
        d=3,
        pair_sample=200,
        random_state=42,
    )

    assert result.selected_geometry == "euclidean"
    assert np.isfinite(result.selected.stress)
    assert result.selected.stress < 1e-10
    assert result.metadata["used_pair_sample"] is True

def test_selector_can_choose_hyperbolic_candidate():
    rng = np.random.default_rng(10)

    n = 60
    kappa = 2.5
    max_norm = 0.8

    angles = rng.uniform(0.0, 2.0 * np.pi, size=n)
    radii = max_norm * np.sqrt(rng.uniform(0.0, 1.0, size=n))

    Y = np.column_stack(
        [
            radii * np.cos(angles),
            radii * np.sin(angles),
        ]
    )

    D = poincare_distance_matrix(Y, kappa=kappa)

    result = select_geometry(
        D,
        d=2,
        geometries=("euclidean", "hyperbolic"),
    )

    assert result.selected_geometry == "hyperbolic"
    assert result.selected.stress < result.candidates[1].stress
    assert result.candidate_table.shape[0] == 2