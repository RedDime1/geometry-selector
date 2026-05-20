import numpy as np
import pytest

from geomselect.metrics import pairwise_euclidean
from geomselect.selector import select_geometry


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
