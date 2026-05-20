from __future__ import annotations

import numpy as np
import pandas as pd

from geomselect.euclidean import fit_euclidean
from geomselect.hyperbolic import fit_hyperbolic
from geomselect.preprocessing import check_distance_matrix, sample_pairs
from geomselect.result import GeometryCandidate, GeometrySelectorConfig, SelectionResult
from geomselect.spherical import fit_spherical


def _make_pairs(
    n: int,
    pair_sample: int | None,
    random_state: int | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    if pair_sample is None:
        return None

    return sample_pairs(
        n,
        max_pairs=pair_sample,
        random_state=random_state,
    )


def _make_recommendation(
    selected: GeometryCandidate,
    candidates: list[GeometryCandidate],
    *,
    close_ratio: float = 1.05,
) -> str:
    if len(candidates) == 1:
        return f"Selected geometry: {selected.geometry}."

    ordered = sorted(candidates, key=lambda item: item.stress)
    best = ordered[0]
    second = ordered[1]

    ratio = second.stress / max(best.stress, 1e-15)

    if ratio <= close_ratio:
        return (
            f"Selected geometry: {best.geometry}. "
            f"The second candidate is close, so the result should be treated cautiously."
        )

    return f"Selected geometry: {best.geometry}. The stress gap is noticeable."


def _fit_candidate(
    geometry: str,
    D: np.ndarray,
    d: int,
    pairs: tuple[np.ndarray, np.ndarray] | None,
) -> GeometryCandidate:
    if geometry == "euclidean":
        return fit_euclidean(
            D,
            d=d,
            pairs=pairs,
        )

    if geometry == "hyperbolic":
        return fit_hyperbolic(
            D,
            d=d,
            pairs=pairs,
        )

    if geometry == "spherical":
        return fit_spherical(
            D,
            d=d,
            pairs=pairs,
        )

    raise NotImplementedError(f"Geometry {geometry!r} is not implemented yet.")


def select_geometry(
    D: np.ndarray,
    d: int = 2,
    *,
    geometries: tuple[str, ...] = ("euclidean", "hyperbolic", "spherical"),
    pair_sample: int | None = None,
    random_state: int | None = None,
) -> SelectionResult:
    D = check_distance_matrix(D)
    n = D.shape[0]

    config = GeometrySelectorConfig(
        d=d,
        geometries=tuple(geometries),
        pair_sample=pair_sample,
        random_state=random_state,
    )

    pairs = _make_pairs(
        n=n,
        pair_sample=pair_sample,
        random_state=random_state,
    )

    candidates: list[GeometryCandidate] = []

    for geometry in geometries:
        candidates.append(
            _fit_candidate(
                geometry,
                D,
                d,
                pairs,
            )
        )

    candidates = sorted(candidates, key=lambda item: item.stress)

    for candidate in candidates:
        candidate.selected = False

    selected = candidates[0]
    selected.selected = True

    table = pd.DataFrame(
        [candidate.as_dict() for candidate in candidates]
    )

    recommendation = _make_recommendation(selected, candidates)

    return SelectionResult(
        selected_geometry=selected.geometry,
        selected=selected,
        candidates=tuple(candidates),
        candidate_table=table,
        recommendation=recommendation,
        config=config,
        metadata={
            "n": int(n),
            "used_pair_sample": pair_sample is not None,
        },
    )