from __future__ import annotations

import numpy as np
import pandas as pd

from geomselect.euclidean import fit_euclidean
from geomselect.hyperbolic import fit_hyperbolic
from geomselect.preprocessing import check_distance_matrix, sample_pairs
from geomselect.result import GeometryCandidate, GeometrySelectorConfig, SelectionResult
from geomselect.spherical import fit_spherical


_IMPLEMENTED_GEOMETRIES = {"euclidean", "hyperbolic", "spherical"}


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


def _validate_geometries(geometries: tuple[str, ...]) -> tuple[str, ...]:
    if not geometries:
        raise ValueError("At least one geometry must be requested.")

    normalized = tuple(str(item).lower() for item in geometries)

    if len(set(normalized)) != len(normalized):
        raise ValueError("Geometry list contains duplicates.")

    unknown = sorted(set(normalized) - _IMPLEMENTED_GEOMETRIES)

    if unknown:
        raise NotImplementedError(
            "Unsupported geometries: " + ", ".join(unknown)
        )

    return normalized


def _safe_stress(candidate: GeometryCandidate) -> float:
    stress = float(candidate.stress)

    if not np.isfinite(stress):
        return np.inf

    return stress


def _make_recommendation(
    candidates: list[GeometryCandidate],
    *,
    close_ratio: float,
    close_abs: float,
) -> str:
    if len(candidates) == 1:
        return f"Selected geometry: {candidates[0].geometry}."

    ordered = sorted(candidates, key=_safe_stress)

    best = ordered[0]
    second = ordered[1]

    best_stress = _safe_stress(best)
    second_stress = _safe_stress(second)

    absolute_gap = second_stress - best_stress
    relative_gap = second_stress / max(best_stress, 1e-15)

    if best_stress < 1e-10:
        is_close = second_stress <= 1e-8
    else:
        is_close = relative_gap <= close_ratio or absolute_gap <= close_abs

    if is_close:
        return (
            f"Selected geometry: {best.geometry}. "
            f"The second candidate is close, so the result should be treated cautiously."
        )

    return f"Selected geometry: {best.geometry}. The stress gap is noticeable."


def _fit_candidate(
    geometry: str,
    D: np.ndarray,
    config: GeometrySelectorConfig,
    pairs: tuple[np.ndarray, np.ndarray] | None,
) -> GeometryCandidate:
    if geometry == "euclidean":
        return fit_euclidean(
            D,
            d=config.d,
            pairs=pairs,
        )

    if geometry == "hyperbolic":
        return fit_hyperbolic(
            D,
            d=config.d,
            pairs=pairs,
            norm_method=config.norm_method,
            grid_num=config.hyper_grid_num,
            span_decades=config.hyper_span_decades,
            center=config.center,
            t_max=config.hyper_t_max,
            n_refine=config.n_refine,
            refine_num=config.refine_num,
            eig_tol=config.eig_tol,
        )

    if geometry == "spherical":
        return fit_spherical(
            D,
            d=config.d,
            pairs=pairs,
            norm_method=config.norm_method,
            mode="min",
            grid_num=config.sphere_grid_num,
            span_decades=config.sphere_span_decades,
            center=config.center,
            r_max_factor=config.sphere_r_max_factor,
            r_max_abs=config.sphere_r_max_abs,
            n_refine=config.n_refine,
            refine_num=config.refine_num,
            eig_tol=config.eig_tol,
        )

    raise NotImplementedError(f"Geometry {geometry!r} is not implemented yet.")


def _make_candidate_table(candidates: list[GeometryCandidate]) -> pd.DataFrame:
    table = pd.DataFrame(
        [candidate.as_dict() for candidate in candidates]
    )

    table.insert(0, "rank", np.arange(1, len(table) + 1))

    return table


def select_geometry(
    D: np.ndarray,
    d: int = 2,
    *,
    geometries: tuple[str, ...] = ("euclidean", "hyperbolic", "spherical"),
    pair_sample: int | None = None,
    random_state: int | None = None,
    norm_method: str | None = "median",
    hyper_grid_num: int = 31,
    sphere_grid_num: int = 31,
    hyper_span_decades: float = 3.0,
    sphere_span_decades: float = 3.0,
    center: float = 1.0,
    hyper_t_max: float = 20.0,
    sphere_r_max_factor: float = 6.0,
    sphere_r_max_abs: float | None = None,
    n_refine: int = 3,
    refine_num: int = 25,
    eig_tol: float = 1e-10,
    close_ratio: float = 1.05,
    close_abs: float = 1e-3,
    fail_fast: bool = False,
) -> SelectionResult:
    D = check_distance_matrix(D)
    n = D.shape[0]

    geometries = _validate_geometries(tuple(geometries))

    config = GeometrySelectorConfig(
        d=d,
        geometries=geometries,
        pair_sample=pair_sample,
        random_state=random_state,
        norm_method=norm_method,
        hyper_grid_num=hyper_grid_num,
        sphere_grid_num=sphere_grid_num,
        hyper_span_decades=hyper_span_decades,
        sphere_span_decades=sphere_span_decades,
        center=center,
        hyper_t_max=hyper_t_max,
        sphere_r_max_factor=sphere_r_max_factor,
        sphere_r_max_abs=sphere_r_max_abs,
        n_refine=n_refine,
        refine_num=refine_num,
        eig_tol=eig_tol,
        close_ratio=close_ratio,
        close_abs=close_abs,
        fail_fast=fail_fast,
    )

    pairs = _make_pairs(
        n=n,
        pair_sample=pair_sample,
        random_state=random_state,
    )

    candidates: list[GeometryCandidate] = []
    candidate_errors: dict[str, str] = {}

    for geometry in geometries:
        try:
            candidate = _fit_candidate(
                geometry,
                D,
                config,
                pairs,
            )
            candidates.append(candidate)
        except Exception as exc:
            candidate_errors[geometry] = str(exc)

            if fail_fast:
                raise

    if not candidates:
        raise RuntimeError(
            "No geometry candidates were fitted. Errors: "
            + repr(candidate_errors)
        )

    candidates = sorted(candidates, key=_safe_stress)

    for candidate in candidates:
        candidate.selected = False

    selected = candidates[0]
    selected.selected = True

    table = _make_candidate_table(candidates)

    recommendation = _make_recommendation(
        candidates,
        close_ratio=close_ratio,
        close_abs=close_abs,
    )

    if len(candidates) >= 2:
        second = candidates[1]
        second_geometry = second.geometry
        second_stress = _safe_stress(second)
        absolute_margin = second_stress - _safe_stress(selected)
        relative_margin = second_stress / max(_safe_stress(selected), 1e-15)
    else:
        second_geometry = None
        second_stress = np.nan
        absolute_margin = np.nan
        relative_margin = np.nan

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
            "pair_sample": pair_sample,
            "candidate_errors": candidate_errors,
            "best_stress": _safe_stress(selected),
            "second_geometry": second_geometry,
            "second_stress": second_stress,
            "absolute_margin": absolute_margin,
            "relative_margin": relative_margin,
        },
    )