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


def _safe_float(x: object, default: float = np.nan) -> float:
    try:
        value = float(x)
    except Exception:
        return default

    if not np.isfinite(value):
        return default

    return value


def _scaled_distance_quantile(
    D: np.ndarray,
    scale_s: float,
    q: float,
) -> float:
    D_scaled = np.asarray(D, dtype=float) / max(float(scale_s), 1e-15)
    values = D_scaled[D_scaled > 0]

    if values.size == 0:
        return 0.0

    return float(np.quantile(values, q))


def _curvature_regime(
    signal: float,
    *,
    flat_signal_threshold: float,
    weak_signal_threshold: float,
) -> str:
    signal = _safe_float(signal, default=0.0)

    if signal < flat_signal_threshold:
        return "near_flat"

    if signal < weak_signal_threshold:
        return "weak_curvature"

    return "curved"


def _attach_curvature_diagnostics(
    candidate: GeometryCandidate,
    D: np.ndarray,
    config: GeometrySelectorConfig,
) -> None:
    metadata = candidate.metadata

    if candidate.geometry == "euclidean":
        metadata["signed_curvature"] = 0.0
        metadata["curvature_signal"] = 0.0
        metadata["curvature_regime"] = "flat"
        return

    scale_s = _safe_float(metadata.get("scale_s", 1.0), default=1.0)
    qD = _scaled_distance_quantile(
        D,
        scale_s=scale_s,
        q=config.curvature_quantile,
    )

    if candidate.geometry == "hyperbolic":
        kappa = _safe_float(candidate.parameter_value, default=np.nan)
        kappa_scaled = _safe_float(metadata.get("kappa_scaled", np.nan), default=np.nan)

        signal = np.sqrt(max(kappa_scaled, 0.0)) * qD

        metadata["signed_curvature"] = -kappa
        metadata["curvature_signal"] = float(signal)
        metadata["curvature_regime"] = _curvature_regime(
            signal,
            flat_signal_threshold=config.flat_signal_threshold,
            weak_signal_threshold=config.weak_signal_threshold,
        )
        return

    if candidate.geometry == "spherical":
        R = _safe_float(candidate.parameter_value, default=np.nan)
        R_scaled = _safe_float(metadata.get("R_scaled", np.nan), default=np.nan)

        signal = qD / max(R_scaled, 1e-15)

        metadata["signed_curvature"] = 1.0 / max(R * R, 1e-15)
        metadata["curvature_signal"] = float(signal)
        metadata["curvature_regime"] = _curvature_regime(
            signal,
            flat_signal_threshold=config.flat_signal_threshold,
            weak_signal_threshold=config.weak_signal_threshold,
        )
        return

    metadata["signed_curvature"] = np.nan
    metadata["curvature_signal"] = np.nan
    metadata["curvature_regime"] = "unknown"


def _make_recommendation(
    *,
    selected: GeometryCandidate,
    best_by_stress: GeometryCandidate,
    second_by_stress: GeometryCandidate | None,
    euclidean_candidate: GeometryCandidate | None,
    close_ratio: float,
    close_abs: float,
    used_near_flat_override: bool,
) -> tuple[str, str]:
    if second_by_stress is None:
        return f"Selected geometry: {selected.geometry}.", "high"

    best_stress = _safe_stress(best_by_stress)
    second_stress = _safe_stress(second_by_stress)

    absolute_gap = second_stress - best_stress
    relative_gap = second_stress / max(best_stress, 1e-15)

    if best_stress < 1e-10:
        is_close = second_stress <= 1e-8
    else:
        is_close = relative_gap <= close_ratio or absolute_gap <= close_abs

    best_regime = str(best_by_stress.metadata.get("curvature_regime", "unknown"))

    if used_near_flat_override:
        return (
            f"Best geometry by stress is {best_by_stress.geometry}, but its curvature is close to zero "
            f"and the Euclidean stress is close. Selected geometry: euclidean. "
            f"The result should be interpreted as an almost flat case.",
            "low_or_medium",
        )

    if best_by_stress.geometry in {"hyperbolic", "spherical"} and best_regime == "near_flat":
        if euclidean_candidate is not None:
            euclidean_stress = _safe_stress(euclidean_candidate)
            if euclidean_stress > best_stress:
                return (
                    f"Selected geometry: {best_by_stress.geometry}. "
                    f"However, the selected curvature is close to zero, so the curvature sign is unreliable. "
                    f"The Euclidean model is worse by stress, but the case should be treated as weakly curved or ambiguous.",
                    "low_or_medium",
                )

        return (
            f"Selected geometry: {best_by_stress.geometry}. "
            f"However, the selected curvature is close to zero, so the result should be treated cautiously.",
            "low_or_medium",
        )

    if is_close:
        return (
            f"Selected geometry: {best_by_stress.geometry}. "
            f"The second candidate is close, so the result should be treated cautiously.",
            "low_or_medium",
        )

    return (
        f"Selected geometry: {best_by_stress.geometry}. The stress gap is noticeable.",
        "high",
    )


def _fit_candidate(
    geometry: str,
    D: np.ndarray,
    config: GeometrySelectorConfig,
    pairs: tuple[np.ndarray, np.ndarray] | None,
    plus_pairs: tuple[np.ndarray, np.ndarray] | None,
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
            do_plus=config.do_plus,
            plus_pairs=plus_pairs,
            plus_maxiter=config.plus_maxiter_hyper,
            plus_gtol=config.plus_gtol,
            rollback_plus=config.rollback_plus,
            random_state=config.random_state,
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
            do_plus=config.do_plus,
            plus_pairs=plus_pairs,
            plus_maxiter=config.plus_maxiter_sphere,
            plus_gtol=config.plus_gtol,
            rollback_plus=config.rollback_plus,
            random_state=config.random_state,
        )

    raise NotImplementedError(f"Geometry {geometry!r} is not implemented yet.")


def _make_candidate_table(candidates: list[GeometryCandidate]) -> pd.DataFrame:
    rows = []

    extra_keys = [
        "stress_before_plus",
        "stress_after_plus",
        "used_plus",
        "selection_score",
        "kappa_scaled",
        "R_scaled",
        "scale_s",
        "signed_curvature",
        "curvature_signal",
        "curvature_regime",
    ]

    for candidate in candidates:
        row = candidate.as_dict()
        metadata = candidate.metadata or {}

        for key in extra_keys:
            if key in metadata:
                row[key] = metadata[key]

        rows.append(row)

    table = pd.DataFrame(rows)
    table.insert(0, "rank", np.arange(1, len(table) + 1))

    return table


def _select_with_near_flat_rule(
    candidates: list[GeometryCandidate],
    config: GeometrySelectorConfig,
) -> tuple[GeometryCandidate, GeometryCandidate, GeometryCandidate | None, bool]:
    ordered = sorted(candidates, key=_safe_stress)
    best = ordered[0]
    second = ordered[1] if len(ordered) > 1 else None

    euclidean_candidate = next(
        (candidate for candidate in ordered if candidate.geometry == "euclidean"),
        None,
    )

    used_near_flat_override = False
    selected = best

    if (
        best.geometry in {"hyperbolic", "spherical"}
        and str(best.metadata.get("curvature_regime")) == "near_flat"
        and euclidean_candidate is not None
    ):
        best_stress = _safe_stress(best)
        euclidean_stress = _safe_stress(euclidean_candidate)

        euclidean_is_close = (
            euclidean_stress <= best_stress * config.euclidean_flat_close_ratio
            or euclidean_stress - best_stress <= config.euclidean_flat_close_abs
        )

        if euclidean_is_close:
            selected = euclidean_candidate
            used_near_flat_override = True

    return selected, best, second, used_near_flat_override


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
    do_plus: bool = False,
    plus_pair_sample: int | None = None,
    plus_maxiter_hyper: int = 200,
    plus_maxiter_sphere: int = 300,
    plus_gtol: float = 1e-6,
    rollback_plus: bool = True,
    flat_signal_threshold: float = 0.15,
    weak_signal_threshold: float = 0.35,
    euclidean_flat_close_ratio: float = 1.10,
    euclidean_flat_close_abs: float = 1e-3,
    curvature_quantile: float = 0.90,
    plot: bool = False,
    plot_geometry: str | None = None,
    plot_hyperbolic_model: str = "poincare",
    plot_labels=None,
    plot_show: bool = True,
    plot_marker_size: int = 5,
    plot_show_surface: bool = True,
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
        do_plus=do_plus,
        plus_pair_sample=plus_pair_sample,
        plus_maxiter_hyper=plus_maxiter_hyper,
        plus_maxiter_sphere=plus_maxiter_sphere,
        plus_gtol=plus_gtol,
        rollback_plus=rollback_plus,
        flat_signal_threshold=flat_signal_threshold,
        weak_signal_threshold=weak_signal_threshold,
        euclidean_flat_close_ratio=euclidean_flat_close_ratio,
        euclidean_flat_close_abs=euclidean_flat_close_abs,
        curvature_quantile=curvature_quantile,
    )

    pairs = _make_pairs(
        n=n,
        pair_sample=pair_sample,
        random_state=random_state,
    )

    if random_state is None:
        plus_random_state = None
    else:
        plus_random_state = random_state + 1000

    plus_pairs = _make_pairs(
        n=n,
        pair_sample=plus_pair_sample,
        random_state=plus_random_state,
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
                plus_pairs,
            )
            _attach_curvature_diagnostics(candidate, D, config)
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

    selected, best_by_stress, second_by_stress, used_near_flat_override = (
        _select_with_near_flat_rule(candidates, config)
    )

    for candidate in candidates:
        candidate.selected = candidate is selected

    table = _make_candidate_table(candidates)

    euclidean_candidate = next(
        (candidate for candidate in candidates if candidate.geometry == "euclidean"),
        None,
    )

    recommendation, confidence = _make_recommendation(
        selected=selected,
        best_by_stress=best_by_stress,
        second_by_stress=second_by_stress,
        euclidean_candidate=euclidean_candidate,
        close_ratio=close_ratio,
        close_abs=close_abs,
        used_near_flat_override=used_near_flat_override,
    )

    if second_by_stress is not None:
        second_geometry = second_by_stress.geometry
        second_stress = _safe_stress(second_by_stress)
        absolute_margin = second_stress - _safe_stress(best_by_stress)
        relative_margin = second_stress / max(_safe_stress(best_by_stress), 1e-15)
    else:
        second_geometry = None
        second_stress = np.nan
        absolute_margin = np.nan
        relative_margin = np.nan

    if euclidean_candidate is None:
        euclidean_stress = np.nan
    else:
        euclidean_stress = _safe_stress(euclidean_candidate)

    result = SelectionResult(
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
            "used_plus_pair_sample": plus_pair_sample is not None,
            "plus_pair_sample": plus_pair_sample,
            "candidate_errors": candidate_errors,
            "selected_geometry": selected.geometry,
            "best_geometry_by_stress": best_by_stress.geometry,
            "best_stress": _safe_stress(best_by_stress),
            "selected_stress": _safe_stress(selected),
            "second_geometry": second_geometry,
            "second_stress": second_stress,
            "absolute_margin": absolute_margin,
            "relative_margin": relative_margin,
            "euclidean_stress": euclidean_stress,
            "selected_curvature_signal": selected.metadata.get("curvature_signal", np.nan),
            "selected_curvature_regime": selected.metadata.get("curvature_regime", "unknown"),
            "best_curvature_signal": best_by_stress.metadata.get("curvature_signal", np.nan),
            "best_curvature_regime": best_by_stress.metadata.get("curvature_regime", "unknown"),
            "used_near_flat_override": bool(used_near_flat_override),
            "confidence": confidence,
            "do_plus": bool(do_plus),
            "rollback_plus": bool(rollback_plus),
        },
    )
    if plot:
        if d != 2:
            result.metadata["plot_warning"] = "Plotting is available only for d=2."
        else:
            from geomselect.visualization import plot_embedding

            fig = plot_embedding(
                result,
                geometry=plot_geometry,
                labels=plot_labels,
                marker_size=plot_marker_size,
                show_surface=plot_show_surface,
                show=plot_show,
                hyperbolic_model=plot_hyperbolic_model,
            )
            result.metadata["figure"] = fig
    return result
