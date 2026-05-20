from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import eigsh

from geomselect.metrics import stress1, stress1_from_vectors
from geomselect.preprocessing import check_distance_matrix, normalize_distance_matrix
from geomselect.result import GeometryCandidate
from geomselect.plus import sphere_plus_refine


def spherical_distance_matrix(
    U: np.ndarray,
    *,
    R: float = 1.0,
    eps: float = 1e-15,
) -> np.ndarray:
    U = np.asarray(U, dtype=float)

    if U.ndim != 2:
        raise ValueError("U must be a two-dimensional array.")

    if R <= 0 or not np.isfinite(R):
        raise ValueError("R must be positive and finite.")

    norms = np.linalg.norm(U, axis=1, keepdims=True)
    U = U / np.maximum(norms, eps)

    dots = np.clip(U @ U.T, -1.0, 1.0)
    D_hat = R * np.arccos(dots)
    D_hat = 0.5 * (D_hat + D_hat.T)
    np.fill_diagonal(D_hat, 0.0)

    return D_hat


def spherical_stress(
    D: np.ndarray,
    U: np.ndarray,
    *,
    R: float,
    pairs: tuple[np.ndarray, np.ndarray] | None = None,
) -> float:
    D = check_distance_matrix(D)
    U = np.asarray(U, dtype=float)

    if R <= 0 or not np.isfinite(R):
        raise ValueError("R must be positive and finite.")

    norms = np.linalg.norm(U, axis=1, keepdims=True)
    U = U / np.maximum(norms, 1e-15)

    if pairs is None:
        D_hat = spherical_distance_matrix(U, R=R)
        return stress1(D, D_hat)

    rows, cols = pairs

    dots = np.sum(U[rows] * U[cols], axis=1)
    dots = np.clip(dots, -1.0, 1.0)

    d_hat = R * np.arccos(dots)
    d_true = D[rows, cols]

    return stress1_from_vectors(d_true, d_hat)


def spherical_matrix_A(D_scaled: np.ndarray, R_scaled: float) -> np.ndarray:
    D_scaled = check_distance_matrix(D_scaled)
    R_scaled = float(R_scaled)

    if R_scaled <= 0 or not np.isfinite(R_scaled):
        raise ValueError("R_scaled must be positive and finite.")

    A = np.cos(D_scaled / R_scaled)
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 1.0)

    return A


def _safe_min_eigval(
    A: np.ndarray,
    *,
    tol: float = 1e-8,
    maxiter: int | None = None,
) -> float:
    A = np.asarray(A, dtype=float)
    A = 0.5 * (A + A.T)

    n = A.shape[0]

    if n <= 3:
        vals = np.linalg.eigvalsh(A)
        return float(vals[0])

    val = eigsh(
        A,
        k=1,
        which="SA",
        return_eigenvectors=False,
        tol=tol,
        maxiter=maxiter,
    )[0]

    return float(val)


def _safe_tail_abs_sum_eigvals(
    A: np.ndarray,
    keep_rank: int,
    *,
    tol: float = 1e-8,
    maxiter: int | None = None,
) -> float:
    A = np.asarray(A, dtype=float)
    A = 0.5 * (A + A.T)

    n = A.shape[0]
    keep_rank = int(keep_rank)
    tail_size = n - keep_rank

    if tail_size <= 0:
        return 0.0

    if n <= 300:
        vals = np.linalg.eigvalsh(A)
        vals = np.sort(vals)[::-1]
        tail = vals[keep_rank:]
        return float(np.sum(np.abs(tail)))

    k = min(tail_size, max(1, min(50, n - 2)))

    vals = eigsh(
        A,
        k=k,
        which="SA",
        return_eigenvectors=False,
        tol=tol,
        maxiter=maxiter,
    )

    return float(np.sum(np.abs(vals)))


def spherical_min_score(
    D_scaled: np.ndarray,
    R_scaled: float,
    *,
    eig_tol: float = 1e-8,
    eig_maxiter: int | None = None,
) -> float:
    D_scaled = check_distance_matrix(D_scaled)
    R_scaled = float(R_scaled)

    if R_scaled <= 0 or not np.isfinite(R_scaled):
        return np.inf

    max_D = float(np.max(D_scaled))

    if R_scaled < max_D / np.pi:
        return np.inf

    A = spherical_matrix_A(D_scaled, R_scaled)

    lambda_min = _safe_min_eigval(
        A,
        tol=eig_tol,
        maxiter=eig_maxiter,
    )

    return float((R_scaled * R_scaled) * abs(lambda_min))


def spherical_tail_score(
    D_scaled: np.ndarray,
    d: int,
    R_scaled: float,
    *,
    eig_tol: float = 1e-8,
    eig_maxiter: int | None = None,
) -> float:
    D_scaled = check_distance_matrix(D_scaled)
    R_scaled = float(R_scaled)

    if R_scaled <= 0 or not np.isfinite(R_scaled):
        return np.inf

    max_D = float(np.max(D_scaled))

    if R_scaled < max_D / np.pi:
        return np.inf

    A = spherical_matrix_A(D_scaled, R_scaled)

    tail_sum = _safe_tail_abs_sum_eigvals(
        A,
        keep_rank=d + 1,
        tol=eig_tol,
        maxiter=eig_maxiter,
    )

    return float((R_scaled * R_scaled) * tail_sum)


def _spherical_min_details(
    D_scaled: np.ndarray,
    R_scaled: float,
    *,
    eig_tol: float = 1e-8,
    eig_maxiter: int | None = None,
) -> dict[str, float]:
    D_scaled = check_distance_matrix(D_scaled)
    R_scaled = float(R_scaled)

    if R_scaled <= 0 or not np.isfinite(R_scaled):
        return {
            "score": np.inf,
            "lambda_min": np.nan,
        }

    max_D = float(np.max(D_scaled))

    if R_scaled < max_D / np.pi:
        return {
            "score": np.inf,
            "lambda_min": np.nan,
        }

    A = spherical_matrix_A(D_scaled, R_scaled)

    lambda_min = _safe_min_eigval(
        A,
        tol=eig_tol,
        maxiter=eig_maxiter,
    )

    score = (R_scaled * R_scaled) * abs(lambda_min)

    return {
        "score": float(score),
        "lambda_min": float(lambda_min),
    }


def _make_R_grid_scaled_capped(
    D_scaled: np.ndarray,
    *,
    num: int = 21,
    span_decades: float = 2.0,
    center: float = 1.0,
    r_max_factor: float = 6.0,
    r_max_abs: float | None = None,
) -> np.ndarray:
    D_scaled = check_distance_matrix(D_scaled)

    if num < 2:
        raise ValueError("num must be at least 2.")

    if center <= 0 or not np.isfinite(center):
        raise ValueError("center must be positive and finite.")

    max_D = float(np.max(D_scaled))
    R_min = max_D / np.pi if max_D > 0 else 1.0

    lo = max(R_min, center * 10.0 ** (-span_decades))

    if r_max_abs is None:
        hi_cap = r_max_factor * max(max_D, 1e-12)
    else:
        hi_cap = float(r_max_abs)

    hi = min(center * 10.0 ** span_decades, hi_cap)

    if not np.isfinite(hi) or hi <= lo:
        hi = lo * 10.0

    grid = np.logspace(np.log10(lo), np.log10(hi), int(num))
    grid[0] = lo

    return np.unique(grid)


def multisection_argmin_on_grid(
    objective,
    initial_grid: np.ndarray,
    *,
    n_refine: int = 3,
    num: int = 21,
    verbose: bool = False,
) -> tuple[float, float, list[dict[str, object]]]:
    grid = np.asarray(initial_grid, dtype=float)
    grid = grid[np.isfinite(grid) & (grid > 0)]
    grid = np.unique(np.sort(grid))

    if len(grid) < 2:
        raise ValueError("Initial grid must contain at least two positive points.")

    history: list[dict[str, object]] = []

    best_x = None
    best_score = np.inf
    current_grid = grid

    for step in range(int(n_refine) + 1):
        scores = []

        for x in current_grid:
            try:
                score = float(objective(float(x)))
            except Exception:
                score = np.inf

            scores.append(score)

            if np.isfinite(score) and score < best_score:
                best_score = score
                best_x = float(x)

        scores_arr = np.asarray(scores, dtype=float)

        history.append(
            {
                "step": step,
                "grid": current_grid.copy(),
                "scores": scores_arr.copy(),
                "best_x": best_x,
                "best_score": best_score,
            }
        )

        if verbose:
            if np.any(np.isfinite(scores_arr)):
                idx_print = int(np.nanargmin(scores_arr))
                print(
                    f"[step {step}] best_grid={current_grid[idx_print]:.6g}, "
                    f"score={scores_arr[idx_print]:.6g}"
                )

        if step == int(n_refine):
            break

        if not np.any(np.isfinite(scores_arr)):
            raise RuntimeError("All objective values are invalid.")

        idx = int(np.nanargmin(scores_arr))

        if idx == 0:
            lo, hi = current_grid[0], current_grid[1]
        elif idx == len(current_grid) - 1:
            lo, hi = current_grid[-2], current_grid[-1]
        else:
            lo, hi = current_grid[idx - 1], current_grid[idx + 1]

        if lo <= 0 or hi <= 0 or lo >= hi:
            break

        current_grid = np.logspace(
            np.log10(lo),
            np.log10(hi),
            num=int(num),
        )

        current_grid[0] = lo
        current_grid[-1] = hi
        current_grid = np.unique(current_grid)

    if best_x is None or not np.isfinite(best_score):
        raise RuntimeError("No valid minimum was found.")

    return float(best_x), float(best_score), history


def select_R_by_spectral(
    D: np.ndarray,
    d: int = 2,
    *,
    norm_method: str | None = "median",
    mode: str = "min",
    grid_num: int = 21,
    span_decades: float = 2.0,
    center: float = 1.0,
    r_max_factor: float = 6.0,
    r_max_abs: float | None = None,
    n_refine: int = 3,
    refine_num: int = 21,
    eig_tol: float = 1e-8,
    eig_maxiter: int | None = None,
    verbose: bool = False,
) -> dict[str, object]:
    D = check_distance_matrix(D)

    if d < 1:
        raise ValueError("Embedding dimension d must be positive.")

    D_scaled, scale_s = normalize_distance_matrix(D, method=norm_method)

    if r_max_abs is None:
        r_max_scaled_abs = None
    else:
        r_max_scaled_abs = float(r_max_abs) / max(float(scale_s), 1e-15)

    initial_grid = _make_R_grid_scaled_capped(
        D_scaled,
        num=grid_num,
        span_decades=span_decades,
        center=center,
        r_max_factor=r_max_factor,
        r_max_abs=r_max_scaled_abs,
    )

    details_cache: dict[float, dict[str, float]] = {}

    def get_details(R_scaled: float) -> dict[str, float]:
        key = float(f"{float(R_scaled):.14g}")

        if key in details_cache:
            return details_cache[key]

        if mode == "min":
            details = _spherical_min_details(
                D_scaled,
                key,
                eig_tol=eig_tol,
                eig_maxiter=eig_maxiter,
            )
        elif mode == "tail":
            score = spherical_tail_score(
                D_scaled,
                d=d,
                R_scaled=key,
                eig_tol=eig_tol,
                eig_maxiter=eig_maxiter,
            )
            details = {
                "score": float(score),
                "lambda_min": np.nan,
            }
        else:
            raise ValueError("mode must be 'min' or 'tail'.")

        details_cache[key] = details
        return details

    def objective(R_scaled: float) -> float:
        return float(get_details(R_scaled)["score"])

    best_R_scaled, best_score, history = multisection_argmin_on_grid(
        objective,
        initial_grid=initial_grid,
        n_refine=n_refine,
        num=refine_num,
        verbose=verbose,
    )

    best_details = get_details(best_R_scaled)
    best_R = best_R_scaled * float(scale_s)

    grid_R_scaled = []
    grid_score = []
    grid_lambda_min = []

    for item in history:
        grid = np.asarray(item["grid"], dtype=float)

        for R_scaled in grid:
            key = float(f"{float(R_scaled):.14g}")
            details = get_details(key)

            grid_R_scaled.append(key)
            grid_score.append(float(details["score"]))
            grid_lambda_min.append(float(details["lambda_min"]))

    grid_R_scaled_arr = np.asarray(grid_R_scaled, dtype=float)
    grid_score_arr = np.asarray(grid_score, dtype=float)
    grid_lambda_min_arr = np.asarray(grid_lambda_min, dtype=float)

    order = np.argsort(grid_R_scaled_arr)

    if verbose:
        print()
        print("Spherical parameter selection")
        print(f"scale_s        = {float(scale_s):.6g}")
        print(f"R_scaled       = {best_R_scaled:.6g}")
        print(f"R_original     = {best_R:.6g}")
        print(f"score          = {best_score:.6g}")

    return {
        "R": float(best_R),
        "R_scaled": float(best_R_scaled),
        "scale_s": float(scale_s),
        "score": float(best_score),
        "lambda_min": float(best_details["lambda_min"]),
        "history": history,
        "grid_R_scaled": grid_R_scaled_arr[order],
        "grid_R": grid_R_scaled_arr[order] * float(scale_s),
        "grid_score": grid_score_arr[order],
        "grid_lambda_min": grid_lambda_min_arr[order],
        "norm_method": norm_method,
        "mode": mode,
    }


def select_R_by_spectral_multisection(
    D: np.ndarray,
    d: int = 2,
    *,
    norm_method: str | None = "median",
    mode: str = "min",
    grid_num: int = 21,
    span_decades: float = 2.0,
    center: float = 1.0,
    r_max_factor: float = 6.0,
    r_max_abs: float | None = None,
    n_refine: int = 3,
    refine_num: int = 21,
    verbose: bool = False,
) -> dict[str, object]:
    return select_R_by_spectral(
        D,
        d=d,
        norm_method=norm_method,
        mode=mode,
        grid_num=grid_num,
        span_decades=span_decades,
        center=center,
        r_max_factor=r_max_factor,
        r_max_abs=r_max_abs,
        n_refine=n_refine,
        refine_num=refine_num,
        verbose=verbose,
    )


def spherical_fixed_R_eigsh(
    D: np.ndarray,
    d: int,
    R: float = 1.0,
    *,
    tol: float = 1e-10,
    maxiter: int | None = None,
    full_threshold: int = 250,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    D = check_distance_matrix(D)

    if d < 1:
        raise ValueError("Embedding dimension d must be positive.")

    if R <= 0 or not np.isfinite(R):
        raise ValueError("R must be positive and finite.")

    A = spherical_matrix_A(D, R)
    n = A.shape[0]
    k = min(d + 1, n)

    if n <= full_threshold or k >= n:
        evals, evecs = np.linalg.eigh(A)
        order = np.argsort(evals)[::-1]
        vals = evals[order][:k]
        vecs = evecs[:, order][:, :k]
    else:
        vals, vecs = eigsh(
            A,
            k=k,
            which="LA",
            return_eigenvectors=True,
            tol=tol,
            maxiter=maxiter,
        )
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vecs = vecs[:, order]

    coords = vecs * np.sqrt(np.maximum(vals, 0.0))[None, :]

    if coords.shape[1] < d + 1:
        coords = np.pad(
            coords,
            ((0, 0), (0, d + 1 - coords.shape[1])),
            mode="constant",
        )

    norms = np.linalg.norm(coords, axis=1, keepdims=True)
    U = coords / np.maximum(norms, 1e-15)

    bad = norms[:, 0] <= 1e-15

    if np.any(bad):
        U[bad] = 0.0
        U[bad, 0] = 1.0

    X = R * U

    return U, X, A, vals

def _safe_float_for_plus(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return np.inf

    if not np.isfinite(x):
        return np.inf

    return x


def fit_spherical(
    D: np.ndarray,
    d: int = 2,
    *,
    pairs: tuple[np.ndarray, np.ndarray] | None = None,
    norm_method: str | None = "median",
    mode: str = "min",
    grid_num: int = 21,
    span_decades: float = 2.0,
    center: float = 1.0,
    r_max_factor: float = 6.0,
    r_max_abs: float | None = None,
    n_refine: int = 3,
    refine_num: int = 21,
    eig_tol: float = 1e-8,
    do_plus: bool = False,
    plus_pairs: tuple[np.ndarray, np.ndarray] | None = None,
    plus_maxiter: int = 300,
    plus_gtol: float = 1e-6,
    rollback_plus: bool = True,
    random_state: int | None = None,
) -> GeometryCandidate:
    D = check_distance_matrix(D)

    selection = select_R_by_spectral_multisection(
        D,
        d=d,
        norm_method=norm_method,
        mode=mode,
        grid_num=grid_num,
        span_decades=span_decades,
        center=center,
        r_max_factor=r_max_factor,
        r_max_abs=r_max_abs,
        n_refine=n_refine,
        refine_num=refine_num,
        verbose=False,
    )

    scale_s = float(selection["scale_s"])
    R_scaled = float(selection["R_scaled"])
    R = float(selection["R"])

    D_scaled = D / max(scale_s, 1e-15)

    U0, X_scaled, A, evals = spherical_fixed_R_eigsh(
        D_scaled,
        d=d,
        R=R_scaled,
        tol=eig_tol,
    )

    stress_before = spherical_stress(
        D,
        U0,
        R=R,
        pairs=pairs,
    )

    U_final = U0
    stress_after = np.nan
    used_plus = False
    plus_info = None

    if do_plus:
        try:
            U1, plus_info = sphere_plus_refine(
                D_scaled,
                U0,
                R=R_scaled,
                pairs=plus_pairs,
                seed=random_state,
                maxiter=plus_maxiter,
                gtol=plus_gtol,
                verbose=False,
            )

            stress_after = spherical_stress(
                D,
                U1,
                R=R,
                pairs=pairs,
            )

            if (not rollback_plus) or (
                _safe_float_for_plus(stress_after)
                <= _safe_float_for_plus(stress_before)
            ):
                U_final = U1
                used_plus = True

        except Exception as exc:
            plus_info = {
                "success": False,
                "message": str(exc),
            }

    stress_final = stress_after if used_plus else stress_before

    return GeometryCandidate(
        geometry="spherical",
        stress=float(stress_final),
        embedding=R * U_final,
        parameter_name="R",
        parameter_value=float(R),
        metadata={
            "d": int(d),
            "R": float(R),
            "R_scaled": float(R_scaled),
            "scale_s": float(scale_s),
            "selection_score": float(selection["score"]),
            "lambda_min": float(selection["lambda_min"]),
            "unit_embedding": U_final,
            "initial_unit_embedding": U0,
            "scaled_embedding": X_scaled,
            "A": A,
            "eigenvalues": np.asarray(evals, dtype=float),
            "selection": selection,
            "stress_before_plus": float(stress_before),
            "stress_after_plus": float(stress_after) if np.isfinite(stress_after) else np.nan,
            "used_plus": bool(used_plus),
            "plus_info": plus_info,
        },
    )