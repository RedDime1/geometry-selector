from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import eigsh

from geomselect.metrics import stress1, stress1_from_vectors
from geomselect.preprocessing import check_distance_matrix, normalize_distance_matrix
from geomselect.result import GeometryCandidate
from geomselect.plus import hydra_plus_refine


def poincare_distance_matrix(
    Y: np.ndarray,
    *,
    kappa: float = 1.0,
    eps: float = 1e-15,
) -> np.ndarray:
    Y = np.asarray(Y, dtype=float)

    if Y.ndim != 2:
        raise ValueError("Y must be a two-dimensional array.")

    if kappa <= 0 or not np.isfinite(kappa):
        raise ValueError("kappa must be positive and finite.")

    sq_norms = np.sum(Y * Y, axis=1)

    if np.any(sq_norms >= 1.0):
        norms = np.sqrt(np.maximum(sq_norms, eps))
        mask = norms >= 1.0
        Y = Y.copy()
        Y[mask] = Y[mask] / norms[mask, None] * (1.0 - 1e-12)
        sq_norms = np.sum(Y * Y, axis=1)

    diff = Y[:, None, :] - Y[None, :, :]
    diff2 = np.sum(diff * diff, axis=2)

    denom = np.maximum((1.0 - sq_norms[:, None]) * (1.0 - sq_norms[None, :]), eps)
    arg = 1.0 + 2.0 * diff2 / denom
    arg = np.maximum(arg, 1.0)

    D = np.arccosh(arg) / np.sqrt(kappa)
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)

    return D


def hyperbolic_stress(
    D: np.ndarray,
    Y: np.ndarray,
    *,
    kappa: float,
    pairs: tuple[np.ndarray, np.ndarray] | None = None,
) -> float:
    D = check_distance_matrix(D)
    Y = np.asarray(Y, dtype=float)

    if pairs is None:
        D_hat = poincare_distance_matrix(Y, kappa=kappa)
        return stress1(D, D_hat)

    rows, cols = pairs
    D_hat = poincare_distance_matrix(Y, kappa=kappa)

    return stress1_from_vectors(
        D[rows, cols],
        D_hat[rows, cols],
    )


def hyperbolic_matrix_A(D_scaled: np.ndarray, kappa_scaled: float) -> np.ndarray:
    D_scaled = check_distance_matrix(D_scaled)

    if kappa_scaled <= 0 or not np.isfinite(kappa_scaled):
        raise ValueError("kappa_scaled must be positive and finite.")

    return np.cosh(np.sqrt(kappa_scaled) * D_scaled)


def _extreme_eigs(
    A: np.ndarray,
    *,
    d: int,
    full_threshold: int = 250,
    tol: float = 1e-10,
) -> dict[str, np.ndarray]:
    A = np.asarray(A, dtype=float)
    n = A.shape[0]

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square.")

    bottom_k = min(d + 1, n)
    top_k = min(2, n)

    if n <= full_threshold or bottom_k + top_k >= n:
        evals, evecs = np.linalg.eigh(A)
        order = np.argsort(evals)
        evals = evals[order]
        evecs = evecs[:, order]

        return {
            "bottom_vals": evals[:bottom_k],
            "bottom_vecs": evecs[:, :bottom_k],
            "top_vals": evals[-top_k:],
            "top_vecs": evecs[:, -top_k:],
            "all_vals": evals,
        }

    bottom_vals, bottom_vecs = eigsh(
        A,
        k=bottom_k,
        which="SA",
        tol=tol,
        return_eigenvectors=True,
    )

    order_bottom = np.argsort(bottom_vals)
    bottom_vals = bottom_vals[order_bottom]
    bottom_vecs = bottom_vecs[:, order_bottom]

    top_vals, top_vecs = eigsh(
        A,
        k=top_k,
        which="LA",
        tol=tol,
        return_eigenvectors=True,
    )

    order_top = np.argsort(top_vals)
    top_vals = top_vals[order_top]
    top_vecs = top_vecs[:, order_top]

    return {
        "bottom_vals": bottom_vals,
        "bottom_vecs": bottom_vecs,
        "top_vals": top_vals,
        "top_vecs": top_vecs,
        "all_vals": None,
    }


def hyperbolic_signature_score(
    D_scaled: np.ndarray,
    kappa_scaled: float,
    d: int,
    *,
    t_max: float = 40.0,
    eps: float = 1e-15,
) -> float:
    D_scaled = check_distance_matrix(D_scaled)

    if d < 1:
        raise ValueError("Embedding dimension d must be positive.")

    if kappa_scaled <= 0 or not np.isfinite(kappa_scaled):
        return np.inf

    T = np.sqrt(kappa_scaled) * D_scaled

    if np.max(T) > t_max:
        return np.inf

    A = np.cosh(T)
    eig = _extreme_eigs(A, d=d)

    bottom_vals = eig["bottom_vals"]
    top_vals = eig["top_vals"]

    n = A.shape[0]

    lambda_top1 = float(top_vals[-1])

    if len(top_vals) >= 2:
        lambda_top2 = float(top_vals[-2])
    else:
        lambda_top2 = np.inf

    if len(bottom_vals) >= d + 1:
        lambda_left_zero = float(bottom_vals[d])
    else:
        lambda_left_zero = np.inf

    if len(bottom_vals) >= d:
        negative_signal = float(np.sum(np.abs(bottom_vals[:d])))
    else:
        negative_signal = float(np.sum(np.abs(bottom_vals)))

    positive_nontrivial = max(lambda_top1 - n, 0.0)

    allowed_signal = negative_signal + positive_nontrivial
    forbidden_signal = abs(lambda_left_zero) + abs(lambda_top2)

    return float(forbidden_signal / max(allowed_signal, eps))


def _kappa_scaled_max_from_D(
    D_scaled: np.ndarray,
    *,
    t_max: float = 20.0,
    eps: float = 1e-15,
) -> float:
    D_scaled = check_distance_matrix(D_scaled)
    max_D = float(np.max(D_scaled))

    if max_D <= eps:
        return 1.0

    return float((t_max / max_D) ** 2)


def _make_kappa_grid(
    D_scaled: np.ndarray,
    *,
    grid_num: int = 31,
    span_decades: float = 3.0,
    center: float = 1.0,
    t_max: float = 20.0,
    min_kappa_scaled: float = 0.01,
) -> np.ndarray:
    if grid_num < 3:
        raise ValueError("grid_num must be at least 3.")

    if center <= 0 or not np.isfinite(center):
        raise ValueError("center must be positive and finite.")

    kappa_max = _kappa_scaled_max_from_D(D_scaled, t_max=t_max)

    lo = center * 10.0 ** (-span_decades)
    hi = center * 10.0 ** span_decades

    lo = max(lo, min_kappa_scaled)
    hi = min(hi, kappa_max)

    if hi < lo:
        lo = max(min_kappa_scaled, min(kappa_max, center * 0.1))
        hi = max(kappa_max, lo)

    if hi <= lo:
        hi = lo * 1.0001

    return np.logspace(
        np.log10(lo),
        np.log10(hi),
        int(grid_num),
    )


def select_kappa_by_signature(
    D: np.ndarray,
    d: int = 2,
    *,
    norm_method: str | None = "median",
    grid_num: int = 31,
    span_decades: float = 3.0,
    center: float = 1.0,
    t_max: float = 20.0,
    n_refine: int = 3,
    refine_num: int = 25,
    min_kappa_scaled: float = 0.01,
) -> dict[str, object]:
    D = check_distance_matrix(D)
    D_scaled, scale_s = normalize_distance_matrix(D, method=norm_method)

    cache: dict[float, float] = {}

    def eval_one(kappa_scaled: float) -> float:
        kappa_scaled = float(kappa_scaled)

        if not np.isfinite(kappa_scaled) or kappa_scaled <= 0:
            return np.inf

        key = float(f"{kappa_scaled:.14g}")

        if key not in cache:
            cache[key] = hyperbolic_signature_score(
                D_scaled,
                key,
                d=d,
                t_max=max(t_max, 40.0),
            )

        return cache[key]

    grid = _make_kappa_grid(
        D_scaled,
        grid_num=grid_num,
        span_decades=span_decades,
        center=center,
        t_max=t_max,
        min_kappa_scaled=min_kappa_scaled,
    )

    all_kappas: list[float] = []
    all_scores: list[float] = []

    current = np.asarray(grid, dtype=float)

    for step in range(max(int(n_refine), 0) + 1):
        scores = np.array([eval_one(k) for k in current], dtype=float)

        all_kappas.extend([float(k) for k in current])
        all_scores.extend([float(s) for s in scores])

        best_idx = int(np.nanargmin(scores))
        best_kappa = float(current[best_idx])

        if step == n_refine:
            break

        sorted_current = np.sort(np.unique(current))

        pos = int(np.searchsorted(sorted_current, best_kappa))

        if pos <= 0:
            left = best_kappa / 10.0 ** (span_decades / max(grid_num - 1, 1))
        else:
            left = float(sorted_current[pos - 1])

        if pos >= len(sorted_current) - 1:
            right = best_kappa * 10.0 ** (span_decades / max(grid_num - 1, 1))
        else:
            right = float(sorted_current[pos + 1])

        left = max(left, min_kappa_scaled)

        kappa_max = _kappa_scaled_max_from_D(D_scaled, t_max=t_max)
        right = min(right, kappa_max)

        if right <= left:
            left = max(min_kappa_scaled, best_kappa * 0.8)
            right = min(kappa_max, best_kappa * 1.2)

        if right <= left:
            break

        current = np.logspace(
            np.log10(left),
            np.log10(right),
            int(refine_num),
        )

    kappas = np.asarray(all_kappas, dtype=float)
    scores = np.asarray(all_scores, dtype=float)

    finite = np.isfinite(kappas) & np.isfinite(scores) & (kappas > 0)

    if not np.any(finite):
        raise RuntimeError("No valid kappa was found.")

    kappas = kappas[finite]
    scores = scores[finite]

    order = np.argsort(kappas)
    kappas = kappas[order]
    scores = scores[order]

    best_idx = int(np.argmin(scores))
    best_kappa_scaled = float(kappas[best_idx])
    best_score = float(scores[best_idx])
    best_kappa = float(best_kappa_scaled / (scale_s * scale_s))

    return {
        "kappa": best_kappa,
        "kappa_scaled": best_kappa_scaled,
        "scale_s": float(scale_s),
        "score": best_score,
        "grid_kappa_scaled": kappas,
        "grid_kappa": kappas / (scale_s * scale_s),
        "grid_score": scores,
        "norm_method": norm_method,
    }


def select_kappa_by_signature_multisection(
    D: np.ndarray,
    d: int = 2,
    *,
    norm_method: str | None = "median",
    grid_num: int = 31,
    span_decades: float = 3.0,
    center: float = 1.0,
    t_max: float = 20.0,
    n_refine: int = 3,
    refine_num: int = 25,
    verbose: bool = False,
) -> dict[str, object]:
    result = select_kappa_by_signature(
        D,
        d=d,
        norm_method=norm_method,
        grid_num=grid_num,
        span_decades=span_decades,
        center=center,
        t_max=t_max,
        n_refine=n_refine,
        refine_num=refine_num,
    )

    if verbose:
        print(
            "kappa_scaled=",
            result["kappa_scaled"],
            "kappa=",
            result["kappa"],
            "score=",
            result["score"],
        )

    return result


def _poincare_projection_hydra(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)

    if X.ndim != 2 or X.shape[1] < 2:
        raise ValueError("Lorentz coordinates X must have shape (n, d + 1).")

    spatial = X[:, 1:]
    spatial_norm = np.linalg.norm(spatial, axis=1, keepdims=True)

    U = np.zeros_like(spatial)
    mask = spatial_norm[:, 0] > 1e-15
    U[mask] = spatial[mask] / spatial_norm[mask]

    x0 = X[:, 0]
    x_min = min(1.0, float(np.min(x0)))

    numer = np.maximum(x0 - x_min, 0.0)
    denom = np.maximum(x0 + x_min, 1e-15)

    r = np.sqrt(numer / denom)
    Y = U * r[:, None]

    norms = np.linalg.norm(Y, axis=1)
    mask = norms >= 1.0

    if np.any(mask):
        Y[mask] = Y[mask] / norms[mask, None] * (1.0 - 1e-12)

    return Y


def hydra_fixed_kappa(
    D_scaled: np.ndarray,
    d: int = 2,
    *,
    kappa_scaled: float = 1.0,
    eig_tol: float = 1e-10,
    full_threshold: int = 250,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    D_scaled = check_distance_matrix(D_scaled)

    if d < 1:
        raise ValueError("Embedding dimension d must be positive.")

    if kappa_scaled <= 0 or not np.isfinite(kappa_scaled):
        raise ValueError("kappa_scaled must be positive and finite.")

    A = hyperbolic_matrix_A(D_scaled, kappa_scaled)
    n = A.shape[0]

    if d >= n:
        raise ValueError("Embedding dimension d must be smaller than n.")

    if n <= full_threshold or d + 1 >= n:
        evals, evecs = np.linalg.eigh(A)
        order = np.argsort(evals)
        evals = evals[order]
        evecs = evecs[:, order]

        bottom_vals = evals[:d]
        bottom_vecs = evecs[:, :d]

        top_val = float(evals[-1])
        top_vec = evecs[:, -1]
    else:
        top_vals, top_vecs = eigsh(
            A,
            k=1,
            which="LA",
            tol=eig_tol,
            return_eigenvectors=True,
        )

        top_val = float(top_vals[0])
        top_vec = top_vecs[:, 0]

        bottom_vals, bottom_vecs = eigsh(
            A,
            k=d,
            which="SA",
            tol=eig_tol,
            return_eigenvectors=True,
        )

        order = np.argsort(bottom_vals)
        bottom_vals = bottom_vals[order]
        bottom_vecs = bottom_vecs[:, order]

    if np.sum(top_vec) < 0:
        top_vec = -top_vec

    X = np.zeros((n, d + 1), dtype=float)
    X[:, 0] = np.sqrt(max(top_val, 0.0)) * top_vec

    for j in range(d):
        X[:, j + 1] = np.sqrt(max(-float(bottom_vals[j]), 0.0)) * bottom_vecs[:, j]

    Y = _poincare_projection_hydra(X)

    return Y, X, top_val, np.asarray(bottom_vals, dtype=float)


def hydra_fixed_kappa_eigsh(
    D: np.ndarray,
    d: int,
    kappa: float = 1.0,
    tol: float = 1e-10,
    maxiter: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    D = check_distance_matrix(D)

    if kappa <= 0 or not np.isfinite(kappa):
        raise ValueError("kappa must be positive and finite.")

    A = hyperbolic_matrix_A(D, kappa)
    n = A.shape[0]

    if n <= 250 or d + 1 >= n:
        evals, evecs = np.linalg.eigh(A)
        order = np.argsort(evals)
        evals = evals[order]
        evecs = evecs[:, order]

        lam_bottom = evals[:d]
        q_bottom = evecs[:, :d]

        lam_top = float(evals[-1])
        q_top = evecs[:, -1]
    else:
        lam_top_arr, q_top_arr = eigsh(
            A,
            k=1,
            which="LA",
            tol=tol,
            maxiter=maxiter,
            return_eigenvectors=True,
        )

        lam_top = float(lam_top_arr[0])
        q_top = q_top_arr[:, 0]

        lam_bottom, q_bottom = eigsh(
            A,
            k=d,
            which="SA",
            tol=tol,
            maxiter=maxiter,
            return_eigenvectors=True,
        )

        idx = np.argsort(lam_bottom)
        lam_bottom = lam_bottom[idx]
        q_bottom = q_bottom[:, idx]

    if np.sum(q_top) < 0:
        q_top = -q_top

    X = np.zeros((n, d + 1), dtype=float)
    X[:, 0] = np.sqrt(max(lam_top, 0.0)) * q_top

    for j in range(d):
        X[:, j + 1] = np.sqrt(max(-float(lam_bottom[j]), 0.0)) * q_bottom[:, j]

    Y = _poincare_projection_hydra(X)

    return Y, X, lam_top, np.asarray(lam_bottom, dtype=float), A

def _safe_float_for_plus(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return np.inf

    if not np.isfinite(x):
        return np.inf

    return x


def fit_hyperbolic(
    D: np.ndarray,
    d: int = 2,
    *,
    pairs: tuple[np.ndarray, np.ndarray] | None = None,
    norm_method: str | None = "median",
    grid_num: int = 31,
    span_decades: float = 3.0,
    center: float = 1.0,
    t_max: float = 20.0,
    n_refine: int = 3,
    refine_num: int = 25,
    eig_tol: float = 1e-10,
    do_plus: bool = False,
    plus_pairs: tuple[np.ndarray, np.ndarray] | None = None,
    plus_maxiter: int = 200,
    plus_gtol: float = 1e-6,
    rollback_plus: bool = True,
    random_state: int | None = None,
) -> GeometryCandidate:
    D = check_distance_matrix(D)

    selection = select_kappa_by_signature(
        D,
        d=d,
        norm_method=norm_method,
        grid_num=grid_num,
        span_decades=span_decades,
        center=center,
        t_max=t_max,
        n_refine=n_refine,
        refine_num=refine_num,
    )

    scale_s = float(selection["scale_s"])
    kappa_scaled = float(selection["kappa_scaled"])
    kappa = float(selection["kappa"])

    D_scaled = D / max(scale_s, 1e-15)

    Y0, X, top_eigenvalue, bottom_eigenvalues = hydra_fixed_kappa(
        D_scaled,
        d=d,
        kappa_scaled=kappa_scaled,
        eig_tol=eig_tol,
    )

    stress_before = hyperbolic_stress(
        D,
        Y0,
        kappa=kappa,
        pairs=pairs,
    )

    Y_final = Y0
    stress_after = np.nan
    used_plus = False
    plus_info = None

    if do_plus:
        try:
            Y1, plus_info = hydra_plus_refine(
                D_scaled,
                Y0,
                kappa=kappa_scaled,
                pairs=plus_pairs,
                maxiter=plus_maxiter,
                gtol=plus_gtol,
                seed=random_state,
                verbose=False,
            )

            stress_after = hyperbolic_stress(
                D,
                Y1,
                kappa=kappa,
                pairs=pairs,
            )

            if (not rollback_plus) or (
                _safe_float_for_plus(stress_after)
                <= _safe_float_for_plus(stress_before)
            ):
                Y_final = Y1
                used_plus = True

        except Exception as exc:
            plus_info = {
                "success": False,
                "message": str(exc),
            }

    stress_final = stress_after if used_plus else stress_before

    return GeometryCandidate(
        geometry="hyperbolic",
        stress=float(stress_final),
        embedding=Y_final,
        parameter_name="kappa",
        parameter_value=kappa,
        metadata={
            "d": int(d),
            "kappa_scaled": kappa_scaled,
            "scale_s": scale_s,
            "selection_score": float(selection["score"]),
            "top_eigenvalue": float(top_eigenvalue),
            "bottom_eigenvalues": np.asarray(bottom_eigenvalues, dtype=float),
            "lorentz_embedding": X,
            "selection": selection,
            "stress_before_plus": float(stress_before),
            "stress_after_plus": float(stress_after) if np.isfinite(stress_after) else np.nan,
            "used_plus": bool(used_plus),
            "plus_info": plus_info,
        },
    )