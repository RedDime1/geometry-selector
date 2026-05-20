from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from geomselect.preprocessing import sample_pairs


def poincare_to_hyperboloid_spatial(Y: np.ndarray) -> np.ndarray:
    Y = np.asarray(Y, dtype=float)
    r2 = np.sum(Y * Y, axis=1, keepdims=True)
    denom = np.maximum(1.0 - r2, 1e-15)
    return 2.0 * Y / denom


def hyperboloid_spatial_to_poincare(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    u = np.sqrt(1.0 + np.sum(x * x, axis=1, keepdims=True))
    Y = x / np.maximum(u + 1.0, 1e-15)

    norms = np.linalg.norm(Y, axis=1)
    mask = norms >= 1.0

    if np.any(mask):
        Y = Y.copy()
        Y[mask] = Y[mask] / norms[mask, None] * (1.0 - 1e-12)

    return Y


def hydra_plus_objective_and_grad(
    x_flat: np.ndarray,
    D: np.ndarray,
    kappa: float,
    pairs: tuple[np.ndarray, np.ndarray],
    d: int,
    weights=None,
    eps: float = 1e-12,
) -> tuple[float, np.ndarray]:
    D = np.asarray(D, dtype=float)
    x = np.asarray(x_flat, dtype=float).reshape(D.shape[0], d)

    i, j = pairs

    xi = x[i]
    xj = x[j]

    ui = np.sqrt(1.0 + np.sum(xi * xi, axis=1))
    uj = np.sqrt(1.0 + np.sum(xj * xj, axis=1))

    dot = np.sum(xi * xj, axis=1)
    z = ui * uj - dot
    z = np.maximum(z, 1.0 + 1e-12)

    d_hat = np.arccosh(z) / np.sqrt(kappa)
    d_true = D[i, j]
    e = d_hat - d_true

    if weights is None:
        w = 1.0
    else:
        w = weights(d_true)

    f = float(np.sum(w * e * e))

    denom = np.sqrt(np.maximum(z * z - 1.0, eps))
    dd_dz = (1.0 / np.sqrt(kappa)) * (1.0 / denom)

    xi_over_ui = xi / np.maximum(ui[:, None], 1e-15)
    xj_over_uj = xj / np.maximum(uj[:, None], 1e-15)

    dz_dxi = uj[:, None] * xi_over_ui - xj
    dz_dxj = ui[:, None] * xj_over_uj - xi

    coef = 2.0 * w * e * dd_dz

    grad = np.zeros_like(x)
    np.add.at(grad, i, coef[:, None] * dz_dxi)
    np.add.at(grad, j, coef[:, None] * dz_dxj)

    return f, grad.reshape(-1)


def hydra_plus_refine(
    D: np.ndarray,
    Y0: np.ndarray,
    kappa: float,
    pairs: tuple[np.ndarray, np.ndarray] | None = None,
    maxiter: int = 200,
    gtol: float = 1e-6,
    seed: int | None = 0,
    weights=None,
    jitter: float = 1e-4,
    verbose: bool = False,
) -> tuple[np.ndarray, dict[str, object]]:
    D = np.asarray(D, dtype=float)
    Y0 = np.asarray(Y0, dtype=float)

    n, d = Y0.shape

    if pairs is None:
        pairs = sample_pairs(
            n,
            max_pairs=min(50000, n * (n - 1) // 2),
            random_state=seed,
        )

    x0 = poincare_to_hyperboloid_spatial(Y0)

    rng = np.random.default_rng(seed)
    x0 = x0 + jitter * rng.normal(size=x0.shape)

    x0_flat = x0.reshape(-1)

    def fun(x_flat: np.ndarray) -> float:
        f, _ = hydra_plus_objective_and_grad(
            x_flat,
            D,
            kappa,
            pairs,
            d,
            weights=weights,
        )
        return f

    def jac(x_flat: np.ndarray) -> np.ndarray:
        _, g = hydra_plus_objective_and_grad(
            x_flat,
            D,
            kappa,
            pairs,
            d,
            weights=weights,
        )
        return g

    res = minimize(
        fun,
        x0_flat,
        jac=jac,
        method="L-BFGS-B",
        options={
            "maxiter": int(maxiter),
            "gtol": float(gtol),
        },
    )

    x_opt = res.x.reshape(n, d)
    Y_opt = hyperboloid_spatial_to_poincare(x_opt)

    info = {
        "success": bool(res.success),
        "status": int(res.status),
        "message": str(res.message),
        "nit": int(res.nit),
        "final_obj": float(res.fun),
        "max_norm_Y": float(np.max(np.linalg.norm(Y_opt, axis=1))),
    }

    if verbose:
        print("hydra_plus_refine:", info)

    return Y_opt, info


def _normalize_rows(U: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    U = np.asarray(U, dtype=float)
    norms = np.linalg.norm(U, axis=1, keepdims=True)
    return U / np.maximum(norms, eps)


def _sphere_sse_and_riemannian_grad(
    D: np.ndarray,
    U: np.ndarray,
    R: float,
    pairs: tuple[np.ndarray, np.ndarray],
    weights=None,
    dot_eps: float = 1e-15,
    denom_eps: float = 1e-15,
) -> tuple[float, np.ndarray]:
    D = np.asarray(D, dtype=float)
    U = np.asarray(U, dtype=float)

    i, j = pairs

    ui = U[i]
    uj = U[j]

    dots = np.sum(ui * uj, axis=1)
    dots = np.clip(dots, -1.0 + dot_eps, 1.0 - dot_eps)

    d_hat = R * np.arccos(dots)
    d_true = D[i, j]
    e = d_hat - d_true

    if weights is None:
        w = 1.0
    else:
        w = weights(d_true)

    sse = float(np.sum(w * e * e))

    denom = np.sqrt(np.maximum(1.0 - dots * dots, denom_eps))
    ddot = -R / denom
    coef = 2.0 * w * e * ddot

    grad = np.zeros_like(U)
    np.add.at(grad, i, coef[:, None] * uj)
    np.add.at(grad, j, coef[:, None] * ui)

    proj = np.sum(grad * U, axis=1, keepdims=True)
    grad_R = grad - proj * U

    return sse, grad_R


def _sphere_sse_only(
    D: np.ndarray,
    U: np.ndarray,
    R: float,
    pairs: tuple[np.ndarray, np.ndarray],
    weights=None,
    dot_eps: float = 1e-15,
) -> float:
    D = np.asarray(D, dtype=float)
    U = np.asarray(U, dtype=float)

    i, j = pairs

    ui = U[i]
    uj = U[j]

    dots = np.sum(ui * uj, axis=1)
    dots = np.clip(dots, -1.0 + dot_eps, 1.0 - dot_eps)

    d_hat = R * np.arccos(dots)
    d_true = D[i, j]
    e = d_hat - d_true

    if weights is None:
        w = 1.0
    else:
        w = weights(d_true)

    return float(np.sum(w * e * e))


def sphere_plus_refine(
    D: np.ndarray,
    U0: np.ndarray,
    R: float,
    pairs: tuple[np.ndarray, np.ndarray] | None = None,
    pairs_m: int = 50000,
    seed: int | None = 0,
    maxiter: int = 300,
    gtol: float = 1e-6,
    step0: float = 0.5,
    line_search: bool = True,
    ls_c: float = 1e-4,
    ls_beta: float = 0.5,
    ls_max_steps: int = 25,
    weights=None,
    dot_eps: float = 1e-15,
    denom_eps: float = 1e-15,
    early_stop_sse: float = 1e-9,
    verbose: bool = False,
) -> tuple[np.ndarray, dict[str, object]]:
    D = np.asarray(D, dtype=float)
    U = _normalize_rows(U0)

    n = U.shape[0]

    if pairs is None:
        if n <= 400:
            i, j = np.triu_indices(n, k=1)
            pairs = (i, j)
        else:
            pairs = sample_pairs(
                n,
                max_pairs=int(pairs_m),
                random_state=seed,
            )

    sse, grad_R = _sphere_sse_and_riemannian_grad(
        D,
        U,
        R,
        pairs,
        weights=weights,
        dot_eps=dot_eps,
        denom_eps=denom_eps,
    )

    if sse <= early_stop_sse:
        info = {
            "success": True,
            "nit": 0,
            "final_sse": float(sse),
            "final_gnorm": 0.0,
            "max_norm_U": float(np.max(np.linalg.norm(U, axis=1))),
            "min_norm_U": float(np.min(np.linalg.norm(U, axis=1))),
            "note": "early_stop_sse triggered",
        }
        return U, info

    it = 0

    for it in range(1, int(maxiter) + 1):
        gnorm = float(np.sqrt(np.mean(np.sum(grad_R * grad_R, axis=1))))

        if verbose and (it == 1 or it % 25 == 0):
            print(f"iter {it:4d} | sse={sse:.6g} | gnorm={gnorm:.3e}")

        if gnorm < gtol:
            break

        direction = -grad_R
        dir_norm_sq = float(np.sum(direction * direction))
        alpha = float(step0)

        if line_search:
            accepted = False

            for _ in range(int(ls_max_steps)):
                U_try = _normalize_rows(U + alpha * direction)

                sse_try = _sphere_sse_only(
                    D,
                    U_try,
                    R,
                    pairs,
                    weights=weights,
                    dot_eps=dot_eps,
                )

                if sse_try <= sse - ls_c * alpha * dir_norm_sq:
                    U = U_try
                    sse = sse_try
                    accepted = True
                    break

                alpha *= ls_beta

            if not accepted:
                U = _normalize_rows(U + alpha * direction)

        else:
            U = _normalize_rows(U + alpha * direction)

        sse, grad_R = _sphere_sse_and_riemannian_grad(
            D,
            U,
            R,
            pairs,
            weights=weights,
            dot_eps=dot_eps,
            denom_eps=denom_eps,
        )

    final_gnorm = float(np.sqrt(np.mean(np.sum(grad_R * grad_R, axis=1))))

    info = {
        "success": bool(it < maxiter),
        "nit": int(it),
        "final_sse": float(sse),
        "final_gnorm": final_gnorm,
        "max_norm_U": float(np.max(np.linalg.norm(U, axis=1))),
        "min_norm_U": float(np.min(np.linalg.norm(U, axis=1))),
    }

    return U, info