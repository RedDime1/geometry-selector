from __future__ import annotations

import numpy as np

from geomselect.result import GeometryCandidate, SelectionResult


def _import_plotly():
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "Plotly is required for visualization. Install it with: pip install -e '.[viz]'"
        ) from exc

    return go


def _get_candidate(
    result_or_candidate: SelectionResult | GeometryCandidate,
    geometry: str | None = None,
) -> GeometryCandidate:
    if isinstance(result_or_candidate, GeometryCandidate):
        if geometry is not None and result_or_candidate.geometry != geometry:
            raise ValueError("The provided candidate does not match the requested geometry.")
        return result_or_candidate

    if isinstance(result_or_candidate, SelectionResult):
        if geometry is None:
            return result_or_candidate.selected

        for candidate in result_or_candidate.candidates:
            if candidate.geometry == geometry:
                return candidate

        raise ValueError(f"Geometry {geometry!r} is not present in the selection result.")

    raise TypeError("Expected SelectionResult or GeometryCandidate.")


def _make_marker(labels=None, marker_size: int = 7) -> dict:
    marker = {
        "size": marker_size,
    }

    if labels is None:
        return marker

    labels_array = np.asarray(labels)

    if np.issubdtype(labels_array.dtype, np.number):
        marker["color"] = labels_array
        marker["showscale"] = True

    return marker


def _make_text(labels=None):
    if labels is None:
        return None

    return [str(item) for item in labels]


def _plot_euclidean(
    candidate: GeometryCandidate,
    labels=None,
    title: str | None = None,
    marker_size: int = 7,
):
    go = _import_plotly()

    X = np.asarray(candidate.embedding, dtype=float)

    if X.ndim != 2:
        raise ValueError("Euclidean embedding must be a 2D array.")

    if X.shape[1] == 1:
        X = np.column_stack([X[:, 0], np.zeros(X.shape[0])])

    if X.shape[1] < 2:
        raise ValueError("Euclidean embedding must have at least one coordinate.")

    if title is None:
        title = f"Euclidean embedding, stress={candidate.stress:.6g}"

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=X[:, 0],
            y=X[:, 1],
            mode="markers",
            marker=_make_marker(labels, marker_size),
            text=_make_text(labels),
            name="points",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="x1",
        yaxis_title="x2",
        width=700,
        height=600,
    )

    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )

    return fig


def _plot_poincare_disk(
    candidate: GeometryCandidate,
    labels=None,
    title: str | None = None,
    marker_size: int = 7,
):
    go = _import_plotly()

    Y = np.asarray(candidate.embedding, dtype=float)

    if Y.ndim != 2 or Y.shape[1] != 2:
        raise ValueError("Poincare disk visualization is available only for d=2 embeddings.")

    if title is None:
        kappa = candidate.parameter_value
        title = f"Poincare disk, kappa={kappa:.6g}, stress={candidate.stress:.6g}"

    theta = np.linspace(0.0, 2.0 * np.pi, 400)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=np.cos(theta),
            y=np.sin(theta),
            mode="lines",
            name="unit disk",
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=Y[:, 0],
            y=Y[:, 1],
            mode="markers",
            marker=_make_marker(labels, marker_size),
            text=_make_text(labels),
            name="points",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="y1",
        yaxis_title="y2",
        width=700,
        height=700,
    )

    fig.update_xaxes(range=[-1.05, 1.05])

    fig.update_yaxes(
        range=[-1.05, 1.05],
        scaleanchor="x",
        scaleratio=1,
    )

    return fig


def _poincare_to_hyperboloid(Y: np.ndarray, kappa: float) -> np.ndarray:
    Y = np.asarray(Y, dtype=float)

    r2 = np.sum(Y * Y, axis=1, keepdims=True)
    denom = np.maximum(1.0 - r2, 1e-15)

    x0 = (1.0 + r2) / denom
    xs = 2.0 * Y / denom

    scale = 1.0 / np.sqrt(max(float(kappa), 1e-15))

    return scale * np.column_stack([x0[:, 0], xs[:, 0], xs[:, 1]])


def _plot_hyperboloid(
    candidate: GeometryCandidate,
    labels=None,
    title: str | None = None,
    marker_size: int = 5,
    show_surface: bool = True,
):
    go = _import_plotly()

    Y = np.asarray(candidate.embedding, dtype=float)

    if Y.ndim != 2 or Y.shape[1] != 2:
        raise ValueError("Hyperboloid visualization is available only for d=2 Poincare embeddings.")

    kappa = float(candidate.parameter_value)
    H = _poincare_to_hyperboloid(Y, kappa=kappa)

    if title is None:
        title = f"Hyperboloid model, kappa={kappa:.6g}, stress={candidate.stress:.6g}"

    fig = go.Figure()

    if show_surface:
        spatial = H[:, 1:3]
        max_rho = float(np.max(np.linalg.norm(spatial, axis=1)))
        max_rho = max(max_rho * 1.15, 1.0 / np.sqrt(max(kappa, 1e-15)))

        rho = np.linspace(0.0, max_rho, 40)
        theta = np.linspace(0.0, 2.0 * np.pi, 80)

        rr, tt = np.meshgrid(rho, theta)

        x1 = rr * np.cos(tt)
        x2 = rr * np.sin(tt)

        radius = 1.0 / np.sqrt(max(kappa, 1e-15))
        x0 = np.sqrt(radius * radius + x1 * x1 + x2 * x2)

        fig.add_trace(
            go.Surface(
                x=x1,
                y=x2,
                z=x0,
                opacity=0.22,
                showscale=False,
                name="hyperboloid",
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scatter3d(
            x=H[:, 1],
            y=H[:, 2],
            z=H[:, 0],
            mode="markers",
            marker=_make_marker(labels, marker_size),
            text=_make_text(labels),
            name="points",
        )
    )

    fig.update_layout(
        title=title,
        width=750,
        height=700,
        scene={
            "xaxis_title": "x1",
            "yaxis_title": "x2",
            "zaxis_title": "x0",
            "aspectmode": "data",
        },
    )

    return fig


def _plot_hyperbolic(
    candidate: GeometryCandidate,
    labels=None,
    title: str | None = None,
    marker_size: int = 7,
    hyperbolic_model: str = "poincare",
    show_surface: bool = True,
):
    if hyperbolic_model == "poincare":
        return _plot_poincare_disk(
            candidate,
            labels=labels,
            title=title,
            marker_size=marker_size,
        )

    if hyperbolic_model == "hyperboloid":
        return _plot_hyperboloid(
            candidate,
            labels=labels,
            title=title,
            marker_size=marker_size,
            show_surface=show_surface,
        )

    raise ValueError("hyperbolic_model must be 'poincare' or 'hyperboloid'.")


def _sphere_radius_from_candidate(candidate: GeometryCandidate, X: np.ndarray) -> float:
    R = candidate.parameter_value

    try:
        R = float(R)
    except Exception:
        R = np.nan

    if np.isfinite(R) and R > 0:
        return R

    norms = np.linalg.norm(X, axis=1)
    positive = norms[norms > 0]

    if positive.size == 0:
        return 1.0

    return float(np.median(positive))


def _plot_spherical(
    candidate: GeometryCandidate,
    labels=None,
    title: str | None = None,
    marker_size: int = 5,
    show_surface: bool = True,
):
    go = _import_plotly()

    X = np.asarray(candidate.embedding, dtype=float)

    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("Spherical visualization is available only for d=2 sphere embeddings in R^3.")

    R = _sphere_radius_from_candidate(candidate, X)

    if title is None:
        title = f"Spherical embedding, R={R:.6g}, stress={candidate.stress:.6g}"

    fig = go.Figure()

    if show_surface:
        u = np.linspace(0.0, 2.0 * np.pi, 50)
        v = np.linspace(0.0, np.pi, 25)

        xs = R * np.outer(np.cos(u), np.sin(v))
        ys = R * np.outer(np.sin(u), np.sin(v))
        zs = R * np.outer(np.ones_like(u), np.cos(v))

        fig.add_trace(
            go.Surface(
                x=xs,
                y=ys,
                z=zs,
                opacity=0.18,
                showscale=False,
                name="sphere",
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scatter3d(
            x=X[:, 0],
            y=X[:, 1],
            z=X[:, 2],
            mode="markers",
            marker=_make_marker(labels, marker_size),
            text=_make_text(labels),
            name="points",
        )
    )

    fig.update_layout(
        title=title,
        width=750,
        height=700,
        scene={
            "xaxis_title": "x1",
            "yaxis_title": "x2",
            "zaxis_title": "x3",
            "aspectmode": "data",
        },
    )

    return fig


def plot_embedding(
    result_or_candidate: SelectionResult | GeometryCandidate,
    *,
    geometry: str | None = None,
    labels=None,
    title: str | None = None,
    marker_size: int = 7,
    show_surface: bool = True,
    show: bool = True,
    hyperbolic_model: str = "poincare",
):
    candidate = _get_candidate(
        result_or_candidate,
        geometry=geometry,
    )

    if candidate.geometry == "euclidean":
        fig = _plot_euclidean(
            candidate,
            labels=labels,
            title=title,
            marker_size=marker_size,
        )

    elif candidate.geometry == "hyperbolic":
        fig = _plot_hyperbolic(
            candidate,
            labels=labels,
            title=title,
            marker_size=marker_size,
            hyperbolic_model=hyperbolic_model,
            show_surface=show_surface,
        )

    elif candidate.geometry == "spherical":
        fig = _plot_spherical(
            candidate,
            labels=labels,
            title=title,
            marker_size=marker_size,
            show_surface=show_surface,
        )

    else:
        raise ValueError(f"Unsupported geometry: {candidate.geometry!r}")

    if show:
        fig.show()

    return fig