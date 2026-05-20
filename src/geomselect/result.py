from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True, eq=False)
class GeometryCandidate:
    geometry: str
    stress: float
    embedding: np.ndarray | None = None
    parameter_name: str | None = None
    parameter_value: float | None = None
    selected: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self, *, include_embedding: bool = False) -> dict[str, Any]:
        row = {
            "geometry": self.geometry,
            "stress": self.stress,
            "parameter_name": self.parameter_name,
            "parameter_value": self.parameter_value,
            "selected": self.selected,
        }

        if include_embedding:
            row["embedding"] = self.embedding

        return row


@dataclass(slots=True, eq=False)
class GeometrySelectorConfig:
    d: int = 2
    geometries: tuple[str, ...] = ("euclidean", "hyperbolic", "spherical")
    pair_sample: int | None = None
    random_state: int | None = None
    norm_method: str | None = "median"
    hyper_grid_num: int = 31
    sphere_grid_num: int = 31
    hyper_span_decades: float = 3.0
    sphere_span_decades: float = 3.0
    center: float = 1.0
    hyper_t_max: float = 20.0
    sphere_r_max_factor: float = 6.0
    sphere_r_max_abs: float | None = None
    n_refine: int = 3
    refine_num: int = 25
    eig_tol: float = 1e-10
    close_ratio: float = 1.05
    close_abs: float = 1e-3
    fail_fast: bool = False
    do_plus: bool = False
    plus_pair_sample: int | None = None
    plus_maxiter_hyper: int = 200
    plus_maxiter_sphere: int = 300
    plus_gtol: float = 1e-6
    rollback_plus: bool = True
    flat_signal_threshold: float = 0.15
    weak_signal_threshold: float = 0.35
    euclidean_flat_close_ratio: float = 1.10
    euclidean_flat_close_abs: float = 1e-3
    curvature_quantile: float = 0.90


@dataclass(slots=True, eq=False)
class SelectionResult:
    selected_geometry: str
    selected: GeometryCandidate
    candidates: tuple[GeometryCandidate, ...]
    candidate_table: pd.DataFrame
    recommendation: str
    config: GeometrySelectorConfig
    metadata: dict[str, Any] = field(default_factory=dict)