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
    geometries: tuple[str, ...] = ("euclidean",)
    pair_sample: int | None = None
    random_state: int | None = None


@dataclass(slots=True, eq=False)
class SelectionResult:
    selected_geometry: str
    selected: GeometryCandidate
    candidates: tuple[GeometryCandidate, ...]
    candidate_table: pd.DataFrame
    recommendation: str
    config: GeometrySelectorConfig
    metadata: dict[str, Any] = field(default_factory=dict)