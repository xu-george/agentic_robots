from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Sam3Mask:
    mask: np.ndarray
    box: tuple[float, float, float, float]
    score: float
    label: str


@dataclass(frozen=True)
class PointPromptResult:
    mask: np.ndarray
    score: float

