from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IkSolution:
    joint_positions: np.ndarray


@dataclass(frozen=True)
class MotionPlan:
    waypoints: np.ndarray
    dt: float
