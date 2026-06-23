from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GraspPlan:
    grasps: np.ndarray
    scores: np.ndarray
    contact_points: np.ndarray

    @property
    def empty(self) -> bool:
        return self.grasps.size == 0

    @property
    def best_grasp(self) -> np.ndarray | None:
        if self.empty:
            return None
        best_idx = int(np.argmax(self.scores)) if self.scores.size else 0
        return self.grasps[best_idx]

