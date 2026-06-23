from __future__ import annotations

import os

import numpy as np

from agentic_robots.services.base import ServiceConfig, post_json
from agentic_robots.services.motion.pyroki.types import IkSolution, MotionPlan


DEFAULT_PYROKI_URL = os.environ.get("PYROKI_SERVICE_URL", "http://127.0.0.1:8116")


class PyrokiClient:
    """HTTP client for PyRoKi IK and pose-to-pose planning."""

    def __init__(self, url: str = DEFAULT_PYROKI_URL, timeout_s: float = 60.0):
        self.config = ServiceConfig(url=url, timeout_s=timeout_s)

    def solve_ik(
        self,
        target_pose_wxyz_xyz: np.ndarray,
        prev_cfg: np.ndarray | None = None,
    ) -> IkSolution:
        payload = {
            "target_pose_wxyz_xyz": np.asarray(target_pose_wxyz_xyz, dtype=float).tolist(),
            "prev_cfg": None if prev_cfg is None else np.asarray(prev_cfg, dtype=float).tolist(),
        }
        data = post_json(self.config, "/ik", payload)
        return IkSolution(joint_positions=np.asarray(data["joint_positions"], dtype=np.float32))

    def plan(
        self,
        start_pose_wxyz_xyz: np.ndarray,
        end_pose_wxyz_xyz: np.ndarray,
        *,
        timesteps: int = 20,
        dt: float = 0.02,
    ) -> MotionPlan:
        payload = {
            "start_pose_wxyz_xyz": np.asarray(start_pose_wxyz_xyz, dtype=float).tolist(),
            "end_pose_wxyz_xyz": np.asarray(end_pose_wxyz_xyz, dtype=float).tolist(),
            "timesteps": timesteps,
            "dt": dt,
        }
        data = post_json(self.config, "/plan", payload)
        return MotionPlan(
            waypoints=np.asarray(data["waypoints"], dtype=np.float32),
            dt=float(data["dt"]),
        )
