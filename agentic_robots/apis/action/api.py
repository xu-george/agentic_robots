from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from agentic_robots.apis.base import ApiBase
from agentic_robots.envs.pybullet import StackingEnv


class ActionApi(ApiBase):
    """Low-level robot action API backed by the PyBullet FrankaSim robot."""

    def __init__( 
        self,
        env: StackingEnv,
        *,
        grasp_distance_m: float = 0.12,
    ) -> None:
        self.env = env
        self.grasp_distance_m = grasp_distance_m
        self.last_grasp: dict[str, Any] | None = None

    def functions(self) -> dict[str, Callable[..., Any]]:
        return {
            "get_end_effector_pose": self.get_end_effector_pose,
            "move_to_pose": self.move_to_pose,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
        }

    def get_end_effector_pose(self):
        """Return the current end-effector pose with position and quaternion_wxyz fields."""
        self._require_robot()
        return self.env.robot.get_end_effector_pose()

    def move_to_pose(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
        *,
        plan_steps: int = 40,
        sim_steps_per_waypoint: int = 4,
    ) -> np.ndarray:
        """Move the gripper to a target world pose using FrankaSim joint interpolation."""
        self._require_robot()
        trajectory = self.env.robot.plan_to_pose(position, quaternion_wxyz, steps=plan_steps)
        for waypoint in trajectory:
            self.env.robot.move_to_joints(waypoint, steps=sim_steps_per_waypoint)
            self.env._record_frame()
        return trajectory

    def open_gripper(self, *, steps: int = 60) -> None:
        """Open the Franka gripper."""
        self._require_robot()
        self.env.robot.open_gripper(steps=steps)
        self.env.step_sim(60)

    def close_gripper(self, *, steps: int = 60) -> None:
        """Close the Franka gripper and record nearest-object diagnostics."""
        self._require_robot()
        self.env.robot.close_gripper(steps=steps)
        self._record_nearest_object()
        self.env.step_sim(24)

    def _record_nearest_object(self) -> None:
        eef_position = self.env.robot.get_end_effector_pose().position
        nearest_name = None
        nearest_distance = float("inf")
        nearest_position = None
        for name in self.env.object_ids:
            object_position = self.env.get_object_position(name)
            distance = float(np.linalg.norm(object_position - eef_position))
            if distance < nearest_distance:
                nearest_name = name
                nearest_distance = distance
                nearest_position = object_position

        self.last_grasp = {
            "nearest_object": nearest_name,
            "nearest_distance_m": round(nearest_distance, 4),
            "grasp_distance_m": self.grasp_distance_m,
            "eef_position": eef_position.round(4).tolist(),
            "nearest_object_position": (
                nearest_position.round(4).tolist() if nearest_position is not None else None
            ),
        }

    def _require_robot(self) -> None:
        if self.env.robot is None:
            raise RuntimeError("StackingEnv.reset() must be called before using ActionApi")
