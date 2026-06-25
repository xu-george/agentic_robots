from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from agentic_robots.apis.base import ApiBase

# register the ActionApi class in the agentic_robots.apis.action module so that it can be imported from there
class ActionApi(ApiBase):
    """Low-level action API backed directly by a robot object."""

    def __init__(self, robot: Any) -> None:
        self.robot = robot

    def functions(self) -> dict[str, Callable[..., Any]]:
        return {
            "get_end_effector_pose": self.get_end_effector_pose,
            "move_to_pose": self.move_to_pose,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
        }

    # tcp pose functions
    def get_end_effector_pose(self):
        """Return the current end-effector pose with position and quaternion_wxyz fields."""
        self._require_robot()
        return self.robot.get_end_effector_pose()

    def move_to_pose(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Move the gripper to a target world pose."""
        self._require_robot()
        return self.robot.move_to_pose(position, quaternion_wxyz, **kwargs)

    def open_gripper(self, **kwargs) -> None:
        """Open the Franka gripper."""
        self._require_robot()
        self.robot.open_gripper(**kwargs)

    def close_gripper(self, **kwargs) -> None:
        """Close the Franka gripper."""
        self._require_robot()
        self.robot.close_gripper(**kwargs)

    def _require_robot(self) -> None:
        if self.robot is None:
            raise RuntimeError("A loaded robot must be provided before using ActionApi")
