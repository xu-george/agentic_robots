from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from agentic_robots.envs.real.franka import FrankaReal
from agentic_robots.envs.real.realsense_camera import RealSenseCamera


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CALIBRATION = PROJECT_ROOT / "configs" / "robot" / "realsense_d455_franka_calibration.yaml"


class RealObjectLiftEnv:
    """Real tabletop lift env: Franka action backend plus calibrated RealSense RGB-D."""

    def __init__(
        self,
        *,
        object_prompt: str,
        robot: FrankaReal | None = None,
        camera: RealSenseCamera | None = None,
        calibration_path: str | Path = DEFAULT_CALIBRATION,
        motion_enabled: bool = False,
        move_home_on_reset: bool = False,
    ) -> None:
        self.object_prompt = object_prompt
        self.motion_enabled = motion_enabled
        self.move_home_on_reset = move_home_on_reset
        self.calibration_path = Path(calibration_path)
        self.calibration = _load_yaml(self.calibration_path)
        self.intrinsics = np.asarray(self.calibration["intrinsics"], dtype=np.float64).reshape(3, 3)
        self.camera_pose = np.asarray(
            self.calibration["camera_pose"]["matrix"],
            dtype=np.float64,
        ).reshape(4, 4)
        camera_cfg = self.calibration["camera"]
        self.camera = camera or RealSenseCamera(
            width=int(camera_cfg["width"]),
            height=int(camera_cfg["height"]),
            serial=str(camera_cfg["serial"]) if camera_cfg.get("serial") else None,
        )
        self.robot = robot or FrankaReal(motion_enabled=motion_enabled)

    def reset(self) -> dict[str, Any]:
        self.robot.load()
        if self.motion_enabled and self.move_home_on_reset:
            self.robot.move_home()
        return self.get_obs()

    def get_obs(self) -> dict[str, Any]:
        frame = self.camera.capture()
        pose = self.robot.get_end_effector_pose()
        gripper = self.robot.read_gripper()
        return {
            "rgb": frame.rgb,
            "depth": frame.depth_m,
            "intrinsics": self.intrinsics,
            "camera_pose": self.camera_pose,
            "images": {"rgb": frame.rgb, "depth": frame.depth_m},
            "robot": {
                "eef_position": pose.position,
                "eef_quaternion_wxyz": pose.quaternion_wxyz,
                "gripper": gripper,
                "gripper_width": np.array([gripper.width], dtype=np.float64),
            },
            "ground_truth": {"objects": {}},
        }

    def is_success(self) -> bool:
        return bool(self.robot.read_gripper().is_grasped)

    def close(self) -> None:
        self.camera.stop()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
