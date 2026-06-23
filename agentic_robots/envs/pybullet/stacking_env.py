from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pybullet as p
import pybullet_data

from agentic_robots.envs.pybullet.franka_sim import FrankaSim, _xyzw_to_wxyz


@dataclass(frozen=True)
class CameraConfig:
    width: int = 320
    height: int = 240
    fov_deg: float = 55.0
    near: float = 0.02
    far: float = 3.0
    eye: tuple[float, float, float] = (0.65, -0.75, 0.65)
    target: tuple[float, float, float] = (0.45, 0.0, 0.08)
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)


@dataclass
class CameraInfo:
    intrinsics: np.ndarray
    extrinsics: np.ndarray
    view_matrix: list[float] | None = None
    projection_matrix: list[float] | None = None


class StackingEnv:
    """Minimal PyBullet stacking task with ground-truth perception values."""

    def __init__(
        self,
        *,
        gui: bool = False,
        camera: CameraConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self.gui = gui
        self.camera = camera or CameraConfig()
        self.rng = np.random.default_rng(seed)
        self.client_id: int | None = None
        self.robot: FrankaSim | None = None
        self.object_ids: dict[str, int] = {}
        self.table_id: int | None = None
        self.held_object: str | None = None
        self._video_writer: cv2.VideoWriter | None = None
        self._video_path: Path | None = None
        self._video_fps = 24.0

        self.camera_info = CameraInfo(
            intrinsics=self._compute_intrinsics(),
            extrinsics=np.eye(4, dtype=np.float64),
        )

    def reset(self) -> dict[str, Any]:
        """Create the PyBullet world, robot, table, and stacking cubes."""
        self.close()
        self.client_id = p.connect(p.GUI if self.gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        p.resetSimulation(physicsClientId=self.client_id)
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=self.client_id)
        p.setTimeStep(1.0 / 240.0, physicsClientId=self.client_id)

        p.loadURDF("plane.urdf", physicsClientId=self.client_id)
        self.table_id = self._create_box(
            name="table",
            half_extents=(0.45, 0.35, 0.025),
            position=(0.45, 0.0, 0.025),
            rgba=(0.55, 0.45, 0.35, 1.0),
        )

        red_position, blue_position = self._sample_cube_positions()
        self.object_ids = {
            "red_cube": self._create_box(
                name="red_cube",
                half_extents=(0.025, 0.025, 0.025),
                position=red_position,
                rgba=(0.9, 0.1, 0.1, 1.0),
            ),
            "blue_cube": self._create_box(
                name="blue_cube",
                half_extents=(0.025, 0.025, 0.025),
                position=blue_position,
                rgba=(0.1, 0.25, 0.9, 1.0),
            ),
        }

        self.robot = FrankaSim(client_id=self.client_id, base_position=(0.0, 0.0, 0.0))
        self.robot.load()
        self._update_camera_matrices()
        self.step_sim(60)
        return self.get_obs()

    def step_sim(self, steps: int = 1) -> None:
        """Step the simulation."""
        self._require_client()
        for _ in range(max(0, int(steps))):
            p.stepSimulation(physicsClientId=self.client_id)
        self._record_frame()

    def get_obs(self) -> dict[str, Any]:
        """Return image observation, robot state, and ground-truth perception values.

        Camera matrices are intentionally kept on the environment as
        `camera_info`, instead of being nested inside obs.
        """
        self._require_client()
        self._require_robot()
        images = self.render_images()
        eef_pose = self.robot.get_end_effector_pose()
        return {
            "images": images,
            "robot": {
                "joint_positions": self.robot.get_joint_positions(),
                "gripper_positions": self.robot.get_gripper_positions(),
                "eef_position": eef_pose.position,
                "eef_quaternion_wxyz": eef_pose.quaternion_wxyz,
            },
            "ground_truth": {
                "objects": self.get_ground_truth(),
            },
        }

    def render_images(self) -> dict[str, np.ndarray]:
        """Render RGB, depth, and segmentation images from the task camera."""
        self._require_client()
        self._update_camera_matrices()
        _, _, rgb, depth_buffer, segmentation = p.getCameraImage(
            width=self.camera.width,
            height=self.camera.height,
            viewMatrix=self.camera_info.view_matrix,
            projectionMatrix=self.camera_info.projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=self.client_id,
        )
        rgb_array = np.asarray(rgb, dtype=np.uint8).reshape(
            self.camera.height,
            self.camera.width,
            4,
        )[:, :, :3]
        depth = self._depth_buffer_to_meters(np.asarray(depth_buffer, dtype=np.float64))
        segmentation_array = np.asarray(segmentation, dtype=np.int32).reshape(
            self.camera.height,
            self.camera.width,
        )
        return {
            "rgb": rgb_array,
            "depth": depth,
            "segmentation": segmentation_array,
        }

    def start_video_recording(self, path: str | Path, *, fps: float = 24.0) -> Path:
        """Record task-camera RGB frames to an MP4 file."""
        self._require_client()
        self.stop_video_recording()
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._video_writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (self.camera.width, self.camera.height),
        )
        if not self._video_writer.isOpened():
            self._video_writer = None
            raise RuntimeError(f"Could not open video writer for {output_path}")
        self._video_path = output_path
        self._video_fps = fps
        self._record_frame()
        return output_path

    def stop_video_recording(self) -> Path | None:
        """Stop recording and return the video path, if recording was active."""
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
        return self._video_path

    def get_ground_truth(self) -> dict[str, dict[str, Any]]:
        """Return ground-truth object states for fake perception APIs."""
        self._require_client()
        result: dict[str, dict[str, Any]] = {}
        for name, body_id in self.object_ids.items():
            position, quaternion_xyzw = p.getBasePositionAndOrientation(
                body_id,
                physicsClientId=self.client_id,
            )
            result[name] = {
                "body_id": body_id,
                "position": np.asarray(position, dtype=np.float64),
                "quaternion_wxyz": _xyzw_to_wxyz(np.asarray(quaternion_xyzw, dtype=np.float64)),
            }
        return result

    def get_object_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return a ground-truth object pose as position and wxyz quaternion."""
        state = self.get_ground_truth()[name]
        return state["position"], state["quaternion_wxyz"]

    def get_object_position(self, name: str) -> np.ndarray:
        """Return a ground-truth object XYZ position."""
        return self.get_object_pose(name)[0]

    def is_success(self) -> bool:
        """Check whether red cube is stacked on blue cube."""
        red = self.get_object_position("red_cube")
        blue = self.get_object_position("blue_cube")
        xy_close = np.linalg.norm(red[:2] - blue[:2]) < 0.04
        z_stacked = red[2] > blue[2] + 0.035
        return bool(xy_close and z_stacked)

    def close(self) -> None:
        """Disconnect from PyBullet if connected."""
        if self.client_id is not None and p.isConnected(self.client_id):
            self.stop_video_recording()
            p.disconnect(physicsClientId=self.client_id)
        self.client_id = None
        self.robot = None
        self.object_ids = {}
        self.table_id = None
        self.held_object = None
        self._video_path = None

    def _create_box(
        self,
        *,
        name: str,
        half_extents: tuple[float, float, float],
        position: tuple[float, float, float],
        rgba: tuple[float, float, float, float],
    ) -> int:
        self._require_client()
        collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            physicsClientId=self.client_id,
        )
        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=rgba,
            physicsClientId=self.client_id,
        )
        body_id = p.createMultiBody(
            baseMass=0.0 if name == "table" else 0.05,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=position,
            physicsClientId=self.client_id,
        )
        p.changeDynamics(
            body_id,
            -1,
            lateralFriction=1.0,
            spinningFriction=0.01,
            rollingFriction=0.01,
            physicsClientId=self.client_id,
        )
        return body_id

    def _sample_cube_positions(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Sample non-overlapping cube centers in the Panda's reachable table workspace."""
        x_range = (0.36, 0.58)
        y_range = (-0.18, 0.18)
        z = 0.075
        min_separation = 0.12

        for _ in range(100):
            red_xy = self.rng.uniform((x_range[0], y_range[0]), (x_range[1], y_range[1]))
            blue_xy = self.rng.uniform((x_range[0], y_range[0]), (x_range[1], y_range[1]))
            if np.linalg.norm(red_xy - blue_xy) >= min_separation:
                return (
                    (float(red_xy[0]), float(red_xy[1]), z),
                    (float(blue_xy[0]), float(blue_xy[1]), z),
                )

        # Extremely unlikely, but keep reset deterministic and valid if random sampling misses.
        return (0.42, -0.08, z), (0.50, 0.08, z)

    def _compute_intrinsics(self) -> np.ndarray:
        f = 0.5 * self.camera.height / np.tan(0.5 * np.deg2rad(self.camera.fov_deg))
        return np.array(
            [
                [f, 0.0, self.camera.width / 2.0],
                [0.0, f, self.camera.height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def _update_camera_matrices(self) -> None:
        self.camera_info.view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.camera.eye,
            cameraTargetPosition=self.camera.target,
            cameraUpVector=self.camera.up,
        )
        self.camera_info.projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.camera.fov_deg,
            aspect=self.camera.width / self.camera.height,
            nearVal=self.camera.near,
            farVal=self.camera.far,
        )
        self.camera_info.extrinsics = _camera_pose_from_eye_target_up(
            np.asarray(self.camera.eye, dtype=np.float64),
            np.asarray(self.camera.target, dtype=np.float64),
            np.asarray(self.camera.up, dtype=np.float64),
        )

    def _depth_buffer_to_meters(self, depth_buffer: np.ndarray) -> np.ndarray:
        near = self.camera.near
        far = self.camera.far
        return far * near / (far - (far - near) * depth_buffer)

    def _move_to_pose_recorded(self, position: np.ndarray, quaternion_wxyz: np.ndarray) -> None:
        self._require_robot()
        trajectory = self.robot.plan_to_pose(position, quaternion_wxyz, steps=40)
        for waypoint in trajectory:
            self.robot.move_to_joints(waypoint, steps=4)
            self._record_frame()

    def _record_frame(self) -> None:
        if self._video_writer is None:
            return
        rgb = self.render_images()["rgb"]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self._video_writer.write(bgr)

    def _require_client(self) -> None:
        if self.client_id is None or not p.isConnected(self.client_id):
            raise RuntimeError("StackingEnv.reset() must be called before using the env")

    def _require_robot(self) -> None:
        if self.robot is None:
            raise RuntimeError("FrankaSim robot has not been created")


def _camera_pose_from_eye_target_up(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray,
) -> np.ndarray:
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)

    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose
