from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pybullet as p
import pybullet_data
import yaml

from agentic_robots.envs.pybullet.franka_sim import FrankaSim, _xyzw_to_wxyz
from agentic_robots.envs.pybullet.stacking_env import (
    CameraConfig,
    CameraInfo,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OBJECT_CONFIG = PROJECT_ROOT / "configs" / "lift_objects.yaml"
DEFAULT_YCB_DIR = PROJECT_ROOT / "data" / "ycb"


@dataclass(frozen=True)
class LiftObjectConfig:
    name: str
    ycb_id: str
    mesh: str
    scale: float
    position: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    mass: float
    prompt: str
    camera: CameraConfig | None = None


class LiftEnv:
    """PyBullet tabletop lift task with one configurable YCB object."""

    def __init__(
        self,
        *,
        object_name: str = "banana",
        object_config_path: Path = DEFAULT_OBJECT_CONFIG,
        ycb_dir: Path = DEFAULT_YCB_DIR,
        gui: bool = False,
        camera: CameraConfig | None = None,
    ) -> None:
        self.object_name = object_name
        self.object_config_path = object_config_path
        self.ycb_dir = ycb_dir
        self.gui = gui
        self._camera_overridden = camera is not None
        self.camera = camera or CameraConfig(
            eye=(0.65, -0.75, 0.65),
            target=(0.48, 0.0, 0.08),
        )
        self.client_id: int | None = None
        self.robot: FrankaSim | None = None
        self.object_ids: dict[str, int] = {}
        self.table_id: int | None = None
        self.object_start_z: float | None = None
        self._video_writer: cv2.VideoWriter | None = None
        self._video_path: Path | None = None
        self._debug_item_ids: list[int] = []

        self.camera_info = CameraInfo(
            intrinsics=self._compute_intrinsics(),
            extrinsics=np.eye(4, dtype=np.float64),
        )

    def reset(self) -> dict[str, Any]:
        """Create the PyBullet world, robot, table, and configured YCB object."""
        config = self._load_object_config()
        if config.camera is not None and not self._camera_overridden:
            self.camera = config.camera
            self.camera_info = CameraInfo(
                intrinsics=self._compute_intrinsics(),
                extrinsics=np.eye(4, dtype=np.float64),
            )
        if self._can_reset_scene_in_place(config.name):
            self.clear_debug_items()
            self.robot.reset()
            self._reset_body_pose(body_id=self.object_ids[config.name], config=config)
            self.step_sim(120)
            self.object_start_z = float(self.get_object_position(config.name)[2])
            return self.get_obs()

        self.close()
        self.client_id = p.connect(p.GUI if self.gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        p.resetSimulation(physicsClientId=self.client_id)
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=self.client_id)
        p.setTimeStep(1.0 / 240.0, physicsClientId=self.client_id)

        p.loadURDF("plane.urdf", physicsClientId=self.client_id)
        self.table_id = self._create_table()
        body_id = self._create_ycb_object(config)
        self.object_ids = {config.name: body_id}

        self.robot = FrankaSim(
            client_id=self.client_id,
            base_position=(0.0, 0.0, 0.0),
            frame_recorder=self._record_frame,
        )
        self.robot.load()
        self._update_camera_matrices()
        self.step_sim(120)
        self.object_start_z = float(self.get_object_position(config.name)[2])
        return self.get_obs()

    def step_sim(self, steps: int = 1) -> None:
        self._require_client()
        for _ in range(max(0, int(steps))):
            p.stepSimulation(physicsClientId=self.client_id)
        self._record_frame()

    def get_obs(self) -> dict[str, Any]:
        self._require_client()
        self._require_robot()
        images = self.render_images()
        eef_pose = self.robot.get_end_effector_pose()
        return {
            "rgb": images["rgb"],
            "depth": images["depth"],
            "segmentation": images["segmentation"],
            "intrinsics": self.camera_info.intrinsics,
            "camera_pose": self.camera_info.extrinsics,
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
        return {"rgb": rgb_array, "depth": depth, "segmentation": segmentation_array}

    def start_video_recording(self, path: str | Path, *, fps: float = 24.0) -> Path:
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
        self._record_frame()
        return output_path

    def stop_video_recording(self) -> Path | None:
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
        return self._video_path

    def get_ground_truth(self) -> dict[str, dict[str, Any]]:
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
        state = self.get_ground_truth()[name]
        return state["position"], state["quaternion_wxyz"]

    def get_object_position(self, name: str) -> np.ndarray:
        return self.get_object_pose(name)[0]

    def is_success(self) -> bool:
        if self.object_start_z is None:
            return False
        position = self.get_object_position(self.object_name)
        return bool(position[2] > self.object_start_z + 0.08)

    def clear_debug_items(self) -> None:
        """Remove debug lines/text previously added by this environment."""
        self._require_client()
        for item_id in self._debug_item_ids:
            p.removeUserDebugItem(item_id, physicsClientId=self.client_id)
        self._debug_item_ids = []

    def close(self) -> None:
        if self.client_id is not None and p.isConnected(self.client_id):
            self.stop_video_recording()
            p.disconnect(physicsClientId=self.client_id)
        self.client_id = None
        self.robot = None
        self.object_ids = {}
        self.table_id = None
        self.object_start_z = None
        self._video_path = None
        self._debug_item_ids = []

    def _load_object_config(self) -> LiftObjectConfig:
        with self.object_config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        raw = data["objects"][self.object_name]
        return LiftObjectConfig(
            name=self.object_name,
            ycb_id=raw["ycb_id"],
            mesh=raw["mesh"],
            scale=float(raw.get("scale", 1.0)),
            position=tuple(raw["position"]),
            orientation_xyzw=tuple(raw.get("orientation_xyzw", [0.0, 0.0, 0.0, 1.0])),
            mass=float(raw.get("mass", 0.05)),
            prompt=raw.get("prompt", self.object_name),
            camera=CameraConfig(**raw["camera"]) if "camera" in raw else None,
        )

    def _create_table(self) -> int:
        collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=(0.45, 0.35, 0.025),
            physicsClientId=self.client_id,
        )
        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=(0.45, 0.35, 0.025),
            rgbaColor=(0.55, 0.45, 0.35, 1.0),
            physicsClientId=self.client_id,
        )
        return p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=(0.45, 0.0, 0.025),
            physicsClientId=self.client_id,
        )

    def _create_ycb_object(self, config: LiftObjectConfig) -> int:
        mesh_path = self.ycb_dir / config.ycb_id / config.mesh
        if not mesh_path.exists():
            raise FileNotFoundError(
                f"YCB mesh not found: {mesh_path}. Run scripts/download_ycb_objects.py first."
            )
        mesh_scale = [config.scale] * 3

        collision = p.createCollisionShape(
            p.GEOM_MESH,
            fileName=str(mesh_path),
            meshScale=mesh_scale,
            physicsClientId=self.client_id,
        )
        visual = p.createVisualShape(
            p.GEOM_MESH,
            fileName=str(mesh_path),
            meshScale=mesh_scale,
            physicsClientId=self.client_id,
        )
        body_id = p.createMultiBody(
            baseMass=config.mass,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=config.position,
            baseOrientation=config.orientation_xyzw,
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

    def _can_reset_scene_in_place(self, object_name: str) -> bool:
        return bool(
            self.gui
            and self.client_id is not None
            and p.isConnected(self.client_id)
            and self.robot is not None
            and object_name in self.object_ids
        )

    def _reset_body_pose(self, *, body_id: int, config: LiftObjectConfig) -> None:
        p.resetBasePositionAndOrientation(
            body_id,
            config.position,
            config.orientation_xyzw,
            physicsClientId=self.client_id,
        )
        p.resetBaseVelocity(
            body_id,
            linearVelocity=(0.0, 0.0, 0.0),
            angularVelocity=(0.0, 0.0, 0.0),
            physicsClientId=self.client_id,
        )

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
        self.camera_info.extrinsics = _camera_optical_pose_from_eye_target_up(
            np.asarray(self.camera.eye, dtype=np.float64),
            np.asarray(self.camera.target, dtype=np.float64),
            np.asarray(self.camera.up, dtype=np.float64),
        )

    def _depth_buffer_to_meters(self, depth_buffer: np.ndarray) -> np.ndarray:
        near = self.camera.near
        far = self.camera.far
        return far * near / (far - (far - near) * depth_buffer)

    def _record_frame(self) -> None:
        if self._video_writer is None:
            return
        rgb = self.render_images()["rgb"]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self._video_writer.write(bgr)

    def _require_client(self) -> None:
        if self.client_id is None or not p.isConnected(self.client_id):
            raise RuntimeError("LiftEnv.reset() must be called before using the env")

    def _require_robot(self) -> None:
        if self.robot is None:
            raise RuntimeError("FrankaSim robot has not been created")


def _camera_optical_pose_from_eye_target_up(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray,
) -> np.ndarray:
    """Return camera-to-world pose for pinhole points where +z looks forward and +y is down."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)

    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = right
    pose[:3, 1] = -true_up
    pose[:3, 2] = forward
    pose[:3, 3] = eye
    return pose


