from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml


OPEN_WIDTH_M = 0.04
CLOSE_WIDTH_M = 0.01
GRIPPER_SPEED_M_S = 0.03
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "franka_real.yaml"
BRIDGE_RELATIVE_PATH = Path("franka_action_bridge/build/franka_action_bridge")
BRIDGE_BUILD_COMMAND = (
    "cmake -S agentic_robots/envs/real/franka/franka_action_bridge "
    "-B agentic_robots/envs/real/franka/franka_action_bridge/build "
    "-DFranka_DIR=third_party/libfranka/build\n"
    "cmake --build agentic_robots/envs/real/franka/franka_action_bridge/build"
)


@dataclass(frozen=True)
class FrankaWorkspaceLimits:
    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]

    def bridge_args(self) -> list[str]:
        return [
            f"{self.x[0]:.9f}",
            f"{self.x[1]:.9f}",
            f"{self.y[0]:.9f}",
            f"{self.y[1]:.9f}",
            f"{self.z[0]:.9f}",
            f"{self.z[1]:.9f}",
        ]

    def contains(self, position: np.ndarray) -> bool:
        x, y, z = np.asarray(position, dtype=np.float64).reshape(3)
        return bool(
            self.x[0] <= x <= self.x[1]
            and self.y[0] <= y <= self.y[1]
            and self.z[0] <= z <= self.z[1]
        )


@dataclass(frozen=True)
class FrankaTcpConfig:
    offset_from_ee: tuple[float, float, float]

    def bridge_args(self) -> list[str]:
        return [f"{value:.9f}" for value in self.offset_from_ee]


@dataclass(frozen=True)
class FrankaRealConfig:
    host: str
    default_motion_duration_s: float
    max_translation_m: float
    orientation_tolerance: float
    workspace: FrankaWorkspaceLimits
    tcp: FrankaTcpConfig


@dataclass(frozen=True)
class FrankaConnectionStatus:
    host: str
    ping_ok: bool
    bridge_available: bool
    motion_enabled: bool


@dataclass(frozen=True)
class FrankaPose:
    position: np.ndarray
    quaternion_wxyz: np.ndarray


@dataclass(frozen=True)
class FrankaGripperState:
    width: float
    max_width: float
    is_grasped: bool
    temperature: float


class FrankaReal:
    """Real Franka backend using the local libfranka bridge."""

    def __init__(
        self,
        *,
        host: str | None = None,
        bridge_binary: str | Path | None = None,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        motion_enabled: bool = False,
        max_translation_m: float | None = None,
        default_motion_duration_s: float | None = None,
        orientation_tolerance: float | None = None,
        workspace: FrankaWorkspaceLimits | None = None,
        tcp: FrankaTcpConfig | None = None,
    ) -> None:
        config = load_franka_real_config(config_path)
        self.host = config.host if host is None else host
        self.bridge_binary = Path(bridge_binary) if bridge_binary else _default_bridge_binary()
        self.motion_enabled = motion_enabled
        self.max_translation_m = (
            config.max_translation_m if max_translation_m is None else max_translation_m
        )
        self.default_motion_duration_s = (
            config.default_motion_duration_s
            if default_motion_duration_s is None
            else default_motion_duration_s
        )
        self.orientation_tolerance = (
            config.orientation_tolerance
            if orientation_tolerance is None
            else orientation_tolerance
        )
        self.workspace = config.workspace if workspace is None else workspace
        self.tcp = config.tcp if tcp is None else tcp
        self.last_command: dict[str, object] | None = None

    def check_connection(self) -> FrankaConnectionStatus:
        return FrankaConnectionStatus(
            host=self.host,
            ping_ok=self.ping_robot(self.host),
            bridge_available=self.bridge_binary.exists(),
            motion_enabled=self.motion_enabled,
        )

    def load(self) -> None:
        status = self.check_connection()
        if not status.bridge_available:
            raise RuntimeError(_missing_bridge_message())
        if not status.ping_ok:
            raise RuntimeError(f"Franka robot is not reachable at {self.host}")

    def get_end_effector_pose(self) -> FrankaPose:
        payload = self._run_json("pose", *self.tcp.bridge_args())
        return FrankaPose(
            position=np.asarray(payload["position"], dtype=np.float64),
            quaternion_wxyz=_normalize_quaternion(np.asarray(payload["quaternion_wxyz"])),
        )

    def move_to_pose(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
        *,
        duration_s: float | None = None,
    ) -> np.ndarray:
        self._require_motion_enabled()
        start = self.get_end_effector_pose()
        target_position = np.asarray(position, dtype=np.float64).reshape(3)
        target_quaternion = _normalize_quaternion(np.asarray(quaternion_wxyz).reshape(4))
        delta = target_position - start.position
        duration = self.default_motion_duration_s if duration_s is None else float(duration_s)
        self._validate_move(delta, start.quaternion_wxyz, target_quaternion, duration)

        result = self._run_json(
            "move-relative",
            f"{delta[0]:.9f}",
            f"{delta[1]:.9f}",
            f"{delta[2]:.9f}",
            f"{duration:.3f}",
            *self.workspace.bridge_args(),
            *self.tcp.bridge_args(),
        )
        self.last_command = {
            "command": "move_to_pose",
            "start_position": start.position.tolist(),
            "target_position": target_position.tolist(),
            "final_position": result["position"],
            "duration_s": duration,
        }
        return target_position[None, :]

    def open_gripper(self, *, width_m: float = OPEN_WIDTH_M) -> FrankaGripperState:
        self._require_motion_enabled()
        payload = self._run_json(
            "gripper-open",
            f"{float(width_m):.6f}",
            f"{GRIPPER_SPEED_M_S:.6f}",
        )
        state = FrankaGripperState(
            width=float(payload["width"]),
            max_width=float(payload["max_width"]),
            is_grasped=bool(payload["is_grasped"]),
            temperature=float(payload["temperature"]),
        )
        self.last_command = {"command": "open_gripper", "gripper": asdict(state)}
        return state

    def close_gripper(self, *, width_m: float = CLOSE_WIDTH_M) -> FrankaGripperState:
        self._require_motion_enabled()
        payload = self._run_json(
            "gripper-close",
            f"{float(width_m):.6f}",
            f"{GRIPPER_SPEED_M_S:.6f}",
        )
        state = FrankaGripperState(
            width=float(payload["width"]),
            max_width=float(payload["max_width"]),
            is_grasped=bool(payload["is_grasped"]),
            temperature=float(payload["temperature"]),
        )
        self.last_command = {"command": "close_gripper", "gripper": asdict(state)}
        return state

    def read_gripper(self) -> FrankaGripperState:
        payload = self._run_json("gripper-read")
        return FrankaGripperState(
            width=float(payload["width"]),
            max_width=float(payload["max_width"]),
            is_grasped=bool(payload["is_grasped"]),
            temperature=float(payload["temperature"]),
        )

    @staticmethod
    def ping_robot(host: str) -> bool:
        return (
            subprocess.run(
                ["ping", "-c", "1", "-W", "1", host],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            ).returncode
            == 0
        )

    def _run_json(self, *args: str) -> dict[str, object]:
        if not self.bridge_binary.exists():
            raise RuntimeError(_missing_bridge_message())
        result = subprocess.run(
            [str(self.bridge_binary), self.host, *args],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"Real Franka bridge failed (exit {result.returncode}): {detail}")
        try:
            return json.loads(result.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Real Franka bridge returned invalid JSON: {result.stdout}") from exc

    def _require_motion_enabled(self) -> None:
        if not self.motion_enabled:
            raise RuntimeError(
                "Real robot motion is disabled. Construct FrankaReal with "
                "motion_enabled=True only after the workspace and stop button are ready."
            )

    def _validate_move(
        self,
        delta: np.ndarray,
        current_quaternion: np.ndarray,
        target_quaternion: np.ndarray,
        duration_s: float,
    ) -> None:
        if np.max(np.abs(delta)) > self.max_translation_m:
            raise ValueError(
                f"Real Franka move exceeds max_translation_m={self.max_translation_m}: "
                f"delta={delta.tolist()}"
            )
        if not 2.0 <= duration_s <= 20.0:
            raise ValueError("duration_s must be in [2.0, 20.0]")
        if not _same_orientation(current_quaternion, target_quaternion, self.orientation_tolerance):
            raise ValueError(
                "FrankaReal currently supports translation-only Cartesian moves. "
                "Use the current quaternion_wxyz from get_end_effector_pose()."
            )


def load_franka_real_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> FrankaRealConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    motion = data.get("motion") or {}
    workspace = data.get("workspace") or {}
    tcp = data.get("tcp") or {}
    return FrankaRealConfig(
        host=str(data["host"]),
        default_motion_duration_s=float(motion["default_duration_s"]),
        max_translation_m=float(motion["max_translation_m"]),
        orientation_tolerance=float(motion["orientation_tolerance"]),
        workspace=FrankaWorkspaceLimits(
            x=_read_limits(workspace, "x"),
            y=_read_limits(workspace, "y"),
            z=_read_limits(workspace, "z"),
        ),
        tcp=FrankaTcpConfig(offset_from_ee=_read_vector3(tcp, "offset_from_ee")),
    )


def _default_bridge_binary() -> Path:
    return Path(__file__).resolve().parent / BRIDGE_RELATIVE_PATH


def _missing_bridge_message() -> str:
    return f"Real Franka bridge binary is missing. Build it with:\n{BRIDGE_BUILD_COMMAND}"


def _read_limits(data: dict[str, object], axis: str) -> tuple[float, float]:
    raw_limits = data.get(axis)
    if not isinstance(raw_limits, list | tuple) or len(raw_limits) != 2:
        raise ValueError(f"workspace.{axis} must be a two-value [min, max] list")
    limits = (float(raw_limits[0]), float(raw_limits[1]))
    if limits[0] >= limits[1]:
        raise ValueError(f"workspace.{axis} min must be smaller than max")
    return limits


def _read_vector3(data: dict[str, object], key: str) -> tuple[float, float, float]:
    raw_value = data.get(key)
    if not isinstance(raw_value, list | tuple) or len(raw_value) != 3:
        raise ValueError(f"tcp.{key} must be a three-value [x, y, z] list")
    return (float(raw_value[0]), float(raw_value[1]), float(raw_value[2]))


def _normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quaternion))
    if norm == 0.0:
        raise ValueError("Quaternion norm must be non-zero")
    return quaternion / norm


def _same_orientation(left: np.ndarray, right: np.ndarray, tolerance: float) -> bool:
    return bool(
        np.linalg.norm(left - right) <= tolerance or np.linalg.norm(left + right) <= tolerance
    )
