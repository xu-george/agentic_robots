from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CALIBRATION = PROJECT_ROOT / "configs" / "robot" / "realsense_d455_franka_calibration.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tests" / "api_test" / "outputs" / "marker_calibration_check"
DEFAULT_MARKER_ID = 4
DEFAULT_TARGET_CLEARANCE_M = 0.05
DEFAULT_MAX_STEP_M = 0.04


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check RealSense-to-Franka calibration by moving the closed gripper TCP "
            "above a detected ArUco marker."
        )
    )
    parser.add_argument("--host", default=None, help="Override Franka host from franka_real.yaml.")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--marker-id", type=int, default=None)
    parser.add_argument("--target-clearance-m", type=float, default=DEFAULT_TARGET_CLEARANCE_M)
    parser.add_argument("--max-step-m", type=float, default=DEFAULT_MAX_STEP_M)
    parser.add_argument("--duration-s", type=float, default=5.0)
    parser.add_argument("--home-duration-s", type=float, default=None)
    parser.add_argument("--settle-frames", type=int, default=30)
    parser.add_argument(
        "--skip-home",
        action="store_true",
        help="Do not move to the configured home joint pose before marker detection.",
    )
    parser.add_argument("--execute-real", action="store_true")
    parser.add_argument("--confirm-clear-workspace", action="store_true")
    args = parser.parse_args()

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from agentic_robots.envs.real import FrankaReal, load_franka_real_config

    calibration = _load_yaml(args.calibration)
    validation = _load_validation(calibration)
    marker_id = args.marker_id
    if marker_id is None:
        marker_id = int(validation.get("marker_id", DEFAULT_MARKER_ID))

    camera_matrix = np.asarray(calibration["intrinsics"], dtype=np.float64).reshape(3, 3)
    camera_to_base = np.asarray(calibration["camera_pose"]["matrix"], dtype=np.float64).reshape(4, 4)

    robot = FrankaReal(host=args.host, motion_enabled=args.execute_real)
    real_config = load_franka_real_config()
    robot.load()
    status = robot.check_connection()
    home_pose = None

    if args.execute_real:
        if not args.confirm_clear_workspace:
            raise RuntimeError(
                "Refusing real motion. Re-run with --confirm-clear-workspace only after "
                "the marker area is clear and the user stop is reachable."
            )
        if args.target_clearance_m < 0.0 or args.target_clearance_m > 0.10:
            raise ValueError("--target-clearance-m must be in [0.0, 0.10]")
        if not args.skip_home:
            print("moving_home_joints", np.round(real_config.home_joint_positions, 5).tolist())
            home_pose = robot.move_home(duration_s=args.home_duration_s)
            print("home_tcp_base_m", np.round(home_pose.position, 5).tolist())

    detection = _detect_marker_position(
        marker_id=marker_id,
        camera_matrix=camera_matrix,
        camera_width=int(calibration["camera"]["width"]),
        camera_height=int(calibration["camera"]["height"]),
        settle_frames=args.settle_frames,
    )
    marker_base = _transform_point(camera_to_base, detection.position_camera_m)
    target_base = marker_base + np.array([0.0, 0.0, args.target_clearance_m], dtype=np.float64)

    start_pose = robot.get_end_effector_pose()
    target_waypoints = _make_waypoints(
        start_pose.position,
        target_base,
        max_step_m=args.max_step_m,
    )
    target_path_inside_workspace = all(
        real_config.workspace.contains(waypoint) for waypoint in target_waypoints
    )
    validation_marker_base = validation.get("marker_position_base_m")
    validation_error = None
    if validation_marker_base is not None:
        validation_marker_base = np.asarray(validation_marker_base, dtype=np.float64).reshape(3)
        validation_error = marker_base - validation_marker_base

    result = {
        "marker_id": marker_id,
        "marker_center_px": detection.center_px.tolist(),
        "marker_center_depth_m": float(detection.depth_m),
        "marker_position_camera_m": detection.position_camera_m.tolist(),
        "marker_position_base_m": marker_base.tolist(),
        "target_clearance_m": float(args.target_clearance_m),
        "target_tcp_base_m": target_base.tolist(),
        "start_tcp_base_m": start_pose.position.tolist(),
        "start_quaternion_wxyz": start_pose.quaternion_wxyz.tolist(),
        "waypoint_count": len(target_waypoints),
        "target_inside_bridge_workspace": real_config.workspace.contains(target_base),
        "target_path_inside_bridge_workspace": target_path_inside_workspace,
        "workspace_limits": {
            "x": list(real_config.workspace.x),
            "y": list(real_config.workspace.y),
            "z": list(real_config.workspace.z),
        },
        "connection": {
            "host": status.host,
            "ping_ok": status.ping_ok,
            "bridge_available": status.bridge_available,
            "motion_enabled": status.motion_enabled,
        },
        "home_joint_positions": list(real_config.home_joint_positions),
        "home_executed": bool(home_pose is not None),
    }
    if home_pose is not None:
        result["home_tcp_base_m"] = home_pose.position.tolist()
        result["home_quaternion_wxyz"] = home_pose.quaternion_wxyz.tolist()
    if validation_marker_base is not None and validation_error is not None:
        result["validation_marker_position_base_m"] = validation_marker_base.tolist()
        result["validation_marker_error_m"] = validation_error.tolist()
        result["validation_marker_error_norm_m"] = float(np.linalg.norm(validation_error))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = args.output_dir / "marker_detection_overlay.jpg"
    cv2.imwrite(str(overlay_path), detection.overlay_bgr)
    result_path = args.output_dir / "result.json"

    print("marker_position_base_m", np.round(marker_base, 5).tolist())
    print("target_tcp_base_m", np.round(target_base, 5).tolist())
    print("start_tcp_base_m", np.round(start_pose.position, 5).tolist())
    print("target_waypoint_count", len(target_waypoints))
    print("workspace_z_limits", list(real_config.workspace.z))
    print("target_inside_bridge_workspace", real_config.workspace.contains(target_base))
    print("target_path_inside_bridge_workspace", target_path_inside_workspace)
    if validation_error is not None:
        print("validation_marker_error_m", np.round(validation_error, 5).tolist())
        print("validation_marker_error_norm_m", round(float(np.linalg.norm(validation_error)), 6))
    print("overlay", overlay_path)

    if not args.execute_real:
        result["executed"] = False
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("Read-only calibration check completed without robot motion.")
        if not target_path_inside_workspace:
            print("Do not execute real motion until the marker target is inside workspace.")
        else:
            print("Add --execute-real to close the gripper and move above the marker.")
        print("result", result_path)
        return

    if not target_path_inside_workspace:
        raise ValueError(
            "Refusing real motion. Target path leaves the bridge workspace limits: "
            f"x={real_config.workspace.x}, y={real_config.workspace.y}, z={real_config.workspace.z}"
        )

    print("closing_gripper")
    result["closed_gripper"] = _state_dict(robot.close_gripper())
    print("moving_to_marker_clearance")
    for index, waypoint in enumerate(target_waypoints, start=1):
        print(
            f"moving_waypoint_{index}_of_{len(target_waypoints)}",
            np.round(waypoint, 5).tolist(),
        )
        robot.move_to_pose(waypoint, start_pose.quaternion_wxyz, duration_s=args.duration_s)

    final_pose = robot.get_end_effector_pose()
    position_error = final_pose.position - target_base
    result.update(
        {
            "executed": True,
            "final_tcp_base_m": final_pose.position.tolist(),
            "final_quaternion_wxyz": final_pose.quaternion_wxyz.tolist(),
            "target_error_m": position_error.tolist(),
            "target_error_norm_m": float(np.linalg.norm(position_error)),
        }
    )
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("final_tcp_base_m", np.round(final_pose.position, 5).tolist())
    print("target_error_m", np.round(position_error, 5).tolist())
    print("target_error_norm_m", round(float(np.linalg.norm(position_error)), 6))
    print("result", result_path)


class MarkerDetection:
    def __init__(
        self,
        *,
        center_px: np.ndarray,
        depth_m: float,
        position_camera_m: np.ndarray,
        overlay_bgr: np.ndarray,
    ) -> None:
        self.center_px = center_px
        self.depth_m = depth_m
        self.position_camera_m = position_camera_m
        self.overlay_bgr = overlay_bgr


def _detect_marker_position(
    *,
    marker_id: int,
    camera_matrix: np.ndarray,
    camera_width: int,
    camera_height: int,
    settle_frames: int,
) -> MarkerDetection:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, camera_width, camera_height, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, camera_width, camera_height, rs.format.z16, 30)
    align = rs.align(rs.stream.color)

    pipeline.start(config)
    try:
        frames = None
        for _ in range(max(1, settle_frames)):
            frames = align.process(pipeline.wait_for_frames(2000))
        if frames is None:
            raise RuntimeError("No RealSense frames received")
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("RealSense did not provide aligned color/depth frames")

        color_bgr = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids = _detect_aruco(gray)
        if ids is None or len(ids) == 0:
            raise RuntimeError(f"No ArUco markers detected; expected marker id {marker_id}")

        ids_flat = ids.reshape(-1)
        matches = np.flatnonzero(ids_flat == marker_id)
        if matches.size == 0:
            raise RuntimeError(f"Detected marker ids {ids_flat.tolist()}, but not id {marker_id}")

        marker_corners = corners[int(matches[0])].reshape(4, 2)
        center_px = marker_corners.mean(axis=0)
        depth_m = _sample_depth(depth_frame, center_px)
        position_camera_m = _pixel_to_camera(center_px, depth_m, camera_matrix)

        overlay = color_bgr.copy()
        cv2.aruco.drawDetectedMarkers(overlay, [marker_corners.reshape(1, 4, 2)], np.array([[marker_id]]))
        cv2.drawMarker(
            overlay,
            tuple(np.round(center_px).astype(int)),
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=24,
            thickness=2,
        )
        cv2.putText(
            overlay,
            f"id={marker_id} depth={depth_m:.3f}m",
            tuple(np.round(center_px + np.array([10.0, -10.0])).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return MarkerDetection(
            center_px=center_px.astype(np.float64),
            depth_m=float(depth_m),
            position_camera_m=position_camera_m,
            overlay_bgr=overlay,
        )
    finally:
        pipeline.stop()


def _detect_aruco(gray: np.ndarray):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        return detector.detectMarkers(gray)[:2]
    return cv2.aruco.detectMarkers(gray, dictionary)[:2]


def _sample_depth(depth_frame, center_px: np.ndarray) -> float:
    cx, cy = np.round(center_px).astype(int)
    width = int(depth_frame.get_width())
    height = int(depth_frame.get_height())
    samples: list[float] = []
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            x = int(np.clip(cx + dx, 0, width - 1))
            y = int(np.clip(cy + dy, 0, height - 1))
            depth = float(depth_frame.get_distance(x, y))
            if np.isfinite(depth) and depth > 0.0:
                samples.append(depth)
    if not samples:
        raise RuntimeError(f"No valid depth around marker center pixel {center_px.tolist()}")
    return float(np.median(samples))


def _pixel_to_camera(pixel_xy: np.ndarray, depth_m: float, camera_matrix: np.ndarray) -> np.ndarray:
    u, v = np.asarray(pixel_xy, dtype=np.float64).reshape(2)
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    return np.array([(u - cx) * depth_m / fx, (v - cy) * depth_m / fy, depth_m], dtype=np.float64)


def _transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    point_h = np.append(np.asarray(point, dtype=np.float64).reshape(3), 1.0)
    return (np.asarray(transform, dtype=np.float64).reshape(4, 4) @ point_h)[:3]


def _make_waypoints(
    start: np.ndarray,
    target: np.ndarray,
    *,
    max_step_m: float,
) -> list[np.ndarray]:
    if max_step_m <= 0.0 or max_step_m > 0.05:
        raise ValueError("--max-step-m must be in (0.0, 0.05]")
    delta = target - start
    step_count = max(1, int(np.ceil(float(np.max(np.abs(delta))) / max_step_m)))
    return [start + delta * (step / step_count) for step in range(1, step_count + 1)]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _load_validation(calibration: dict[str, Any]) -> dict[str, Any]:
    validation = ((calibration.get("calibration") or {}).get("validation") or {})
    result_file = validation.get("result_file")
    if not result_file:
        return {}
    path = Path(str(result_file))
    if not path.exists():
        return {}
    return _load_yaml(path)


def _state_dict(state) -> dict[str, Any]:
    return {
        "width": float(state.width),
        "max_width": float(state.max_width),
        "is_grasped": bool(state.is_grasped),
        "temperature": float(state.temperature),
    }


if __name__ == "__main__":
    main()
