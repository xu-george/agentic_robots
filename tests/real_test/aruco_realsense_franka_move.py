from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CALIBRATION = PROJECT_ROOT / "configs" / "robot" / "realsense_d455_franka_calibration.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tests" / "real_test" / "outputs" / "aruco_realsense_franka_move"
DEFAULT_MARKER_ID = 4
DEFAULT_MARKER_SIZE_M = 0.10
DEFAULT_TARGET_CLEARANCE_M = 0.05
DEFAULT_MAX_STEP_M = 0.04


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect an ArUco marker with RealSense and optionally move Franka above it."
    )
    parser.add_argument("--host", default=None, help="Override Franka host from franka_real.yaml.")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--marker-id", type=int, default=DEFAULT_MARKER_ID)
    parser.add_argument("--marker-size-m", type=float, default=DEFAULT_MARKER_SIZE_M)
    parser.add_argument("--target-clearance-m", type=float, default=DEFAULT_TARGET_CLEARANCE_M)
    parser.add_argument("--max-step-m", type=float, default=DEFAULT_MAX_STEP_M)
    parser.add_argument("--duration-s", type=float, default=5.0)
    parser.add_argument("--settle-frames", type=int, default=30)
    parser.add_argument("--move-home-first", action="store_true")
    parser.add_argument("--execute-real", action="store_true")
    parser.add_argument("--confirm-clear-workspace", action="store_true")
    args = parser.parse_args()

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from agentic_robots.envs.real import FrankaReal, RealSenseCamera, load_franka_real_config

    calibration = _load_yaml(args.calibration)
    camera_matrix = np.asarray(calibration["intrinsics"], dtype=np.float64).reshape(3, 3)
    camera_to_base = np.asarray(calibration["camera_pose"]["matrix"], dtype=np.float64).reshape(4, 4)
    dist_coeffs = _distortion_coeffs(calibration)
    camera_cfg = calibration["camera"]

    camera = RealSenseCamera(
        width=int(camera_cfg["width"]),
        height=int(camera_cfg["height"]),
        serial=str(camera_cfg["serial"]) if camera_cfg.get("serial") else None,
    )
    robot = FrankaReal(host=args.host, motion_enabled=args.execute_real)
    real_config = load_franka_real_config()

    try:
        robot.load()
        if args.execute_real and not args.confirm_clear_workspace:
            raise RuntimeError(
                "Refusing real motion. Re-run with --confirm-clear-workspace only after "
                "the marker area is clear and the user stop is reachable."
            )
        if args.execute_real and args.move_home_first:
            print("moving_home_joints", np.round(real_config.home_joint_positions, 5).tolist())
            home_pose = robot.move_home()
            print("home_tcp_base_m", np.round(home_pose.position, 5).tolist())

        detection = _detect_marker_pose(
            camera=camera,
            marker_id=args.marker_id,
            marker_size_m=args.marker_size_m,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            settle_frames=args.settle_frames,
        )
    finally:
        camera.stop()

    marker_pose_base = camera_to_base @ detection.marker_pose_camera
    marker_position_base = marker_pose_base[:3, 3]
    target_position_base = marker_position_base + np.array(
        [0.0, 0.0, args.target_clearance_m],
        dtype=np.float64,
    )

    start_pose = robot.get_end_effector_pose()
    waypoints = _make_waypoints(
        start_pose.position,
        target_position_base,
        max_step_m=args.max_step_m,
    )
    path_inside_workspace = all(real_config.workspace.contains(waypoint) for waypoint in waypoints)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = args.output_dir / "aruco_pose_overlay.jpg"
    result_path = args.output_dir / "result.json"
    cv2.imwrite(str(overlay_path), detection.overlay_bgr)

    result: dict[str, Any] = {
        "marker_id": int(args.marker_id),
        "marker_size_m": float(args.marker_size_m),
        "marker_center_px": detection.center_px.tolist(),
        "marker_center_depth_m": float(detection.center_depth_m),
        "marker_pose_camera": detection.marker_pose_camera.tolist(),
        "marker_pose_base": marker_pose_base.tolist(),
        "marker_position_base_m": marker_position_base.tolist(),
        "target_clearance_m": float(args.target_clearance_m),
        "target_tcp_base_m": target_position_base.tolist(),
        "start_tcp_base_m": start_pose.position.tolist(),
        "start_quaternion_wxyz": start_pose.quaternion_wxyz.tolist(),
        "waypoint_count": len(waypoints),
        "path_inside_workspace": bool(path_inside_workspace),
        "workspace_limits": {
            "x": list(real_config.workspace.x),
            "y": list(real_config.workspace.y),
            "z": list(real_config.workspace.z),
        },
        "overlay_path": str(overlay_path),
    }

    print("marker_position_base_m", np.round(marker_position_base, 5).tolist())
    print("target_tcp_base_m", np.round(target_position_base, 5).tolist())
    print("start_tcp_base_m", np.round(start_pose.position, 5).tolist())
    print("path_inside_workspace", path_inside_workspace)
    print("overlay", overlay_path)

    if not args.execute_real:
        result["executed"] = False
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("Read-only ArUco pose check completed without robot motion.")
        print("Add --execute-real --confirm-clear-workspace to move the robot above the marker.")
        print("result", result_path)
        return

    if not path_inside_workspace:
        raise ValueError(
            "Refusing real motion. Target path leaves workspace limits: "
            f"x={real_config.workspace.x}, y={real_config.workspace.y}, z={real_config.workspace.z}"
        )

    print("moving_to_marker_clearance")
    for index, waypoint in enumerate(waypoints, start=1):
        print(f"moving_waypoint_{index}_of_{len(waypoints)}", np.round(waypoint, 5).tolist())
        robot.move_to_pose(waypoint, start_pose.quaternion_wxyz, duration_s=args.duration_s)

    final_pose = robot.get_end_effector_pose()
    target_error = final_pose.position - target_position_base
    result.update(
        {
            "executed": True,
            "final_tcp_base_m": final_pose.position.tolist(),
            "final_quaternion_wxyz": final_pose.quaternion_wxyz.tolist(),
            "target_error_m": target_error.tolist(),
            "target_error_norm_m": float(np.linalg.norm(target_error)),
        }
    )
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("final_tcp_base_m", np.round(final_pose.position, 5).tolist())
    print("target_error_m", np.round(target_error, 5).tolist())
    print("target_error_norm_m", round(float(np.linalg.norm(target_error)), 6))
    print("result", result_path)


class MarkerPoseDetection:
    def __init__(
        self,
        *,
        center_px: np.ndarray,
        center_depth_m: float,
        marker_pose_camera: np.ndarray,
        overlay_bgr: np.ndarray,
    ) -> None:
        self.center_px = center_px
        self.center_depth_m = center_depth_m
        self.marker_pose_camera = marker_pose_camera
        self.overlay_bgr = overlay_bgr


def _detect_marker_pose(
    *,
    camera,
    marker_id: int,
    marker_size_m: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    settle_frames: int,
) -> MarkerPoseDetection:
    if marker_size_m <= 0.0:
        raise ValueError("--marker-size-m must be positive")

    frame = camera.capture(settle_frames=settle_frames)
    color_bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids = _detect_aruco(gray)
    if ids is None or len(ids) == 0:
        raise RuntimeError(f"No ArUco markers detected; expected marker id {marker_id}")

    ids_flat = ids.reshape(-1)
    matches = np.flatnonzero(ids_flat == marker_id)
    if matches.size == 0:
        raise RuntimeError(f"Detected marker ids {ids_flat.tolist()}, but not id {marker_id}")

    marker_corners = corners[int(matches[0])].reshape(4, 2).astype(np.float64)
    center_px = marker_corners.mean(axis=0)
    center_depth_m = _sample_depth_image(frame.depth_m, center_px)
    marker_pose_camera = _estimate_marker_pose(
        marker_corners,
        marker_size_m=marker_size_m,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
    )

    overlay = color_bgr.copy()
    cv2.aruco.drawDetectedMarkers(
        overlay,
        [marker_corners.reshape(1, 4, 2).astype(np.float32)],
        np.array([[marker_id]], dtype=np.int32),
    )
    _draw_axes(overlay, camera_matrix, dist_coeffs, marker_pose_camera, marker_size_m)
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
        f"id={marker_id} z_pnp={marker_pose_camera[2, 3]:.3f}m z_depth={center_depth_m:.3f}m",
        tuple(np.round(center_px + np.array([10.0, -10.0])).astype(int)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    return MarkerPoseDetection(
        center_px=center_px,
        center_depth_m=center_depth_m,
        marker_pose_camera=marker_pose_camera,
        overlay_bgr=overlay,
    )


def _estimate_marker_pose(
    corners_px: np.ndarray,
    *,
    marker_size_m: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    half = marker_size_m / 2.0
    object_points = np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        corners_px.astype(np.float64),
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not success:
        raise RuntimeError("cv2.solvePnP failed for ArUco marker")
    rotation, _ = cv2.Rodrigues(rvec)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = tvec.reshape(3)
    return pose


def _draw_axes(
    image_bgr: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    marker_pose_camera: np.ndarray,
    marker_size_m: float,
) -> None:
    axis = marker_size_m * 0.75
    points_3d = np.array(
        [
            [0.0, 0.0, 0.0],
            [axis, 0.0, 0.0],
            [0.0, axis, 0.0],
            [0.0, 0.0, axis],
        ],
        dtype=np.float64,
    )
    rvec, _ = cv2.Rodrigues(marker_pose_camera[:3, :3])
    tvec = marker_pose_camera[:3, 3].reshape(3, 1)
    points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, camera_matrix, dist_coeffs)
    pts = np.round(points_2d.reshape(-1, 2)).astype(int)
    origin = tuple(pts[0])
    cv2.line(image_bgr, origin, tuple(pts[1]), (0, 0, 255), 3)
    cv2.line(image_bgr, origin, tuple(pts[2]), (0, 255, 0), 3)
    cv2.line(image_bgr, origin, tuple(pts[3]), (255, 0, 0), 3)


def _detect_aruco(gray: np.ndarray):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        return detector.detectMarkers(gray)[:2]
    return cv2.aruco.detectMarkers(gray, dictionary)[:2]


def _sample_depth_image(depth_m: np.ndarray, center_px: np.ndarray) -> float:
    cx, cy = np.round(center_px).astype(int)
    height, width = depth_m.shape[:2]
    samples: list[float] = []
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            x = int(np.clip(cx + dx, 0, width - 1))
            y = int(np.clip(cy + dy, 0, height - 1))
            depth = float(depth_m[y, x])
            if np.isfinite(depth) and depth > 0.0:
                samples.append(depth)
    if not samples:
        raise RuntimeError(f"No valid depth around marker center pixel {center_px.tolist()}")
    return float(np.median(samples))


def _make_waypoints(
    start: np.ndarray,
    target: np.ndarray,
    *,
    max_step_m: float,
) -> list[np.ndarray]:
    if max_step_m <= 0.0 or max_step_m > 0.05:
        raise ValueError("--max-step-m must be in (0.0, 0.05]")
    start = np.asarray(start, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    delta = target - start
    step_count = max(1, int(np.ceil(float(np.max(np.abs(delta))) / max_step_m)))
    return [start + delta * (step / step_count) for step in range(1, step_count + 1)]


def _distortion_coeffs(calibration: dict[str, Any]) -> np.ndarray:
    coeffs = ((calibration.get("camera") or {}).get("intrinsics") or {}).get("coeffs")
    if coeffs is None:
        return np.zeros((5, 1), dtype=np.float64)
    return np.asarray(coeffs, dtype=np.float64).reshape(-1, 1)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


if __name__ == "__main__":
    main()
