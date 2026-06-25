from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BRIDGE_MAX_STEP_M = 0.04


def main() -> None:
    parser = argparse.ArgumentParser(description="Real Franka hardware sequence test")
    parser.add_argument("--host", default="172.16.0.2")
    parser.add_argument("--dz", type=float, default=0.01)
    parser.add_argument("--duration-s", type=float, default=5.0)
    parser.add_argument(
        "--max-step-m",
        type=float,
        default=BRIDGE_MAX_STEP_M,
        help="Maximum per-axis Cartesian step for segmented target moves.",
    )
    parser.add_argument(
        "--target-position",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Optional absolute Cartesian target position to check/use.",
    )
    parser.add_argument(
        "--target-quaternion-wxyz",
        nargs=4,
        type=float,
        metavar=("W", "X", "Y", "Z"),
        help="Optional absolute target orientation to check/use.",
    )
    parser.add_argument(
        "--execute-real",
        action="store_true",
        help="Enable real robot motion and gripper commands.",
    )
    parser.add_argument(
        "--confirm-clear-workspace",
        action="store_true",
        help="Confirm workspace is clear and user stop is reachable.",
    )
    args = parser.parse_args()

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from agentic_robots.envs.real import FrankaReal, load_franka_real_config

    robot = FrankaReal(host=args.host, motion_enabled=args.execute_real)
    real_config = load_franka_real_config()
    robot.load()

    status = robot.check_connection()
    # The current pose is treated as "home" for this bridge-level test; no
    # separate joint-space homing command exists in FrankaReal yet.
    home_pose = robot.get_end_effector_pose()
    gripper = robot.read_gripper()

    print("connection", status)
    print("home_position_current_start", np.round(home_pose.position, 5).tolist())
    print("home_quaternion_wxyz", np.round(home_pose.quaternion_wxyz, 5).tolist())
    print("gripper", asdict(gripper))

    if args.max_step_m <= 0.0 or args.max_step_m > 0.05:
        raise ValueError("--max-step-m must be in (0.0, 0.05]")

    requested_target = None
    target_quaternion = home_pose.quaternion_wxyz
    if args.target_position is not None:
        requested_target = np.asarray(args.target_position, dtype=np.float64)
        target_quaternion = (
            home_pose.quaternion_wxyz
            if args.target_quaternion_wxyz is None
            else _normalize(np.asarray(args.target_quaternion_wxyz, dtype=np.float64))
        )
        delta = requested_target - home_pose.position
        orientation_match = _same_orientation(home_pose.quaternion_wxyz, target_quaternion)
        target_waypoints = _make_waypoints(
            home_pose.position,
            requested_target,
            max_step_m=args.max_step_m,
        )
        target_path_inside_workspace = all(
            real_config.workspace.contains(waypoint) for waypoint in target_waypoints
        )
        reachable = bool(orientation_match and target_path_inside_workspace)
        print("requested_target_position", np.round(requested_target, 5).tolist())
        print("requested_target_quaternion_wxyz", np.round(target_quaternion, 5).tolist())
        print("target_delta_from_current_m", np.round(delta, 5).tolist())
        print("target_max_axis_delta_m", round(float(np.max(np.abs(delta))), 5))
        print("target_waypoint_count", len(target_waypoints))
        print("target_inside_bridge_workspace", real_config.workspace.contains(requested_target))
        print("target_path_inside_bridge_workspace", target_path_inside_workspace)
        print("target_orientation_matches_current", orientation_match)
        print("target_reachable_by_segmented_franka_real_test", reachable)

    if not args.execute_real:
        print("Read-only FrankaReal hardware sequence check passed")
        return

    if not args.confirm_clear_workspace:
        raise RuntimeError(
            "Refusing real motion. Re-run with --confirm-clear-workspace only after "
            "the robot workspace is clear and the user stop is reachable."
        )
    if abs(args.dz) > 0.02:
        raise ValueError("--dz is limited to +/-0.02 m for this test")

    target_position = (
        requested_target
        if requested_target is not None
        else home_pose.position + np.array([0.0, 0.0, args.dz], dtype=np.float64)
    )
    target_waypoints = _make_waypoints(
        home_pose.position,
        target_position,
        max_step_m=args.max_step_m,
    )
    if not _same_orientation(home_pose.quaternion_wxyz, target_quaternion):
        raise ValueError(
            "Refusing real motion. This bridge-level test only supports translation "
            "while holding the current end-effector orientation."
        )
    if not all(real_config.workspace.contains(waypoint) for waypoint in target_waypoints):
        raise ValueError(
            "Refusing real motion. The target path leaves the bridge workspace limits: "
            f"x={real_config.workspace.x}, y={real_config.workspace.y}, "
            f"z={real_config.workspace.z}."
        )

    print("moving_to_target", np.round(target_position, 5).tolist())
    for index, waypoint in enumerate(target_waypoints, start=1):
        print(
            f"moving_to_target_waypoint_{index}_of_{len(target_waypoints)}",
            np.round(waypoint, 5).tolist(),
        )
        robot.move_to_pose(waypoint, home_pose.quaternion_wxyz, duration_s=args.duration_s)
    target_readback = robot.get_end_effector_pose()
    print("target_readback", np.round(target_readback.position, 5).tolist())

    print("returning_home", np.round(home_pose.position, 5).tolist())
    return_waypoints = _make_waypoints(
        target_readback.position,
        home_pose.position,
        max_step_m=args.max_step_m,
    )
    for index, waypoint in enumerate(return_waypoints, start=1):
        print(
            f"returning_home_waypoint_{index}_of_{len(return_waypoints)}",
            np.round(waypoint, 5).tolist(),
        )
        robot.move_to_pose(waypoint, home_pose.quaternion_wxyz, duration_s=args.duration_s)
    home_readback = robot.get_end_effector_pose()
    print("home_readback", np.round(home_readback.position, 5).tolist())

    print("opening_gripper")
    print("after_open", asdict(robot.open_gripper()))
    print("closing_gripper")
    print("after_close", asdict(robot.close_gripper()))
    print("reopening_gripper")
    print("after_reopen", asdict(robot.open_gripper()))

    final_pose = robot.get_end_effector_pose()
    print("final_position", np.round(final_pose.position, 5).tolist())
    print("last_command", robot.last_command)
    print("Real Franka hardware sequence test passed")


def _make_waypoints(
    start: np.ndarray,
    target: np.ndarray,
    *,
    max_step_m: float,
) -> list[np.ndarray]:
    delta = target - start
    step_count = max(1, int(np.ceil(float(np.max(np.abs(delta))) / max_step_m)))
    return [start + delta * (step / step_count) for step in range(1, step_count + 1)]


def _normalize(quaternion: np.ndarray) -> np.ndarray:
    return quaternion / np.linalg.norm(quaternion)


def _same_orientation(left: np.ndarray, right: np.ndarray, tolerance: float = 5e-3) -> bool:
    return bool(
        np.linalg.norm(left - right) <= tolerance or np.linalg.norm(left + right) <= tolerance
    )


if __name__ == "__main__":
    main()
