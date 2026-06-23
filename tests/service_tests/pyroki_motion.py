from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = Path("tests/service_tests/outputs/pyroki_motion")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a simple PyRoKi IK/planning test.")
    parser.add_argument("--url", default="http://127.0.0.1:8116", help="PyRoKi service URL.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timesteps", type=int, default=5)
    return parser.parse_args()


def check_service(url: str) -> None:
    try:
        response = requests.get(f"{url.rstrip('/')}/health", timeout=3.0)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise SystemExit(
            f"PyRoKi service is not reachable at {url}. "
            "Start it with: python -m agentic_robots.services.motion.pyroki.server"
        ) from exc


def main() -> None:
    args = parse_args()

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from agentic_robots.services.motion.pyroki.client import PyrokiClient

    check_service(args.url)

    client = PyrokiClient(url=args.url, timeout_s=120.0)

    start_pose = np.array([1.0, 0.0, 0.0, 0.0, 0.45, 0.0, 0.45], dtype=np.float64)
    end_pose = np.array([1.0, 0.0, 0.0, 0.0, 0.50, 0.0, 0.45], dtype=np.float64)

    start_ik = client.solve_ik(start_pose)
    end_ik = client.solve_ik(end_pose, prev_cfg=start_ik.joint_positions)
    plan = client.plan(start_pose, end_pose, timesteps=args.timesteps)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "start_joint_positions.npy", start_ik.joint_positions)
    np.save(args.output_dir / "end_joint_positions.npy", end_ik.joint_positions)
    np.save(args.output_dir / "waypoints.npy", plan.waypoints)

    result_path = args.output_dir / "results.json"
    result_path.write_text(
        json.dumps(
            {
                "service_url": args.url,
                "start_pose_wxyz_xyz": start_pose.tolist(),
                "end_pose_wxyz_xyz": end_pose.tolist(),
                "start_joint_positions": start_ik.joint_positions.tolist(),
                "end_joint_positions": end_ik.joint_positions.tolist(),
                "waypoints_shape": list(plan.waypoints.shape),
                "dt": plan.dt,
                "start_joint_positions_path": str(
                    args.output_dir / "start_joint_positions.npy"
                ),
                "end_joint_positions_path": str(args.output_dir / "end_joint_positions.npy"),
                "waypoints_path": str(args.output_dir / "waypoints.npy"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Start IK joints: {np.round(start_ik.joint_positions, 4).tolist()}")
    print(f"End IK joints: {np.round(end_ik.joint_positions, 4).tolist()}")
    print(f"Waypoints shape: {plan.waypoints.shape}")
    print(f"Wrote {result_path}")


if __name__ == "__main__":
    main()
