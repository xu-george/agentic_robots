from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only smoke test for the real lift env.")
    parser.add_argument("--object-prompt", default="object")
    parser.add_argument("--calibration", type=Path, default=None)
    args = parser.parse_args()

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from agentic_robots.envs.real import RealObjectLiftEnv

    kwargs = {"object_prompt": args.object_prompt}
    if args.calibration is not None:
        kwargs["calibration_path"] = args.calibration

    env = RealObjectLiftEnv(**kwargs, motion_enabled=False)
    try:
        obs = env.reset()
        print("real_lift_env smoke ok")
        print("rgb_shape", obs["rgb"].shape)
        print("depth_shape", obs["depth"].shape)
        print("intrinsics_shape", obs["intrinsics"].shape)
        print("camera_pose_shape", obs["camera_pose"].shape)
        print("eef_position", obs["robot"]["eef_position"].round(5).tolist())
        print("gripper_width", round(float(obs["robot"]["gripper_width"][0]), 5))
    finally:
        env.close()


if __name__ == "__main__":
    main()
