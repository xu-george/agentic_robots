from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from agentic_robots.envs.pybullet import StackingEnv

    env = StackingEnv(gui=False, seed=7)
    try:
        obs = env.reset()

        assert set(obs) == {"images", "robot", "ground_truth"}
        assert env.camera_info.intrinsics.shape == (3, 3)
        assert env.camera_info.extrinsics.shape == (4, 4)
        assert env.camera_info.view_matrix is not None
        assert env.camera_info.projection_matrix is not None

        images = obs["images"]
        assert images["rgb"].shape == (env.camera.height, env.camera.width, 3)
        assert images["depth"].shape == (env.camera.height, env.camera.width)
        assert images["segmentation"].shape == (env.camera.height, env.camera.width)
        assert images["rgb"].dtype == np.uint8

        robot_state = obs["robot"]
        assert robot_state["joint_positions"].shape == (7,)
        assert robot_state["gripper_positions"].shape == (2,)
        assert robot_state["eef_position"].shape == (3,)
        assert robot_state["eef_quaternion_wxyz"].shape == (4,)

        objects = obs["ground_truth"]["objects"]
        assert {"red_cube", "blue_cube"} <= set(objects)
        assert objects["red_cube"]["position"].shape == (3,)
        assert objects["red_cube"]["quaternion_wxyz"].shape == (4,)

        red_position, _ = env.get_object_pose("red_cube")
        assert red_position.shape == (3,)
        blue_position, _ = env.get_object_pose("blue_cube")
        for cube_position in (red_position, blue_position):
            assert 0.36 <= cube_position[0] <= 0.58
            assert -0.18 <= cube_position[1] <= 0.18
            assert np.isclose(cube_position[2], 0.075, atol=1e-3)
        assert np.linalg.norm(red_position[:2] - blue_position[:2]) >= 0.12

        current_pose = env.robot.get_end_effector_pose()
        target_position = current_pose.position + np.array([0.0, 0.0, 0.03])
        joints = env.robot.solve_ik(target_position, current_pose.quaternion_wxyz)
        assert joints.shape == (7,)

        trajectory = env.robot.plan_to_pose(
            target_position,
            current_pose.quaternion_wxyz,
            steps=5,
        )
        assert trajectory.shape == (5, 7)

        env.robot.open_gripper(steps=5)
        env.robot.close_gripper(steps=5)

        print("PyBullet stacking env smoke test passed")
        print(f"RGB shape: {images['rgb'].shape}")
        print(f"Camera intrinsics shape: {env.camera_info.intrinsics.shape}")
        print(f"Red cube position: {red_position.tolist()}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
