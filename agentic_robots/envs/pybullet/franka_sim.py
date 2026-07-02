from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pybullet as p
import pybullet_data


ARM_JOINT_INDICES = tuple(range(7))
FINGER_JOINT_INDICES = (9, 10)
EEF_LINK_INDEX = 11
IK_LOWER_LIMITS = np.array(
    [-2.9671, -1.8326, -2.9671, -3.1416, -2.9671, -0.0873, -2.9671, 0.0, 0.0]
)
IK_UPPER_LIMITS = np.array(
    [2.9671, 1.8326, 2.9671, 0.0, 2.9671, 3.8223, 2.9671, 0.04, 0.04]
)
IK_JOINT_RANGES = IK_UPPER_LIMITS - IK_LOWER_LIMITS


@dataclass(frozen=True)
class FrankaPose:
    position: np.ndarray
    quaternion_wxyz: np.ndarray


class FrankaSim:
    """Small PyBullet Franka interface with action-style robot controls."""

    def __init__(
        self,
        *,
        client_id: int,
        base_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        base_orientation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        use_fixed_base: bool = True,
        frame_recorder: Callable[[], None] | None = None,
    ) -> None:
        self.client_id = client_id
        self.base_position = base_position
        self.base_orientation_xyzw = base_orientation_xyzw
        self.use_fixed_base = use_fixed_base
        self.frame_recorder = frame_recorder
        self.body_id: int | None = None
        self.home_joints = np.array(
            [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
            dtype=np.float64,
        )

    def load(self) -> int:
        """Load the Panda URDF and reset it to the default home pose."""
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        self.body_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=self.base_position,
            baseOrientation=self.base_orientation_xyzw,
            useFixedBase=self.use_fixed_base,
            flags=p.URDF_USE_SELF_COLLISION,
            physicsClientId=self.client_id,
        )
        self.reset()
        return self.body_id

    def reset(self) -> None:
        """Reset arm and gripper joints to a neutral state."""
        self._require_loaded()
        for joint_index, value in zip(ARM_JOINT_INDICES, self.home_joints):
            p.resetJointState(
                self.body_id,
                joint_index,
                float(value),
                targetVelocity=0.0,
                physicsClientId=self.client_id,
            )
        self.open_gripper(steps=1)

    def get_joint_positions(self) -> np.ndarray:
        """Return the 7 arm joint positions."""
        self._require_loaded()
        states = p.getJointStates(self.body_id, ARM_JOINT_INDICES, physicsClientId=self.client_id)
        return np.array([state[0] for state in states], dtype=np.float64)

    def get_gripper_positions(self) -> np.ndarray:
        """Return left/right finger joint positions."""
        self._require_loaded()
        states = p.getJointStates(
            self.body_id,
            FINGER_JOINT_INDICES,
            physicsClientId=self.client_id,
        )
        return np.array([state[0] for state in states], dtype=np.float64)

    def get_end_effector_pose(self) -> FrankaPose:
        """Return the end-effector pose as position and wxyz quaternion."""
        self._require_loaded()
        link_state = p.getLinkState(
            self.body_id,
            EEF_LINK_INDEX,
            computeForwardKinematics=True,
            physicsClientId=self.client_id,
        )
        position = np.array(link_state[4], dtype=np.float64)
        quaternion_wxyz = _xyzw_to_wxyz(np.array(link_state[5], dtype=np.float64))
        return FrankaPose(position=position, quaternion_wxyz=quaternion_wxyz)

    def solve_ik(self, position: np.ndarray, quaternion_wxyz: np.ndarray) -> np.ndarray:
        """Solve IK for the Panda grasp target link.

        Args:
            position: Target XYZ in world frame.
            quaternion_wxyz: Target orientation as [w, x, y, z].

        Returns:
            joints: 7 arm joint values.
        """
        self._require_loaded()
        target_position = np.asarray(position, dtype=np.float64).reshape(3)
        target_quaternion_xyzw = _wxyz_to_xyzw(
            np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        )
        solution = p.calculateInverseKinematics(
            self.body_id,
            EEF_LINK_INDEX,
            targetPosition=target_position.tolist(),
            targetOrientation=target_quaternion_xyzw.tolist(),
            lowerLimits=IK_LOWER_LIMITS.tolist(),
            upperLimits=IK_UPPER_LIMITS.tolist(),
            jointRanges=IK_JOINT_RANGES.tolist(),
            restPoses=[*self.home_joints.tolist(), 0.04, 0.04],
            maxNumIterations=1000,
            residualThreshold=1e-6,
            physicsClientId=self.client_id,
        )
        return np.asarray(solution[:7], dtype=np.float64)

    def plan_to_pose(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
        *,
        steps: int = 40,
    ) -> np.ndarray:
        """Return a simple joint-space interpolation to a target pose."""
        start = self.get_joint_positions()
        goal = self.solve_ik(position, quaternion_wxyz)
        if steps <= 1:
            return goal[None, :]
        alpha = np.linspace(0.0, 1.0, steps, dtype=np.float64)[:, None]
        return (1.0 - alpha) * start[None, :] + alpha * goal[None, :]

    def move_to_joints(
        self,
        joints: np.ndarray,
        *,
        steps: int = 120,
        max_force: float = 240.0,
    ) -> None:
        """Move the arm toward target joints using PyBullet position control."""
        self._require_loaded()
        target = np.asarray(joints, dtype=np.float64).reshape(7)
        p.setJointMotorControlArray(
            self.body_id,
            ARM_JOINT_INDICES,
            p.POSITION_CONTROL,
            targetPositions=target.tolist(),
            forces=[max_force] * len(ARM_JOINT_INDICES),
            physicsClientId=self.client_id,
        )
        self.step(steps)

    def move_to_pose(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
        *,
        plan_steps: int = 40,
        sim_steps_per_waypoint: int = 4,
    ) -> np.ndarray:
        """Plan a simple joint path to a pose and execute it."""
        trajectory = self.plan_to_pose(position, quaternion_wxyz, steps=plan_steps)
        for waypoint in trajectory:
            self.move_to_joints(waypoint, steps=sim_steps_per_waypoint)
        return trajectory

    def open_gripper(self, *, steps: int = 60) -> None:
        """Open both Panda fingers."""
        self._set_gripper(width=0.04, steps=steps)

    def close_gripper(self, *, steps: int = 60) -> None:
        """Close both Panda fingers."""
        self._set_gripper(width=0.0, steps=steps)

    def step(self, steps: int = 1) -> None:
        """Step the PyBullet simulation."""
        for _ in range(max(0, int(steps))):
            p.stepSimulation(physicsClientId=self.client_id)
        if self.frame_recorder is not None:
            self.frame_recorder()

    def _set_gripper(self, *, width: float, steps: int) -> None:
        self._require_loaded()
        p.setJointMotorControlArray(
            self.body_id,
            FINGER_JOINT_INDICES,
            p.POSITION_CONTROL,
            targetPositions=[width, width],
            forces=[40.0, 40.0],
            physicsClientId=self.client_id,
        )
        self.step(steps)

    def _require_loaded(self) -> None:
        if self.body_id is None:
            raise RuntimeError("FrankaSim.load() must be called before using the robot")


def _xyzw_to_wxyz(quaternion_xyzw: np.ndarray) -> np.ndarray:
    return np.array(
        [quaternion_xyzw[3], quaternion_xyzw[0], quaternion_xyzw[1], quaternion_xyzw[2]],
        dtype=np.float64,
    )


def _wxyz_to_xyzw(quaternion_wxyz: np.ndarray) -> np.ndarray:
    return np.array(
        [quaternion_wxyz[1], quaternion_wxyz[2], quaternion_wxyz[3], quaternion_wxyz[0]],
        dtype=np.float64,
    )
