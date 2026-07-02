from __future__ import annotations

from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def panda_control_points() -> np.ndarray:
    """Return Panda gripper control points, falling back to calibrated constants."""
    try:
        import sys

        graspnet_root = PROJECT_ROOT / "third_party" / "contact_graspnet_pytorch"
        if str(graspnet_root) not in sys.path:
            sys.path.append(str(graspnet_root))
        import contact_graspnet_pytorch.mesh_utils as mesh_utils

        gripper = mesh_utils.create_gripper("panda")
        return np.asarray(
            gripper.get_control_point_tensor(1, False, convex_hull=False).squeeze(),
            dtype=np.float64,
        )
    except Exception:
        return np.array(
            [
                [0.0, 0.0, 0.0],
                [0.04, 0.0, 0.058],
                [-0.04, 0.0, 0.058],
                [0.04, 0.0, 0.112],
                [-0.04, 0.0, 0.112],
            ],
            dtype=np.float64,
        )


def panda_gripper_wire_points(grasp: np.ndarray, *, opening: float) -> np.ndarray:
    """Build a 7-point two-finger Panda gripper polyline in world/camera space."""
    grasp_transform = np.asarray(grasp, dtype=np.float64).reshape(4, 4)
    control_points = panda_control_points()
    z_base = float(np.abs(control_points[1, 2]))
    z_tip = float(np.abs(control_points[3, 2]))
    half = float(opening) * 0.5
    line_points = np.array(
        [
            [-half, 0.0, z_tip],
            [-half, 0.0, z_base],
            [0.0, 0.0, z_base],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, z_base],
            [half, 0.0, z_base],
            [half, 0.0, z_tip],
        ],
        dtype=np.float64,
    )
    return line_points @ grasp_transform[:3, :3].T + grasp_transform[:3, 3]
