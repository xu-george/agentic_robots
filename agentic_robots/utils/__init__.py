"""Visualization and API schema utilities for PyBullet simulation."""

from __future__ import annotations

from agentic_robots.utils.api_schema import (
    build_api_description,
    format_api_description,
    load_api_schema,
)
from agentic_robots.utils.gripper_geometry import panda_gripper_wire_points
from agentic_robots.utils.transforms import quaternion_wxyz_to_rotation_matrix

from typing import Any

import numpy as np

try:
    import pybullet as p
except ImportError:
    p = None


_panda_gripper_wire_points = panda_gripper_wire_points

__all__ = [
    "build_api_description",
    "clear_debug",
    "draw_grasp_axes",
    "draw_grasp_candidates",
    "draw_gripper_wireframe",
    "format_api_description",
    "load_api_schema",
    "panda_gripper_wire_points",
    "show_grasp",
]


def draw_grasp_axes(
    client_id: int,
    position: np.ndarray,
    quaternion_wxyz: np.ndarray,
    *,
    axis_length: float = 0.08,
    line_width: float = 3.0,
    life_time: float = 0.0,
    label: str = "grasp",
    item_ids: list[int] | None = None,
) -> list[int]:
    """Draw RGB coordinate axes at a grasp pose in the PyBullet simulation.

    Args:
        client_id: PyBullet physics client ID.
        position: 3D world position.
        quaternion_wxyz: Orientation as [w, x, y, z].
        axis_length: Length of each axis in meters.
        line_width: Line width in pixels.
        life_time: How long the debug lines persist (0 = forever).
        label: Text label placed above the axes.
        item_ids: Optional list to append debug item IDs into (for later cleanup).

    Returns:
        List of PyBullet debug item IDs.
    """
    origin = np.asarray(position, dtype=np.float64).reshape(3)
    rotation = quaternion_wxyz_to_rotation_matrix(quaternion_wxyz)
    axis_colors = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.1, 0.35, 1.0))
    ids: list[int] = []

    for axis_index, color in enumerate(axis_colors):
        end = origin + rotation[:, axis_index] * axis_length
        ids.append(
            p.addUserDebugLine(
                origin.tolist(),
                end.tolist(),
                lineColorRGB=color,
                lineWidth=line_width,
                lifeTime=life_time,
                physicsClientId=client_id,
            )
        )

    cross = axis_length * 0.12
    ids.extend([
        p.addUserDebugLine(
            (origin + np.array([-cross, 0.0, 0.0])).tolist(),
            (origin + np.array([cross, 0.0, 0.0])).tolist(),
            lineColorRGB=(1.0, 1.0, 0.0),
            lineWidth=line_width,
            lifeTime=life_time,
            physicsClientId=client_id,
        ),
        p.addUserDebugLine(
            (origin + np.array([0.0, -cross, 0.0])).tolist(),
            (origin + np.array([0.0, cross, 0.0])).tolist(),
            lineColorRGB=(1.0, 1.0, 0.0),
            lineWidth=line_width,
            lifeTime=life_time,
            physicsClientId=client_id,
        ),
        p.addUserDebugText(
            label,
            (origin + np.array([0.0, 0.0, axis_length * 0.35])).tolist(),
            textColorRGB=(1.0, 1.0, 0.0),
            textSize=1.2,
            lifeTime=life_time,
            physicsClientId=client_id,
        ),
    ])

    if item_ids is not None:
        item_ids.extend(ids)
    return ids


def draw_gripper_wireframe(
    client_id: int,
    transform: np.ndarray,
    *,
    opening: float = 0.08,
    color: tuple[float, float, float] = (0.1, 1.0, 0.0),
    line_width: float = 3.0,
    life_time: float = 0.0,
    label: str | None = None,
    item_ids: list[int] | None = None,
) -> list[int]:
    """Draw a two-tine fork (two-finger gripper gesture) at a 4x4 transform in the PyBullet sim.

    Args:
        client_id: PyBullet physics client ID.
        transform: 4x4 grasp transform (world frame).
        opening: Gripper opening width in meters.
        color: RGB color of the wireframe.
        line_width: Line width in pixels.
        life_time: How long the debug lines persist (0 = forever).
        label: Optional text label above the gripper.
        item_ids: Optional list to append debug item IDs into.

    Returns:
        List of PyBullet debug item IDs.
    """
    grasp = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    points = _panda_gripper_wire_points(grasp, opening=opening)
    segments = tuple((i, i + 1) for i in range(len(points) - 1))
    ids: list[int] = []

    for start, end in segments:
        ids.append(
            p.addUserDebugLine(
                points[start].tolist(),
                points[end].tolist(),
                lineColorRGB=color,
                lineWidth=line_width,
                lifeTime=life_time,
                physicsClientId=client_id,
            )
        )

    if label is not None:
        ids.append(
            p.addUserDebugText(
                label,
                (grasp[:3, 3] + np.array([0.0, 0.0, 0.035])).tolist(),
                textColorRGB=color,
                textSize=1.2,
                lifeTime=life_time,
                physicsClientId=client_id,
            )
        )

    if item_ids is not None:
        item_ids.extend(ids)
    return ids


def draw_grasp_candidates(
    client_id: int,
    transforms: np.ndarray,
    *,
    scores: np.ndarray | None = None,
    best_transform: np.ndarray | None = None,
    max_grasps: int = 5,
    opening: float = 0.08,
    line_width: float = 1.5,
    life_time: float = 0.0,
    item_ids: list[int] | None = None,
) -> list[int]:
    """Draw top-N candidate gripper wireframes in the PyBullet sim.

    Args:
        client_id: PyBullet physics client ID.
        transforms: Array of 4x4 grasp transforms, shape (N, 4, 4).
        scores: Optional array of scores, shape (N,).
        best_transform: Optional single 4x4 transform to highlight in green.
        max_grasps: Maximum number of candidates to draw.
        opening: Gripper opening width in meters.
        line_width: Line width in pixels.
        life_time: How long the debug lines persist (0 = forever).
        item_ids: Optional list to append debug item IDs into.

    Returns:
        List of PyBullet debug item IDs.
    """
    grasps = np.asarray(transforms, dtype=np.float64).reshape(-1, 4, 4)
    if grasps.size == 0:
        return []

    ids: list[int] = []

    if scores is not None:
        scores_arr = np.asarray(scores, dtype=np.float64).reshape(-1)
        if scores_arr.size == grasps.shape[0]:
            order = np.argsort(scores_arr)[::-1]
            selected = order[: max(1, min(max_grasps, grasps.shape[0]))]
        else:
            selected = np.arange(min(max_grasps, grasps.shape[0]))
    else:
        selected = np.arange(min(max_grasps, grasps.shape[0]))

    for rank, grasp_index in enumerate(selected):
        color = _candidate_color(rank, len(selected))
        draw_gripper_wireframe(
            client_id,
            grasps[grasp_index],
            opening=opening,
            color=color,
            line_width=line_width,
            life_time=life_time,
            item_ids=ids,
        )

    if best_transform is not None:
        best = np.asarray(best_transform, dtype=np.float64).reshape(4, 4)
        draw_gripper_wireframe(
            client_id,
            best,
            opening=opening,
            color=(0.1, 1.0, 0.0),
            line_width=max(4.0, line_width * 3.0),
            life_time=life_time,
            label="best",
            item_ids=ids,
        )

    if item_ids is not None:
        item_ids.extend(ids)
    return ids


def show_grasp(
    client_id: int,
    grasp_pose: Any,
    *,
    draw_axes: bool = True,
    draw_candidates: bool = True,
    max_candidates: int = 5,
    opening: float = 0.08,
    life_time: float = 0.0,
    clear_previous: bool = True,
) -> list[int]:
    """Convenience: draw a GraspPose (best pose + candidate wireframes) in the sim.

    Args:
        client_id: PyBullet physics client ID.
        grasp_pose: A GraspPose object from get_finger_grasp().
        draw_axes: Whether to draw RGB axes at the best grasp.
        draw_candidates: Whether to draw candidate wireframes from the plan.
        max_candidates: Max candidates to draw from the plan.
        opening: Gripper opening width in meters.
        life_time: How long the debug lines persist (0 = forever).
        clear_previous: If True, remove all prior debug items with the same client_id.

    Returns:
        List of PyBullet debug item IDs.
    """
    ids: list[int] = []

    if clear_previous:
        p.removeAllUserDebugItems(physicsClientId=client_id)

    if draw_candidates:
        plan = getattr(grasp_pose, "plan", None)
        if plan is not None:
            grasps_arr = getattr(plan, "grasps", None)
            scores_arr = getattr(plan, "scores", None)
            if grasps_arr is not None and grasps_arr.size:
                # plan.grasps are in the CAMERA frame; transform them into the
                # world frame so candidate wireframes land at the object in the
                # PyBullet sim instead of near the world origin. Recover the
                # camera pose from the best raw grasp: graspnet_transform ==
                # camera_pose @ plan.grasps[best_idx]. The fork wireframe is
                # defined in the GraspNet gripper frame, so draw the best at
                # grasp_pose.graspnet_transform (not the panda command frame).
                world_grasps = _world_frame_candidates(
                    grasps_arr, grasp_pose, scores_arr
                )
                best_world = (
                    grasp_pose.graspnet_transform
                    if getattr(grasp_pose, "graspnet_transform", None) is not None
                    else grasp_pose.transform
                )
                draw_grasp_candidates(
                    client_id,
                    world_grasps,
                    scores=scores_arr,
                    best_transform=best_world,
                    max_grasps=max_candidates,
                    opening=opening,
                    life_time=life_time,
                    item_ids=ids,
                )

    if draw_axes:
        draw_grasp_axes(
            client_id,
            grasp_pose.position,
            grasp_pose.quaternion_wxyz,
            life_time=life_time,
            label="best grasp",
            item_ids=ids,
        )

    return ids


def clear_debug(client_id: int) -> None:
    """Remove all user debug items from the PyBullet simulation."""
    p.removeAllUserDebugItems(physicsClientId=client_id)


def _world_frame_candidates(
    grasps_arr: np.ndarray,
    grasp_pose: Any,
    scores_arr: np.ndarray | None,
) -> np.ndarray:
    """Transform camera-frame plan grasps into world-frame grasp transforms.

    ``plan.grasps`` are stored in the camera frame. ``grasp_pose.graspnet_transform``
    is the world-frame raw grasp == ``camera_pose @ plan.grasps[best_idx]``, so
    the camera pose is recovered as
    ``grasp_pose.graspnet_transform @ inv(plan.grasps[best_idx])`` and applied to
    every candidate. Falls back to the raw camera-frame grasps when the camera
    pose cannot be recovered (e.g. missing graspnet_transform or degenerate
    best grasp), preserving prior behavior.
    """
    grasps = np.asarray(grasps_arr, dtype=np.float64).reshape(-1, 4, 4)
    graspnet_transform = getattr(grasp_pose, "graspnet_transform", None)
    if graspnet_transform is None:
        return grasps
    scores = (
        np.asarray(scores_arr, dtype=np.float64).reshape(-1)
        if scores_arr is not None
        else None
    )
    if scores is not None and scores.size == grasps.shape[0]:
        best_idx = int(np.argmax(scores))
    else:
        best_idx = 0
    best_cam = grasps[best_idx]
    try:
        camera_pose = np.asarray(graspnet_transform, dtype=np.float64).reshape(4, 4) @ np.linalg.inv(
            best_cam
        )
    except np.linalg.LinAlgError:
        return grasps
    world = camera_pose @ grasps
    return world.reshape(-1, 4, 4)


def _candidate_color(rank: int, count: int) -> tuple[float, float, float]:
    if count <= 1:
        value = 0.0
    else:
        value = rank / float(count - 1)
    palette = (
        np.array([0.0, 1.0, 0.75], dtype=np.float64),
        np.array([0.1, 0.4, 1.0], dtype=np.float64),
        np.array([0.95, 0.8, 0.35], dtype=np.float64),
        np.array([1.0, 0.25, 0.0], dtype=np.float64),
        np.array([0.55, 0.0, 1.0], dtype=np.float64),
    )
    scaled = value * (len(palette) - 1)
    lower = int(np.floor(scaled))
    upper = min(lower + 1, len(palette) - 1)
    alpha = scaled - lower
    color = palette[lower] * (1.0 - alpha) + palette[upper] * alpha
    return tuple(float(c) for c in color)
