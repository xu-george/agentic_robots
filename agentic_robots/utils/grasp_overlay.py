from __future__ import annotations

import cv2
import numpy as np

from agentic_robots.utils.gripper_geometry import panda_gripper_wire_points


def draw_grasp_overlay(
    *,
    rgb: np.ndarray,
    mask: np.ndarray | None,
    grasps: np.ndarray,
    scores: np.ndarray,
    intrinsics: np.ndarray,
    top_k: int,
) -> np.ndarray:
    """Draw top grasp hypotheses and an optional segmentation mask on an RGB image."""
    image = np.asarray(rgb, dtype=np.uint8).copy()
    if mask is not None:
        image = blend_mask(image, np.asarray(mask, dtype=bool))

    grasp_array = np.asarray(grasps, dtype=np.float64).reshape(-1, 4, 4)
    score_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    if score_array.size == len(grasp_array):
        order = np.argsort(score_array)[::-1]
    else:
        order = np.arange(len(grasp_array))

    selected = order[: max(1, min(top_k, len(order)))]
    for rank, grasp_index in enumerate(selected[::-1]):
        is_best = grasp_index == order[0]
        color = (255, 40, 40) if is_best else rank_color(rank, len(selected))
        draw_grasp(
            image,
            grasp_array[int(grasp_index)],
            intrinsics,
            color=color,
            width=4 if is_best else 2,
        )
    return image


def blend_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = rgb.copy()
    overlay[mask] = (0.55 * overlay[mask] + 0.45 * np.array([40, 190, 255])).astype(np.uint8)
    return overlay


def draw_grasp(
    image: np.ndarray,
    grasp: np.ndarray,
    intrinsics: np.ndarray,
    *,
    color: tuple[int, int, int],
    width: int,
) -> None:
    points = panda_gripper_wire_points(grasp, opening=0.08)
    uv, valid = project_points(points, intrinsics, image.shape[:2])
    for start, end in zip(range(0, 6), range(1, 7)):
        if valid[start] and valid[end]:
            cv2.line(
                image,
                tuple(np.round(uv[start]).astype(int)),
                tuple(np.round(uv[end]).astype(int)),
                color,
                width,
                cv2.LINE_AA,
            )


def project_points(
    points: np.ndarray,
    intrinsics: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image_shape
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    uvw = (np.asarray(intrinsics, dtype=np.float64).reshape(3, 3) @ points.T).T
    uv = uvw[:, :2] / np.maximum(uvw[:, 2:3], 1e-8)
    valid = (
        (points[:, 2] > 0.0)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )
    return uv, valid


def rank_color(rank: int, count: int) -> tuple[int, int, int]:
    value = rank / max(count - 1, 1)
    return (int(40 + 180 * value), int(230 * (1.0 - value)), 255)
