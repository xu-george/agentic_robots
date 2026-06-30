from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OrientedBoundingBox:
    """Object bounding box estimated from 3D points."""

    center: np.ndarray
    extent: np.ndarray
    rotation: np.ndarray


@dataclass(frozen=True)
class ObjectPose:
    """3D object pose estimate derived from an RGB-D segmentation mask."""

    object_name: str
    position: np.ndarray
    orientation_wxyz: np.ndarray
    mask: np.ndarray
    box: tuple[float, float, float, float]
    score: float
    points: np.ndarray
    oriented_bounding_box: OrientedBoundingBox


@dataclass(frozen=True)
class GraspPose:
    """6-DoF grasp pose selected from a GraspNet plan.

    `graspnet_transform` stores the raw Contact-GraspNet gripper frame. The
    `transform` field stores the returned command frame, usually the project
    grasp point between the Panda finger tips.
    """

    position: np.ndarray
    orientation_wxyz: np.ndarray
    score: float
    transform: np.ndarray
    plan: object
    graspnet_transform: np.ndarray | None = None
    frame: str = "panda_grasp_point"
    grasp_target_offset_m: np.ndarray | None = None

    @property
    def quaternion_wxyz(self) -> np.ndarray:
        """Alias that matches get_end_effector_pose().quaternion_wxyz."""
        return self.orientation_wxyz

    @property
    def raw_graspnet_position(self) -> np.ndarray | None:
        """Raw Contact-GraspNet gripper-frame origin in the output/world frame."""
        if self.graspnet_transform is None:
            return None
        return self.graspnet_transform[:3, 3].copy()
