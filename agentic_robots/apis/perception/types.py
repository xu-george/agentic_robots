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
