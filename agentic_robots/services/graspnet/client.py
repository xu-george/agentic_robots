from __future__ import annotations

import os

import numpy as np

from agentic_robots.services.base import ServiceConfig, decode_numpy, encode_numpy, post_json
from agentic_robots.services.graspnet.types import GraspPlan


DEFAULT_GRASPNET_URL = os.environ.get("GRASPNET_SERVICE_URL", "http://127.0.0.1:8115")


class GraspNetClient:
    """HTTP client for Contact-GraspNet planning."""

    def __init__(self, url: str = DEFAULT_GRASPNET_URL, timeout_s: float = 120.0):
        self.config = ServiceConfig(url=url, timeout_s=timeout_s)

    def plan_from_depth(
        self,
        depth: np.ndarray,
        camera_intrinsics: np.ndarray,
        segmap: np.ndarray,
        segmap_id: int = 1,
        *,
        local_regions: bool = True,
        filter_grasps: bool = True,
        skip_border_objects: bool = False,
        z_range: tuple[float, float] = (0.2, 2.0),
        forward_passes: int = 2,
        max_retries: int = 10,
    ) -> GraspPlan:
        payload = {
            "depth_base64": encode_numpy(depth),
            "cam_K_base64": encode_numpy(camera_intrinsics),
            "segmap_base64": encode_numpy(segmap),
            "segmap_id": segmap_id,
            "local_regions": local_regions,
            "filter_grasps": filter_grasps,
            "skip_border_objects": skip_border_objects,
            "z_range": list(z_range),
            "forward_passes": forward_passes,
            "max_retries": max_retries,
        }
        return _decode_plan(post_json(self.config, "/plan", payload))

    def plan_from_point_clouds(
        self,
        pc_full: np.ndarray,
        pc_segment: np.ndarray,
        segmap_id: int = 1,
        *,
        local_regions: bool = True,
        filter_grasps: bool = True,
        forward_passes: int = 2,
        max_retries: int = 10,
    ) -> GraspPlan:
        payload = {
            "pc_full_base64": encode_numpy(pc_full),
            "pc_segment_base64": encode_numpy(pc_segment),
            "segmap_id": segmap_id,
            "local_regions": local_regions,
            "filter_grasps": filter_grasps,
            "forward_passes": forward_passes,
            "max_retries": max_retries,
        }
        return _decode_plan(post_json(self.config, "/plan_point_clouds", payload))


def _decode_plan(data: dict) -> GraspPlan:
    return GraspPlan(
        grasps=decode_numpy(data["grasps_base64"]),
        scores=decode_numpy(data["scores_base64"]),
        contact_points=decode_numpy(data["contact_pts_base64"]),
    )

