from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from agentic_robots.apis.base import ApiBase
from agentic_robots.apis.perception.types import GraspPose, ObjectPose, OrientedBoundingBox
from agentic_robots.services.graspnet.client import GraspNetClient
from agentic_robots.services.graspnet.types import GraspPlan
from agentic_robots.services.sam3.client import Sam3Client
from agentic_robots.services.sam3.types import PointPromptResult, Sam3Mask

PANDA_GRASPTARGET_FROM_GRASPNET = np.array(
    [
        [0.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.112204],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


class PerceptionApi(ApiBase):
    """CAP-X-style perception API backed by SAM3 and GraspNet services."""

    def __init__(
        self,
        *,
        sam3_client: Sam3Client | None = None,
        graspnet_client: GraspNetClient | None = None,
        observation_provider: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self.sam3_client = sam3_client or Sam3Client()
        self.graspnet_client = graspnet_client or GraspNetClient()
        self._observation_provider = observation_provider

    def functions(self) -> dict[str, Callable[..., Any]]:
        return {
            "get_observation": self.get_observation,
            "segment_object": self.segment_object,
            "segment_point": self.segment_point,
            "get_object_pose": self.get_object_pose,
            "mask_to_3d_points": self.mask_to_3d_points,
            "get_oriented_bounding_box_from_3d_points": (
                self.get_oriented_bounding_box_from_3d_points
            ),
            "plan_grasp": self.plan_grasp,
            "get_best_grasp_pose": self.get_best_grasp_pose,
        }

    def get_observation(self) -> dict[str, Any]:
        """Get the latest RGB-D observation from the configured provider.

        Returns:
            observation:
                A dictionary containing camera data. Expected keys depend on the
                provider, but perception methods use:

                - rgb: np.ndarray of shape (H, W, 3), dtype uint8
                - depth: np.ndarray of shape (H, W), meters
                - intrinsics: np.ndarray of shape (3, 3)
                - camera_pose: optional np.ndarray of shape (4, 4)
        """
        if self._observation_provider is None:
            raise RuntimeError("No observation_provider was configured")
        return self._observation_provider()

    def segment_object(self, rgb: np.ndarray, text_prompt: str) -> list[Sam3Mask]:
        """Segment objects in an RGB image from a text prompt using SAM3.

        Args:
            rgb:
                RGB image array of shape (H, W, 3), dtype uint8.
            text_prompt:
                Natural language object prompt, for example "bowl" or "red cube".

        Returns:
            masks:
                List of SAM3 masks. Each item contains mask, bounding box,
                confidence score, and label.
        """
        return self.sam3_client.segment(rgb, text_prompt)

    def segment_point(
        self,
        rgb: np.ndarray,
        point_xy: tuple[float, float],
    ) -> list[PointPromptResult]:
        """Segment objects in an RGB image from an image point prompt using SAM3.

        Args:
            rgb:
                RGB image array of shape (H, W, 3), dtype uint8.
            point_xy:
                Pixel coordinate as (x, y).

        Returns:
            masks:
                List of point-prompt segmentation masks and scores.
        """
        return self.sam3_client.segment_point(rgb, point_xy)

    def get_object_pose(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        object_name: str,
        *,
        camera_pose: np.ndarray | None = None,
        min_depth_m: float = 0.05,
    ) -> ObjectPose:
        """Estimate a named object's 3D pose from RGB-D data.

        This is perception-only. It segments the object with SAM3, projects the
        selected mask through the depth map, then returns the object centroid and
        PCA-oriented bounding box. If camera_pose is supplied, output points and
        position are transformed into that world frame; otherwise they are in the
        camera frame.

        Args:
            rgb:
                RGB image array of shape (H, W, 3), dtype uint8.
            depth:
                Depth image in meters, shape (H, W) or (H, W, 1).
            intrinsics:
                Camera intrinsic matrix, shape (3, 3).
            object_name:
                Natural language object name to segment.
            camera_pose:
                Optional camera-to-world transform, shape (4, 4).
            min_depth_m:
                Minimum valid depth in meters.

        Returns:
            pose:
                ObjectPose with centroid position, orientation quaternion wxyz,
                selected segmentation mask, 3D points, and oriented bounding box.
        """
        masks = self.segment_object(rgb, object_name)
        if not masks:
            raise RuntimeError(f"No SAM3 masks found for object '{object_name}'")

        best = max(masks, key=lambda item: item.score)
        points = self.mask_to_3d_points(
            depth=depth,
            intrinsics=intrinsics,
            mask=best.mask,
            camera_pose=camera_pose,
            min_depth_m=min_depth_m,
        )
        if points.size == 0:
            raise RuntimeError(f"No valid depth points found for object '{object_name}'")

        obb = self.get_oriented_bounding_box_from_3d_points(points)
        return ObjectPose(
            object_name=object_name,
            position=obb.center,
            orientation_wxyz=_rotation_matrix_to_quaternion_wxyz(obb.rotation),
            mask=best.mask,
            box=best.box,
            score=best.score,
            points=points,
            oriented_bounding_box=obb,
        )

    def mask_to_3d_points(
        self,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        mask: np.ndarray,
        *,
        camera_pose: np.ndarray | None = None,
        min_depth_m: float = 0.05,
    ) -> np.ndarray:
        """Project masked depth pixels into a 3D point cloud.

        Args:
            depth:
                Depth image in meters, shape (H, W) or (H, W, 1).
            intrinsics:
                Camera intrinsic matrix, shape (3, 3).
            mask:
                Boolean object mask, shape (H, W).
            camera_pose:
                Optional camera-to-world transform, shape (4, 4).
            min_depth_m:
                Minimum valid depth in meters.

        Returns:
            points:
                Array of shape (N, 3). Points are in camera frame unless
                camera_pose is provided.
        """
        depth_2d = np.asarray(depth)
        if depth_2d.ndim == 3 and depth_2d.shape[-1] == 1:
            depth_2d = depth_2d[:, :, 0]
        if depth_2d.ndim != 2:
            raise ValueError(f"depth must have shape (H, W), got {depth_2d.shape}")

        mask_bool = np.asarray(mask).astype(bool)
        if mask_bool.shape != depth_2d.shape:
            raise ValueError(
                f"mask shape {mask_bool.shape} must match depth shape {depth_2d.shape}"
            )

        valid = mask_bool & np.isfinite(depth_2d) & (depth_2d > min_depth_m)
        v, u = np.nonzero(valid)
        z = depth_2d[v, u].astype(np.float64)
        if z.size == 0:
            return np.empty((0, 3), dtype=np.float64)

        k = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
        fx, fy = k[0, 0], k[1, 1]
        cx, cy = k[0, 2], k[1, 2]

        x = (u.astype(np.float64) - cx) * z / fx
        y = (v.astype(np.float64) - cy) * z / fy
        points = np.column_stack([x, y, z])

        if camera_pose is None:
            return points

        pose = np.asarray(camera_pose, dtype=np.float64).reshape(4, 4)
        points_h = np.column_stack([points, np.ones(points.shape[0])])
        return (pose @ points_h.T).T[:, :3]

    def get_oriented_bounding_box_from_3d_points(
        self,
        points: np.ndarray,
    ) -> OrientedBoundingBox:
        """Estimate an oriented bounding box from 3D points using PCA.

        Args:
            points:
                Point cloud array of shape (N, 3).

        Returns:
            oriented_bounding_box:
                Center, extent, and rotation matrix. Rotation columns are the
                principal axes of the point cloud.
        """
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if pts.shape[0] == 0:
            raise ValueError("points must contain at least one point")

        center = pts.mean(axis=0)
        if pts.shape[0] < 3:
            rotation = np.eye(3, dtype=np.float64)
        else:
            _, _, vh = np.linalg.svd(pts - center, full_matrices=False)
            rotation = vh.T
            if np.linalg.det(rotation) < 0:
                rotation[:, -1] *= -1.0

        local = (pts - center) @ rotation
        min_corner = local.min(axis=0)
        max_corner = local.max(axis=0)
        extent = max_corner - min_corner
        local_center = 0.5 * (min_corner + max_corner)
        world_center = center + local_center @ rotation.T
        return OrientedBoundingBox(center=world_center, extent=extent, rotation=rotation)

    def plan_grasp(
        self,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        segmentation: Any,
        *,
        segmap_id: int = 1,
        z_range: tuple[float, float] = (0.2, 2.0),
        forward_passes: int = 4,
        max_retries: int = 10,
    ) -> GraspPlan:
        """Plan grasp candidates for one segmented object with GraspNet.

        Args:
            depth:
                Depth image in meters, shape (H, W).
            intrinsics:
                Camera intrinsic matrix, shape (3, 3).
            segmentation:
                Instance segmentation map. Pixels equal to segmap_id are treated
                as the target object.
            segmap_id:
                Target object ID in the segmentation map.
            z_range:
                Valid depth range for Contact-GraspNet.
            forward_passes:
                Number of Contact-GraspNet forward passes.
            max_retries:
                Service retry count for robust planning.

        Returns:
            grasp_plan:
                GraspPlan with 4x4 grasp transforms, scores, and contact points.
        """
        segmap = _coerce_segmentation(segmentation)
        if segmap.dtype == bool:
            segmap = segmap.astype(np.int32)
            segmap_id = 1

        return self.graspnet_client.plan_from_depth(
            depth=np.asarray(depth),
            camera_intrinsics=np.asarray(intrinsics),
            segmap=segmap,
            segmap_id=segmap_id,
            z_range=z_range,
            forward_passes=forward_passes,
            max_retries=max_retries,
        )

    def get_best_grasp_pose(
        self,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        segmentation: Any,
        camera_pose: np.ndarray | None = None,
        *,
        segmap_id: int = 1,
        z_range: tuple[float, float] = (0.2, 2.0),
        forward_passes: int = 4,
        max_retries: int = 10,
    ) -> GraspPose:
        """Return the highest-scoring GraspNet 6-DoF grasp as a pose.

        The returned pose is in the camera frame by default. If `camera_pose`
        is supplied, the grasp transform is converted into that world frame.
        """
        plan = self.plan_grasp(
            depth=depth,
            intrinsics=intrinsics,
            segmentation=segmentation,
            segmap_id=segmap_id,
            z_range=z_range,
            forward_passes=forward_passes,
            max_retries=max_retries,
        )
        if plan.empty:
            raise RuntimeError("GraspNet returned no grasp candidates")

        best_idx = int(np.argmax(plan.scores)) if plan.scores.size else 0
        if camera_pose is None and self._observation_provider is not None:
            observation = self._observation_provider()
            camera_pose = observation.get("camera_pose")
        if plan.contact_points.size:
            best_idx = _select_contact_grasp_index(
                depth=np.asarray(depth),
                intrinsics=np.asarray(intrinsics),
                segmentation=segmentation,
                plan=plan,
                camera_pose=camera_pose,
                fallback_index=best_idx,
            )

        graspnet_transform = np.asarray(plan.grasps[best_idx], dtype=np.float64).reshape(4, 4)
        if camera_pose is not None:
            camera_pose = np.asarray(camera_pose, dtype=np.float64).reshape(4, 4)
            graspnet_transform = camera_pose @ graspnet_transform

        transform = graspnet_transform @ PANDA_GRASPTARGET_FROM_GRASPNET

        return GraspPose(
            position=transform[:3, 3].copy(),
            orientation_wxyz=_rotation_matrix_to_quaternion_wxyz(transform[:3, :3]),
            score=float(plan.scores[best_idx]) if plan.scores.size else 0.0,
            transform=transform,
            plan=plan,
            graspnet_transform=graspnet_transform,
        )


def _rotation_matrix_to_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a normalized wxyz quaternion."""
    r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(r))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r[2, 1] - r[1, 2]) / s
        qy = (r[0, 2] - r[2, 0]) / s
        qz = (r[1, 0] - r[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(r)))
        if idx == 0:
            s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            qw = (r[2, 1] - r[1, 2]) / s
            qx = 0.25 * s
            qy = (r[0, 1] + r[1, 0]) / s
            qz = (r[0, 2] + r[2, 0]) / s
        elif idx == 1:
            s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            qw = (r[0, 2] - r[2, 0]) / s
            qx = (r[0, 1] + r[1, 0]) / s
            qy = 0.25 * s
            qz = (r[1, 2] + r[2, 1]) / s
        else:
            s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            qw = (r[1, 0] - r[0, 1]) / s
            qx = (r[0, 2] + r[2, 0]) / s
            qy = (r[1, 2] + r[2, 1]) / s
            qz = 0.25 * s

    quat = np.array([qw, qx, qy, qz], dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def _coerce_segmentation(segmentation: Any) -> np.ndarray:
    if isinstance(segmentation, list):
        if not segmentation:
            raise ValueError("segmentation list is empty")
        segmentation = segmentation[0]
    if hasattr(segmentation, "mask"):
        segmentation = segmentation.mask
    return np.asarray(segmentation)


def _select_contact_grasp_index(
    *,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    segmentation: Any,
    plan: GraspPlan,
    camera_pose: np.ndarray | None,
    fallback_index: int,
) -> int:
    """Prefer contact points near the segmented object while keeping score as a tiebreaker."""
    try:
        mask = _coerce_segmentation(segmentation).astype(bool)
        depth_2d = np.asarray(depth)
        if depth_2d.ndim == 3 and depth_2d.shape[-1] == 1:
            depth_2d = depth_2d[:, :, 0]
        valid = mask & np.isfinite(depth_2d) & (depth_2d > 0.05)
        v, u = np.nonzero(valid)
        if u.size == 0:
            return fallback_index

        z = depth_2d[v, u].astype(np.float64)
        k = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
        x = (u.astype(np.float64) - k[0, 2]) * z / k[0, 0]
        y = (v.astype(np.float64) - k[1, 2]) * z / k[1, 1]
        centroid = np.column_stack([x, y, z]).mean(axis=0)
        contacts = np.asarray(plan.contact_points, dtype=np.float64).reshape(-1, 3)
        if camera_pose is not None:
            pose = np.asarray(camera_pose, dtype=np.float64).reshape(4, 4)
            contacts_h = np.column_stack([contacts, np.ones(contacts.shape[0])])
            contacts = (pose @ contacts_h.T).T[:, :3]
            centroid = (pose @ np.append(centroid, 1.0))[:3]

        scores = np.asarray(plan.scores, dtype=np.float64).reshape(-1)
        count = min(len(contacts), len(scores))
        if count == 0:
            return fallback_index
        contacts = contacts[:count]
        scores = scores[:count]
        distances = np.linalg.norm(contacts - centroid, axis=1)
        score_span = max(float(scores.max() - scores.min()), 1e-8)
        normalized_scores = (scores - float(scores.min())) / score_span
        return int(np.argmin(distances - 0.03 * normalized_scores))
    except Exception:
        return fallback_index
