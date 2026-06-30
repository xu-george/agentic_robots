from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class FakeSam3Client:
    def segment(self, image, prompt):
        from agentic_robots.services.sam3.types import Sam3Mask

        mask = np.zeros(image.shape[:2], dtype=bool)
        mask[1:3, 1:3] = True
        return [
            Sam3Mask(
                mask=mask,
                box=(1.0, 1.0, 2.0, 2.0),
                score=0.9,
                label=prompt,
            )
        ]

    def segment_point(self, image, point_xy):
        from agentic_robots.services.sam3.types import PointPromptResult

        mask = np.zeros(image.shape[:2], dtype=bool)
        mask[int(point_xy[1]), int(point_xy[0])] = True
        return [PointPromptResult(mask=mask, score=1.0)]


class FakeGraspNetClient:
    def plan_from_depth(self, depth, camera_intrinsics, segmap, **kwargs):
        from agentic_robots.services.graspnet.types import GraspPlan

        return GraspPlan(
            grasps=np.eye(4, dtype=np.float64)[None, :, :],
            scores=np.array([0.75], dtype=np.float64),
            contact_points=np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
        )


def main() -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from agentic_robots.apis.perception import PerceptionApi
    from agentic_robots.apis.perception.api import (
        PANDA_GRASPTARGET_FROM_GRASPNET,
        PANDA_GRASPTARGET_OFFSET_M,
    )

    api = PerceptionApi(
        sam3_client=FakeSam3Client(),
        graspnet_client=FakeGraspNetClient(),
    )
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    depth = np.ones((4, 4), dtype=np.float64)
    intrinsics = np.array(
        [
            [100.0, 0.0, 1.5],
            [0.0, 100.0, 1.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    segments = api.segment_object(rgb, "cube")
    assert len(segments) == 1
    assert segments[0].label == "cube"

    pose = api.get_object_pose(rgb, depth, intrinsics, "cube")
    assert pose.object_name == "cube"
    assert pose.points.shape == (4, 3)
    assert pose.position.shape == (3,)
    assert pose.orientation_wxyz.shape == (4,)

    plan = api.plan_grasp(depth, intrinsics, pose.mask)
    assert plan.grasps.shape == (1, 4, 4)
    assert plan.best_grasp is not None
    grasp_pose = api.get_best_grasp_pose(depth, intrinsics, pose.mask)
    assert grasp_pose.position.shape == (3,)
    assert grasp_pose.orientation_wxyz.shape == (4,)
    assert grasp_pose.transform.shape == (4, 4)
    assert grasp_pose.frame == "panda_grasp_point"
    assert np.allclose(grasp_pose.raw_graspnet_position, np.zeros(3))
    assert np.allclose(grasp_pose.position, PANDA_GRASPTARGET_OFFSET_M)
    assert np.allclose(grasp_pose.transform, PANDA_GRASPTARGET_FROM_GRASPNET)
    assert np.allclose(grasp_pose.grasp_target_offset_m, PANDA_GRASPTARGET_OFFSET_M)

    raw_grasp_pose = api.get_best_grasp_pose(
        depth,
        intrinsics,
        pose.mask,
        apply_grasp_target_offset=False,
    )
    assert raw_grasp_pose.frame == "graspnet_gripper"
    assert np.allclose(raw_grasp_pose.position, np.zeros(3))
    assert np.allclose(raw_grasp_pose.grasp_target_offset_m, np.zeros(3))

    docs = api.combined_doc()
    assert "segment_object" in docs
    assert "get_object_pose" in docs
    assert "plan_grasp" in docs
    assert "get_best_grasp_pose" in docs

    print("Perception API smoke test passed")


if __name__ == "__main__":
    main()
