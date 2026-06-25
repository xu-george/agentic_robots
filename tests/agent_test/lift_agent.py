from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tests" / "agent_test" / "outputs" / "lift_agent"
DEFAULT_OBJECT_CONFIG = PROJECT_ROOT / "configs" / "lift_objects.yaml"

LIFT_SYSTEM_PROMPT = """You write short Python robot task programs.
Use only the provided perception and action functions. Do not import modules or define helper functions.
get_best_grasp_pose(...) returns a grasp object, not a dict.
Use grasp.position and grasp.quaternion_wxyz; never use grasp["position"] or grasp["quaternion_wxyz"].
Positions are usually numpy arrays, so vector addition with small offset lists is allowed.
Return only executable Python code, no markdown, no explanation.
"""

LIFT_API_DESCRIPTION = """Perception:
- get_observation() -> observation dict with rgb, depth, intrinsics, camera_pose
- segment_object(rgb, text_prompt) -> masks for the requested object
- get_best_grasp_pose(depth, intrinsics, segmentation, camera_pose) -> grasp object
  with position, quaternion_wxyz, score, and transform fields
Action:
- get_end_effector_pose() -> current robot pose with position and quaternion_wxyz
- move_to_pose(position, quaternion_wxyz, plan_steps=40, sim_steps_per_waypoint=4)
- open_gripper()
- close_gripper()
"""

LIFT_REVISION_GUIDANCE = (
    "Revise the lift code. Use the same grasp result. "
    "Do this sequence: open gripper, move above grasp, move to grasp, "
    "close gripper, then move upward to lift. "
    "Do not import modules or stop after closing the gripper."
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the YCB lift task with the general code agent.")
    parser.add_argument("--object", default="banana")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--fake-perception", action="store_true")
    parser.add_argument("--show-grasp-pose", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--hold-s", type=float, default=0.0)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from agentic_robots.agent import (
        CodeAgent,
        CodeAgentConfig,
        make_robot_diagnostics,
        make_scene_summary,
    )
    from agentic_robots.apis.action import ActionApi
    from agentic_robots.apis.perception import PerceptionApi
    from agentic_robots.envs.pybullet import LiftEnv

    object_prompt = _load_object_prompt(args.object)
    env = LiftEnv(gui=args.gui, object_name=args.object)
    try:
        env.reset()
        video_path = None
        if args.record_video:
            video_path = env.start_video_recording(args.output_dir / "lift_agent.mp4")

        perception_api: PerceptionApi | None = None
        if args.fake_perception:
            perception_api = PybulletLiftPerceptionApi(env, object_prompt=object_prompt)
        if args.show_grasp_pose:
            if perception_api is None:
                perception_api = PerceptionApi(observation_provider=env.get_obs)
            perception_api = VisualizingLiftPerceptionApi(env, perception_api)
        if perception_api is None:
            perception_api = PerceptionApi(observation_provider=env.get_obs)

        action_api = ActionApi(env)
        api_functions = perception_api.functions()
        api_functions.update(action_api.functions())
        config = CodeAgentConfig(
            task_description=f"Use perception and grasp planning to lift the {object_prompt}.",
            api_functions=api_functions,
            success_fn=env.is_success,
            system_prompt=LIFT_SYSTEM_PROMPT,
            api_description=LIFT_API_DESCRIPTION,
            scene_context_fn=lambda: {
                "objects": make_scene_summary(env),
                "target_object_prompt": object_prompt,
            },
            diagnostics_fn=lambda: {
                "objects": make_scene_summary(env),
                "robot": make_robot_diagnostics(env),
                "last_grasp": action_api.last_grasp,
            },
            revision_guidance=LIFT_REVISION_GUIDANCE,
            fallback_code=_fallback_lift_code(object_prompt),
        )
        agent = CodeAgent(
            env,
            config,
            allow_fallback=not args.no_fallback,
            max_attempts=args.max_attempts,
        )
        result = agent.run()
        if args.record_video:
            video_path = env.stop_video_recording()
        saved_paths = _save_result(args.output_dir, result, video_path=video_path)

        print("LLM used:", result.llm_used)
        print("Generated code:")
        print(result.code)
        print("Saved generated code:", saved_paths["code"])
        if video_path is not None:
            print("Saved video:", video_path)
        print("Runtime stdout:")
        print(result.runtime.stdout.strip())
        if result.runtime.error:
            print("Runtime error:", result.runtime.error)
        print("Task success:", result.task_success)

        if args.hold_s > 0:
            print(f"Holding PyBullet window for {args.hold_s:.1f} seconds...")
            end_time = time.time() + args.hold_s
            while time.time() < end_time:
                env.step_sim(1)
                time.sleep(1 / 240)

        if not result.task_success:
            raise SystemExit(1)
    finally:
        env.close()


class PybulletLiftPerceptionApi:
    """Local perception stand-in for testing the LLM/action path without GPU services."""

    def __init__(self, env, *, object_prompt: str) -> None:
        self.env = env
        self.object_prompt = object_prompt
        self._base_api = None

    def functions(self) -> dict[str, Any]:
        return {
            "get_observation": self.env.get_obs,
            "segment_object": self.segment_object,
            "segment_point": self.segment_point,
            "get_object_pose": self.get_object_pose,
            "mask_to_3d_points": self._api().mask_to_3d_points,
            "get_oriented_bounding_box_from_3d_points": (
                self._api().get_oriented_bounding_box_from_3d_points
            ),
            "plan_grasp": self.plan_grasp,
            "get_best_grasp_pose": self.get_best_grasp_pose,
        }

    def segment_object(self, rgb: np.ndarray, text_prompt: str):
        from agentic_robots.services.sam3.types import Sam3Mask

        obs = self.env.get_obs()
        body_id = self.env.object_ids[self.env.object_name]
        mask = obs["segmentation"] == body_id
        return [Sam3Mask(mask=mask, box=_mask_box(mask), score=1.0, label=text_prompt)]

    def segment_point(self, rgb: np.ndarray, point_xy: tuple[float, float]):
        from agentic_robots.services.sam3.types import PointPromptResult

        mask = np.zeros(rgb.shape[:2], dtype=bool)
        x = int(np.clip(point_xy[0], 0, rgb.shape[1] - 1))
        y = int(np.clip(point_xy[1], 0, rgb.shape[0] - 1))
        mask[y, x] = True
        return [PointPromptResult(mask=mask, score=1.0)]

    def get_object_pose(self, rgb, depth, intrinsics, object_name, *, camera_pose=None, **kwargs):
        position, quaternion_wxyz = self.env.get_object_pose(self.env.object_name)
        mask = self.segment_object(rgb, object_name)[0]
        from agentic_robots.apis.perception.types import ObjectPose, OrientedBoundingBox

        return ObjectPose(
            object_name=object_name,
            position=position,
            orientation_wxyz=quaternion_wxyz,
            mask=mask.mask,
            box=mask.box,
            score=mask.score,
            points=np.asarray([position], dtype=np.float64),
            oriented_bounding_box=OrientedBoundingBox(
                center=position,
                extent=np.array([0.08, 0.08, 0.08], dtype=np.float64),
                rotation=np.eye(3, dtype=np.float64),
            ),
        )

    def plan_grasp(self, depth, intrinsics, segmentation, **kwargs):
        from agentic_robots.services.graspnet.types import GraspPlan

        position, _ = self.env.get_object_pose(self.env.object_name)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, 3] = position
        return GraspPlan(
            grasps=transform[None, :, :],
            scores=np.array([1.0], dtype=np.float64),
            contact_points=position[None, :],
        )

    def get_best_grasp_pose(self, depth, intrinsics, segmentation, *args, **kwargs):
        from agentic_robots.apis.perception.types import GraspPose

        plan = self.plan_grasp(depth, intrinsics, segmentation)
        position, _ = self.env.get_object_pose(self.env.object_name)
        grasp_position = position + np.array([0.0, 0.0, 0.04], dtype=np.float64)
        orientation_wxyz = self.env.robot.get_end_effector_pose().quaternion_wxyz
        transform = np.eye(4, dtype=np.float64)
        transform[:3, 3] = grasp_position
        return GraspPose(
            position=grasp_position,
            orientation_wxyz=orientation_wxyz,
            score=1.0,
            transform=transform,
            plan=plan,
        )

    def _api(self):
        if self._base_api is None:
            from agentic_robots.apis.perception import PerceptionApi

            self._base_api = PerceptionApi(observation_provider=self.env.get_obs)
        return self._base_api


class VisualizingLiftPerceptionApi:
    """Perception API wrapper that refreshes PyBullet GraspNet markers per call."""

    def __init__(self, env, base_api) -> None:
        self.env = env
        self.base_api = base_api

    def functions(self) -> dict[str, Any]:
        functions = self.base_api.functions()
        functions["get_best_grasp_pose"] = self.get_best_grasp_pose
        return functions

    def get_best_grasp_pose(self, depth, intrinsics, segmentation, camera_pose=None, **kwargs):
        grasp = self.base_api.get_best_grasp_pose(
            depth,
            intrinsics,
            segmentation,
            camera_pose,
            **kwargs,
        )
        self.draw_grasp_pose(grasp, camera_pose)
        return grasp

    def draw_grasp_pose(self, grasp, camera_pose=None) -> None:
        if camera_pose is None:
            obs = self.env.get_obs()
            camera_pose = obs.get("camera_pose")

        marker_transform = (
            grasp.graspnet_transform if grasp.graspnet_transform is not None else grasp.transform
        )
        candidate_transforms = _candidate_marker_transforms(grasp, camera_pose)
        self.env.draw_grasp_candidates(
            candidate_transforms,
            scores=grasp.plan.scores,
            best_transform=marker_transform,
            max_grasps=5,
            clear_previous=True,
        )

        object_position = self.env.get_object_position(self.env.object_name)
        distance = float(np.linalg.norm(grasp.position - object_position))
        displayed_candidates = min(5, len(candidate_transforms))
        print("Best grasp pose marker:")
        print("  marker_position:", np.round(marker_transform[:3, 3], 5).tolist())
        print("  executable_position:", np.round(grasp.position, 5).tolist())
        print("  executable_quaternion_wxyz:", np.round(grasp.quaternion_wxyz, 5).tolist())
        print("  score:", round(float(grasp.score), 5))
        print("  distance_to_object_origin_m:", round(distance, 5))
        print("  displayed_candidates:", displayed_candidates)


def _mask_box(mask: np.ndarray) -> tuple[float, float, float, float]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


def _load_object_prompt(name: str) -> str:
    with DEFAULT_OBJECT_CONFIG.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return config["objects"][name].get("prompt", name)


def _fallback_lift_code(object_prompt: str) -> str:
    return f"""
obs = get_observation()
masks = segment_object(obs["rgb"], {object_prompt!r})
grasp = get_best_grasp_pose(obs["depth"], obs["intrinsics"], masks, obs["camera_pose"])
open_gripper()
move_to_pose(grasp.position + [0.0, 0.0, 0.10], grasp.quaternion_wxyz)
move_to_pose(grasp.position, grasp.quaternion_wxyz)
close_gripper()
move_to_pose(grasp.position + [0.0, 0.0, 0.18], grasp.quaternion_wxyz)
print("lift fallback grasp score", grasp.score)
""".strip()


def _draw_best_grasp_pose(env, object_prompt: str, *, perception_api=None) -> None:
    if perception_api is None:
        from agentic_robots.apis.perception import PerceptionApi

        perception_api = PerceptionApi(observation_provider=env.get_obs)

    env.clear_debug_items()
    obs = env.get_obs()
    masks = perception_api.segment_object(obs["rgb"], object_prompt)
    grasp = perception_api.get_best_grasp_pose(
        obs["depth"],
        obs["intrinsics"],
        masks,
        obs["camera_pose"],
    )
    object_position = env.get_object_position(env.object_name)
    distance = float(np.linalg.norm(grasp.position - object_position))
    candidate_transforms = _world_grasp_transforms(grasp.plan.grasps, obs["camera_pose"])
    marker_transform = grasp.graspnet_transform if grasp.graspnet_transform is not None else grasp.transform
    env.draw_grasp_candidates(
        candidate_transforms,
        scores=grasp.plan.scores,
        best_transform=marker_transform,
        max_grasps=5,
    )
    displayed_candidates = min(5, len(candidate_transforms))
    print("Best grasp pose marker:")
    print("  marker_position:", np.round(marker_transform[:3, 3], 5).tolist())
    print("  executable_position:", np.round(grasp.position, 5).tolist())
    print("  executable_quaternion_wxyz:", np.round(grasp.quaternion_wxyz, 5).tolist())
    print("  score:", round(float(grasp.score), 5))
    print("  distance_to_object_origin_m:", round(distance, 5))
    print("  displayed_candidates:", displayed_candidates)


def _candidate_marker_transforms(grasp, camera_pose: np.ndarray | None) -> np.ndarray:
    candidate_transforms = np.asarray(grasp.plan.grasps, dtype=np.float64).reshape(-1, 4, 4)
    if getattr(grasp, "graspnet_transform", None) is not None and camera_pose is not None:
        return _world_grasp_transforms(candidate_transforms, camera_pose)
    return candidate_transforms


def _world_grasp_transforms(grasps: np.ndarray, camera_pose: np.ndarray) -> np.ndarray:
    camera_pose = np.asarray(camera_pose, dtype=np.float64).reshape(4, 4)
    grasp_transforms = np.asarray(grasps, dtype=np.float64).reshape(-1, 4, 4)
    return camera_pose[None, :, :] @ grasp_transforms


def _save_result(output_dir: Path, result, *, video_path: Path | None = None) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    code_path = output_dir / "generated_code.py"
    stdout_path = output_dir / "stdout.txt"
    result_path = output_dir / "result.json"

    code_path.write_text(result.code + "\n", encoding="utf-8")
    stdout_path.write_text(result.runtime.stdout, encoding="utf-8")
    result_path.write_text(
        json.dumps(
            {
                "llm_used": result.llm_used,
                "task_success": result.task_success,
                "runtime_success": result.runtime.success,
                "runtime_error": result.runtime.error,
                "attempt_count": len(result.attempts),
                "attempts": [
                    {
                        "index": index,
                        "task_success": attempt.task_success,
                        "runtime_success": attempt.runtime.success,
                        "runtime_error": attempt.runtime.error,
                        "stdout": attempt.runtime.stdout,
                        "code": attempt.code,
                    }
                    for index, attempt in enumerate(result.attempts, start=1)
                ],
                "generated_code_path": str(code_path),
                "stdout_path": str(stdout_path),
                "video_path": str(video_path) if video_path is not None else None,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"code": code_path, "stdout": stdout_path, "result": result_path}


if __name__ == "__main__":
    main()
