from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tests" / "agent_test" / "outputs" / "lift_agent"
DEFAULT_OBJECT_CONFIG = PROJECT_ROOT / "configs" / "lift_objects.yaml"

LIFT_SYSTEM_PROMPT = """You write short Python robot task programs.
Use only the provided perception and action functions. NumPy is pre-imported
as `np` (and `numpy`); you do NOT need to write `import numpy` yourself, and
writing any other `import` statement will fail the run. Do not define helper functions.
get_finger_grasp(...) returns a GraspPose with .position and .quaternion_wxyz (alias for .orientation_wxyz).
Positions are usually numpy arrays, so vector addition with small offset lists is allowed.
Return only executable Python code, no markdown, no explanation.
"""

LIFT_API_FUNCTION_NAMES = [
    "get_observation",
    "segment_object",
    "get_finger_grasp",
    "get_end_effector_pose",
    "move_to_pose",
    "open_gripper",
    "close_gripper",
]

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
    parser.add_argument("--use-real-services", action="store_true")
    parser.add_argument(
        "--show-grasp",
        action="store_true",
        help="Run perception once and draw grasp candidates in the sim before the agent starts.",
    )
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--hold-s", type=float, default=10)
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

        if args.use_real_services:
            perception = PerceptionApi(observation_provider=env.get_obs)
        else:
            perception = PerceptionApi(
                observation_provider=env.get_obs,
                sam3_client=_FakeSam3Client(env=env),
                graspnet_client=_FakeGraspNetClient(object_name=args.object, env=env),
            )
        action = ActionApi(env.robot)

        if args.show_grasp:
            from agentic_robots.utils import show_grasp, clear_debug

            obs = env.get_obs()
            masks = perception.segment_object(obs["rgb"], object_prompt)
            grasp = perception.get_finger_grasp(obs["depth"], obs["intrinsics"], masks, obs["camera_pose"])
            show_grasp(
                env.client_id,
                grasp,
                draw_axes=True,
                draw_candidates=True,
                max_candidates=5,
                opening=0.08,
                life_time=0.0,
            )
            print(
                f"Grasp at {np.round(grasp.position, 4).tolist()}, "
                f"score={grasp.score:.3f}, {len(getattr(grasp.plan, 'grasps', []))} candidates"
            )
            for _ in range(100):
                env.step_sim(1)
                time.sleep(0.01)

        api_functions = {}
        api_functions.update(perception.functions())
        api_functions.update(action.functions())

        config = CodeAgentConfig(
            task_description=f"Use perception and grasp planning to lift the {object_prompt}.",
            api_functions=api_functions,
            success_fn=env.is_success,
            system_prompt=LIFT_SYSTEM_PROMPT,
            api_schema_path=PROJECT_ROOT / "configs" / "api_schema.json",
            api_function_names=LIFT_API_FUNCTION_NAMES,
            scene_context_fn=lambda: {
                "objects": make_scene_summary(env),
                "target_object_prompt": object_prompt,
            },
            diagnostics_fn=lambda: {
                "objects": make_scene_summary(env),
                "robot": make_robot_diagnostics(env),
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


class _FakeSam3Client:
    """Uses PyBullet ground-truth segmentation instead of real SAM3."""

    def __init__(self, *, env) -> None:
        self.env = env

    def segment(self, image, prompt):
        from agentic_robots.services.sam3.types import Sam3Mask

        obs = self.env.get_obs()
        body_id = self.env.object_ids[self.env.object_name]
        mask = obs["segmentation"] == body_id
        return [Sam3Mask(mask=mask, box=_mask_box(mask), score=1.0, label=prompt)]

    def segment_point(self, image, point_xy):
        from agentic_robots.services.sam3.types import PointPromptResult

        mask = np.zeros(image.shape[:2], dtype=bool)
        x = int(np.clip(point_xy[0], 0, image.shape[1] - 1))
        y = int(np.clip(point_xy[1], 0, image.shape[0] - 1))
        mask[y, x] = True
        return [PointPromptResult(mask=mask, score=1.0)]


class _FakeGraspNetClient:
    """Returns a camera-frame grasp at the object's world position.

    PerceptionApi.get_finger_grasp does:
        world_frame = camera_pose @ camera_frame_grasp
        final = world_frame @ PANDA_GRASPTARGET_FROM_GRASPNET

    We compute the camera-frame transform so that final[:3, 3] lands
    at the object's world position.
    """

    def __init__(self, *, object_name: str, env) -> None:
        self.object_name = object_name
        self.env = env

    def plan_from_depth(self, depth, camera_intrinsics, segmap, **kwargs):
        from agentic_robots.services.graspnet.types import GraspPlan

        obs = self.env.get_obs()
        camera_pose = np.asarray(obs["camera_pose"], dtype=np.float64).reshape(4, 4)
        world_pos, _ = self.env.get_object_pose(self.object_name)

        from agentic_robots.apis.perception.api import PANDA_GRASPTARGET_OFFSET_M

        offset = PANDA_GRASPTARGET_OFFSET_M  # [0, 0, 0.105]
        R_cam = camera_pose[:3, :3]
        t_cam = camera_pose[:3, 3]
        cam_frame_pos = R_cam.T @ (world_pos - t_cam - R_cam @ offset)

        transform = np.eye(4, dtype=np.float64)
        transform[:3, 3] = cam_frame_pos
        contact_cam = cam_frame_pos.copy()
        return GraspPlan(
            grasps=transform[None, :, :],
            scores=np.array([1.0], dtype=np.float64),
            contact_points=contact_cam[None, :],
        )


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
grasp = get_finger_grasp(obs["depth"], obs["intrinsics"], masks, obs["camera_pose"])
open_gripper()
move_to_pose(grasp.position + [0.0, 0.0, 0.10], grasp.quaternion_wxyz)
move_to_pose(grasp.position, grasp.quaternion_wxyz)
close_gripper()
move_to_pose(grasp.position + [0.0, 0.0, 0.18], grasp.quaternion_wxyz)
print("lift fallback grasp score", grasp.score)
""".strip()


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
