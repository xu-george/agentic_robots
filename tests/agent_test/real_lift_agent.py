from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tests" / "agent_test" / "outputs" / "real_lift_agent"

REAL_LIFT_SYSTEM_PROMPT = """You write short Python robot task programs.
Use only the provided perception and action functions. Do not import modules or define helper functions.
get_best_grasp_pose(...) returns a grasp object, not a dict.
Use grasp.position and grasp.quaternion_wxyz; never use grasp["position"] or grasp["quaternion_wxyz"].
Positions are usually numpy arrays, so vector addition with small offset lists is allowed.
Return only executable Python code, no markdown, no explanation.
"""

REAL_LIFT_API_DESCRIPTION = """Perception:
- get_observation() -> observation dict with rgb, depth, intrinsics, camera_pose
- segment_object(rgb, text_prompt) -> SAM3 masks for the requested object
- get_best_grasp_pose(depth, intrinsics, segmentation, camera_pose) -> grasp in Franka base frame
Action:
- get_end_effector_pose() -> current robot pose with position and quaternion_wxyz
- move_to_pose(position, quaternion_wxyz, duration_s=5.0)
- open_gripper()
- close_gripper()
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the real Franka + RealSense lift task.")
    parser.add_argument("--object-prompt", required=True)
    parser.add_argument("--execute-real", action="store_true")
    parser.add_argument("--confirm-clear-workspace", action="store_true")
    parser.add_argument("--move-home-first", action="store_true")
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--no-grasp-window", action="store_true")
    parser.add_argument("--grasp-window-top-k", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from agentic_robots.agent import CodeAgent, CodeAgentConfig
    from agentic_robots.apis.perception import PerceptionApi
    from agentic_robots.envs.real import RealObjectLiftEnv

    if args.execute_real and not args.confirm_clear_workspace:
        raise SystemExit("Refusing real motion without --confirm-clear-workspace")

    env = RealObjectLiftEnv(
        object_prompt=args.object_prompt,
        motion_enabled=args.execute_real,
        move_home_on_reset=args.move_home_first,
    )
    try:
        obs = env.reset()
        if not args.execute_real:
            print("Read-only real lift env initialized.")
            print("rgb_shape", obs["rgb"].shape)
            print("depth_shape", obs["depth"].shape)
            print("eef_position", obs["robot"]["eef_position"].round(5).tolist())
            print("Add --execute-real --confirm-clear-workspace to run robot motion.")
            return

        base_perception_api = PerceptionApi(observation_provider=env.get_obs)
        perception_api = (
            base_perception_api
            if args.no_grasp_window
            else LiveGraspPerceptionApi(
                base_perception_api,
                output_dir=args.output_dir,
                top_k=args.grasp_window_top_k,
            )
        )
        action_api = ChunkedRealActionApi(env.robot)
        api_functions = perception_api.functions()
        api_functions.update(action_api.functions())

        config = CodeAgentConfig(
            task_description=f"Use RealSense perception and grasp planning to lift {args.object_prompt}.",
            api_functions=api_functions,
            success_fn=env.is_success,
            system_prompt=REAL_LIFT_SYSTEM_PROMPT,
            api_description=REAL_LIFT_API_DESCRIPTION,
            scene_context_fn=lambda: {
                "target_object_prompt": args.object_prompt,
                "camera_pose_frame": "camera optical frame to Franka base",
            },
            diagnostics_fn=lambda: _robot_diagnostics(env),
            revision_guidance=(
                "Use the same grasp result. Open gripper, move above grasp, move down, "
                "close gripper, then lift upward. Use the current end-effector quaternion."
            ),
            fallback_code=_fallback_real_lift_code(args.object_prompt),
        )
        result = CodeAgent(
            env,
            config,
            llm=None if args.use_llm else FallbackOnlyLlm(),
            max_attempts=args.max_attempts,
        ).run()
        saved_paths = _save_result(args.output_dir, result)

        print("LLM used:", result.llm_used)
        print("Generated code:")
        print(result.code)
        print("Saved generated code:", saved_paths["code"])
        print("Runtime stdout:")
        print(result.runtime.stdout.strip())
        if result.runtime.error:
            print("Runtime error:", result.runtime.error)
        print("Task success:", result.task_success)
        if not result.task_success:
            raise SystemExit(1)
    finally:
        env.close()
        if not args.no_grasp_window:
            cv2.destroyAllWindows()


def _fallback_real_lift_code(object_prompt: str) -> str:
    return f"""
obs = get_observation()
masks = segment_object(obs["rgb"], {object_prompt!r})
grasp = get_best_grasp_pose(obs["depth"], obs["intrinsics"], masks, obs["camera_pose"])
pose = get_end_effector_pose()
open_gripper()
move_to_pose(grasp.position + [0.0, 0.0, 0.04], pose.quaternion_wxyz)
move_to_pose(grasp.position, pose.quaternion_wxyz)
close_gripper()
move_to_pose(grasp.position + [0.0, 0.0, 0.04], pose.quaternion_wxyz)
print("real lift fallback grasp score", grasp.score)
""".strip()


class ChunkedRealActionApi:
    """Action API that keeps each real Cartesian command inside bridge limits."""

    def __init__(self, robot: Any) -> None:
        self.robot = robot

    def functions(self) -> dict[str, Any]:
        return {
            "get_end_effector_pose": self.get_end_effector_pose,
            "move_to_pose": self.move_to_pose,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
        }

    def get_end_effector_pose(self):
        self._require_robot()
        return self.robot.get_end_effector_pose()

    def move_to_pose(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
        *,
        max_step_m: float = 0.04,
        duration_s: float = 5.0,
        **_: Any,
    ) -> np.ndarray:
        self._require_robot()
        start = self.robot.get_end_effector_pose().position
        target = self._clip_to_workspace(np.asarray(position, dtype=np.float64).reshape(3))
        steps = max(1, int(np.ceil(np.max(np.abs(target - start)) / max_step_m)))
        for waypoint in np.linspace(start, target, steps + 1, dtype=np.float64)[1:]:
            self.robot.move_to_pose(waypoint, quaternion_wxyz, duration_s=duration_s)
        return target[None, :]

    def open_gripper(self, **kwargs) -> None:
        self._require_robot()
        self.robot.open_gripper(**kwargs)

    def close_gripper(self, **kwargs) -> None:
        self._require_robot()
        self.robot.close_gripper(**kwargs)

    def _require_robot(self) -> None:
        if self.robot is None:
            raise RuntimeError("A loaded robot must be provided before using ChunkedRealActionApi")

    def _clip_to_workspace(self, target: np.ndarray, *, margin_m: float = 0.005) -> np.ndarray:
        workspace = self.robot.workspace
        lower = np.array(
            [workspace.x[0] + margin_m, workspace.y[0] + margin_m, workspace.z[0] + margin_m],
            dtype=np.float64,
        )
        upper = np.array(
            [workspace.x[1] - margin_m, workspace.y[1] - margin_m, workspace.z[1] - margin_m],
            dtype=np.float64,
        )
        clipped = np.clip(target, lower, upper)
        if not np.allclose(clipped, target):
            print(
                "adjusted_move_target_to_workspace",
                "requested=",
                np.round(target, 5).tolist(),
                "used=",
                np.round(clipped, 5).tolist(),
            )
        return clipped


class LiveGraspPerceptionApi:
    """Test-local wrapper that visualizes grasp plans without changing PerceptionApi."""

    def __init__(self, base_api: Any, *, output_dir: Path, top_k: int) -> None:
        self.base_api = base_api
        self.output_dir = output_dir
        self.top_k = top_k
        self._last_rgb: np.ndarray | None = None
        self._last_mask: np.ndarray | None = None

    def functions(self) -> dict[str, Any]:
        funcs = self.base_api.functions()
        funcs["segment_object"] = self.segment_object
        funcs["get_best_grasp_pose"] = self.get_best_grasp_pose
        return funcs

    def get_observation(self):
        return self.base_api.get_observation()

    def segment_object(self, rgb: np.ndarray, text_prompt: str, *args, **kwargs):
        masks = self.base_api.segment_object(rgb, text_prompt, *args, **kwargs)
        self._last_rgb = np.asarray(rgb, dtype=np.uint8).copy()
        self._last_mask = masks[0].mask if masks else None
        return masks

    def get_best_grasp_pose(self, depth, intrinsics, segmentation, camera_pose, *args, **kwargs):
        grasp = self.base_api.get_best_grasp_pose(
            depth,
            intrinsics,
            segmentation,
            camera_pose,
            *args,
            **kwargs,
        )
        if self._last_rgb is not None:
            overlay = draw_grasp_overlay(
                rgb=self._last_rgb,
                mask=self._last_mask,
                grasps=grasp.plan.grasps,
                scores=grasp.plan.scores,
                intrinsics=intrinsics,
                top_k=self.top_k,
            )
            self.output_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.output_dir / "live_grasp_overlay.png"
            cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            cv2.imshow("Real lift grasp hypotheses", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
            print(f"live_grasp_overlay={output_path}")
        return grasp


def draw_grasp_overlay(
    *,
    rgb: np.ndarray,
    mask: np.ndarray | None,
    grasps: np.ndarray,
    scores: np.ndarray,
    intrinsics: np.ndarray,
    top_k: int,
) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.uint8).copy()
    if mask is not None:
        image = _blend_mask(image, np.asarray(mask, dtype=bool))

    grasp_array = np.asarray(grasps, dtype=np.float64).reshape(-1, 4, 4)
    score_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    if grasp_array.size == 0:
        return image

    if score_array.size == len(grasp_array):
        order = np.argsort(score_array)[::-1]
    else:
        order = np.arange(len(grasp_array))
    selected = order[: max(1, min(top_k, len(order)))]
    for rank, grasp_index in enumerate(selected[::-1]):
        color = (255, 40, 40) if grasp_index == selected[0] else _rank_color(rank, len(selected))
        width = 4 if grasp_index == selected[0] else 2
        _draw_grasp(image, grasp_array[int(grasp_index)], intrinsics, color=color, width=width)
    return image


def _blend_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = rgb.copy()
    overlay[mask] = (0.55 * overlay[mask] + 0.45 * np.array([40, 190, 255])).astype(np.uint8)
    return overlay


def _draw_grasp(
    image: np.ndarray,
    grasp: np.ndarray,
    intrinsics: np.ndarray,
    *,
    color: tuple[int, int, int],
    width: int,
) -> None:
    from agentic_robots.envs.pybullet.lift_env import _panda_gripper_wire_points

    points = _panda_gripper_wire_points(grasp, opening=0.08)
    uv, valid = _project(points, intrinsics, image.shape[:2])
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


def _project(
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


def _rank_color(rank: int, count: int) -> tuple[int, int, int]:
    value = rank / max(count - 1, 1)
    return (int(40 + 180 * value), int(230 * (1.0 - value)), 255)


class FallbackOnlyLlm:
    def complete(self, *args, **kwargs):
        raise RuntimeError("LLM disabled for deterministic real robot run")


def _robot_diagnostics(env) -> dict[str, Any]:
    obs = env.get_obs()
    return {
        "eef_position": obs["robot"]["eef_position"].round(4).tolist(),
        "gripper_width": round(float(obs["robot"]["gripper_width"][0]), 5),
    }


def _save_result(output_dir: Path, result) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    code_path = output_dir / "generated_code.py"
    result_path = output_dir / "result.json"
    code_path.write_text(result.code + "\n", encoding="utf-8")
    result_path.write_text(
        json.dumps(
            {
                "llm_used": result.llm_used,
                "task_success": result.task_success,
                "runtime_success": result.runtime.success,
                "runtime_error": result.runtime.error,
                "stdout": result.runtime.stdout,
                "generated_code_path": str(code_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"code": code_path, "result": result_path}


if __name__ == "__main__":
    main()
