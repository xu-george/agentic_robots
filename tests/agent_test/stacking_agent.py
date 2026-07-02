from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tests" / "agent_test" / "outputs" / "stacking_agent"

STACKING_SYSTEM_PROMPT = """You write short Python robot task programs.
Use only the provided functions. NumPy is pre-imported as `np`; do not write
any `import` statement. get_end_effector_pose() returns a FrankaPose object;
use pose.position and pose.quaternion_wxyz, never dict access. Positions are
numpy arrays, so vector addition with small offset lists is allowed. Approach
every target from above first, then descend; never go straight to a table-level
pose or the arm will sweep through the cubes. Return ONLY executable Python
code, no markdown, no explanation, NO comments at all.
"""

STACKING_API_FUNCTION_NAMES = [
    "get_observation",
    "get_object_position",
    "is_success",
    "get_end_effector_pose",
    "move_to_pose",
    "open_gripper",
    "close_gripper",
]

STACKING_EXTRA_API_FUNCTIONS = {
    "get_object_position": {
        "description": "Get the world position of a named object in the scene.",
        "params": [
            {
                "name": "name",
                "type": "str",
                "description": 'Object name, e.g. "red_cube" or "blue_cube".',
                "required": True,
            }
        ],
        "returns": {
            "type": "np.ndarray",
            "shape": "(3,)",
            "dtype": "float64",
            "description": "Object center position [x, y, z]. Index [2] is the cube center z.",
        },
    },
    "is_success": {
        "description": "Check if the stacking task is complete (red cube centered above blue cube).",
        "params": [],
        "returns": {
            "type": "bool",
            "description": "True if red cube center is ~0.05m above blue cube center and aligned in xy.",
        },
    },
}

STACKING_REVISION_GUIDANCE = (
    "Revise the stacking code. Approach the red cube from above, close the gripper, "
    "lift, move over the blue cube, lower to stack height, then open the gripper. "
    "Use get_end_effector_pose().quaternion_wxyz with dot access, not dict access."
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the stacking task with the general code agent.")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--hold-s", type=float, default=0.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from agentic_robots.agent import (
        CodeAgent,
        CodeAgentConfig,
        make_scene_summary,
    )
    from agentic_robots.apis.action import ActionApi
    from agentic_robots.envs.pybullet import StackingEnv

    env = StackingEnv(gui=args.gui)
    try:
        env.reset()
        video_path = None
        if args.record_video:
            video_path = env.start_video_recording(args.output_dir / "stacking_agent.mp4")

        action_api = ActionApi(env.robot)
        api_functions = {
            "get_observation": env.get_obs,
            "get_object_position": env.get_object_position,
            "is_success": env.is_success,
        }
        api_functions.update(action_api.functions())
        config = CodeAgentConfig(
            task_description="Stack the red cube on top of the blue cube.",
            api_functions=api_functions,
            success_fn=env.is_success,
            system_prompt=STACKING_SYSTEM_PROMPT,
            api_schema_path=PROJECT_ROOT / "configs" / "api_schema.json",
            api_function_names=STACKING_API_FUNCTION_NAMES,
            extra_api_functions=STACKING_EXTRA_API_FUNCTIONS,
            scene_context_fn=lambda: {
                "objects": make_scene_summary(env),
                "goal": "red_cube centered above blue_cube",
            },
            diagnostics_fn=lambda: _stacking_diagnostics(env),
            revision_guidance=STACKING_REVISION_GUIDANCE,
            fallback_code=_fallback_stacking_code(),
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


def _stacking_diagnostics(env) -> dict[str, object]:
    from agentic_robots.agent import make_robot_diagnostics

    red = env.get_object_position("red_cube")
    blue = env.get_object_position("blue_cube")
    return {
        "objects": {
            "red_cube": red.round(4).tolist(),
            "blue_cube": blue.round(4).tolist(),
        },
        "robot": make_robot_diagnostics(env),
        "xy_error_m": round(float(np.linalg.norm(red[:2] - blue[:2])), 4),
        "z_delta_m": round(float(red[2] - blue[2]), 4),
        "task_success": env.is_success(),
    }


def _fallback_stacking_code() -> str:
    return """
red = get_object_position("red_cube")
blue = get_object_position("blue_cube")
pose = get_end_effector_pose()
open_gripper()
move_to_pose(red + [0.0, 0.0, 0.14], pose.quaternion_wxyz)
move_to_pose(red + [0.0, 0.0, 0.02], pose.quaternion_wxyz, plan_steps=50, sim_steps_per_waypoint=6)
close_gripper(steps=120)
move_to_pose(red + [0.0, 0.0, 0.20], pose.quaternion_wxyz, plan_steps=60, sim_steps_per_waypoint=6)
move_to_pose(blue + [0.0, 0.0, 0.20], pose.quaternion_wxyz, plan_steps=60, sim_steps_per_waypoint=6)
move_to_pose(blue + [0.0, 0.0, 0.095], pose.quaternion_wxyz, plan_steps=50, sim_steps_per_waypoint=6)
open_gripper(steps=120)
move_to_pose(blue + [0.0, 0.0, 0.20], pose.quaternion_wxyz, plan_steps=50, sim_steps_per_waypoint=6)
print("stacking fallback success", is_success())
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
