from __future__ import annotations

import argparse
import json
import sys
from threading import RLock
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tests" / "agent_test" / "outputs" / "real_lift_agent"

REAL_LIFT_SYSTEM_PROMPT = """You write short Python robot task programs for a real Franka robot.
Use only the provided perception and action functions. NumPy is pre-imported
as `np` (and `numpy`); you do NOT need to write `import numpy` yourself, and
writing any other import statement will fail the run. Do not define helper functions.
get_finger_grasp(...) returns a GraspPose with .position and .quaternion_wxyz
(alias for .orientation_wxyz). Positions are usually numpy arrays, so vector
addition with small offset lists is allowed. For safety, move above the grasp
first, descend to the grasp, close the gripper, then lift upward. Return only
executable Python code, no markdown, no explanation.
get_observation() returns a dict; use obs["rgb"], obs["depth"],
obs["intrinsics"], and obs["camera_pose"], not dot access.
"""

REAL_LIFT_API_FUNCTION_NAMES = [
    "get_observation",
    "segment_object",
    "get_finger_grasp",
    "get_end_effector_pose",
    "move_to_pose",
    "open_gripper",
    "close_gripper",
]

REAL_LIFT_REVISION_GUIDANCE = (
    "Revise the real lift code. Use the same grasp result. "
    "Do this sequence: open gripper, move above grasp, move to grasp, "
    "close gripper, then move upward to lift. "
    "Use dict access for observations. "
    "Do not import modules or stop after closing the gripper."
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the real Franka + RealSense lift task.")
    parser.add_argument("--object-prompt", required=True)
    parser.add_argument("--execute-real", action="store_true")
    parser.add_argument("--confirm-clear-workspace", action="store_true")
    parser.add_argument("--move-home-first", action="store_true")
    parser.add_argument("--gui", action="store_true", help="Start a read-only RealSense web viewer.")
    parser.add_argument("--gui-host", default="127.0.0.1")
    parser.add_argument("--gui-port", type=int, default=8120)
    parser.add_argument("--gui-refresh-s", type=float, default=1.5)
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from agentic_robots.agent import CodeAgent, CodeAgentConfig
    from agentic_robots.apis.action import ActionApi
    from agentic_robots.apis.perception import PerceptionApi
    from agentic_robots.envs.real import RealObjectLiftEnv
    from agentic_robots.web.realsense_viewer import start_realsense_viewer

    if args.execute_real and not args.confirm_clear_workspace:
        raise SystemExit("Refusing real motion without --confirm-clear-workspace")

    env = RealObjectLiftEnv(
        object_prompt=args.object_prompt,
        motion_enabled=args.execute_real,
        move_home_on_reset=args.move_home_first,
    )
    viewer = None
    camera_lock = RLock()

    def get_obs_locked() -> dict[str, Any]:
        with camera_lock:
            return env.get_obs()

    def get_camera_obs_locked() -> dict[str, Any]:
        with camera_lock:
            frame = env.camera.capture()
        return {
            "rgb": frame.rgb,
            "depth": frame.depth_m,
            "intrinsics": env.intrinsics,
            "camera_pose": env.camera_pose,
        }

    try:
        env.reset()
        if args.gui:
            viewer = start_realsense_viewer(
                observation_provider=get_camera_obs_locked,
                object_prompt=args.object_prompt,
                host=args.gui_host,
                port=args.gui_port,
                refresh_s=args.gui_refresh_s,
            )
            print(f"RealSense viewer: {viewer.url}")

        if not args.execute_real:
            print("Read-only real lift env initialized.")
            obs = get_obs_locked()
            print("rgb_shape", obs["rgb"].shape)
            print("depth_shape", obs["depth"].shape)
            print("eef_position", obs["robot"]["eef_position"].round(5).tolist())
            print("Add --execute-real --confirm-clear-workspace to run robot motion.")
            return

        perception = PerceptionApi(observation_provider=get_obs_locked)
        action = ActionApi(env.robot)

        api_functions = {}
        api_functions.update(perception.functions())
        api_functions.update(action.functions())

        config = CodeAgentConfig(
            task_description=f"Use RealSense perception and grasp planning to lift {args.object_prompt}.",
            api_functions=api_functions,
            success_fn=env.is_success,
            system_prompt=REAL_LIFT_SYSTEM_PROMPT,
            api_schema_path=PROJECT_ROOT / "configs" / "api_schema.json",
            api_function_names=REAL_LIFT_API_FUNCTION_NAMES,
            scene_context_fn=lambda: {
                "target_object_prompt": args.object_prompt,
                "camera_pose_frame": "RealSense camera optical frame to Franka base",
            },
            diagnostics_fn=lambda: _robot_diagnostics(get_obs_locked),
            revision_guidance=REAL_LIFT_REVISION_GUIDANCE,
            fallback_code=_fallback_real_lift_code(args.object_prompt),
        )
        result = CodeAgent(
            env,
            config,
            allow_fallback=not args.no_fallback,
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
        if viewer is not None:
            viewer.stop()
        env.close()


def _fallback_real_lift_code(object_prompt: str) -> str:
    return f"""
obs = get_observation()
masks = segment_object(obs["rgb"], {object_prompt!r})
grasp = get_finger_grasp(obs["depth"], obs["intrinsics"], masks, obs["camera_pose"])
open_gripper()
move_to_pose(grasp.position + [0.0, 0.0, 0.10], grasp.quaternion_wxyz)
move_to_pose(grasp.position, grasp.quaternion_wxyz)
close_gripper()
move_to_pose(grasp.position + [0.0, 0.0, 0.18], grasp.quaternion_wxyz)
print("real lift fallback grasp score", grasp.score)
""".strip()


def _robot_diagnostics(observation_provider) -> dict[str, Any]:
    obs = observation_provider()
    gripper = obs["robot"]["gripper"]
    return {
        "eef_position": np.round(obs["robot"]["eef_position"], 5).tolist(),
        "eef_quaternion_wxyz": np.round(obs["robot"]["eef_quaternion_wxyz"], 5).tolist(),
        "gripper_width": round(float(obs["robot"]["gripper_width"][0]), 5),
        "gripper_is_grasped": bool(gripper.is_grasped),
    }


def _save_result(output_dir: Path, result) -> dict[str, Path]:
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
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"code": code_path, "stdout": stdout_path, "result": result_path}


if __name__ == "__main__":
    main()
