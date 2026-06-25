from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "tests" / "agent_test" / "outputs"


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerun saved LLM-generated robot code.")
    parser.add_argument("--task", choices=("stacking", "lift"), required=True)
    parser.add_argument("--code", type=Path, default=None)
    parser.add_argument("--object", default="banana", help="Object name for --task lift.")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--hold-s", type=float, default=0.0)
    parser.add_argument("--fake-perception", action="store_true")
    parser.add_argument("--show-grasp-pose", action="store_true")
    args = parser.parse_args()

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    code_path = args.code or _default_code_path(args.task)
    code = code_path.read_text(encoding="utf-8")

    if args.task == "stacking":
        result = _run_stacking(code, args)
    else:
        result = _run_lift(code, args)

    print("Reran code:", code_path)
    print("Runtime success:", result["runtime_success"])
    print("Task success:", result["task_success"])
    print("Runtime stdout:")
    print(result["stdout"].strip())
    if result["error"]:
        print("Runtime error:", result["error"])
    if result["video_path"] is not None:
        print("Saved video:", result["video_path"])

    if not result["task_success"]:
        raise SystemExit(1)


def _default_code_path(task: str) -> Path:
    return DEFAULT_OUTPUT_ROOT / f"{task}_agent" / "generated_code.py"


def _run_stacking(code: str, args) -> dict[str, object]:
    from agentic_robots.agent.runtime import AgentRuntime
    from agentic_robots.apis.action import ActionApi
    from agentic_robots.envs.pybullet import StackingEnv

    env = StackingEnv(gui=args.gui)
    video_path = None
    try:
        env.reset()
        if args.record_video:
            video_path = env.start_video_recording(
                DEFAULT_OUTPUT_ROOT / "stacking_agent" / "rerun_generated_code.mp4"
            )

        action_api = ActionApi(env.robot)
        api_functions = {
            "get_observation": env.get_obs,
            "get_object_position": env.get_object_position,
            "is_success": env.is_success,
        }
        api_functions.update(action_api.functions())
        runtime = AgentRuntime(api_functions).run(code)

        if args.record_video:
            video_path = env.stop_video_recording()
        _hold_window(env, args.hold_s)
        return _result_dict(runtime, env.is_success(), video_path)
    finally:
        env.close()


def _run_lift(code: str, args) -> dict[str, object]:
    from agentic_robots.agent.runtime import AgentRuntime
    from agentic_robots.apis.action import ActionApi
    from agentic_robots.apis.perception import PerceptionApi
    from agentic_robots.envs.pybullet import LiftEnv
    from tests.agent_test.lift_agent import (
        PybulletLiftPerceptionApi,
        VisualizingLiftPerceptionApi,
        _load_object_prompt,
    )

    object_prompt = _load_object_prompt(args.object)
    env = LiftEnv(gui=args.gui, object_name=args.object)
    video_path = None
    try:
        env.reset()
        if args.record_video:
            video_path = env.start_video_recording(
                DEFAULT_OUTPUT_ROOT / "lift_agent" / "rerun_generated_code.mp4"
            )

        perception_api = None
        if args.fake_perception:
            perception_api = PybulletLiftPerceptionApi(env, object_prompt=object_prompt)
        if args.show_grasp_pose:
            if perception_api is None:
                perception_api = PerceptionApi(observation_provider=env.get_obs)
            perception_api = VisualizingLiftPerceptionApi(env, perception_api)
        if perception_api is None:
            perception_api = PerceptionApi(observation_provider=env.get_obs)

        action_api = ActionApi(env.robot)
        api_functions = perception_api.functions()
        api_functions.update(action_api.functions())
        runtime = AgentRuntime(api_functions).run(code)

        if args.record_video:
            video_path = env.stop_video_recording()
        _hold_window(env, args.hold_s)
        return _result_dict(runtime, env.is_success(), video_path)
    finally:
        env.close()


def _hold_window(env, hold_s: float) -> None:
    if hold_s <= 0:
        return
    end_time = time.time() + hold_s
    while time.time() < end_time:
        env.step_sim(1)
        time.sleep(1 / 240)


def _result_dict(runtime, task_success: bool, video_path: Path | None) -> dict[str, object]:
    return {
        "runtime_success": runtime.success,
        "task_success": bool(runtime.success and task_success),
        "stdout": runtime.stdout,
        "error": runtime.error,
        "video_path": video_path,
    }


if __name__ == "__main__":
    main()
