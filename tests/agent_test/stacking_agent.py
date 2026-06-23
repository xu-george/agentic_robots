from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tests" / "agent_test" / "outputs" / "stacking_agent"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the stacking code agent in PyBullet.")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--hold-s", type=float, default=0.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from agentic_robots.agent import StackingCodeAgent
    from agentic_robots.envs.pybullet import StackingEnv

    env = StackingEnv(gui=args.gui)
    try:
        env.reset()
        video_path = None
        if args.record_video:
            video_path = env.start_video_recording(args.output_dir / "stacking_agent.mp4")

        agent = StackingCodeAgent(
            env,
            allow_fallback=not args.no_fallback,
            max_attempts=args.max_attempts,
        )
        result = agent.run("Stack the red cube on top of the blue cube.")
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
