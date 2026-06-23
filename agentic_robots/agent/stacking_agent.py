from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from agentic_robots.agent.runtime import AgentRuntime, RuntimeResult
from agentic_robots.apis.action import ActionApi
from agentic_robots.envs.pybullet import StackingEnv
from agentic_robots.services.foundation_model.qwen.client import QwenClient


SYSTEM_PROMPT = """You write short Python robot task programs.
Use only the provided functions. Do not import modules.
Available functions:
- get_observation()
- get_object_position(name)
- get_end_effector_pose() -> object with position and quaternion_wxyz fields
- move_to_pose(position, quaternion_wxyz)
- open_gripper()
- close_gripper()
- is_success()
- print(...)

Positions are numpy arrays, so you may add offsets like [0.0, 0.0, 0.1].
Use cautious vertical clearance above objects before moving horizontally.
move_to_pose expects quaternion_wxyz; use get_end_effector_pose().quaternion_wxyz as the default
orientation unless the task clearly requires a different one.
Return only executable Python code, no markdown, no explanation.
"""


@dataclass(frozen=True)
class AgentAttempt:
    code: str
    runtime: RuntimeResult
    task_success: bool


@dataclass(frozen=True)
class AgentRunResult:
    code: str
    runtime: RuntimeResult
    task_success: bool
    llm_used: bool
    attempts: tuple[AgentAttempt, ...] = ()


class StackingCodeAgent:
    """Ask an LLM for high-level stacking code and execute it in PyBullet."""

    def __init__(
        self,
        env: StackingEnv,
        *,
        llm: QwenClient | None = None,
        allow_fallback: bool = True,
        max_attempts: int = 2,
    ) -> None:
        self.env = env
        self.llm = llm or QwenClient()
        self.allow_fallback = allow_fallback
        self.max_attempts = max(1, max_attempts)
        self.action_api = ActionApi(env)

    def run(self, instruction: str) -> AgentRunResult:
        attempts: list[AgentAttempt] = []
        code, llm_used = self._generate_code(instruction)
        runtime = AgentRuntime(self._functions()).run(code)
        task_success = bool(runtime.success and self.env.is_success())
        attempts.append(AgentAttempt(code=code, runtime=runtime, task_success=task_success))

        while llm_used and not task_success and len(attempts) < self.max_attempts:
            code = self._revise_code(instruction, attempts[-1])
            runtime = AgentRuntime(self._functions()).run(code)
            task_success = bool(runtime.success and self.env.is_success())
            attempts.append(AgentAttempt(code=code, runtime=runtime, task_success=task_success))

        final_attempt = attempts[-1]
        return AgentRunResult(
            code=final_attempt.code,
            runtime=final_attempt.runtime,
            task_success=final_attempt.task_success,
            llm_used=llm_used,
            attempts=tuple(attempts),
        )

    def _generate_code(self, instruction: str) -> tuple[str, bool]:
        prompt = self._build_prompt(instruction)
        try:
            result = self.llm.complete(
                prompt,
                system=SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=1024,
            )
            return _extract_code(result.content), True
        except Exception:
            if not self.allow_fallback:
                raise
            return _fallback_stacking_code(), False

    def _revise_code(self, instruction: str, previous: AgentAttempt) -> str:
        prompt = (
            f"{self._build_prompt(instruction)}\n\n"
            "The previous code did not complete the task.\n"
            f"Previous code:\n{previous.code}\n\n"
            f"Runtime success: {previous.runtime.success}\n"
            f"Task success: {previous.task_success}\n"
            f"Runtime stdout:\n{previous.runtime.stdout}\n"
            f"Runtime error: {previous.runtime.error}\n"
            f"Failure diagnostics: {self._failure_diagnostics()}\n"
            "Revise the code based on the diagnostics. If the grasp did not attach an object, "
            "move closer to the cube before close_gripper(), especially in z. Do not close the "
            "gripper only from a high clearance pose. Return only executable Python code."
        )
        result = self.llm.complete(
            prompt,
            system=SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=1024,
        )
        return _extract_code(result.content)

    def _build_prompt(self, instruction: str) -> str:
        return (
            f"Task: {instruction}\n"
            f"Objects: {self._scene_summary()}\n"
            "Scene notes: cubes are about 0.05 m wide and sit on a table. "
            "Use the gripper to move the red cube onto the blue cube. "
            "Check is_success() at the end.\n"
            "Write the high-level Python code to complete the task."
        )

    def _scene_summary(self) -> dict[str, dict[str, Any]]:
        obs = self.env.get_obs()
        objects = obs["ground_truth"]["objects"]
        return {
            name: {
                "position": value["position"].round(4).tolist(),
                "quaternion_wxyz": value["quaternion_wxyz"].round(4).tolist(),
            }
            for name, value in objects.items()
        }

    def _failure_diagnostics(self) -> dict[str, Any]:
        obs = self.env.get_obs()
        red = obs["ground_truth"]["objects"]["red_cube"]["position"]
        blue = obs["ground_truth"]["objects"]["blue_cube"]["position"]
        eef = obs["robot"]["eef_position"]
        xy_error = float(((red[:2] - blue[:2]) ** 2).sum() ** 0.5)
        z_error = float(red[2] - blue[2])
        return {
            "current_objects": self._scene_summary(),
            "robot": {
                "eef_position": eef.round(4).tolist(),
                "gripper_positions": obs["robot"]["gripper_positions"].round(4).tolist(),
                "held_object": self.env.held_object,
            },
            "last_grasp": self.action_api.last_grasp,
            "success_check": {
                "xy_distance_red_to_blue_m": round(xy_error, 4),
                "red_minus_blue_z_m": round(z_error, 4),
                "requires_xy_distance_below_m": 0.04,
                "requires_red_above_blue_by_more_than_m": 0.035,
            },
            "grasp_hint": (
                "A 0.05 m cube center is around z=0.075 on the table. "
                "If last_grasp.attached is false or nearest_distance_m is large, "
                "approach from above, then lower the gripper near the cube before closing. "
                "After close_gripper(), lift first, then move horizontally."
            ),
        }

    def _functions(self):
        functions = {
            "get_observation": self.env.get_obs,
            "get_object_position": self.env.get_object_position,
            "is_success": self.env.is_success,
        }
        functions.update(self.action_api.functions())
        return functions


def _extract_code(content: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", content, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped


def _fallback_stacking_code() -> str:
    return "\n".join(
        [
            "print('Stack red_cube on blue_cube')",
            "red_position = get_object_position('red_cube')",
            "blue_position = get_object_position('blue_cube')",
            "quat = get_end_effector_pose().quaternion_wxyz",
            "open_gripper()",
            "move_to_pose(red_position + [0.0, 0.0, 0.18], quat)",
            "move_to_pose(red_position + [0.0, 0.0, 0.08], quat)",
            "close_gripper()",
            "move_to_pose(red_position + [0.0, 0.0, 0.22], quat)",
            "move_to_pose(blue_position + [0.0, 0.0, 0.22], quat)",
            "move_to_pose(blue_position + [0.0, 0.0, 0.13], quat)",
            "open_gripper()",
            "move_to_pose(blue_position + [0.0, 0.0, 0.22], quat)",
            "print('success:', is_success())",
        ]
    )
