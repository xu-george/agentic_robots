from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_robots.agent.runtime import AgentRuntime, RuntimeResult
from agentic_robots.services.foundation_model.qwen.client import QwenClient


DEFAULT_SYSTEM_PROMPT = """You write short Python robot task programs.
Use only the provided functions. Do not import modules.
Positions are usually numpy arrays, so use indexing or vector addition.
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


@dataclass(frozen=True)
class CodeAgentConfig:
    task_description: str
    api_functions: dict[str, Callable[..., Any]]
    success_fn: Callable[[], bool]
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    api_description: str | None = None
    api_schema_path: str | Path | None = None
    api_function_names: list[str] | None = None
    extra_api_functions: dict[str, dict[str, Any]] | None = None
    scene_context_fn: Callable[[], Any] | None = None
    diagnostics_fn: Callable[[], Any] | None = None
    revision_guidance: str = "Revise the code and return only executable Python code."
    fallback_code: str | None = None


class CodeAgent:
    """Generic LLM/runtime loop for robot code tasks.

    The agent has no task-specific branches. Callers provide the environment,
    task description, allowlisted APIs, success check, and optional diagnostics
    through CodeAgentConfig.
    """

    def __init__(
        self,
        env: Any,
        config: CodeAgentConfig,
        *,
        llm: QwenClient | None = None,
        allow_fallback: bool = True,
        max_attempts: int = 2,
    ) -> None:
        self.env = env
        self.config = config
        self.llm = llm or QwenClient()
        self.allow_fallback = allow_fallback
        self.max_attempts = max(1, max_attempts)

    def run(self, task_description: str | None = None) -> AgentRunResult:
        attempts: list[AgentAttempt] = []
        code, llm_used = self._generate_code(task_description)
        attempts.append(self._run_attempt(code))

        while llm_used and not attempts[-1].task_success and len(attempts) < self.max_attempts:
            self._reset_env_for_retry()
            code = self._revise_code(task_description, attempts[-1])
            attempts.append(self._run_attempt(code))

        final_attempt = attempts[-1]
        return AgentRunResult(
            code=final_attempt.code,
            runtime=final_attempt.runtime,
            task_success=final_attempt.task_success,
            llm_used=llm_used,
            attempts=tuple(attempts),
        )

    def _run_attempt(self, code: str) -> AgentAttempt:
        runtime = AgentRuntime(self.config.api_functions).run(code)
        task_success = bool(runtime.success and self.config.success_fn())
        return AgentAttempt(code=code, runtime=runtime, task_success=task_success)

    def _generate_code(self, task_description: str | None) -> tuple[str, bool]:
        try:
            result = self.llm.complete(
                self._build_prompt(task_description),
                system=self.config.system_prompt,
                temperature=0.0,
                max_tokens=2048,
            )
            return extract_code(result.content), True
        except Exception:
            if not self.allow_fallback or self.config.fallback_code is None:
                raise
            return self.config.fallback_code, False

    def _revise_code(self, task_description: str | None, previous: AgentAttempt) -> str:
        prompt = (
            f"{self._build_prompt(task_description)}\n\n"
            "The previous code did not complete the task.\n"
            f"Previous code:\n{previous.code}\n\n"
            f"Runtime success: {previous.runtime.success}\n"
            f"Task success: {previous.task_success}\n"
            f"Runtime stdout:\n{previous.runtime.stdout}\n"
            f"Runtime error: {previous.runtime.error}\n"
            f"Failure diagnostics: {self._diagnostics()}\n"
            f"{self.config.revision_guidance}"
        )
        result = self.llm.complete(
            prompt,
            system=self.config.system_prompt,
            temperature=0.0,
            max_tokens=2048,
        )
        return extract_code(result.content)

    def _build_prompt(self, task_description: str | None) -> str:
        task = task_description or self.config.task_description
        parts = [f"Task: {task}"]

        api_desc = self._get_api_description()
        if api_desc:
            parts.append(f"Available APIs:\n{api_desc}")
        else:
            api_names = "\n".join(f"- {name}" for name in sorted(self.config.api_functions))
            parts.append(f"Available APIs:\n{api_names}")

        if self.config.scene_context_fn is not None:
            parts.append(f"Scene context: {self.config.scene_context_fn()}")

        parts.append("Write high-level Python code using only the available APIs.")
        return "\n".join(parts)

    def _get_api_description(self) -> str | None:
        if self.config.api_schema_path is not None or self.config.extra_api_functions is not None:
            from agentic_robots.utils.api_schema import build_api_description

            return build_api_description(
                function_names=self.config.api_function_names,
                schema_path=self.config.api_schema_path,
                extra_functions=self.config.extra_api_functions,
            )
        return self.config.api_description

    def _diagnostics(self) -> Any:
        if self.config.diagnostics_fn is None:
            return None
        return self.config.diagnostics_fn()

    def _reset_env_for_retry(self) -> None:
        reset = getattr(self.env, "reset", None)
        if callable(reset):
            reset()


def make_scene_summary(env: Any) -> dict[str, dict[str, Any]]:
    obs = env.get_obs()
    objects = obs["ground_truth"]["objects"]
    return {
        name: {
            "position": value["position"].round(4).tolist(),
            "quaternion_wxyz": value["quaternion_wxyz"].round(4).tolist(),
        }
        for name, value in objects.items()
    }


def make_robot_diagnostics(env: Any) -> dict[str, Any]:
    obs = env.get_obs()
    eef = obs["robot"]["eef_position"]
    return {
        "eef_position": eef.round(4).tolist(),
        "gripper_positions": obs["robot"]["gripper_positions"].round(4).tolist(),
    }


def extract_code(content: str) -> str:
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
