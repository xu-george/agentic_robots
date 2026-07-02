from __future__ import annotations

import numpy as np

from agentic_robots.agent.code_agent import CodeAgent, CodeAgentConfig, make_scene_summary


class _RetryLlm:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, prompt: str, **kwargs):
        self.prompts.append(prompt)
        if len(self.prompts) == 1:
            return _Completion("close_env()")
        return _Completion("mark_success()")


class _Completion:
    def __init__(self, content: str) -> None:
        self.content = content


class _ResetRequiredEnv:
    def __init__(self) -> None:
        self.reset_count = 0
        self.is_reset = False
        self.succeeded = False

    def reset(self) -> None:
        self.reset_count += 1
        self.is_reset = True

    def get_obs(self):
        if not self.is_reset:
            raise RuntimeError("reset required")
        return {
            "ground_truth": {
                "objects": {
                    "banana": {
                        "position": np.array([0.5, 0.0, 0.1]),
                        "quaternion_wxyz": np.array([1.0, 0.0, 0.0, 0.0]),
                    }
                }
            }
        }

    def close_env(self) -> None:
        self.is_reset = False

    def mark_success(self) -> None:
        self.succeeded = True

    def is_success(self) -> bool:
        return self.succeeded


def test_code_agent_resets_before_building_revision_prompt() -> None:
    env = _ResetRequiredEnv()
    env.reset()
    llm = _RetryLlm()
    config = CodeAgentConfig(
        task_description="lift banana",
        api_functions={
            "close_env": env.close_env,
            "mark_success": env.mark_success,
        },
        success_fn=env.is_success,
        scene_context_fn=lambda: make_scene_summary(env),
    )

    result = CodeAgent(env, config, llm=llm, allow_fallback=False, max_attempts=2).run()

    assert result.task_success
    assert env.reset_count == 2
    assert len(result.attempts) == 2
