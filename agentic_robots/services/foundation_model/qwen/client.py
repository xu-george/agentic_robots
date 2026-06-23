from __future__ import annotations

import os

from agentic_robots.services.base import ServiceConfig, post_json
from agentic_robots.services.foundation_model.qwen.types import ChatMessage, ChatResult


DEFAULT_QWEN_URL = os.environ.get("QWEN_SERVICE_URL", "http://127.0.0.1:8112")


class QwenClient:
    """HTTP client for the local Qwen foundation model service."""

    def __init__(self, url: str = DEFAULT_QWEN_URL, timeout_s: float = 180.0):
        self.config = ServiceConfig(url=url, timeout_s=timeout_s)

    def chat(
        self,
        messages: list[ChatMessage | dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = 512,
    ) -> ChatResult:
        payload = {
            "messages": [
                message if isinstance(message, dict) else {
                    "role": message.role,
                    "content": message.content,
                }
                for message in messages
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if model is not None:
            payload["model"] = model

        data = post_json(self.config, "/chat", payload)
        return ChatResult(
            model=str(data["model"]),
            content=str(data["content"]),
            done=bool(data.get("done", True)),
        )

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = 512,
    ) -> ChatResult:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
