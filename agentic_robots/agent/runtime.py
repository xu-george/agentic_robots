from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class RuntimeResult:
    success: bool
    stdout: str
    error: str | None = None


class AgentRuntime:
    """Execute agent-written high-level code against an allowlisted API."""

    def __init__(self, functions: dict[str, Callable[..., Any]]) -> None:
        self.functions = functions

    def run(self, code: str) -> RuntimeResult:
        stdout = io.StringIO()
        globals_dict = {
            "__builtins__": {
                "abs": abs,
                "bool": bool,
                "dict": dict,
                "float": float,
                "int": int,
                "len": len,
                "list": list,
                "max": max,
                "min": min,
                "print": print,
                "range": range,
                "str": str,
                "tuple": tuple,
            },
            **self.functions,
        }
        try:
            with contextlib.redirect_stdout(stdout):
                exec(compile(code, "<agent_code>", "exec"), globals_dict, {})
        except Exception as exc:
            return RuntimeResult(success=False, stdout=stdout.getvalue(), error=str(exc))
        return RuntimeResult(success=True, stdout=stdout.getvalue())
