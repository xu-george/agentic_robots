from __future__ import annotations

import contextlib
import importlib
import io
import math
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

_SAFE_IMPORT_ROOTS = frozenset({"numpy", "np", "math", "time"})


def _safe_import(
    name: str,
    globals: dict[str, Any] | None = None,
    locals: dict[str, Any] | None = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
) -> Any:
    """Restricted ``__import__`` for the agent runtime.

    Only a small safelist of pure-compute modules is importable
    (numpy, math, time). Importing anything else (e.g. ``os``, ``subprocess``,
    ``socket``) raises ``ImportError`` so the LLM-generated code cannot touch
    the filesystem, network, or process machinery.
    """
    if level != 0:
        raise ImportError("relative imports are not allowed in agent code")
    root = name.split(".")[0]
    if root not in _SAFE_IMPORT_ROOTS:
        raise ImportError(f"import of {name!r} is not allowed in agent code")
    return importlib.import_module(name)


@dataclass(frozen=True)
class RuntimeResult:
    success: bool
    stdout: str
    error: str | None = None


class AgentRuntime:
    """Execute agent-written high-level code against an allowlisted API.

    The sandbox exposes the allowlisted API functions plus a tiny safelist of
    pure-compute modules (``numpy`` as ``np``, ``math``, ``time``). LLM-written
    ``import`` statements for those modules succeed; anything else is rejected.
    This lets generated code that naturally uses ``np.array(...)`` run without
    crashing on ``__import__ not found`` while still keeping the sandbox sealed.
    """

    def __init__(self, functions: dict[str, Callable[..., Any]]) -> None:
        self.functions = functions

    def run(self, code: str) -> RuntimeResult:
        stdout = io.StringIO()
        globals_dict = {
            "__builtins__": {
                "abs": abs,
                "bool": bool,
                "dict": dict,
                "enumerate": enumerate,
                "float": float,
                "int": int,
                "len": len,
                "list": list,
                "max": max,
                "min": min,
                "print": print,
                "range": range,
                "round": round,
                "str": str,
                "sum": sum,
                "tuple": tuple,
                "zip": zip,
                "__import__": _safe_import,
            },
            "np": np,
            "numpy": np,
            "math": math,
            "time": time,
            **self.functions,
        }
        try:
            with contextlib.redirect_stdout(stdout):
                exec(compile(code, "<agent_code>", "exec"), globals_dict, {})
        except Exception as exc:
            return RuntimeResult(success=False, stdout=stdout.getvalue(), error=str(exc))
        return RuntimeResult(success=True, stdout=stdout.getvalue())
