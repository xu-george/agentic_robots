from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any


class ApiBase(ABC):
    """Base class for CAP-X-style APIs exposed to robot task code."""

    @abstractmethod
    def functions(self) -> dict[str, Callable[..., Any]]:
        """Return public function name to callable mapping."""

    def combined_doc(self) -> str:
        """Return signatures and docstrings for all exposed API functions."""
        lines: list[str] = []
        for name, fn in self.functions().items():
            try:
                signature = str(inspect.signature(fn))
            except (TypeError, ValueError):
                signature = "(...)"
            lines.append(f"{name}{signature}")

            doc = inspect.getdoc(fn)
            if doc:
                lines.append("  Doc:")
                lines.extend(f"    {line}" for line in doc.splitlines())
            lines.append("")

        return "\n".join(lines).strip()
