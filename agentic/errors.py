"""Exception hierarchy for the framework.

All errors derive from :class:`AgenticError` so callers can catch the whole
family with a single ``except``.
"""

from __future__ import annotations

from typing import Any


class AgenticError(Exception):
    """Base class for every error raised by the framework."""


class ModelError(AgenticError):
    """The underlying LLM provider call failed."""


class ToolError(AgenticError):
    """A tool failed.

    By default a tool error is *recoverable*: the message is returned to the
    model as the tool result so it can retry or choose another path. Raise with
    ``fatal=True`` to abort the whole run instead.
    """

    def __init__(self, message: str, *, fatal: bool = False) -> None:
        super().__init__(message)
        self.fatal = fatal


class SkillError(AgenticError):
    """A skill could not be loaded or referenced."""


class MaxTurnsExceeded(AgenticError):
    """The agent reached its per-run turn limit without a final answer.

    ``messages`` holds the conversation captured so far for debugging.
    """

    def __init__(self, message: str, *, messages: list[Any] | None = None) -> None:
        super().__init__(message)
        self.messages = messages or []


class BudgetExceeded(AgenticError):
    """A run exceeded its token, time, or turn budget.

    Attributes:
        kind: one of ``"tokens"``, ``"time"``, ``"turns"``.
        limit: the configured ceiling.
        used: how much had been consumed when the limit tripped.
        messages: the conversation captured so far (for debugging / partial use).
    """

    def __init__(
        self,
        message: str,
        *,
        kind: str,
        limit: Any,
        used: Any,
        messages: list[Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.limit = limit
        self.used = used
        self.messages = messages or []
