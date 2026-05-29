"""Run context (injected into tools) and the result of a run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from .budget import Budget, Usage
from .messages import Message
from .tracing import Trace, Tracer

DepsT = TypeVar("DepsT")


@dataclass
class RunContext(Generic[DepsT]):
    """Carried through a whole run and injected into any tool that asks for it.

    Declare a tool parameter annotated ``RunContext`` (commonly named ``ctx``)
    and the framework will pass it in and hide it from the model's schema. From
    a tool you can reach user dependencies (``ctx.deps``), inspect the live
    budget, or emit custom trace events (``ctx.tracer.event(...)``).
    """

    deps: DepsT
    budget: Budget
    tracer: Tracer
    trace: Trace
    agent_name: str

    @property
    def usage(self) -> Usage:
        return self.budget.usage

    @property
    def remaining_tokens(self) -> int | None:
        return self.budget.remaining_tokens

    @property
    def remaining_time(self) -> float | None:
        return self.budget.remaining_time


@dataclass
class RunResult(Generic[DepsT]):
    """Everything produced by :meth:`Agent.run`."""

    output: str
    """The model's final text answer."""

    messages: list[Message] = field(default_factory=list)
    """New conversation messages from this run (excluding the system prompt).

    Pass this straight back as ``message_history`` to continue the conversation.
    """

    usage: Usage = field(default_factory=Usage)
    """Cumulative token usage for the whole run tree."""

    elapsed: float = 0.0
    """Wall-clock seconds the run took."""

    turns: int = 0
    """Number of LLM round-trips this agent performed."""

    stop_reason: str = "completed"
    """``"completed"`` when the model produced a final answer."""

    trace: Trace = field(default_factory=Trace)
    """The structured span tree. Use ``trace.format()`` or ``trace.as_dict()``."""

    def trace_dict(self) -> dict[str, Any] | None:
        return self.trace.as_dict()

    def __str__(self) -> str:
        return self.output
