"""agentic — a small, clean async agentic framework.

Core concepts
-------------
* :class:`Agent`      — an LLM + tools + skills, optionally delegating to subagents.
* :func:`tool`        — turn a typed Python function into a model-callable tool.
* :class:`Skill`      — a reusable bundle of instructions (+ optional tools),
                        auto-loaded or revealed on demand via progressive disclosure.
* :class:`Budget`     — token / time / turn ceilings enforced across a whole run.
* :class:`RunContext` — injected into tools; exposes deps, budget, and tracing.
* :class:`Trace`      — a structured, serialisable record of everything that happened.

Quick start
-----------
>>> import asyncio
>>> from agentic import Agent, tool
>>>
>>> @tool
... def add(a: int, b: int) -> int:
...     "Add two integers."
...     return a + b
>>>
>>> agent = Agent("gpt-4o-mini", instructions="You are concise.", tools=[add])
>>> asyncio.run(agent.run("What is 21 + 21?")).output  # doctest: +SKIP
'42'
"""

from __future__ import annotations

from .agent import Agent
from .budget import Budget, Usage
from .context import RunContext, RunResult
from .errors import (
    AgenticError,
    BudgetExceeded,
    MaxTurnsExceeded,
    ModelError,
    SkillError,
    ToolError,
)
from .messages import Message, ToolCall
from .models import Model, ModelResponse, OpenAIModel
from .skills import Skill
from .tools import Tool, tool
from .tracing import Span, Trace, Tracer, configure_logging

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "tool",
    "Tool",
    "Skill",
    "Budget",
    "Usage",
    "RunContext",
    "RunResult",
    "Model",
    "ModelResponse",
    "OpenAIModel",
    "Message",
    "ToolCall",
    "Trace",
    "Tracer",
    "Span",
    "configure_logging",
    "AgenticError",
    "ModelError",
    "ToolError",
    "SkillError",
    "BudgetExceeded",
    "MaxTurnsExceeded",
    "__version__",
]
