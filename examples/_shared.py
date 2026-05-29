"""Shared helpers for the example guide.

Every numbered example runs **offline by default** so you can explore the
framework with no API key and no cost. If ``OPENAI_API_KEY`` is set, the same
examples transparently use a real model instead.

The trick is :func:`demo_model`: it returns an :class:`agentic.OpenAIModel` when
a key is present, otherwise an :class:`agentic.testing.FunctionModel` — a model
whose every turn is decided by a small Python "brain" function. The teaching
code (agents, tools, skills) is identical either way; only the brain is demo
scaffolding.
"""

from __future__ import annotations

import os
from typing import Callable

from agentic import OpenAIModel
from agentic.testing import FunctionModel


def demo_model(brain: Callable, *, name: str = "demo-llm", model: str = "gpt-4o-mini"):
    """A real OpenAI model if ``OPENAI_API_KEY`` is set, else an offline brain."""
    if os.getenv("OPENAI_API_KEY"):
        return OpenAIModel(model)
    return FunctionModel(brain, name=name)


def using_real_model() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def banner(title: str) -> None:
    line = "─" * len(title)
    print(f"\n{title}\n{line}")


def steps_taken(messages) -> int:
    """How many assistant turns have happened so far (handy for scripted brains)."""
    return sum(1 for m in messages if m.role == "assistant")
