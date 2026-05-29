"""Offline models for tests, demos, and deterministic local runs.

:class:`FunctionModel` drives an agent from a plain Python function instead of a
network call, so the full run loop — tools, skills, subagents, budgets, tracing
— can be exercised without an API key. (Same idea as pydantic-ai's
``FunctionModel`` / ``TestModel``.)
"""

from __future__ import annotations

import json
from typing import Any, Callable, Sequence, Union

from .messages import Message, ToolCall
from .models import Model, ModelResponse
from .tools import Tool

# What a FunctionModel function may return for one turn:
#   * a string                       -> final assistant answer
#   * a list of (tool_name, kwargs)  -> tool calls to make this turn
#   * a Message                      -> use verbatim
FunctionResult = Union[str, "Message", "list[tuple[str, dict]]"]


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class FunctionModel(Model):
    """A model whose every turn is decided by a Python function.

    The function receives ``(messages, tools)`` and returns one of the
    :data:`FunctionResult` shapes. Token usage is estimated from text length so
    budget behaviour is exercised deterministically.
    """

    def __init__(
        self,
        fn: Callable[[Sequence[Message], Sequence[Tool]], FunctionResult],
        *,
        name: str = "function-model",
    ) -> None:
        self.fn = fn
        self.name = name

    async def generate(
        self,
        messages: Sequence[Message],
        tools: Sequence[Tool],
        *,
        timeout: float | None = None,
    ) -> ModelResponse:
        result = self.fn(messages, tools)
        prompt_tokens = sum(_estimate_tokens(m.content or "") for m in messages)

        if isinstance(result, Message):
            assistant = result
            completion = _estimate_tokens(assistant.content or "") + 4 * len(assistant.tool_calls)
            finish = "tool_calls" if assistant.tool_calls else "stop"
        elif isinstance(result, str):
            assistant = Message.assistant(content=result)
            completion = _estimate_tokens(result)
            finish = "stop"
        else:  # list of (name, kwargs)
            tool_calls = [
                ToolCall(id=f"call_{i}", name=name, arguments=json.dumps(kwargs))
                for i, (name, kwargs) in enumerate(result)
            ]
            assistant = Message.assistant(tool_calls=tool_calls)
            completion = 4 * len(tool_calls)
            finish = "tool_calls"

        return ModelResponse(
            message=assistant,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion,
            total_tokens=prompt_tokens + completion,
            finish_reason=finish,
        )


class ScriptedModel(Model):
    """Returns a fixed sequence of responses, one per call (handy for tests)."""

    def __init__(self, responses: list[FunctionResult], *, name: str = "scripted-model") -> None:
        self._responses = list(responses)
        self._i = 0
        self.name = name

    async def generate(
        self,
        messages: Sequence[Message],
        tools: Sequence[Tool],
        *,
        timeout: float | None = None,
    ) -> ModelResponse:
        if self._i >= len(self._responses):
            # Nothing scripted left: end the conversation cleanly.
            return ModelResponse(message=Message.assistant(content=""), finish_reason="stop")
        response = self._responses[self._i]
        self._i += 1
        delegate = FunctionModel(lambda m, t, _r=response: _r, name=self.name)
        return await delegate.generate(messages, tools, timeout=timeout)
