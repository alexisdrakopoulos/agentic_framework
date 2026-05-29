"""Model abstraction and the OpenAI implementation.

The framework talks to LLMs only through the small :class:`Model` interface, so
swapping providers later means writing one new subclass. :class:`OpenAIModel`
uses the Chat Completions API (the de-facto standard that most providers mirror)
via the official async client.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .errors import ModelError
from .messages import Message, ToolCall
from .tools import Tool


@dataclass
class ModelResponse:
    """One assistant turn returned by a model."""

    message: Message
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str = "stop"
    raw: Any = field(default=None, repr=False)


class Model:
    """Interface every model backend implements."""

    name: str = "model"

    async def generate(
        self,
        messages: Sequence[Message],
        tools: Sequence[Tool],
        *,
        timeout: float | None = None,
    ) -> ModelResponse:
        raise NotImplementedError


class OpenAIModel(Model):
    """A model backed by the OpenAI Chat Completions API.

    Args:
        model: model id, e.g. ``"gpt-4o-mini"``.
        client: a pre-built ``openai.AsyncOpenAI``; one is created if omitted.
        api_key / base_url: forwarded when creating the default client
            (``base_url`` lets you point at any OpenAI-compatible endpoint).
        temperature: optional sampling temperature.
        **default_params: extra parameters merged into every request
            (``max_tokens``, ``top_p``, ``seed``, ...).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        client: Any = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        **default_params: Any,
    ) -> None:
        self.name = model
        self.temperature = temperature
        self.default_params = default_params
        if client is not None:
            self.client = client
        else:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:  # pragma: no cover - import guard
                raise ModelError(
                    "the 'openai' package is required for OpenAIModel; install it with "
                    "`uv add openai` or `pip install openai`"
                ) from exc
            kwargs: dict[str, Any] = {}
            if api_key is not None:
                kwargs["api_key"] = api_key
            if base_url is not None:
                kwargs["base_url"] = base_url
            self.client = AsyncOpenAI(**kwargs)

    async def generate(
        self,
        messages: Sequence[Message],
        tools: Sequence[Tool],
        *,
        timeout: float | None = None,
    ) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": self.name,
            "messages": [m.to_openai() for m in messages],
        }
        if tools:
            payload["tools"] = [t.to_openai() for t in tools]
            payload["tool_choice"] = "auto"
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        payload.update(self.default_params)
        if timeout is not None:
            payload["timeout"] = timeout

        try:
            resp = await self.client.chat.completions.create(**payload)
        except Exception as exc:  # noqa: BLE001 - normalise provider errors
            raise ModelError(f"OpenAI request failed: {exc}") from exc

        choice = resp.choices[0]
        raw_msg = choice.message
        tool_calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments or "{}")
            for tc in (raw_msg.tool_calls or [])
        ]
        assistant = Message(role="assistant", content=raw_msg.content, tool_calls=tool_calls)

        usage = getattr(resp, "usage", None)
        return ModelResponse(
            message=assistant,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
            finish_reason=choice.finish_reason or "stop",
            raw=resp,
        )
