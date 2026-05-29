"""Provider-neutral chat message types.

These mirror the OpenAI chat schema closely but keep the framework decoupled
from any single SDK. Every type knows how to render itself to the dict shape
the OpenAI client expects via ``to_openai()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(slots=True)
class ToolCall:
    """A request from the model to invoke a tool."""

    id: str
    name: str
    arguments: str  # raw JSON string exactly as emitted by the model

    def to_openai(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass(slots=True)
class Message:
    """A single conversation message."""

    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None  # set when role == "tool"

    # --- convenience constructors -------------------------------------------------

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls(role="user", content=content)

    @classmethod
    def assistant(
        cls, content: str | None = None, tool_calls: list[ToolCall] | None = None
    ) -> "Message":
        return cls(role="assistant", content=content, tool_calls=tool_calls or [])

    @classmethod
    def tool(cls, tool_call_id: str, content: str) -> "Message":
        return cls(role="tool", content=content, tool_call_id=tool_call_id)

    # --- serialisation ------------------------------------------------------------

    def to_openai(self) -> dict[str, Any]:
        if self.role == "tool":
            return {
                "role": "tool",
                "tool_call_id": self.tool_call_id,
                "content": self.content or "",
            }

        msg: dict[str, Any] = {"role": self.role}
        # Assistant messages may carry tool calls with null content; everything
        # else needs a content string.
        if self.content is not None or not self.tool_calls:
            msg["content"] = self.content or ""
        if self.tool_calls:
            msg["tool_calls"] = [tc.to_openai() for tc in self.tool_calls]
        return msg

    def short(self, width: int = 80) -> str:
        """A compact one-line summary for logs."""
        if self.tool_calls:
            calls = ", ".join(f"{tc.name}(...)" for tc in self.tool_calls)
            return f"{self.role}: →{calls}"
        text = (self.content or "").replace("\n", " ")
        if len(text) > width:
            text = text[: width - 1] + "…"
        return f"{self.role}: {text}"
