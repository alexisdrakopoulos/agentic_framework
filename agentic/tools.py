"""Tools: ordinary Python functions the model can call.

Wrap a function with :func:`tool` (or pass a bare function to an ``Agent`` and
it is wrapped for you). The parameter schema is derived from type hints and the
docstring; the description is the docstring summary.

A tool may declare a parameter annotated :class:`~agentic.context.RunContext`
(conventionally named ``ctx``). It is injected at call time and hidden from the
model, giving the tool access to dependencies, the budget, and the tracer.
"""

from __future__ import annotations

import asyncio
import enum
import inspect
import json
from dataclasses import dataclass
from typing import Any, Callable, get_args, get_origin

from .context import RunContext
from .schema import (
    build_parameters_schema,
    find_context_param,
    safe_type_hints,
    summarize_docstring,
)


@dataclass
class Tool:
    """A callable exposed to the model, with its JSON schema."""

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]
    takes_context: bool = False
    context_param: str | None = None

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    async def call(self, arguments: dict[str, Any], ctx: RunContext) -> Any:
        """Invoke the underlying function with model-supplied ``arguments``.

        Sync functions are run in a worker thread so they never block the event
        loop. Argument values are lightly coerced (enums, pydantic models) to
        match the function's annotations.
        """
        kwargs = self._coerce_arguments(arguments)
        if self.takes_context and self.context_param:
            kwargs[self.context_param] = ctx
        if inspect.iscoroutinefunction(self.func):
            return await self.func(**kwargs)
        return await asyncio.to_thread(self._call_sync, kwargs)

    def _call_sync(self, kwargs: dict[str, Any]) -> Any:
        return self.func(**kwargs)

    def _coerce_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        hints = safe_type_hints(self.func, include_extras=False)
        out: dict[str, Any] = {}
        for key, value in arguments.items():
            annotation = hints.get(key)
            out[key] = _coerce(value, annotation)
        return out

    @classmethod
    def from_function(
        cls,
        func: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> "Tool":
        ctx_param = find_context_param(func, RunContext)
        skip = {ctx_param} if ctx_param else set()
        parameters = build_parameters_schema(func, skip=skip)
        return cls(
            name=name or func.__name__,
            description=(description or summarize_docstring(inspect.getdoc(func)) or func.__name__),
            parameters=parameters,
            func=func,
            takes_context=ctx_param is not None,
            context_param=ctx_param,
        )


def _coerce(value: Any, annotation: Any) -> Any:
    """Best-effort conversion of a JSON value to its annotated Python type."""
    if annotation is None or value is None:
        return value
    # Unwrap Optional[T] / T | None.
    args = get_args(annotation)
    if get_origin(annotation) is not None and type(None) in args:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            annotation = non_none[0]
    if isinstance(annotation, type):
        if issubclass(annotation, enum.Enum):
            try:
                return annotation(value)
            except ValueError:
                return value
        if hasattr(annotation, "model_validate") and isinstance(value, dict):
            try:
                return annotation.model_validate(value)  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                return value
    return value


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Any:
    """Decorator that turns a function into a :class:`Tool`.

    Usable bare (``@tool``) or with arguments (``@tool(name=..., description=...)``).
    """

    def wrap(f: Callable[..., Any]) -> Tool:
        return Tool.from_function(f, name=name, description=description)

    if func is not None:
        return wrap(func)
    return wrap


def as_tool(obj: Tool | Callable[..., Any]) -> Tool:
    """Coerce a ``Tool`` or plain callable into a ``Tool``."""
    if isinstance(obj, Tool):
        return obj
    return Tool.from_function(obj)


def stringify_result(value: Any) -> str:
    """Render a tool's return value as the string content of a tool message."""
    if value is None:
        return "null"
    if isinstance(value, str):
        return value
    if hasattr(value, "model_dump"):  # pydantic model
        try:
            value = value.model_dump()
        except Exception:  # noqa: BLE001
            return str(value)
    try:
        return json.dumps(value, default=str, ensure_ascii=False)
    except TypeError:
        return str(value)
