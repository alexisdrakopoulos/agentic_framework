"""Typed structured output.

When an agent is given an ``output_type`` (a pydantic model, a dataclass, a
``TypedDict``, or even a plain type like ``list[str]``), it returns a validated
instance of that type instead of free text.

This is implemented provider-neutrally with a synthetic ``final_result`` tool:
its JSON Schema is derived from the output type, the model "answers" by calling
it, and the arguments are validated into the target type. The approach works
with any tool-calling model (including the offline ``FunctionModel``) and gives
the model a chance to self-correct if validation fails.

Schema generation and validation use pydantic's ``TypeAdapter`` (pydantic is
already present as a dependency of the OpenAI client). It is imported lazily so
the rest of the framework stays pydantic-free.
"""

from __future__ import annotations

from typing import Any


class OutputSpec:
    """Describes a run's required output type and how to validate it."""

    def __init__(self, output_type: Any, *, tool_name: str = "final_result") -> None:
        try:
            from pydantic import TypeAdapter
        except ImportError as exc:  # pragma: no cover - pydantic ships with openai
            raise ImportError(
                "structured output requires pydantic (installed automatically with openai)"
            ) from exc

        self.output_type = output_type
        self.tool_name = tool_name
        self._adapter = TypeAdapter(output_type)

        schema = self._adapter.json_schema()
        if schema.get("type") == "object" and "properties" in schema:
            # Object-like outputs (models/dataclasses/TypedDicts) map directly.
            self.parameters = schema
            self._wrapped = False
        else:
            # Scalar/list/etc. must be wrapped in an object for tool parameters.
            defs = schema.pop("$defs", None)
            params: dict[str, Any] = {
                "type": "object",
                "properties": {"value": schema},
                "required": ["value"],
                "additionalProperties": False,
            }
            if defs:
                params["$defs"] = defs  # keep $ref targets resolvable at the top level
            self.parameters = params
            self._wrapped = True

    def validate(self, args: dict[str, Any]) -> Any:
        """Validate model-supplied arguments into an instance of the output type.

        Raises whatever ``pydantic`` raises on invalid data; the caller turns
        that into a recoverable tool error so the model can try again.
        """
        payload = args.get("value") if self._wrapped else args
        return self._adapter.validate_python(payload)
