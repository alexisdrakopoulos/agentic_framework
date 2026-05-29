"""Turn a Python function into a JSON Schema for tool calling.

The goal is to let users write ordinary, type-hinted, docstring-documented
functions and get a correct OpenAI ``function`` schema for free — without
pulling in a heavy dependency. Supported annotations:

* ``str``, ``int``, ``float``, ``bool``, ``None``
* ``list[T]`` / ``set[T]`` / ``tuple[T, ...]`` / ``Sequence[T]``
* ``dict[str, V]``
* ``Optional[T]`` / ``T | None`` and other ``Union`` types
* ``Literal[...]`` and ``enum.Enum`` subclasses (rendered as ``enum``)
* ``Annotated[T, "description"]`` (the string becomes the field description)
* any pydantic ``BaseModel`` (uses its own ``model_json_schema``)

Parameter descriptions are taken from ``Annotated`` metadata or, failing that,
from a Google-/reST-style docstring.
"""

from __future__ import annotations

import enum
import inspect
import re
import types
import typing
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

_PRIMITIVES: dict[type, dict[str, str]] = {
    str: {"type": "string"},
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
    type(None): {"type": "null"},
}

_SEQUENCE_ORIGINS = (list, set, frozenset, tuple, typing.Sequence)


def safe_type_hints(func: Any, *, include_extras: bool = True) -> dict[str, Any]:
    """``typing.get_type_hints`` that degrades gracefully.

    If the whole-function resolution fails (e.g. an annotation references a type
    that is local to the defining function and therefore invisible to
    ``get_type_hints``), fall back to resolving each annotation individually so
    that one bad annotation does not discard the rest.
    """
    try:
        return get_type_hints(func, include_extras=include_extras)
    except Exception:  # noqa: BLE001
        pass

    hints: dict[str, Any] = {}
    raw = getattr(func, "__annotations__", {}) or {}
    func_globals = getattr(func, "__globals__", {})
    for key, value in raw.items():
        if not isinstance(value, str):
            hints[key] = value
            continue
        try:
            hints[key] = eval(value, func_globals)  # noqa: S307 - resolve own annotations
        except Exception:  # noqa: BLE001 - leave this one unresolved
            continue
    return hints


def _is_union(origin: Any) -> bool:
    return origin is Union or origin is getattr(types, "UnionType", ())


def _enum_schema(values: list[Any]) -> dict[str, Any]:
    schema: dict[str, Any] = {"enum": values}
    py_types = {type(v) for v in values}
    if py_types == {str}:
        schema["type"] = "string"
    elif py_types <= {int, bool}:
        schema["type"] = "integer"
    elif py_types <= {int, float}:
        schema["type"] = "number"
    return schema


def type_to_schema(annotation: Any) -> tuple[dict[str, Any], str | None]:
    """Return ``(json_schema, description)`` for a type annotation.

    ``description`` is non-None only when the annotation is ``Annotated`` with a
    string. ``None``-ness for optionality is handled by the caller.
    """
    # Annotated[T, ...] -> unwrap, pull a string description from the metadata.
    if hasattr(annotation, "__metadata__"):
        inner = get_args(annotation)[0]
        schema, _ = type_to_schema(inner)
        description = next((m for m in annotation.__metadata__ if isinstance(m, str)), None)
        return schema, description

    if annotation is inspect.Parameter.empty or annotation is Any:
        return {}, None  # accept anything

    # pydantic models bring their own schema.
    if isinstance(annotation, type) and hasattr(annotation, "model_json_schema"):
        try:
            return annotation.model_json_schema(), None  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - fall through to generic handling
            pass

    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return _enum_schema([m.value for m in annotation]), None

    if annotation in _PRIMITIVES:
        return dict(_PRIMITIVES[annotation]), None

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Literal:
        return _enum_schema(list(args)), None

    if _is_union(origin):
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            # Optional[T] -> just T (optionality is expressed via `required`).
            return type_to_schema(non_none[0])
        return {"anyOf": [type_to_schema(a)[0] for a in non_none]}, None

    if origin in _SEQUENCE_ORIGINS or annotation in _SEQUENCE_ORIGINS:
        item = next((a for a in args if a is not Ellipsis), None)
        items_schema = type_to_schema(item)[0] if item is not None else {}
        return {"type": "array", "items": items_schema}, None

    if origin in (dict, typing.Mapping) or annotation is dict:
        if len(args) == 2:
            return {"type": "object", "additionalProperties": type_to_schema(args[1])[0]}, None
        return {"type": "object"}, None

    # Unknown -> permissive.
    return {}, None


def build_parameters_schema(func: Any, *, skip: set[str] | None = None) -> dict[str, Any]:
    """Build the ``parameters`` object schema for ``func``."""
    skip = skip or set()
    sig = inspect.signature(func)
    hints = safe_type_hints(func, include_extras=True)
    doc_params = parse_docstring_params(inspect.getdoc(func) or "")

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name in skip:
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        annotation = hints.get(name, param.annotation)
        schema, annotated_desc = type_to_schema(annotation)
        description = annotated_desc or doc_params.get(name)
        if description:
            schema = {**schema, "description": description}
        properties[name] = schema
        if param.default is inspect.Parameter.empty:
            required.append(name)

    out: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        out["required"] = required
    out["additionalProperties"] = False
    return out


def find_context_param(func: Any, ctx_type: type) -> str | None:
    """Return the name of the parameter annotated as ``ctx_type`` (or its
    generic alias), if any. Used to inject the run context into tools."""
    hints = safe_type_hints(func, include_extras=False)
    sig = inspect.signature(func)
    for name in sig.parameters:
        annotation = hints.get(name)
        if annotation is None:
            continue
        if annotation is ctx_type or get_origin(annotation) is ctx_type:
            return name
    return None


# --- docstring parsing ------------------------------------------------------------

_SECTION_RE = re.compile(r"^(Args|Arguments|Parameters)\s*:\s*$", re.IGNORECASE)
_GOOGLE_PARAM_RE = re.compile(r"^(\w+)\s*(?:\([^)]*\))?\s*:\s*(.+)$")
_REST_PARAM_RE = re.compile(r"^:param\s+(?:\w+\s+)?(\w+)\s*:\s*(.+)$")


def parse_docstring_params(doc: str) -> dict[str, str]:
    """Extract ``{param_name: description}`` from a Google- or reST-style docstring."""
    params: dict[str, str] = {}
    if not doc:
        return params

    lines = doc.splitlines()
    in_args = False
    current: str | None = None

    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()

        rest = _REST_PARAM_RE.match(stripped)
        if rest:
            params[rest.group(1)] = rest.group(2).strip()
            current = None
            continue

        if _SECTION_RE.match(stripped):
            in_args = True
            current = None
            continue

        if in_args:
            if not stripped:  # blank line ends the section
                in_args = False
                current = None
                continue
            # A new section header (e.g. "Returns:") ends the args block.
            if re.match(r"^[A-Z][A-Za-z ]+:\s*$", stripped) and not _GOOGLE_PARAM_RE.match(stripped):
                in_args = False
                current = None
                continue
            match = _GOOGLE_PARAM_RE.match(stripped)
            if match:
                current = match.group(1)
                params[current] = match.group(2).strip()
            elif current:  # continuation line for the previous param
                params[current] += " " + stripped

    return params


def summarize_docstring(doc: str | None) -> str:
    """Return the leading summary paragraph of a docstring (the tool description)."""
    if not doc:
        return ""
    summary: list[str] = []
    for line in inspect.cleandoc(doc).splitlines():
        if not line.strip():
            break
        if _SECTION_RE.match(line.strip()):
            break
        summary.append(line.strip())
    return " ".join(summary).strip()
