"""Tracing: a structured span tree plus optional live logging.

Every interesting step of a run (the run itself, each turn, each model call,
each tool call, each subagent delegation, each skill load) is recorded as a
:class:`Span`. Spans nest to form a tree that is attached to the
:class:`~agentic.context.RunResult` and is JSON-serialisable for offline
inspection. If a ``logging.Logger`` is supplied, spans are also emitted live as
an indented, human-readable trace.

Nesting is tracked with a :class:`contextvars.ContextVar`, so it stays correct
even when tool calls run concurrently via ``asyncio.gather`` (each task gets its
own copy of the context).
"""

from __future__ import annotations

import contextvars
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

# The span currently in scope for the running task.
_current_span: contextvars.ContextVar["Span | None"] = contextvars.ContextVar(
    "agentic_current_span", default=None
)

DEFAULT_LOGGER_NAME = "agentic.trace"

# Symbols used in the live log, keyed by span status.
_KIND_ICON = {
    "run": "▶",
    "subagent": "⮑",
    "turn": "•",
    "model": "🧠",
    "tool": "🔧",
    "skill": "📦",
}


@dataclass(slots=True)
class Span:
    """A single timed step in a run."""

    id: str
    name: str
    kind: str  # run | subagent | turn | model | tool | skill
    parent_id: str | None
    depth: int
    start: float
    end: float | None = None
    status: str = "ok"  # ok | error
    error: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    children: list["Span"] = field(default_factory=list)

    @property
    def duration(self) -> float | None:
        if self.end is None:
            return None
        return self.end - self.start

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "status": self.status,
            "duration_s": round(self.duration, 4) if self.duration is not None else None,
            "attributes": self.attributes,
            "events": self.events,
            "error": self.error,
            "children": [c.as_dict() for c in self.children],
        }


class Trace:
    """Holds every span produced during a run and exposes the root tree."""

    def __init__(self) -> None:
        self.spans: list[Span] = []
        self.root: Span | None = None

    def as_dict(self) -> dict[str, Any] | None:
        return self.root.as_dict() if self.root else None

    def format(self) -> str:
        """Render the trace as an indented text tree."""
        if self.root is None:
            return "(empty trace)"
        lines: list[str] = []
        self._format_span(self.root, lines)
        return "\n".join(lines)

    def _format_span(self, span: Span, lines: list[str]) -> None:
        icon = _KIND_ICON.get(span.kind, "·")
        if span.status == "error":
            icon = "✖"
        dur = f" ({span.duration:.3f}s)" if span.duration is not None else ""
        attrs = ""
        if span.attributes:
            shown = {
                k: v
                for k, v in span.attributes.items()
                if k in ("model", "total_tokens", "tools", "stop_reason", "result")
            }
            if shown:
                attrs = " " + " ".join(f"{k}={v}" for k, v in shown.items())
        lines.append(f"{'  ' * span.depth}{icon} {span.kind}:{span.name}{dur}{attrs}")
        for event in span.events:
            lines.append(f"{'  ' * (span.depth + 1)}· {event.get('message', '')}")
        for child in span.children:
            self._format_span(child, lines)


class Tracer:
    """Creates spans and (optionally) logs them live.

    Pass ``logger=None`` to capture the trace tree silently; pass a configured
    logger (see :func:`configure_logging`) to also stream the run as it happens.
    """

    def __init__(
        self,
        trace: Trace | None = None,
        *,
        logger: logging.Logger | None = None,
        log_level: int = logging.INFO,
    ) -> None:
        self.trace = trace if trace is not None else Trace()
        self.logger = logger
        self.log_level = log_level

    @asynccontextmanager
    async def span(self, kind: str, name: str, **attributes: Any) -> AsyncIterator[Span]:
        parent = _current_span.get()
        span = Span(
            id=uuid.uuid4().hex,
            name=name,
            kind=kind,
            parent_id=parent.id if parent else None,
            depth=(parent.depth + 1) if parent else 0,
            start=time.monotonic(),
            attributes=dict(attributes),
        )
        self.trace.spans.append(span)
        if parent is not None:
            parent.children.append(span)
        else:
            self.trace.root = span

        token = _current_span.set(span)
        self._log(span, "start")
        try:
            yield span
        except BaseException as exc:  # noqa: BLE001 - record then re-raise
            span.status = "error"
            span.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            span.end = time.monotonic()
            _current_span.reset(token)
            self._log(span, "end")

    def event(self, message: str, **fields: Any) -> None:
        """Attach a point-in-time event to the current span."""
        span = _current_span.get()
        record = {"message": message, **fields}
        if span is not None:
            span.events.append(record)
        if self.logger is not None:
            depth = (span.depth + 1) if span else 0
            self.logger.log(self.log_level, "%s· %s", "  " * depth, message)

    def _log(self, span: Span, phase: str) -> None:
        if self.logger is None or not self.logger.isEnabledFor(self.log_level):
            return
        indent = "  " * span.depth
        if phase == "start":
            icon = _KIND_ICON.get(span.kind, "·")
            extra = ""
            if span.kind == "model" and "model" in span.attributes:
                extra = f" [{span.attributes['model']}]"
            elif span.kind in ("tool", "skill") or span.kind == "subagent":
                extra = ""
            self.logger.log(self.log_level, "%s%s %s:%s%s", indent, icon, span.kind, span.name, extra)
        else:  # end
            icon = "✖" if span.status == "error" else "✓"
            dur = f"{span.duration:.3f}s" if span.duration is not None else "?"
            note = ""
            if span.kind == "model" and "total_tokens" in span.attributes:
                note = f" +{span.attributes['total_tokens']} tok"
            elif span.status == "error" and span.error:
                note = f" {span.error}"
            elif span.kind in ("run", "subagent") and "total_tokens" in span.attributes:
                note = f" {span.attributes['total_tokens']} tok"
            self.logger.log(
                self.log_level, "%s%s %s:%s (%s)%s", indent, icon, span.kind, span.name, dur, note
            )


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Attach a clean console handler to the ``agentic`` logger and return it.

    Call this once from an application/script to see live traces. It is a no-op
    to call more than once (it will not stack duplicate handlers).
    """
    logger = logging.getLogger("agentic")
    logger.setLevel(level)
    if not any(getattr(h, "_agentic", False) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler._agentic = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    logger.propagate = False
    return logging.getLogger(DEFAULT_LOGGER_NAME)
