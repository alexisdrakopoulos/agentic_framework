"""Tracing & logging: a structured span tree plus live structlog output.

Two complementary records of a run:

* A :class:`Trace` — an in-memory tree of :class:`Span` objects (run → turn →
  model/tool/subagent...). It is attached to every
  :class:`~agentic.context.RunResult`, is JSON-serialisable (``trace.as_dict()``),
  and renders as an indented tree (``trace.format()``). Always built, even when
  logging is silent.

* Live **structlog** events — emitted as the run happens when logging is enabled
  via :func:`configure_logging`. Each line carries structured fields: the acting
  ``agent``, token usage and remaining budget after every model call, the tool
  being run with its arguments and result, skill loads, and so on. Nesting is
  shown by indentation and tracked with ``contextvars`` so it stays correct
  under concurrent tool calls.
"""

from __future__ import annotations

import contextvars
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import structlog

# The span currently in scope for the running task (copied per asyncio task).
_current_span: contextvars.ContextVar["Span | None"] = contextvars.ContextVar(
    "agentic_current_span", default=None
)

# Whether configure_logging() has been called. Until then, live logging is off
# so importing the framework never spams stdout; the trace tree is still built.
_LOGGING_CONFIGURED = False

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
        return None if self.end is None else self.end - self.start

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
        """Render the trace as an indented text tree (no logging required)."""
        if self.root is None:
            return "(empty trace)"
        lines: list[str] = []
        self._format_span(self.root, lines)
        return "\n".join(lines)

    def _format_span(self, span: Span, lines: list[str]) -> None:
        icon = "✖" if span.status == "error" else _KIND_ICON.get(span.kind, "·")
        dur = f" ({span.duration:.3f}s)" if span.duration is not None else ""
        shown_keys = ("model", "total_tokens", "tools", "stop_reason", "result", "arguments")
        attrs = " ".join(
            f"{k}={span.attributes[k]}" for k in shown_keys if k in span.attributes
        )
        attrs = f" {attrs}" if attrs else ""
        lines.append(f"{'  ' * span.depth}{icon} {span.kind}:{span.name}{dur}{attrs}")
        for event in span.events:
            lines.append(f"{'  ' * (span.depth + 1)}· {event.get('event', '')}")
        for child in span.children:
            self._format_span(child, lines)


class Tracer:
    """Builds the span tree and (optionally) streams structlog events.

    Args:
        trace: the :class:`Trace` to populate (a fresh one if omitted).
        logger: a structlog logger to emit to; defaults to ``get_logger("agentic")``.
        enabled: force live logging on/off. Defaults to whether
            :func:`configure_logging` has been called.
    """

    def __init__(
        self,
        trace: Trace | None = None,
        *,
        logger: Any = None,
        enabled: bool | None = None,
    ) -> None:
        self.trace = trace if trace is not None else Trace()
        self.log = logger if logger is not None else structlog.get_logger("agentic")
        self.enabled = enabled if enabled is not None else (_LOGGING_CONFIGURED or logger is not None)

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
        (parent.children if parent else _RootSlot(self.trace)).append(span)

        token = _current_span.set(span)
        self._log_start(span)
        try:
            yield span
        except BaseException as exc:  # noqa: BLE001 - record then re-raise
            span.status = "error"
            span.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            span.end = time.monotonic()
            _current_span.reset(token)
            self._log_end(span)

    def event(self, event: str, **fields: Any) -> None:
        """Record a point-in-time event on the current span (and log it)."""
        span = _current_span.get()
        record = {"event": event, **fields}
        depth = (span.depth + 1) if span else 0
        if span is not None:
            span.events.append(record)
        if self.enabled:
            self.log.info(event, _depth=depth, **fields)

    # --- live logging -------------------------------------------------------------

    def _log_start(self, span: Span) -> None:
        if not self.enabled:
            return
        if span.kind in ("run", "subagent"):
            self.log.info(f"{span.kind} started", _depth=span.depth, name=span.name)
        elif span.kind == "turn":
            self.log.info("turn", _depth=span.depth, n=span.name)
        # model/tool spans are logged on completion (when results/usage exist).

    def _log_end(self, span: Span) -> None:
        if not self.enabled:
            return
        dur = round(span.duration, 4) if span.duration is not None else None
        a = span.attributes
        if span.kind == "model":
            self.log.info(
                "llm response" if span.status == "ok" else "llm error",
                _depth=span.depth,
                model=a.get("model"),
                tokens_in=a.get("prompt_tokens"),
                tokens_out=a.get("completion_tokens"),
                tokens=a.get("total_tokens"),
                cum_tokens=a.get("cum_tokens"),
                tokens_left=a.get("tokens_left"),
                time_left=a.get("time_left"),
                dur=dur,
            )
        elif span.kind == "tool":
            self.log.info(
                "tool ok" if span.status == "ok" else "tool error",
                _depth=span.depth,
                tool=span.name,
                args=a.get("arguments"),
                result=a.get("result"),
                error=span.error,
                dur=dur,
            )
        elif span.kind in ("run", "subagent"):
            self.log.info(
                f"{span.kind} finished",
                _depth=span.depth,
                name=span.name,
                stop=a.get("stop_reason"),
                tokens=a.get("total_tokens"),
                dur=dur,
            )


class _RootSlot:
    """Tiny adapter so ``span()`` can ``.append`` to either a parent or the root."""

    __slots__ = ("_trace",)

    def __init__(self, trace: Trace) -> None:
        self._trace = trace

    def append(self, span: Span) -> None:
        self._trace.root = span


# --- structlog configuration ------------------------------------------------------


def _make_depth_processor(json: bool) -> Any:
    """A structlog processor that handles the ``_depth`` field and drops None values.

    Console mode indents the event message by depth (human-readable nesting);
    JSON mode keeps ``depth`` as a numeric field (machine-readable nesting).
    """

    def processor(_logger: Any, _method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
        depth = int(event_dict.pop("_depth", 0) or 0)
        if json:
            if depth:
                event_dict["depth"] = depth
        elif depth:
            event_dict["event"] = "  " * depth + str(event_dict.get("event", ""))
        return {k: v for k, v in event_dict.items() if v is not None}

    return processor


def configure_logging(
    level: str | int = "info",
    *,
    json: bool = False,
    extra_processors: list[Any] | None = None,
) -> Any:
    """Turn on live structured logging for runs and return the ``agentic`` logger.

    Call once from your application/script. Without it, runs are silent (the
    trace tree is still captured on each :class:`~agentic.context.RunResult`).

    Args:
        level: minimum level, e.g. ``"info"`` or ``"debug"`` (or a logging int).
        json: emit one JSON object per line (production) instead of the colourised
            console renderer (development).
        extra_processors: structlog processors inserted into the chain *before*
            rendering. Each receives every event (with bound context vars like
            ``agent`` and any you bind yourself) and must return the event dict.
            This is the hook for taps that mirror agent activity elsewhere — e.g.
            into a progress store a UI can poll. See ``examples/08_progress_polling.py``.
    """
    global _LOGGING_CONFIGURED

    lvl = logging.getLevelName(level.upper()) if isinstance(level, str) else level
    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        *(extra_processors or []),
        _make_depth_processor(json),
        structlog.processors.TimeStamper(fmt="%H:%M:%S"),
    ]
    if json:
        processors += [structlog.processors.format_exc_info, structlog.processors.JSONRenderer()]
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(lvl),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )
    _LOGGING_CONFIGURED = True
    return structlog.get_logger("agentic")
