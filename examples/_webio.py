"""Web glue for examples 08 & 09 — the bridge between an agent run and a
REST/polling UI.

In a real service this is the code that would live in your web app. Three pieces:

* ``ProgressStore`` — per-run state your status endpoint returns as JSON. In-memory
  here; back it with Redis in production so your poll endpoint and worker(s) share it.
* ``make_progress_processor`` — a **structlog processor** that mirrors every agent
  event into the store, translated into human-friendly text. You plug it in with
  ``configure_logging(extra_processors=[make_progress_processor(store)])``. This is
  the "the logger updates the thing the UI polls" idea.
* ``HumanChannel`` — lets an async tool (``ask_user``) block until a human answers
  via a separate REST call, by awaiting an ``asyncio.Future`` that the answer
  endpoint resolves.

Everything keys off a ``run_id`` bound into structlog's context vars (see
``progress_run``), so events from concurrent runs never get mixed up.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
from dataclasses import dataclass, field
from typing import Any

import structlog


# --------------------------------------------------------------------- progress store
@dataclass
class RunProgress:
    run_id: str
    state: str = "running"  # running | awaiting_input | done | error
    current: str = "Starting…"  # latest human-friendly status line
    steps: list[dict] = field(default_factory=list)
    tokens: int = 0
    result: Any = None
    error: str | None = None
    question: dict | None = None


class ProgressStore:
    """Per-run progress. Single-process/asyncio-safe; swap for Redis in production."""

    def __init__(self) -> None:
        self._runs: dict[str, RunProgress] = {}

    def create(self, run_id: str) -> RunProgress:
        self._runs[run_id] = RunProgress(run_id)
        return self._runs[run_id]

    def get(self, run_id: str) -> RunProgress | None:
        return self._runs.get(run_id)

    def step(self, run_id: str, message: str, **detail: Any) -> None:
        p = self._runs.get(run_id)
        if p is None:
            return
        p.current = message
        p.steps.append({"message": message, **detail})

    def set_tokens(self, run_id: str, n: int) -> None:
        p = self._runs.get(run_id)
        if p:
            p.tokens = n

    def set_question(self, run_id: str, question: dict) -> None:
        p = self._runs.get(run_id)
        if p:
            p.question = question
            p.state = "awaiting_input"
            p.current = f"Waiting for your input: {question['text']}"

    def clear_question(self, run_id: str) -> None:
        p = self._runs.get(run_id)
        if p:
            p.question = None
            p.state = "running"

    def finish(self, run_id: str, result: Any) -> None:
        p = self._runs.get(run_id)
        if p:
            p.result = result
            p.state = "done"
            p.current = "Done."

    def fail(self, run_id: str, error: str) -> None:
        p = self._runs.get(run_id)
        if p:
            p.error = error
            p.state = "error"
            p.current = f"Error: {error}"

    def snapshot(self, run_id: str) -> dict:
        """Exactly what ``GET /runs/{run_id}/status`` would return as JSON."""
        p = self._runs.get(run_id)
        if p is None:
            return {"state": "unknown"}
        return {
            "run_id": p.run_id,
            "state": p.state,
            "current": p.current,
            "tokens": p.tokens,
            "steps": p.steps[-20:],
            "question": p.question,
            "result": p.result if p.state == "done" else None,
            "error": p.error,
        }


@contextlib.asynccontextmanager
async def progress_run(store: ProgressStore, run_id: str):
    """Create a run's progress entry and bind ``run_id`` for the duration.

    Binding via ``structlog.contextvars`` is what lets the processor below
    attribute each log event to the right run (it's copied per asyncio task).
    """
    store.create(run_id)
    tokens = structlog.contextvars.bind_contextvars(run_id=run_id)
    try:
        yield
    finally:
        structlog.contextvars.reset_contextvars(**tokens)


# ----------------------------------------------------- humanising structlog processor
# Map a few framework events to friendly phrasing. Anything not listed here and not
# a known low-level event is treated as already human-friendly — that's how a tool's
# own `ctx.tracer.event("Looking up your order…")` shows up verbatim in the UI.
_FRIENDLY = {
    "run started": lambda e: "Getting started…",
    "subagent started": lambda e: f"Consulting the {str(e.get('agent', 'specialist')).replace('_', ' ')}…",
    "final answer": lambda e: "Wrapping up…",
    "final result": lambda e: "Wrapping up…",
    "awaiting structured result": lambda e: "Formatting the result…",
}
# Low-level events we deliberately do NOT surface to end users (too granular).
_INTERNAL = {
    "turn",
    "llm response",
    "calling tools",
    "tool ok",
    "tool error",
    "run finished",
    "subagent finished",
}


def make_progress_processor(store: ProgressStore):
    """A structlog processor that mirrors agent activity into ``store``.

    It never alters the event (normal console/JSON logging still happens) and
    no-ops for events without a bound ``run_id``.
    """

    def processor(_logger: Any, _method: str, event_dict: dict) -> dict:
        run_id = event_dict.get("run_id")
        if run_id and store.get(run_id) is not None:
            if "cum_tokens" in event_dict:
                store.set_tokens(run_id, event_dict["cum_tokens"])
            event = str(event_dict.get("event", ""))
            if event in _FRIENDLY:
                store.step(run_id, _FRIENDLY[event](event_dict), kind="framework")
            elif event not in _INTERNAL:
                # A custom, already-human-friendly message from a tool's tracer.event(...)
                store.step(run_id, event, kind="custom")
        return event_dict

    return processor


# --------------------------------------------------------------- human-in-the-loop bridge
class HumanChannel:
    """Bridges an async tool to a REST 'answer' endpoint.

    ``ask`` registers a pending question (surfacing it to the UI via the store)
    and awaits a Future; the answer endpoint calls ``answer`` to resolve it.

    In a multi-worker deployment, replace the in-process Future with a shared
    inbox (Redis pub/sub, or a DB row the worker polls) so the answer can reach
    whichever worker is running the agent.
    """

    def __init__(self, store: ProgressStore) -> None:
        self._store = store
        self._pending: dict[str, asyncio.Future] = {}
        self._ids = itertools.count(1)

    async def ask(
        self, run_id: str, question: str, *, options: list[str] | None = None, timeout: float = 300.0
    ) -> str:
        qid = f"{run_id}:q{next(self._ids)}"
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[qid] = fut
        self._store.set_question(run_id, {"id": qid, "text": question, "options": options})
        try:
            return await asyncio.wait_for(fut, timeout)
        finally:
            self._pending.pop(qid, None)
            self._store.clear_question(run_id)

    def pending_ids(self) -> list[str]:
        return list(self._pending)

    def answer(self, question_id: str, text: str) -> bool:
        """Resolve a pending question. Returns False if it's unknown/already answered."""
        fut = self._pending.get(question_id)
        if fut is not None and not fut.done():
            fut.set_result(text)
            return True
        return False
