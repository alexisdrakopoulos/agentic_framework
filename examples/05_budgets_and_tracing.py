"""05 · Budgets, logging, and tracing — production safety & observability

Two things you need before putting an agent in production: a hard cap on what a
run may consume, and a clear record of what it did.

Budgets
-------
``Budget(max_tokens=..., max_time=..., max_turns=...)`` is enforced across the
whole run tree (parent + subagents). Limits are checked at call boundaries; when
one trips, the run raises ``BudgetExceeded`` (with ``.kind``, ``.limit``,
``.used``, and the partial ``.messages`` attached). ``max_time`` also caps each
in-flight model/tool call so a single hang can't blow the wall-clock.

(Reaching ``max_turns`` is treated as a soft stop, not an error: the result
comes back with ``stop_reason="max_turns"`` so you can inspect partial work.)

Logging & tracing
-----------------
* ``configure_logging()`` turns on live, structured **structlog** output. Every
  model call logs token usage and remaining budget; every turn and tool call is
  logged with the acting agent. Pass ``json=True`` for one-JSON-object-per-line
  output suited to log pipelines.
* Independently, every ``RunResult`` carries a ``trace`` you can render
  (``trace.format()``) or serialise (``trace.as_dict()``) — even with logging off.

Run it:
    uv run python examples/05_budgets_and_tracing.py
"""

from __future__ import annotations

import asyncio
import json

from agentic import Agent, Budget, BudgetExceeded, configure_logging, tool

from _shared import banner, demo_model


@tool
def expensive_step(note: str) -> str:
    """A tool that always asks for more work (to demonstrate hitting a budget)."""
    return "partial result; more analysis still needed " + "data " * 20


# This brain never finalises — it keeps calling the tool. Good for showing a cap.
def loop_brain(messages, tools):
    return [("expensive_step", {"note": "keep going"})]


agent = Agent(
    demo_model(loop_brain),
    name="runaway",
    instructions="You analyse data thoroughly.",
    tools=[expensive_step],
    max_turns=100,
)


async def main() -> None:
    configure_logging()  # try configure_logging(json=True) for machine-readable logs

    banner("Enforcing a token budget (the run is stopped before it overspends)")
    try:
        await agent.run("Analyse everything in exhaustive detail.", budget=Budget(max_tokens=1500))
    except BudgetExceeded as exc:
        print(f"\nStopped: {exc}")
        print(f"  kind={exc.kind}  limit={exc.limit}  used={exc.used}")
        print(f"  partial conversation captured: {len(exc.messages)} messages")

    banner("The trace is JSON-serialisable for offline inspection / dashboards")
    # Re-run briefly with a tiny turn cap to get a clean tree to print.
    result = await agent.run("Quick look only.", max_turns=2)
    print(f"stop_reason={result.stop_reason}")
    print(json.dumps(result.trace.as_dict(), indent=2)[:600] + "\n...")


if __name__ == "__main__":
    asyncio.run(main())
