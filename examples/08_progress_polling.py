"""08 · Showing live progress to a polling (REST) UI

You can't use websockets, but you want the UI to show — in detail and in plain
language — what the agent is doing *right now*. The pattern:

1. Keep a per-run ``ProgressStore`` (in-memory here; Redis in production).
2. Plug a **structlog processor** into the framework's event stream that mirrors
   every step into the store, translated into human-friendly text:
       configure_logging(extra_processors=[make_progress_processor(store)])
3. Run the agent in the background; the UI polls ``GET /runs/{id}/status`` and
   renders ``store.snapshot(run_id)``.

Two layers of progress text:
* **Automatic** — framework events (started, consulting a subagent, wrapping up)
  are translated for free.
* **Custom** — inside a tool, call ``ctx.tracer.event("Looking up your order…")``
  for domain-specific messages. They flow through the same processor and appear
  verbatim. Use these for the messages that matter to users.

This file *simulates* the web server: it runs the agent as a background task and
a loop that polls the store, printing each new status the way a browser would.
The FastAPI wiring is sketched at the bottom.

Run it:
    uv run python examples/08_progress_polling.py
"""

from __future__ import annotations

import asyncio

from agentic import Agent, Budget, RunContext, configure_logging, tool

from _shared import demo_model, steps_taken
from _webio import ProgressStore, make_progress_processor, progress_run


# --- tools: note the human-friendly `ctx.tracer.event(...)` progress messages -----
@tool
async def get_order_status(ctx: RunContext, order_id: str) -> dict:
    """Look up an order's status."""
    ctx.tracer.event(f"Looking up order {order_id}…")  # <- shown to the user
    await asyncio.sleep(0.4)  # simulate real I/O latency so polling sees the step
    return {"id": order_id, "item": "Wireless mouse", "total": 79.99, "eligible": True}


@tool
async def issue_refund(ctx: RunContext, order_id: str, amount: float) -> str:
    """Issue a refund for an order."""
    ctx.tracer.event(f"Issuing a ${amount:.2f} refund for {order_id}…")
    await asyncio.sleep(0.5)
    return "refund-confirmed-7781"


def brain(messages, tools):
    step = steps_taken(messages)
    if step == 0:
        return [("get_order_status", {"order_id": "A-1042"})]
    if step == 1:
        return [("issue_refund", {"order_id": "A-1042", "amount": 79.99})]
    return "All done — your $79.99 refund is confirmed (ref refund-confirmed-7781)."


agent = Agent(
    demo_model(brain),
    name="support_agent",
    instructions="You resolve refund requests.",
    tools=[get_order_status, issue_refund],
)


async def main() -> None:
    store = ProgressStore()
    # The single line that wires agent activity into the pollable store:
    configure_logging(extra_processors=[make_progress_processor(store)])

    run_id = "refund-demo-001"

    async def run_agent() -> None:
        # `progress_run` binds run_id so the processor can attribute events to it.
        async with progress_run(store, run_id):
            try:
                result = await agent.run(
                    "I'd like a refund for order A-1042.", budget=Budget(max_time=30)
                )
                store.finish(run_id, result.output)
            except Exception as exc:  # noqa: BLE001
                store.fail(run_id, str(exc))

    task = asyncio.create_task(run_agent())

    # ---- This loop is what a browser does: poll the status endpoint repeatedly ----
    print(f"(UI polling GET /runs/{run_id}/status every 150ms)\n")
    last = None
    while not task.done():
        await asyncio.sleep(0.15)
        snap = store.snapshot(run_id)
        if snap["current"] != last:  # only print when the displayed status changes
            last = snap["current"]
            print(f"  [{snap['state']:<8} | {snap['tokens']:>4} tok]  {snap['current']}")
    await task

    snap = store.snapshot(run_id)
    print(f"\nFinal status payload the UI would receive:\n  {snap['state']}: {snap['result']}")


if __name__ == "__main__":
    asyncio.run(main())


# ──────────────────────────────────────────────────────────────────────────────
# How this maps to a real REST API (FastAPI). The store + processor are the same;
# only the transport changes.
#
#     store = ProgressStore()
#     configure_logging(extra_processors=[make_progress_processor(store)])
#     app = FastAPI()
#
#     @app.post("/runs")
#     async def start_run(body: StartBody):
#         run_id = uuid4().hex
#         async def worker():
#             async with progress_run(store, run_id):
#                 result = await agent.run(body.message, budget=Budget(max_tokens=50_000))
#                 store.finish(run_id, result.output)
#         asyncio.create_task(worker())          # fire-and-forget; return immediately
#         return {"run_id": run_id}
#
#     @app.get("/runs/{run_id}/status")          # <- the UI polls this, e.g. every 1s
#     async def status(run_id: str):
#         return store.snapshot(run_id)
# ──────────────────────────────────────────────────────────────────────────────
