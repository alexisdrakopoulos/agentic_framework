"""09 · Human-in-the-loop over a REST-only UI

Sometimes the agent needs to *ask the user something* mid-run ("which account?",
"are you sure?") before it can continue. With only REST + polling (no
websockets), the flow is:

1. The agent calls an ``ask_user`` tool. Because the tool is ``async``, it simply
   ``await``s an answer — without blocking the event loop, so the web server keeps
   serving other requests.
2. ``ask_user`` records the question in the ``ProgressStore`` and sets the run's
   state to ``awaiting_input``. The polling UI sees the question and renders a form.
3. The user submits via ``POST /runs/{id}/answer``; that endpoint resolves the
   Future the tool is awaiting; the tool returns the answer to the model and the
   run continues.

The bridge between "an awaiting async tool" and "a separate REST request" is the
``HumanChannel`` (see ``_webio.py``). It's passed to the tool through **deps** —
a clean, concrete example of what dependency injection is for: the tool needs
access to the channel and the current ``run_id``, which are per-request, so they
arrive via ``ctx.deps`` rather than as globals.

This file simulates the UI: it polls, and when a question appears it "answers"
after a short pause. FastAPI wiring is sketched at the bottom.

Run it:
    uv run python examples/09_human_in_the_loop.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from agentic import Agent, RunContext, ToolError, configure_logging, tool

from _shared import demo_model, steps_taken
from _webio import HumanChannel, ProgressStore, make_progress_processor, progress_run


# --- the per-run dependencies the tools need (see the README "Dependencies" note) --
@dataclass
class WebDeps:
    run_id: str
    store: ProgressStore
    human: HumanChannel


# --- the human-in-the-loop tool ---------------------------------------------------
@tool
async def ask_user(ctx: RunContext, question: str) -> str:
    """Ask the human a clarifying question and wait for their answer.

    Use this when the request is ambiguous and you cannot proceed safely without
    more information.

    Args:
        question: a single, specific question for the user.
    """
    # `ctx.deps` carries the per-run channel + run_id (dependency injection).
    try:
        return await ctx.deps.human.ask(ctx.deps.run_id, question, timeout=120)
    except asyncio.TimeoutError:
        # Recoverable: tell the model the user is unavailable so it can fall back.
        raise ToolError("the user did not respond in time; proceed with sensible defaults")


def brain(messages, tools):
    step = steps_taken(messages)
    if step == 0:
        # The request is ambiguous → ask the human instead of guessing.
        return [("ask_user", {"question": "How many people, and what date & time?"})]
    answer = messages[-1].content  # the human's reply comes back as the tool result
    return f"Great — I've booked a table. Details: {answer}. A confirmation is on its way."


agent = Agent(
    demo_model(brain),
    name="booking_agent",
    instructions=(
        "You book restaurant tables. If the party size or time is missing, ask the user "
        "with ask_user before booking. Never invent details."
    ),
    tools=[ask_user],
)


async def main() -> None:
    store = ProgressStore()
    human = HumanChannel(store)
    configure_logging(extra_processors=[make_progress_processor(store)])

    run_id = "booking-demo-001"
    deps = WebDeps(run_id=run_id, store=store, human=human)

    async def run_agent() -> None:
        async with progress_run(store, run_id):
            try:
                result = await agent.run("Book me a table for dinner.", deps=deps)
                store.finish(run_id, result.output)
            except Exception as exc:  # noqa: BLE001
                store.fail(run_id, str(exc))

    task = asyncio.create_task(run_agent())

    # ---- Simulated UI: poll; when a question appears, gather input and POST it ----
    print(f"(UI polling GET /runs/{run_id}/status)\n")
    answered: set[str] = set()
    last = None
    while not task.done():
        await asyncio.sleep(0.1)
        snap = store.snapshot(run_id)
        if snap["current"] != last:
            last = snap["current"]
            print(f"  [{snap['state']}]  {snap['current']}")

        if snap["state"] == "awaiting_input":
            q = snap["question"]
            if q["id"] not in answered:
                answered.add(q["id"])
                print(f"      ↳ UI renders a form for: {q['text']!r}")
                await asyncio.sleep(0.6)  # the human thinks/types
                ok = human.answer(q["id"], "2 people, tomorrow at 7pm")  # POST /answer
                print(f"      ↳ user submitted answer (POST /answer accepted={ok})")
    await task

    print(f"\nFinal: {store.snapshot(run_id)['result']}")


if __name__ == "__main__":
    asyncio.run(main())


# ──────────────────────────────────────────────────────────────────────────────
# How this maps to a real REST API (FastAPI):
#
#     store = ProgressStore()
#     human = HumanChannel(store)
#     configure_logging(extra_processors=[make_progress_processor(store)])
#     app = FastAPI()
#
#     @app.post("/runs")
#     async def start_run(body: StartBody):
#         run_id = uuid4().hex
#         deps = WebDeps(run_id=run_id, store=store, human=human)
#         async def worker():
#             async with progress_run(store, run_id):
#                 result = await agent.run(body.message, deps=deps)
#                 store.finish(run_id, result.output)
#         asyncio.create_task(worker())
#         return {"run_id": run_id}
#
#     @app.get("/runs/{run_id}/status")          # UI polls this; when state ==
#     async def status(run_id: str):             # "awaiting_input", render question
#         return store.snapshot(run_id)
#
#     @app.post("/runs/{run_id}/answer")         # UI posts the user's reply here
#     async def answer(run_id: str, body: AnswerBody):
#         accepted = human.answer(body.question_id, body.text)
#         return {"accepted": accepted}
#
# Notes for production:
#  * Don't set a tight Budget(max_time=...) on human-in-the-loop runs, or the wait
#    for a human will trip the time budget. Use the per-question `timeout` instead.
#  * With multiple workers, replace HumanChannel's in-process Future with a shared
#    inbox (Redis pub/sub or a DB row the worker polls) so the answer reaches the
#    worker actually running the agent.
# ──────────────────────────────────────────────────────────────────────────────
