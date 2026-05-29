"""06 · End to end with a real OpenAI model

Everything from the guide, wired to an actual LLM: a coordinator agent with a
tool and a skill, delegating to a specialist subagent, all under a shared budget
with live structured logging.

Unlike examples 01–05 (which run offline), this one needs a real model. Set your
key and run:

    export OPENAI_API_KEY=sk-...
    uv run python examples/06_openai_end_to_end.py

You can point at any OpenAI-compatible endpoint with
``OpenAIModel("model-id", base_url="...")``.
"""

from __future__ import annotations

import asyncio
import os

from agentic import Agent, Budget, BudgetExceeded, OpenAIModel, RunContext, Skill, configure_logging, tool


# --- a tool --------------------------------------------------------------------
@tool
def get_weather(city: str) -> str:
    """Get a short current-weather report for a city."""
    # Canned data keeps the example deterministic and free of extra API calls.
    table = {
        "reykjavik": "2°C, windy, light snow",
        "lisbon": "21°C, sunny",
        "tokyo": "16°C, clear",
    }
    return table.get(city.strip().lower(), "15°C, partly cloudy")


# --- a specialist subagent with its own tool -----------------------------------
@tool
def convert_currency(amount: float, rate: float) -> float:
    """Convert an amount using an explicit exchange rate (to = amount * rate)."""
    return round(amount * rate, 2)


fx_specialist = Agent(
    "gpt-4o-mini",
    name="fx_specialist",
    instructions=(
        "You convert currencies. If given an amount and a rate, use convert_currency. "
        "Answer with just the converted figure and currency."
    ),
    tools=[convert_currency],
)


# --- a skill: packing advice, loaded on demand ---------------------------------
packing_skill = Skill(
    name="packing_advice",
    description="Give a short packing list tailored to weather and trip length.",
    instructions=(
        "Produce a 3–5 item packing list as bullet points, chosen for the destination's "
        "weather. Keep it tight; no preamble."
    ),
)


# --- the coordinator -----------------------------------------------------------
coordinator = Agent(
    OpenAIModel("gpt-4o-mini", temperature=0.2),
    name="trip_assistant",
    instructions=(
        "You are a travel assistant. Check the weather with get_weather. If the user "
        "needs a packing list, load the packing_advice skill. If currency conversion is "
        "needed, delegate to the fx_specialist subagent. Be concise."
    ),
    tools=[get_weather],
    skills=[packing_skill],
    subagents=[fx_specialist],
)


async def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run this example (examples 01–05 run offline).")
        return

    configure_logging()  # live token-usage + budget + per-turn activity logs

    budget = Budget(max_tokens=20_000, max_time=60.0, max_turns=10)
    try:
        result = await coordinator.run(
            "I'm going to Reykjavik for 3 days. What should I pack, and what is "
            "200 USD in EUR at a rate of 0.92?",
            budget=budget,
        )
    except BudgetExceeded as exc:
        print(f"\nRun stopped by budget ({exc.kind}): {exc}")
        return

    print("\n=== FINAL ANSWER ===")
    print(result.output)
    print("\n=== USAGE ===")
    print(result.usage.as_dict(), f"in {result.elapsed:.2f}s across {result.turns} turns")


if __name__ == "__main__":
    asyncio.run(main())
