"""07 · Typed structured output

So far ``result.output`` has been free text. Often you want a *validated object*
instead — to store it, branch on it, or hand it to another system.

Key ideas
---------
* Pass ``output_type=`` to an agent: a pydantic model, a dataclass, a
  ``TypedDict``, or even a plain type like ``list[str]``. ``result.output`` is
  then a validated instance of that type.
* Under the hood the agent gains a synthetic ``final_result`` tool whose schema
  is your type. The model "answers" by calling it; the arguments are validated.
  This works with any tool-calling model (and offline here).
* If the model returns data that doesn't validate, the error is fed back and it
  gets to try again — you don't get a half-valid object.

Tools and structured output compose: the agent can call regular tools to gather
information, then call ``final_result`` to return the typed answer.

Run it:
    uv run python examples/07_structured_output.py
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field

from agentic import Agent, tool

from _shared import banner, demo_model, steps_taken


# The shape we want back — a validated object, not a string.
class Ticket(BaseModel):
    title: str
    severity: int = Field(ge=1, le=5, description="1 = trivial, 5 = critical")
    component: str
    needs_oncall: bool


@tool
def get_service_owner(component: str) -> str:
    """Return the team that owns a service component."""
    return {"checkout": "payments-team", "search": "discovery-team"}.get(component, "platform-team")


def brain(messages, tools):
    step = steps_taken(messages)
    if step == 0:
        # First gather info with a normal tool...
        return [("get_service_owner", {"component": "checkout"})]
    # ...then emit the structured result via the auto-generated final_result tool.
    return [
        (
            "final_result",
            {
                "title": "Checkout returns 500 on payment",
                "severity": 5,
                "component": "checkout",
                "needs_oncall": True,
            },
        )
    ]


agent = Agent(
    demo_model(brain),
    name="triage",
    instructions="You triage incident reports into structured tickets.",
    tools=[get_service_owner],
    output_type=Ticket,  # <- the whole point of this example
)


async def main() -> None:
    result = await agent.run(
        "Users report checkout fails with a 500 during payment. Triage this."
    )

    banner("result.output is a validated Ticket, not text")
    ticket = result.output
    print(repr(ticket))
    print(f"\nseverity is an int: {ticket.severity + 0}  | needs_oncall is a bool: {ticket.needs_oncall}")

    banner("Auto-generated schema for the `final_result` tool")
    import json

    print(json.dumps(Ticket.model_json_schema(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
