"""01 · Hello, agent

The smallest useful agent: an instruction (system prompt), one tool, and a run.

Key ideas
---------
* An ``Agent`` wraps a model, a persona (``instructions``), and some ``tools``.
* A *tool* is just a typed Python function. The framework reads its type hints
  and docstring to build the JSON schema the model needs — you write no schema.
* ``agent.run(...)`` is async and returns a ``RunResult`` whose ``.output`` is
  the final text. (``run_sync`` is a blocking convenience wrapper.)

Run it:
    uv run python examples/01_hello_agent.py
"""

from __future__ import annotations

import asyncio

from agentic import Agent, tool

from _shared import banner, demo_model


# A tool is a normal function. The one-line docstring becomes its description;
# the parameter (with its type hint) becomes the schema the model fills in.
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    fake = {"Paris": "18°C and sunny", "Oslo": "3°C and snowing"}
    return fake.get(city, "weather unknown")


# In real use you'd write: Agent("gpt-4o-mini", ...) or Agent(OpenAIModel(...), ...).
# demo_model() falls back to an offline scripted brain so this runs without a key.
def brain(messages, tools):
    # First turn: ask for the weather tool. Second turn: answer using the result.
    if messages[-1].role == "user":
        return [("get_weather", {"city": "Paris"})]
    return f"It's currently {messages[-1].content} in Paris."


agent = Agent(
    demo_model(brain),
    name="assistant",
    instructions="You are a concise, friendly assistant.",
    tools=[get_weather],
)


async def main() -> None:
    result = await agent.run("What's the weather like in Paris?")

    banner("Output")
    print(result.output)

    banner("What happened")
    print(f"turns={result.turns}  stop={result.stop_reason}  tokens={result.usage.total_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
