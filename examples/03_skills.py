"""03 · Skills — packaged instructions (+ tools), revealed on demand

A *skill* is a reusable bundle of know-how: a chunk of instructions plus any
tools that go with it, under a name and one-line description. This is distinct
from a tool (a single function call). Think "a packaged capability".

Two activation modes
---------------------
* ``auto_load=True`` — the skill's instructions are baked into the system prompt
  from the start and its tools are always available.
* default (*progressive disclosure*) — only the skill's name + description are
  advertised. When the model judges it relevant, it calls the built-in
  ``load_skill`` tool; the full instructions are then returned to it and the
  skill's tools switch on for the rest of the run. This keeps the context small
  until a capability is actually needed.

You can define skills in Python (with tools) or load instruction-only skills
from markdown / a ``SKILL.md`` file via ``Skill.from_markdown`` / ``from_directory``.

Run it:
    uv run python examples/03_skills.py
"""

from __future__ import annotations

import asyncio

from agentic import Agent, Skill, configure_logging, tool

from _shared import banner, demo_model, steps_taken


# A tool that belongs to the SQL skill.
@tool
def run_sql(query: str) -> str:
    """Execute a read-only SQL query against the analytics warehouse."""
    return "월 | revenue\n2024-01 | 120000\n2024-02 | 138000"  # pretend rows


# A skill = instructions + the tools that implement it. Loaded on demand.
sql_skill = Skill(
    name="sql_analytics",
    description="Answer questions about business metrics using the data warehouse.",
    instructions=(
        "When asked about metrics, write a single SQL query, run it with `run_sql`, "
        "then summarise the result in one sentence with the key number."
    ),
    tools=[run_sql],
)

# An instruction-only skill, loaded from markdown (note the YAML-ish frontmatter).
tone_skill = Skill.from_markdown(
    """---
name: exec_tone
description: Communicate in a crisp, executive-summary tone.
---
Lead with the answer. One short paragraph, no preamble, no hedging.
""",
    auto_load=True,  # always on -> its instructions are part of the system prompt
)


def brain(messages, tools):
    names = {t.name for t in tools}
    step = steps_taken(messages)
    if step == 0:
        # The sql tool isn't available yet — only `load_skill` is. So load it.
        assert "run_sql" not in names and "load_skill" in names
        return [("load_skill", {"name": "sql_analytics"})]
    if step == 1:
        # After loading, the skill's tool is available.
        assert "run_sql" in names
        return [("run_sql", {"query": "select month, revenue from monthly"})]
    return "Revenue grew from 120,000 in Jan to 138,000 in Feb (+15%)."


agent = Agent(
    demo_model(brain),
    name="analyst",
    instructions="You are a data analyst.",
    skills=[sql_skill, tone_skill],
)


async def main() -> None:
    configure_logging()  # watch the `load_skill` call and the skill's tool switch on

    result = await agent.run("How did revenue trend over the last two months?")

    banner("System prompt (note: exec_tone is baked in; sql_analytics is only advertised)")
    print(agent._compose_system_prompt())

    banner("Output")
    print(result.output)


if __name__ == "__main__":
    asyncio.run(main())
