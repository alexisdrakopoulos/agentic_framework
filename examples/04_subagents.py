"""04 · Subagents — delegating to specialists

Sometimes one agent shouldn't do everything. A *subagent* is just another
``Agent`` — with its own model, instructions, and tools — that a parent can hand
a self-contained task to.

Key ideas
---------
* Pass ``subagents=[...]`` to an agent. Each subagent is exposed to the parent
  as a delegation tool named after it; calling that tool runs the subagent and
  returns its final answer.
* Subagents are different from skills: a skill adds instructions/tools to the
  *same* agent; a subagent is a *separate* agent with its own context window and
  toolset (good for focus, isolation, or a different model per role).
* The whole run tree shares ONE budget and ONE trace. Token/time limits apply
  across parent + all subagents, and the trace nests subagent spans under the
  delegation call so you can see the full story.

Run it:
    uv run python examples/04_subagents.py
"""

from __future__ import annotations

import asyncio

from agentic import Agent, configure_logging, tool

from _shared import banner, demo_model, steps_taken


# --- a research specialist (its own tool) -----------------------------------------
@tool
def web_search(query: str) -> str:
    """Search the web and return the top result snippet."""
    return "Mistletoe is toxic to cats and dogs; ingestion can cause GI upset."


def researcher_brain(messages, tools):
    if steps_taken(messages) == 0:
        return [("web_search", {"query": "is mistletoe toxic to pets"})]
    return "Mistletoe is toxic to pets; it can cause gastrointestinal upset if eaten."


researcher = Agent(
    demo_model(researcher_brain, name="researcher-llm"),
    name="researcher",
    instructions="You find and verify facts using web_search, then answer succinctly.",
    tools=[web_search],
)


# --- a writer specialist ----------------------------------------------------------
def writer_brain(messages, tools):
    return "🎄 Keep mistletoe out of paws' reach — it can upset your pet's tummy!"


writer = Agent(
    demo_model(writer_brain, name="writer-llm"),
    name="writer",
    instructions="You turn facts into a single friendly, punchy sentence.",
)


# --- the orchestrator that delegates ----------------------------------------------
def editor_brain(messages, tools):
    names = {t.name for t in tools}
    step = steps_taken(messages)
    if step == 0:
        assert {"researcher", "writer"} <= names  # subagents appear as tools
        return [("researcher", {"task": "Is mistletoe dangerous to pets? Verify."})]
    if step == 1:
        fact = messages[-1].content
        return [("writer", {"task": f"Write one friendly safety tip based on: {fact}"})]
    return messages[-1].content  # publish the writer's line


editor = Agent(
    demo_model(editor_brain, name="editor-llm"),
    name="editor",
    instructions="You coordinate a researcher and a writer to produce a safety tip.",
    subagents=[researcher, writer],
)


async def main() -> None:
    configure_logging()  # note how the `agent=` field switches as work is delegated

    result = await editor.run("Give me a holiday pet-safety tip about mistletoe.")

    banner("Output")
    print(result.output)

    banner("Run summary (one shared budget across all three agents)")
    print(f"requests={result.usage.requests}  total_tokens={result.usage.total_tokens}")

    banner("Trace tree (subagents nested under the delegation calls)")
    print(result.trace.format())


if __name__ == "__main__":
    asyncio.run(main())
