"""End-to-end tests for the agentic framework, run fully offline.

Everything is driven by :class:`agentic.testing.FunctionModel`, so no API key
or network is needed. A FunctionModel "brain" inspects the conversation and
decides each turn: return a string (final answer) or a list of
``(tool_name, kwargs)`` tool calls.
"""

from __future__ import annotations

import enum
from typing import Annotated, Literal

import pytest

from agentic import Agent, Budget, BudgetExceeded, RunContext, Skill, tool
from agentic.schema import build_parameters_schema
from agentic.testing import FunctionModel


class Color(enum.Enum):
    RED = "red"
    BLUE = "blue"


def steps_taken(messages) -> int:
    """Number of assistant turns already in the conversation."""
    return sum(1 for m in messages if m.role == "assistant")


# --------------------------------------------------------------------------- tools


async def test_basic_tool_call_then_answer():
    seen = []

    @tool
    def get_weather(city: str) -> str:
        "Get the current weather for a city."
        seen.append(city)
        return f"sunny in {city}"

    def brain(messages, tools):
        if messages[-1].role == "user":
            return [("get_weather", {"city": "Paris"})]
        return f"It is {messages[-1].content}."

    agent = Agent(FunctionModel(brain), name="weatherbot", tools=[get_weather])
    result = await agent.run("Weather in Paris?")

    assert seen == ["Paris"]
    assert "sunny in Paris" in result.output
    assert result.stop_reason == "completed"
    assert result.turns == 2
    assert result.usage.requests == 2
    assert result.usage.total_tokens > 0


async def test_parallel_tool_calls_run_concurrently():
    @tool
    async def double(n: int) -> int:
        "Double a number."
        return n * 2

    def brain(messages, tools):
        if messages[-1].role == "user":
            return [("double", {"n": 1}), ("double", {"n": 2}), ("double", {"n": 3})]
        return "done"

    agent = Agent(FunctionModel(brain), tools=[double])
    result = await agent.run("go")

    tool_msgs = [m for m in result.messages if m.role == "tool"]
    assert len(tool_msgs) == 3
    assert {m.content for m in tool_msgs} == {"2", "4", "6"}
    assert result.output == "done"


async def test_tool_error_is_recoverable():
    @tool
    def flaky(x: int) -> int:
        "Return x, but fail for negatives."
        if x < 0:
            raise ValueError("x must be non-negative")
        return x

    def brain(messages, tools):
        step = steps_taken(messages)
        if step == 0:
            return [("flaky", {"x": -1})]
        if step == 1:
            assert "non-negative" in messages[-1].content  # model saw the error
            return [("flaky", {"x": 5})]
        return f"answer is {messages[-1].content}"

    agent = Agent(FunctionModel(brain), tools=[flaky])
    result = await agent.run("go")
    assert "answer is 5" in result.output


async def test_run_context_and_deps_injection():
    @tool
    def whoami(ctx: RunContext) -> str:
        "Report the caller and agent name."
        return f"user={ctx.deps['uid']};agent={ctx.agent_name};left={ctx.remaining_tokens}"

    def brain(messages, tools):
        if messages[-1].role == "user":
            return [("whoami", {})]
        return messages[-1].content

    agent = Agent(FunctionModel(brain), name="ctxbot", tools=[whoami])
    result = await agent.run("who am I?", deps={"uid": 42}, budget=Budget(max_tokens=10_000))
    assert "user=42" in result.output
    assert "agent=ctxbot" in result.output


# --------------------------------------------------------------------------- skills


async def test_skill_progressive_disclosure():
    @tool
    def sql(query: str) -> str:
        "Run a SQL query against the warehouse."
        return "rows=7"

    analytics = Skill(
        name="analytics",
        description="Answer data questions against the warehouse.",
        instructions="Use the `sql` tool, then report the row count.",
        tools=[sql],
    )

    def brain(messages, tools):
        names = {t.name for t in tools}
        step = steps_taken(messages)
        if step == 0:
            # Before loading: skill tool hidden, load_skill offered.
            assert "load_skill" in names and "sql" not in names
            return [("load_skill", {"name": "analytics"})]
        if step == 1:
            # After loading: the skill's tool is now available.
            assert "sql" in names
            return [("sql", {"query": "select count(*) from t"})]
        return "There are 7 rows."

    agent = Agent(FunctionModel(brain), name="analyst", skills=[analytics])
    result = await agent.run("how many rows in t?")

    assert "7" in result.output
    assert "load_skill" in result.trace.format()


async def test_auto_loaded_skill_tools_available_immediately():
    @tool
    def cite() -> str:
        "Produce a citation."
        return "[1]"

    research = Skill(
        name="research",
        description="Always cite sources.",
        instructions="Cite everything.",
        tools=[cite],
        auto_load=True,
    )

    def brain(messages, tools):
        names = {t.name for t in tools}
        if messages[-1].role == "user":
            assert "cite" in names  # available from the very first turn
            assert "load_skill" not in names  # nothing left to load on demand
            return [("cite", {})]
        return f"source {messages[-1].content}"

    agent = Agent(FunctionModel(brain), skills=[research])
    result = await agent.run("cite something")
    assert "[1]" in result.output
    # The auto-skill's instructions are baked into the system prompt.
    assert "Cite everything." in agent._compose_system_prompt()


def test_skill_from_markdown_frontmatter():
    skill = Skill.from_markdown(
        """---
name: summarizer
description: Summarize long text crisply.
---
Write a three-bullet summary. Keep each bullet under 15 words.
""",
    )
    assert skill.name == "summarizer"
    assert skill.description == "Summarize long text crisply."
    assert "three-bullet" in skill.instructions
    assert skill.auto_load is False


# ----------------------------------------------------------------------- subagents


async def test_subagent_delegation_and_shared_budget():
    def translator_brain(messages, tools):
        return "Bonjour le monde"

    translator = Agent(
        FunctionModel(translator_brain),
        name="translator",
        instructions="Translate English into French.",
    )

    def boss_brain(messages, tools):
        names = {t.name for t in tools}
        if steps_taken(messages) == 0:
            assert "translator" in names  # exposed as a delegation tool
            return [("translator", {"task": "Translate 'hello world' to French"})]
        return messages[-1].content  # echo the subagent's answer

    boss = Agent(FunctionModel(boss_brain), name="boss", subagents=[translator])
    result = await boss.run("Say hello world in French")

    assert "Bonjour" in result.output
    # boss(2 turns) + translator(1 turn) all counted against the one shared budget
    assert result.usage.requests == 3

    # The trace nests the subagent under the boss's delegation tool call.
    def find(node, kind):
        if node["kind"] == kind:
            return node
        for child in node["children"]:
            hit = find(child, kind)
            if hit:
                return hit
        return None

    tree = result.trace.as_dict()
    assert tree["kind"] == "run" and tree["name"] == "boss"
    assert find(tree, "subagent")["name"] == "translator"


# ------------------------------------------------------------------------ budgets


async def test_token_budget_raises_and_attaches_partial():
    @tool
    def noop() -> str:
        "Consume tokens forever."
        return "x" * 200

    def brain(messages, tools):
        return [("noop", {})]  # never finalises

    agent = Agent(FunctionModel(brain), tools=[noop], max_turns=100)
    with pytest.raises(BudgetExceeded) as info:
        await agent.run("go", budget=Budget(max_tokens=50))

    assert info.value.kind == "tokens"
    assert info.value.limit == 50
    assert info.value.messages  # partial conversation captured


async def test_time_budget_raises():
    import asyncio

    @tool
    async def slow() -> str:
        "A tool that takes too long."
        await asyncio.sleep(0.5)
        return "done"

    def brain(messages, tools):
        return [("slow", {})]

    agent = Agent(FunctionModel(brain), tools=[slow])
    with pytest.raises(BudgetExceeded) as info:
        await agent.run("go", budget=Budget(max_time=0.05))
    assert info.value.kind == "time"


async def test_max_turns_returns_with_reason():
    @tool
    def noop() -> str:
        "No-op."
        return "x"

    def brain(messages, tools):
        return [("noop", {})]  # never finalises

    agent = Agent(FunctionModel(brain), tools=[noop], max_turns=3)
    result = await agent.run("go")
    assert result.stop_reason == "max_turns"
    assert result.turns == 3


# -------------------------------------------------------------------- schema gen


def test_schema_generation_covers_common_types():
    def fn(
        query: str,
        limit: int = 10,
        ratio: float = 1.0,
        flag: bool = False,
        tags: list[str] | None = None,
        mode: Literal["fast", "slow"] = "fast",
        color: Color = Color.RED,
        note: Annotated[str, "a free-text note"] = "",
    ) -> str:
        """Do a thing.

        Args:
            query: what to search for.
            limit: maximum number of results.
        """
        return query

    schema = build_parameters_schema(fn)
    props = schema["properties"]

    assert schema["required"] == ["query"]
    assert props["query"] == {"type": "string", "description": "what to search for."}
    assert props["limit"]["type"] == "integer"
    assert props["ratio"]["type"] == "number"
    assert props["flag"]["type"] == "boolean"
    assert props["tags"] == {"type": "array", "items": {"type": "string"}}
    assert props["mode"]["enum"] == ["fast", "slow"]
    assert props["color"]["enum"] == ["red", "blue"]
    assert props["note"]["description"] == "a free-text note"
    assert schema["additionalProperties"] is False


async def test_structured_output_pydantic_model():
    from pydantic import BaseModel

    class Sentiment(BaseModel):
        label: str
        score: float

    def brain(messages, tools):
        # The agent should expose a `final_result` tool whose schema is Sentiment.
        names = {t.name for t in tools}
        assert "final_result" in names
        return [("final_result", {"label": "positive", "score": 0.9})]

    agent = Agent(FunctionModel(brain), name="classifier", output_type=Sentiment)
    result = await agent.run("I love it!")

    assert isinstance(result.output, Sentiment)
    assert result.output.label == "positive"
    assert result.output.score == 0.9


async def test_structured_output_scalar_is_wrapped():
    def brain(messages, tools):
        return [("final_result", {"value": [1, 2, 3]})]

    agent = Agent(FunctionModel(brain), output_type=list[int])
    result = await agent.run("give me three numbers")
    assert result.output == [1, 2, 3]


async def test_structured_output_revalidates_on_bad_data():
    from pydantic import BaseModel

    class Point(BaseModel):
        x: int
        y: int

    attempts = {"n": 0}

    def brain(messages, tools):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return [("final_result", {"x": "not-an-int", "y": 2})]  # invalid -> recoverable
        # The model should have seen the validation error in the tool result.
        assert "did not match" in messages[-1].content
        return [("final_result", {"x": 1, "y": 2})]

    agent = Agent(FunctionModel(brain), output_type=Point)
    result = await agent.run("make a point")
    assert isinstance(result.output, Point) and result.output.x == 1
    assert result.turns == 2  # one failed attempt, one success


async def test_structured_output_reprompts_on_prose():
    from pydantic import BaseModel

    class Answer(BaseModel):
        text: str

    state = {"n": 0}

    def brain(messages, tools):
        state["n"] += 1
        if state["n"] == 1:
            return "here is my answer in prose"  # ignores the tool
        return [("final_result", {"text": "done"})]

    agent = Agent(FunctionModel(brain), output_type=Answer)
    result = await agent.run("answer me")
    assert isinstance(result.output, Answer) and result.output.text == "done"
    # A nudge message was injected after the prose turn.
    assert any("final_result" in (m.content or "") for m in result.messages if m.role == "user")


def test_run_sync_wrapper():
    @tool
    def echo(text: str) -> str:
        "Echo the text."
        return text

    def brain(messages, tools):
        if messages[-1].role == "user":
            return [("echo", {"text": "hi"})]
        return messages[-1].content

    agent = Agent(FunctionModel(brain), tools=[echo])
    result = agent.run_sync("say hi")  # blocking convenience wrapper
    assert result.output == "hi"
