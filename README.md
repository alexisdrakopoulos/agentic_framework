# agentic

A small, clean **async** agentic framework for Python. It gives you the handful
of primitives you actually need to build LLM agents in production — **agents,
tools, skills, and subagents** — with **budgets** (token / time / turn) and
**structured tracing & logging** built in, not bolted on.

It deliberately stays small and readable. LLM calls go through the official
OpenAI Python client (or any OpenAI-compatible endpoint), behind a one-method
`Model` interface so you can swap providers later.

```python
import asyncio
from agentic import Agent, tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return {"Paris": "18°C, sunny"}.get(city, "unknown")

agent = Agent("gpt-4o-mini", instructions="Be concise.", tools=[get_weather])

result = asyncio.run(agent.run("What's the weather in Paris?"))
print(result.output)   # -> "It's 18°C and sunny in Paris."
print(result.usage.as_dict())
```

---

## Why this exists

We evaluated [pydantic-ai](https://github.com/pydantic/pydantic-ai) and
[microsoft/agent-framework](https://github.com/microsoft/agent-framework). This
is a focused, self-contained alternative: async-first like both of them, with a
deliberately tiny surface area, first-class observability, and hard run budgets.

## Install

The project is managed with [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev        # install runtime + dev (pytest) deps
uv run pytest              # run the offline test suite
uv run python examples/01_hello_agent.py   # run an example (no API key needed)
```

To use a real model, set `OPENAI_API_KEY`. Dependencies are just `openai` and
`structlog`.

---

## Core concepts

| Concept | What it is |
|--------|------------|
| **`Agent`** | A model + a persona (`instructions`) + `tools` + `skills` + optional `subagents`. Runs a tool-calling loop until the model returns a final answer. |
| **Tool** (`@tool`) | A plain typed Python function the model can call. Its JSON Schema is derived from type hints + docstring — you write no schema. |
| **`Skill`** | A reusable bundle of *instructions* (+ optional tools). Either always-on (`auto_load`) or revealed on demand (*progressive disclosure*) when the model calls the built-in `load_skill` tool. |
| **Subagent** | Another `Agent` exposed to a parent as a delegation tool. Has its own model/tools/context; shares the parent's budget and trace. |
| **`Budget`** | Token / time / turn ceilings enforced across the whole run. Exceeding one raises `BudgetExceeded`. |
| **`RunContext`** | Injected into tools that ask for it. Exposes `deps` (your objects), the live budget, and the tracer. |
| **Structured output** | Set `output_type=` and `result.output` is a validated pydantic model / dataclass / scalar instead of text. |
| **`Trace` / logging** | A structured span tree on every result, plus optional live `structlog` output. |

### Tools

A tool is just a function. Type hints become the schema; the first docstring
line becomes the description; `Args:` lines become parameter descriptions.
Supported types include `str/int/float/bool`, `list[T]`, `dict[str, V]`,
`Optional[T]` / `T | None`, `Literal[...]`, `enum.Enum`, and pydantic models.

```python
from typing import Literal
from agentic import tool, RunContext, ToolError

@tool
def search(ctx: RunContext, query: str, limit: int = 5, mode: Literal["web", "news"] = "web") -> list[str]:
    """Search the web.

    Args:
        query: what to look for.
        limit: maximum number of results.
    """
    if not query:
        raise ToolError("query must not be empty")   # recoverable: fed back to the model
    return ctx.deps.search_client.run(query, limit, mode)
```

- A parameter annotated **`RunContext`** is injected and hidden from the model —
  use it for dependency injection (`ctx.deps`), budget checks, and custom trace
  events (`ctx.tracer.event(...)`).
- A tool that raises is **recoverable** by default: the error text is returned
  to the model so it can adapt. Raise `ToolError(..., fatal=True)` to abort.
- Sync tools run in a thread; multiple tool calls in one turn run concurrently.

#### What are "dependencies" (`deps`)?

`deps` is whatever object **your tools need from the outside world to do their
job** — passed in per run via `agent.run(task, deps=...)` and reached inside a
tool as `ctx.deps`. It's plain dependency injection: instead of tools importing
globals or opening their own connections, the things they depend on are handed
to them. Concrete examples:

- a **database** session / connection pool, an **HTTP client** (`httpx.AsyncClient`),
  a cache (Redis), a vector store or search client;
- **credentials / config / feature flags** (an API key, a `Settings` object);
- the **current request context** — authenticated `user_id`, tenant, locale,
  permissions — which differ on every call;
- app-specific channels, e.g. a progress store or the human-in-the-loop
  `HumanChannel` in [examples 08–09](examples/README.md).

```python
@dataclass
class Deps:
    db: Database
    http: httpx.AsyncClient
    user_id: str

@tool
async def recent_orders(ctx: RunContext, limit: int = 5) -> list[dict]:
    "List the current user's recent orders."
    return await ctx.deps.db.orders_for(ctx.deps.user_id, limit)   # uses injected deps

result = await agent.run("show my orders", deps=Deps(db, http, user_id="u_42"))
```

Why inject instead of using globals? It keeps agents **stateless and reusable**
across concurrent runs (each run carries its own user/connections), makes tools
**easy to test** (pass a fake `Deps` — that's exactly how the offline examples and
tests work), and keeps each tool's requirements **explicit**. `deps` is untyped
by default; `Agent[DepsT]` / `RunContext[DepsT]` let you annotate it for editor
help. If your tools need nothing external, ignore `deps` entirely.

### Skills (instruction bundles)

A skill packages *know-how*: instructions plus the tools that implement them.

```python
from agentic import Skill

refunds = Skill(
    name="refunds",
    description="How to evaluate and process refund requests.",
    instructions="Check the policy, confirm the order is in-window, then issue the refund.",
    tools=[lookup_policy, issue_refund],
)
agent = Agent("gpt-4o-mini", skills=[refunds])
```

By default skills use **progressive disclosure**: only the name + description are
shown to the model up front. When the model decides a skill is relevant it calls
`load_skill("refunds")`; the full instructions are revealed and the skill's tools
switch on for the rest of the run. Set `auto_load=True` to make a skill always
active. Instruction-only skills can be loaded from markdown
(`Skill.from_markdown` / `Skill.from_directory`, with simple `--- key: value ---`
frontmatter).

### Subagents

Pass `subagents=[...]` and each becomes a delegation tool on the parent:

```python
researcher = Agent("gpt-4o-mini", name="researcher", tools=[web_search])
editor = Agent("gpt-4o-mini", name="editor", subagents=[researcher])
# the editor can now call a `researcher` tool with a self-contained task
```

A subagent is a *separate* agent (own context window, own tools, possibly a
different model) — use it for focus and isolation. A skill, by contrast, extends
the *same* agent. The whole tree shares one budget and one trace.

### Structured output

Give an agent an `output_type` and `result.output` comes back as a validated
instance instead of text:

```python
from pydantic import BaseModel
from agentic import Agent

class Ticket(BaseModel):
    title: str
    severity: int
    needs_oncall: bool

agent = Agent("gpt-4o-mini", instructions="Triage incidents.", output_type=Ticket)
result = await agent.run("Checkout 500s during payment.")
result.output.severity   # -> int, validated   (result.output is a Ticket)
```

`output_type` accepts a pydantic model, a dataclass, a `TypedDict`, or a plain
type like `list[str]`. It's implemented provider-neutrally: the agent gains a
synthetic `final_result` tool whose schema is your type, the model answers by
calling it, and the arguments are validated. If the data doesn't validate, the
error is fed back and the model tries again — so you never get a half-valid
object. Tools and structured output compose: gather with tools, then return a
typed result.

### Budgets

```python
from agentic import Budget, BudgetExceeded

try:
    result = await agent.run(task, budget=Budget(max_tokens=50_000, max_time=30.0, max_turns=20))
except BudgetExceeded as exc:
    print(exc.kind, exc.limit, exc.used)   # "tokens" | "time" | "turns"
    partial = exc.messages                 # conversation captured so far
```

Limits cover the **entire run tree** (parent + subagents). They're checked at
call boundaries; `max_time` also caps each in-flight model/tool call so a single
hang can't blow the wall-clock. Reaching `max_turns` is a *soft* stop: the result
returns with `stop_reason="max_turns"` instead of raising.

### Logging & tracing

Two complementary records:

1. **Live structured logs** via `structlog`. Call `configure_logging()` once
   (use `json=True` for log pipelines). You get one line per step, indented by
   nesting:

   ```
   run started            agent=trip_assistant
     turn                 agent=trip_assistant n=1
       llm response       agent=trip_assistant tokens_in=210 tokens_out=18 tokens=228 cum_tokens=228 tokens_left=49772 time_left=29.9
       calling tools      agent=trip_assistant tools=['get_weather']
       tool ok            agent=trip_assistant tool=get_weather args={"city":"Reykjavik"} result=...
     ...
   run finished           agent=trip_assistant stop=completed tokens=228
   ```

   Every model call logs **token usage + remaining budget**; every turn logs
   **what the agent is doing**; the `agent=` field tracks which (sub)agent is
   acting.

2. **A structured trace tree** on every `RunResult`, regardless of logging:

   ```python
   result.trace.format()     # indented text tree
   result.trace.as_dict()    # JSON-serialisable, for dashboards / offline viewers
   ```

---

## How it works

The run loop is intentionally simple:

```
Agent.run(user_input)
  └─ compose system prompt  (instructions + auto-skills + skill catalogue + subagents)
  └─ loop up to max_turns:
       ├─ check budget (time / tokens / turns)        ── raises BudgetExceeded
       ├─ call the model with the currently-active tools
       ├─ record token usage against the budget
       ├─ no tool calls?  →  return final answer  (RunResult)
       └─ run the requested tools concurrently, append results, loop
```

The **active tool set is recomputed every turn**, which is what makes
progressive skill-loading work: calling `load_skill` adds that skill's tools to
the set for subsequent turns. Subagents are delegation tools whose body runs
another `Agent` with the *same* `RunContext`, so budgets and traces are shared
and the trace nests naturally.

### Module map

```
agentic/
  agent.py      Agent: the run loop, subagent wiring, skill activation, budget+trace hookup
  models.py     Model interface + OpenAIModel (Chat Completions; any compatible endpoint)
  tools.py      Tool + @tool decorator; arg coercion; RunContext injection
  schema.py     JSON Schema generation from type hints + docstring parsing
  skills.py     Skill: instruction bundles, progressive disclosure, markdown loader
  budget.py     Usage accounting + Budget enforcement
  tracing.py    Span / Trace tree + structlog wiring (configure_logging)
  context.py    RunContext (injected into tools) + RunResult
  messages.py   Provider-neutral chat message types
  errors.py     Exception hierarchy (AgenticError, ToolError, BudgetExceeded, ...)
  testing.py    FunctionModel / ScriptedModel for offline, deterministic runs & tests
```

### Design choices

- **Async-first.** `run()` is a coroutine; tool calls within a turn run via
  `asyncio.gather`; sync tools are offloaded to threads. There's a `run_sync()`
  convenience wrapper for scripts/notebooks.
- **Provider-thin.** Everything goes through `Model.generate(...)`. `OpenAIModel`
  uses Chat Completions (the broadly-compatible standard); add a provider by
  writing one subclass.
- **Stateless agents.** An `Agent` holds no per-run state, so one instance is
  safe to reuse and run concurrently. All run state lives in the `RunContext`.
- **Observable by default, quiet by default.** The trace tree is always built;
  live logs only appear once you call `configure_logging()`.

---

## Testing your own agents

`FunctionModel` drives an agent from a Python function instead of the network,
so you can test the full loop deterministically with no API key:

```python
from agentic import Agent
from agentic.testing import FunctionModel

def brain(messages, tools):
    if messages[-1].role == "user":
        return [("get_weather", {"city": "Paris"})]   # ask for a tool
    return "It's sunny in Paris."                       # then answer

agent = Agent(FunctionModel(brain), tools=[get_weather])
result = await agent.run("weather in Paris?")
assert "sunny" in result.output
```

See [`tests/test_agentic.py`](tests/test_agentic.py) for tools, skills,
subagents, and budget tests, and the [examples guide](examples/README.md) for a
concept-by-concept walkthrough.

## Status

A clean reference implementation (v0.1). Natural next steps if you adopt it:
streaming responses, automatic retries with backoff, and additional `Model`
backends (e.g. Anthropic). The interfaces are designed to absorb these without
breaking the surface above.
