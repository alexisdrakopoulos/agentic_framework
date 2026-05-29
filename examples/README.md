# A guided tour of `agentic`

These examples build up the framework one concept at a time. Read them in order —
each adds a single new idea on top of the previous one.

**Everything here runs offline with no API key.** Each example uses a small
scripted "brain" (via `FunctionModel`) so the behaviour is deterministic and
free. The teaching code — how you define agents, tools, skills, and subagents —
is exactly what you'd write against a real model. If you set `OPENAI_API_KEY`,
examples 01–05 will transparently use `gpt-4o-mini` instead (see
[`_shared.py`](./_shared.py)).

Run any example with:

```bash
uv run python examples/01_hello_agent.py
```

| # | File | What it teaches |
|---|------|-----------------|
| 01 | [`01_hello_agent.py`](./01_hello_agent.py) | The smallest agent: instructions + one tool + `run()`. How a typed function becomes a tool. |
| 02 | [`02_tools_and_context.py`](./02_tools_and_context.py) | Rich parameter types, `RunContext` dependency injection (`ctx.deps`), and recoverable vs. fatal tool errors. |
| 03 | [`03_skills.py`](./03_skills.py) | Skills as instruction bundles. Progressive disclosure via `load_skill`, `auto_load` skills, and `Skill.from_markdown`. |
| 04 | [`04_subagents.py`](./04_subagents.py) | Delegating to specialist subagents. One shared budget/trace; nested traces; per-agent log context. |
| 05 | [`05_budgets_and_tracing.py`](./05_budgets_and_tracing.py) | Token/time/turn budgets and `BudgetExceeded`. Live structlog logging and the serialisable trace tree. |
| 06 | [`06_openai_end_to_end.py`](./06_openai_end_to_end.py) | Everything together against a **real** OpenAI model (needs `OPENAI_API_KEY`). |
| 07 | [`07_structured_output.py`](./07_structured_output.py) | `output_type=` to get a validated pydantic model (or dataclass / scalar) back instead of text, with auto re-validation. |
| 08 | [`08_progress_polling.py`](./08_progress_polling.py) | Show live, human-friendly progress to a **polling REST UI** by tapping the event stream into a pollable store. |
| 09 | [`09_human_in_the_loop.py`](./09_human_in_the_loop.py) | An `ask_user` tool that pauses the run for a clarifying question and resumes when the answer arrives via a **REST endpoint**. |

Examples 08–09 share a small bit of web glue in [`_webio.py`](./_webio.py) — the
kind of code that would live in your web service (a progress store, a structlog
processor that humanizes events, and the human-in-the-loop channel). Both files
sketch the matching FastAPI endpoints at the bottom.

## What to look for

- **01–02** print results to stdout. Example 02 prints the JSON Schema the
  framework auto-generated for a tool — you never write that by hand.
- **03–05** call `configure_logging()`, so you'll see the live, indented
  **structlog** trace. Watch:
  - each `turn` and the `calling tools` line — *what the agent is doing*;
  - every `llm response` line — `tokens_in/out`, cumulative `cum_tokens`, and
    remaining budget (`tokens_left`, `time_left`);
  - the `agent=` field — in example 04 it switches as work is delegated to
    subagents and back.
- **05** shows a run being stopped mid-flight by a token budget, with the
  partial conversation attached to the exception, and dumps the trace as JSON.

## Going further

- Swap the model anywhere: `Agent(OpenAIModel("gpt-4o-mini"), ...)`, a bare
  `Agent("gpt-4o-mini", ...)`, or any OpenAI-compatible endpoint via
  `OpenAIModel(..., base_url=...)`.
- For deterministic tests of *your* agents, use
  [`agentic.testing.FunctionModel`](../agentic/testing.py) the same way these
  examples do — see [`tests/test_agentic.py`](../tests/test_agentic.py).
