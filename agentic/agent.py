"""The Agent: an LLM plus tools, skills, and (optionally) subagents.

An agent runs a simple, well-traced loop:

1. Compose a system prompt from its instructions, any auto-loaded skills, a
   catalogue of loadable skills, and its subagents.
2. Call the model with the currently-active tools.
3. If the model asked for tool calls, run them (concurrently) and loop.
4. Otherwise, return the model's text as the final answer.

A token / time / turn :class:`~agentic.budget.Budget` is shared across the whole
run tree, and every step is recorded in a :class:`~agentic.tracing.Trace`.
Subagents are just ``Agent`` instances exposed to a parent as delegation tools;
they share the parent's budget and trace.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Generic, Iterable, Sequence, TypeVar

import structlog

from .budget import Budget
from .context import RunContext, RunResult
from .errors import BudgetExceeded, ToolError
from .messages import Message, ToolCall
from .models import Model, OpenAIModel
from .output import OutputSpec
from .skills import Skill
from .tools import Tool, as_tool, stringify_result
from .tracing import Trace, Tracer

DepsT = TypeVar("DepsT")

DEFAULT_MAX_TURNS = 12


class _RunState:
    """Mutable per-run state (kept off ``Agent`` so agents are reusable/concurrent)."""

    __slots__ = ("loaded_skills", "load_skill_tool", "final_result_tool", "final_output", "has_output")

    def __init__(self) -> None:
        self.loaded_skills: set[str] = set()
        self.load_skill_tool: Tool | None = None
        self.final_result_tool: Tool | None = None
        self.final_output: Any = None
        self.has_output: bool = False  # set once final_result validates (output may be falsy)


class Agent(Generic[DepsT]):
    """A configurable agent.

    Args:
        model: a :class:`~agentic.models.Model`, or a model-id string which is
            wrapped in :class:`~agentic.models.OpenAIModel`.
        name: identifier used in traces and as the base for its delegation tool.
        instructions: the system prompt / persona.
        tools: callables or :class:`~agentic.tools.Tool` objects the model may call.
        skills: :class:`~agentic.skills.Skill` bundles (auto-loaded or on-demand).
        subagents: other agents this one may delegate to.
        output_type: if set (a pydantic model, dataclass, TypedDict, or any type),
            ``RunResult.output`` is a validated instance of it instead of text.
        max_turns: maximum model round-trips before the run stops with
            ``stop_reason="max_turns"``.
    """

    def __init__(
        self,
        model: Model | str,
        *,
        name: str = "agent",
        instructions: str = "",
        tools: Iterable[Tool | Any] = (),
        skills: Iterable[Skill] = (),
        subagents: Iterable["Agent"] = (),
        output_type: Any = None,
        max_turns: int = DEFAULT_MAX_TURNS,
    ) -> None:
        self.model: Model = OpenAIModel(model) if isinstance(model, str) else model
        self.name = name
        self.instructions = instructions
        self.max_turns = max_turns
        # Structured output is off when output_type is None or plain `str`.
        self._output: OutputSpec | None = (
            OutputSpec(output_type) if output_type not in (None, str) else None
        )

        self._base_tools: list[Tool] = [as_tool(t) for t in tools]
        skills = list(skills)
        self._auto_skills: list[Skill] = [s for s in skills if s.auto_load]
        self._loadable_skills: dict[str, Skill] = {s.name: s for s in skills if not s.auto_load}

        self._subagents: list[Agent] = list(subagents)
        self._subagent_tools: list[Tool] = []
        self._subagent_tool_names: dict[str, str] = {}
        used_names = {t.name for t in self._base_tools}
        for sub in self._subagents:
            tool_name = _unique(_sanitize_tool_name(sub.name), used_names)
            used_names.add(tool_name)
            self._subagent_tool_names[sub.name] = tool_name
            self._subagent_tools.append(self._make_subagent_tool(sub, tool_name))

    # --- public API ---------------------------------------------------------------

    async def run(
        self,
        user_input: str,
        *,
        deps: DepsT | None = None,
        budget: Budget | None = None,
        message_history: Sequence[Message] | None = None,
        max_turns: int | None = None,
        logger: Any = None,
        _parent_ctx: RunContext | None = None,
    ) -> RunResult[Any]:
        """Run the agent on ``user_input`` and return a :class:`RunResult`.

        Raises :class:`~agentic.errors.BudgetExceeded` if a token/time/turn
        ceiling trips. Reaching ``max_turns`` is not an error: the result is
        returned with ``stop_reason="max_turns"``.
        """
        is_subagent = _parent_ctx is not None
        if is_subagent:
            assert _parent_ctx is not None
            budget = _parent_ctx.budget
            tracer = _parent_ctx.tracer
            trace = _parent_ctx.trace
            if deps is None:
                deps = _parent_ctx.deps
        else:
            budget = (budget or Budget()).start()
            trace = Trace()
            tracer = Tracer(trace, logger=logger)

        ctx: RunContext = RunContext(
            deps=deps, budget=budget, tracer=tracer, trace=trace, agent_name=self.name
        )
        max_turns = max_turns if max_turns is not None else self.max_turns

        state = _RunState()
        if self._loadable_skills:
            state.load_skill_tool = self._make_load_skill_tool(state)
        if self._output is not None:
            state.final_result_tool = self._make_final_result_tool(state)

        convo: list[Message] = [Message.system(self._compose_system_prompt())]
        if message_history:
            convo.extend(m for m in message_history if m.role != "system")
        convo.append(Message.user(user_input))

        output = ""
        last_text = ""
        stop_reason = "completed"
        turns_done = 0

        span_kind = "subagent" if is_subagent else "run"
        # Tag every log line emitted during this (sub)agent's turn with its name.
        bind_tokens = structlog.contextvars.bind_contextvars(agent=self.name)
        try:
            async with tracer.span(span_kind, self.name, model=self.model.name) as run_span:
                try:
                    for turn in range(1, max_turns + 1):
                        budget.tick_turn()
                        budget.check()
                        turns_done = turn

                        async with tracer.span("turn", str(turn)) as turn_span:
                            remaining = budget.time_left_or_raise()
                            tools_by_name = self._active_tools(state)
                            tool_list = list(tools_by_name.values())

                            async with tracer.span(
                                "model",
                                self.model.name,
                                model=self.model.name,
                                tools=len(tool_list),
                            ) as model_span:
                                resp = await _with_deadline(
                                    self.model.generate(convo, tool_list, timeout=remaining),
                                    budget,
                                )
                                budget.add_usage(
                                    resp.prompt_tokens, resp.completion_tokens, resp.total_tokens
                                )
                                model_span.attributes.update(
                                    prompt_tokens=resp.prompt_tokens,
                                    completion_tokens=resp.completion_tokens,
                                    total_tokens=resp.total_tokens,
                                    finish_reason=resp.finish_reason,
                                    # usage statistics + remaining budget for the log line
                                    cum_tokens=budget.usage.total_tokens,
                                    tokens_left=budget.remaining_tokens,
                                    time_left=(
                                        round(budget.remaining_time, 2)
                                        if budget.remaining_time is not None
                                        else None
                                    ),
                                )

                            budget.check()  # stop promptly if that call pushed us over

                            assistant = resp.message
                            convo.append(assistant)
                            if assistant.content:
                                last_text = assistant.content

                            if not assistant.tool_calls:
                                if self._output is not None and not state.has_output:
                                    # Wanted structured output but got prose: nudge & retry.
                                    convo.append(
                                        Message.user(
                                            "Please provide the final answer by calling the "
                                            f"`{self._output.tool_name}` tool."
                                        )
                                    )
                                    tracer.event("awaiting structured result")
                                    continue
                                output = assistant.content or ""
                                turn_span.attributes["result"] = "final answer"
                                tracer.event("final answer", chars=len(output))
                                break

                            # Surface what the agent decided to do this turn.
                            tool_names = [tc.name for tc in assistant.tool_calls]
                            turn_span.attributes["tool_calls"] = tool_names
                            tracer.event("calling tools", tools=tool_names)
                            results = await self._run_tools(
                                assistant.tool_calls, tools_by_name, ctx, state
                            )
                            convo.extend(results)

                            # A validated structured result ends the run.
                            if state.has_output:
                                output = state.final_output
                                turn_span.attributes["result"] = "structured result"
                                tracer.event("final result", type=type(output).__name__)
                                break
                    else:
                        stop_reason = "max_turns"
                        output = last_text
                except BudgetExceeded as exc:
                    # Attach the partial conversation for debugging / partial use.
                    exc.messages = convo[1:]
                    run_span.attributes["stop_reason"] = f"budget:{exc.kind}"
                    run_span.attributes["total_tokens"] = budget.usage.total_tokens
                    raise

                run_span.attributes["stop_reason"] = stop_reason
                run_span.attributes["total_tokens"] = budget.usage.total_tokens
        finally:
            structlog.contextvars.reset_contextvars(**bind_tokens)

        return RunResult(
            output=output,
            messages=convo[1:],  # drop the system prompt; round-trips as message_history
            usage=budget.usage.copy(),
            elapsed=budget.elapsed,
            turns=turns_done,
            stop_reason=stop_reason,
            trace=trace,
        )

    def run_sync(self, user_input: str, **kwargs: Any) -> RunResult[Any]:
        """Convenience blocking wrapper around :meth:`run` for scripts/notebooks."""
        return asyncio.run(self.run(user_input, **kwargs))

    # --- tool execution -----------------------------------------------------------

    async def _run_tools(
        self,
        tool_calls: list[ToolCall],
        tools_by_name: dict[str, Tool],
        ctx: RunContext,
        state: _RunState,
    ) -> list[Message]:
        async def run_one(tc: ToolCall) -> Message:
            async with ctx.tracer.span("tool", tc.name) as span:
                tool = tools_by_name.get(tc.name)
                if tool is None:
                    span.status = "error"
                    span.error = "unknown tool"
                    return Message.tool(tc.id, f"Error: unknown tool '{tc.name}'.")
                try:
                    args = json.loads(tc.arguments or "{}")
                    if not isinstance(args, dict):
                        raise ValueError("tool arguments must be a JSON object")
                except (json.JSONDecodeError, ValueError) as exc:
                    span.status = "error"
                    span.error = str(exc)
                    return Message.tool(tc.id, f"Error: could not parse arguments — {exc}")

                span.attributes["arguments"] = _preview(args)
                try:
                    result = await _with_deadline(tool.call(args, ctx), ctx.budget)
                except BudgetExceeded:
                    raise  # hard cap: abort the whole run
                except ToolError as exc:
                    if exc.fatal:
                        raise
                    span.status = "error"
                    span.error = str(exc)
                    return Message.tool(tc.id, f"Error: {exc}")
                except Exception as exc:  # noqa: BLE001 - surface to the model to recover
                    span.status = "error"
                    span.error = f"{type(exc).__name__}: {exc}"
                    return Message.tool(
                        tc.id, f"Error: tool '{tc.name}' raised {type(exc).__name__}: {exc}"
                    )

                content = stringify_result(result)
                span.attributes["result"] = _preview(content)
                return Message.tool(tc.id, content)

        return list(await asyncio.gather(*(run_one(tc) for tc in tool_calls)))

    # --- composition helpers ------------------------------------------------------

    def _active_tools(self, state: _RunState) -> dict[str, Tool]:
        """The tools available this turn: base + subagents + active skills + load_skill."""
        ordered: list[Tool] = list(self._base_tools)
        ordered.extend(self._subagent_tools)
        for skill in self._auto_skills:
            ordered.extend(skill.tools)
        for name in state.loaded_skills:
            ordered.extend(self._loadable_skills[name].tools)
        if state.load_skill_tool is not None:
            ordered.append(state.load_skill_tool)
        if state.final_result_tool is not None:
            ordered.append(state.final_result_tool)

        by_name: dict[str, Tool] = {}
        for tool in ordered:
            by_name.setdefault(tool.name, tool)
        return by_name

    def _compose_system_prompt(self) -> str:
        parts: list[str] = []
        if self.instructions.strip():
            parts.append(self.instructions.strip())

        for skill in self._auto_skills:
            parts.append(f"## Skill: {skill.name}\n{skill.instructions.strip()}")

        if self._loadable_skills:
            lines = "\n".join(
                f"- {s.name}: {s.description}" for s in self._loadable_skills.values()
            )
            parts.append(
                "## Available skills\n"
                "Each skill below provides extra instructions and tools. When one is relevant "
                "to the task, call the `load_skill` tool with its exact name to activate it "
                "before continuing.\n"
                f"{lines}"
            )

        if self._subagents:
            lines = "\n".join(
                f"- `{self._subagent_tool_names[sa.name]}`: "
                f"{_first_line(sa.instructions) or sa.name}"
                for sa in self._subagents
            )
            parts.append(
                "## Delegation\n"
                "You can hand a self-contained sub-task to a specialist subagent by calling "
                "one of these tools; each returns that subagent's final answer:\n"
                f"{lines}"
            )

        if self._output is not None:
            parts.append(
                "## Final answer\n"
                f"When you have the complete answer, call the `{self._output.tool_name}` tool "
                "with the result as structured data. Do not write the final answer as plain text."
            )

        return "\n\n".join(parts) if parts else "You are a helpful assistant."

    def _make_final_result_tool(self, state: _RunState) -> Tool:
        spec = self._output
        assert spec is not None

        async def final_result(ctx: RunContext, **fields: Any) -> str:
            try:
                state.final_output = spec.validate(fields)
            except Exception as exc:  # noqa: BLE001 - recoverable: the model retries
                raise ToolError(f"the result did not match the required schema: {exc}")
            state.has_output = True
            return "Final result accepted."

        return Tool(
            name=spec.tool_name,
            description=(
                "Provide the final answer as structured data matching the required schema. "
                "Call this exactly once, when you have the complete answer."
            ),
            parameters=spec.parameters,
            func=final_result,
            takes_context=True,
            context_param="ctx",
        )

    def _make_load_skill_tool(self, state: _RunState) -> Tool:
        loadable = self._loadable_skills

        async def load_skill(ctx: RunContext, name: str) -> str:
            """Load a skill by name to reveal its full instructions and enable its tools.

            Args:
                name: the skill name, exactly as listed under "Available skills".
            """
            skill = loadable.get(name)
            if skill is None:
                return f"Error: unknown skill '{name}'. Available: {sorted(loadable)}"
            already = name in state.loaded_skills
            state.loaded_skills.add(name)
            ctx.tracer.event(
                f"loaded skill '{name}'", tools=[t.name for t in skill.tools]
            )
            tool_note = (
                f" New tools now available: {', '.join(t.name for t in skill.tools)}."
                if skill.tools
                else ""
            )
            prefix = "(already loaded) " if already else ""
            return (
                f"{prefix}Skill '{name}' is now active.{tool_note}\n\n"
                f"## {skill.name} instructions\n{skill.instructions.strip()}"
            )

        return Tool.from_function(load_skill, name="load_skill")

    def _make_subagent_tool(self, sub: "Agent", tool_name: str) -> Tool:
        async def delegate(ctx: RunContext, task: str) -> str:
            result = await sub.run(task, _parent_ctx=ctx)
            return result.output or "(the subagent produced no output)"

        description = (
            f"Delegate a self-contained sub-task to the '{sub.name}' specialist subagent "
            "and return its answer."
        )
        if sub.instructions.strip():
            description += f" Specialty: {_first_line(sub.instructions)}"
        # Give the injected `task` parameter a helpful description.
        delegate.__doc__ = (
            "Delegate a self-contained sub-task to a specialist subagent.\n\n"
            "Args:\n    task: a complete, standalone description of the sub-task to perform."
        )
        return Tool.from_function(delegate, name=tool_name, description=description)


# --- module helpers ---------------------------------------------------------------


async def _with_deadline(coro: Any, budget: Budget) -> Any:
    """Await ``coro`` but never beyond the run's remaining time budget."""
    remaining = budget.time_left_or_raise()
    if remaining is None:
        return await coro
    try:
        return await asyncio.wait_for(coro, timeout=remaining)
    except asyncio.TimeoutError as exc:
        raise BudgetExceeded(
            f"time budget exceeded: {budget.elapsed:.2f}s of {budget.max_time:.2f}s",
            kind="time",
            limit=budget.max_time,
            used=round(budget.elapsed, 3),
        ) from exc


_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def _sanitize_tool_name(name: str) -> str:
    cleaned = _NAME_RE.sub("_", name.strip()).strip("_")
    return (cleaned or "subagent")[:48]


def _unique(name: str, used: set[str]) -> str:
    if name not in used:
        return name
    i = 2
    while f"{name}_{i}" in used:
        i += 1
    return f"{name}_{i}"


def _first_line(text: str) -> str:
    for line in (text or "").splitlines():
        if line.strip():
            return line.strip()
    return ""


def _preview(value: Any, limit: int = 200) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    return text if len(text) <= limit else text[: limit - 1] + "…"
