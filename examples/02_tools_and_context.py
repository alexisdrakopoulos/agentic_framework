"""02 · Tools, typed parameters, and dependency injection

Tools get more interesting when they take rich arguments and need access to
your application's state (a database handle, the current user, config, ...).

Key ideas
---------
* Rich parameter types just work: ``int``, ``float``, ``bool``, ``list[T]``,
  ``Optional[T]`` / ``T | None``, ``Literal[...]``, and ``enum.Enum`` all map to
  correct JSON Schema. Defaults make a parameter optional.
* Dependency injection: declare a parameter annotated ``RunContext`` (by
  convention ``ctx``). The framework injects it and hides it from the model.
  From it you reach ``ctx.deps`` (your object, passed to ``run(deps=...)``), the
  live budget, and the tracer.
* Errors are recoverable by default: if a tool raises, the error text is fed
  back to the model as the tool result so it can retry or adapt. Raise
  ``ToolError(..., fatal=True)`` to abort the whole run instead.

Run it:
    uv run python examples/02_tools_and_context.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Literal

from agentic import Agent, RunContext, ToolError, tool

from _shared import banner, demo_model, steps_taken


# --- your application's dependencies ----------------------------------------------
@dataclass
class Deps:
    catalog: dict[str, float] = field(default_factory=dict)  # product -> price


# --- tools ------------------------------------------------------------------------
@tool
def lookup_price(ctx: RunContext, product: str) -> float:
    """Look up the price of a product from the catalog.

    Args:
        product: the product name to price.
    """
    # `ctx.deps` is whatever you passed to `agent.run(deps=...)`.
    price = ctx.deps.catalog.get(product)
    if price is None:
        # Recoverable: the model sees this message and can pick a real product.
        raise ToolError(f"unknown product {product!r}; known: {list(ctx.deps.catalog)}")
    return price


@tool
def quote(
    items: list[str],
    discount_pct: float = 0.0,
    currency: Literal["USD", "EUR"] = "USD",
) -> str:
    """Produce a formatted price quote.

    Args:
        items: product names to include.
        discount_pct: percentage discount to apply to the subtotal.
        currency: the currency to render the total in.
    """
    # (kept trivial; a real tool might call lookup_price internally)
    symbol = {"USD": "$", "EUR": "€"}[currency]
    return f"{len(items)} item(s), {discount_pct:.0f}% off, totals in {symbol}"


def brain(messages, tools):
    step = steps_taken(messages)
    if step == 0:  # try a product that doesn't exist -> tool raises
        return [("lookup_price", {"product": "Gadget"})]
    if step == 1:  # the model saw the error and the valid options; retry
        assert "known" in messages[-1].content
        return [("lookup_price", {"product": "Widget"})]
    if step == 2:
        return [("quote", {"items": ["Widget"], "discount_pct": 10, "currency": "EUR"})]
    return "A single Widget is €9.00; with 10% off your quote is ready."


agent = Agent(
    demo_model(brain),
    name="sales-assistant",
    instructions="You help customers price products. Use the catalog; never invent prices.",
    tools=[lookup_price, quote],
)


async def main() -> None:
    deps = Deps(catalog={"Widget": 9.0, "Sprocket": 12.5})
    result = await agent.run("Quote me one Gadget, 10% off, in euros.", deps=deps)

    banner("Output")
    print(result.output)

    banner("Generated schema for `quote` (derived from type hints + docstring)")
    import json

    print(json.dumps(agent._base_tools[1].parameters, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
