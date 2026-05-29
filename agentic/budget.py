"""Token / time / turn accounting and budget enforcement.

A single :class:`Budget` instance is shared across an entire run *tree* (the
top-level agent and every subagent it delegates to), so limits apply to the
whole logical run rather than to each agent in isolation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .errors import BudgetExceeded


@dataclass(slots=True)
class Usage:
    """Cumulative token usage."""

    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(
        self,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int | None = None,
    ) -> None:
        self.requests += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += (
            total_tokens if total_tokens is not None else prompt_tokens + completion_tokens
        )

    def copy(self) -> "Usage":
        return Usage(
            requests=self.requests,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.total_tokens,
        )

    def __sub__(self, other: "Usage") -> "Usage":
        return Usage(
            requests=self.requests - other.requests,
            prompt_tokens=self.prompt_tokens - other.prompt_tokens,
            completion_tokens=self.completion_tokens - other.completion_tokens,
            total_tokens=self.total_tokens - other.total_tokens,
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "requests": self.requests,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class Budget:
    """Ceilings for a single run.

    Any limit left as ``None`` is unbounded. Limits are enforced at call
    boundaries: the run is stopped *before* a model call that would start over
    budget, and immediately *after* a call that pushed token usage over the top.
    Exceeding a limit raises :class:`BudgetExceeded`.
    """

    max_tokens: int | None = None
    max_time: float | None = None  # wall-clock seconds for the whole run
    max_turns: int | None = None  # total LLM round-trips across the run tree

    usage: Usage = field(default_factory=Usage)
    turns: int = 0
    _start: float | None = field(default=None, repr=False)

    def start(self) -> "Budget":
        """Begin the clock (idempotent)."""
        if self._start is None:
            self._start = time.monotonic()
        return self

    @property
    def elapsed(self) -> float:
        if self._start is None:
            return 0.0
        return time.monotonic() - self._start

    @property
    def remaining_time(self) -> float | None:
        if self.max_time is None:
            return None
        return self.max_time - self.elapsed

    @property
    def remaining_tokens(self) -> int | None:
        if self.max_tokens is None:
            return None
        return self.max_tokens - self.usage.total_tokens

    # --- mutation -----------------------------------------------------------------

    def add_usage(
        self, prompt_tokens: int, completion_tokens: int, total_tokens: int | None = None
    ) -> None:
        self.usage.add(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    def tick_turn(self) -> None:
        self.turns += 1

    # --- enforcement --------------------------------------------------------------

    def check(self) -> None:
        """Raise :class:`BudgetExceeded` if any limit has been reached."""
        if self.max_time is not None and self.elapsed > self.max_time:
            raise BudgetExceeded(
                f"time budget exceeded: {self.elapsed:.2f}s of {self.max_time:.2f}s",
                kind="time",
                limit=self.max_time,
                used=round(self.elapsed, 3),
            )
        if self.max_tokens is not None and self.usage.total_tokens >= self.max_tokens:
            raise BudgetExceeded(
                f"token budget exceeded: {self.usage.total_tokens} of {self.max_tokens}",
                kind="tokens",
                limit=self.max_tokens,
                used=self.usage.total_tokens,
            )
        if self.max_turns is not None and self.turns > self.max_turns:
            raise BudgetExceeded(
                f"turn budget exceeded: {self.turns} of {self.max_turns}",
                kind="turns",
                limit=self.max_turns,
                used=self.turns,
            )

    def time_left_or_raise(self) -> float | None:
        """Return remaining seconds (or ``None`` if unbounded); raise if already out."""
        rt = self.remaining_time
        if rt is not None and rt <= 0:
            raise BudgetExceeded(
                f"time budget exceeded: {self.elapsed:.2f}s of {self.max_time:.2f}s",
                kind="time",
                limit=self.max_time,
                used=round(self.elapsed, 3),
            )
        return rt
