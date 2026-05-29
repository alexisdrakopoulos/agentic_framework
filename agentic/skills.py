"""Skills: reusable, named bundles of instructions (and optional tools).

A skill packages *know-how* — a chunk of system-prompt instructions plus any
tools that go with it — under a name and one-line description. This mirrors the
Claude "Agent Skills" model.

Two activation modes:

* ``auto_load=True`` — the skill's instructions are baked into the system prompt
  from the start and its tools are always available.
* ``auto_load=False`` (default, *progressive disclosure*) — only the name and
  description are advertised to the model. When the model decides it is
  relevant, it calls the built-in ``load_skill`` tool; the full instructions are
  then returned to it and the skill's tools become available for the rest of the
  run.

Define skills in Python (with tools), or load instruction-only skills from a
``SKILL.md`` file/directory with simple ``key: value`` frontmatter.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable

from .errors import SkillError
from .tools import Tool, as_tool


@dataclass
class Skill:
    """A named instruction bundle, optionally carrying tools."""

    name: str
    description: str
    instructions: str
    tools: list[Tool] = field(default_factory=list)
    auto_load: bool = False

    def __post_init__(self) -> None:
        self.tools = [as_tool(t) for t in self.tools]

    @classmethod
    def from_markdown(
        cls,
        text: str,
        *,
        name: str | None = None,
        description: str | None = None,
        tools: list[Tool | Callable] | None = None,
        auto_load: bool = False,
    ) -> "Skill":
        """Build a skill from markdown with optional ``--- key: value ---`` frontmatter."""
        meta, body = _parse_frontmatter(text)
        resolved_name = name or meta.get("name")
        if not resolved_name:
            raise SkillError("skill is missing a name (pass name= or add it to frontmatter)")
        return cls(
            name=resolved_name,
            description=description or meta.get("description", ""),
            instructions=body.strip(),
            tools=[as_tool(t) for t in (tools or [])],
            auto_load=auto_load,
        )

    @classmethod
    def from_directory(
        cls,
        path: str,
        *,
        tools: list[Tool | Callable] | None = None,
        auto_load: bool = False,
    ) -> "Skill":
        """Load a skill from ``<path>/SKILL.md`` (or ``<path>`` if it is a file)."""
        skill_file = path if os.path.isfile(path) else os.path.join(path, "SKILL.md")
        if not os.path.isfile(skill_file):
            raise SkillError(f"no SKILL.md found at {skill_file!r}")
        with open(skill_file, encoding="utf-8") as fh:
            text = fh.read()
        name_default = os.path.basename(os.path.dirname(skill_file) or skill_file)
        skill = cls.from_markdown(text, tools=tools, auto_load=auto_load)
        if not skill.name:
            skill.name = name_default
        return skill


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split simple ``--- ... ---`` ``key: value`` frontmatter from the body.

    Intentionally tiny — no YAML dependency. Anything beyond flat ``key: value``
    lines should live in the instruction body.
    """
    stripped = text.lstrip()
    if not stripped.startswith("---"):
        return {}, text
    lines = stripped.splitlines()
    meta: dict[str, str] = {}
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
        if ":" in lines[i]:
            key, _, value = lines[i].partition(":")
            meta[key.strip()] = value.strip().strip("\"'")
    if end is None:
        return {}, text
    body = "\n".join(lines[end + 1 :])
    return meta, body
