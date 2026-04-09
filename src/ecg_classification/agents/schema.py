from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class AgentSpec:
    name: str
    role: str
    objective: str
    model: str | None
    temperature: float | None
    handoff_to: str | None
    tools: tuple[str, ...] = field(default_factory=tuple)
    system_prompt: str = ""
    source_path: Path | None = None
