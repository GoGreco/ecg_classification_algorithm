from pathlib import Path

from ecg_classification.agents.loader import load_agent_spec


def test_load_agent_spec_parses_frontmatter_and_prompt(tmp_path: Path) -> None:
    agent_path = tmp_path / "reviewer.md"
    agent_path.write_text(
        """---
name: reviewer
role: senior reviewer
objective: review manuscripts
model: gpt-test
temperature: 0.1
handoff_to: researcher
tools:
  - manuscript
  - figures
---
Act critically.
""",
        encoding="utf-8",
    )

    spec = load_agent_spec(agent_path)

    assert spec.name == "reviewer"
    assert spec.role == "senior reviewer"
    assert spec.objective == "review manuscripts"
    assert spec.model == "gpt-test"
    assert spec.temperature == 0.1
    assert spec.handoff_to == "researcher"
    assert spec.tools == ("manuscript", "figures")
    assert spec.system_prompt == "Act critically."
