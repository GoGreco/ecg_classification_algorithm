from pathlib import Path

from ecg_classification.agents.orchestration import (
    PLACEHOLDER_RESEARCHER_RESPONSE,
    PLACEHOLDER_REVIEWER_REPORT,
    create_review_session,
    progress_review_session,
)


def _write_agent(path: Path, name: str) -> None:
    path.write_text(
        f"""---
name: {name}
role: {name} role
objective: {name} objective
model: gpt-test
temperature: 0.2
---
Prompt for {name}.
""",
        encoding="utf-8",
    )


def test_create_review_session_builds_first_round(tmp_path: Path) -> None:
    manuscript_path = tmp_path / "manuscript.md"
    reviewer_path = tmp_path / "reviewer.md"
    researcher_path = tmp_path / "researcher.md"
    session_dir = tmp_path / "session"
    manuscript_path.write_text("# Manuscript\n\nResults section.", encoding="utf-8")
    _write_agent(reviewer_path, "reviewer")
    _write_agent(researcher_path, "researcher")

    create_review_session(manuscript_path, reviewer_path, researcher_path, session_dir)

    assert (session_dir / "current_manuscript.md").exists()
    assert (session_dir / "round_01" / "reviewer_prompt.md").exists()
    assert (session_dir / "round_01" / "reviewer_report.md").read_text(encoding="utf-8") == PLACEHOLDER_REVIEWER_REPORT


def test_progress_review_session_generates_bidirectional_cycle(tmp_path: Path) -> None:
    manuscript_path = tmp_path / "manuscript.md"
    reviewer_path = tmp_path / "reviewer.md"
    researcher_path = tmp_path / "researcher.md"
    session_dir = tmp_path / "session"
    manuscript_path.write_text("# Manuscript\n\nOriginal.", encoding="utf-8")
    _write_agent(reviewer_path, "reviewer")
    _write_agent(researcher_path, "researcher")

    create_review_session(manuscript_path, reviewer_path, researcher_path, session_dir)

    reviewer_report_path = session_dir / "round_01" / "reviewer_report.md"
    reviewer_report_path.write_text("# Parecer\n\nFaltam testes estatisticos.", encoding="utf-8")

    first_status = progress_review_session(session_dir, reviewer_path, researcher_path)

    assert first_status == "researcher_prompt_created"
    assert (session_dir / "round_01" / "researcher_prompt.md").exists()
    assert (session_dir / "round_01" / "researcher_response.md").read_text(encoding="utf-8") == PLACEHOLDER_RESEARCHER_RESPONSE

    researcher_response_path = session_dir / "round_01" / "researcher_response.md"
    revised_manuscript_path = session_dir / "round_01" / "revised_manuscript.md"
    researcher_response_path.write_text("# Resposta\n\nIncluimos novos testes.", encoding="utf-8")
    revised_manuscript_path.write_text("# Manuscript\n\nOriginal.\n\nAdded statistical tests.", encoding="utf-8")

    second_status = progress_review_session(session_dir, reviewer_path, researcher_path)

    assert second_status == "next_reviewer_prompt_created"
    assert (session_dir / "round_02" / "reviewer_prompt.md").exists()
