from __future__ import annotations

from pathlib import Path
import shutil

from ecg_classification.agents.loader import load_agent_spec
from ecg_classification.agents.schema import AgentSpec

PLACEHOLDER_REVIEWER_REPORT = """# Parecer do revisor

Substitua este arquivo pelo parecer gerado pelo agente revisor.
O parecer deve conter, no minimo:

- resumo executivo da decisao editorial;
- lista priorizada de problemas metodologicos;
- critica detalhada de resultados, ablaçoes, estatistica e generalizacao;
- demandas objetivas de revisao para o pesquisador.
"""

PLACEHOLDER_RESEARCHER_RESPONSE = """# Resposta do pesquisador

Substitua este arquivo pela resposta do agente pesquisador.
A resposta deve conter, no minimo:

- plano de acao para cada critica do revisor;
- alteracoes efetuadas no manuscrito;
- novas analises, testes ou experimentos;
- limitacoes remanescentes e justificativas tecnicas.
"""


def create_review_session(
    manuscript_path: str | Path,
    reviewer_path: str | Path,
    researcher_path: str | Path,
    output_dir: str | Path,
) -> Path:
    manuscript_path = Path(manuscript_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reviewer = load_agent_spec(reviewer_path)
    researcher = load_agent_spec(researcher_path)

    current_manuscript_path = output_dir / "current_manuscript.md"
    shutil.copyfile(manuscript_path, current_manuscript_path)

    round_dir = output_dir / "round_01"
    round_dir.mkdir(exist_ok=True)
    reviewer_prompt = build_reviewer_prompt(
        reviewer=reviewer,
        researcher=researcher,
        manuscript_text=current_manuscript_path.read_text(encoding="utf-8"),
        round_number=1,
    )
    (round_dir / "reviewer_prompt.md").write_text(reviewer_prompt, encoding="utf-8")
    (round_dir / "reviewer_report.md").write_text(PLACEHOLDER_REVIEWER_REPORT, encoding="utf-8")
    return output_dir


def progress_review_session(
    session_dir: str | Path,
    reviewer_path: str | Path,
    researcher_path: str | Path,
) -> str:
    session_dir = Path(session_dir)
    reviewer = load_agent_spec(reviewer_path)
    researcher = load_agent_spec(researcher_path)

    round_dir = _latest_round_dir(session_dir)
    reviewer_report_path = round_dir / "reviewer_report.md"
    researcher_prompt_path = round_dir / "researcher_prompt.md"
    researcher_response_path = round_dir / "researcher_response.md"
    revised_manuscript_path = round_dir / "revised_manuscript.md"

    if _is_completed(reviewer_report_path, PLACEHOLDER_REVIEWER_REPORT) and not researcher_prompt_path.exists():
        manuscript_text = (session_dir / "current_manuscript.md").read_text(encoding="utf-8")
        reviewer_report = reviewer_report_path.read_text(encoding="utf-8")
        researcher_prompt = build_researcher_prompt(
            researcher=researcher,
            reviewer=reviewer,
            manuscript_text=manuscript_text,
            reviewer_report=reviewer_report,
            round_number=_round_number(round_dir),
        )
        researcher_prompt_path.write_text(researcher_prompt, encoding="utf-8")
        researcher_response_path.write_text(PLACEHOLDER_RESEARCHER_RESPONSE, encoding="utf-8")
        revised_manuscript_path.write_text(manuscript_text, encoding="utf-8")
        return "researcher_prompt_created"

    if (
        researcher_prompt_path.exists()
        and _is_completed(researcher_response_path, PLACEHOLDER_RESEARCHER_RESPONSE)
        and _is_completed(revised_manuscript_path, "")
    ):
        shutil.copyfile(revised_manuscript_path, session_dir / "current_manuscript.md")
        next_round_number = _round_number(round_dir) + 1
        next_round_dir = session_dir / f"round_{next_round_number:02d}"
        next_round_dir.mkdir(exist_ok=True)
        reviewer_prompt = build_reviewer_prompt(
            reviewer=reviewer,
            researcher=researcher,
            manuscript_text=revised_manuscript_path.read_text(encoding="utf-8"),
            round_number=next_round_number,
            prior_report=reviewer_report_path.read_text(encoding="utf-8"),
            prior_response=researcher_response_path.read_text(encoding="utf-8"),
        )
        (next_round_dir / "reviewer_prompt.md").write_text(reviewer_prompt, encoding="utf-8")
        (next_round_dir / "reviewer_report.md").write_text(PLACEHOLDER_REVIEWER_REPORT, encoding="utf-8")
        return "next_reviewer_prompt_created"

    return "no_action"


def build_reviewer_prompt(
    reviewer: AgentSpec,
    researcher: AgentSpec,
    manuscript_text: str,
    round_number: int,
    prior_report: str | None = None,
    prior_response: str | None = None,
) -> str:
    sections = [
        f"# Ciclo de revisao {round_number:02d}",
        "",
        "## Agente ativo",
        _render_agent(reviewer),
        "",
        "## Objetivo da rodada",
        (
            "Atue como revisor de revista de alto fator de impacto em sinais biomédicos. "
            f"Seu parecer sera encaminhado ao agente pesquisador `{researcher.name}`, "
            "que devera executar as correcoes propostas."
        ),
    ]

    if prior_report:
        sections.extend(["", "## Parecer anterior", prior_report.strip()])
    if prior_response:
        sections.extend(["", "## Resposta anterior do pesquisador", prior_response.strip()])

    sections.extend(
        [
            "",
            "## Manuscrito atual",
            manuscript_text.strip(),
            "",
            "## Formato exigido do parecer",
            "- decisao editorial provisoria;",
            "- principais falhas metodologicas e experimentais;",
            "- critica especifica sobre resultados, significancia estatistica, comparadores e reproducibilidade;",
            "- exigencias de revisao acionaveis para o pesquisador;",
            "- lista de verificacoes obrigatorias antes de recomendacao positiva.",
        ]
    )
    return "\n".join(sections).strip() + "\n"


def build_researcher_prompt(
    researcher: AgentSpec,
    reviewer: AgentSpec,
    manuscript_text: str,
    reviewer_report: str,
    round_number: int,
) -> str:
    return (
        f"# Resposta ao ciclo de revisao {round_number:02d}\n\n"
        "## Agente ativo\n"
        f"{_render_agent(researcher)}\n\n"
        "## Objetivo da rodada\n"
        f"Atue como pesquisador sênior. Responda ao parecer do agente `{reviewer.name}` "
        "com rigor técnico e implemente no manuscrito todas as alteracoes justificadas.\n\n"
        "## Manuscrito atual\n"
        f"{manuscript_text.strip()}\n\n"
        "## Parecer do revisor\n"
        f"{reviewer_report.strip()}\n\n"
        "## Entregaveis obrigatorios\n"
        "- resposta ponto a ponto ao parecer;\n"
        "- manuscrito revisado com correcoes incorporadas;\n"
        "- justificativa explicita para cada ponto aceito, recusado ou parcialmente atendido;\n"
        "- novos experimentos, controles, testes estatisticos ou analises adicionais quando necessarios.\n"
    )


def _render_agent(agent: AgentSpec) -> str:
    lines = [
        f"- nome: {agent.name}",
        f"- papel: {agent.role}",
        f"- objetivo: {agent.objective}",
    ]
    if agent.model:
        lines.append(f"- modelo sugerido: {agent.model}")
    if agent.temperature is not None:
        lines.append(f"- temperatura sugerida: {agent.temperature}")
    if agent.tools:
        lines.append(f"- ferramentas: {', '.join(agent.tools)}")
    lines.extend(["", "## Instrucoes do agente", agent.system_prompt.strip()])
    return "\n".join(lines)


def _latest_round_dir(session_dir: Path) -> Path:
    rounds = sorted(path for path in session_dir.iterdir() if path.is_dir() and path.name.startswith("round_"))
    if not rounds:
        raise FileNotFoundError("Nenhuma rodada encontrada na sessao.")
    return rounds[-1]


def _round_number(round_dir: Path) -> int:
    return int(round_dir.name.split("_")[1])


def _is_completed(path: Path, placeholder: str) -> bool:
    if not path.exists():
        return False
    content = path.read_text(encoding="utf-8").strip()
    placeholder_content = placeholder.strip()
    if not content:
        return False
    if placeholder_content and content == placeholder_content:
        return False
    return True
