from __future__ import annotations

import argparse
from pathlib import Path

try:
    from scripts._bootstrap import bootstrap_src_path
except ModuleNotFoundError:  # pragma: no cover
    from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from ecg_classification.agents.orchestration import create_review_session, progress_review_session


def main() -> None:
    parser = argparse.ArgumentParser(description="Gerencia ciclos de revisao com agentes definidos em Markdown.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("manuscript")
    init_parser.add_argument("--output-dir", required=True)
    init_parser.add_argument(
        "--reviewer",
        default="agents/high_impact_biomedical_signals_reviewer.md",
    )
    init_parser.add_argument(
        "--researcher",
        default="agents/nonlinear_dynamics_signal_researcher.md",
    )

    progress_parser = subparsers.add_parser("progress")
    progress_parser.add_argument("session_dir")
    progress_parser.add_argument(
        "--reviewer",
        default="agents/high_impact_biomedical_signals_reviewer.md",
    )
    progress_parser.add_argument(
        "--researcher",
        default="agents/nonlinear_dynamics_signal_researcher.md",
    )

    args = parser.parse_args()

    if args.command == "init":
        output_dir = create_review_session(
            manuscript_path=Path(args.manuscript),
            reviewer_path=Path(args.reviewer),
            researcher_path=Path(args.researcher),
            output_dir=Path(args.output_dir),
        )
        print(output_dir)
        return

    status = progress_review_session(
        session_dir=Path(args.session_dir),
        reviewer_path=Path(args.reviewer),
        researcher_path=Path(args.researcher),
    )
    print(status)


if __name__ == "__main__":
    main()
