from __future__ import annotations

from pathlib import Path

from ecg_classification.config import ProjectPaths


def get_project_paths() -> ProjectPaths:
    return ProjectPaths()


def resolve_record_csv(record_id: str, base_dir: Path | None = None) -> Path:
    paths = get_project_paths()
    directory = base_dir or paths.data_interim
    return directory / f"{record_id}_record.csv"


def resolve_annotation_csv(record_id: str, base_dir: Path | None = None) -> Path:
    paths = get_project_paths()
    directory = base_dir or paths.data_interim
    return directory / f"{record_id}_annotation.csv"
