from pathlib import Path

from ecg_classification.config import ProjectPaths


def test_project_paths_derive_expected_directories(tmp_path: Path) -> None:
    paths = ProjectPaths(root=tmp_path)

    assert paths.docs == tmp_path / "docs"
    assert paths.data_raw == tmp_path / "data" / "raw" / "mit_bih"
    assert paths.data_interim == tmp_path / "data" / "interim" / "signal_tables"
    assert paths.data_processed == tmp_path / "data" / "processed"
    assert paths.reports == tmp_path / "reports"
    assert paths.experiments == tmp_path / "experiments"


def test_project_paths_ensure_structure_creates_directories(tmp_path: Path) -> None:
    paths = ProjectPaths(root=tmp_path)

    paths.ensure_structure()

    assert paths.docs.exists()
    assert paths.data_raw.exists()
    assert paths.data_interim.exists()
    assert paths.data_processed.exists()
    assert paths.reports.exists()
    assert paths.experiments.exists()
