from pathlib import Path

from ecg_classification.utils.paths import resolve_annotation_csv, resolve_record_csv


def test_resolve_record_csv_uses_expected_suffix(tmp_path: Path) -> None:
    assert resolve_record_csv("100", base_dir=tmp_path) == tmp_path / "100_record.csv"


def test_resolve_annotation_csv_uses_expected_suffix(tmp_path: Path) -> None:
    assert resolve_annotation_csv("100", base_dir=tmp_path) == tmp_path / "100_annotation.csv"
