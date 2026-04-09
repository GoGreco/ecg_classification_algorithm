from __future__ import annotations

from pathlib import Path

import pandas as pd

from ecg_classification.config import ProjectPaths
from ecg_classification.io.load_wfdb import load_record_names, read_annotations, read_signal


def export_dataframe(dataframe: pd.DataFrame, destination: Path, overwrite: bool = False) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        return destination
    dataframe.to_csv(destination, index=False)
    return destination


def export_record_csv(record_name: str, output_dir: Path | None = None, overwrite: bool = False) -> tuple[Path, Path]:
    paths = ProjectPaths()
    target_dir = output_dir or paths.data_interim
    signal_frame, _ = read_signal(record_name)
    annotation_frame = read_annotations(record_name)
    record_path = export_dataframe(signal_frame, target_dir / f"{record_name}_record.csv", overwrite=overwrite)
    annotation_path = export_dataframe(annotation_frame, target_dir / f"{record_name}_annotation.csv", overwrite=overwrite)
    return record_path, annotation_path


def export_all_records(output_dir: Path | None = None, overwrite: bool = False) -> list[tuple[Path, Path]]:
    results: list[tuple[Path, Path]] = []
    for record_name in load_record_names():
        results.append(export_record_csv(record_name, output_dir=output_dir, overwrite=overwrite))
    return results
