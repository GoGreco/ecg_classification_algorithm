from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import wfdb

from ecg_classification.config import ProjectPaths


def load_record_names(record_file_path: Path | None = None) -> list[str]:
    paths = ProjectPaths()
    record_file = record_file_path or paths.data_raw / "RECORDS"
    return record_file.read_text(encoding="utf-8").splitlines()


def read_signal(record_name: str, data_dir: Path | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    paths = ProjectPaths()
    root = data_dir or paths.data_raw
    signal, metadata = wfdb.rdsamp(str(root / record_name))
    frame = pd.DataFrame(signal, columns=metadata["sig_name"])
    return frame, metadata


def read_annotations(record_name: str, data_dir: Path | None = None, extension: str = "atr") -> pd.DataFrame:
    paths = ProjectPaths()
    root = data_dir or paths.data_raw
    annotations = wfdb.rdann(str(root / record_name), extension)
    return pd.DataFrame({"Sample": annotations.sample, "Symbol": annotations.symbol})


def read_metadata(record_name: str, data_dir: Path | None = None) -> dict[str, Any]:
    paths = ProjectPaths()
    root = data_dir or paths.data_raw
    header = wfdb.rdheader(str(root / record_name))
    return {
        "record_name": header.record_name,
        "sampling_frequency": header.fs,
        "signal_names": header.sig_name,
        "units": header.units,
        "num_samples": header.sig_len,
    }
