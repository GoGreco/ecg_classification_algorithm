from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

try:
    from scripts._bootstrap import bootstrap_src_path
except ModuleNotFoundError:  # pragma: no cover
    from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from ecg_classification.config import ExperimentConfig, ProjectPaths
from ecg_classification.preprocessing.quality import estimate_ecg_nsr


NSR_COLUMNS = [
    "record_id",
    "lead",
    "n_samples",
    "sampling_rate",
    "lowcut_hz",
    "highcut_hz",
    "signal_power",
    "noise_power",
    "nsr",
    "nsr_db",
    "snr_db",
]


def read_numeric_columns(record_csv: Path) -> dict[str, np.ndarray]:
    with record_csv.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        values: dict[str, list[float]] = {field: [] for field in reader.fieldnames or []}
        for row in reader:
            for field in values:
                try:
                    values[field].append(float(row[field]))
                except (TypeError, ValueError):
                    continue
    return {field: np.asarray(column, dtype=float) for field, column in values.items()}


def estimate_record_nsr(
    record_csv: Path,
    sampling_rate: int,
    lead: str = "MLII",
    lowcut: float = 0.5,
    highcut: float = 40.0,
) -> list[dict[str, float | int | str]]:
    record_id = record_csv.stem.replace("_record", "")
    signal = read_numeric_columns(record_csv).get(lead)
    if signal is None or signal.size == 0:
        return []

    metrics = estimate_ecg_nsr(signal, sampling_rate=sampling_rate, lowcut=lowcut, highcut=highcut)
    return [
        {
            "record_id": record_id,
            "lead": lead,
            "n_samples": int(signal.size),
            "sampling_rate": int(sampling_rate),
            "lowcut_hz": float(lowcut),
            "highcut_hz": float(highcut),
            **metrics,
        }
    ]


def main() -> None:
    paths = ProjectPaths()
    config = ExperimentConfig()
    rows: list[dict[str, float | int | str]] = []
    skipped_records: list[str] = []

    for record_csv in sorted(paths.data_interim.glob("*_record.csv")):
        record_rows = estimate_record_nsr(
            record_csv,
            sampling_rate=config.sampling_rate,
            lead=config.default_lead,
        )
        if record_rows:
            rows.extend(record_rows)
        else:
            skipped_records.append(record_csv.stem.replace("_record", ""))

    destination = paths.reports / "signal_nsr.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=NSR_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"NSR metrics saved to {destination}.")
    if skipped_records:
        print(f"Skipped records without {config.default_lead}: {', '.join(skipped_records)}.")


if __name__ == "__main__":
    main()
