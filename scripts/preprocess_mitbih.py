from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from scripts._bootstrap import bootstrap_src_path
except ModuleNotFoundError:  # pragma: no cover
    from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from ecg_classification.config import ExperimentConfig, ProjectPaths
from ecg_classification.preprocessing.baseline import remove_baseline_wander
from ecg_classification.preprocessing.filtering import clean_ecg


def select_preprocessing_lead(frame: pd.DataFrame, preferred_lead: str = "MLII") -> str:
    if preferred_lead in frame.columns:
        return preferred_lead
    return str(frame.columns[0])


def preprocess_record(record_csv: Path, sampling_rate: int, preferred_lead: str = "MLII") -> pd.DataFrame:
    frame = pd.read_csv(record_csv)
    lead = select_preprocessing_lead(frame, preferred_lead=preferred_lead)
    cleaned = clean_ecg(frame[lead].to_numpy(), sampling_rate=sampling_rate)
    corrected, baseline = remove_baseline_wander(cleaned, sampling_rate=sampling_rate)
    return pd.DataFrame(
        {
            "raw": frame[lead].to_numpy(),
            "cleaned": cleaned,
            "baseline": baseline,
            "baseline_corrected": corrected,
        }
    )


def main() -> None:
    paths = ProjectPaths()
    config = ExperimentConfig()
    paths.data_processed.mkdir(parents=True, exist_ok=True)
    for record_csv in sorted(paths.data_interim.glob("*_record.csv")):
        processed = preprocess_record(
            record_csv,
            sampling_rate=config.sampling_rate,
            preferred_lead=config.default_lead,
        )
        processed.to_csv(paths.data_processed / f"{record_csv.stem}_processed.csv", index=False)
    print("Processed signals saved to data/processed.")


if __name__ == "__main__":
    main()
