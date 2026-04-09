from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

from ecg_classification.features.labels import is_beat_label
from ecg_classification.features.morphology import beat_morphology_features
from ecg_classification.features.symbolic_dynamics import symbolic_feature_vector


def select_lead(record_frame: pd.DataFrame, preferred_leads: tuple[str, ...] = ("MLII", "V5", "V1", "V2")) -> str:
    for lead in preferred_leads:
        if lead in record_frame.columns:
            return lead
    return str(record_frame.columns[0])


def build_labeled_beat_dataset(
    record_frame: pd.DataFrame,
    annotation_frame: pd.DataFrame,
    record_id: str,
    sampling_rate: int,
    lead: str | None = None,
    left_size: int = 100,
    right_size: int = 150,
    label_transform: Callable[[str], str | None] | None = None,
) -> pd.DataFrame:
    selected_lead = lead or select_lead(record_frame)
    signal = record_frame[selected_lead].to_numpy()
    annotations = annotation_frame.copy()
    annotations.columns = annotations.columns.str.strip()
    annotations = annotations[["Sample", "Symbol"]].dropna().reset_index(drop=True)

    rows: list[dict[str, float | int | str]] = []
    kept_peaks: list[int] = []
    kept_labels: list[str] = []

    for _, row_data in annotations.iterrows():
        peak = int(row_data["Sample"])
        raw_label = str(row_data["Symbol"])
        if not is_beat_label(raw_label):
            continue

        label = label_transform(raw_label) if label_transform is not None else raw_label
        if label is None:
            continue

        start = peak - left_size
        end = peak + right_size
        if start < 0 or end > signal.size:
            continue

        kept_peaks.append(peak)
        kept_labels.append(label)

    for idx, (peak, label) in enumerate(zip(kept_peaks, kept_labels)):
        start = peak - left_size
        end = peak + right_size
        window = signal[start:end]
        row: dict[str, float | int | str] = {
            "record_id": record_id,
            "lead": selected_lead,
            "peak_index": peak,
            "label": label,
        }
        row.update(beat_morphology_features(window))
        row.update(symbolic_feature_vector(window))

        prev_rr = (peak - kept_peaks[idx - 1]) / sampling_rate if idx > 0 else 0.0
        next_rr = (kept_peaks[idx + 1] - peak) / sampling_rate if idx < len(kept_peaks) - 1 else 0.0
        row["rr_prev"] = float(prev_rr)
        row["rr_next"] = float(next_rr)
        rows.append(row)

    return pd.DataFrame(rows)


def build_dataset_from_csv_paths(
    record_paths: list[Path],
    annotation_dir: Path,
    sampling_rate: int,
    lead: str | None = None,
    label_transform: Callable[[str], str | None] | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for record_path in record_paths:
        record_id = record_path.stem.replace("_record", "")
        annotation_path = annotation_dir / f"{record_id}_annotation.csv"
        if not annotation_path.exists():
            continue
        record_frame = pd.read_csv(record_path)
        annotation_frame = pd.read_csv(annotation_path)
        beat_frame = build_labeled_beat_dataset(
            record_frame=record_frame,
            annotation_frame=annotation_frame,
            record_id=record_id,
            sampling_rate=sampling_rate,
            lead=lead,
            label_transform=label_transform,
        )
        if not beat_frame.empty:
            frames.append(beat_frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
