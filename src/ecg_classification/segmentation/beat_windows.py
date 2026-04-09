from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def match_annotation(peak_idx: int, annotations: dict[int, str], tolerance: int = 15) -> str | None:
    for idx in range(peak_idx - tolerance, peak_idx + tolerance + 1):
        if idx in annotations:
            return annotations[idx]
    return None


def annotations_to_dict(annotation_frame: pd.DataFrame) -> dict[int, str]:
    frame = annotation_frame.copy()
    frame.columns = frame.columns.str.strip()
    return {int(row["Sample"]): str(row["Symbol"]) for _, row in frame.iterrows()}


def build_beat_windows(
    signal: np.ndarray,
    rpeaks: Iterable[int],
    left_size: int = 100,
    right_size: int = 150,
) -> list[np.ndarray]:
    signal = np.asarray(signal)
    windows: list[np.ndarray] = []
    for peak in rpeaks:
        start = peak - left_size
        end = peak + right_size
        if start < 0 or end > signal.size:
            continue
        windows.append(signal[start:end])
    return windows
