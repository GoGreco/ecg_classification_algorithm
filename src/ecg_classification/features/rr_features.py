from __future__ import annotations

import numpy as np


def rr_intervals(rpeaks: np.ndarray, sampling_rate: int) -> np.ndarray:
    rpeaks = np.asarray(rpeaks)
    if rpeaks.size < 2:
        return np.array([], dtype=float)
    return np.diff(rpeaks) / sampling_rate


def rr_summary(rpeaks: np.ndarray, sampling_rate: int) -> dict[str, float]:
    intervals = rr_intervals(rpeaks, sampling_rate)
    if intervals.size == 0:
        return {"rr_mean": 0.0, "rr_std": 0.0}
    return {"rr_mean": float(intervals.mean()), "rr_std": float(intervals.std())}
