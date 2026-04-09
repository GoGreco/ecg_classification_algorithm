from __future__ import annotations

import numpy as np


def detect_rpeaks(signal: np.ndarray, sampling_rate: int) -> np.ndarray:
    import neurokit2 as nk

    _, peaks = nk.ecg_peaks(signal, sampling_rate=sampling_rate)
    return np.asarray(peaks["ECG_R_Peaks"], dtype=int)
