from __future__ import annotations

import numpy as np


def bessel_highcut_filter(signal: np.ndarray, highcut: float = 3.0, order: int = 5) -> np.ndarray:
    import neurokit2 as nk

    return np.asarray(nk.signal_filter(signal, highcut=highcut, method="bessel", order=order))


def clean_ecg(signal: np.ndarray, sampling_rate: int, method: str = "neurokit") -> np.ndarray:
    import neurokit2 as nk

    return np.asarray(nk.ecg_clean(signal, sampling_rate=sampling_rate, method=method))


def smooth_signal(signal: np.ndarray, window: int = 20) -> np.ndarray:
    if window <= 1:
        return np.asarray(signal)
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(np.asarray(signal), kernel, mode="same")
