from __future__ import annotations

import numpy as np


def zscore(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=float)
    std = signal.std()
    if std == 0:
        return np.zeros_like(signal)
    return (signal - signal.mean()) / std


def minmax(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=float)
    minimum = signal.min()
    maximum = signal.max()
    if maximum == minimum:
        return np.zeros_like(signal)
    return (signal - minimum) / (maximum - minimum)
