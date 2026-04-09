from __future__ import annotations

import math

import numpy as np


def delineate_ecg(signal: np.ndarray, rpeaks: np.ndarray, sampling_rate: int, method: str = "dwt") -> dict[str, list[int]]:
    import neurokit2 as nk

    _, waves = nk.ecg_delineate(signal, rpeaks, sampling_rate=sampling_rate, method=method)
    delineated: dict[str, list[int]] = {}
    for key, values in waves.items():
        delineated[key] = [int(value) for value in values if not math.isnan(value)]
    return delineated
