from __future__ import annotations

import numpy as np


def signal_quality_metrics(signal: np.ndarray) -> dict[str, float]:
    signal = np.asarray(signal, dtype=float)
    diff = np.diff(signal) if signal.size > 1 else np.array([0.0])
    return {
        "mean": float(signal.mean()),
        "std": float(signal.std()),
        "peak_to_peak": float(np.ptp(signal)),
        "derivative_std": float(diff.std()),
        "clipping_ratio": float(np.mean(np.isclose(signal, signal.min()) | np.isclose(signal, signal.max()))),
    }
