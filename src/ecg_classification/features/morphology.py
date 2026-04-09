from __future__ import annotations

import numpy as np


def beat_morphology_features(window: np.ndarray) -> dict[str, float]:
    window = np.asarray(window, dtype=float)
    return {
        "morph_mean": float(window.mean()),
        "morph_std": float(window.std()),
        "morph_max": float(window.max()),
        "morph_min": float(window.min()),
        "morph_peak_to_peak": float(np.ptp(window)),
    }
