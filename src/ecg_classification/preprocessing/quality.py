from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


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


def bandpass_filter_ecg(
    ecg: np.ndarray,
    sampling_rate: float,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    ecg = np.asarray(ecg, dtype=float)
    if ecg.ndim != 1:
        raise ValueError("ECG signal must be one-dimensional.")
    if ecg.size == 0:
        raise ValueError("ECG signal must not be empty.")
    nyquist = sampling_rate / 2.0
    if not 0 < lowcut < highcut < nyquist:
        raise ValueError("Expected 0 < lowcut < highcut < sampling_rate / 2.")

    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return np.asarray(filtfilt(b, a, ecg))


def estimate_ecg_nsr(
    ecg: np.ndarray,
    sampling_rate: float,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    order: int = 4,
) -> dict[str, float]:
    ecg = np.asarray(ecg, dtype=float)
    filtered = bandpass_filter_ecg(
        ecg,
        sampling_rate=sampling_rate,
        lowcut=lowcut,
        highcut=highcut,
        order=order,
    )
    estimated_noise = ecg - filtered

    signal_power = float(np.mean(filtered**2))
    noise_power = float(np.mean(estimated_noise**2))
    nsr = noise_power / signal_power if signal_power > 0 else np.inf
    nsr_db = float(10 * np.log10(nsr)) if nsr > 0 else -np.inf
    snr_db = -nsr_db

    return {
        "signal_power": signal_power,
        "noise_power": noise_power,
        "nsr": float(nsr),
        "nsr_db": float(nsr_db),
        "snr_db": float(snr_db),
    }
