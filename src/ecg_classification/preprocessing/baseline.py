from __future__ import annotations

import numpy as np
import pywt


def add_synthetic_baseline_wander(
    signal: np.ndarray,
    sampling_rate: int,
    baseline_frequency: float = 0.3,
    baseline_amplitude: float = 0.5,
    noise_amplitude: float = 0.08,
) -> tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(signal)
    time = np.arange(signal.size) / sampling_rate
    contamination = baseline_amplitude * np.sin(2 * np.pi * baseline_frequency * time)
    contamination += noise_amplitude * np.random.randn(signal.size)
    return signal + contamination, contamination


def remove_baseline_wander(
    signal: np.ndarray,
    sampling_rate: int,
    cutoff_frequency: float = 0.5,
    wavelet: str = "sym10",
) -> tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(signal)
    target_level = int(np.ceil(np.log2(sampling_rate / cutoff_frequency)))
    max_level = pywt.dwt_max_level(signal.size, pywt.Wavelet(wavelet).dec_len)
    level = min(target_level, max_level)
    coefficients = pywt.wavedec(signal, wavelet, level=level)
    approximation_only = [coefficients[0]] + [np.zeros_like(c) for c in coefficients[1:]]
    baseline = pywt.waverec(approximation_only, wavelet)[: signal.size]
    corrected = signal - baseline
    return corrected, baseline
