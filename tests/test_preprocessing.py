import numpy as np

from ecg_classification.preprocessing.normalization import minmax, zscore
from ecg_classification.preprocessing.quality import estimate_ecg_nsr


def test_zscore_constant_signal_returns_zeros() -> None:
    signal = np.ones(10)
    assert np.allclose(zscore(signal), 0.0)


def test_minmax_rescales_range() -> None:
    signal = np.array([1.0, 2.0, 3.0])
    assert np.allclose(minmax(signal), np.array([0.0, 0.5, 1.0]))


def test_estimate_ecg_nsr_reports_noise_and_signal_power() -> None:
    sampling_rate = 360
    time = np.arange(0, 8, 1 / sampling_rate)
    clean_signal = np.sin(2 * np.pi * 5 * time)
    high_frequency_noise = 0.5 * np.sin(2 * np.pi * 80 * time)
    metrics = estimate_ecg_nsr(clean_signal + high_frequency_noise, sampling_rate=sampling_rate)

    assert metrics["signal_power"] > 0
    assert metrics["noise_power"] > 0
    assert metrics["nsr"] > 0
    assert np.isclose(metrics["snr_db"], -metrics["nsr_db"])
