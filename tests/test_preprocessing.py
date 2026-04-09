import numpy as np

from ecg_classification.preprocessing.normalization import minmax, zscore


def test_zscore_constant_signal_returns_zeros() -> None:
    signal = np.ones(10)
    assert np.allclose(zscore(signal), 0.0)


def test_minmax_rescales_range() -> None:
    signal = np.array([1.0, 2.0, 3.0])
    assert np.allclose(minmax(signal), np.array([0.0, 0.5, 1.0]))
