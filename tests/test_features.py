import numpy as np

from ecg_classification.features.symbolic_dynamics import symbolic_feature_vector, symbolize


def test_symbolize_length_matches_signal() -> None:
    signal = np.array([0.1, 0.2, -0.4, 0.8, 0.0])
    assert len(symbolize(signal, num_bins=3)) == len(signal)


def test_symbolic_feature_vector_has_expected_keys() -> None:
    features = symbolic_feature_vector(np.sin(np.linspace(0, 1, 50)))
    assert {"symbolic_entropy", "num_unique_words", "mean_symbol", "std_symbol"} <= set(features)
