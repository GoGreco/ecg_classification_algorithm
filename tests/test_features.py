import numpy as np

from ecg_classification.features.symbolic_dynamics import (
    dense_word_probability_vector,
    euclidean_probability_distance,
    minmax_normalize,
    probability_partition_limits,
    symbolic_feature_vector,
    symbolize,
    symbolize_with_partitions,
    word_probability_distribution,
)


def test_symbolize_length_matches_signal() -> None:
    signal = np.array([0.1, 0.2, -0.4, 0.8, 0.0])
    assert len(symbolize(signal, num_bins=3)) == len(signal)


def test_symbolic_feature_vector_has_expected_keys() -> None:
    features = symbolic_feature_vector(np.sin(np.linspace(0, 1, 50)))
    assert {"symbolic_entropy", "num_unique_words", "mean_symbol", "std_symbol"} <= set(features)


def test_probability_partition_limits_cover_signal_range() -> None:
    signal = minmax_normalize(np.array([4.0, 2.0, 8.0, 6.0, 10.0]))
    limits = probability_partition_limits(signal, num_partitions=3)
    symbols = symbolize_with_partitions(signal, limits)

    assert limits[0] == 0.0
    assert limits[-1] == 1.0
    assert set(symbols) <= {0, 1, 2}


def test_word_probability_distribution_and_distance() -> None:
    symbols = np.array([0, 1, 0, 1])
    distribution = word_probability_distribution(symbols, word_size=2)
    vector = dense_word_probability_vector(symbols, alphabet_size=2, word_size=2)

    assert distribution[(0, 1)] == 2 / 3
    assert np.isclose(vector.sum(), 1.0)
    assert euclidean_probability_distance(distribution, distribution) == 0.0
