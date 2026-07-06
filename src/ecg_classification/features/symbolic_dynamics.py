from __future__ import annotations

from collections import Counter
from itertools import product
from math import log2

import numpy as np

from ecg_classification.preprocessing.normalization import zscore


def minmax_normalize(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=float)
    if signal.size == 0:
        return signal
    lower = float(signal.min())
    upper = float(signal.max())
    if np.isclose(upper, lower):
        return np.zeros_like(signal, dtype=float)
    return (signal - lower) / (upper - lower)


def probability_partition_limits(signal: np.ndarray, num_partitions: int) -> np.ndarray:
    if num_partitions < 2:
        raise ValueError("num_partitions must be at least 2.")
    values = np.sort(np.asarray(signal, dtype=float))
    if values.size == 0:
        raise ValueError("signal must not be empty.")

    limits = [float(values[0])]
    n_samples = values.size
    for partition_idx in range(1, num_partitions):
        position = partition_idx * n_samples / num_partitions
        lower_idx = int(np.floor(position))
        upper_idx = int(np.ceil(position))
        if lower_idx >= n_samples:
            limit = float(values[-1])
        elif lower_idx == upper_idx:
            limit = float(values[lower_idx])
        else:
            limit = float((values[lower_idx] + values[min(upper_idx, n_samples - 1)]) / 2.0)
        limits.append(limit)
    limits.append(float(values[-1]))
    return np.asarray(limits, dtype=float)


def symbolize_with_partitions(signal: np.ndarray, partition_limits: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=float)
    partition_limits = np.asarray(partition_limits, dtype=float)
    if partition_limits.ndim != 1 or partition_limits.size < 3:
        raise ValueError("partition_limits must define at least two partitions.")
    symbols = np.digitize(signal, partition_limits[1:-1], right=False)
    return np.clip(symbols, 0, partition_limits.size - 2).astype(int)


def symbol_entropy(symbols: np.ndarray) -> float:
    symbols = np.asarray(symbols, dtype=int)
    if symbols.size == 0:
        return 0.0
    counts = Counter(symbols.tolist())
    total = symbols.size
    return -sum((count / total) * log2(count / total) for count in counts.values())


def select_partition_count_by_entropy(
    reference_signal: np.ndarray,
    symbol_signal: np.ndarray | None = None,
    min_partitions: int = 2,
    max_partitions: int = 40,
    epsilon: float = 0.2,
) -> tuple[int, list[dict[str, float]]]:
    if min_partitions < 2:
        raise ValueError("min_partitions must be at least 2.")
    if max_partitions < min_partitions:
        raise ValueError("max_partitions must be greater than or equal to min_partitions.")

    reference_signal = np.asarray(reference_signal, dtype=float)
    signal_to_symbolize = np.asarray(symbol_signal if symbol_signal is not None else reference_signal, dtype=float)
    previous_entropy: float | None = None
    history: list[dict[str, float]] = []
    selected_partitions = min_partitions

    for num_partitions in range(min_partitions, max_partitions + 1):
        limits = probability_partition_limits(reference_signal, num_partitions=num_partitions)
        symbols = symbolize_with_partitions(signal_to_symbolize, limits)
        entropy = symbol_entropy(symbols)
        delta = 0.0 if previous_entropy is None else entropy - previous_entropy
        history.append(
            {
                "num_partitions": float(num_partitions),
                "entropy": float(entropy),
                "entropy_delta": float(delta),
            }
        )
        if previous_entropy is not None and delta < epsilon:
            selected_partitions = max(min_partitions, num_partitions - 1)
            break
        selected_partitions = num_partitions
        previous_entropy = entropy

    return selected_partitions, history


def word_probability_distribution(symbols: np.ndarray, word_size: int) -> dict[tuple[int, ...], float]:
    words = symbolic_words(np.asarray(symbols, dtype=int), word_size=word_size)
    if not words:
        return {}
    counts = Counter(words)
    total = len(words)
    return {word: count / total for word, count in counts.items()}


def dense_word_probability_vector(symbols: np.ndarray, alphabet_size: int, word_size: int) -> np.ndarray:
    distribution = word_probability_distribution(symbols, word_size=word_size)
    words = product(range(alphabet_size), repeat=word_size)
    return np.asarray([distribution.get(tuple(word), 0.0) for word in words], dtype=float)


def euclidean_probability_distance(
    first: dict[tuple[int, ...], float],
    second: dict[tuple[int, ...], float],
) -> float:
    words = set(first) | set(second)
    return float(np.sqrt(sum((first.get(word, 0.0) - second.get(word, 0.0)) ** 2 for word in words)))


def build_partitions(signal: np.ndarray, num_bins: int = 4) -> np.ndarray:
    normalized = zscore(signal)
    return np.linspace(normalized.min(), normalized.max(), num_bins + 1)


def symbolize(signal: np.ndarray, num_bins: int = 4) -> np.ndarray:
    partitions = build_partitions(signal, num_bins=num_bins)
    normalized = zscore(signal)
    symbols = np.digitize(normalized, partitions[1:-1], right=False)
    return symbols.astype(int)


def symbolic_words(symbols: np.ndarray, word_size: int = 3) -> list[tuple[int, ...]]:
    return [tuple(symbols[idx : idx + word_size]) for idx in range(0, len(symbols) - word_size + 1)]


def shannon_entropy(words: list[tuple[int, ...]]) -> float:
    if not words:
        return 0.0
    counts = Counter(words)
    total = sum(counts.values())
    return -sum((count / total) * log2(count / total) for count in counts.values())


def symbolic_feature_vector(signal: np.ndarray, num_bins: int = 4, word_size: int = 3) -> dict[str, float]:
    symbols = symbolize(signal, num_bins=num_bins)
    words = symbolic_words(symbols, word_size=word_size)
    return {
        "symbolic_entropy": shannon_entropy(words),
        "num_unique_words": float(len(set(words))),
        "mean_symbol": float(symbols.mean()) if len(symbols) else 0.0,
        "std_symbol": float(symbols.std()) if len(symbols) else 0.0,
    }
