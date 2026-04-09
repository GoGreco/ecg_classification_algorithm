from __future__ import annotations

from collections import Counter
from math import log2

import numpy as np

from ecg_classification.preprocessing.normalization import zscore


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
