from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_signal(signal: np.ndarray, sampling_rate: int, title: str) -> None:
    time = np.arange(len(signal)) / sampling_rate
    plt.figure(figsize=(12, 4))
    plt.plot(time, signal, linewidth=1)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
