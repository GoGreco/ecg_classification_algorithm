"""Visualiza o efeito de t sobre um mesmo batimento do conjunto de dados."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts._bootstrap import bootstrap_src_path

bootstrap_src_path()

from lstm_training import montar_dataset
from symbolic_lstm_training import simbolizar_janelas
from scripts.evaluate_lstm_repeated_binary import coarse_grain


def load_limits(path: Path) -> np.ndarray:
    table = pd.read_csv(path)
    return np.r_[table["lower_limit"].iloc[0], table["upper_limit"].to_numpy()]


def main() -> None:
    output_dir = Path("reports/symbolic_lstm_scale_sweep_binary_test")
    X, labels, _ = montar_dataset(Path("data/filtered"))
    # Mantém um exemplo normal e um não normal para ilustrar apenas a codificação.
    indices = [int(np.flatnonzero(labels == label)[0]) for label in ("N", "V")]
    scales = (2, 5, 10)
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharey="row")
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, 7))

    for row, index in enumerate(indices):
        for column, scale in enumerate(scales):
            continuous = coarse_grain(X[index : index + 1], scale)[0]
            limits = load_limits(
                output_dir
                / "partitions"
                / f"scale_{scale}"
                / "symbol_partition_limits_rep0.csv"
            )
            symbolic = simbolizar_janelas(continuous[None, :], limits)[0]
            axis = axes[row, column]
            axis.plot(np.arange(len(continuous)), continuous, color="#222222", linewidth=1.3)
            axis.step(np.arange(len(continuous)), symbolic / 6.0, where="mid", color="#d62728", alpha=0.75, linewidth=1.0)
            axis.scatter(np.arange(len(continuous)), symbolic / 6.0, c=symbolic, cmap="viridis", vmin=0, vmax=6, s=12)
            axis.set_title(f"classe {labels[index]} — t={scale} — L={len(continuous)}")
            axis.grid(alpha=0.2)
            axis.set_xlabel("posição agregada")
            if column == 0:
                axis.set_ylabel("amplitude / símbolo normalizado")

    fig.suptitle("Do ECG contínuo à sequência simbólica em diferentes escalas", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "scale_process_examples.png", dpi=160)
    plt.close(fig)
    print(output_dir / "scale_process_examples.png")


if __name__ == "__main__":
    main()
