from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    output_dir = Path("reports/symbolic_lstm_scale5_binary_test")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, model, title in zip(
        axes,
        ("continuous", "symbolic"),
        ("LSTM contínua — 50 amplitudes", "LSTM simbólica — 50 símbolos"),
    ):
        matrix = pd.read_csv(output_dir / f"{model}_confusion_matrix.csv", index_col=0)
        values = matrix.to_numpy()
        image = axis.imshow(values, cmap="Blues")
        axis.set_xticks(range(2), ["N", "não-N"])
        axis.set_yticks(range(2), ["N", "não-N"])
        axis.set_xlabel("Predito")
        axis.set_ylabel("Real")
        axis.set_title(title)
        for row in range(2):
            for column in range(2):
                axis.text(
                    column,
                    row,
                    str(values[row, column]),
                    ha="center",
                    va="center",
                    color="white" if values[row, column] > values.max() / 2 else "black",
                )
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Quantidade")
    fig.suptitle("Matrizes de confusão — classificação N versus não-N")
    fig.tight_layout()
    fig.savefig(output_dir / "comparison_confusion_matrices.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
