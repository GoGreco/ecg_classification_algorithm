"""Gera gráficos comparando uma janela de ECG com sua sequência simbólica.

Os limites usados são lidos de ``reports/symbolic_lstm`` para garantir que a
visualização reproduza exatamente a simbolização usada no treinamento.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts._bootstrap import bootstrap_src_path

bootstrap_src_path()

from ecg_classification.features.labels import map_to_aami
from symbolic_lstm_training import normalizar_janelas_minmax, simbolizar_janelas


LEFT = 100
RIGHT = 150
WINDOW_SIZE = LEFT + RIGHT
CLASS_ORDER = ("N", "S", "V", "F", "Q")
COLORS = ("#313695", "#74add1", "#fee090", "#f46d43", "#a50026", "#762a83", "#1b7837")


def load_limits(path: Path) -> np.ndarray:
    table = pd.read_csv(path)
    return np.r_[table["lower_limit"].iloc[0], table["upper_limit"].to_numpy()]


def find_examples(filtered_dir: Path) -> dict[str, tuple[str, int, str, np.ndarray]]:
    examples: dict[str, tuple[str, int, str, np.ndarray]] = {}
    for csv_path in sorted(filtered_dir.glob("*_filtered.csv")):
        frame = pd.read_csv(csv_path)
        signal = frame["amplitude_filtered"].to_numpy(dtype="float32")
        annotations = frame["annotation"]
        for peak, raw_label in annotations[annotations.notna() & (annotations != "")].items():
            aami = map_to_aami(str(raw_label))
            if aami not in CLASS_ORDER or aami in examples:
                continue
            if peak < LEFT or peak + RIGHT > len(signal):
                continue
            examples[aami] = (
                csv_path.stem.removesuffix("_filtered"),
                int(peak),
                str(raw_label),
                signal[peak - LEFT : peak + RIGHT],
            )
        if len(examples) == len(CLASS_ORDER):
            break
    return examples


def plot_comparison(
    signal: np.ndarray,
    symbols: np.ndarray,
    record_id: str,
    peak: int,
    raw_label: str,
    aami_label: str,
    output_path: Path,
) -> None:
    x = np.arange(WINDOW_SIZE)
    fig, (signal_ax, symbol_ax) = plt.subplots(
        2,
        1,
        figsize=(14, 6),
        sharex=True,
        gridspec_kw={"height_ratios": (3, 1)},
    )

    normalized = normalizar_janelas_minmax(signal[None, :])[0]
    signal_ax.plot(x, normalized, color="#222222", linewidth=1.4)
    signal_ax.axvline(LEFT, color="#d73027", linestyle="--", linewidth=1, label="batimento anotado")
    signal_ax.set_ylabel("Amplitude normalizada")
    signal_ax.set_title(
        f"Registro {record_id} — anotação {raw_label} / classe AAMI {aami_label} — pico {peak}"
    )
    signal_ax.grid(alpha=0.2)
    signal_ax.legend(loc="upper right")

    symbol_ax.step(np.arange(WINDOW_SIZE + 1), np.r_[symbols, symbols[-1]], where="post", color="#222222")
    symbol_ax.scatter(x + 0.5, symbols, c=[COLORS[int(symbol)] for symbol in symbols], s=9, zorder=3)
    symbol_ax.set_yticks(range(len(COLORS)))
    symbol_ax.set_ylabel("Símbolo")
    symbol_ax.set_xlabel("Amostras na janela (pico em 100)")
    symbol_ax.set_ylim(-0.5, len(COLORS) - 0.5)
    symbol_ax.set_xlim(0, WINDOW_SIZE)
    symbol_ax.grid(axis="y", alpha=0.2)
    for symbol in range(len(COLORS)):
        symbol_ax.axhspan(symbol - 0.5, symbol + 0.5, color=COLORS[symbol], alpha=0.08)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filtered-dir", type=Path, default=Path("data/filtered"))
    parser.add_argument(
        "--limits",
        type=Path,
        default=Path("reports/symbolic_lstm/symbol_partition_limits.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/symbolic_lstm/figures"),
    )
    args = parser.parse_args()

    limits = load_limits(args.limits)
    examples = find_examples(args.filtered_dir)
    if len(examples) != len(CLASS_ORDER):
        missing = sorted(set(CLASS_ORDER) - set(examples))
        raise RuntimeError(f"Não foi possível encontrar exemplos das classes: {missing}")

    summary_path = args.output_dir / "symbolic_comparison_summary.png"
    summary_fig, summary_axes = plt.subplots(
        len(CLASS_ORDER), 2, figsize=(14, 13), gridspec_kw={"width_ratios": (3, 1)}
    )

    for row, aami_label in enumerate(CLASS_ORDER):
        record_id, peak, raw_label, signal = examples[aami_label]
        symbols = simbolizar_janelas(normalizar_janelas_minmax(signal[None, :]), limits)[0]
        output_path = args.output_dir / f"symbolic_comparison_{aami_label}_{record_id}_{peak}.png"
        plot_comparison(signal, symbols, record_id, peak, raw_label, aami_label, output_path)

        normalized = normalizar_janelas_minmax(signal[None, :])[0]
        summary_axes[row, 0].plot(np.arange(WINDOW_SIZE), normalized, color="#222222", linewidth=1)
        summary_axes[row, 0].axvline(LEFT, color="#d73027", linestyle="--", linewidth=0.8)
        summary_axes[row, 0].set_ylabel(aami_label)
        summary_axes[row, 0].grid(alpha=0.2)
        summary_axes[row, 1].step(np.arange(WINDOW_SIZE + 1), np.r_[symbols, symbols[-1]], where="post", color="#222222")
        summary_axes[row, 1].set_yticks(range(len(COLORS)))
        summary_axes[row, 1].set_ylim(-0.5, len(COLORS) - 0.5)
        summary_axes[row, 1].grid(axis="y", alpha=0.2)

    summary_axes[0, 0].set_title("ECG normalizado")
    summary_axes[0, 1].set_title("Sequência simbólica")
    summary_axes[-1, 0].set_xlabel("Amostras na janela (pico em 100)")
    summary_axes[-1, 1].set_xlabel("Amostras na janela")
    summary_fig.suptitle("Comparação visual: ECG contínuo versus simbolização", fontsize=14)
    summary_fig.tight_layout()
    summary_fig.savefig(summary_path, dpi=160)
    plt.close(summary_fig)

    print(f"Resumo salvo em {summary_path}")
    for aami_label in CLASS_ORDER:
        print(f"{aami_label}: {args.output_dir / f'symbolic_comparison_{aami_label}_{examples[aami_label][0]}_{examples[aami_label][1]}.png'}")


if __name__ == "__main__":
    main()
