"""Testa codificações simbólicas temporais alternativas.

Estratégias:

1. ``grouped``: run-length encoding dos símbolos consecutivos. A sequência é
   reduzida no eixo temporal, mas pode ser reconstruída exatamente.
2. ``extrema_direction``: representa a janela por segmentos crescentes ou
   decrescentes e pelos máximos/mínimos locais relevantes.

Os resultados são gravados em ``reports/symbolic_lstm_test``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from scripts._bootstrap import bootstrap_src_path

bootstrap_src_path()

from ecg_classification.features.labels import map_to_aami
from symbolic_lstm_training import normalizar_janelas_minmax, simbolizar_janelas


LEFT = 100
RIGHT = 150
WINDOW_SIZE = LEFT + RIGHT
CLASS_ORDER = ("N", "S", "V", "F", "Q")


def load_limits(path: Path) -> np.ndarray:
    table = pd.read_csv(path)
    return np.r_[table["lower_limit"].iloc[0], table["upper_limit"].to_numpy()]


def run_length_encode(symbols: np.ndarray) -> tuple[str, str, int]:
    """Retorna ``símbolo:comprimento`` e a sequência de símbolos agrupados."""
    if len(symbols) == 0:
        return "", "", 0
    values: list[int] = []
    lengths: list[int] = []
    current = int(symbols[0])
    length = 1
    for symbol in symbols[1:]:
        symbol = int(symbol)
        if symbol == current:
            length += 1
        else:
            values.append(current)
            lengths.append(length)
            current, length = symbol, 1
    values.append(current)
    lengths.append(length)
    grouped = " ".join(map(str, values))
    runs = "|".join(f"{symbol}:{length}" for symbol, length in zip(values, lengths))
    return runs, grouped, len(values)


def coarse_grain_and_symbolize(
    signal: np.ndarray,
    limits: np.ndarray,
    scale: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Agrega a amplitude contínua em blocos e simboliza depois.

    A agregação é feita antes da discretização para evitar operações
    aritméticas sem interpretação física sobre os símbolos categóricos.
    """
    normalized = normalizar_janelas_minmax(signal[None, :])[0]
    if len(normalized) % scale:
        raise ValueError("O tamanho da janela deve ser divisível pela escala.")
    coarse_signal = normalized.reshape(-1, scale).mean(axis=1)
    symbols = simbolizar_janelas(coarse_signal[None, :], limits)[0]
    return coarse_signal, symbols


def extrema_direction_encoding(
    signal: np.ndarray,
    prominence: float = 0.03,
    distance: int = 5,
) -> tuple[str, str, list[int]]:
    """Codifica segmentos direcionais e extremos locais.

    A sequência contém ``I`` (início), ``U``/``D`` para segmentos ascendentes
    ou descendentes e ``M``/``m`` para máximos/mínimos. Os extremos são
    detectados com distância e proeminência mínimas para evitar codificar
    pequenas oscilações residuais como eventos independentes.
    """
    maxima, _ = find_peaks(signal, prominence=prominence, distance=distance)
    minima, _ = find_peaks(-signal, prominence=prominence, distance=distance)
    extrema = sorted([(int(index), "M") for index in maxima] + [(int(index), "m") for index in minima])
    if not extrema:
        direction = "U" if signal[-1] >= signal[0] else "D"
        return f"I {direction} E", f"0 {len(signal) - 1}", []

    tokens = ["I"]
    positions: list[int] = []
    previous_index = 0
    previous_value = float(signal[0])
    for index, marker in extrema:
        if index <= previous_index:
            continue
        direction = "U" if signal[index] >= previous_value else "D"
        tokens.extend((direction, marker))
        positions.append(index)
        previous_index = index
        previous_value = float(signal[index])
    direction = "U" if signal[-1] >= previous_value else "D"
    tokens.extend((direction, "E"))
    return " ".join(tokens), " ".join(map(str, positions)), positions


def plot_example(
    signal: np.ndarray,
    original_symbols: np.ndarray,
    block5_symbols: np.ndarray,
    grouped_values: list[int],
    extrema_positions: list[int],
    record_id: str,
    peak: int,
    label: str,
    output_path: Path,
) -> None:
    normalized = normalizar_janelas_minmax(signal[None, :])[0]
    x = np.arange(WINDOW_SIZE)
    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=False)
    axes[0].plot(x, normalized, color="#222222")
    axes[0].axvline(LEFT, color="#d73027", linestyle="--", linewidth=1)
    axes[0].set_ylabel("ECG")
    axes[0].set_title(f"Registro {record_id}, pico {peak}, classe {label}")
    axes[1].step(np.arange(WINDOW_SIZE + 1), np.r_[original_symbols, original_symbols[-1]], where="post")
    axes[1].set_ylabel("Original")
    axes[1].set_title("Sequência simbólica original (250 símbolos)")
    axes[2].step(np.arange(len(block5_symbols) + 1), np.r_[block5_symbols, block5_symbols[-1]], where="post")
    axes[2].set_ylabel("Escala 5")
    axes[2].set_title("Coarse-graining contínuo: 50 símbolos (5 amostras por bloco)")
    axes[2].set_xlim(0, len(block5_symbols))
    if grouped_values:
        axes[3].step(np.arange(len(grouped_values) + 1), np.r_[grouped_values, grouped_values[-1]], where="post")
    axes[3].set_ylabel("Agrupada")
    axes[3].set_title("Run-length encoding dos símbolos originais")
    axes[3].set_xlim(0, len(grouped_values))
    if extrema_positions:
        axes[4].scatter(extrema_positions, normalized[extrema_positions], color="#d73027", zorder=3)
    axes[4].plot(x, normalized, color="#bbbbbb", linewidth=1)
    axes[4].set_ylabel("Extremos")
    axes[4].set_xlabel("Índice temporal da janela")
    axes[4].set_title("Máximos e mínimos usados na codificação")
    axes[4].set_xlim(0, WINDOW_SIZE)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filtered-dir", type=Path, default=Path("data/filtered"))
    parser.add_argument("--limits", type=Path, default=Path("reports/symbolic_lstm/symbol_partition_limits.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/symbolic_lstm_test"))
    args = parser.parse_args()

    limits = load_limits(args.limits)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    grouped_rows: list[dict[str, object]] = []
    extrema_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    block5_rows: list[dict[str, object]] = []
    examples: dict[str, tuple[str, int, str, np.ndarray, np.ndarray, list[int], list[int]]] = {}

    for csv_path in sorted(args.filtered_dir.glob("*_filtered.csv")):
        record_id = csv_path.stem.removesuffix("_filtered")
        frame = pd.read_csv(csv_path)
        signal = frame["amplitude_filtered"].to_numpy(dtype="float32")
        mask = frame["annotation"].notna() & (frame["annotation"] != "")
        for peak, raw_label in frame.loc[mask, "annotation"].items():
            aami = map_to_aami(str(raw_label))
            if aami is None or peak < LEFT or peak + RIGHT > len(signal):
                continue
            window = signal[peak - LEFT : peak + RIGHT]
            normalized = normalizar_janelas_minmax(window[None, :])
            symbols = simbolizar_janelas(normalized, limits)[0]
            _, block5_symbols = coarse_grain_and_symbolize(window, limits, scale=5)
            runs, grouped, n_runs = run_length_encode(symbols)
            event_sequence, event_positions, extrema = extrema_direction_encoding(normalized[0])
            base = {
                "record_id": record_id,
                "peak_index": int(peak),
                "raw_label": str(raw_label),
                "aami_label": aami,
                "original_length": len(symbols),
            }
            grouped_rows.append({**base, "grouped_sequence": grouped, "run_lengths": runs, "grouped_length": n_runs, "compression_ratio": n_runs / len(symbols)})
            block5_rows.append({**base, "scale": 5, "block_length_samples": 5, "block5_sequence": " ".join(map(str, block5_symbols)), "block5_length": len(block5_symbols), "compression_ratio": len(block5_symbols) / len(symbols)})
            extrema_rows.append({**base, "event_sequence": event_sequence, "extrema_positions": event_positions, "event_length": len(event_sequence.split()), "extrema_count": len(extrema), "event_compression_ratio": len(event_sequence.split()) / len(symbols)})
            summary_rows.append({**base, "original_length": len(symbols), "block5_length": len(block5_symbols), "grouped_length": n_runs, "event_length": len(event_sequence.split()), "extrema_count": len(extrema)})
            if aami in CLASS_ORDER and aami not in examples:
                grouped_values = [int(value) for value in grouped.split()]
                examples[aami] = (record_id, int(peak), aami, window, symbols, block5_symbols, grouped_values, extrema)

    grouped_frame = pd.DataFrame(grouped_rows)
    block5_frame = pd.DataFrame(block5_rows)
    extrema_frame = pd.DataFrame(extrema_rows)
    summary_frame = pd.DataFrame(summary_rows)
    grouped_frame.to_csv(args.output_dir / "grouped_symbol_sequences.csv", index=False)
    block5_frame.to_csv(args.output_dir / "block5_symbol_sequences.csv", index=False)
    extrema_frame.to_csv(args.output_dir / "extrema_direction_sequences.csv", index=False)
    summary_frame.to_csv(args.output_dir / "encoding_lengths.csv", index=False)
    summary_frame.groupby("aami_label")[["block5_length", "grouped_length", "event_length", "extrema_count"]].agg(["count", "mean", "median"]).to_csv(args.output_dir / "encoding_lengths_by_class.csv")

    for aami, example in examples.items():
        record_id, peak, label, window, symbols, block5_symbols, grouped_values, extrema = example
        plot_example(window, symbols, block5_symbols, grouped_values, extrema, record_id, peak, label, args.output_dir / "figures" / f"encoding_comparison_{aami}_{record_id}_{peak}.png")
    print(f"Batimentos processados: {len(summary_frame)}")
    print(f"Resultados salvos em: {args.output_dir}")


if __name__ == "__main__":
    main()
