from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from scripts._bootstrap import bootstrap_src_path
except ModuleNotFoundError:  # pragma: no cover
    from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from ecg_classification.config import ExperimentConfig, ProjectPaths
from ecg_classification.features.labels import is_beat_label
from ecg_classification.features.symbolic_dynamics import (
    euclidean_probability_distance,
    minmax_normalize,
    probability_partition_limits,
    select_partition_count_by_entropy,
    symbol_entropy,
    symbolize_with_partitions,
    word_probability_distribution,
)


DEFAULT_REFERENCE_RECORD = "100"
DEFAULT_WORD_SIZES = (2, 4, 6)
SYMBOL_SOURCE = "mitbih_beat_annotations"
PARTITION_SOURCE = "reference_filtered_series"


def record_id_from_processed_path(processed_csv: Path) -> str:
    return processed_csv.stem.replace("_record_processed", "")


def record_has_lead(record_csv: Path, lead: str) -> bool:
    if not record_csv.exists():
        return False
    columns = pd.read_csv(record_csv, nrows=0).columns
    return lead in columns


def load_processed_signal(processed_csv: Path, signal_column: str = "baseline_corrected") -> pd.Series:
    frame = pd.read_csv(processed_csv)
    if signal_column in frame.columns:
        return frame[signal_column]
    if "cleaned" in frame.columns:
        return frame["cleaned"]
    return frame.iloc[:, 0]


def load_beat_samples(annotation_csv: Path, signal_size: int) -> list[int]:
    annotation_frame = pd.read_csv(annotation_csv)
    annotation_frame.columns = annotation_frame.columns.str.strip()
    samples: list[int] = []
    for _, row in annotation_frame.iterrows():
        sample = int(row["Sample"])
        symbol = str(row["Symbol"]).strip()
        if is_beat_label(symbol) and 0 <= sample < signal_size:
            samples.append(sample)
    return samples


def build_symbolic_record(
    record_id: str,
    normalized_signal,
    beat_samples: list[int],
    partition_limits,
    reference_distributions: dict[int, dict[tuple[int, ...], float]],
    word_sizes: tuple[int, ...] = DEFAULT_WORD_SIZES,
) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]]]:
    peak_values = normalized_signal[beat_samples]
    symbols = symbolize_with_partitions(peak_values, partition_limits)
    feature_row: dict[str, float | int | str] = {
        "record_id": record_id,
        "lead": "MLII",
        "n_peaks": int(len(beat_samples)),
        "symbolic_entropy": symbol_entropy(symbols),
    }

    alphabet_size = int(len(partition_limits) - 1)
    for symbol in range(alphabet_size):
        feature_row[f"symbol_{symbol}_probability"] = float((symbols == symbol).mean()) if symbols.size else 0.0

    for word_size in word_sizes:
        distribution = word_probability_distribution(symbols, word_size=word_size)
        feature_row[f"unique_words_{word_size}"] = int(len(distribution))
        feature_row[f"abnormality_measure_w{word_size}"] = euclidean_probability_distance(
            reference_distributions[word_size],
            distribution,
        )

    sequence_rows = [
        {
            "record_id": record_id,
            "lead": "MLII",
            "peak_index": int(sample),
            "normalized_peak_amplitude": float(value),
            "symbol": int(symbol),
        }
        for sample, value, symbol in zip(beat_samples, peak_values, symbols, strict=False)
    ]
    return feature_row, sequence_rows


def build_symbolic_dataset(
    paths: ProjectPaths,
    lead: str = "MLII",
    reference_record: str = DEFAULT_REFERENCE_RECORD,
    max_partitions: int = 40,
    entropy_epsilon: float = 0.2,
    word_sizes: tuple[int, ...] = DEFAULT_WORD_SIZES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prepared_records: dict[str, tuple[object, list[int]]] = {}
    skipped_records: list[dict[str, str]] = []

    for processed_csv in sorted(paths.data_processed.glob("*_record_processed.csv")):
        record_id = record_id_from_processed_path(processed_csv)
        record_csv = paths.data_interim / f"{record_id}_record.csv"
        annotation_csv = paths.data_interim / f"{record_id}_annotation.csv"
        if not record_has_lead(record_csv, lead=lead):
            skipped_records.append({"record_id": record_id, "reason": f"missing_{lead}"})
            continue
        if not annotation_csv.exists():
            skipped_records.append({"record_id": record_id, "reason": "missing_annotation"})
            continue

        signal = load_processed_signal(processed_csv).to_numpy(dtype=float)
        normalized_signal = minmax_normalize(signal)
        beat_samples = load_beat_samples(annotation_csv, signal_size=len(normalized_signal))
        if not beat_samples:
            skipped_records.append({"record_id": record_id, "reason": "missing_beat_samples"})
            continue
        prepared_records[record_id] = (normalized_signal, beat_samples)

    if reference_record not in prepared_records:
        raise RuntimeError(f"Reference record {reference_record} is not available for lead {lead}.")

    reference_signal, reference_samples = prepared_records[reference_record]
    reference_peaks = reference_signal[reference_samples]
    selected_partitions, entropy_history_rows = select_partition_count_by_entropy(
        reference_signal,
        symbol_signal=reference_peaks,
        max_partitions=max_partitions,
        epsilon=entropy_epsilon,
    )
    partition_limits = probability_partition_limits(reference_signal, num_partitions=selected_partitions)
    reference_symbols = symbolize_with_partitions(reference_peaks, partition_limits)
    reference_distributions = {
        word_size: word_probability_distribution(reference_symbols, word_size=word_size) for word_size in word_sizes
    }

    feature_rows: list[dict[str, float | int | str]] = []
    sequence_rows: list[dict[str, float | int | str]] = []
    for record_id, (normalized_signal, beat_samples) in sorted(prepared_records.items()):
        feature_row, record_sequence_rows = build_symbolic_record(
            record_id,
            normalized_signal,
            beat_samples,
            partition_limits,
            reference_distributions,
            word_sizes=word_sizes,
        )
        feature_row.update(
            {
                "reference_record": reference_record,
                "num_partitions": int(selected_partitions),
                "entropy_epsilon": float(entropy_epsilon),
                "symbol_source": SYMBOL_SOURCE,
            }
        )
        feature_rows.append(feature_row)
        sequence_rows.extend(record_sequence_rows)

    metadata_rows = [
        {
            "kind": "method",
            "name": "symbol_source",
            "value": SYMBOL_SOURCE,
        },
        {
            "kind": "method",
            "name": "partition_source",
            "value": PARTITION_SOURCE,
        },
        {
            "kind": "method",
            "name": "reference_record",
            "value": reference_record,
        },
        {
            "kind": "method",
            "name": "word_sizes",
            "value": "-".join(str(size) for size in word_sizes),
        },
        {
            "kind": "method",
            "name": "entropy_epsilon",
            "value": float(entropy_epsilon),
        },
        {
            "kind": "method",
            "name": "selected_partitions",
            "value": int(selected_partitions),
        },
    ]
    metadata_rows.extend(
        [
            {
                "kind": "partition_limit",
                "name": f"limit_{idx}",
                "value": float(value),
            }
            for idx, value in enumerate(partition_limits)
        ]
    )
    metadata_rows.extend(
        {
            "kind": "entropy_history",
            "name": f"k_{int(row['num_partitions'])}",
            "value": float(row["entropy"]),
            "entropy_delta": float(row["entropy_delta"]),
        }
        for row in entropy_history_rows
    )
    metadata_rows.extend(
        {"kind": "skipped_record", "name": row["record_id"], "value": row["reason"]} for row in skipped_records
    )

    return pd.DataFrame(feature_rows), pd.DataFrame(sequence_rows), pd.DataFrame(metadata_rows)


def main() -> None:
    paths = ProjectPaths()
    config = ExperimentConfig()
    features, sequences, metadata = build_symbolic_dataset(paths, lead=config.default_lead)

    paths.data_processed.mkdir(parents=True, exist_ok=True)
    features_destination = paths.data_processed / "symbolic_features.csv"
    sequences_destination = paths.data_processed / "symbolic_sequences.csv"
    metadata_destination = paths.data_processed / "symbolic_metadata.csv"

    features.to_csv(features_destination, index=False)
    sequences.to_csv(sequences_destination, index=False)
    metadata.to_csv(metadata_destination, index=False)

    print(f"Symbolic features saved to {features_destination}.")
    print(f"Symbolic sequences saved to {sequences_destination}.")
    print(f"Symbolic metadata saved to {metadata_destination}.")


if __name__ == "__main__":
    main()
