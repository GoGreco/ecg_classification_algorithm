from __future__ import annotations

import argparse
import math

import matplotlib.pyplot as plt
import pandas as pd

try:
    from scripts._bootstrap import bootstrap_src_path
except ModuleNotFoundError:  # pragma: no cover
    from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from ecg_classification.config import ExperimentConfig, ProjectPaths
from ecg_classification.evaluation.rpeak_metrics import summarize_rpeak_detection
from ecg_classification.features.labels import is_beat_label
from ecg_classification.preprocessing.baseline import remove_baseline_wander
from ecg_classification.preprocessing.filtering import clean_ecg
from ecg_classification.segmentation.rpeaks import detect_rpeaks


def load_reference_peaks(annotation_path) -> pd.Series:
    frame = pd.read_csv(annotation_path)
    columns = {column: column.strip() for column in frame.columns}
    frame = frame.rename(columns=columns)
    beats = frame[frame["Symbol"].astype(str).map(is_beat_label)]
    return beats["Sample"].astype(int)


def evaluate_record(record_path, annotation_path, sampling_rate: int, tolerance: int) -> dict[str, float | int | str]:
    signal_frame = pd.read_csv(record_path)
    lead = signal_frame.columns[0]
    raw_signal = signal_frame[lead].to_numpy()
    cleaned_signal = clean_ecg(raw_signal, sampling_rate=sampling_rate)
    corrected_signal, _ = remove_baseline_wander(cleaned_signal, sampling_rate=sampling_rate)

    reference_peaks = load_reference_peaks(annotation_path).to_numpy()
    detected_peaks = detect_rpeaks(corrected_signal, sampling_rate=sampling_rate)
    metrics = summarize_rpeak_detection(reference_peaks, detected_peaks, tolerance=tolerance)

    record_id = record_path.stem.replace("_record", "")
    return {
        "record_id": record_id,
        "lead": lead,
        "reference_peaks": int(reference_peaks.size),
        "detected_peaks": int(detected_peaks.size),
        "true_positives": metrics.true_positives,
        "false_positives": metrics.false_positives,
        "false_negatives": metrics.false_negatives,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "mean_absolute_error_samples": metrics.mean_absolute_error_samples,
        "median_absolute_error_samples": metrics.median_absolute_error_samples,
        "mean_absolute_error_ms": metrics.mean_absolute_error_samples * 1000.0 / sampling_rate
        if not math.isnan(metrics.mean_absolute_error_samples)
        else float("nan"),
        "median_absolute_error_ms": metrics.median_absolute_error_samples * 1000.0 / sampling_rate
        if not math.isnan(metrics.median_absolute_error_samples)
        else float("nan"),
    }


def build_figure(metrics_frame: pd.DataFrame, output_path) -> None:
    ordered = metrics_frame.sort_values("f1", ascending=False).reset_index(drop=True)
    summary = ordered[["precision", "recall", "f1"]].mean()

    figure, axes = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2.2, 1.3])

    x = range(len(ordered))
    axes[0].plot(x, ordered["precision"], label="Precision", linewidth=2)
    axes[0].plot(x, ordered["recall"], label="Recall", linewidth=2)
    axes[0].plot(x, ordered["f1"], label="F1", linewidth=2.5)
    axes[0].set_title("Desempenho da deteccao de picos R por registro")
    axes[0].set_ylabel("Score")
    axes[0].set_xlabel("Registros ordenados por F1")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    summary.plot(kind="bar", ax=axes[1], color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    axes[1].set_title("Media global das metricas")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].grid(True, axis="y", alpha=0.3)
    for bar in axes[1].patches:
        height = bar.get_height()
        axes[1].annotate(
            f"{height:.3f}",
            (bar.get_x() + bar.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            xytext=(0, 4),
            textcoords="offset points",
        )

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tolerance", type=int, default=15, help="Tolerance in samples for a correct peak match.")
    args = parser.parse_args()

    paths = ProjectPaths()
    config = ExperimentConfig()

    rows: list[dict[str, float | int | str]] = []
    for record_path in sorted(paths.data_interim.glob("*_record.csv")):
        annotation_path = paths.data_interim / f"{record_path.stem.replace('_record', '')}_annotation.csv"
        if not annotation_path.exists():
            continue
        rows.append(
            evaluate_record(
                record_path=record_path,
                annotation_path=annotation_path,
                sampling_rate=config.sampling_rate,
                tolerance=args.tolerance,
            )
        )

    if not rows:
        raise RuntimeError("No record and annotation CSV pairs were found in data/interim/signal_tables.")

    metrics_frame = pd.DataFrame(rows).sort_values("record_id").reset_index(drop=True)
    reports_tables = paths.reports / "tables"
    reports_figures = paths.reports / "figures"
    reports_tables.mkdir(parents=True, exist_ok=True)
    reports_figures.mkdir(parents=True, exist_ok=True)

    csv_path = reports_tables / "rpeak_detection_metrics.csv"
    figure_path = reports_figures / "rpeak_detection_metrics.png"
    metrics_frame.to_csv(csv_path, index=False)
    build_figure(metrics_frame, figure_path)

    summary = metrics_frame[["precision", "recall", "f1"]].mean().to_dict()
    print(f"Metrics saved to {csv_path}.")
    print(f"Figure saved to {figure_path}.")
    print(
        "Global mean metrics: "
        f"precision={summary['precision']:.4f}, "
        f"recall={summary['recall']:.4f}, "
        f"f1={summary['f1']:.4f}"
    )


if __name__ == "__main__":
    main()
