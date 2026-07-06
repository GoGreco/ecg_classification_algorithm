import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts._bootstrap import bootstrap_src_path

bootstrap_src_path()


def detectar_extremos_manualmente(sinal: np.ndarray) -> tuple[list[int], list[int]]:
    picos, vales = [], []
    for i in range(1, len(sinal) - 1):
        if sinal[i] > sinal[i - 1] and sinal[i] > sinal[i + 1]:
            picos.append(i)
        elif sinal[i] < sinal[i - 1] and sinal[i] < sinal[i + 1]:
            vales.append(i)
    return picos, vales


def main() -> None:
    from ecg_classification.config import ExperimentConfig, ProjectPaths
    from ecg_classification.preprocessing.filtering import clean_ecg, smooth_signal
    from ecg_classification.segmentation.beat_windows import annotations_to_dict, match_annotation
    from ecg_classification.segmentation.delineation import delineate_ecg
    from ecg_classification.segmentation.rpeaks import detect_rpeaks

    paths = ProjectPaths()
    config = ExperimentConfig()
    record_csv = paths.data_interim / "200_record.csv"
    annotation_csv = paths.data_interim / "200_annotation.csv"

    frame = pd.read_csv(record_csv)
    lead = "MLII" if "MLII" in frame.columns else frame.columns[min(1, len(frame.columns) - 1)]
    ecg_signal = frame[lead].to_numpy()

    annotation_frame = pd.read_csv(annotation_csv)
    annotation_map = annotations_to_dict(annotation_frame)

    ecg_cleaned = clean_ecg(ecg_signal, sampling_rate=config.sampling_rate)
    ecg_smoothed = smooth_signal(ecg_cleaned, window=20)
    r_indices = detect_rpeaks(ecg_cleaned, sampling_rate=config.sampling_rate)
    waves = delineate_ecg(ecg_cleaned, r_indices, sampling_rate=config.sampling_rate)
    starts = waves.get("ECG_P_Onsets", [])
    ends = waves.get("ECG_T_Offsets", [])

    time = np.arange(len(ecg_smoothed)) / config.sampling_rate

    plt.figure(figsize=(20, 8))
    plt.plot(time, ecg_smoothed, label="Smoothed Signal", color="black", alpha=0.7)

    label_added_start = False
    for idx in starts:
        if idx < len(time):
            plt.axvline(
                x=time[idx],
                color="green",
                linestyle="--",
                alpha=0.5,
                label="Start (P-On)" if not label_added_start else "",
            )
            label_added_start = True

    label_added_end = False
    for idx in ends:
        if idx < len(time):
            plt.axvline(
                x=time[idx],
                color="purple",
                linestyle=":",
                alpha=0.5,
                label="End (T-Off)" if not label_added_end else "",
            )
            label_added_end = True

    detected_labels_count = 0
    for r_idx in r_indices:
        if r_idx < len(time):
            label = match_annotation(int(r_idx), annotation_map, tolerance=15)
            if label:
                detected_labels_count += 1
                plt.annotate(
                    f"{label}",
                    xy=(time[r_idx], ecg_smoothed[r_idx]),
                    xytext=(0, 25),
                    textcoords="offset points",
                    ha="center",
                    fontsize=12,
                    color="white",
                    fontweight="bold",
                    bbox=dict(boxstyle="circle,pad=0.3", fc="blue", ec="darkblue", alpha=0.8),
                )

    picos_manuais, vales_manuais = detectar_extremos_manualmente(ecg_smoothed)
    for idx in picos_manuais:
        plt.text(time[idx], ecg_smoothed[idx] + 0.02, "P", color="red", fontsize=9, ha="center", va="bottom", fontweight="bold")
    for idx in vales_manuais:
        plt.text(time[idx], ecg_smoothed[idx] - 0.02, "V", color="blue", fontsize=9, ha="center", va="top", fontweight="bold")

    plt.title(f"ECG Delineation and Annotation Alignment ({detected_labels_count} matched annotations)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
