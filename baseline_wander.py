import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts._bootstrap import bootstrap_src_path

bootstrap_src_path()


def main() -> None:
    from ecg_classification.config import ExperimentConfig, ProjectPaths
    from ecg_classification.preprocessing.baseline import add_synthetic_baseline_wander, remove_baseline_wander

    paths = ProjectPaths()
    config = ExperimentConfig()
    record_csv = sorted(paths.data_interim.glob("*_record.csv"))[0]
    frame = pd.read_csv(record_csv)
    lead = frame.columns[0]
    clean_signal = frame[lead].to_numpy()[:40000]
    contaminated, _ = add_synthetic_baseline_wander(clean_signal, sampling_rate=config.sampling_rate)
    corrected, baseline = remove_baseline_wander(contaminated, sampling_rate=config.sampling_rate)

    time = np.arange(clean_signal.size) / config.sampling_rate

    plt.figure(figsize=(12, 6))
    plt.plot(time, clean_signal, label="Original Signal", color="green", linewidth=1.5)
    plt.plot(time, contaminated, label="Contaminated Signal", color="red", alpha=0.5)
    plt.title("Original vs Contaminated ECG Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.figure(figsize=(12, 6))
    plt.plot(time, contaminated, label="Contaminated Signal", color="red", alpha=0.5)
    plt.plot(time, baseline, label="Estimated Baseline", color="blue", linewidth=1.5)
    plt.title("Estimated Baseline Wander")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.figure(figsize=(12, 6))
    plt.plot(time, corrected, label="Baseline Corrected Signal", color="orange")
    plt.title("Signal After Baseline Removal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
