from pathlib import Path
import importlib
import sys
import types

import pandas as pd

def test_data_load_wrapper_delegates_to_make_dataset(monkeypatch) -> None:
    import data_load

    called = {"count": 0}

    def fake_main() -> None:
        called["count"] += 1

    monkeypatch.setattr(data_load, "main", fake_main)

    data_load.main()

    assert called["count"] == 1


def test_make_dataset_main_calls_export_all_records(monkeypatch) -> None:
    import scripts.make_dataset as make_dataset

    called = {"count": 0}

    def fake_export_all_records():
        called["count"] += 1
        return []

    monkeypatch.setattr(make_dataset, "export_all_records", fake_export_all_records)

    make_dataset.main()

    assert called["count"] == 1


def test_preprocess_record_returns_expected_columns(monkeypatch, tmp_path: Path) -> None:
    fake_nk = types.SimpleNamespace(
        signal_filter=lambda signal, **kwargs: signal,
        ecg_clean=lambda signal, sampling_rate, method="neurokit": signal,
    )
    monkeypatch.setitem(sys.modules, "neurokit2", fake_nk)

    preprocess_mitbih = importlib.import_module("scripts.preprocess_mitbih")
    record_csv = tmp_path / "100_record.csv"
    pd.DataFrame({"MLII": [0.1, 0.2, 0.3]}).to_csv(record_csv, index=False)

    monkeypatch.setattr(preprocess_mitbih, "clean_ecg", lambda signal, sampling_rate: signal + 1.0)
    monkeypatch.setattr(
        preprocess_mitbih,
        "remove_baseline_wander",
        lambda signal, sampling_rate: (signal - 0.5, signal * 0.1),
    )

    processed = preprocess_mitbih.preprocess_record(record_csv, sampling_rate=360)

    assert list(processed.columns) == ["raw", "cleaned", "baseline", "baseline_corrected"]
    assert processed["raw"].tolist() == [0.1, 0.2, 0.3]
    assert processed["cleaned"].tolist() == [1.1, 1.2, 1.3]


def test_preprocess_record_prefers_mlii_when_not_first(monkeypatch, tmp_path: Path) -> None:
    fake_nk = types.SimpleNamespace(
        signal_filter=lambda signal, **kwargs: signal,
        ecg_clean=lambda signal, sampling_rate, method="neurokit": signal,
    )
    monkeypatch.setitem(sys.modules, "neurokit2", fake_nk)

    preprocess_mitbih = importlib.import_module("scripts.preprocess_mitbih")
    record_csv = tmp_path / "114_record.csv"
    pd.DataFrame({"V5": [0.1, 0.2, 0.3], "MLII": [0.4, 0.5, 0.6]}).to_csv(record_csv, index=False)

    monkeypatch.setattr(preprocess_mitbih, "clean_ecg", lambda signal, sampling_rate: signal + 1.0)
    monkeypatch.setattr(
        preprocess_mitbih,
        "remove_baseline_wander",
        lambda signal, sampling_rate: (signal - 0.5, signal * 0.1),
    )

    processed = preprocess_mitbih.preprocess_record(record_csv, sampling_rate=360)

    assert processed["raw"].tolist() == [0.4, 0.5, 0.6]
    assert processed["cleaned"].tolist() == [1.4, 1.5, 1.6]


def test_extract_symbolic_features_main_writes_output(monkeypatch, tmp_path: Path) -> None:
    extract_symbolic_features = importlib.import_module("scripts.extract_symbolic_features")
    interim_dir = tmp_path / "interim"
    processed_dir = tmp_path / "processed"
    interim_dir.mkdir()
    processed_dir.mkdir()
    pd.DataFrame({"MLII": [0.1, 0.2, 0.3, 0.4], "V5": [0.4, 0.3, 0.2, 0.1]}).to_csv(
        interim_dir / "100_record.csv",
        index=False,
    )
    pd.DataFrame({"Sample": [1, 2, 3], "Symbol": ["N", "N", "N"]}).to_csv(
        interim_dir / "100_annotation.csv",
        index=False,
    )
    pd.DataFrame({"baseline_corrected": [0.0, 0.2, 0.6, 1.0]}).to_csv(
        processed_dir / "100_record_processed.csv",
        index=False,
    )
    pd.DataFrame({"V5": [0.1, 0.2, 0.3, 0.4], "V2": [0.4, 0.3, 0.2, 0.1]}).to_csv(
        interim_dir / "102_record.csv",
        index=False,
    )
    pd.DataFrame({"Sample": [1, 2, 3], "Symbol": ["N", "N", "N"]}).to_csv(
        interim_dir / "102_annotation.csv",
        index=False,
    )
    pd.DataFrame({"baseline_corrected": [0.0, 0.2, 0.6, 1.0]}).to_csv(
        processed_dir / "102_record_processed.csv",
        index=False,
    )

    class DummyPaths:
        def __init__(self) -> None:
            self.data_interim = interim_dir
            self.data_processed = processed_dir

    class DummyConfig:
        default_lead = "MLII"

    monkeypatch.setattr(extract_symbolic_features, "ProjectPaths", DummyPaths)
    monkeypatch.setattr(extract_symbolic_features, "ExperimentConfig", DummyConfig)

    extract_symbolic_features.main()

    output = pd.read_csv(processed_dir / "symbolic_features.csv")
    sequences = pd.read_csv(processed_dir / "symbolic_sequences.csv")
    metadata = pd.read_csv(processed_dir / "symbolic_metadata.csv")

    assert output["record_id"].astype(str).tolist() == ["100"]
    assert output["lead"].tolist() == ["MLII"]
    assert output["reference_record"].astype(str).tolist() == ["100"]
    assert output["abnormality_measure_w2"].tolist() == [0.0]
    assert sequences["record_id"].astype(str).unique().tolist() == ["100"]
    assert set(metadata["kind"]) >= {"partition_limit", "entropy_history", "skipped_record"}


def test_estimate_signal_nsr_main_writes_report(monkeypatch, tmp_path: Path) -> None:
    estimate_signal_nsr = importlib.import_module("scripts.estimate_signal_nsr")
    interim_dir = tmp_path / "interim"
    reports_dir = tmp_path / "reports"
    interim_dir.mkdir()
    reports_dir.mkdir()
    pd.DataFrame({"MLII": [0.1, 0.2, 0.3], "V5": [0.4, 0.5, 0.6]}).to_csv(
        interim_dir / "100_record.csv",
        index=False,
    )

    class DummyPaths:
        def __init__(self) -> None:
            self.data_interim = interim_dir
            self.reports = reports_dir

    class DummyConfig:
        sampling_rate = 360
        default_lead = "MLII"

    monkeypatch.setattr(estimate_signal_nsr, "ProjectPaths", DummyPaths)
    monkeypatch.setattr(estimate_signal_nsr, "ExperimentConfig", DummyConfig)
    monkeypatch.setattr(
        estimate_signal_nsr,
        "estimate_ecg_nsr",
        lambda signal, sampling_rate, lowcut, highcut: {
            "signal_power": 2.0,
            "noise_power": 0.5,
            "nsr": 0.25,
            "nsr_db": -6.0,
            "snr_db": 6.0,
        },
    )

    estimate_signal_nsr.main()

    output = pd.read_csv(reports_dir / "signal_nsr.csv")
    assert output["record_id"].astype(str).tolist() == ["100"]
    assert output["lead"].tolist() == ["MLII"]
    assert output["n_samples"].tolist() == [3]
    assert output["nsr"].tolist() == [0.25]
