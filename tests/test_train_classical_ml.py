import json

import pandas as pd

import scripts.train_classical_ml as train_classical_ml


def test_feature_columns_excludes_metadata() -> None:
    frame = pd.DataFrame(columns=["record_id", "lead", "peak_index", "label", "morph_mean", "rr_prev"])
    assert train_classical_ml.feature_columns(frame) == ["morph_mean", "rr_prev"]


def test_train_classical_ml_main_writes_outputs(monkeypatch, tmp_path) -> None:
    data_interim = tmp_path / "interim"
    data_processed = tmp_path / "processed"
    reports = tmp_path / "reports"
    reports_tables = reports / "tables"
    data_interim.mkdir()
    data_processed.mkdir()
    reports_tables.mkdir(parents=True)

    pd.DataFrame({"dummy": [1]}).to_csv(data_interim / "100_record.csv", index=False)

    dataset = pd.DataFrame(
        {
            "record_id": ["100", "100", "101", "101", "102", "102", "103", "103"],
            "lead": ["MLII"] * 8,
            "peak_index": [1, 2, 3, 4, 5, 6, 7, 8],
            "label": ["N", "S", "N", "S", "V", "V", "Q", "Q"],
            "morph_mean": [0.1, 0.2, 0.11, 0.19, 0.12, 0.18, 0.13, 0.17],
            "rr_prev": [0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2],
        }
    )

    class DummyPaths:
        def __init__(self) -> None:
            self.data_interim = data_interim
            self.data_processed = data_processed
            self.reports = reports

    class DummyConfig:
        sampling_rate = 360
        random_seed = 42

    monkeypatch.setattr(train_classical_ml, "ProjectPaths", DummyPaths)
    monkeypatch.setattr(train_classical_ml, "ExperimentConfig", DummyConfig)
    monkeypatch.setattr(train_classical_ml, "build_dataset_from_csv_paths", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(
        train_classical_ml,
        "fit_classifier",
        lambda features, labels, random_state: {"trained": True, "columns": list(features.columns)},
    )
    monkeypatch.setattr(
        train_classical_ml,
        "predict_classifier",
        lambda classifier, features: pd.Series(["Q", "Q"] if len(features) == 2 else ["N"] * len(features)),
    )

    train_classical_ml.main()

    saved_dataset = pd.read_csv(data_processed / "beat_feature_dataset.csv")
    saved_metrics = json.loads((reports_tables / "classical_ml_metrics.json").read_text(encoding="utf-8"))
    saved_confusion = pd.read_csv(reports_tables / "classical_ml_confusion_matrix.csv", index_col=0)

    assert len(saved_dataset) == 8
    assert "accuracy" in saved_metrics
    assert list(saved_confusion.index) == ["N", "Q", "V"]
