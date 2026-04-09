from __future__ import annotations

import json

import pandas as pd

try:
    from scripts._bootstrap import bootstrap_src_path
except ModuleNotFoundError:  # pragma: no cover
    from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from ecg_classification.config import ExperimentConfig, ProjectPaths
from ecg_classification.evaluation.metrics import classification_metrics, confusion_matrix_frame
from ecg_classification.evaluation.validation import split_records
from ecg_classification.features.beat_dataset import build_dataset_from_csv_paths
from ecg_classification.features.labels import map_to_aami
from ecg_classification.models.classical_ml import fit_classifier, predict_classifier


METADATA_COLUMNS = {"record_id", "lead", "peak_index", "label"}


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in METADATA_COLUMNS]


def main() -> None:
    paths = ProjectPaths()
    config = ExperimentConfig()

    record_paths = sorted(paths.data_interim.glob("*_record.csv"))
    dataset = build_dataset_from_csv_paths(
        record_paths,
        paths.data_interim,
        sampling_rate=config.sampling_rate,
        label_transform=map_to_aami,
    )
    if dataset.empty:
        raise RuntimeError("No labeled beat dataset could be built from the available records.")

    dataset_path = paths.data_processed / "beat_feature_dataset.csv"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(dataset_path, index=False)

    record_splits = split_records(sorted(dataset["record_id"].astype(str).unique()))
    train_dataset = dataset[dataset["record_id"].astype(str).isin(record_splits["train"])].copy()
    test_dataset = dataset[dataset["record_id"].astype(str).isin(record_splits["test"])].copy()

    if train_dataset.empty or test_dataset.empty:
        raise RuntimeError("Train/test split by record produced an empty partition.")

    columns = feature_columns(dataset)
    classifier = fit_classifier(train_dataset[columns], train_dataset["label"], random_state=config.random_seed)
    predictions = predict_classifier(classifier, test_dataset[columns])

    metrics = classification_metrics(test_dataset["label"], predictions)
    confusion = confusion_matrix_frame(test_dataset["label"], predictions)

    paths.reports.mkdir(parents=True, exist_ok=True)
    paths.reports.joinpath("tables").mkdir(parents=True, exist_ok=True)
    metrics_path = paths.reports / "tables" / "classical_ml_metrics.json"
    confusion_path = paths.reports / "tables" / "classical_ml_confusion_matrix.csv"

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    confusion.to_csv(confusion_path, index=True)

    print(f"Beat feature dataset saved to {dataset_path}.")
    print(f"Metrics saved to {metrics_path}.")
    print(f"Confusion matrix saved to {confusion_path}.")


if __name__ == "__main__":
    main()
