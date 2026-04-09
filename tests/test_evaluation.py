import pandas as pd

from ecg_classification.evaluation.metrics import classification_metrics, confusion_matrix_frame
from ecg_classification.evaluation.validation import split_records


def test_classification_metrics_perfect_predictions() -> None:
    y_true = ["N", "V", "N", "S"]
    y_pred = ["N", "V", "N", "S"]

    metrics = classification_metrics(y_true, y_pred)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision_macro"] == 1.0
    assert metrics["recall_macro"] == 1.0
    assert metrics["f1_macro"] == 1.0


def test_confusion_matrix_frame_uses_union_of_labels() -> None:
    matrix = confusion_matrix_frame(["N", "V"], ["N", "S"])

    assert list(matrix.index) == ["N", "S", "V"]
    assert list(matrix.columns) == ["N", "S", "V"]
    assert isinstance(matrix, pd.DataFrame)


def test_split_records_preserves_order_and_sizes() -> None:
    splits = split_records(["100", "101", "102", "103", "104", "105"], train=0.5, validation=0.25)

    assert splits == {
        "train": ["100", "101", "102"],
        "validation": ["103"],
        "test": ["104", "105"],
    }
