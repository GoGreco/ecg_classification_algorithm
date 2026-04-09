import pandas as pd

from ecg_classification.features.beat_dataset import build_dataset_from_csv_paths, build_labeled_beat_dataset, select_lead
from ecg_classification.features.labels import map_to_aami


def test_select_lead_prefers_clinical_priority() -> None:
    frame = pd.DataFrame({"V2": [0.0], "MLII": [1.0]})
    assert select_lead(frame) == "MLII"


def test_build_labeled_beat_dataset_combines_features(monkeypatch) -> None:
    record_frame = pd.DataFrame({"MLII": [0.1] * 500})
    annotation_frame = pd.DataFrame({"Sample": [150, 250], "Symbol": ["N", "V"]})

    dataset = build_labeled_beat_dataset(
        record_frame=record_frame,
        annotation_frame=annotation_frame,
        record_id="100",
        sampling_rate=360,
        left_size=50,
        right_size=50,
    )

    assert len(dataset) == 2
    assert {"record_id", "lead", "peak_index", "label", "rr_prev", "rr_next"} <= set(dataset.columns)
    assert dataset["label"].tolist() == ["N", "V"]


def test_build_labeled_beat_dataset_filters_non_beat_annotations() -> None:
    record_frame = pd.DataFrame({"MLII": [0.1] * 500})
    annotation_frame = pd.DataFrame({"Sample": [150, 250, 300], "Symbol": ["N", "+", "V"]})

    dataset = build_labeled_beat_dataset(
        record_frame=record_frame,
        annotation_frame=annotation_frame,
        record_id="100",
        sampling_rate=360,
        left_size=50,
        right_size=50,
    )

    assert dataset["label"].tolist() == ["N", "V"]


def test_build_labeled_beat_dataset_supports_label_mapping() -> None:
    record_frame = pd.DataFrame({"MLII": [0.1] * 500})
    annotation_frame = pd.DataFrame({"Sample": [150, 250, 300], "Symbol": ["L", "A", "/"]})

    dataset = build_labeled_beat_dataset(
        record_frame=record_frame,
        annotation_frame=annotation_frame,
        record_id="100",
        sampling_rate=360,
        left_size=50,
        right_size=50,
        label_transform=map_to_aami,
    )

    assert dataset["label"].tolist() == ["N", "S", "Q"]


def test_build_dataset_from_csv_paths_joins_multiple_records(monkeypatch, tmp_path) -> None:
    interim = tmp_path / "interim"
    interim.mkdir()

    pd.DataFrame({"MLII": [0.1] * 500}).to_csv(interim / "100_record.csv", index=False)
    pd.DataFrame({"Sample": [150], "Symbol": ["N"]}).to_csv(interim / "100_annotation.csv", index=False)
    pd.DataFrame({"MLII": [0.2] * 500}).to_csv(interim / "101_record.csv", index=False)
    pd.DataFrame({"Sample": [200], "Symbol": ["V"]}).to_csv(interim / "101_annotation.csv", index=False)

    dataset = build_dataset_from_csv_paths(
        record_paths=[interim / "100_record.csv", interim / "101_record.csv"],
        annotation_dir=interim,
        sampling_rate=360,
    )

    assert set(dataset["record_id"].astype(str)) == {"100", "101"}
