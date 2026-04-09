import pandas as pd

from ecg_classification.segmentation.beat_windows import annotations_to_dict, match_annotation


def test_annotations_to_dict_normalizes_columns() -> None:
    frame = pd.DataFrame({"Sample ": [100, 200], "Symbol ": ["N", "V"]})
    assert annotations_to_dict(frame) == {100: "N", 200: "V"}


def test_match_annotation_respects_tolerance() -> None:
    annotations = {100: "N"}
    assert match_annotation(110, annotations, tolerance=10) == "N"
