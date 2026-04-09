from pathlib import Path

import pandas as pd

from ecg_classification.io.export_csv import export_dataframe


def test_export_dataframe_creates_file(tmp_path: Path) -> None:
    destination = tmp_path / "record.csv"
    frame = pd.DataFrame({"a": [1, 2, 3]})

    result = export_dataframe(frame, destination)

    assert result == destination
    assert destination.exists()
    assert pd.read_csv(destination).equals(frame)


def test_export_dataframe_respects_overwrite_flag(tmp_path: Path) -> None:
    destination = tmp_path / "record.csv"
    export_dataframe(pd.DataFrame({"a": [1]}), destination)

    export_dataframe(pd.DataFrame({"a": [2]}), destination, overwrite=False)
    assert pd.read_csv(destination)["a"].tolist() == [1]

    export_dataframe(pd.DataFrame({"a": [3]}), destination, overwrite=True)
    assert pd.read_csv(destination)["a"].tolist() == [3]
