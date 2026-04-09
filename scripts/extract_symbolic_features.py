from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from scripts._bootstrap import bootstrap_src_path
except ModuleNotFoundError:  # pragma: no cover
    from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from ecg_classification.config import ProjectPaths
from ecg_classification.features.symbolic_dynamics import symbolic_feature_vector


def main() -> None:
    paths = ProjectPaths()
    rows: list[dict[str, float | str]] = []
    for record_csv in sorted(paths.data_interim.glob("*_record.csv")):
        frame = pd.read_csv(record_csv)
        lead = frame.columns[0]
        features = symbolic_feature_vector(frame[lead].to_numpy())
        rows.append({"record_id": record_csv.stem.replace("_record", ""), "lead": lead, **features})
    output = pd.DataFrame(rows)
    destination = paths.data_processed / "symbolic_features.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    print(f"Symbolic features saved to {destination}.")


if __name__ == "__main__":
    main()
