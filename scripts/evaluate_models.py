from __future__ import annotations

import argparse

import pandas as pd

try:
    from scripts._bootstrap import bootstrap_src_path
except ModuleNotFoundError:  # pragma: no cover
    from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from ecg_classification.evaluation.metrics import classification_metrics, confusion_matrix_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", required=True, help="CSV file with a y_true column.")
    parser.add_argument("--predictions", required=True, help="CSV file with a y_pred column.")
    args = parser.parse_args()

    truth = pd.read_csv(args.truth)["y_true"]
    predictions = pd.read_csv(args.predictions)["y_pred"]
    print(classification_metrics(truth, predictions))
    print(confusion_matrix_frame(truth, predictions))


if __name__ == "__main__":
    main()
