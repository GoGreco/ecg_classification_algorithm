from __future__ import annotations

import argparse

try:
    from scripts._bootstrap import bootstrap_src_path
except ModuleNotFoundError:  # pragma: no cover
    from _bootstrap import bootstrap_src_path

bootstrap_src_path()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment",
        choices=["make_dataset", "preprocess", "symbolic_features", "train_classical_ml", "train_lstm"],
    )
    args = parser.parse_args()

    if args.experiment == "make_dataset":
        from scripts.make_dataset import main as command
    elif args.experiment == "preprocess":
        from scripts.preprocess_mitbih import main as command
    elif args.experiment == "symbolic_features":
        from scripts.extract_symbolic_features import main as command
    elif args.experiment == "train_classical_ml":
        from scripts.train_classical_ml import main as command
    else:
        from scripts.train_lstm import main as command
    command()


if __name__ == "__main__":
    main()
