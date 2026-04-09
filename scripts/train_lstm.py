from __future__ import annotations

try:
    from scripts._bootstrap import bootstrap_src_path
except ModuleNotFoundError:  # pragma: no cover
    from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from ecg_classification.models.lstm import build_lstm_baseline


def main() -> None:
    build_lstm_baseline()


if __name__ == "__main__":
    main()
