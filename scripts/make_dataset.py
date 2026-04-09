from __future__ import annotations

try:
    from scripts._bootstrap import bootstrap_src_path
except ModuleNotFoundError:  # pragma: no cover
    from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from ecg_classification.io.export_csv import export_all_records


def main() -> None:
    export_all_records()
    print("WFDB records exported to CSV in data/interim/signal_tables.")


if __name__ == "__main__":
    main()
