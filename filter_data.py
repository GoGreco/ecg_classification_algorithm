import matplotlib.pyplot as plt
import pandas as pd

from scripts._bootstrap import bootstrap_src_path

bootstrap_src_path()

from ecg_classification.config import ProjectPaths
from ecg_classification.preprocessing.filtering import bessel_highcut_filter


def main() -> None:
    paths = ProjectPaths()
    record_csv = sorted(paths.data_interim.glob("*_record.csv"))[0]
    frame = pd.read_csv(record_csv)
    lead = frame.columns[0]
    raw = frame[lead].to_numpy()
    filtered = bessel_highcut_filter(raw, highcut=3.0, order=5)
    plot_frame = pd.DataFrame({"Raw": raw, "Bessel": filtered})
    plot_frame.plot(title=f"Filtering demo for {record_csv.stem} ({lead})")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
