from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    docs: Path = field(init=False)
    data_raw: Path = field(init=False)
    data_interim: Path = field(init=False)
    data_processed: Path = field(init=False)
    reports: Path = field(init=False)
    experiments: Path = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "docs", self.root / "docs")
        object.__setattr__(self, "data_raw", self.root / "data" / "raw" / "mit_bih")
        object.__setattr__(self, "data_interim", self.root / "data" / "interim" / "signal_tables")
        object.__setattr__(self, "data_processed", self.root / "data" / "processed")
        object.__setattr__(self, "reports", self.root / "reports")
        object.__setattr__(self, "experiments", self.root / "experiments")

    def ensure_structure(self) -> None:
        for path in (
            self.docs,
            self.data_raw,
            self.data_interim,
            self.data_processed,
            self.reports,
            self.experiments,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ExperimentConfig:
    sampling_rate: int = 360
    default_lead: str = "MLII"
    annotation_extension: str = "atr"
    random_seed: int = 42
    train_split: float = 0.7
    validation_split: float = 0.15
    test_split: float = 0.15
