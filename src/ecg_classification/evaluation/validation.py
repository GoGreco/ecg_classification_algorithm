from __future__ import annotations

from collections.abc import Sequence


def split_records(record_ids: Sequence[str], train: float = 0.7, validation: float = 0.15) -> dict[str, list[str]]:
    total = len(record_ids)
    train_end = int(total * train)
    validation_end = train_end + int(total * validation)
    return {
        "train": list(record_ids[:train_end]),
        "validation": list(record_ids[train_end:validation_end]),
        "test": list(record_ids[validation_end:]),
    }
