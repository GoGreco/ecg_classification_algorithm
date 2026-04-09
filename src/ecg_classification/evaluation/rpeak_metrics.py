from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RPeakDetectionResult:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    mean_absolute_error_samples: float
    median_absolute_error_samples: float


def match_rpeaks(
    reference_peaks: np.ndarray,
    detected_peaks: np.ndarray,
    tolerance: int,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    reference = np.asarray(reference_peaks, dtype=int)
    detected = np.asarray(detected_peaks, dtype=int)

    matches: list[tuple[int, int]] = []
    unmatched_reference: list[int] = []
    used_detected: set[int] = set()

    for ref_peak in reference:
        best_index: int | None = None
        best_distance: int | None = None
        candidate_index = int(np.searchsorted(detected, ref_peak - tolerance, side="left"))
        while candidate_index < detected.size and detected[candidate_index] <= ref_peak + tolerance:
            if candidate_index not in used_detected:
                distance = abs(int(detected[candidate_index]) - int(ref_peak))
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_index = candidate_index
            candidate_index += 1

        if best_index is None:
            unmatched_reference.append(int(ref_peak))
            continue

        used_detected.add(best_index)
        matches.append((int(ref_peak), int(detected[best_index])))

    unmatched_detected = [
        int(peak) for index, peak in enumerate(detected) if index not in used_detected
    ]
    return matches, unmatched_reference, unmatched_detected


def summarize_rpeak_detection(
    reference_peaks: np.ndarray,
    detected_peaks: np.ndarray,
    tolerance: int,
) -> RPeakDetectionResult:
    matches, unmatched_reference, unmatched_detected = match_rpeaks(
        reference_peaks=reference_peaks,
        detected_peaks=detected_peaks,
        tolerance=tolerance,
    )

    true_positives = len(matches)
    false_negatives = len(unmatched_reference)
    false_positives = len(unmatched_detected)

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    if matches:
        absolute_errors = np.abs([detected - reference for reference, detected in matches])
        mean_absolute_error = float(np.mean(absolute_errors))
        median_absolute_error = float(np.median(absolute_errors))
    else:
        mean_absolute_error = float("nan")
        median_absolute_error = float("nan")

    return RPeakDetectionResult(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        mean_absolute_error_samples=mean_absolute_error,
        median_absolute_error_samples=median_absolute_error,
    )
