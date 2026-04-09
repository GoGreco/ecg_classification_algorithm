import numpy as np

from ecg_classification.evaluation.rpeak_metrics import match_rpeaks, summarize_rpeak_detection


def test_match_rpeaks_uses_closest_peak_within_tolerance() -> None:
    reference = np.array([100, 200, 300])
    detected = np.array([98, 110, 198, 315])

    matches, unmatched_reference, unmatched_detected = match_rpeaks(reference, detected, tolerance=15)

    assert matches == [(100, 98), (200, 198), (300, 315)]
    assert unmatched_reference == []
    assert unmatched_detected == [110]


def test_summarize_rpeak_detection_computes_metrics() -> None:
    reference = np.array([100, 200, 300, 400])
    detected = np.array([98, 205, 390, 500])

    metrics = summarize_rpeak_detection(reference, detected, tolerance=12)

    assert metrics.true_positives == 3
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.precision == 0.75
    assert metrics.recall == 0.75
    assert metrics.f1 == 0.75
    assert metrics.mean_absolute_error_samples == 5.666666666666667
