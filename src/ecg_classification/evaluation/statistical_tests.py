from __future__ import annotations

from scipy.stats import ttest_rel


def paired_t_test(sample_a, sample_b) -> dict[str, float]:
    statistic, pvalue = ttest_rel(sample_a, sample_b)
    return {"statistic": float(statistic), "pvalue": float(pvalue)}
