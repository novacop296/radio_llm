"""
Statistical Analysis & Uncertainty Engine (Phase 1.6)
=====================================================
Calculates descriptive statistics, distribution metrics, quartiles, and bootstrap
confidence intervals for research metrics.

Handles edge cases (empty collections, single observations, zero denominators,
identical values) gracefully without generating NaN or Infinity.

RESEARCH EVALUATION ONLY — NOT CLINICAL PERFORMANCE.
"""

import math
import random
from typing import List, Dict, Optional, Any


def sanitize_float(val: Optional[float], default: float = 0.0, min_val: Optional[float] = None, max_val: Optional[float] = None) -> float:
    """Ensures a float value is finite, not NaN, not Infinite, and optionally clamped."""
    if val is None or math.isnan(val) or math.isinf(val):
        val = default
    if min_val is not None and val < min_val:
        val = min_val
    if max_val is not None and val > max_val:
        val = max_val
    return round(float(val), 4)


def compute_distribution_summary(values: List[float]) -> Dict[str, Any]:
    """
    Computes summary distribution metrics: mean, median, std, min, max, q25, q75, iqr, count.
    Guaranteed to never produce NaN or Infinity.
    """
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "q25": 0.0,
            "q75": 0.0,
            "iqr": 0.0,
            "count": 0
        }

    # Filter out invalid values
    clean_vals = [float(v) for v in values if v is not None and not math.isnan(v) and not math.isinf(v)]
    n = len(clean_vals)
    if n == 0:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "q25": 0.0,
            "q75": 0.0,
            "iqr": 0.0,
            "count": 0
        }

    sorted_vals = sorted(clean_vals)
    mean_val = sum(sorted_vals) / n

    if n == 1:
        v = round(sorted_vals[0], 4)
        return {
            "mean": v,
            "median": v,
            "std": 0.0,
            "min": v,
            "max": v,
            "q25": v,
            "q75": v,
            "iqr": 0.0,
            "count": 1
        }

    # Standard deviation (sample standard deviation)
    variance = sum((x - mean_val) ** 2 for x in sorted_vals) / (n - 1)
    std_val = math.sqrt(variance)

    # Median & Quartiles
    def get_percentile(data: List[float], p: float) -> float:
        k = (len(data) - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return data[int(k)]
        d0 = data[int(f)] * (c - k)
        d1 = data[int(c)] * (k - f)
        return d0 + d1

    median_val = get_percentile(sorted_vals, 0.5)
    q25 = get_percentile(sorted_vals, 0.25)
    q75 = get_percentile(sorted_vals, 0.75)
    iqr = q75 - q25

    return {
        "mean": sanitize_float(mean_val),
        "median": sanitize_float(median_val),
        "std": sanitize_float(std_val),
        "min": sanitize_float(sorted_vals[0]),
        "max": sanitize_float(sorted_vals[-1]),
        "q25": sanitize_float(q25),
        "q75": sanitize_float(q75),
        "iqr": sanitize_float(iqr),
        "count": n
    }


def compute_bootstrap_ci(
    values: List[float],
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Computes statistical uncertainty intervals (e.g. 95% bootstrap confidence interval).
    Labeled strictly as statistical uncertainty interval, NOT clinical confidence.
    """
    clean_vals = [float(v) for v in values if v is not None and not math.isnan(v) and not math.isinf(v)]
    n = len(clean_vals)
    if n == 0:
        return {
            "lower_95": 0.0,
            "upper_95": 0.0,
            "method": "Bootstrap Percentile Method (empty sample)"
        }
    if n == 1:
        v = sanitize_float(clean_vals[0])
        return {
            "lower_95": v,
            "upper_95": v,
            "method": "Bootstrap Percentile Method (n=1)"
        }

    rng = random.Random(seed)
    bootstrap_means = []
    for _ in range(n_bootstraps):
        sample = [rng.choice(clean_vals) for _ in range(n)]
        bootstrap_means.append(sum(sample) / n)

    bootstrap_means.sort()
    alpha = (1.0 - confidence_level) / 2.0
    lower_idx = int(math.floor(alpha * n_bootstraps))
    upper_idx = int(math.ceil((1.0 - alpha) * n_bootstraps)) - 1
    upper_idx = max(0, min(upper_idx, n_bootstraps - 1))

    lower_val = sanitize_float(bootstrap_means[lower_idx])
    upper_val = sanitize_float(bootstrap_means[upper_idx])

    return {
        "lower_95": lower_val,
        "upper_95": upper_val,
        "method": f"Bootstrap Percentile Method ({n_bootstraps} resamples, seed={seed})"
    }
