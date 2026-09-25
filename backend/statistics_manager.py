"""
LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 1.9 — Statistical Analysis & Reporting Engine

Module: statistics_manager.py
Purpose:
- Implements deterministic, high-precision descriptive statistics and distribution summaries.
- Computes empirical bootstrap confidence intervals (default 95% CI with fixed seed).
- Computes absolute (delta_abs) and safe relative differences (delta_rel) with zero-division safeguards.
- Enforces strict neutral research language with zero normative claims ("better", "superior", "winner").
- Rejects NaN, Infinity, and invalid sample counts.
"""

import math
import random
from typing import List, Dict, Optional, Any, Tuple

RESEARCH_DISCLAIMER = (
    "RESEARCH EVALUATION ONLY — NOT CLINICAL PERFORMANCE. "
    "Calculated statistical distributions and uncertainty intervals are empirical research summaries. "
    "They do not provide clinical diagnostic guarantees, patient-level validation, or therapeutic efficacy."
)


def sanitize_float(
    val: Optional[float],
    default: float = 0.0,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    precision: int = 4
) -> float:
    """Safely converts a float value, guaranteeing no NaN or Infinity."""
    if val is None:
        return default
    try:
        f = float(val)
    except (ValueError, TypeError):
        return default

    if math.isnan(f) or math.isinf(f):
        return default

    if min_val is not None and f < min_val:
        f = min_val
    if max_val is not None and f > max_val:
        f = max_val

    return round(f, precision)


def check_finite_numbers(obj: Any) -> Tuple[bool, Optional[str]]:
    """Recursively checks if all numbers in an object are finite (no NaN or Infinity)."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return False, "NaN or Infinity detected in numeric value."
    elif isinstance(obj, dict):
        for k, v in obj.items():
            ok, err = check_finite_numbers(v)
            if not ok:
                return False, f"Key '{k}': {err}"
    elif isinstance(obj, (list, tuple)):
        for idx, item in enumerate(obj):
            ok, err = check_finite_numbers(item)
            if not ok:
                return False, f"Index {idx}: {err}"
    return True, None


class StatisticsManager:
    """Provides research-oriented descriptive statistics, bootstrap CIs, and neutral comparisons."""

    def __init__(self, default_seed: int = 42, default_bootstrap_iterations: int = 1000):
        self.default_seed = default_seed
        self.default_bootstrap_iterations = default_bootstrap_iterations

    def compute_distribution(self, values: List[float]) -> Dict[str, Any]:
        """
        Computes descriptive statistics: mean, median, std, min, max, q25, q75, iqr, count.
        Rejects NaN/Inf and handles empty/single-element collections gracefully.
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

        clean = [float(v) for v in values if v is not None and not math.isnan(float(v)) and not math.isinf(float(v))]
        n = len(clean)
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

        sorted_vals = sorted(clean)
        mean_val = sum(sorted_vals) / n

        if n == 1:
            v = sanitize_float(sorted_vals[0])
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

        # Sample standard deviation (Bessel's correction)
        variance = sum((x - mean_val) ** 2 for x in sorted_vals) / (n - 1)
        std_val = math.sqrt(max(0.0, variance))

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
        iqr = max(0.0, q75 - q25)

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
        self,
        values: List[float],
        n_bootstraps: Optional[int] = None,
        confidence_level: float = 0.95,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Computes empirical bootstrap confidence interval with fixed random seed.
        Clearly labeled as statistical uncertainty interval.
        """
        boot_n = n_bootstraps or self.default_bootstrap_iterations
        if boot_n < 100:
            boot_n = 100
        rng_seed = self.default_seed if seed is None else seed

        clean = [float(v) for v in values if v is not None and not math.isnan(float(v)) and not math.isinf(float(v))]
        n = len(clean)

        if n == 0:
            return {
                "lower": 0.0,
                "upper": 0.0,
                "method": "Bootstrap Percentile Method (empty sample)",
                "sample_size": 0,
                "iterations": boot_n,
                "confidence_level": confidence_level
            }

        if n == 1:
            v = sanitize_float(clean[0])
            return {
                "lower": v,
                "upper": v,
                "method": "Bootstrap Percentile Method (n=1)",
                "sample_size": 1,
                "iterations": boot_n,
                "confidence_level": confidence_level
            }

        rng = random.Random(rng_seed)
        boot_means = []
        for _ in range(boot_n):
            sample = [rng.choice(clean) for _ in range(n)]
            boot_means.append(sum(sample) / n)

        boot_means.sort()
        alpha = (1.0 - confidence_level) / 2.0
        lower_idx = int(math.floor(alpha * boot_n))
        upper_idx = int(math.ceil((1.0 - alpha) * boot_n)) - 1
        lower_idx = max(0, min(lower_idx, boot_n - 1))
        upper_idx = max(0, min(upper_idx, boot_n - 1))

        return {
            "lower": sanitize_float(boot_means[lower_idx]),
            "upper": sanitize_float(boot_means[upper_idx]),
            "method": f"Empirical Bootstrap Percentile ({confidence_level * 100:.1f}%)",
            "sample_size": n,
            "iterations": boot_n,
            "confidence_level": confidence_level
        }

    def compute_metric_summary(
        self,
        metric_values: Dict[str, List[float]],
        confidence_level: float = 0.95,
        n_bootstraps: Optional[int] = None,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """Computes distributions and bootstrap CIs across a dictionary of metric lists."""
        result = {}
        sample_size = 0
        for name, vals in metric_values.items():
            dist = self.compute_distribution(vals)
            ci = self.compute_bootstrap_ci(vals, n_bootstraps=n_bootstraps, confidence_level=confidence_level, seed=seed)
            dist["ci_95"] = ci
            result[name] = dist
            if dist["count"] > sample_size:
                sample_size = dist["count"]

        return {
            "sample_size": sample_size,
            "confidence_level": confidence_level,
            "bootstrap_iterations": n_bootstraps or self.default_bootstrap_iterations,
            "random_seed": self.default_seed if seed is None else seed,
            "metrics": result,
            "research_disclaimer": RESEARCH_DISCLAIMER
        }

    def compute_deltas(
        self,
        baseline: Dict[str, float],
        comparator: Dict[str, float],
        baseline_cis: Optional[Dict[str, Any]] = None,
        comparator_cis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Computes delta_abs and safe delta_rel between baseline and comparator.
        Handles zero-denominator safely.
        """
        all_keys = sorted(set(list(baseline.keys()) + list(comparator.keys())))
        deltas = {}

        for k in all_keys:
            base_val = sanitize_float(baseline.get(k, 0.0))
            comp_val = sanitize_float(comparator.get(k, 0.0))
            delta_abs = sanitize_float(comp_val - base_val)

            if abs(base_val) > 1e-7:
                delta_rel = sanitize_float((comp_val - base_val) / abs(base_val) * 100.0)
            else:
                delta_rel = 0.0

            entry = {
                "baseline_value": base_val,
                "comparator_value": comp_val,
                "delta_abs": delta_abs,
                "delta_rel": delta_rel
            }
            if baseline_cis and k in baseline_cis:
                entry["baseline_ci_95"] = baseline_cis[k]
            if comparator_cis and k in comparator_cis:
                entry["comparator_ci_95"] = comparator_cis[k]

            deltas[k] = entry

        return deltas

    def generate_neutral_summary(
        self,
        deltas: Dict[str, Any],
        exp_a_name: str = "Experiment A",
        exp_b_name: str = "Experiment B"
    ) -> str:
        """
        Generates neutral descriptive comparison summary text.
        Strictly avoids evaluative/normative words ("superior", "best", "winner", "improved").
        """
        lines = [f"Comparative Metric Analysis: {exp_a_name} vs. {exp_b_name}."]
        for metric, data in deltas.items():
            base_v = data.get("baseline_value", 0.0)
            comp_v = data.get("comparator_value", 0.0)
            d_abs = data.get("delta_abs", 0.0)
            d_rel = data.get("delta_rel", 0.0)
            sign_str = "+" if d_abs >= 0 else ""
            lines.append(
                f"- {metric}: Baseline={base_v:.4f}, Comparator={comp_v:.4f}, Difference={sign_str}{d_abs:.4f} ({sign_str}{d_rel:.1f}% relative difference)."
            )
        lines.append(
            "Note: Differences reflect quantitative measurements under specified configurations. "
            "They do not imply clinical advantage, diagnostic preference, or medical recommendation."
        )
        return "\n".join(lines)


# Global Singleton
global_statistics_manager = StatisticsManager()
