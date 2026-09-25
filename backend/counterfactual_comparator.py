"""
LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 2.0 — Interactive Counterfactual Explanations & Clinical Reasoning Synthesis

Module: counterfactual_comparator.py
Purpose:
- Implements comparative analysis across counterfactual experiments.
- Supports:
    1. Baseline vs Counterfactual comparison
    2. Perturbation Method A vs Perturbation Method B comparison
    3. Target ROI vs Random Control ROI comparison
    4. Multi-strength sensitivity curve analysis
- Computes rigorous descriptive statistics and bootstrap confidence intervals.
- Enforces strictly neutral research terminology without causal claims.

DISCLAIMER:
RESEARCH USE ONLY.
Comparison metrics reflect computational model response sensitivity, NOT clinical causality.
"""

import os
import sys
import math
from typing import Dict, List, Optional, Any, Tuple, Union

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.counterfactual_manager import (
    validate_finite_number,
    RESEARCH_DISCLAIMER,
    global_counterfactual_manager
)


class CounterfactualComparator:
    """
    Compares counterfactual experiments and generates neutral quantitative reports.
    """

    def __init__(self, manager=global_counterfactual_manager):
        self.manager = manager

    def compute_summary_statistics(self, values: List[float]) -> Dict[str, float]:
        """
        Computes descriptive statistics (mean, median, std, min, max, Q1, Q3, IQR).
        Validates all inputs are finite.
        """
        if not values:
            return {
                "count": 0, "mean": 0.0, "median": 0.0, "std": 0.0,
                "min": 0.0, "max": 0.0, "q1": 0.0, "q3": 0.0, "iqr": 0.0
            }

        clean = [validate_finite_number(v, f"value_{i}") for i, v in enumerate(values)]
        arr = np.array(clean, dtype=np.float64)

        mean_val = float(np.mean(arr))
        median_val = float(np.median(arr))
        std_val = float(np.std(arr)) if len(arr) > 1 else 0.0
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        q1_val = float(np.percentile(arr, 25))
        q3_val = float(np.percentile(arr, 75))
        iqr_val = float(q3_val - q1_val)

        return {
            "count": len(clean),
            "mean": round(mean_val, 4),
            "median": round(median_val, 4),
            "std": round(std_val, 4),
            "min": round(min_val, 4),
            "max": round(max_val, 4),
            "q1": round(q1_val, 4),
            "q3": round(q3_val, 4),
            "iqr": round(iqr_val, 4)
        }

    def compute_bootstrap_ci(
        self,
        values: List[float],
        num_resamples: int = 1000,
        confidence_level: float = 0.95,
        seed: int = 42
    ) -> Dict[str, float]:
        """
        Computes non-parametric bootstrap confidence interval for the mean.
        """
        if not values or len(values) < 2:
            val = float(values[0]) if values else 0.0
            return {"ci_lower": round(val, 4), "ci_upper": round(val, 4), "confidence_level": confidence_level}

        clean = np.array([validate_finite_number(v) for v in values], dtype=np.float64)
        rng = np.random.RandomState(seed)
        n = len(clean)
        resampled_means = []

        for _ in range(num_resamples):
            sample = rng.choice(clean, size=n, replace=True)
            resampled_means.append(np.mean(sample))

        alpha = (1.0 - confidence_level) / 2.0
        lower = float(np.percentile(resampled_means, alpha * 100))
        upper = float(np.percentile(resampled_means, (1.0 - alpha) * 100))

        return {
            "ci_lower": round(lower, 4),
            "ci_upper": round(upper, 4),
            "confidence_level": confidence_level
        }

    def compare_two_experiments(
        self,
        exp_a: Dict[str, Any],
        exp_b: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compares two counterfactual experiments (e.g. Target ROI vs Control ROI, or Method A vs Method B).
        Generates objective comparative text with strictly non-causal descriptions.
        """
        deltas_a = {d["pathology"]: d["delta_abs"] for d in exp_a.get("finding_deltas", [])}
        deltas_b = {d["pathology"]: d["delta_abs"] for d in exp_b.get("finding_deltas", [])}

        all_paths = sorted(list(set(deltas_a.keys()).union(set(deltas_b.keys()))))
        comparison_table = []
        diffs = []

        for p in all_paths:
            da = deltas_a.get(p, 0.0)
            db = deltas_b.get(p, 0.0)
            diff = round(da - db, 4)
            diffs.append(abs(diff))

            comparison_table.append({
                "pathology": p,
                "delta_experiment_a": da,
                "delta_experiment_b": db,
                "difference": diff,
                "greater_sensitivity": "Experiment A" if abs(da) > abs(db) else ("Experiment B" if abs(db) > abs(da) else "Equal")
            })

        stats = self.compute_summary_statistics(diffs)
        ci = self.compute_bootstrap_ci(diffs)

        # Objective neutral comparative statement
        mean_diff = stats["mean"]
        method_a = exp_a.get("perturbation", {}).get("method", "Experiment A")
        method_b = exp_b.get("perturbation", {}).get("method", "Experiment B")

        neutral_summary = (
            f"Under the evaluated configurations, {method_a} and {method_b} exhibited a mean absolute "
            f"activation difference of {mean_diff}. This reflects differential computational model sensitivity "
            f"across the tested perturbation regions and parameters."
        )

        return {
            "experiment_a_id": exp_a.get("counterfactual_id"),
            "experiment_b_id": exp_b.get("counterfactual_id"),
            "comparison_table": comparison_table,
            "statistics": stats,
            "bootstrap_ci": ci,
            "neutral_summary": neutral_summary,
            "disclaimer": RESEARCH_DISCLAIMER
        }

    def run_control_comparison(
        self,
        study_id: str,
        counterfactual_id: str,
        actor: str = "researcher"
    ) -> Dict[str, Any]:
        """
        Runs an automated control comparison: executes the same perturbation on a non-overlapping control ROI.
        """
        target_exp = self.manager.get_counterfactual(study_id, counterfactual_id)
        if target_exp.get("status") not in ("COMPLETED", "VALIDATED", "PUBLISHED"):
            target_exp = self.manager.run_counterfactual(study_id, counterfactual_id, actor)

        target_roi = target_exp["perturbation"]["roi"]
        control_roi = self.manager.generate_random_control_roi((512, 512), target_roi, seed=target_exp["perturbation"]["seed"] + 100)

        # Create control experiment
        ctrl_exp = self.manager.create_counterfactual(study_id, target_exp.get("source_view", "PA"), created_by=f"{actor}_control")
        ctrl_exp = self.manager.configure_counterfactual(
            study_id,
            ctrl_exp["counterfactual_id"],
            method=target_exp["perturbation"]["method"],
            roi=control_roi,
            strength=target_exp["perturbation"]["strength"],
            seed=target_exp["perturbation"]["seed"] + 100,
            target_pathology=target_exp["perturbation"].get("target_pathology"),
            actor=actor
        )
        ctrl_exp = self.manager.run_counterfactual(study_id, ctrl_exp["counterfactual_id"], actor)

        return self.compare_two_experiments(target_exp, ctrl_exp)


global_counterfactual_comparator = CounterfactualComparator()
