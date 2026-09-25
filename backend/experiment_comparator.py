"""
LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 1.7 — Multi-Experiment & Longitudinal Research Comparator Module

Module: experiment_comparator.py
Purpose:
- Compares multiple completed experiment runs, model versions, and dataset versions side-by-side.
- Detects configuration differences (architectures, weights, layers, thresholds, cohorts).
- Computes non-evaluative metric deltas and relative deltas with safe zero-denominator handling.
- Detects dataset and methodology compatibility across runs.
- Strictly eliminates subjective rankings, "winner" labels, or clinical superiority claims.
"""

import os
import sys
import json
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

try:
    from backend.experiment_registry import global_experiment_registry, RESEARCH_DISCLAIMER
    from backend.model_registry import global_model_registry
    from backend.dataset_version_manager import global_dataset_version_manager
except ImportError:
    from experiment_registry import global_experiment_registry, RESEARCH_DISCLAIMER
    from model_registry import global_model_registry
    from dataset_version_manager import global_dataset_version_manager

EVALUATION_NOTICE = "RESEARCH EVALUATION ONLY — NOT CLINICAL PERFORMANCE"


class ExperimentComparator:
    """Performs deterministic side-by-side comparison across experiments and versions."""

    def __init__(
        self,
        experiment_registry=None,
        model_registry=None,
        dataset_version_manager=None,
        experiment_manager=None
    ):
        self.experiment_registry = experiment_manager or experiment_registry or global_experiment_registry
        self.experiment_manager = self.experiment_registry
        self.model_registry = model_registry or global_model_registry
        self.dataset_version_manager = dataset_version_manager or global_dataset_version_manager

    def compare_experiments(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """Compares two or more experiments side-by-side with detailed difference detection (Phase 1.5/1.7)."""
        if not experiment_ids or len(experiment_ids) < 2:
            raise ValueError("At least 2 experiment IDs are required for comparison.")

        experiments: List[Dict[str, Any]] = []
        for eid in experiment_ids:
            clean_id = eid.strip()
            exp = self.experiment_registry.get_experiment(clean_id)
            if not exp:
                raise ValueError(f"Experiment '{clean_id}' not found.")
            experiments.append(exp)

        # 1. Detect Configuration Differences
        config_keys = [
            "model_id", "dataset_version_id", "evaluation_dataset_id", "methodology",
            "model_name", "model_version", "target_layer", "qa_threshold", "qa_top_k",
            "llm_provider", "llm_model", "normalization", "device"
        ]

        configuration_differences: List[Dict[str, Any]] = []
        for k in config_keys:
            val_map = {}
            for exp in experiments:
                fp = exp.get("configuration_fingerprint") or exp.get("configuration", {}) or {}
                val = exp.get(k) or fp.get(k)
                val_map[exp["experiment_id"]] = val

            distinct_vals = set(str(v) for v in val_map.values() if v is not None)
            if len(distinct_vals) > 1:
                configuration_differences.append({
                    "configuration_field": k,
                    "has_difference": True,
                    "values_by_experiment": val_map,
                    "description": f"Observed configuration difference across experiments for parameter '{k}'."
                })

        # Dataset snapshot / version differences
        snap_map = {exp["experiment_id"]: (exp.get("dataset_version_id") or exp.get("dataset_snapshot_id")) for exp in experiments}
        if len(set(snap_map.values())) > 1:
            configuration_differences.append({
                "configuration_field": "dataset_version_id",
                "has_difference": True,
                "values_by_experiment": snap_map,
                "description": "Experiments evaluated against different dataset versions / cohorts."
            })

        # 2. Extract and Compare Metrics (Side-by-Side)
        metric_comparisons: Dict[str, Any] = {
            "review_coverage": {},
            "consensus_coverage": {},
            "inter_rater_reliability": {},
            "explainability_coverage": {},
            "human_corrections": {},
            "machine_reviewer_agreement": {}
        }

        for exp in experiments:
            eid = exp["experiment_id"]
            metrics = exp.get("metrics") or {}

            # Review Coverage
            rc = metrics.get("review_coverage", {})
            metric_comparisons["review_coverage"][eid] = {
                "total_studies": rc.get("total_studies", 0),
                "reviewed_studies": rc.get("reviewed_studies", 0),
                "finalized_studies": rc.get("finalized_studies", 0),
                "review_completion_ratio": rc.get("review_completion_ratio", 0.0)
            }

            # Consensus Coverage
            cc = metrics.get("consensus_coverage", {})
            metric_comparisons["consensus_coverage"][eid] = {
                "total_consensus_eligible": cc.get("total_consensus_eligible", 0),
                "completed_consensus_count": cc.get("completed_consensus_count", 0),
                "consensus_completion_ratio": cc.get("consensus_completion_ratio", 0.0)
            }

            # Inter-rater Reliability
            irr = metrics.get("inter_rater_reliability", {})
            metric_comparisons["inter_rater_reliability"][eid] = {
                "cohens_kappa_2_reviewers": irr.get("cohens_kappa_2_reviewers", {}),
                "fleiss_kappa_multi_reviewers": irr.get("fleiss_kappa_multi_reviewers", {}),
                "multi_reviewer_study_count": irr.get("multi_reviewer_study_count", 0)
            }

            # Explainability Coverage
            ec = metrics.get("explainability_coverage", {})
            metric_comparisons["explainability_coverage"][eid] = {
                "grounded_studies_count": ec.get("grounded_studies_count", 0),
                "explainability_ratio": ec.get("explainability_ratio", 0.0)
            }

            # Human Corrections
            hc = metrics.get("human_corrections", {})
            metric_comparisons["human_corrections"][eid] = {
                "total_qa_corrections": hc.get("total_qa_corrections", 0),
                "total_finding_status_modifications": hc.get("total_finding_status_modifications", 0),
                "correction_rate_per_study": hc.get("correction_rate_per_study", 0.0)
            }

            # Machine-Reviewer Agreement
            mra = metrics.get("machine_reviewer_agreement", {})
            metric_comparisons["machine_reviewer_agreement"][eid] = {
                "overall_agreement_ratio": mra.get("overall_agreement_ratio", 0.0)
            }

        # 3. Numeric Deltas between first two experiments (Baseline vs Variant)
        baseline_id = experiments[0]["experiment_id"]
        variant_id = experiments[1]["experiment_id"]
        deltas = self._compute_pairwise_deltas(experiments[0], experiments[1])

        # 4. Compatibility Checks
        compatibility = self.check_compatibility(experiments[0], experiments[1])

        return {
            "comparison_id": f"cmp_{baseline_id}_{variant_id}",
            "experiment_a_id": baseline_id,
            "experiment_b_id": variant_id,
            "experiment_ids": [exp["experiment_id"] for exp in experiments],
            "configuration_differences": configuration_differences,
            "metric_comparisons": metric_comparisons,
            "metric_deltas": deltas.get("absolute_deltas", {}),
            "relative_deltas": deltas.get("relative_deltas", {}),
            "dataset_compatibility": compatibility.get("dataset_compatibility"),
            "methodology_compatibility": compatibility.get("methodology_compatibility"),
            "is_directly_comparable": compatibility.get("is_directly_comparable"),
            "comparison_notes": compatibility.get("notes"),
            "evaluation_notice": EVALUATION_NOTICE,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "disclaimer": RESEARCH_DISCLAIMER,
            "research_disclaimer": RESEARCH_DISCLAIMER
        }

    def check_compatibility(self, exp_a: Dict[str, Any], exp_b: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluates whether two experiments are directly comparable across datasets and methodology."""
        dsv_a = exp_a.get("dataset_version_id") or exp_a.get("dataset_snapshot_id")
        dsv_b = exp_b.get("dataset_version_id") or exp_b.get("dataset_snapshot_id")

        eval_ds_a = exp_a.get("evaluation_dataset_id")
        eval_ds_b = exp_b.get("evaluation_dataset_id")

        meth_a = exp_a.get("methodology", "standard")
        meth_b = exp_b.get("methodology", "standard")

        mod_a = exp_a.get("model_id") or exp_a.get("model_name")
        mod_b = exp_b.get("model_id") or exp_b.get("model_name")

        dataset_compat = (dsv_a == dsv_b)
        eval_ds_compat = (eval_ds_a == eval_ds_b)
        meth_compat = (meth_a == meth_b)

        incompatibilities = []
        if not dataset_compat:
            incompatibilities.append(f"Different dataset cohorts: '{dsv_a}' vs '{dsv_b}'")
        if not eval_ds_compat:
            incompatibilities.append(f"Different evaluation reference datasets: '{eval_ds_a}' vs '{eval_ds_b}'")
        if not meth_compat:
            incompatibilities.append(f"Different evaluation methodologies: '{meth_a}' vs '{meth_b}'")

        is_comparable = (dataset_compat and eval_ds_compat and meth_compat)
        notes = "Direct numerical comparison valid on identical dataset and methodology." if is_comparable else (
            f"Caution: Runs are NOT directly comparable due to: {'; '.join(incompatibilities)}."
        )

        return {
            "dataset_compatibility": {
                "compatible": dataset_compat,
                "dataset_version_a": dsv_a,
                "dataset_version_b": dsv_b
            },
            "evaluation_dataset_compatibility": {
                "compatible": eval_ds_compat,
                "evaluation_dataset_a": eval_ds_a,
                "evaluation_dataset_b": eval_ds_b
            },
            "methodology_compatibility": {
                "compatible": meth_compat,
                "methodology_a": meth_a,
                "methodology_b": meth_b
            },
            "is_directly_comparable": is_comparable,
            "notes": notes
        }

    def _compute_pairwise_deltas(self, exp_a: Dict[str, Any], exp_b: Dict[str, Any]) -> Dict[str, Any]:
        """Calculates absolute and relative deltas between two experiments with zero-division handling."""
        metrics_a = exp_a.get("metrics") or {}
        metrics_b = exp_b.get("metrics") or {}

        abs_deltas = {}
        rel_deltas = {}

        # 1. Flatten scalar numbers
        flat_a = self._flatten_metrics(metrics_a)
        flat_b = self._flatten_metrics(metrics_b)

        all_keys = set(flat_a.keys()) | set(flat_b.keys())
        for k in sorted(all_keys):
            val_a = flat_a.get(k)
            val_b = flat_b.get(k)

            if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                diff = float(val_b) - float(val_a)
                abs_deltas[k] = round(diff, 4)
                if abs(float(val_a)) > 1e-7:
                    rel_deltas[k] = round(diff / abs(float(val_a)), 4)
                else:
                    rel_deltas[k] = None

        return {
            "absolute_deltas": abs_deltas,
            "relative_deltas": rel_deltas
        }

    def _flatten_metrics(self, d: Any, parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
        """Recursively flattens metric dictionary."""
        items = []
        if isinstance(d, dict):
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(self._flatten_metrics(v, new_key, sep=sep).items())
                elif isinstance(v, (int, float)):
                    items.append((new_key, v))
        return dict(items)

    def compare_evaluations(self, eval_a: Dict[str, Any], eval_b: Dict[str, Any]) -> Dict[str, Any]:
        """Compares two Phase 1.6 evaluation runs with structured metric deltas."""
        id_a = eval_a.get("evaluation_id", "EVAL-A")
        id_b = eval_b.get("evaluation_id", "EVAL-B")
        exp_a = eval_a.get("experiment_id", "EXP-A")
        exp_b = eval_b.get("experiment_id", "EXP-B")

        metrics_a = eval_a.get("aggregate_metrics", {})
        metrics_b = eval_b.get("aggregate_metrics", {})

        all_metric_keys = sorted(set(list(metrics_a.keys()) + list(metrics_b.keys())))
        metric_comparisons = []

        for m_key in all_metric_keys:
            m_obj_a = metrics_a.get(m_key, {})
            m_obj_b = metrics_b.get(m_key, {})

            val_a = m_obj_a.get("metric_value")
            val_b = m_obj_b.get("metric_value")
            name = m_obj_a.get("metric_name") or m_obj_b.get("metric_name") or m_key

            delta = None
            rel_delta = None
            if val_a is not None and val_b is not None:
                delta = round(float(val_b) - float(val_a), 4)
                if abs(float(val_a)) > 1e-6:
                    rel_delta = round((float(val_b) - float(val_a)) / float(val_a), 4)

            metric_comparisons.append({
                "metric_name": name,
                "value_a": val_a,
                "value_b": val_b,
                "delta": delta,
                "relative_delta": rel_delta,
                "sample_size_a": m_obj_a.get("sample_size", 0),
                "sample_size_b": m_obj_b.get("sample_size", 0),
                "methodology": m_obj_b.get("methodology") or m_obj_a.get("methodology") or "Standard benchmark"
            })

        return {
            "comparison_id": f"COMP_{id_a}_{id_b}",
            "evaluation_id_a": id_a,
            "evaluation_id_b": id_b,
            "experiment_id_a": exp_a,
            "experiment_id_b": exp_b,
            "metric_comparisons": metric_comparisons,
            "created_at": eval_b.get("completed_at") or eval_a.get("completed_at"),
            "disclaimer": RESEARCH_DISCLAIMER
        }

    def compare_external_benchmarks(self, bench_a: Dict[str, Any], bench_b: Dict[str, Any]) -> Dict[str, Any]:
        """Compares two Phase 1.8 benchmarks across multi-view or external test set dimensions."""
        id_a = bench_a.get("benchmark_id", "BENCH-A")
        id_b = bench_b.get("benchmark_id", "BENCH-B")

        modality_a = bench_a.get("modality", "UNKNOWN")
        modality_b = bench_b.get("modality", "UNKNOWN")
        view_a = bench_a.get("view_configuration", "SINGLE_VIEW")
        view_b = bench_b.get("view_configuration", "SINGLE_VIEW")
        ds_a = bench_a.get("external_dataset_id") or bench_a.get("dataset_version_id")
        ds_b = bench_b.get("external_dataset_id") or bench_b.get("dataset_version_id")

        is_same_dataset = (ds_a == ds_b)
        is_same_view = (view_a == view_b)
        is_same_modality = (modality_a == modality_b)

        compatibility_warnings = []
        if not is_same_dataset:
            compatibility_warnings.append("Benchmarks evaluated on distinct datasets / cohorts (distribution shift likely).")
        if not is_same_view:
            compatibility_warnings.append(f"View configurations differ: {view_a} vs {view_b}.")
        if not is_same_modality:
            compatibility_warnings.append(f"Modalities differ: {modality_a} vs {modality_b}.")

        metrics_a = bench_a.get("metrics", {})
        metrics_b = bench_b.get("metrics", {})
        all_keys = sorted(set(list(metrics_a.keys()) + list(metrics_b.keys())))

        metric_deltas = []
        for k in all_keys:
            val_a = metrics_a.get(k)
            val_b = metrics_b.get(k)
            if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                abs_delta = round(float(val_b) - float(val_a), 4)
                rel_delta = round(abs_delta / float(val_a), 4) if abs(float(val_a)) > 1e-6 else None
                direction = "Observed increase" if abs_delta > 0 else ("Observed decrease" if abs_delta < 0 else "Metrics were unchanged")
                metric_deltas.append({
                    "metric_name": k,
                    "value_a": val_a,
                    "value_b": val_b,
                    "absolute_delta": abs_delta,
                    "relative_delta": rel_delta,
                    "observation": direction
                })

        now = datetime.now(timezone.utc).isoformat()
        return {
            "comparison_id": f"COMP_BENCH_{id_a}_{id_b}",
            "benchmark_id_a": id_a,
            "benchmark_id_b": id_b,
            "is_directly_comparable": (len(compatibility_warnings) == 0),
            "compatibility_warnings": compatibility_warnings,
            "dimension_comparison": {
                "dataset_a": ds_a,
                "dataset_b": ds_b,
                "view_configuration_a": view_a,
                "view_configuration_b": view_b,
                "modality_a": modality_a,
                "modality_b": modality_b
            },
            "metric_deltas": metric_deltas,
            "created_at": now,
            "disclaimer": RESEARCH_DISCLAIMER
        }


# Global Singleton
global_experiment_comparator = ExperimentComparator()

