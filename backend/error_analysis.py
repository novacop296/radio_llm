"""
Research Error Analysis Engine (Phase 1.6)
=========================================
Computes finding-level error summaries, confusion matrices, and observed disagreement
distributions across machine predictions, human reviewer judgements, and reference annotations.

Uses strictly neutral research terminology without causal or clinical claims.

RESEARCH EVALUATION ONLY — NOT CLINICAL PERFORMANCE.
"""

from typing import Dict, List, Any, Optional
from backend.statistical_analysis import sanitize_float


class ErrorAnalysisEngine:
    """Computes neutral error and disagreement metrics for research evaluations."""

    @staticmethod
    def build_confusion_matrix(tp: int, tn: int, fp: int, fn: int) -> Dict[str, int]:
        return {
            "tp": max(0, int(tp)),
            "tn": max(0, int(tn)),
            "fp": max(0, int(fp)),
            "fn": max(0, int(fn))
        }

    @staticmethod
    def calculate_classification_metrics(tp: int, tn: int, fp: int, fn: int) -> Dict[str, Optional[float]]:
        """
        Calculates classification metrics with zero-division handling and bounded outputs [0.0, 1.0].
        """
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0.0

        pos_pred = tp + fp
        precision = tp / pos_pred if pos_pred > 0 else 0.0

        actual_pos = tp + fn
        recall = tp / actual_pos if actual_pos > 0 else 0.0
        sensitivity = recall

        actual_neg = tn + fp
        specificity = tn / actual_neg if actual_neg > 0 else 0.0

        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        balanced_accuracy = (sensitivity + specificity) / 2.0

        return {
            "accuracy": sanitize_float(accuracy, min_val=0.0, max_val=1.0),
            "precision": sanitize_float(precision, min_val=0.0, max_val=1.0),
            "recall": sanitize_float(recall, min_val=0.0, max_val=1.0),
            "f1": sanitize_float(f1, min_val=0.0, max_val=1.0),
            "specificity": sanitize_float(specificity, min_val=0.0, max_val=1.0),
            "sensitivity": sanitize_float(sensitivity, min_val=0.0, max_val=1.0),
            "balanced_accuracy": sanitize_float(balanced_accuracy, min_val=0.0, max_val=1.0)
        }

    @classmethod
    def analyze_finding_disagreements(
        cls,
        eval_findings: List[Dict[str, Any]],
        reviewer_sessions: Optional[List[Dict[str, Any]]] = None,
        reference_annotations: Optional[Dict[str, Dict[str, int]]] = None
    ) -> Dict[str, Any]:
        """
        Aggregates per-finding and overall disagreement observations.
        """
        per_finding_disagreements: Dict[str, int] = {}
        total_machine_reviewer_disagreements = 0
        total_inter_reviewer_disagreements = 0
        disagreement_details = []

        for item in eval_findings:
            ftype = item.get("finding_type", "unknown")
            cm = item.get("confusion_matrix", {})
            fp = cm.get("fp", 0)
            fn = cm.get("fn", 0)
            finding_disagreements = fp + fn
            per_finding_disagreements[ftype] = finding_disagreements
            total_machine_reviewer_disagreements += finding_disagreements

            if finding_disagreements > 0:
                disagreement_details.append({
                    "finding_type": ftype,
                    "observed_fp_count": fp,
                    "observed_fn_count": fn,
                    "sample_size": item.get("sample_count", 0),
                    "disagreement_rate": sanitize_float(
                        finding_disagreements / item.get("sample_count", 1) if item.get("sample_count", 0) > 0 else 0.0
                    )
                })

        # Calculate inter-reviewer disagreements if sessions provided
        if reviewer_sessions and len(reviewer_sessions) > 1:
            # Check pairwise disagreements
            by_study = {}
            for sess in reviewer_sessions:
                sid = sess.get("study_id")
                if sid:
                    by_study.setdefault(sid, []).append(sess)
            
            for sid, s_list in by_study.items():
                if len(s_list) >= 2:
                    for i in range(len(s_list)):
                        for j in range(i + 1, len(s_list)):
                            rev_a = s_list[i].get("decisions", {})
                            rev_b = s_list[j].get("decisions", {})
                            for k, val_a in rev_a.items():
                                if k in rev_b and rev_b[k] != val_a:
                                    total_inter_reviewer_disagreements += 1

        total_disagreements = total_machine_reviewer_disagreements + total_inter_reviewer_disagreements
        adjudication_frequency = sanitize_float(
            total_inter_reviewer_disagreements / max(1, len(eval_findings)),
            min_val=0.0,
            max_val=1.0
        )

        return {
            "total_disagreements": total_disagreements,
            "machine_reviewer_disagreements": total_machine_reviewer_disagreements,
            "inter_reviewer_disagreements": total_inter_reviewer_disagreements,
            "adjudication_frequency": adjudication_frequency,
            "per_finding_disagreements": per_finding_disagreements,
            "disagreement_details": disagreement_details
        }

    @classmethod
    def compute_error_breakdown(
        cls,
        confusion_matrix: Dict[str, int],
        finding_evaluations: Optional[Dict[str, Any]] = None,
        reviewer_disagreements: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Calculates a comprehensive error analysis breakdown from confusion matrices and disagreements."""
        tp = confusion_matrix.get("tp", 0)
        tn = confusion_matrix.get("tn", 0)
        fp = confusion_matrix.get("fp", 0)
        fn = confusion_matrix.get("fn", 0)
        total = tp + tn + fp + fn
        clf = cls.calculate_classification_metrics(tp, tn, fp, fn)

        return {
            "confusion_matrix": cls.build_confusion_matrix(tp, tn, fp, fn),
            "classification_metrics": clf,
            "total_samples": total,
            "error_count": fp + fn,
            "error_rate": sanitize_float((fp + fn) / total if total > 0 else 0.0),
            "false_positive_rate": sanitize_float(fp / (fp + tn) if (fp + tn) > 0 else 0.0),
            "false_negative_rate": sanitize_float(fn / (fn + tp) if (fn + tp) > 0 else 0.0),
            "finding_evaluations": finding_evaluations or {},
            "reviewer_disagreements": reviewer_disagreements or [],
            "disclaimer": "RESEARCH ERROR ANALYSIS — NOT CLINICAL DIAGNOSTIC SENSITIVITY"
        }
