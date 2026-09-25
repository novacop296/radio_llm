"""
Evaluation Report Generator (Phase 1.6)
=======================================
Generates structured, reproducible research evaluation reports in JSON and plain text formats.

Adheres strictly to research disclaimers, neutral terminology, and zero clinical claims.

RESEARCH EVALUATION ONLY — NOT CLINICAL PERFORMANCE.
"""

import json
import datetime
from typing import Dict, Any, Optional
from backend.evaluation_dataset_manager import RESEARCH_DISCLAIMER


class EvaluationReportGenerator:
    """Generates structured evaluation reports for academic and research provenance."""

    @classmethod
    def generate_json_report(
        cls,
        eval_doc: Dict[str, Any],
        comparison: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generates a comprehensive, schema-compliant JSON evaluation report."""
        eval_id = eval_doc.get("evaluation_id", "EVAL-UNKNOWN")
        exp_id = eval_doc.get("experiment_id", "EXP-UNKNOWN")
        dataset_id = eval_doc.get("evaluation_dataset_id", "DATASET-UNKNOWN")
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()

        report = {
            "report_id": f"REP_{eval_id}",
            "evaluation_id": eval_id,
            "experiment_id": exp_id,
            "dataset_id": dataset_id,
            "created_at": now,
            "evaluation_metadata": {
                "title": eval_doc.get("title", ""),
                "description": eval_doc.get("description", ""),
                "status": eval_doc.get("status", "UNKNOWN"),
                "created_at": eval_doc.get("created_at"),
                "completed_at": eval_doc.get("completed_at"),
                "study_count": eval_doc.get("study_count", 0),
                "reviewer_count": eval_doc.get("reviewer_count", 0)
            },
            "dataset_description": {
                "dataset_id": dataset_id,
                "dataset_fingerprint": eval_doc.get("dataset_fingerprint"),
                "source": "Isolated Research Benchmark Dataset"
            },
            "experiment_configuration": {
                "experiment_id": exp_id,
                "configuration_fingerprint": eval_doc.get("configuration_fingerprint")
            },
            "methodology": {
                "classification": "Micro-averaged binary classification over supported radiological finding categories.",
                "inter_rater_agreement": "Chance-corrected Cohen's Kappa (2-rater) and Fleiss' Kappa (3+-rater).",
                "uncertainty_estimation": "Non-parametric bootstrap percentile method (95% CI, 1000 resamples)."
            },
            "observed_metrics": eval_doc.get("aggregate_metrics", {}),
            "finding_evaluations": eval_doc.get("finding_evaluations", []),
            "agreement_evaluation": eval_doc.get("agreement_evaluation", {}),
            "error_analysis": eval_doc.get("error_analysis", {}),
            "statistical_summary": eval_doc.get("statistical_summary", {}),
            "experiment_comparison": comparison,
            "reproducibility": {
                "evaluation_fingerprint": eval_doc.get("evaluation_fingerprint"),
                "dataset_fingerprint": eval_doc.get("dataset_fingerprint"),
                "configuration_fingerprint": eval_doc.get("configuration_fingerprint"),
                "verification_standard": "SHA-256 Cryptographic Digest"
            },
            "limitations": [
                "Evaluations are conducted on academic datasets and do not represent clinical diagnostic accuracy.",
                "Machine activations represent feature attribution patterns rather than diagnostic probabilities.",
                "Inter-rater metrics reflect cohort agreement under experimental review protocols."
            ],
            "disclaimer": RESEARCH_DISCLAIMER
        }

        return report

    @classmethod
    def generate_text_report(
        cls,
        eval_doc: Dict[str, Any],
        comparison: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generates a plain-text research evaluation report."""
        report = cls.generate_json_report(eval_doc, comparison)
        meta = report["evaluation_metadata"]
        metrics = report["observed_metrics"]
        err = report["error_analysis"]

        lines = [
            "=" * 78,
            "EXPLAINABLE RADIOLOGY RESEARCH PROTOTYPE — RESEARCH EVALUATION REPORT",
            "=" * 78,
            f"Report ID:             {report['report_id']}",
            f"Evaluation ID:         {report['evaluation_id']}",
            f"Experiment ID:         {report['experiment_id']}",
            f"Evaluation Dataset:    {report['dataset_id']}",
            f"Status:                {meta['status']}",
            f"Studies Evaluated:     {meta['study_count']}",
            f"Evaluation Timestamp:  {report['created_at']}",
            "-" * 78,
            "REPRODUCIBILITY & CRYPTOGRAPHIC PROVENANCE",
            "-" * 78,
            f"Evaluation Fingerprint:    {report['reproducibility']['evaluation_fingerprint']}",
            f"Dataset Fingerprint:       {report['reproducibility']['dataset_fingerprint']}",
            f"Configuration Fingerprint: {report['reproducibility']['configuration_fingerprint']}",
            "-" * 78,
            "OBSERVED AGGREGATE METRICS",
            "-" * 78,
        ]

        for m_key, m_obj in metrics.items():
            val = m_obj.get("metric_value", 0.0)
            ci = m_obj.get("confidence_interval_95")
            ci_str = f" [95% CI: {ci['lower']} - {ci['upper']}]" if ci else ""
            lines.append(f"  * {m_obj.get('metric_name', m_key):<26}: {val:.4f}{ci_str}")

        lines.extend([
            "-" * 78,
            "ERROR & DISAGREEMENT SUMMARY",
            "-" * 78,
            f"  * Total Disagreements:              {err.get('total_disagreements', 0)}",
            f"  * Machine-Reviewer Disagreements:   {err.get('machine_reviewer_disagreements', 0)}",
            f"  * Inter-Reviewer Disagreements:     {err.get('inter_reviewer_disagreements', 0)}",
            f"  * Adjudication Frequency:           {err.get('adjudication_frequency', 0.0):.4f}",
            "-" * 78,
            "LIMITATIONS",
            "-" * 78,
        ])
        for lim in report["limitations"]:
            lines.append(f"  - {lim}")

        lines.extend([
            "-" * 78,
            "MANDATORY RESEARCH DISCLAIMER",
            "-" * 78,
            report["disclaimer"],
            "=" * 78
        ])

        return "\n".join(lines)
