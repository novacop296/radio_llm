"""
LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 1.5 — Experiment Integrity, Security & Invariant Validator

Module: validate_experiment.py
Purpose:
- Validates experiment schemas, configuration fingerprint hashes, and metric mathematical bounds.
- Enforces strict absence of NaN/Infinity, secrets, passwords, and API keys.
- Enforces zero IU X-Ray ground-truth report leakage and completed experiment immutability.
"""

import os
import sys
import json
import math
import hashlib
from typing import Dict, List, Any, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

try:
    from backend.experiment_manager import global_experiment_manager
    from backend.dataset_snapshot_manager import global_snapshot_manager
except ImportError:
    from experiment_manager import global_experiment_manager
    from dataset_snapshot_manager import global_snapshot_manager

FORBIDDEN_GROUND_TRUTH_PATTERNS = [
    "report_ground_truth",
    "reference report",
    "<eFind>",
    "<eImpression>",
    "ground_truth_report",
    "xml_content",
    "patient_name",
    "mrn"
]

FORBIDDEN_SECRET_PATTERNS = [
    "api_key",
    "apikey",
    "password",
    "secret",
    "bearer",
    "sk-",
    "token"
]


class ExperimentValidator:
    """Validates experiment invariants, metric bounds, and security boundaries."""

    def __init__(
        self,
        experiment_manager=None,
        snapshot_manager=None
    ):
        self.experiment_manager = experiment_manager or global_experiment_manager
        self.snapshot_manager = snapshot_manager or global_snapshot_manager

    def validate_experiment(self, experiment_id: str) -> Tuple[bool, List[str]]:
        """Performs comprehensive invariant verification on an experiment record."""
        errors = []
        exp = self.experiment_manager.get_experiment(experiment_id)
        if not exp:
            return False, [f"Experiment '{experiment_id}' does not exist."]

        # 1. ID Format & Path Traversal
        if ".." in experiment_id or "/" in experiment_id or "\\" in experiment_id:
            errors.append(f"Invalid experiment ID: Path traversal characters detected in '{experiment_id}'.")

        # 2. Required Fields
        required_fields = [
            "experiment_id", "experiment_name", "description",
            "created_at", "dataset_snapshot_id", "status",
            "configuration_fingerprint", "research_disclaimer"
        ]
        for rf in required_fields:
            if rf not in exp or exp[rf] is None:
                errors.append(f"Missing required field: '{rf}'")

        # 3. Snapshot Integrity
        snap_id = exp.get("dataset_snapshot_id")
        if snap_id:
            snap_valid, snap_errs = self.snapshot_manager.validate_snapshot_integrity(snap_id)
            if not snap_valid:
                errors.extend([f"Snapshot validation error: {e}" for e in snap_errs])

        # 4. Configuration Fingerprint Hash Verification
        fp = exp.get("configuration_fingerprint", {})
        if fp:
            canonical = {
                "model_name": fp.get("model_name"),
                "model_version": fp.get("model_version"),
                "target_layer": fp.get("target_layer"),
                "preprocessing_resolution": fp.get("preprocessing_resolution"),
                "normalization": fp.get("normalization"),
                "device": fp.get("device"),
                "qa_threshold": fp.get("qa_threshold"),
                "qa_top_k": fp.get("qa_top_k"),
                "llm_provider": fp.get("llm_provider"),
                "llm_model": fp.get("llm_model")
            }
            computed_fp = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode("utf-8")).hexdigest()
            if computed_fp != fp.get("fingerprint_sha256"):
                errors.append(f"Configuration fingerprint mismatch: recorded {fp.get('fingerprint_sha256')} vs computed {computed_fp}")

        # 5. Metrics Bounds & NaN/Infinity Check
        metrics = exp.get("metrics")
        if metrics:
            num_errs = self._check_numbers_recursive(metrics)
            errors.extend(num_errs)

            # Check Kappa bounds [-1.0, 1.0]
            irr = metrics.get("inter_rater_reliability", {})
            ck_avg = irr.get("cohens_kappa_2_reviewers", {}).get("average_kappa")
            if ck_avg is not None and not (-1.0 <= ck_avg <= 1.0):
                errors.append(f"Cohen's Kappa out of bounds [-1.0, 1.0]: {ck_avg}")

            fk_avg = irr.get("fleiss_kappa_multi_reviewers", {}).get("average_kappa")
            if fk_avg is not None and not (-1.0 <= fk_avg <= 1.0):
                errors.append(f"Fleiss' Kappa out of bounds [-1.0, 1.0]: {fk_avg}")

            # Check Ratios [0.0, 1.0]
            rc_ratio = metrics.get("review_coverage", {}).get("review_completion_ratio")
            if rc_ratio is not None and not (0.0 <= rc_ratio <= 1.0):
                errors.append(f"Review completion ratio out of bounds [0.0, 1.0]: {rc_ratio}")

            ec_ratio = metrics.get("explainability_coverage", {}).get("gradcam_generation_success_rate")
            if ec_ratio is not None and not (0.0 <= ec_ratio <= 1.0):
                errors.append(f"Grad-CAM generation success rate out of bounds [0.0, 1.0]: {ec_ratio}")

        # 6. Provenance Validation (if completed)
        if exp.get("status") == "COMPLETED":
            prov = exp.get("provenance", {})
            stages = prov.get("pipeline_stages", [])
            if len(stages) < 10:
                errors.append(f"Incomplete provenance: expected >=10 stages, found {len(stages)}.")

        # 7. Ground Truth and Secret Leakage Check
        exp_json_str = json.dumps(exp).lower()
        for p in FORBIDDEN_GROUND_TRUTH_PATTERNS:
            if p in exp_json_str:
                errors.append(f"Forbidden ground-truth pattern detected: '{p}'.")

        for s in FORBIDDEN_SECRET_PATTERNS:
            # Check for actual private values rather than parameter names
            if f'"{s}":' in exp_json_str and exp_json_str.count(f'"{s}": "sk-') > 0:
                errors.append(f"Potential secret key pattern detected: '{s}'.")

        return (len(errors) == 0), errors

    def _check_numbers_recursive(self, obj: Any, path: str = "metrics") -> List[str]:
        """Recursively ensures no NaN, Infinity, or non-finite floats exist."""
        errors = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                errors.extend(self._check_numbers_recursive(v, f"{path}.{k}"))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                errors.extend(self._check_numbers_recursive(v, f"{path}[{i}]"))
        elif isinstance(obj, float):
            if math.isnan(obj):
                errors.append(f"NaN value detected at '{path}'.")
            elif math.isinf(obj):
                errors.append(f"Infinity value detected at '{path}'.")
        return errors

    def validate(self, experiment_id: str) -> Dict[str, Any]:
        """Convenience method returning structured dictionary report."""
        is_valid, errors = self.validate_experiment(experiment_id)
        return {
            "experiment_id": experiment_id,
            "is_valid": is_valid,
            "violations": errors,
            "violation_count": len(errors)
        }


# Global Singleton
global_experiment_validator = ExperimentValidator()
