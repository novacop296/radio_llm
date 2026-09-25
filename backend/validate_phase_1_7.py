"""
LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 1.7 — Safety, Invariant & Registry Validator

Module: validate_phase_1_7.py
Purpose:
- Validates Model Version, Dataset Version, and Experiment schemas.
- Enforces strict finite numbers (zero NaN, zero Infinity).
- Enforces zero secret leakage (API keys, passwords, bearer tokens).
- Enforces zero ground-truth XML report leakage in registry payloads.
- Verifies machine artifact byte-level immutability.
"""

import os
import sys
import json
import re
import math
from typing import Dict, List, Any, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


class Phase17Validator:
    """Validates Phase 1.7 models, dataset versions, experiments, and security invariants."""

    FORBIDDEN_XML_TAGS = [r"<eFind>", r"</eFind>", r"<eImpression>", r"</eImpression>", r"<AbstractText>", r"</AbstractText>"]
    SECRET_PATTERNS = [r"sk-[a-zA-Z0-9]{20,}", r"bearer\s+[a-zA-Z0-9_\-\.]{20,}", r"password\s*[:=]\s*['\"].+?['\"]"]

    @classmethod
    def validate_no_ground_truth_leakage(cls, data: Any) -> Tuple[bool, Optional[str]]:
        """Scans stringified payloads for raw XML reference report tags."""
        text = json.dumps(data) if isinstance(data, (dict, list)) else str(data)
        for pat in cls.FORBIDDEN_XML_TAGS:
            if re.search(pat, text, re.IGNORECASE):
                return False, f"Ground-truth XML tag matched pattern '{pat}'."
        return True, None

    @classmethod
    def validate_no_secrets(cls, data: Any) -> Tuple[bool, Optional[str]]:
        """Scans payloads for exposed credentials, bearer tokens, or API keys."""
        text = json.dumps(data) if isinstance(data, (dict, list)) else str(data)
        for pat in cls.SECRET_PATTERNS:
            if re.search(pat, text, re.IGNORECASE):
                return False, f"Secret credential matched pattern '{pat}'."
        return True, None

    @classmethod
    def validate_finite_numbers(cls, data: Any) -> Tuple[bool, Optional[str]]:
        """Recursively ensures no NaN, Infinity, or -Infinity values exist."""
        if isinstance(data, dict):
            for k, v in data.items():
                ok, err = cls.validate_finite_numbers(v)
                if not ok:
                    return False, f"Key '{k}': {err}"
        elif isinstance(data, list):
            for idx, item in enumerate(data):
                ok, err = cls.validate_finite_numbers(item)
                if not ok:
                    return False, f"Index {idx}: {err}"
        elif isinstance(data, float):
            if math.isnan(data) or math.isinf(data):
                return False, f"Invalid non-finite float value: {data}"
        return True, None

    @classmethod
    def validate_model_version(cls, model: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validates model version structure and constraints."""
        errors = []
        required = ["model_id", "model_name", "architecture", "framework", "weights_identifier", "status"]
        for req in required:
            if not model.get(req):
                errors.append(f"Missing required field '{req}' in ModelVersion.")

        ok_sec, err_sec = cls.validate_no_secrets(model)
        if not ok_sec:
            errors.append(err_sec)

        ok_fin, err_fin = cls.validate_finite_numbers(model)
        if not ok_fin:
            errors.append(err_fin)

        return len(errors) == 0, errors

    @classmethod
    def validate_dataset_version(cls, dsv: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validates dataset version manifest and study identifiers."""
        errors = []
        required = ["dataset_version_id", "source_dataset", "study_ids", "study_count", "manifest_sha256", "status"]
        for req in required:
            if req not in dsv:
                errors.append(f"Missing required field '{req}' in DatasetVersion.")

        study_ids = dsv.get("study_ids", [])
        if len(study_ids) != dsv.get("study_count", -1):
            errors.append("study_count does not match length of study_ids.")

        ok_gt, err_gt = cls.validate_no_ground_truth_leakage(dsv)
        if not ok_gt:
            errors.append(err_gt)

        return len(errors) == 0, errors

    @classmethod
    def validate_experiment_record(cls, exp: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validates experiment record schema and safety invariants."""
        errors = []
        required = ["experiment_id", "status", "fingerprint_sha256"]
        for req in required:
            if req not in exp:
                errors.append(f"Missing required field '{req}' in Experiment.")

        ok_gt, err_gt = cls.validate_no_ground_truth_leakage(exp)
        if not ok_gt:
            errors.append(err_gt)

        ok_sec, err_sec = cls.validate_no_secrets(exp)
        if not ok_sec:
            errors.append(err_sec)

        ok_fin, err_fin = cls.validate_finite_numbers(exp)
        if not ok_fin:
            errors.append(err_fin)

        return len(errors) == 0, errors

    @classmethod
    def verify_zero_ground_truth_leakage(cls, data: Any) -> Tuple[bool, Optional[str]]:
        """Alias for validate_no_ground_truth_leakage."""
        return cls.validate_no_ground_truth_leakage(data)

    @classmethod
    def verify_zero_secrets(cls, data: Any) -> Tuple[bool, Optional[str]]:
        """Alias for validate_no_secrets."""
        return cls.validate_no_secrets(data)

    @classmethod
    def verify_machine_artifacts_immutability(cls, hashes_before: Dict[str, str], hashes_after: Dict[str, str]) -> Tuple[bool, List[str]]:
        """Verifies that all machine artifact hashes remain 100% byte-identical."""
        issues = []
        for path, hash_val in hashes_before.items():
            if path not in hashes_after:
                issues.append(f"Machine artifact file deleted: {path}")
            elif hashes_after[path] != hash_val:
                issues.append(f"Machine artifact file modified: {path}")
        return len(issues) == 0, issues


global_phase17_validator = Phase17Validator()
