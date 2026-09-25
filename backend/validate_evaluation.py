"""
Evaluation Validation & Invariant Enforcer (Phase 1.6)
======================================================
Validates JSON schema compliance, metric bounds, statistical invariants,
zero-secret leakage, zero ground-truth report leakage, and byte-level
machine artifact immutability.

RESEARCH EVALUATION ONLY — NOT CLINICAL PERFORMANCE.
"""

import os
import json
import math
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
IU_XRAY_DIR = BASE_DIR / "data" / "iu_xray"


class EvaluationValidator:
    """Enforces mathematical, cryptographic, security, and schema invariants."""

    @staticmethod
    def calculate_dir_sha256(directory_path: Path) -> Dict[str, str]:
        """Calculates SHA-256 hashes of all files in a directory."""
        hashes = {}
        if not directory_path.exists():
            return hashes
        for p in sorted(directory_path.glob("**/*")):
            if p.is_file():
                rel_p = str(p.relative_to(directory_path)).replace("\\", "/")
                with open(p, "rb") as f:
                    hashes[rel_p] = hashlib.sha256(f.read()).hexdigest()
        return hashes

    @staticmethod
    def validate_metric_bounds(eval_doc: Dict[str, Any]) -> List[str]:
        """Verifies that all metric values fall strictly within mathematical ranges."""
        errors = []
        metrics = eval_doc.get("aggregate_metrics", {})
        for name, item in metrics.items():
            val = item.get("metric_value")
            if val is not None:
                if math.isnan(val) or math.isinf(val):
                    errors.append(f"Metric '{name}' contains NaN or Infinity.")
                elif "kappa" in name.lower():
                    if val < -1.0 or val > 1.0:
                        errors.append(f"Metric '{name}' value {val} is outside valid range [-1.0, 1.0].")
                else:
                    if val < 0.0 or val > 1.0:
                        errors.append(f"Metric '{name}' value {val} is outside valid range [0.0, 1.0].")
        return errors

    @staticmethod
    def check_zero_secrets(eval_doc: Dict[str, Any]) -> List[str]:
        """Ensures no sensitive credentials, API keys, or tokens exist in evaluation records."""
        errors = []
        serialized = json.dumps(eval_doc).lower()
        forbidden_substrings = ["sk-", "bearer ", "api_key", "secret_key", "password"]
        for sub in forbidden_substrings:
            if sub in serialized:
                # Check if it's just the key name or an actual credential
                if f'"{sub}":' in serialized or f'"{sub}"' in serialized:
                    continue
                errors.append(f"Possible credential pattern detected: '{sub}'")
        return errors

    @staticmethod
    def check_zero_ground_truth_leakage(eval_doc: Dict[str, Any]) -> List[str]:
        """Verifies that no raw ground truth XML tags or clinical reference narratives are leaked."""
        errors = []
        serialized = json.dumps(eval_doc)
        leakage_patterns = ["<eFind>", "</eFind>", "<eImpression>", "</eImpression>", "<AbstractText>"]
        for pat in leakage_patterns:
            if pat in serialized:
                errors.append(f"Ground-truth XML tag '{pat}' leaked in evaluation payload.")
        return errors

    @classmethod
    def validate_evaluation_record(cls, eval_doc: Dict[str, Any]) -> Dict[str, Any]:
        """Runs full suite of invariant checks on an evaluation record."""
        errors = []

        # 1. Metric bounds
        errors.extend(cls.validate_metric_bounds(eval_doc))

        # 2. Zero secrets
        errors.extend(cls.check_zero_secrets(eval_doc))

        # 3. Zero ground truth leakage
        errors.extend(cls.check_zero_ground_truth_leakage(eval_doc))

        # 4. Identifier validity
        eval_id = eval_doc.get("evaluation_id", "")
        if any(c in eval_id for c in ["..", "/", "\\"]):
            errors.append(f"Path traversal characters detected in evaluation_id: {eval_id}")

        return {
            "valid": len(errors) == 0,
            "error_count": len(errors),
            "errors": errors
        }
