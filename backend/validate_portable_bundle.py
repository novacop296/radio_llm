"""
LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 1.8 — Portable Bundle & Security Validator

Module: validate_portable_bundle.py
Purpose:
- Validates bundle directory structure, schema compliance, cryptographic file hashes, and security invariants.
- Enforces strict path traversal prevention, zero secret leakage, and zero XML report leakage.
"""

import os
import sys
import json
import re
import hashlib
from typing import Dict, List, Any, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

SCHEMA_PATH = os.path.join(BASE_DIR, "docs", "experiment_bundle_schema.json")


class PortableBundleValidator:
    """Validator for portable research experiment bundles."""

    @classmethod
    def validate_bundle_manifest_schema(cls, manifest: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors = []
        required = [
            "bundle_id", "experiment_id", "schema_version", "manifest_sha256",
            "fingerprint_sha256", "created_at", "files", "model_metadata",
            "dataset_metadata", "experiment_config", "software_environment", "status"
        ]
        for req in required:
            if req not in manifest:
                errors.append(f"Missing required property '{req}' in bundle manifest.")

        if "files" in manifest and not isinstance(manifest["files"], list):
            errors.append("'files' property must be a list.")

        return len(errors) == 0, errors

    @classmethod
    def validate_file_hashes(cls, bundle_dir: str, file_entries: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        errors = []
        for fe in file_entries:
            rel = fe.get("relative_path", "")
            expected_h = fe.get("sha256", "")
            if ".." in rel or rel.startswith("/") or rel.startswith("\\"):
                errors.append(f"Illegal relative path in bundle: {rel}")
                continue
            full_path = os.path.join(bundle_dir, rel)
            if not os.path.exists(full_path):
                errors.append(f"Missing file in bundle: {rel}")
                continue

            hasher = hashlib.sha256()
            with open(full_path, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
            actual_h = hasher.hexdigest()
            if actual_h != expected_h:
                errors.append(f"File hash mismatch for {rel}: expected {expected_h}, got {actual_h}")

        return len(errors) == 0, errors

    @classmethod
    def validate_no_secrets(cls, obj: Any) -> Tuple[bool, Optional[str]]:
        secret_keys = {"api_key", "password", "secret", "token", "auth", "private_key"}
        if isinstance(obj, dict):
            for k, v in obj.items():
                if any(sk in k.lower() for sk in secret_keys):
                    if v and str(v).strip():
                        return False, f"Potential secret field detected: '{k}'"
                ok, err = cls.validate_no_secrets(v)
                if not ok:
                    return False, err
        elif isinstance(obj, list):
            for item in obj:
                ok, err = cls.validate_no_secrets(item)
                if not ok:
                    return False, err
        return True, None

    @classmethod
    def validate_no_gt_leakage(cls, obj: Any) -> Tuple[bool, Optional[str]]:
        leakage_patterns = [
            r"<eFind>", r"</eFind>", r"<eImpression>", r"</eImpression>",
            r"<AbstractText", r"</AbstractText>"
        ]
        if isinstance(obj, str):
            for pat in leakage_patterns:
                if re.search(pat, obj, re.IGNORECASE):
                    return False, f"Ground-truth XML report tag detected: {pat}"
        elif isinstance(obj, dict):
            for k, v in obj.items():
                ok, err = cls.validate_no_gt_leakage(v)
                if not ok:
                    return False, err
        elif isinstance(obj, list):
            for item in obj:
                ok, err = cls.validate_no_gt_leakage(item)
                if not ok:
                    return False, err
        return True, None
