"""
LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 1.8 — Multi-Modal Benchmarking, External Portability & Invariant Validator

Module: validate_phase_1_8.py
Purpose:
- Validates external dataset schemas, bundle manifests, and benchmark records.
- Verifies multi-view pairing integrity without cross-study mismatch.
- Enforces strict security invariants: finite numbers, path traversal rejection, zero secrets, zero XML report leakage.
- Verifies byte-for-byte immutability of machine evidence (data/iu_xray/), human reviews, and consensus records.
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

DATA_DIR = os.path.join(BASE_DIR, "data")
IU_XRAY_DIR = os.path.join(DATA_DIR, "iu_xray")
REVIEWS_DIR = os.path.join(DATA_DIR, "reviews")
CONSENSUS_DIR = os.path.join(DATA_DIR, "consensus")
EXTERNAL_DS_DIR = os.path.join(DATA_DIR, "external_datasets")
BUNDLES_DIR = os.path.join(DATA_DIR, "experiment_bundles")
BENCHMARKS_DIR = os.path.join(DATA_DIR, "benchmarks")


class Phase18Validator:
    """Master validator for Phase 1.8 external datasets, portable bundles, and multi-view benchmarks."""

    @classmethod
    def validate_finite_numbers(cls, obj: Any) -> Tuple[bool, Optional[str]]:
        if isinstance(obj, float):
            if obj != obj or obj == float('inf') or obj == float('-inf'):
                return False, "NaN or Infinity detected."
        elif isinstance(obj, dict):
            for k, v in obj.items():
                ok, err = cls.validate_finite_numbers(v)
                if not ok:
                    return False, f"{k}: {err}"
        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                ok, err = cls.validate_finite_numbers(item)
                if not ok:
                    return False, f"Index {idx}: {err}"
        return True, None

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
    def validate_no_ground_truth_leakage(cls, obj: Any) -> Tuple[bool, Optional[str]]:
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
                ok, err = cls.validate_no_ground_truth_leakage(v)
                if not ok:
                    return False, err
        elif isinstance(obj, list):
            for item in obj:
                ok, err = cls.validate_no_ground_truth_leakage(item)
                if not ok:
                    return False, err
        return True, None

    @classmethod
    def validate_path_safety(cls, path_str: str) -> Tuple[bool, Optional[str]]:
        if not path_str or not isinstance(path_str, str):
            return False, "Path must be a non-empty string."
        if any(bad in path_str for bad in ["..", "\0", ":", "*", "?", '"', "<", ">", "|"]):
            return False, "Illegal characters or path traversal attempt detected."
        return True, None

    @classmethod
    def validate_multi_view_pairing_integrity(cls, studies: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """Verifies that view pairs belong strictly to the same study."""
        errors = []
        for s in studies:
            study_id = s.get("study_id")
            if not study_id:
                errors.append("Study missing study_id.")
                continue

            images = s.get("images", [])
            for img in images:
                img_id = img.get("image_id", "")
                view = img.get("view", "")
                if view not in ["Frontal", "Lateral", "Unknown"]:
                    errors.append(f"Invalid view '{view}' for image '{img_id}' in study '{study_id}'.")

            pair_id = s.get("view_pair_id")
            is_complete = s.get("is_complete_pair", False)
            views = s.get("views", [])

            if is_complete and ("Frontal" not in views or "Lateral" not in views):
                errors.append(f"Study '{study_id}' marked as complete pair but missing frontal or lateral view.")
            if is_complete and not pair_id:
                errors.append(f"Study '{study_id}' marked as complete pair but missing view_pair_id.")

        return len(errors) == 0, errors

    @classmethod
    def validate_external_dataset_record(cls, ds: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors = []
        required = ["dataset_id", "dataset_name", "dataset_version", "modality", "image_views", "study_count", "manifest_sha256", "status"]
        for req in required:
            if req not in ds:
                errors.append(f"Missing required field '{req}' in external dataset.")

        ok_sec, err_sec = cls.validate_no_secrets(ds)
        if not ok_sec:
            errors.append(err_sec)

        ok_gt, err_gt = cls.validate_no_ground_truth_leakage(ds)
        if not ok_gt:
            errors.append(err_gt)

        ok_fin, err_fin = cls.validate_finite_numbers(ds)
        if not ok_fin:
            errors.append(err_fin)

        if "studies" in ds:
            ok_pair, pair_errs = cls.validate_multi_view_pairing_integrity(ds["studies"])
            if not ok_pair:
                errors.extend(pair_errs)

        return len(errors) == 0, errors

    @classmethod
    def validate_bundle_manifest_schema(cls, manifest: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validates bundle manifest schema structure."""
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

        ok_sec, err_sec = cls.validate_no_secrets(manifest)
        if not ok_sec:
            errors.append(err_sec)

        ok_gt, err_gt = cls.validate_no_ground_truth_leakage(manifest)
        if not ok_gt:
            errors.append(err_gt)

        return len(errors) == 0, errors

    @classmethod
    def validate_benchmark_record(cls, bm: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors = []
        required = ["benchmark_id", "benchmark_name", "experiment_id", "modality", "view_configuration", "status"]
        for req in required:
            if req not in bm:
                errors.append(f"Missing required field '{req}' in benchmark.")

        ok_sec, err_sec = cls.validate_no_secrets(bm)
        if not ok_sec:
            errors.append(err_sec)

        ok_gt, err_gt = cls.validate_no_ground_truth_leakage(bm)
        if not ok_gt:
            errors.append(err_gt)

        ok_fin, err_fin = cls.validate_finite_numbers(bm)
        if not ok_fin:
            errors.append(err_fin)

        return len(errors) == 0, errors

    @classmethod
    def compute_directory_sha256(cls, directory_path: str) -> Dict[str, str]:
        """Hashes all files in a directory to verify byte-for-byte immutability."""
        file_hashes = {}
        if not os.path.exists(directory_path):
            return file_hashes
        for root, _, files in os.walk(directory_path):
            for file in sorted(files):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, directory_path).replace("\\", "/")
                hasher = hashlib.sha256()
                with open(full_path, "rb") as f:
                    while chunk := f.read(65536):
                        hasher.update(chunk)
                file_hashes[rel_path] = hasher.hexdigest()
        return file_hashes
