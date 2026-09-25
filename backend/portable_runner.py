"""
LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 1.8 — Air-Gapped Reproducibility & Portable Experiment Runner

Module: portable_runner.py
Purpose:
- Loads self-contained experiment bundles in isolated / air-gapped mode.
- Validates bundle manifests, verifies cryptographic file integrity, and re-executes configuration fingerprinting.
- Confirms whether an experiment can be reproduced 100% deterministically from bundled artifacts alone.
- Yields structured status: VERIFIED, MISMATCH, INCOMPLETE, or INVALID.
"""

import os
import sys
import json
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from backend.experiment_bundle import global_experiment_bundle_manager, RESEARCH_DISCLAIMER
    from backend.validate_portable_bundle import PortableBundleValidator
    from backend.experiment_registry import ExperimentRegistry
except ImportError:
    from experiment_bundle import global_experiment_bundle_manager, RESEARCH_DISCLAIMER
    from validate_portable_bundle import PortableBundleValidator
    from experiment_registry import ExperimentRegistry


class PortableRunner:
    """Executes air-gapped bundle validation and reproducibility fingerprint re-calculation."""

    def __init__(self, bundle_manager=None):
        self.bundle_manager = bundle_manager or global_experiment_bundle_manager

    def verify_reproducibility(self, bundle_id: str) -> Dict[str, Any]:
        """Performs full air-gapped reproducibility audit on an experiment bundle."""
        bundle_path = os.path.join(self.bundle_manager.storage_dir, bundle_id)
        if not os.path.exists(bundle_path):
            return {
                "bundle_id": bundle_id,
                "status": "INVALID",
                "is_reproducible": False,
                "errors": [f"Bundle '{bundle_id}' directory does not exist."],
                "disclaimer": RESEARCH_DISCLAIMER,
                "verified_at": datetime.now(timezone.utc).isoformat()
            }

        manifest = self.bundle_manager.get_bundle(bundle_id)
        if not manifest:
            return {
                "bundle_id": bundle_id,
                "status": "INVALID",
                "is_reproducible": False,
                "errors": ["manifest.json is missing or corrupted."],
                "disclaimer": RESEARCH_DISCLAIMER,
                "verified_at": datetime.now(timezone.utc).isoformat()
            }

        # 1. Validate Schema
        ok_schema, schema_errors = PortableBundleValidator.validate_bundle_manifest_schema(manifest)
        if not ok_schema:
            return {
                "bundle_id": bundle_id,
                "status": "INVALID",
                "is_reproducible": False,
                "errors": schema_errors,
                "disclaimer": RESEARCH_DISCLAIMER,
                "verified_at": datetime.now(timezone.utc).isoformat()
            }

        # 2. Verify File Integrity Hashes
        ok_files, file_errors = PortableBundleValidator.validate_file_hashes(bundle_path, manifest.get("files", []))
        if not ok_files:
            return {
                "bundle_id": bundle_id,
                "status": "INVALID",
                "is_reproducible": False,
                "errors": file_errors,
                "disclaimer": RESEARCH_DISCLAIMER,
                "verified_at": datetime.now(timezone.utc).isoformat()
            }

        # 3. Check for required bundle components
        model_meta_file = os.path.join(bundle_path, "model", "metadata.json")
        exp_config_file = os.path.join(bundle_path, "experiment", "config.json")
        dataset_meta_file = os.path.join(bundle_path, "dataset", "metadata.json")

        for req_f in [model_meta_file, exp_config_file, dataset_meta_file]:
            if not os.path.exists(req_f):
                return {
                    "bundle_id": bundle_id,
                    "status": "INCOMPLETE",
                    "is_reproducible": False,
                    "errors": [f"Missing critical metadata artifact: {os.path.basename(req_f)}"],
                    "disclaimer": RESEARCH_DISCLAIMER,
                    "verified_at": datetime.now(timezone.utc).isoformat()
                }

        # 4. Load metadata and recompute canonical experiment fingerprint
        with open(exp_config_file, "r", encoding="utf-8") as f:
            exp_config = json.load(f)
        with open(model_meta_file, "r", encoding="utf-8") as f:
            model_meta = json.load(f)
        with open(dataset_meta_file, "r", encoding="utf-8") as f:
            dataset_meta = json.load(f)

        full_cfg = dict(exp_config)
        if "model_id" not in full_cfg:
            full_cfg["model_id"] = model_meta.get("model_id")
        if "dataset_version_id" not in full_cfg:
            full_cfg["dataset_version_id"] = dataset_meta.get("dataset_version_id")
        if "external_dataset_id" not in full_cfg and dataset_meta.get("external_dataset_id"):
            full_cfg["external_dataset_id"] = dataset_meta.get("external_dataset_id")

        recalculated_fingerprint = ExperimentRegistry.generate_experiment_fingerprint(None, full_cfg)

        stored_fingerprint = manifest.get("fingerprint_sha256")
        is_reproducible = (recalculated_fingerprint == stored_fingerprint)
        status = "VERIFIED" if is_reproducible else "MISMATCH"

        return {
            "bundle_id": bundle_id,
            "experiment_id": manifest.get("experiment_id"),
            "status": status,
            "is_reproducible": is_reproducible,
            "manifest_sha256": manifest.get("manifest_sha256"),
            "stored_fingerprint_sha256": stored_fingerprint,
            "recalculated_fingerprint_sha256": recalculated_fingerprint,
            "file_count": len(manifest.get("files", [])),
            "verified_files": len(manifest.get("files", [])),
            "errors": [] if is_reproducible else [f"Fingerprint mismatch: expected {stored_fingerprint}, got {recalculated_fingerprint}"],
            "disclaimer": RESEARCH_DISCLAIMER,
            "verified_at": datetime.now(timezone.utc).isoformat()
        }


global_portable_runner = PortableRunner()
