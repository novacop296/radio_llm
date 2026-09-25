"""
LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 1.8 — External Dataset Registry & Multi-View Cohort Manager

Module: external_dataset_manager.py
Purpose:
- Manages registration, validation, manifest generation, and lifecycle transitions for external research datasets.
- Isolates external dataset metadata under data/external_datasets/.
- Supports multi-view cohort discovery, frontal/lateral view pairing, and missing-view detection.
- Generates deterministic SHA-256 manifest fingerprints.
- Enforces strict immutability once finalized and guards against ground-truth/secret leakage.
"""

import os
import sys
import json
import time
import re
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

EXTERNAL_DATASETS_DIR = os.path.join(BASE_DIR, "data", "external_datasets")
IU_XRAY_DIR = os.path.join(BASE_DIR, "data", "iu_xray")

RESEARCH_DISCLAIMER = (
    "RESEARCH PROTOTYPE — NOT CLINICAL PERFORMANCE EVIDENCE. External datasets and benchmark definitions "
    "are registered for reproducibility and cross-distribution research evaluation only."
)


class ExternalDatasetManager:
    """Manages isolated external dataset metadata, multi-view pairing, and manifest fingerprinting."""

    def __init__(self, storage_dir: str = EXTERNAL_DATASETS_DIR, iu_xray_dir: str = IU_XRAY_DIR):
        self.storage_dir = storage_dir
        self.iu_xray_dir = iu_xray_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self._registry_file = os.path.join(self.storage_dir, "registry.json")
        self._ensure_registry()

    def _ensure_registry(self):
        if not os.path.exists(self._registry_file):
            initial = {
                "schema_version": "1.8.0",
                "disclaimer": RESEARCH_DISCLAIMER,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "datasets": {}
            }
            with open(self._registry_file, "w", encoding="utf-8") as f:
                json.dump(initial, f, indent=2)

    def _get_dataset_file(self, dataset_id: str) -> str:
        self._validate_id(dataset_id)
        return os.path.join(self.storage_dir, f"{dataset_id}.json")

    def _validate_id(self, dataset_id: str):
        if not dataset_id or not isinstance(dataset_id, str):
            raise ValueError("Dataset ID must be a non-empty string.")
        if any(bad in dataset_id for bad in ["..", "/", "\\", "\0", ":", "*", "?", '"', "<", ">", "|"]):
            raise ValueError(f"Invalid dataset ID '{dataset_id}': path traversal or illegal characters detected.")
        if not re.match(r"^[a-zA-Z0-9_-]+$", dataset_id):
            raise ValueError(f"Dataset ID '{dataset_id}' must match ^[a-zA-Z0-9_-]+$")

    def _validate_finite_numbers(self, obj: Any) -> Tuple[bool, Optional[str]]:
        if isinstance(obj, float):
            if obj != obj or obj == float('inf') or obj == float('-inf'):
                return False, "NaN or Infinity detected in numeric field."
        elif isinstance(obj, dict):
            for k, v in obj.items():
                ok, err = self._validate_finite_numbers(v)
                if not ok:
                    return False, f"{k}: {err}"
        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                ok, err = self._validate_finite_numbers(item)
                if not ok:
                    return False, f"Index {idx}: {err}"
        return True, None

    def _validate_no_secrets(self, obj: Any) -> Tuple[bool, Optional[str]]:
        secret_keys = {"api_key", "password", "secret", "token", "auth", "private_key"}
        if isinstance(obj, dict):
            for k, v in obj.items():
                if any(sk in k.lower() for sk in secret_keys):
                    if v and str(v).strip():
                        return False, f"Potential secret field detected: '{k}'"
                ok, err = self._validate_no_secrets(v)
                if not ok:
                    return False, err
        elif isinstance(obj, list):
            for item in obj:
                ok, err = self._validate_no_secrets(item)
                if not ok:
                    return False, err
        return True, None

    def _validate_no_gt_leakage(self, obj: Any) -> Tuple[bool, Optional[str]]:
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
                ok, err = self._validate_no_gt_leakage(v)
                if not ok:
                    return False, err
        elif isinstance(obj, list):
            for item in obj:
                ok, err = self._validate_no_gt_leakage(item)
                if not ok:
                    return False, err
        return True, None

    def compute_manifest_sha256(self, dataset_record: Dict[str, Any]) -> str:
        """Generates deterministic canonical SHA-256 fingerprint for the external dataset manifest."""
        canonical_dict = {
            "dataset_id": dataset_record.get("dataset_id"),
            "dataset_name": dataset_record.get("dataset_name"),
            "dataset_version": dataset_record.get("dataset_version"),
            "modality": dataset_record.get("modality", "CHEST_XRAY"),
            "image_views": sorted(dataset_record.get("image_views", [])),
            "study_count": dataset_record.get("study_count", 0),
            "image_count": dataset_record.get("image_count", 0),
            "label_schema": sorted(dataset_record.get("label_schema", [])),
            "permitted_annotations": sorted(dataset_record.get("permitted_annotations", [])),
            "preprocessing_definition": dataset_record.get("preprocessing_definition", {}),
            "split_definition": dataset_record.get("split_definition", {}),
            "studies": [
                {
                    "study_id": s.get("study_id"),
                    "views": sorted(s.get("views", [])),
                    "view_pair_id": s.get("view_pair_id"),
                    "is_complete_pair": s.get("is_complete_pair", False),
                    "images": [
                        {
                            "image_id": img.get("image_id"),
                            "view": img.get("view"),
                            "relative_path": img.get("relative_path", "").replace("\\", "/")
                        }
                        for img in sorted(s.get("images", []), key=lambda x: x.get("image_id", ""))
                    ]
                }
                for s in sorted(dataset_record.get("studies", []), key=lambda x: x.get("study_id", ""))
            ]
        }
        raw_json = json.dumps(canonical_dict, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(raw_json.encode('utf-8')).hexdigest()

    def detect_view_pairs(self, studies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validates and attaches view pair metadata to studies without cross-study mismatch."""
        processed_studies = []
        for s in studies:
            s_copy = dict(s)
            study_id = s_copy.get("study_id", "")
            images = s_copy.get("images", [])
            views = set()
            frontal_imgs = []
            lateral_imgs = []

            for img in images:
                v = img.get("view", "Unknown")
                views.add(v)
                if v == "Frontal":
                    frontal_imgs.append(img)
                elif v == "Lateral":
                    lateral_imgs.append(img)

            s_copy["views"] = sorted(list(views))
            if frontal_imgs and lateral_imgs:
                s_copy["view_pair_id"] = f"pair_{study_id}_F{len(frontal_imgs)}_L{len(lateral_imgs)}"
                s_copy["is_complete_pair"] = True
            else:
                s_copy["view_pair_id"] = None
                s_copy["is_complete_pair"] = False

            processed_studies.append(s_copy)
        return processed_studies

    def register_external_dataset(
        self,
        dataset_id: str,
        dataset_name: str,
        dataset_version: str = "v1.0",
        source_description: str = "",
        institution_or_source: str = "External Research Institution",
        modality: str = "CHEST_XRAY",
        image_views: Optional[List[str]] = None,
        studies: Optional[List[Dict[str, Any]]] = None,
        label_schema: Optional[List[str]] = None,
        permitted_annotations: Optional[List[str]] = None,
        preprocessing_definition: Optional[Dict[str, Any]] = None,
        split_definition: Optional[Dict[str, Any]] = None,
        license_metadata: Optional[Dict[str, Any]] = None,
        access_status: str = "SYNTHETIC_FIXTURE",
        local_path_or_reference: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Registers a new external dataset reference with deterministic fingerprint."""
        self._validate_id(dataset_id)
        filepath = self._get_dataset_file(dataset_id)

        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if existing.get("status") in ["FINALIZED", "ARCHIVED"]:
                raise ValueError(f"External dataset '{dataset_id}' is finalized/archived and immutable.")

        raw_studies = studies or []
        paired_studies = self.detect_view_pairs(raw_studies)
        image_count = sum(len(s.get("images", [])) for s in paired_studies)
        study_count = len(paired_studies)

        inferred_views = set(image_views or [])
        for s in paired_studies:
            for v in s.get("views", []):
                inferred_views.add(v)
        if not inferred_views:
            inferred_views = {"Frontal"}

        now = datetime.now(timezone.utc).isoformat()
        record = {
            "schema_version": "1.8.0",
            "disclaimer": RESEARCH_DISCLAIMER,
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "dataset_version": dataset_version,
            "source_description": source_description,
            "institution_or_source": institution_or_source,
            "modality": modality,
            "image_views": sorted(list(inferred_views)),
            "study_count": study_count,
            "image_count": image_count,
            "studies": paired_studies,
            "label_schema": label_schema or ["Normal", "Cardiomegaly", "Pulmonary Edema", "Consolidation", "Pleural Effusion"],
            "permitted_annotations": permitted_annotations or [],
            "preprocessing_definition": preprocessing_definition or {"resize": [224, 224], "normalization": "standard_cxr"},
            "split_definition": split_definition or {"train": 0.0, "val": 0.0, "test": 1.0},
            "license_metadata": license_metadata or {"license_name": "Research Only", "commercial_use": False, "attribution_required": True},
            "access_status": access_status,
            "local_path_or_reference": local_path_or_reference,
            "status": "REGISTERED",
            "created_at": now,
            "finalized_at": None,
            "provenance": [
                {
                    "timestamp": now,
                    "action": "REGISTERED",
                    "actor": "research_system",
                    "details": f"Registered external dataset '{dataset_id}' with {study_count} studies, {image_count} images."
                }
            ],
            "metadata": metadata or {}
        }

        # Safety checks
        for validator in [self._validate_finite_numbers, self._validate_no_secrets, self._validate_no_gt_leakage]:
            ok, err = validator(record)
            if not ok:
                raise ValueError(f"Safety validation failed: {err}")

        manifest_hash = self.compute_manifest_sha256(record)
        record["manifest_sha256"] = manifest_hash

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        self._update_registry(record)
        return record

    def _update_registry(self, record: Dict[str, Any]):
        with open(self._registry_file, "r", encoding="utf-8") as f:
            reg = json.load(f)

        reg["datasets"][record["dataset_id"]] = {
            "dataset_id": record["dataset_id"],
            "dataset_name": record["dataset_name"],
            "dataset_version": record["dataset_version"],
            "modality": record["modality"],
            "image_views": record["image_views"],
            "study_count": record["study_count"],
            "image_count": record["image_count"],
            "access_status": record["access_status"],
            "status": record["status"],
            "manifest_sha256": record["manifest_sha256"],
            "created_at": record["created_at"],
            "finalized_at": record["finalized_at"]
        }
        reg["updated_at"] = datetime.now(timezone.utc).isoformat()

        with open(self._registry_file, "w", encoding="utf-8") as f:
            json.dump(reg, f, indent=2)

    def get_external_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        self._validate_id(dataset_id)
        filepath = self._get_dataset_file(dataset_id)
        if not os.path.exists(filepath):
            return None
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_external_datasets(self) -> List[Dict[str, Any]]:
        with open(self._registry_file, "r", encoding="utf-8") as f:
            reg = json.load(f)
        return list(reg.get("datasets", {}).values())

    def validate_external_dataset(self, dataset_id: str) -> Tuple[bool, List[str]]:
        """Performs full schema, fingerprint, and view pairing validation."""
        record = self.get_external_dataset(dataset_id)
        if not record:
            return False, [f"Dataset '{dataset_id}' not found."]

        errors = []
        required = [
            "dataset_id", "dataset_name", "dataset_version", "modality",
            "image_views", "study_count", "image_count", "manifest_sha256", "status"
        ]
        for req in required:
            if req not in record:
                errors.append(f"Missing required property: '{req}'")

        # Fingerprint match
        current_hash = self.compute_manifest_sha256(record)
        if current_hash != record.get("manifest_sha256"):
            errors.append(f"Manifest fingerprint mismatch: expected {record.get('manifest_sha256')}, calculated {current_hash}")

        # Invariant checks
        for val_fn, name in [
            (self._validate_finite_numbers, "finite numbers"),
            (self._validate_no_secrets, "no secrets"),
            (self._validate_no_gt_leakage, "no ground truth leakage")
        ]:
            ok, err = val_fn(record)
            if not ok:
                errors.append(f"Safety check '{name}' failed: {err}")

        # If valid and in REGISTERED / VALIDATING, transition to VALIDATED
        if not errors and record["status"] in ["REGISTERED", "VALIDATING"]:
            record["status"] = "VALIDATED"
            now = datetime.now(timezone.utc).isoformat()
            record["provenance"].append({
                "timestamp": now,
                "action": "VALIDATED",
                "actor": "system_validator",
                "details": "Dataset schema and manifest fingerprint successfully validated."
            })
            filepath = self._get_dataset_file(dataset_id)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
            self._update_registry(record)

        return len(errors) == 0, errors

    def finalize_external_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """Locks an external dataset definition permanently as FINALIZED (immutable)."""
        record = self.get_external_dataset(dataset_id)
        if not record:
            raise ValueError(f"Dataset '{dataset_id}' not found.")

        if record.get("status") == "FINALIZED":
            return record

        is_valid, errors = self.validate_external_dataset(dataset_id)
        if not is_valid:
            raise ValueError(f"Cannot finalize invalid dataset '{dataset_id}': {'; '.join(errors)}")

        now = datetime.now(timezone.utc).isoformat()
        record["status"] = "FINALIZED"
        record["finalized_at"] = now
        record["provenance"].append({
            "timestamp": now,
            "action": "FINALIZED",
            "actor": "researcher",
            "details": "Dataset version manifest frozen permanently."
        })

        filepath = self._get_dataset_file(dataset_id)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        self._update_registry(record)
        return record

    def archive_external_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """Transitions dataset to ARCHIVED state."""
        record = self.get_external_dataset(dataset_id)
        if not record:
            raise ValueError(f"Dataset '{dataset_id}' not found.")

        now = datetime.now(timezone.utc).isoformat()
        record["status"] = "ARCHIVED"
        record["provenance"].append({
            "timestamp": now,
            "action": "ARCHIVED",
            "actor": "researcher",
            "details": "Dataset archived."
        })

        filepath = self._get_dataset_file(dataset_id)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        self._update_registry(record)
        return record

    def verify_dataset_integrity(self, dataset_id: str) -> Tuple[bool, str]:
        """Verifies cryptographic hash consistency."""
        record = self.get_external_dataset(dataset_id)
        if not record:
            return False, f"Dataset '{dataset_id}' does not exist."
        calculated = self.compute_manifest_sha256(record)
        if calculated == record.get("manifest_sha256"):
            return True, f"Integrity verified: {calculated}"
        return False, f"Tampering detected: stored={record.get('manifest_sha256')}, calculated={calculated}"


global_external_dataset_manager = ExternalDatasetManager()
