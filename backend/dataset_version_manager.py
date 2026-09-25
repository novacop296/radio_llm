"""
LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 1.7 — Reproducible Experiment Registry & Dataset Versioning Manager

Module: dataset_version_manager.py
Purpose:
- Manages reproducible dataset versions with cryptographic SHA-256 manifests.
- Tracks study inclusion/exclusion rules, study IDs, parent lineage, and allowed reference annotation scopes.
- Prevents raw reference XML report leakage into manifests, fingerprints, or API responses.
- Enforces dataset version immutability upon finalization.
"""

import os
import sys
import json
import time
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

DATASET_VERSIONS_DIR = os.path.join(BASE_DIR, "data", "dataset_versions")
os.makedirs(DATASET_VERSIONS_DIR, exist_ok=True)

RESEARCH_DISCLAIMER = (
    "RESEARCH DATASET VERSION RECORD — NOT CLINICAL PERFORMANCE EVIDENCE. "
    "Dataset versions, study identifiers, and inclusion criteria are research artifacts. "
    "They do not provide clinical diagnostic guarantees or patient-level medical validation."
)


class DatasetVersionManager:
    """Manages versioned study cohorts, cryptographic manifests, and isolation boundaries."""

    def __init__(self, versions_dir: str = DATASET_VERSIONS_DIR, data_dir: Optional[str] = None):
        self.versions_dir = versions_dir
        self.data_dir = data_dir or os.path.join(BASE_DIR, "data", "iu_xray")
        os.makedirs(self.versions_dir, exist_ok=True)
        self.index_file = os.path.join(self.versions_dir, "registry.json")
        self._ensure_index()

    def _sanitize_id(self, version_id: str) -> str:
        """Sanitizes dataset version ID and prevents path traversal."""
        if not version_id or ".." in version_id or "/" in version_id or "\\" in version_id:
            raise ValueError(f"Invalid dataset version ID: '{version_id}'. Path traversal not permitted.")
        clean = version_id.strip()
        if not clean.startswith("dsv_"):
            clean = f"dsv_{clean}"
        return clean

    def _ensure_index(self):
        """Ensures index file exists."""
        if not os.path.exists(self.index_file):
            with open(self.index_file, "w", encoding="utf-8") as f:
                json.dump({"dataset_versions": []}, f, indent=2)

    def _update_index(self, version_id: str, record: Dict[str, Any]):
        """Updates master dataset versions index."""
        index_data = {"dataset_versions": []}
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, "r", encoding="utf-8") as f:
                    index_data = json.load(f)
            except Exception:
                index_data = {"dataset_versions": []}

        versions_list = index_data.get("dataset_versions", [])
        new_list = [v for v in versions_list if v.get("dataset_version_id") != version_id]
        new_list.append({
            "dataset_version_id": version_id,
            "source_dataset": record.get("source_dataset"),
            "study_count": record.get("study_count", 0),
            "parent_dataset_version": record.get("parent_dataset_version"),
            "manifest_sha256": record.get("manifest_sha256"),
            "status": record.get("status", "DRAFT"),
            "created_at": record.get("created_at"),
            "created_by": record.get("created_by")
        })
        index_data["dataset_versions"] = new_list
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2)

    def compute_manifest_sha256(
        self,
        source_dataset: str,
        study_ids: List[str],
        allowed_reference_annotations: List[str],
        inclusion_rules: Dict[str, Any],
        exclusion_rules: Dict[str, Any]
    ) -> str:
        """Computes deterministic SHA-256 manifest hash excluding all raw XML reference text."""
        canonical = {
            "source_dataset": str(source_dataset),
            "study_ids": sorted(list(set(study_ids))),
            "study_count": len(set(study_ids)),
            "allowed_reference_annotations": sorted(list(set(allowed_reference_annotations))),
            "inclusion_rules": inclusion_rules or {},
            "exclusion_rules": exclusion_rules or {}
        }
        raw_bytes = json.dumps(canonical, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw_bytes).hexdigest()

    def create_dataset_version(
        self,
        dataset_version_id: str,
        source_dataset: str = "IU_XRAY",
        study_ids: Optional[List[str]] = None,
        allowed_reference_annotations: Optional[List[str]] = None,
        inclusion_rules: Optional[Dict[str, Any]] = None,
        exclusion_rules: Optional[Dict[str, Any]] = None,
        parent_dataset_version: Optional[str] = None,
        created_by: str = "researcher",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Creates a new dataset version record."""
        clean_id = self._sanitize_id(dataset_version_id)
        version_file = os.path.join(self.versions_dir, f"{clean_id}.json")

        if os.path.exists(version_file):
            raise ValueError(f"Dataset Version '{clean_id}' already exists.")

        # Default study discovery if not provided
        if study_ids is None:
            discovered = []
            if os.path.exists(self.data_dir):
                for item in os.listdir(self.data_dir):
                    if item.startswith("CXR") and os.path.isdir(os.path.join(self.data_dir, item)):
                        discovered.append(item)
            study_ids = sorted(discovered) if discovered else ["CXR1122"]

        allowed_annos = allowed_reference_annotations or [
            "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax",
            "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia",
            "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia",
            "Lung Lesion", "Fracture", "Lung Opacity", "Enlarged Cardiomediastinum", "Normal"
        ]

        actual_inclusion = inclusion_rules if inclusion_rules is not None else {"min_views": 1, "has_frontal": True}
        actual_exclusion = exclusion_rules if exclusion_rules is not None else {"exclude_corrupt": True}

        manifest_hash = self.compute_manifest_sha256(
            source_dataset=source_dataset,
            study_ids=study_ids,
            allowed_reference_annotations=allowed_annos,
            inclusion_rules=actual_inclusion,
            exclusion_rules=actual_exclusion
        )

        now_iso = datetime.now(timezone.utc).isoformat()
        record = {
            "dataset_version_id": clean_id,
            "source_dataset": str(source_dataset),
            "parent_dataset_version": parent_dataset_version,
            "study_ids": sorted(list(set(study_ids))),
            "study_count": len(set(study_ids)),
            "allowed_reference_annotations": sorted(list(set(allowed_annos))),
            "manifest_sha256": manifest_hash,
            "created_at": now_iso,
            "created_by": str(created_by),
            "inclusion_rules": actual_inclusion,
            "exclusion_rules": actual_exclusion,
            "status": "DRAFT",
            "metadata": metadata or {},
            "disclaimer": RESEARCH_DISCLAIMER
        }

        with open(version_file, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        self._update_index(clean_id, record)
        return record

    def get_dataset_version(self, dataset_version_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a dataset version by ID."""
        clean_id = self._sanitize_id(dataset_version_id)
        version_file = os.path.join(self.versions_dir, f"{clean_id}.json")
        if not os.path.exists(version_file):
            return None
        with open(version_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_dataset_versions(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Lists all registered dataset versions."""
        if not os.path.exists(self.index_file):
            return []
        with open(self.index_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        versions = data.get("dataset_versions", [])
        if status:
            versions = [v for v in versions if v.get("status") == status]
        return versions

    def validate_dataset_version(self, dataset_version_id: str) -> Dict[str, Any]:
        """Validates manifest integrity and consistency."""
        record = self.get_dataset_version(dataset_version_id)
        if not record:
            raise FileNotFoundError(f"Dataset version '{dataset_version_id}' not found.")

        expected_hash = self.compute_manifest_sha256(
            source_dataset=record.get("source_dataset", ""),
            study_ids=record.get("study_ids", []),
            allowed_reference_annotations=record.get("allowed_reference_annotations", []),
            inclusion_rules=record.get("inclusion_rules", {}),
            exclusion_rules=record.get("exclusion_rules", {})
        )

        is_valid = (expected_hash == record.get("manifest_sha256"))
        
        # Check that study_count matches study_ids length
        count_valid = len(record.get("study_ids", [])) == record.get("study_count", 0)

        # Transition status to VALIDATED if in DRAFT
        if is_valid and count_valid and record.get("status") == "DRAFT":
            record["status"] = "VALIDATED"
            version_file = os.path.join(self.versions_dir, f"{record['dataset_version_id']}.json")
            with open(version_file, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
            self._update_index(record["dataset_version_id"], record)

        return {
            "dataset_version_id": record["dataset_version_id"],
            "manifest_sha256": record.get("manifest_sha256"),
            "computed_sha256": expected_hash,
            "manifest_valid": is_valid,
            "count_valid": count_valid,
            "status": record.get("status"),
            "study_count": record.get("study_count"),
            "disclaimer": RESEARCH_DISCLAIMER
        }

    def finalize_dataset_version(self, dataset_version_id: str) -> Dict[str, Any]:
        """Locks dataset version as permanently finalized and immutable."""
        record = self.get_dataset_version(dataset_version_id)
        if not record:
            raise FileNotFoundError(f"Dataset version '{dataset_version_id}' not found.")

        if record.get("status") == "FINALIZED":
            return record

        validation = self.validate_dataset_version(dataset_version_id)
        if not validation.get("manifest_valid") or not validation.get("count_valid"):
            raise ValueError(f"Cannot finalize dataset version '{dataset_version_id}': Validation failed.")

        record = self.get_dataset_version(dataset_version_id)
        record["status"] = "FINALIZED"
        record["finalized_at"] = datetime.now(timezone.utc).isoformat()

        version_file = os.path.join(self.versions_dir, f"{record['dataset_version_id']}.json")
        with open(version_file, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        self._update_index(record["dataset_version_id"], record)
        return record

    def fingerprint_dataset_version(self, dataset_version_id: str) -> Dict[str, Any]:
        """Returns fingerprint payload for reproducible dataset version referencing."""
        record = self.get_dataset_version(dataset_version_id)
        if not record:
            raise FileNotFoundError(f"Dataset version '{dataset_version_id}' not found.")

        return {
            "dataset_version_id": record["dataset_version_id"],
            "manifest_sha256": record.get("manifest_sha256"),
            "study_count": record.get("study_count"),
            "source_dataset": record.get("source_dataset"),
            "status": record.get("status"),
            "allowed_reference_annotations": record.get("allowed_reference_annotations"),
            "created_at": record.get("created_at"),
            "disclaimer": RESEARCH_DISCLAIMER
        }

    def compare_dataset_versions(self, version_a_id: str, version_b_id: str) -> Dict[str, Any]:
        """Compares two dataset versions (cohort overlap, rules, study deltas)."""
        ver_a = self.get_dataset_version(version_a_id)
        ver_b = self.get_dataset_version(version_b_id)
        if not ver_a:
            raise FileNotFoundError(f"Dataset Version A '{version_a_id}' not found.")
        if not ver_b:
            raise FileNotFoundError(f"Dataset Version B '{version_b_id}' not found.")

        studies_a = set(ver_a.get("study_ids", []))
        studies_b = set(ver_b.get("study_ids", []))

        overlap = sorted(list(studies_a & studies_b))
        only_in_a = sorted(list(studies_a - studies_b))
        only_in_b = sorted(list(studies_b - studies_a))

        jaccard = len(overlap) / len(studies_a | studies_b) if (studies_a | studies_b) else 1.0

        return {
            "comparison_id": f"cmp_dsv_{ver_a['dataset_version_id']}_{ver_b['dataset_version_id']}",
            "dataset_version_a": ver_a["dataset_version_id"],
            "dataset_version_b": ver_b["dataset_version_id"],
            "study_count_a": len(studies_a),
            "study_count_b": len(studies_b),
            "shared_study_count": len(overlap),
            "only_in_a_count": len(only_in_a),
            "only_in_b_count": len(only_in_b),
            "jaccard_similarity": round(float(jaccard), 4),
            "is_identical_cohort": (len(only_in_a) == 0 and len(only_in_b) == 0),
            "manifest_a": ver_a.get("manifest_sha256"),
            "manifest_b": ver_b.get("manifest_sha256"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "disclaimer": RESEARCH_DISCLAIMER
        }


global_dataset_version_manager = DatasetVersionManager()
