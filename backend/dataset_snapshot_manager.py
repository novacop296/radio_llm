"""
LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 1.5 — Immutable Dataset Snapshot Manager

Module: dataset_snapshot_manager.py
Purpose:
- Creates and manages immutable dataset snapshots for reproducible experiment tracking.
- Computes SHA-256 study manifest hashes and machine source artifact checksums.
- Strictly separates dataset versions to ensure experiment evaluation consistency.
"""

import os
import sys
import json
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

try:
    from backend.study_manager import global_study_manager
except ImportError:
    from study_manager import global_study_manager

DATA_DIR = os.path.join(BASE_DIR, "data", "iu_xray")
SNAPSHOTS_DIR = os.path.join(BASE_DIR, "data", "snapshots")
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

RESEARCH_DISCLAIMER = (
    "RESEARCH EVALUATION ONLY — NOT CLINICAL PERFORMANCE. "
    "Dataset snapshots and experiment metrics are for investigative reproducibility only. "
    "They do not constitute certified clinical trial cohorts or diagnostic validation standards."
)


class DatasetSnapshotManager:
    """Manages immutable dataset snapshots and cryptographic manifest hashes."""

    def __init__(
        self,
        snapshots_dir: str = SNAPSHOTS_DIR,
        data_dir: str = DATA_DIR,
        study_manager=None
    ):
        self.snapshots_dir = snapshots_dir
        self.data_dir = data_dir
        self.study_manager = study_manager or global_study_manager
        os.makedirs(self.snapshots_dir, exist_ok=True)

    def _sanitize_id(self, snapshot_id: str) -> str:
        """Sanitizes snapshot ID and prevents directory traversal."""
        if not snapshot_id or ".." in snapshot_id or "/" in snapshot_id or "\\" in snapshot_id:
            raise ValueError(f"Invalid snapshot ID: '{snapshot_id}'. Path traversal not permitted.")
        clean = snapshot_id.strip()
        if not clean.startswith("snap_"):
            clean = f"snap_{clean}"
        return clean

    def create_snapshot(
        self,
        name: str,
        description: str = "",
        study_ids: Optional[List[str]] = None,
        created_by: str = "researcher",
        custom_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Creates a new immutable dataset snapshot.
        If study_ids is None, discovers all currently available studies.
        """
        if not name or len(name.strip()) == 0:
            raise ValueError("Snapshot 'name' is required.")
        if len(name) > 200:
            raise ValueError("Snapshot 'name' must not exceed 200 characters.")
        if len(description) > 2000:
            raise ValueError("Snapshot 'description' must not exceed 2000 characters.")

        # Determine study IDs
        if study_ids is None:
            summaries = self.study_manager.discover_studies()
            resolved_study_ids = [s["study_id"] for s in summaries]
        else:
            if not isinstance(study_ids, list) or len(study_ids) == 0:
                raise ValueError("Provided 'study_ids' must be a non-empty list of strings.")
            resolved_study_ids = []
            for sid in study_ids:
                clean_s = sid.strip().upper()
                if clean_s not in resolved_study_ids:
                    resolved_study_ids.append(clean_s)

        # Sort study IDs deterministically
        sorted_study_ids = sorted(resolved_study_ids)

        # Compute deterministic manifest hash
        manifest_raw = json.dumps(sorted_study_ids, sort_keys=True).encode("utf-8")
        manifest_sha256 = hashlib.sha256(manifest_raw).hexdigest()

        # Generate Snapshot ID
        now_ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if custom_id:
            snap_id = self._sanitize_id(custom_id)
            file_path = os.path.join(self.snapshots_dir, f"{snap_id}.json")
            if os.path.exists(file_path):
                raise ValueError(f"Dataset snapshot '{snap_id}' already exists and is immutable. Use a new ID.")
        else:
            time_slug = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
            hash_slug = manifest_sha256[:8]
            snap_id = f"snap_{time_slug}_{hash_slug}"
            idx = 1
            while os.path.exists(os.path.join(self.snapshots_dir, f"{snap_id}.json")):
                snap_id = f"snap_{time_slug}_{hash_slug}_{idx}"
                idx += 1
            file_path = os.path.join(self.snapshots_dir, f"{snap_id}.json")


        # Compute view counts and artifact checksums
        view_counts: Dict[str, int] = {}
        source_artifact_hashes: Dict[str, str] = {}

        for sid in sorted_study_ids:
            summary = self.study_manager.get_study_summary(sid)
            if summary:
                for v in summary.get("views", ["Frontal"]):
                    view_counts[v] = view_counts.get(v, 0) + 1
            else:
                view_counts["Frontal"] = view_counts.get("Frontal", 0) + 1

        # Check primary baseline files checksums if available
        baseline_files = [
            ("evidence", os.path.join(self.data_dir, "evidence", "CXR1122_evidence_output.json")),
            ("grounding", os.path.join(self.data_dir, "grounding", "CXR1122_CXR1122_IM-0080-1001-0002_infiltration_overlay.png")),
            ("report", os.path.join(self.data_dir, "reports", "CXR1122_llm_output.json"))
        ]
        for name_key, fp in baseline_files:
            if os.path.exists(fp):
                try:
                    with open(fp, "rb") as f:
                        source_artifact_hashes[name_key] = hashlib.sha256(f.read()).hexdigest()
                except Exception:
                    pass

        snapshot_record = {
            "snapshot_id": snap_id,
            "name": name.strip(),
            "description": description.strip(),
            "created_at": now_ts,
            "created_by": created_by.strip() or "researcher",
            "study_count": len(sorted_study_ids),
            "view_counts": view_counts,
            "dataset_source": "IU_XRAY_RESEARCH_CORPUS",
            "study_ids": sorted_study_ids,
            "manifest_sha256": manifest_sha256,
            "source_artifact_hashes": source_artifact_hashes,
            "schema_version": "1.0",
            "disclaimer": RESEARCH_DISCLAIMER
        }

        # Write immutable record
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(snapshot_record, f, indent=2)

        return snapshot_record

    def get_snapshot(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a specific dataset snapshot by ID."""
        try:
            clean_id = self._sanitize_id(snapshot_id)
        except ValueError:
            return None

        file_path = os.path.join(self.snapshots_dir, f"{clean_id}.json")
        if not os.path.exists(file_path):
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def list_snapshots(self) -> List[Dict[str, Any]]:
        """Lists all available dataset snapshots sorted by creation date."""
        snapshots = []
        if not os.path.exists(self.snapshots_dir):
            return []

        for fn in sorted(os.listdir(self.snapshots_dir)):
            if not fn.endswith(".json"):
                continue
            fp = os.path.join(self.snapshots_dir, fn)
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    snapshots.append(data)
            except Exception:
                pass

        snapshots.sort(key=lambda s: s.get("created_at", ""), reverse=True)
        return snapshots

    def validate_snapshot_integrity(self, snapshot_id: str) -> Tuple[bool, List[str]]:
        """Verifies cryptographic hash consistency of a dataset snapshot."""
        errors = []
        snap = self.get_snapshot(snapshot_id)
        if not snap:
            return False, [f"Snapshot '{snapshot_id}' does not exist."]

        study_ids = snap.get("study_ids", [])
        sorted_study_ids = sorted(study_ids)
        manifest_raw = json.dumps(sorted_study_ids, sort_keys=True).encode("utf-8")
        computed_hash = hashlib.sha256(manifest_raw).hexdigest()

        if computed_hash != snap.get("manifest_sha256"):
            errors.append(f"Manifest hash mismatch: recorded {snap.get('manifest_sha256')} vs computed {computed_hash}")

        if snap.get("study_count") != len(study_ids):
            errors.append(f"Study count mismatch: recorded {snap.get('study_count')} vs actual {len(study_ids)}")

        return (len(errors) == 0), errors


# Global Singleton
global_snapshot_manager = DatasetSnapshotManager()
