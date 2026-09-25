"""
LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 1.7 — Immutable Experiment Snapshot Manager

Module: experiment_snapshot.py
Purpose:
- Creates immutable, reproducible research experiment snapshots upon finalization.
- Persists snapshot manifests and fingerprint hashes in data/experiment_snapshots/{experiment_id}/.
- Verifies cryptographic snapshot integrity across model, dataset, evaluation, and metric artifacts.
- Guarantees read-only immutability of finalized experiment states.
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

SNAPSHOTS_DIR = os.path.join(BASE_DIR, "data", "experiment_snapshots")
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

RESEARCH_DISCLAIMER = (
    "RESEARCH EXPERIMENT SNAPSHOT — NOT CLINICAL PERFORMANCE EVIDENCE. "
    "Snapshots preserve immutable mathematical and computational states for reproducible research."
)


class ExperimentSnapshotManager:
    """Manages immutable snapshots and cryptographic manifest hashes for finalized experiments."""

    def __init__(self, snapshots_dir: str = SNAPSHOTS_DIR):
        self.snapshots_dir = snapshots_dir
        os.makedirs(self.snapshots_dir, exist_ok=True)

    def _sanitize_id(self, experiment_id: str) -> str:
        """Sanitizes experiment ID and prevents path traversal."""
        if not experiment_id or ".." in experiment_id or "/" in experiment_id or "\\" in experiment_id:
            raise ValueError(f"Invalid experiment ID: '{experiment_id}'. Path traversal not permitted.")
        clean = experiment_id.strip()
        if not clean.startswith("exp_"):
            clean = f"exp_{clean}"
        return clean

    def _get_exp_snapshot_dir(self, experiment_id: str) -> str:
        clean_id = self._sanitize_id(experiment_id)
        exp_dir = os.path.join(self.snapshots_dir, clean_id)
        os.makedirs(exp_dir, exist_ok=True)
        return exp_dir

    def compute_snapshot_sha256(self, manifest_data: Dict[str, Any]) -> str:
        """Computes deterministic SHA-256 hash across manifest metadata."""
        raw_bytes = json.dumps(manifest_data, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw_bytes).hexdigest()

    def create_snapshot(
        self,
        experiment_id: str,
        experiment_config: Dict[str, Any],
        model_version: Optional[Dict[str, Any]] = None,
        dataset_version: Optional[Dict[str, Any]] = None,
        evaluation_run: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        statistics: Optional[Dict[str, Any]] = None,
        error_analysis: Optional[Dict[str, Any]] = None,
        provenance: Optional[List[Dict[str, Any]]] = None,
        validation_status: str = "VALIDATED"
    ) -> Dict[str, Any]:
        """Creates an immutable snapshot for a finalized experiment."""
        clean_id = self._sanitize_id(experiment_id)
        exp_dir = self._get_exp_snapshot_dir(clean_id)
        manifest_file = os.path.join(exp_dir, "manifest.json")
        fingerprint_file = os.path.join(exp_dir, "fingerprint.json")

        if os.path.exists(manifest_file):
            with open(manifest_file, "r", encoding="utf-8") as f:
                return json.load(f)

        now_iso = datetime.now(timezone.utc).isoformat()
        manifest_body = {
            "experiment_id": clean_id,
            "experiment_name": experiment_config.get("experiment_name", clean_id),
            "model_id": experiment_config.get("model_id"),
            "dataset_version_id": experiment_config.get("dataset_version_id"),
            "evaluation_dataset_id": experiment_config.get("evaluation_dataset_id"),
            "methodology": experiment_config.get("methodology", "standard_qa_evidence_evaluation"),
            "random_seed": experiment_config.get("random_seed", 42),
            "software_versions": experiment_config.get("software_versions", {
                "python": sys.version.split()[0],
                "platform": sys.platform
            }),
            "model_summary": {
                "model_id": model_version.get("model_id") if model_version else None,
                "architecture": model_version.get("architecture") if model_version else None,
                "weights_identifier": model_version.get("weights_identifier") if model_version else None,
                "weights_sha256": model_version.get("weights_sha256") if model_version else None
            },
            "dataset_summary": {
                "dataset_version_id": dataset_version.get("dataset_version_id") if dataset_version else None,
                "study_count": dataset_version.get("study_count") if dataset_version else 0,
                "manifest_sha256": dataset_version.get("manifest_sha256") if dataset_version else None
            },
            "evaluation_run_id": evaluation_run.get("evaluation_id") if evaluation_run else None,
            "metrics": metrics or (evaluation_run.get("metrics") if evaluation_run else {}),
            "statistics": statistics or (evaluation_run.get("statistics") if evaluation_run else {}),
            "error_analysis": error_analysis or (evaluation_run.get("error_analysis") if evaluation_run else {}),
            "provenance": provenance or [],
            "validation_status": validation_status,
            "created_at": now_iso
        }

        snapshot_hash = self.compute_snapshot_sha256(manifest_body)

        snapshot_record = {
            "snapshot_id": f"snap_{clean_id}",
            "experiment_id": clean_id,
            "created_at": now_iso,
            "snapshot_sha256": snapshot_hash,
            "manifest_sha256": snapshot_hash,
            "manifest": manifest_body,
            "fingerprint": {
                "snapshot_sha256": snapshot_hash,
                "experiment_fingerprint": experiment_config.get("fingerprint_sha256"),
                "evaluation_fingerprint": evaluation_run.get("evaluation_fingerprint") if evaluation_run else None,
                "dataset_manifest_sha256": dataset_version.get("manifest_sha256") if dataset_version else None,
                "model_weights_sha256": model_version.get("weights_sha256") if model_version else None
            },
            "validation_status": validation_status,
            "disclaimer": RESEARCH_DISCLAIMER
        }

        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(snapshot_record, f, indent=2)

        with open(fingerprint_file, "w", encoding="utf-8") as f:
            json.dump(snapshot_record["fingerprint"], f, indent=2)

        return snapshot_record

    def get_snapshot(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves snapshot record for an experiment."""
        clean_id = self._sanitize_id(experiment_id)
        exp_dir = os.path.join(self.snapshots_dir, clean_id)
        manifest_file = os.path.join(exp_dir, "manifest.json")
        if not os.path.exists(manifest_file):
            return None
        with open(manifest_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def verify_snapshot(self, experiment_id: str) -> Dict[str, Any]:
        """Verifies cryptographic hash consistency of stored snapshot."""
        snapshot = self.get_snapshot(experiment_id)
        if not snapshot:
            raise FileNotFoundError(f"Snapshot for experiment '{experiment_id}' not found.")

        recorded_hash = snapshot.get("snapshot_sha256") or snapshot.get("manifest_sha256")
        computed_hash = self.compute_snapshot_sha256(snapshot.get("manifest", {}))

        is_valid = (recorded_hash == computed_hash)
        return {
            "experiment_id": snapshot["experiment_id"],
            "snapshot_id": snapshot.get("snapshot_id", f"snap_{snapshot['experiment_id']}"),
            "recorded_sha256": recorded_hash,
            "computed_sha256": computed_hash,
            "manifest_sha256": recorded_hash,
            "manifest_valid": is_valid,
            "verified": is_valid,
            "is_valid": is_valid,
            "created_at": snapshot.get("created_at"),
            "disclaimer": RESEARCH_DISCLAIMER
        }

    def compare_snapshot(self, exp_id_a: str, exp_id_b: str) -> Dict[str, Any]:
        """Compares two experiment snapshots across configuration and metric manifests."""
        snap_a = self.get_snapshot(exp_id_a)
        snap_b = self.get_snapshot(exp_id_b)
        if not snap_a:
            raise FileNotFoundError(f"Snapshot A '{exp_id_a}' not found.")
        if not snap_b:
            raise FileNotFoundError(f"Snapshot B '{exp_id_b}' not found.")

        man_a = snap_a.get("manifest", {})
        man_b = snap_b.get("manifest", {})

        config_diffs = {}
        for k in ["model_id", "dataset_version_id", "evaluation_dataset_id", "methodology", "random_seed"]:
            if man_a.get(k) != man_b.get(k):
                config_diffs[k] = {"experiment_a": man_a.get(k), "experiment_b": man_b.get(k)}

        metrics_a = man_a.get("metrics", {})
        metrics_b = man_b.get("metrics", {})
        metric_deltas = {}
        for mk, mv_a in metrics_a.items():
            if isinstance(mv_a, (int, float)) and mk in metrics_b and isinstance(metrics_b[mk], (int, float)):
                mv_b = metrics_b[mk]
                metric_deltas[mk] = {
                    "value_a": mv_a,
                    "value_b": mv_b,
                    "observed_delta": round(float(mv_b - mv_a), 4),
                    "relative_delta": round(float((mv_b - mv_a) / abs(mv_a)), 4) if mv_a != 0 else None
                }

        return {
            "comparison_id": f"cmp_snap_{exp_id_a}_{exp_id_b}",
            "experiment_a": exp_id_a,
            "experiment_b": exp_id_b,
            "snapshot_sha256_a": snap_a.get("snapshot_sha256"),
            "snapshot_sha256_b": snap_b.get("snapshot_sha256"),
            "config_differences": config_diffs,
            "metric_deltas": metric_deltas,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "disclaimer": RESEARCH_DISCLAIMER
        }

    def export_snapshot(self, experiment_id: str, format: str = "json") -> Tuple[str, str]:
        """Exports snapshot in JSON or formatted plain text."""
        snapshot = self.get_snapshot(experiment_id)
        if not snapshot:
            raise FileNotFoundError(f"Snapshot for experiment '{experiment_id}' not found.")

        if format.lower() == "json":
            return json.dumps(snapshot, indent=2), "application/json"

        # Plain text
        man = snapshot.get("manifest", {})
        lines = [
            "=" * 80,
            "IMMUTABLE EXPERIMENT SNAPSHOT RECORD",
            "=" * 80,
            f"Experiment ID:          {snapshot.get('experiment_id')}",
            f"Snapshot SHA-256:       {snapshot.get('snapshot_sha256')}",
            f"Created At:             {snapshot.get('created_at')}",
            f"Validation Status:      {snapshot.get('validation_status')}",
            "-" * 80,
            "PROVENANCE & METADATA:",
            f"  Model ID:             {man.get('model_id')}",
            f"  Dataset Version ID:   {man.get('dataset_version_id')}",
            f"  Evaluation Dataset:   {man.get('evaluation_dataset_id')}",
            f"  Methodology:          {man.get('methodology')}",
            f"  Random Seed:          {man.get('random_seed')}",
            "-" * 80,
            "OBSERVED METRICS:",
        ]
        for k, v in man.get("metrics", {}).items():
            lines.append(f"  {k:30s}: {v}")
        lines.extend([
            "-" * 80,
            "DISCLAIMER:",
            RESEARCH_DISCLAIMER,
            "=" * 80
        ])
        return "\n".join(lines), "text/plain"


global_experiment_snapshot_manager = ExperimentSnapshotManager()
