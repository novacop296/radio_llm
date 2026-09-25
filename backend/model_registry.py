"""
LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 1.7 — Reproducible Experiment Registry & Model Versioning

Module: model_registry.py
Purpose:
- Manages model configurations and model versions.
- Cryptographically tracks model identities, architecture, target layers, weights hashes, and metadata.
- Enforces model lifecycle: REGISTERED -> FINALIZED -> ARCHIVED.
- Guarantees finalized model record immutability.
- Provides model comparison and provenance trails.
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

MODEL_REGISTRY_DIR = os.path.join(BASE_DIR, "data", "model_registry")
os.makedirs(MODEL_REGISTRY_DIR, exist_ok=True)

RESEARCH_DISCLAIMER = (
    "RESEARCH MODEL REGISTRY RECORD — NOT CLINICAL PERFORMANCE EVIDENCE. "
    "Model configurations, weight hashes, and architecture specifications are research metadata. "
    "They do not provide clinical diagnostic guarantees or patient-level medical validation."
)

TXRV_DENSENET_DEFAULT = {
    "model_id": "model_densenet121_txrv",
    "model_name": "TorchXRayVision DenseNet-121 (res224-all)",
    "architecture": "DenseNet-121",
    "framework": "PyTorch / TorchXRayVision",
    "framework_version": "torchxrayvision 1.2.0+",
    "weights_identifier": "densenet121-res224-all",
    "weights_sha256": None,  # Computed if local cache exists or noted
    "input_dimensions": [1, 224, 224],
    "preprocessing": {
        "resize": [224, 224],
        "color_mode": "grayscale_to_1ch",
        "normalization": "txrv_rescale_minmax_to_neg1024_pos1024"
    },
    "target_labels": [
        "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax",
        "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia",
        "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia",
        "Lung Lesion", "Fracture", "Lung Opacity", "Enlarged Cardiomediastinum"
    ],
    "target_layer": "model.features.norm5",
    "description": "Pre-trained DenseNet-121 model specialized for 18 multi-label chest radiograph pathologies.",
    "registered_by": "system_bootstrap",
    "status": "FINALIZED",
    "metadata": {
        "source_library": "torchxrayvision",
        "feature_dim": 1024,
        "gradcam_supported": True,
        "attribution_resolution": [7, 7]
    }
}


class ModelRegistry:
    """Manages registered model architectures, versions, and weight fingerprints."""

    def __init__(self, registry_dir: str = MODEL_REGISTRY_DIR, models_dir: Optional[str] = None):
        self.registry_dir = models_dir or registry_dir
        os.makedirs(self.registry_dir, exist_ok=True)
        self.index_file = os.path.join(self.registry_dir, "registry.json")
        self._ensure_bootstrap_defaults()

    def _sanitize_id(self, model_id: str) -> str:
        """Sanitizes model ID and prevents path traversal."""
        if not model_id or ".." in model_id or "/" in model_id or "\\" in model_id:
            raise ValueError(f"Invalid model ID: '{model_id}'. Path traversal not permitted.")
        clean = model_id.strip()
        if not clean.startswith("model_"):
            clean = f"model_{clean}"
        return clean

    def _ensure_bootstrap_defaults(self):
        """Initializes default DenseNet-121 baseline model in registry."""
        if not os.path.exists(self.index_file):
            with open(self.index_file, "w", encoding="utf-8") as f:
                json.dump({"models": []}, f, indent=2)

        default_id = TXRV_DENSENET_DEFAULT["model_id"]
        model_file = os.path.join(self.registry_dir, f"{default_id}.json")
        if not os.path.exists(model_file):
            now_iso = datetime.now(timezone.utc).isoformat()
            default_entry = dict(TXRV_DENSENET_DEFAULT)
            default_entry["created_at"] = now_iso
            default_entry["disclaimer"] = RESEARCH_DISCLAIMER
            
            # Attempt to locate local weight cache
            weights_hash, note = self._compute_local_weight_hash(default_entry["weights_identifier"])
            default_entry["weights_sha256"] = weights_hash
            if note:
                default_entry["metadata"]["weights_hash_note"] = note

            with open(model_file, "w", encoding="utf-8") as f:
                json.dump(default_entry, f, indent=2)

            self._update_index(default_id, default_entry)

    def _compute_local_weight_hash(self, weights_identifier: str) -> Tuple[Optional[str], Optional[str]]:
        """Computes SHA-256 of downloaded weights cache or notes fallback."""
        cache_dirs = [
            os.path.expanduser(r"~\.torchxrayvision\models_data"),
            os.path.expanduser("~/.torchxrayvision/models_data"),
            os.path.expanduser("~/.cache/torch/hub/checkpoints")
        ]
        for cdir in cache_dirs:
            if os.path.exists(cdir):
                for fname in os.listdir(cdir):
                    if weights_identifier in fname or "densenet121-res224" in fname:
                        fpath = os.path.join(cdir, fname)
                        if os.path.isfile(fpath):
                            try:
                                h = hashlib.sha256()
                                with open(fpath, "rb") as bf:
                                    while chunk := bf.read(65536):
                                        h.update(chunk)
                                return h.hexdigest(), f"Computed from local cache: {fname}"
                            except Exception:
                                pass
        return None, "Weights downloaded dynamically on demand via TorchXRayVision CDN; local weight cache not verified at bootstrap."

    def _update_index(self, model_id: str, model_data: Dict[str, Any]):
        """Updates the master model registry index."""
        index_data = {"models": []}
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, "r", encoding="utf-8") as f:
                    index_data = json.load(f)
            except Exception:
                index_data = {"models": []}

        models_list = index_data.get("models", [])
        # Replace or append
        new_list = [m for m in models_list if m.get("model_id") != model_id]
        new_list.append({
            "model_id": model_id,
            "model_name": model_data.get("model_name"),
            "architecture": model_data.get("architecture"),
            "framework": model_data.get("framework"),
            "weights_identifier": model_data.get("weights_identifier"),
            "status": model_data.get("status", "REGISTERED"),
            "created_at": model_data.get("created_at"),
            "registered_by": model_data.get("registered_by")
        })
        index_data["models"] = new_list
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2)

    def register_model(
        self,
        model_id: str,
        model_name: str,
        architecture: str,
        framework: str,
        framework_version: str,
        weights_identifier: str,
        input_dimensions: List[int],
        preprocessing: Dict[str, Any],
        target_labels: List[str],
        target_layer: str,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        registered_by: str = "researcher",
        weights_sha256: Optional[str] = None
    ) -> Dict[str, Any]:
        """Registers a new model version record."""
        clean_id = self._sanitize_id(model_id)
        model_file = os.path.join(self.registry_dir, f"{clean_id}.json")

        if os.path.exists(model_file):
            raise ValueError(f"Model ID '{clean_id}' is already registered.")

        if not weights_sha256:
            computed_hash, note = self._compute_local_weight_hash(weights_identifier)
            weights_sha256 = computed_hash
            if note and metadata is not None:
                metadata["weights_hash_note"] = note
            elif note:
                metadata = {"weights_hash_note": note}

        now_iso = datetime.now(timezone.utc).isoformat()
        record = {
            "model_id": clean_id,
            "model_name": str(model_name)[:200],
            "architecture": str(architecture),
            "framework": str(framework),
            "framework_version": str(framework_version),
            "weights_identifier": str(weights_identifier),
            "weights_sha256": weights_sha256,
            "input_dimensions": [int(x) for x in input_dimensions],
            "preprocessing": preprocessing or {},
            "target_labels": [str(x) for x in target_labels],
            "target_layer": str(target_layer),
            "created_at": now_iso,
            "registered_by": str(registered_by),
            "description": str(description)[:2000],
            "status": "REGISTERED",
            "metadata": metadata or {},
            "disclaimer": RESEARCH_DISCLAIMER
        }

        with open(model_file, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        self._update_index(clean_id, record)
        return record

    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a registered model by ID."""
        clean_id = self._sanitize_id(model_id)
        model_file = os.path.join(self.registry_dir, f"{clean_id}.json")
        if not os.path.exists(model_file):
            return None
        with open(model_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_models(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Lists all registered models, optionally filtered by status."""
        if not os.path.exists(self.index_file):
            return []
        with open(self.index_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        models = data.get("models", [])
        if status:
            models = [m for m in models if m.get("status") == status]
        return models

    def update_model_before_finalization(self, model_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Updates a model record before it is finalized."""
        clean_id = self._sanitize_id(model_id)
        model_file = os.path.join(self.registry_dir, f"{clean_id}.json")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model '{clean_id}' not found.")

        with open(model_file, "r", encoding="utf-8") as f:
            record = json.load(f)

        if record.get("status") in ["FINALIZED", "ARCHIVED"]:
            raise ValueError(f"Cannot update model '{clean_id}'. Model is already {record.get('status')} and immutable.")

        # Allowed update fields
        allowed_keys = ["model_name", "description", "preprocessing", "target_labels", "target_layer", "metadata"]
        for k, v in updates.items():
            if k in allowed_keys:
                record[k] = v

        record["updated_at"] = datetime.now(timezone.utc).isoformat()
        with open(model_file, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        self._update_index(clean_id, record)
        return record

    def finalize_model(self, model_id: str) -> Dict[str, Any]:
        """Finalizes a registered model, locking it as permanently immutable."""
        clean_id = self._sanitize_id(model_id)
        model_file = os.path.join(self.registry_dir, f"{clean_id}.json")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model '{clean_id}' not found.")

        with open(model_file, "r", encoding="utf-8") as f:
            record = json.load(f)

        if record.get("status") == "FINALIZED":
            return record

        record["status"] = "FINALIZED"
        record["finalized_at"] = datetime.now(timezone.utc).isoformat()

        with open(model_file, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        self._update_index(clean_id, record)
        return record

    def archive_model(self, model_id: str) -> Dict[str, Any]:
        """Archives a model version."""
        clean_id = self._sanitize_id(model_id)
        model_file = os.path.join(self.registry_dir, f"{clean_id}.json")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model '{clean_id}' not found.")

        with open(model_file, "r", encoding="utf-8") as f:
            record = json.load(f)

        record["status"] = "ARCHIVED"
        record["archived_at"] = datetime.now(timezone.utc).isoformat()

        with open(model_file, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        self._update_index(clean_id, record)
        return record

    def verify_model_hash(self, model_id: str) -> Dict[str, Any]:
        """Verifies local weights integrity against recorded hash if weights are accessible."""
        model = self.get_model(model_id)
        if not model:
            raise FileNotFoundError(f"Model '{model_id}' not found.")

        weights_id = model.get("weights_identifier", "")
        recorded_hash = model.get("weights_sha256")
        current_hash, note = self._compute_local_weight_hash(weights_id)

        is_verified = (recorded_hash is not None and current_hash == recorded_hash)
        return {
            "model_id": model["model_id"],
            "weights_identifier": weights_id,
            "weights_sha256": recorded_hash,
            "recorded_sha256": recorded_hash,
            "computed_sha256": current_hash,
            "verified": is_verified,
            "is_valid": (recorded_hash is not None),
            "status_note": note or ("Weights verified matching" if is_verified else "Recorded and computed hashes differ or not accessible"),
            "disclaimer": RESEARCH_DISCLAIMER
        }

    def compare_model_versions(self, model_id_a: str, model_id_b: str) -> Dict[str, Any]:
        """Compares two model versions across architecture, layers, labels, and preprocessing."""
        model_a = self.get_model(model_id_a)
        model_b = self.get_model(model_id_b)
        if not model_a:
            raise FileNotFoundError(f"Model A '{model_id_a}' not found.")
        if not model_b:
            raise FileNotFoundError(f"Model B '{model_id_b}' not found.")

        diffs = {}
        for key in ["architecture", "framework", "framework_version", "weights_identifier", "target_layer", "input_dimensions"]:
            val_a = model_a.get(key)
            val_b = model_b.get(key)
            if val_a != val_b:
                diffs[key] = {"model_a": val_a, "model_b": val_b}

        labels_a = set(model_a.get("target_labels", []))
        labels_b = set(model_b.get("target_labels", []))
        if labels_a != labels_b:
            diffs["target_labels"] = {
                "only_in_a": sorted(list(labels_a - labels_b)),
                "only_in_b": sorted(list(labels_b - labels_a)),
                "shared_count": len(labels_a & labels_b)
            }

        return {
            "comparison_id": f"cmp_models_{model_a['model_id']}_{model_b['model_id']}",
            "model_a": model_a["model_id"],
            "model_b": model_b["model_id"],
            "differences": diffs,
            "is_identical_architecture": len(diffs) == 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "disclaimer": RESEARCH_DISCLAIMER
        }

    def get_model_provenance(self, model_id: str) -> List[Dict[str, Any]]:
        """Returns provenance audit trail for model registration."""
        model = self.get_model(model_id)
        if not model:
            raise FileNotFoundError(f"Model '{model_id}' not found.")

        provenance = [
            {
                "stage": "MODEL_REGISTRATION",
                "timestamp": model.get("created_at"),
                "actor": model.get("registered_by", "system"),
                "status": "REGISTERED",
                "details": f"Registered model architecture {model.get('architecture')} with weights {model.get('weights_identifier')}"
            }
        ]
        if model.get("finalized_at"):
            provenance.append({
                "stage": "MODEL_FINALIZATION",
                "timestamp": model.get("finalized_at"),
                "actor": "system",
                "status": "FINALIZED",
                "details": "Model version locked permanently as immutable."
            })
        if model.get("archived_at"):
            provenance.append({
                "stage": "MODEL_ARCHIVE",
                "timestamp": model.get("archived_at"),
                "actor": "system",
                "status": "ARCHIVED",
                "details": "Model version marked as archived."
            })
        return provenance


global_model_registry = ModelRegistry()
