"""
LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 1.7 — Longitudinal Experiment Tracking & History Manager

Module: experiment_history.py
Purpose:
- Records append-only audit trail events across the lifecycle of experiments.
- Tracks configuration revisions, execution runs, validation transitions, comparisons, and finalization.
- Generates chronological longitudinal timelines for research reproducibility audits.
"""

import os
import sys
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

HISTORY_DIR = os.path.join(BASE_DIR, "data", "experiments", "history")
os.makedirs(HISTORY_DIR, exist_ok=True)

RESEARCH_DISCLAIMER = (
    "RESEARCH AUDIT TRAIL RECORD — NOT CLINICAL PERFORMANCE EVIDENCE. "
    "History entries and lifecycle events are research development audit records."
)


class ExperimentHistoryTracker:
    """Maintains append-only audit logs and longitudinal timelines for experiments."""

    def __init__(self, history_dir: str = HISTORY_DIR):
        self.history_dir = history_dir
        os.makedirs(self.history_dir, exist_ok=True)

    def _sanitize_id(self, experiment_id: str) -> str:
        """Sanitizes experiment ID and prevents path traversal."""
        if not experiment_id or ".." in experiment_id or "/" in experiment_id or "\\" in experiment_id:
            raise ValueError(f"Invalid experiment ID: '{experiment_id}'. Path traversal not permitted.")
        clean = experiment_id.strip()
        if not clean.startswith("exp_"):
            clean = f"exp_{clean}"
        return clean

    def _get_history_file(self, experiment_id: str) -> str:
        clean_id = self._sanitize_id(experiment_id)
        return os.path.join(self.history_dir, f"{clean_id}.json")

    def record_event(
        self,
        experiment_id: str,
        action: str,
        actor: str = "researcher",
        object_type: str = "EXPERIMENT",
        object_id: Optional[str] = None,
        field: Optional[str] = None,
        old_value: Any = None,
        new_value: Any = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Records an append-only audit event."""
        clean_id = self._sanitize_id(experiment_id)
        hfile = self._get_history_file(clean_id)

        events = []
        if os.path.exists(hfile):
            try:
                with open(hfile, "r", encoding="utf-8") as f:
                    events = json.load(f)
            except Exception:
                events = []

        now_iso = datetime.now(timezone.utc).isoformat()
        event_record = {
            "timestamp": now_iso,
            "actor": str(actor),
            "action": str(action),
            "object_type": str(object_type),
            "object_id": str(object_id or clean_id),
            "field": field,
            "old_value": old_value,
            "new_value": new_value,
            "metadata": metadata or {}
        }
        events.append(event_record)

        with open(hfile, "w", encoding="utf-8") as f:
            json.dump(events, f, indent=2)

        return event_record

    def get_history(self, experiment_id: str) -> List[Dict[str, Any]]:
        """Retrieves raw audit event list for an experiment."""
        clean_id = self._sanitize_id(experiment_id)
        hfile = self._get_history_file(clean_id)
        if not os.path.exists(hfile):
            return []
        try:
            with open(hfile, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def get_timeline(self, experiment_id: str) -> Dict[str, Any]:
        """Formats events into a structured chronological research timeline."""
        clean_id = self._sanitize_id(experiment_id)
        events = self.get_history(clean_id)

        formatted_steps = []
        for idx, ev in enumerate(events, 1):
            formatted_steps.append({
                "step": idx,
                "timestamp": ev.get("timestamp"),
                "action": ev.get("action"),
                "actor": ev.get("actor"),
                "summary": f"[{ev.get('action')}] by {ev.get('actor')} on {ev.get('object_type')}:{ev.get('object_id')}" + (f" (field: {ev.get('field')})" if ev.get("field") else ""),
                "details": ev.get("metadata", {})
            })

        return {
            "experiment_id": clean_id,
            "total_events": len(events),
            "first_event_at": events[0]["timestamp"] if events else None,
            "latest_event_at": events[-1]["timestamp"] if events else None,
            "timeline": formatted_steps,
            "disclaimer": RESEARCH_DISCLAIMER
        }


global_experiment_history = ExperimentHistoryTracker()
