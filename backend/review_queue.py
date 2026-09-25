"""
LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 1.4 — Review Queue Module

Module: review_queue.py
Purpose:
- Implements deterministic review queue generation, filtering, and sorting for multi-study research workflows.
- Strictly avoids subjective priority scores, clinical urgency rankings, or model confidence weightings.
- Provides sanitized ReviewQueueItem representations compliant with study_schema.json.
- Supports lifecycle status filtering (UNREVIEWED, IN_REVIEW, AWAITING_CONSENSUS, ADJUDICATION_REQUIRED, READY_FOR_FINALIZATION, FINALIZED).
"""

import os
import sys
from typing import Dict, List, Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

try:
    from study_manager import global_study_manager, RESEARCH_DISCLAIMER
except ImportError:
    from backend.study_manager import global_study_manager, RESEARCH_DISCLAIMER

LIFECYCLE_ORDER = [
    "UNREVIEWED",
    "IN_REVIEW",
    "AWAITING_CONSENSUS",
    "ADJUDICATION_REQUIRED",
    "READY_FOR_FINALIZATION",
    "FINALIZED"
]


class ReviewQueue:
    """Provides deterministic review queue filtering, indexing, and sorting."""

    def __init__(self, study_manager=None):
        self.study_manager = study_manager or global_study_manager

    def get_queue_items(
        self,
        status: Optional[str] = None,
        consensus_status: Optional[str] = None,
        adjudication_required: Optional[bool] = None,
        finalized: Optional[bool] = None,
        reviewer_count: Optional[int] = None,
        query: Optional[str] = None,
        sort_by: str = "study_id",
        sort_dir: str = "asc"
    ) -> List[Dict[str, Any]]:
        """
        Retrieves review queue items with deterministic filtering and sorting.
        """
        # 1. Fetch all study summaries
        summaries = self.study_manager.discover_studies()

        # 2. Apply Filters
        filtered = []
        for s in summaries:
            # Filter by workflow status
            if status and status.upper() not in ("ALL", ""):
                if s.get("workflow_status", "").upper() != status.upper():
                    continue

            # Filter by consensus status
            if consensus_status and consensus_status.upper() not in ("ALL", ""):
                if s.get("consensus_status", "").upper() != consensus_status.upper():
                    continue

            # Filter by adjudication requirement
            if adjudication_required is not None:
                if s.get("adjudication_required", False) != adjudication_required:
                    continue

            # Filter by finalization state
            if finalized is not None:
                if s.get("consensus_finalized", False) != finalized:
                    continue

            # Filter by reviewer count
            if reviewer_count is not None:
                if s.get("reviewer_count", 0) != reviewer_count:
                    continue

            # Search Query (Study ID prefix or name)
            if query and query.strip():
                clean_q = query.strip().upper()
                if clean_q not in s.get("study_id", "").upper():
                    continue

            # Format as ReviewQueueItem
            views = s.get("views", ["Frontal"])
            primary_view = views[0] if views else "Frontal"

            item = {
                "study_id": s["study_id"],
                "primary_view": primary_view,
                "views": views,
                "evidence_count": s.get("evidence_count", 0),
                "machine_report": s.get("machine_report_available", False),
                "reviewer_count": s.get("reviewer_count", 0),
                "review_completion": s.get("review_completion", 0.0),
                "consensus_status": s.get("consensus_status", "NOT_STARTED"),
                "adjudication_required": s.get("adjudication_required", False),
                "consensus_finalized": s.get("consensus_finalized", False),
                "workflow_status": s.get("workflow_status", "UNREVIEWED"),
                "last_updated": s.get("last_updated", ""),
                "disclaimer": RESEARCH_DISCLAIMER
            }
            filtered.append(item)

        # 3. Deterministic Sorting (Zero subjective clinical prioritization)
        reverse = (sort_dir.lower() == "desc")

        if sort_by == "last_updated":
            filtered.sort(key=lambda x: x.get("last_updated", ""), reverse=reverse)
        elif sort_by == "reviewer_count":
            filtered.sort(key=lambda x: (x.get("reviewer_count", 0), x.get("study_id", "")), reverse=reverse)
        elif sort_by == "review_completion":
            filtered.sort(key=lambda x: (x.get("review_completion", 0.0), x.get("study_id", "")), reverse=reverse)
        elif sort_by == "status" or sort_by == "workflow_status":
            filtered.sort(
                key=lambda x: (
                    LIFECYCLE_ORDER.index(x.get("workflow_status")) if x.get("workflow_status") in LIFECYCLE_ORDER else 99,
                    x.get("study_id", "")
                ),
                reverse=reverse
            )
        else:
            # Default: sort by study_id (natural/alphanumeric order)
            def natural_sort_key(item):
                sid = item.get("study_id", "")
                # Extract trailing digits if any for natural numerical ordering
                import re
                parts = re.split(r'(\d+)', sid)
                return [int(p) if p.isdigit() else p for p in parts]

            filtered.sort(key=natural_sort_key, reverse=reverse)

        return filtered


# Global ReviewQueue Singleton
global_review_queue = ReviewQueue()
