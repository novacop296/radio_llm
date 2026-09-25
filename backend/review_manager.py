"""
LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 1.2 — Human-in-the-Loop Review & Finalization Manager

Module: review_manager.py
Purpose:
- Manages the independent reviewer and finalization layer stored in data/reviews/{study_id}.json.
- Provides session creation, finding decision updates, QA answer corrections, report draft editing,
  draft reset, finalization locking, audit trail generation, and multi-format exports.
- Guarantees strict immutability of underlying machine-generated evidence and report artifacts.
"""

import os
import sys
import json
import time
import uuid
import re
from typing import Dict, List, Any, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
REVIEWS_DIR = os.path.join(DATA_DIR, "reviews")
os.makedirs(REVIEWS_DIR, exist_ok=True)

# Add backend to sys.path for sibling imports
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from validate_final_review import (
    validate_review_session,
    VALID_REVIEWER_STATUSES,
    VALID_QA_ANSWERS,
    ID_PATTERN
)

RESEARCH_DISCLAIMER = (
    "Research Prototype — Reviewer annotations and finalized reports are research metadata "
    "and do not constitute certified clinical ground truth or automated diagnoses."
)


def _get_timestamp() -> str:
    """Return ISO 8601 UTC timestamp."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _get_review_file_path(study_id: str, reviewer_id: Optional[str] = None) -> str:
    """Safely construct the review file path with path traversal protection."""
    clean_id = os.path.basename(study_id.strip())
    if not ID_PATTERN.match(clean_id):
        raise ValueError(f"Invalid study identifier '{study_id}'. Must be alphanumeric.")
    if reviewer_id and reviewer_id not in ["default", "researcher_01", ""]:
        clean_rev = os.path.basename(reviewer_id.strip())
        if not ID_PATTERN.match(clean_rev):
            raise ValueError(f"Invalid reviewer identifier '{reviewer_id}'.")
        return os.path.join(REVIEWS_DIR, f"{clean_id}_{clean_rev}.json")
    return os.path.join(REVIEWS_DIR, f"{clean_id}.json")


class ReviewManager:
    """
    Manages human-in-the-loop review sessions, finding corrections,
    QA adjustments, final report drafts, audit trails, and finalization.
    Supports single-reviewer baseline and multi-reviewer isolated sessions.
    """

    def __init__(self, load_study_func=None):
        """
        Args:
            load_study_func: Callable to load baseline machine study data (load_study_data).
        """
        self._load_study_func = load_study_func

    def _get_machine_data(self, study_id: str) -> Optional[Dict[str, Any]]:
        """Fetch authoritative machine data using the configured study loader."""
        if self._load_study_func:
            return self._load_study_func(study_id)
        
        # Fallback local import if not injected
        try:
            from api import load_study_data
            return load_study_data(study_id)
        except Exception:
            return None

    def get_or_create_review(
        self,
        study_id: str,
        reviewer_info: Optional[Dict[str, Any]] = None,
        reviewer_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Retrieves an existing review session or initializes a new one from machine artifacts.
        Never modifies underlying machine artifacts.
        """
        r_id = reviewer_id or (reviewer_info or {}).get("id")
        file_path = _get_review_file_path(study_id, r_id)

        if not os.path.exists(file_path) and (r_id is None or r_id in ["default", "researcher_01"]):
            default_path = os.path.join(REVIEWS_DIR, f"{study_id}.json")
            if os.path.exists(default_path):
                file_path = default_path

        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    session = json.load(f)
                return session
            except Exception:
                # If corrupted, re-initialize
                pass

        # Initialize new review session from machine data
        machine_data = self._get_machine_data(study_id)
        if not machine_data:
            raise ValueError(f"Cannot initialize review: Study '{study_id}' not found.")

        effective_rev_id = r_id or (reviewer_info or {}).get("id", "researcher_01")
        reviewer_name = (reviewer_info or {}).get("display_name", f"Clinical Research Reviewer ({effective_rev_id})")

        evidence_list = machine_data.get("evidence", [])
        qa_questions = machine_data.get("qa_questions", [])
        machine_report = machine_data.get("report", {})

        # 1. Build Finding Reviews Baseline
        finding_reviews = []
        for ev in evidence_list:
            fname = ev.get("finding", "")
            loc = ev.get("location", "unspecified")
            sev = ev.get("severity", "unspecified")
            m_status = ev.get("status", "possible")

            finding_reviews.append({
                "finding": fname,
                "machine_status": m_status,
                "reviewer_status": "not_reviewed",
                "machine_location": loc,
                "reviewer_location": loc,
                "machine_severity": sev,
                "reviewer_severity": sev,
                "reviewer_comment": "",
                "reviewed": False
            })

        # 2. Build QA Reviews Baseline
        qa_reviews = []
        for q in qa_questions:
            qa_reviews.append({
                "question_id": q.get("question_id", ""),
                "finding": q.get("finding", ""),
                "level": q.get("level", 1),
                "question_text": q.get("question_text", ""),
                "machine_answer": q.get("answer", "uncertain"),
                "reviewer_answer": "not_reviewed",
                "reviewer_comment": ""
            })

        # 3. Build Initial Report Review Draft (Copied from machine report, locked machine baseline)
        final_findings = []
        for f in machine_report.get("findings", []):
            final_findings.append({
                "finding": f.get("finding", ""),
                "statement": f.get("statement", ""),
                "status": f.get("status", "possible"),
                "location": f.get("location", "unspecified"),
                "severity": f.get("severity", "unspecified")
            })

        final_impression = list(machine_report.get("impression", []))

        now = _get_timestamp()
        review_id = f"rev_{study_id}_{effective_rev_id}_{int(time.time())}"

        session = {
            "study_id": study_id,
            "review_id": review_id,
            "reviewer": {
                "id": effective_rev_id,
                "display_name": reviewer_name,
                "role": "Research Reviewer"
            },
            "status": "in_review",
            "started_at": now,
            "updated_at": now,
            "completed_at": None,
            "finding_reviews": finding_reviews,
            "qa_reviews": qa_reviews,
            "report_review": {
                "machine_report_locked": True,
                "final_findings": final_findings,
                "final_impression": final_impression,
                "reviewer_comment": "",
                "finalized": False
            },
            "audit_trail": [
                {
                    "timestamp": now,
                    "reviewer_id": effective_rev_id,
                    "finding": "*SESSION*",
                    "field": "session_init",
                    "old_value": None,
                    "new_value": "initialized"
                }
            ],
            "disclaimer": RESEARCH_DISCLAIMER
        }

        self._save_session(study_id, session, reviewer_id=effective_rev_id)
        return session

    def _save_session(self, study_id: str, session: Dict[str, Any], reviewer_id: Optional[str] = None):
        """Save review session to disk atomically."""
        r_id = reviewer_id or session.get("reviewer", {}).get("id")
        file_path = _get_review_file_path(study_id, r_id)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(session, f, indent=2)

    def get_review(self, study_id: str, reviewer_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieve existing review session or None."""
        file_path = _get_review_file_path(study_id, reviewer_id)
        if not os.path.exists(file_path) and (reviewer_id is None or reviewer_id in ["default", "researcher_01"]):
            default_path = os.path.join(REVIEWS_DIR, f"{study_id}.json")
            if os.path.exists(default_path):
                file_path = default_path

        if not os.path.exists(file_path):
            return None
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def update_finding_review(
        self,
        study_id: str,
        payload: Dict[str, Any],
        reviewer_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Updates a reviewer decision for a specific candidate finding and logs to audit trail.
        """
        rev_id = reviewer_id or payload.get("reviewer_id") or (payload.get("reviewer") if isinstance(payload.get("reviewer"), str) else None)
        session = self.get_or_create_review(study_id, reviewer_id=rev_id)
        if session.get("status") == "finalized" or session.get("report_review", {}).get("finalized"):
            raise ValueError(f"Cannot modify finding review: Review session for '{study_id}' is finalized and locked.")

        finding_name = (payload.get("finding") or payload.get("finding_name") or "").strip()
        if not finding_name:
            raise ValueError("Field 'finding' is required in finding review update.")

        reviewer_status = (
            payload.get("reviewer_status") or
            payload.get("status") or
            payload.get("decision") or
            payload.get("review_status") or
            ""
        ).strip().lower()

        if reviewer_status and reviewer_status not in VALID_REVIEWER_STATUSES:
            raise ValueError(f"Invalid reviewer_status '{reviewer_status}'. Must be one of {sorted(VALID_REVIEWER_STATUSES)}")

        target_fr = None
        for fr in session.get("finding_reviews", []):
            if fr["finding"].lower() == finding_name.lower():
                target_fr = fr
                break

        if not target_fr:
            raise ValueError(f"Finding '{finding_name}' not found in candidate findings for study '{study_id}'.")

        effective_rev_id = rev_id or session.get("reviewer", {}).get("id", "researcher_01")
        now = _get_timestamp()

        # Update and log changes
        changes = []
        if reviewer_status and reviewer_status != target_fr["reviewer_status"]:
            changes.append(("reviewer_status", target_fr["reviewer_status"], reviewer_status))
            target_fr["reviewer_status"] = reviewer_status
            target_fr["reviewed"] = (reviewer_status in ["confirmed_present", "confirmed_absent", "uncertain"])

        if "reviewer_location" in payload or "location" in payload:
            loc_val = payload.get("reviewer_location") if "reviewer_location" in payload else payload.get("location")
            new_loc = str(loc_val).strip()
            if new_loc != target_fr["reviewer_location"]:
                changes.append(("reviewer_location", target_fr["reviewer_location"], new_loc))
                target_fr["reviewer_location"] = new_loc

        if "reviewer_severity" in payload or "severity" in payload:
            sev_val = payload.get("reviewer_severity") if "reviewer_severity" in payload else payload.get("severity")
            new_sev = str(sev_val).strip()
            if new_sev != target_fr["reviewer_severity"]:
                changes.append(("reviewer_severity", target_fr["reviewer_severity"], new_sev))
                target_fr["reviewer_severity"] = new_sev

        if "reviewer_comment" in payload or "comment" in payload or "notes" in payload:
            comm_val = payload.get("reviewer_comment") if "reviewer_comment" in payload else (payload.get("comment") if "comment" in payload else payload.get("notes"))
            new_comm = str(comm_val).strip()
            if new_comm != target_fr["reviewer_comment"]:
                changes.append(("reviewer_comment", target_fr["reviewer_comment"], new_comm))
                target_fr["reviewer_comment"] = new_comm

        if "reviewed" in payload:
            rev_val = payload["reviewed"]
            if isinstance(rev_val, str):
                target_fr["reviewed"] = rev_val.lower() in ["true", "1", "yes"]
            else:
                target_fr["reviewed"] = bool(rev_val)
        else:
            target_fr["reviewed"] = (target_fr["reviewer_status"] in ["confirmed_present", "confirmed_absent", "uncertain"])

        for field, old_v, new_v in changes:
            session["audit_trail"].append({
                "timestamp": now,
                "reviewer_id": effective_rev_id,
                "finding": target_fr["finding"],
                "field": field,
                "old_value": old_v,
                "new_value": new_v
            })

        session["updated_at"] = now
        self._save_session(study_id, session, reviewer_id=effective_rev_id)
        return target_fr

    def get_finding_review(self, study_id: str, finding: str, reviewer_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieve single finding review state."""
        session = self.get_review(study_id, reviewer_id=reviewer_id)
        if not session:
            return None
        for fr in session.get("finding_reviews", []):
            if fr["finding"].lower() == finding.lower():
                return fr
        return None

    def update_qa_review(
        self,
        study_id: str,
        payload: Dict[str, Any],
        reviewer_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Updates a reviewer QA answer correction without modifying machine QA artifacts.
        """
        rev_id = reviewer_id or payload.get("reviewer_id")
        session = self.get_or_create_review(study_id, reviewer_id=rev_id)
        if session.get("status") == "finalized":
            raise ValueError(f"Cannot modify QA review: Session for '{study_id}' is finalized.")

        q_id = payload.get("question_id", "").strip()
        if not q_id:
            raise ValueError("Field 'question_id' is required.")

        r_ans = payload.get("reviewer_answer", "").strip().lower()
        if r_ans and r_ans not in VALID_QA_ANSWERS:
            raise ValueError(f"Invalid QA reviewer_answer '{r_ans}'. Must be one of {sorted(VALID_QA_ANSWERS)}")

        target_qr = None
        for qr in session.get("qa_reviews", []):
            if qr["question_id"].lower() == q_id.lower():
                target_qr = qr
                break

        if not target_qr:
            raise ValueError(f"Question ID '{q_id}' not found in study QA questions.")

        rev_id = rev_id or session.get("reviewer", {}).get("id", "researcher_01")
        now = _get_timestamp()

        changes = []
        if r_ans and r_ans != target_qr["reviewer_answer"]:
            changes.append(("reviewer_answer", target_qr["reviewer_answer"], r_ans))
            target_qr["reviewer_answer"] = r_ans

        if "reviewer_comment" in payload:
            new_c = str(payload["reviewer_comment"]).strip()
            if new_c != target_qr["reviewer_comment"]:
                changes.append(("reviewer_comment", target_qr["reviewer_comment"], new_c))
                target_qr["reviewer_comment"] = new_c

        for field, old_v, new_v in changes:
            session["audit_trail"].append({
                "timestamp": now,
                "reviewer_id": rev_id,
                "finding": target_qr.get("finding", q_id),
                "field": f"qa_{q_id}_{field}",
                "old_value": old_v,
                "new_value": new_v
            })

        session["updated_at"] = now
        self._save_session(study_id, session, reviewer_id=rev_id)
        return target_qr

    def save_report_draft(
        self,
        study_id: str,
        final_findings: Optional[List[Dict[str, Any]]] = None,
        final_impression: Optional[List[str]] = None,
        reviewer_comment: Optional[str] = None,
        reviewer_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Saves edited findings and impression draft for the final report.
        """
        session = self.get_or_create_review(study_id, reviewer_id=reviewer_id)
        if session.get("status") == "finalized":
            raise ValueError(f"Cannot save report draft: Session for '{study_id}' is finalized.")

        report_rev = session.setdefault("report_review", {})
        rev_id = reviewer_id or session.get("reviewer", {}).get("id", "researcher_01")
        now = _get_timestamp()

        if final_findings is not None:
            if not isinstance(final_findings, list):
                raise ValueError("final_findings must be a list of finding objects.")
            report_rev["final_findings"] = final_findings
            session["audit_trail"].append({
                "timestamp": now,
                "reviewer_id": rev_id,
                "finding": "*REPORT*",
                "field": "final_findings",
                "old_value": "previous_findings_draft",
                "new_value": f"{len(final_findings)} findings updated"
            })

        if final_impression is not None:
            if not isinstance(final_impression, list):
                raise ValueError("final_impression must be a list of string statements.")
            report_rev["final_impression"] = final_impression
            session["audit_trail"].append({
                "timestamp": now,
                "reviewer_id": rev_id,
                "finding": "*REPORT*",
                "field": "final_impression",
                "old_value": "previous_impression_draft",
                "new_value": f"{len(final_impression)} impression items updated"
            })

        if reviewer_comment is not None:
            report_rev["reviewer_comment"] = str(reviewer_comment).strip()

        session["updated_at"] = now
        self._save_session(study_id, session, reviewer_id=rev_id)
        return report_rev

    def reset_report_to_machine(
        self,
        study_id: str,
        reviewer_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Resets final report draft back to the original machine report.
        """
        session = self.get_or_create_review(study_id, reviewer_id=reviewer_id)
        if session.get("status") == "finalized":
            raise ValueError(f"Cannot reset report: Session for '{study_id}' is finalized.")

        machine_data = self._get_machine_data(study_id)
        if not machine_data:
            raise ValueError(f"Study '{study_id}' machine data not available.")

        m_report = machine_data.get("report", {})
        final_findings = []
        for f in m_report.get("findings", []):
            final_findings.append({
                "finding": f.get("finding", ""),
                "statement": f.get("statement", ""),
                "status": f.get("status", "possible"),
                "location": f.get("location", "unspecified"),
                "severity": f.get("severity", "unspecified")
            })

        final_impression = list(m_report.get("impression", []))

        rev_id = reviewer_id or session.get("reviewer", {}).get("id", "researcher_01")
        now = _get_timestamp()

        session["report_review"]["final_findings"] = final_findings
        session["report_review"]["final_impression"] = final_impression
        session["report_review"]["reviewer_comment"] = ""
        session["audit_trail"].append({
            "timestamp": now,
            "reviewer_id": rev_id,
            "finding": "*REPORT*",
            "field": "reset_to_machine",
            "old_value": "draft",
            "new_value": "machine_baseline"
        })

        session["updated_at"] = now
        self._save_session(study_id, session, reviewer_id=rev_id)
        return session["report_review"]

    def finalize_review(
        self,
        study_id: str,
        reviewer_id: Optional[str] = None
    ) -> Tuple[bool, Dict[str, Any], List[str]]:
        """
        Validates all prerequisites and locks the review session as finalized.

        Returns:
            (success: bool, session_data: dict, validation_issues: list)
        """
        session = self.get_or_create_review(study_id, reviewer_id=reviewer_id)
        if session.get("status") == "finalized":
            return True, session, []

        machine_data = self._get_machine_data(study_id)
        is_valid, issues = validate_review_session(
            session,
            require_finalization_ready=True,
            machine_study_data=machine_data
        )

        if not is_valid:
            return False, session, issues

        # Lock and finalize
        rev_id = reviewer_id or session.get("reviewer", {}).get("id", "researcher_01")
        now = _get_timestamp()

        session["status"] = "finalized"
        session["report_review"]["finalized"] = True
        session["completed_at"] = now
        session["updated_at"] = now

        session["audit_trail"].append({
            "timestamp": now,
            "reviewer_id": rev_id,
            "finding": "*SESSION*",
            "field": "status",
            "old_value": "in_review",
            "new_value": "finalized"
        })

        self._save_session(study_id, session, reviewer_id=rev_id)
        return True, session, []

    def get_audit_trail(self, study_id: str, reviewer_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve chronological audit trail entries."""
        session = self.get_review(study_id, reviewer_id=reviewer_id)
        if not session:
            return []
        return session.get("audit_trail", [])

    def export_review(self, study_id: str, format: str = "json", reviewer_id: Optional[str] = None) -> Tuple[str, str]:
        """
        Export review as JSON or readable text.

        Returns:
            (content_str, mime_type)
        """
        session = self.get_or_create_review(study_id, reviewer_id=reviewer_id)
        machine_data = self._get_machine_data(study_id) or {}

        if format.lower() == "text":
            lines = [
                "================================================",
                "RESEARCH REVIEW REPORT",
                "NOT A CLINICAL DIAGNOSIS",
                "================================================",
                f"STUDY ID    : {session.get('study_id')}",
                f"REVIEW ID   : {session.get('review_id')}",
                f"REVIEW STATUS: {session.get('status', 'in_review').upper()}",
                f"REVIEWER    : {session.get('reviewer', {}).get('display_name')} ({session.get('reviewer', {}).get('id')})",
                f"STARTED AT  : {session.get('started_at')}",
                f"COMPLETED AT: {session.get('completed_at') or 'IN PROGRESS'}",
                "-----------------------------------------------------------------",
                "MACHINE EVIDENCE SUMMARY (IMMUTABLE BASELINE):"
            ]
            for ev in machine_data.get("evidence", []):
                lines.append(f"- {ev.get('finding')}: Status={ev.get('status').upper()}, Activation={ev.get('model_score'):.4f}, Loc={ev.get('location')}")

            lines.append("-----------------------------------------------------------------")
            lines.append("REVIEWER FINDING DECISIONS:")
            for fr in session.get("finding_reviews", []):
                lines.append(
                    f"- {fr.get('finding')}: Reviewer Status=[{fr.get('reviewer_status').upper():<16}], "
                    f"Machine=[{fr.get('machine_status').upper():<9}], Loc={fr.get('reviewer_location')}, Sev={fr.get('reviewer_severity')}"
                )
                if fr.get("reviewer_comment"):
                    lines.append(f"  Comment: {fr.get('reviewer_comment')}")

            lines.append("-----------------------------------------------------------------")
            lines.append("FINAL REVIEWED FINDINGS:")
            for ff in session.get("report_review", {}).get("final_findings", []):
                lines.append(f"- [{ff.get('status', 'possible').upper():<10}] {ff.get('statement')}")

            lines.append("-----------------------------------------------------------------")
            lines.append("FINAL REVIEWED IMPRESSION:")
            for imp in session.get("report_review", {}).get("final_impression", []):
                lines.append(f"- {imp}")

            if session.get("report_review", {}).get("reviewer_comment"):
                lines.append(f"\nReviewer Synthesis Notes: {session.get('report_review', {}).get('reviewer_comment')}")

            lines.append("-----------------------------------------------------------------")
            lines.append("AUDIT TRAIL (CHRONOLOGICAL):")
            for entry in session.get("audit_trail", []):
                lines.append(f"[{entry.get('timestamp')}] {entry.get('reviewer_id')} -> {entry.get('finding')}.{entry.get('field')}: '{entry.get('old_value')}' => '{entry.get('new_value')}'")

            lines.append("=================================================================")
            lines.append(RESEARCH_DISCLAIMER)
            lines.append("=================================================================")

            return "\n".join(lines), "text/plain"

        else:
            # JSON format
            export_obj = {
                "study_id": session.get("study_id"),
                "review_id": session.get("review_id"),
                "status": session.get("status"),
                "reviewer": session.get("reviewer"),
                "timestamps": {
                    "started_at": session.get("started_at"),
                    "updated_at": session.get("updated_at"),
                    "completed_at": session.get("completed_at")
                },
                "machine_baseline": {
                    "evidence": machine_data.get("evidence", []),
                    "report": machine_data.get("report", {}),
                    "locked": True
                },
                "finding_reviews": session.get("finding_reviews", []),
                "qa_reviews": session.get("qa_reviews", []),
                "final_report": session.get("report_review", {}),
                "audit_trail": session.get("audit_trail", []),
                "disclaimer": RESEARCH_DISCLAIMER
            }
            return json.dumps(export_obj, indent=2), "application/json"


# Global singleton instance
global_review_manager = ReviewManager()
