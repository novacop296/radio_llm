"""
LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 1.3 — Multi-Reviewer Consensus, Inter-Rater Agreement & Adjudication Manager

Module: consensus_manager.py
Purpose:
- Manages multi-reviewer consensus sessions stored in data/consensus/{study_id}.json.
- Collects independent reviewer sessions without cross-contaminating reviewer records.
- Computes deterministic finding, location, severity, and QA consensus.
- Calculates statistical inter-rater agreement metrics:
  * Reviewer Agreement Ratio
  * Cohen's Kappa (for exactly 2 reviewers)
  * Fleiss' Kappa (for 3 or more reviewers)
- Manages dispute adjudication workflow with mandatory justifications.
- Synthesizes and finalizes immutable consensus reports.
- Exports consensus packages in JSON and human-readable plain-text.
"""

import os
import sys
import json
import time
import math
import uuid
import re
from typing import Dict, List, Any, Optional, Tuple, Set

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CONSENSUS_DIR = os.path.join(DATA_DIR, "consensus")
REVIEWS_DIR = os.path.join(DATA_DIR, "reviews")
os.makedirs(CONSENSUS_DIR, exist_ok=True)
os.makedirs(REVIEWS_DIR, exist_ok=True)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from review_manager import global_review_manager, _get_review_file_path

ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")

RESEARCH_DISCLAIMER = (
    "RESEARCH CONSENSUS OUTPUT — NOT A CLINICAL DIAGNOSIS. "
    "Consensus results, reviewer annotations, inter-rater agreement metrics, "
    "adjudication decisions, and consensus reports are research metadata only."
)

VALID_FINDING_CATEGORIES = ["confirmed_present", "confirmed_absent", "uncertain", "needs_review"]


def _get_timestamp() -> str:
    """Return ISO 8601 UTC timestamp."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _get_consensus_file_path(study_id: str) -> str:
    """Construct path for study consensus file with security validation."""
    clean_id = os.path.basename(study_id.strip())
    if not ID_PATTERN.match(clean_id):
        raise ValueError(f"Invalid study identifier '{study_id}'. Must be alphanumeric.")
    return os.path.join(CONSENSUS_DIR, f"{clean_id}.json")


class ConsensusManager:
    """
    Manages multi-reviewer consensus sessions, inter-rater agreement calculations,
    adjudications, and consensus report finalization.
    """

    def __init__(self, review_mgr=None, load_study_func=None):
        self._review_mgr = review_mgr or global_review_manager
        self._load_study_func = load_study_func

    def _get_machine_data(self, study_id: str) -> Optional[Dict[str, Any]]:
        """Fetch authoritative machine baseline data."""
        if self._load_study_func:
            return self._load_study_func(study_id)
        try:
            from api import load_study_data
            return load_study_data(study_id)
        except Exception:
            return None

    def get_consensus_session(self, study_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve existing consensus session or None."""
        file_path = _get_consensus_file_path(study_id)
        if not os.path.exists(file_path):
            return None
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_consensus_session(self, study_id: str, session: Dict[str, Any]):
        """Save consensus session to disk."""
        file_path = _get_consensus_file_path(study_id)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(session, f, indent=2)

    def create_consensus_session(
        self,
        study_id: str,
        required_reviewers: int = 3,
        minimum_reviewers: int = 2,
        reviewers_list: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Initializes a new or updated consensus session for a study.
        """
        machine_data = self._get_machine_data(study_id)
        if not machine_data:
            raise ValueError(f"Cannot initialize consensus: Study '{study_id}' not found.")

        existing = self.get_consensus_session(study_id)
        now = _get_timestamp()

        if existing and existing.get("status") == "finalized":
            return existing

        consensus_id = existing.get("consensus_id") if existing else f"cons_{study_id}_{int(time.time())}"
        created_at = existing.get("created_at") if existing else now

        # Build / merge reviewer roster
        reviewers = existing.get("reviewers", []) if existing else []
        if reviewers_list:
            for r in reviewers_list:
                r_id = r.get("id", "").strip()
                if not r_id:
                    continue
                if not ID_PATTERN.match(r_id):
                    raise ValueError(f"Invalid reviewer ID '{r_id}'.")
                if not any(ex["id"] == r_id for ex in reviewers):
                    reviewers.append({
                        "id": r_id,
                        "display_name": r.get("display_name", f"Reviewer {r_id}"),
                        "role": r.get("role", "Clinical Reviewer"),
                        "review_status": "registered",
                        "session_file": f"{study_id}_{r_id}.json",
                        "completed_at": None
                    })

        # Default standard 3-reviewer cohort if none provided
        if not reviewers:
            for i in range(1, required_reviewers + 1):
                r_id = f"reviewer_{i:03d}"
                reviewers.append({
                    "id": r_id,
                    "display_name": f"Clinical Reviewer {i:03d}",
                    "role": "Clinical Reviewer",
                    "review_status": "registered",
                    "session_file": f"{study_id}_{r_id}.json",
                    "completed_at": None
                })

        session = {
            "study_id": study_id,
            "consensus_id": consensus_id,
            "status": "collecting",
            "created_at": created_at,
            "updated_at": now,
            "completed_at": None,
            "required_reviewers": max(2, int(required_reviewers)),
            "minimum_reviewers": max(2, int(minimum_reviewers)),
            "reviewers": reviewers,
            "finding_consensus": [],
            "qa_consensus": [],
            "agreement_metrics": {
                "reviewer_count": 0,
                "finding_count": 0,
                "average_agreement_ratio": None,
                "cohens_kappa": None,
                "fleiss_kappa": None
            },
            "adjudication": {
                "items_requiring_adjudication": [],
                "adjudication_records": [],
                "all_resolved": True
            },
            "consensus_report": {
                "final_findings": [],
                "final_impression": [],
                "reviewer_summary": "",
                "agreement_summary": "",
                "adjudication_summary": "",
                "reviewer_comment": "",
                "finalized": False
            },
            "audit_trail": existing.get("audit_trail", []) if existing else [
                {
                    "timestamp": now,
                    "user_id": "system",
                    "action": "consensus_initialized",
                    "target": "*SESSION*",
                    "old_value": None,
                    "new_value": "initialized"
                }
            ],
            "disclaimer": RESEARCH_DISCLAIMER
        }

        # Auto-collect existing reviewer data and calculate initial consensus
        self._sync_and_recalculate(study_id, session)
        self._save_consensus_session(study_id, session)
        return session

    def register_reviewer(
        self,
        study_id: str,
        reviewer_id: str,
        display_name: str,
        role: str = "Clinical Reviewer"
    ) -> Dict[str, Any]:
        """Registers a reviewer into the study consensus cohort."""
        clean_rev = reviewer_id.strip()
        if not ID_PATTERN.match(clean_rev):
            raise ValueError(f"Invalid reviewer ID '{reviewer_id}'.")

        session = self.get_consensus_session(study_id)
        if not session:
            session = self.create_consensus_session(study_id)

        if session.get("status") == "finalized":
            raise ValueError(f"Cannot register reviewer: Consensus session for '{study_id}' is finalized.")

        for r in session["reviewers"]:
            if r["id"] == clean_rev:
                r["display_name"] = display_name
                r["role"] = role
                self._save_consensus_session(study_id, session)
                return r

        new_entry = {
            "id": clean_rev,
            "display_name": display_name,
            "role": role,
            "review_status": "registered",
            "session_file": f"{study_id}_{clean_rev}.json",
            "completed_at": None
        }
        session["reviewers"].append(new_entry)
        session["audit_trail"].append({
            "timestamp": _get_timestamp(),
            "user_id": clean_rev,
            "action": "reviewer_registered",
            "target": clean_rev,
            "old_value": None,
            "new_value": display_name
        })

        self._sync_and_recalculate(study_id, session)
        self._save_consensus_session(study_id, session)
        return new_entry

    def collect_reviewer_sessions(self, study_id: str) -> List[Dict[str, Any]]:
        """
        Collects all independent reviewer session objects for the study.
        Guarantees non-destructive read-only collection.
        """
        session = self.get_consensus_session(study_id)
        if not session:
            return []

        completed_sessions = []
        for r in session.get("reviewers", []):
            r_id = r["id"]
            rev_data = self._review_mgr.get_review(study_id, reviewer_id=r_id)
            if rev_data:
                is_done = rev_data.get("status") in ["reviewed", "finalized"] or all(
                    fr.get("reviewed", False) for fr in rev_data.get("finding_reviews", [])
                )
                r["review_status"] = "completed" if is_done else "in_progress"
                r["completed_at"] = rev_data.get("completed_at")
                if is_done:
                    completed_sessions.append(rev_data)
            else:
                r["review_status"] = "registered"

        return completed_sessions

    def _sync_and_recalculate(self, study_id: str, session: Dict[str, Any]):
        """Internal helper to refresh reviewer data, consensus, and metrics."""
        machine_data = self._get_machine_data(study_id) or {}
        evidence_list = machine_data.get("evidence", [])
        qa_questions = machine_data.get("qa_questions", [])

        # 1. Collect reviewer sessions
        all_reviewer_sessions = []
        completed_sessions = []
        for r in session.get("reviewers", []):
            r_id = r["id"]
            rev_data = self._review_mgr.get_review(study_id, reviewer_id=r_id)
            if rev_data:
                # Check completeness
                f_reviews = rev_data.get("finding_reviews", [])
                all_reviewed = len(f_reviews) > 0 and all(fr.get("reviewed", False) for fr in f_reviews)
                is_completed = rev_data.get("status") in ["reviewed", "finalized"] or all_reviewed
                r["review_status"] = "completed" if is_completed else "in_progress"
                r["completed_at"] = rev_data.get("completed_at")
                all_reviewer_sessions.append(rev_data)
                if is_completed:
                    completed_sessions.append(rev_data)
            else:
                r["review_status"] = "registered"

        # Active sessions that have any finding reviews
        active_sessions = [
            s for s in all_reviewer_sessions
            if any(fr.get("reviewer_status") != "not_reviewed" for fr in s.get("finding_reviews", []))
        ]
        eval_sessions = active_sessions if len(active_sessions) >= 2 else (completed_sessions if len(completed_sessions) >= 2 else all_reviewer_sessions)

        n_completed = len(completed_sessions)
        min_req = session.get("minimum_reviewers", 2)

        # 2. Finding Consensus Calculation
        finding_consensus_list = []
        adjudication_items = []
        adjudication_records = session.get("adjudication", {}).get("adjudication_records", [])

        # Map existing adjudication decisions
        adj_map = {f"{r['target_type']}_{r['target_id']}": r for r in adjudication_records}

        for ev in evidence_list:
            fname = ev.get("finding", "")
            m_stat = ev.get("status", "possible")
            m_score = ev.get("model_score", 0.0)
            m_loc = ev.get("location", "unspecified")
            m_sev = ev.get("severity", "unspecified")

            reviewer_decisions = []
            for s in eval_sessions:
                r_info = s.get("reviewer", {})
                fr = next((f for f in s.get("finding_reviews", []) if f.get("finding", "").lower() == fname.lower()), None)
                if fr and fr.get("reviewer_status") != "not_reviewed":
                    reviewer_decisions.append({
                        "reviewer_id": r_info.get("id", "unknown"),
                        "reviewer_name": r_info.get("display_name", "Reviewer"),
                        "decision": fr.get("reviewer_status", "not_reviewed"),
                        "location": fr.get("reviewer_location", "unspecified"),
                        "severity": fr.get("reviewer_severity", "unspecified"),
                        "comment": fr.get("reviewer_comment", ""),
                        "reviewed": fr.get("reviewed", False)
                    })

            # Consensus logic
            n_rev = len(reviewer_decisions)
            dist: Dict[str, int] = {}
            for rd in reviewer_decisions:
                d = rd["decision"]
                dist[d] = dist.get(d, 0) + 1

            consensus_decision = None
            consensus_status = "pending_reviews"
            agreement_ratio = None
            loc_consensus = "unspecified"
            loc_status = "unspecified"
            sev_consensus = "unspecified"
            sev_status = "unspecified"

            if n_rev >= min_req:
                # Check for needs_review
                if dist.get("needs_review", 0) > 0:
                    consensus_status = "adjudication_required"
                else:
                    # Find modal decision
                    modal_dec, modal_count = None, 0
                    for dec_name, count in dist.items():
                        if dec_name == "not_reviewed":
                            continue
                        if count > modal_count:
                            modal_dec, modal_count = dec_name, count
                        elif count == modal_count:
                            modal_dec = None  # Tie

                    agreement_ratio = round(modal_count / n_rev, 4) if n_rev > 0 else 0.0

                    if modal_count == n_rev and modal_dec is not None:
                        consensus_status = "unanimous"
                        consensus_decision = modal_dec
                    elif modal_dec is not None and modal_count > (n_rev / 2.0):
                        consensus_status = "majority"
                        consensus_decision = modal_dec
                    else:
                        consensus_status = "adjudication_required"
                        consensus_decision = None

                # Location consensus
                loc_dist: Dict[str, int] = {}
                for rd in reviewer_decisions:
                    loc = rd.get("location", "unspecified")
                    loc_dist[loc] = loc_dist.get(loc, 0) + 1
                modal_loc, modal_loc_cnt = max(loc_dist.items(), key=lambda x: x[1]) if loc_dist else ("unspecified", 0)
                if modal_loc_cnt == n_rev:
                    loc_consensus = modal_loc
                    loc_status = "unanimous" if modal_loc != "unspecified" else "unspecified"
                elif modal_loc_cnt > (n_rev / 2.0):
                    loc_consensus = modal_loc
                    loc_status = "majority"
                else:
                    loc_consensus = None
                    loc_status = "adjudication_required"

                # Severity consensus
                sev_dist: Dict[str, int] = {}
                for rd in reviewer_decisions:
                    sev = rd.get("severity", "unspecified")
                    sev_dist[sev] = sev_dist.get(sev, 0) + 1
                modal_sev, modal_sev_cnt = max(sev_dist.items(), key=lambda x: x[1]) if sev_dist else ("unspecified", 0)
                if modal_sev_cnt == n_rev:
                    sev_consensus = modal_sev
                    sev_status = "unanimous" if modal_sev != "unspecified" else "unspecified"
                elif modal_sev_cnt > (n_rev / 2.0):
                    sev_consensus = modal_sev
                    sev_status = "majority"
                else:
                    sev_consensus = None
                    sev_status = "adjudication_required"

            # Apply adjudication if exists
            adj_key = f"finding_{fname}"
            if adj_key in adj_map:
                consensus_decision = adj_map[adj_key]["decision"]
                consensus_status = "majority"  # Resolved by adjudication

            adj_loc_key = f"location_{fname}"
            if adj_loc_key in adj_map:
                loc_consensus = adj_map[adj_loc_key]["decision"]
                loc_status = "majority"

            adj_sev_key = f"severity_{fname}"
            if adj_sev_key in adj_map:
                sev_consensus = adj_map[adj_sev_key]["decision"]
                sev_status = "majority"

            # Flag for adjudication if unresolved
            if consensus_status == "adjudication_required" and adj_key not in adj_map:
                adjudication_items.append({
                    "target_type": "finding",
                    "target_id": fname,
                    "issue": f"No consensus reached on status for '{fname}' ({dist})",
                    "status": "pending",
                    "resolved_decision": None
                })
            elif adj_key in adj_map:
                adjudication_items.append({
                    "target_type": "finding",
                    "target_id": fname,
                    "issue": f"Adjudicated finding status",
                    "status": "resolved",
                    "resolved_decision": adj_map[adj_key]["decision"]
                })

            if loc_status == "adjudication_required" and adj_loc_key not in adj_map:
                adjudication_items.append({
                    "target_type": "location",
                    "target_id": fname,
                    "issue": f"Location disagreement for '{fname}' ({loc_dist})",
                    "status": "pending",
                    "resolved_decision": None
                })

            if sev_status == "adjudication_required" and adj_sev_key not in adj_map:
                adjudication_items.append({
                    "target_type": "severity",
                    "target_id": fname,
                    "issue": f"Severity disagreement for '{fname}' ({sev_dist})",
                    "status": "pending",
                    "resolved_decision": None
                })

            finding_consensus_list.append({
                "finding": fname,
                "machine_status": m_stat,
                "machine_score": m_score,
                "machine_location": m_loc,
                "machine_severity": m_sev,
                "reviewer_decisions": reviewer_decisions,
                "decision_distribution": dist,
                "consensus_decision": consensus_decision,
                "consensus_status": consensus_status,
                "agreement_ratio": agreement_ratio,
                "location_consensus": loc_consensus,
                "location_status": loc_status,
                "severity_consensus": sev_consensus,
                "severity_status": sev_status
            })

        # 3. QA Consensus Calculation
        qa_consensus_list = []
        for q in qa_questions:
            q_id = q.get("question_id", "")
            f_name = q.get("finding", "")
            lvl = q.get("level", 1)
            q_text = q.get("question_text", "")
            m_ans = q.get("answer", "uncertain")

            r_answers = []
            for s in eval_sessions:
                r_info = s.get("reviewer", {})
                qr = next((x for x in s.get("qa_reviews", []) if x.get("question_id", "").lower() == q_id.lower()), None)
                if qr and qr.get("reviewer_answer") != "not_reviewed":
                    r_answers.append({
                        "reviewer_id": r_info.get("id", "unknown"),
                        "answer": qr.get("reviewer_answer", ""),
                        "comment": qr.get("reviewer_comment", "")
                    })

            # Consensus
            qa_dist: Dict[str, int] = {}
            for ra in r_answers:
                a = ra["answer"]
                qa_dist[a] = qa_dist.get(a, 0) + 1

            cons_ans = None
            agr_status = "pending"
            n_qa_rev = len(r_answers)

            if n_qa_rev >= min_req:
                m_ans_top, top_cnt = max(qa_dist.items(), key=lambda x: x[1]) if qa_dist else (None, 0)
                if top_cnt == n_qa_rev:
                    cons_ans = m_ans_top
                    agr_status = "unanimous"
                elif top_cnt > (n_qa_rev / 2.0):
                    cons_ans = m_ans_top
                    agr_status = "majority"
                else:
                    cons_ans = None
                    agr_status = "adjudication_required"

            adj_qa_key = f"qa_{q_id}"
            if adj_qa_key in adj_map:
                cons_ans = adj_map[adj_qa_key]["decision"]
                agr_status = "majority"

            if agr_status == "adjudication_required" and adj_qa_key not in adj_map:
                adjudication_items.append({
                    "target_type": "qa",
                    "target_id": q_id,
                    "issue": f"QA answer disagreement for '{q_id}' ({qa_dist})",
                    "status": "pending",
                    "resolved_decision": None
                })

            qa_consensus_list.append({
                "question_id": q_id,
                "finding": f_name,
                "level": lvl,
                "question_text": q_text,
                "machine_answer": m_ans,
                "reviewer_answers": r_answers,
                "consensus_answer": cons_ans,
                "agreement_status": agr_status
            })

        # 4. Statistical Agreement Metrics (Cohen's & Fleiss' Kappa)
        agreement_metrics = self._compute_all_agreement_metrics(
            eval_sessions,
            evidence_list
        )

        session["finding_consensus"] = finding_consensus_list
        session["qa_consensus"] = qa_consensus_list
        session["agreement_metrics"] = agreement_metrics
        session["adjudication"]["items_requiring_adjudication"] = adjudication_items
        session["adjudication"]["all_resolved"] = all(item["status"] == "resolved" for item in adjudication_items)

        # 5. Update Status
        if session.get("status") != "finalized":
            if n_completed < min_req:
                session["status"] = "collecting"
            elif not session["adjudication"]["all_resolved"]:
                session["status"] = "adjudication_required"
            elif len(adjudication_records) > 0:
                session["status"] = "adjudicated"
            else:
                session["status"] = "consensus_reached"

        # 6. Auto-generate / sync draft consensus report if not manually authored
        self._sync_consensus_report_draft(session, machine_data)

    def _compute_all_agreement_metrics(
        self,
        completed_sessions: List[Dict[str, Any]],
        evidence_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute average agreement ratio, Cohen's Kappa, and Fleiss' Kappa."""
        n_raters = len(completed_sessions)
        finding_names = [e.get("finding", "") for e in evidence_list]
        n_findings = len(finding_names)

        if n_raters == 0 or n_findings == 0:
            return {
                "reviewer_count": n_raters,
                "finding_count": n_findings,
                "average_agreement_ratio": None,
                "cohens_kappa": None,
                "fleiss_kappa": None
            }

        # Build matrix: findings x raters -> decision
        rating_matrix: List[List[str]] = []
        for fname in finding_names:
            row = []
            for s in completed_sessions:
                fr = next((f for f in s.get("finding_reviews", []) if f.get("finding", "").lower() == fname.lower()), None)
                dec = fr.get("reviewer_status", "not_reviewed") if fr else "not_reviewed"
                row.append(dec)
            rating_matrix.append(row)

        # Average Agreement Ratio
        ratios = []
        for row in rating_matrix:
            counts = {}
            for d in row:
                counts[d] = counts.get(d, 0) + 1
            max_c = max(counts.values()) if counts else 0
            ratios.append(max_c / n_raters if n_raters > 0 else 0)
        avg_ratio = round(sum(ratios) / len(ratios), 4) if ratios else None

        # Cohen's Kappa (Exactly 2 raters)
        cohens_kappa = None
        if n_raters == 2:
            cohens_kappa = self.calculate_cohens_kappa(
                rating_matrix,
                r1_id=completed_sessions[0].get("reviewer", {}).get("id", "r1"),
                r2_id=completed_sessions[1].get("reviewer", {}).get("id", "r2")
            )

        # Fleiss' Kappa (3 or more raters)
        fleiss_kappa = None
        if n_raters >= 3:
            fleiss_kappa = self.calculate_fleiss_kappa(rating_matrix, n_raters)

        return {
            "reviewer_count": n_raters,
            "finding_count": n_findings,
            "average_agreement_ratio": avg_ratio,
            "cohens_kappa": cohens_kappa,
            "fleiss_kappa": fleiss_kappa
        }

    def calculate_cohens_kappa(
        self,
        rating_matrix: List[List[str]],
        r1_id: str = "r1",
        r2_id: str = "r2"
    ) -> Dict[str, Any]:
        """
        Calculates Cohen's Kappa for 2 raters on finding decisions.
        Handles zero-variance, single category, and edge cases safely.
        """
        n_items = len(rating_matrix)
        if n_items == 0:
            return {
                "metric": "cohens_kappa",
                "value": None,
                "observed_agreement": None,
                "expected_agreement": None,
                "reviewer_pair": [r1_id, r2_id],
                "finding_count": 0
            }

        # Categories observed
        categories = sorted(list(set(r[0] for r in rating_matrix) | set(r[1] for r in rating_matrix)))
        if len(categories) <= 1:
            # Single category used exclusively -> 100% agreement but 0 chance variance
            return {
                "metric": "cohens_kappa",
                "value": 1.0 if all(r[0] == r[1] for r in rating_matrix) else None,
                "observed_agreement": 1.0 if all(r[0] == r[1] for r in rating_matrix) else 0.0,
                "expected_agreement": 1.0,
                "reviewer_pair": [r1_id, r2_id],
                "finding_count": n_items
            }

        # Observed agreement P_o
        matches = sum(1 for r in rating_matrix if r[0] == r[1])
        p_o = matches / n_items

        # Expected agreement P_e
        p_e = 0.0
        for c in categories:
            cnt_r1 = sum(1 for r in rating_matrix if r[0] == c)
            cnt_r2 = sum(1 for r in rating_matrix if r[1] == c)
            p_e += (cnt_r1 / n_items) * (cnt_r2 / n_items)

        # Kappa computation
        if math.isclose(p_e, 1.0):
            kappa_val = 1.0 if math.isclose(p_o, 1.0) else None
        else:
            denom = 1.0 - p_e
            if denom == 0:
                kappa_val = None
            else:
                kappa_val = (p_o - p_e) / denom

        if kappa_val is not None:
            if math.isnan(kappa_val) or math.isinf(kappa_val):
                kappa_val = None
            else:
                kappa_val = round(max(-1.0, min(1.0, kappa_val)), 4)

        return {
            "metric": "cohens_kappa",
            "value": kappa_val,
            "observed_agreement": round(p_o, 4),
            "expected_agreement": round(p_e, 4),
            "reviewer_pair": [r1_id, r2_id],
            "finding_count": n_items
        }

    def calculate_fleiss_kappa(
        self,
        rating_matrix: List[List[str]],
        n_raters: int
    ) -> Dict[str, Any]:
        """
        Calculates Fleiss' Kappa for k >= 3 raters.
        N = items (findings), k = raters, C = categories.
        Guarantees zero NaN / Infinity outputs.
        """
        n_items = len(rating_matrix)
        if n_items == 0 or n_raters < 3:
            return {
                "metric": "fleiss_kappa",
                "value": None,
                "observed_agreement_mean": None,
                "expected_agreement": None,
                "reviewer_count": n_raters,
                "finding_count": n_items,
                "categories": []
            }

        # Categories list
        cat_set = set()
        for row in rating_matrix:
            for val in row:
                cat_set.add(val)
        categories = sorted(list(cat_set))
        n_cat = len(categories)

        if n_cat <= 1:
            return {
                "metric": "fleiss_kappa",
                "value": 1.0,
                "observed_agreement_mean": 1.0,
                "expected_agreement": 1.0,
                "reviewer_count": n_raters,
                "finding_count": n_items,
                "categories": categories
            }

        # Count matrix: N x C
        # n_ij is number of raters assigning item i to category j
        matrix = []
        for row in rating_matrix:
            counts = [row.count(c) for c in categories]
            matrix.append(counts)

        # 1. Category proportions p_j = sum_i(n_ij) / (N * k)
        total_ratings = n_items * n_raters
        p_j = [sum(matrix[i][j] for i in range(n_items)) / total_ratings for j in range(n_cat)]

        # 2. Expected agreement P_e_bar = sum_j(p_j^2)
        p_e_bar = sum(p ** 2 for p in p_j)

        # 3. Observed agreement for item i: P_i = 1/(k(k-1)) * (sum_j(n_ij^2) - k)
        k_factor = n_raters * (n_raters - 1)
        p_i_list = []
        for i in range(n_items):
            sum_sq = sum(matrix[i][j] ** 2 for j in range(n_cat))
            p_i = (sum_sq - n_raters) / k_factor
            p_i_list.append(p_i)

        # 4. Mean observed agreement P_bar
        p_bar = sum(p_i_list) / n_items

        # 5. Fleiss Kappa = (P_bar - P_e_bar) / (1 - P_e_bar)
        if math.isclose(p_e_bar, 1.0):
            kappa_val = 1.0 if math.isclose(p_bar, 1.0) else None
        else:
            denom = 1.0 - p_e_bar
            if denom == 0:
                kappa_val = None
            else:
                kappa_val = (p_bar - p_e_bar) / denom

        if kappa_val is not None:
            if math.isnan(kappa_val) or math.isinf(kappa_val):
                kappa_val = None
            else:
                kappa_val = round(max(-1.0, min(1.0, kappa_val)), 4)

        return {
            "metric": "fleiss_kappa",
            "value": kappa_val,
            "observed_agreement_mean": round(p_bar, 4),
            "expected_agreement": round(p_e_bar, 4),
            "reviewer_count": n_raters,
            "finding_count": n_items,
            "categories": categories
        }

    def record_adjudication(
        self,
        study_id: str,
        target_type: str,
        target_id: str,
        decision: str,
        reason: str,
        adjudicator_id: str = "senior_adjudicator",
        adjudicator_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Records a binding adjudication decision for an unresolvable disagreement.
        Requires a non-empty clinical justification reason.
        """
        clean_reason = reason.strip()
        if not clean_reason:
            raise ValueError("Mandatory adjudication reason is missing or empty.")

        valid_types = ["finding", "location", "severity", "qa", "report"]
        if target_type not in valid_types:
            raise ValueError(f"Invalid target_type '{target_type}'. Must be one of {valid_types}")

        session = self.get_consensus_session(study_id)
        if not session:
            session = self.create_consensus_session(study_id)

        if session.get("status") == "finalized":
            raise ValueError(f"Cannot record adjudication: Consensus session for '{study_id}' is finalized.")

        now = _get_timestamp()
        adj_id = f"adj_{study_id}_{target_type}_{target_id}_{int(time.time())}"

        # Find prior consensus if exists
        prev_consensus = None
        for fc in session.get("finding_consensus", []):
            if fc["finding"].lower() == target_id.lower():
                prev_consensus = fc.get("consensus_decision")
                break

        record = {
            "adjudication_id": adj_id,
            "study_id": study_id,
            "target_type": target_type,
            "target_id": target_id,
            "previous_consensus": prev_consensus,
            "adjudicator_id": adjudicator_id,
            "adjudicator_name": adjudicator_name or f"Adjudicator ({adjudicator_id})",
            "decision": decision,
            "reason": clean_reason,
            "timestamp": now
        }

        # Filter out existing record for this target if re-adjudicating
        adj_records = [r for r in session["adjudication"].get("adjudication_records", []) if not (r["target_type"] == target_type and r["target_id"].lower() == target_id.lower())]
        adj_records.append(record)
        session["adjudication"]["adjudication_records"] = adj_records

        session["audit_trail"].append({
            "timestamp": now,
            "user_id": adjudicator_id,
            "action": "adjudication_recorded",
            "target": f"{target_type}:{target_id}",
            "old_value": prev_consensus,
            "new_value": decision
        })

        session["updated_at"] = now
        self._sync_and_recalculate(study_id, session)
        self._save_consensus_session(study_id, session)
        return record

    def _sync_consensus_report_draft(self, session: Dict[str, Any], machine_data: Dict[str, Any]):
        """Generates draft consensus report from agreed findings and impression."""
        crep = session.setdefault("consensus_report", {})
        if crep.get("finalized", False):
            return

        final_findings = []
        for fc in session.get("finding_consensus", []):
            fname = fc["finding"]
            status = fc.get("consensus_decision") or fc.get("machine_status", "possible")
            loc = fc.get("location_consensus") or "unspecified"
            sev = fc.get("severity_consensus") or "unspecified"
            basis = fc.get("consensus_status", "pending")

            # Map statement
            stmt = f"Evaluation of {fname.lower()} is {status}."
            if status == "confirmed_present":
                stmt = f"Definite radiographic evidence of {fname.lower()} is present ({loc}, {sev})."
            elif status == "confirmed_absent":
                stmt = f"No radiographic evidence of {fname.lower()}."
            elif status == "uncertain":
                stmt = f"Equivocal or borderline evidence of {fname.lower()}."

            final_findings.append({
                "finding": fname,
                "statement": stmt,
                "status": status,
                "location": loc,
                "severity": sev,
                "consensus_basis": basis
            })

        # Consensus Impression
        active_findings = [f["finding"] for f in final_findings if f["status"] == "confirmed_present"]
        if active_findings:
            impression = [f"1. Consensus findings confirm: {', '.join(active_findings)}.", "2. Follow-up recommended as clinically indicated."]
        else:
            impression = ["1. No acute consensus cardiopulmonary abnormality."]

        n_rev = len([r for r in session.get("reviewers", []) if r.get("review_status") == "completed"])
        req_rev = session.get("required_reviewers", 3)
        avg_agr = session.get("agreement_metrics", {}).get("average_agreement_ratio")
        agr_pct = f"{int(avg_agr * 100)}%" if avg_agr is not None else "N/A"

        adj_count = len(session.get("adjudication", {}).get("adjudication_records", []))

        crep["final_findings"] = final_findings
        crep["final_impression"] = impression
        crep["reviewer_summary"] = f"Synthesized from {n_rev}/{req_rev} independent completed reviewer evaluations."
        crep["agreement_summary"] = f"Average inter-rater agreement ratio: {agr_pct}."
        crep["adjudication_summary"] = f"Adjudicated items: {adj_count} dispute(s) resolved."
        crep["finalized"] = False

    def save_consensus_report_draft(
        self,
        study_id: str,
        final_findings: Optional[List[Dict[str, Any]]] = None,
        final_impression: Optional[List[str]] = None,
        reviewer_comment: Optional[str] = None,
        user_id: str = "consensus_coordinator"
    ) -> Dict[str, Any]:
        """Saves custom edited draft of the consensus report."""
        session = self.get_consensus_session(study_id)
        if not session:
            session = self.create_consensus_session(study_id)

        if session.get("status") == "finalized":
            raise ValueError(f"Cannot edit consensus report: Session for '{study_id}' is finalized.")

        crep = session.setdefault("consensus_report", {})
        now = _get_timestamp()

        if final_findings is not None:
            if not isinstance(final_findings, list):
                raise ValueError("final_findings must be a list of finding objects.")
            crep["final_findings"] = final_findings

        if final_impression is not None:
            if not isinstance(final_impression, list):
                raise ValueError("final_impression must be a list of string statements.")
            crep["final_impression"] = final_impression

        if reviewer_comment is not None:
            crep["reviewer_comment"] = str(reviewer_comment).strip()

        session["updated_at"] = now
        session["audit_trail"].append({
            "timestamp": now,
            "user_id": user_id,
            "action": "consensus_report_draft_saved",
            "target": "*REPORT*",
            "old_value": "draft",
            "new_value": f"{len(crep.get('final_findings', []))} findings"
        })

        self._save_consensus_session(study_id, session)
        return crep

    def finalize_consensus(
        self,
        study_id: str,
        user_id: str = "consensus_chair"
    ) -> Tuple[bool, Dict[str, Any], List[str]]:
        """
        Validates consensus readiness and permanently locks the consensus session.
        """
        session = self.get_consensus_session(study_id)
        if not session:
            raise ValueError(f"Consensus session for '{study_id}' does not exist.")

        if session.get("status") == "finalized":
            return True, session, []

        from validate_consensus import validate_consensus_session
        machine_data = self._get_machine_data(study_id)
        is_valid, issues = validate_consensus_session(
            session,
            machine_study_data=machine_data,
            require_finalization_ready=True
        )

        if not is_valid:
            return False, session, issues

        now = _get_timestamp()
        session["status"] = "finalized"
        session["completed_at"] = now
        session["updated_at"] = now
        session["consensus_report"]["finalized"] = True

        session["audit_trail"].append({
            "timestamp": now,
            "user_id": user_id,
            "action": "consensus_finalized",
            "target": "*SESSION*",
            "old_value": "adjudicated",
            "new_value": "finalized"
        })

        self._save_consensus_session(study_id, session)
        return True, session, []

    def get_consensus_audit(self, study_id: str) -> List[Dict[str, Any]]:
        """Retrieve chronological consensus audit trail."""
        session = self.get_consensus_session(study_id)
        if not session:
            return []
        return session.get("audit_trail", [])

    def export_consensus(self, study_id: str, format: str = "json") -> Tuple[str, str]:
        """Export consensus review package as JSON or Text."""
        session = self.get_consensus_session(study_id)
        if not session:
            session = self.create_consensus_session(study_id)

        machine_data = self._get_machine_data(study_id) or {}

        if format.lower() == "text":
            lines = [
                "================================================",
                "MULTI-REVIEWER CONSENSUS REPORT",
                "RESEARCH OUTPUT — NOT A CLINICAL DIAGNOSIS",
                "================================================",
                f"STUDY ID        : {session.get('study_id')}",
                f"CONSENSUS ID    : {session.get('consensus_id')}",
                f"STATUS          : {session.get('status', 'collecting').upper()}",
                f"CREATED AT      : {session.get('created_at')}",
                f"COMPLETED AT    : {session.get('completed_at') or 'IN PROGRESS'}",
                f"REQUIRED RATERS : {session.get('required_reviewers')}",
                "------------------------------------------------",
                "REVIEWERS ROSTER:"
            ]
            for r in session.get("reviewers", []):
                lines.append(f"- {r.get('id')}: {r.get('display_name')} [{r.get('review_status').upper()}]")

            agr = session.get("agreement_metrics", {})
            lines.append("------------------------------------------------")
            lines.append("INTER-RATER AGREEMENT METRICS:")
            lines.append(f"- Reviewers Compared    : {agr.get('reviewer_count')}")
            lines.append(f"- Findings Evaluated    : {agr.get('finding_count')}")
            lines.append(f"- Average Agreement     : {agr.get('average_agreement_ratio') or 'N/A'}")
            if agr.get("cohens_kappa"):
                lines.append(f"- Cohen's Kappa (N=2)   : {agr['cohens_kappa'].get('value')} (Po={agr['cohens_kappa'].get('observed_agreement')}, Pe={agr['cohens_kappa'].get('expected_agreement')})")
            if agr.get("fleiss_kappa"):
                lines.append(f"- Fleiss' Kappa (N>=3)  : {agr['fleiss_kappa'].get('value')} (Pbar={agr['fleiss_kappa'].get('observed_agreement_mean')}, Pe={agr['fleiss_kappa'].get('expected_agreement')})")

            lines.append("------------------------------------------------")
            lines.append("FINDING CONSENSUS & DECISIONS:")
            for fc in session.get("finding_consensus", []):
                lines.append(
                    f"- {fc['finding']:<14}: Machine=[{fc['machine_status'].upper():<9}], "
                    f"Consensus=[{str(fc.get('consensus_decision')).upper():<16}], Status=[{fc.get('consensus_status').upper()}]"
                )

            lines.append("------------------------------------------------")
            lines.append("ADJUDICATION RECORDS:")
            records = session.get("adjudication", {}).get("adjudication_records", [])
            if not records:
                lines.append("None (Unanimous or majority consensus achieved without adjudication).")
            else:
                for rec in records:
                    lines.append(f"- [{rec.get('target_type').upper()}:{rec.get('target_id')}] Decision='{rec.get('decision')}' by {rec.get('adjudicator_id')}")
                    lines.append(f"  Reason: {rec.get('reason')}")

            lines.append("------------------------------------------------")
            lines.append("FINAL CONSENSUS FINDINGS:")
            for ff in session.get("consensus_report", {}).get("final_findings", []):
                lines.append(f"- [{ff.get('status', '').upper():<16}] {ff.get('statement')}")

            lines.append("------------------------------------------------")
            lines.append("FINAL CONSENSUS IMPRESSION:")
            for imp in session.get("consensus_report", {}).get("final_impression", []):
                lines.append(f"- {imp}")

            lines.append("------------------------------------------------")
            lines.append("AUDIT TRAIL:")
            for e in session.get("audit_trail", []):
                lines.append(f"[{e.get('timestamp')}] {e.get('user_id')} -> {e.get('action')}: {e.get('target')} ('{e.get('old_value')}' => '{e.get('new_value')}')")

            lines.append("================================================")
            lines.append(RESEARCH_DISCLAIMER)
            lines.append("================================================")

            return "\n".join(lines), "text/plain"

        else:
            # JSON format
            export_obj = {
                "study_id": session.get("study_id"),
                "consensus_id": session.get("consensus_id"),
                "status": session.get("status"),
                "timestamps": {
                    "created_at": session.get("created_at"),
                    "updated_at": session.get("updated_at"),
                    "completed_at": session.get("completed_at")
                },
                "reviewers": session.get("reviewers", []),
                "machine_baseline": {
                    "evidence": machine_data.get("evidence", []),
                    "report": machine_data.get("report", {}),
                    "locked": True
                },
                "finding_consensus": session.get("finding_consensus", []),
                "qa_consensus": session.get("qa_consensus", []),
                "agreement_metrics": session.get("agreement_metrics", {}),
                "adjudication": session.get("adjudication", {}),
                "consensus_report": session.get("consensus_report", {}),
                "audit_trail": session.get("audit_trail", []),
                "disclaimer": RESEARCH_DISCLAIMER
            }
            return json.dumps(export_obj, indent=2), "application/json"


# Global singleton instance
global_consensus_manager = ConsensusManager()
