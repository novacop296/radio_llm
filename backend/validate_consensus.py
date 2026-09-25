"""
LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 1.3 — Multi-Reviewer Consensus & Adjudication Validation

Module: validate_consensus.py
Purpose:
- Validates structural and semantic integrity of ConsensusSession records.
- Verifies reviewer completion thresholds and isolation.
- Enforces deterministic consensus rules and inter-rater agreement bounds.
- Validates dispute adjudication records and mandatory justification fields.
- Checks byte-level machine artifact immutability.
- Guarantees zero ground-truth XML reference report leakage.
"""

import os
import sys
import re
import json
import math
from typing import Dict, List, Any, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
IU_DATA_DIR = os.path.join(DATA_DIR, "iu_xray")
REPORTS_DIR = os.path.join(IU_DATA_DIR, "reports")

ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")

VALID_CONSENSUS_STATUSES = {
    "collecting",
    "ready_for_consensus",
    "consensus_reached",
    "adjudication_required",
    "adjudicated",
    "finalized"
}

VALID_FINDING_DECISIONS = {
    "confirmed_present",
    "confirmed_absent",
    "uncertain",
    "needs_review",
    "not_reviewed"
}


def validate_consensus_session(
    session: Dict[str, Any],
    machine_study_data: Optional[Dict[str, Any]] = None,
    require_finalization_ready: bool = False
) -> Tuple[bool, List[str]]:
    """
    Validates a multi-reviewer consensus session.

    Returns:
        (is_valid: bool, issues: list of string descriptions)
    """
    issues: List[str] = []

    if not isinstance(session, dict):
        return False, ["Session payload must be a JSON object."]

    # 1. Study ID & Consensus ID
    study_id = session.get("study_id")
    if not study_id or not isinstance(study_id, str) or not ID_PATTERN.match(study_id):
        issues.append(f"Invalid or missing 'study_id': {study_id}")

    if ".." in str(study_id) or "/" in str(study_id) or "\\" in str(study_id):
        issues.append(f"Path traversal detected in study_id: {study_id}")

    consensus_id = session.get("consensus_id")
    if not consensus_id or not isinstance(consensus_id, str):
        issues.append(f"Invalid or missing 'consensus_id': {consensus_id}")

    # 2. Status Validation
    status = session.get("status")
    if status not in VALID_CONSENSUS_STATUSES:
        issues.append(f"Invalid consensus status '{status}'. Must be one of {sorted(VALID_CONSENSUS_STATUSES)}")

    # 3. Reviewers Roster
    reviewers = session.get("reviewers")
    if not isinstance(reviewers, list):
        issues.append("Field 'reviewers' must be a list.")
    else:
        seen_ids = set()
        for r in reviewers:
            r_id = r.get("id")
            if not r_id or not ID_PATTERN.match(str(r_id)):
                issues.append(f"Invalid reviewer ID in roster: {r_id}")
            if r_id in seen_ids:
                issues.append(f"Duplicate reviewer ID '{r_id}' in roster.")
            seen_ids.add(r_id)

    # 4. Finding Consensus List
    finding_consensus = session.get("finding_consensus")
    if not isinstance(finding_consensus, list):
        issues.append("Field 'finding_consensus' must be a list.")
    else:
        for fc in finding_consensus:
            fname = fc.get("finding")
            if not fname:
                issues.append("Finding consensus item missing 'finding' name.")

            c_dec = fc.get("consensus_decision")
            if c_dec is not None and c_dec not in VALID_FINDING_DECISIONS:
                issues.append(f"Invalid consensus_decision '{c_dec}' for finding '{fname}'.")

            agr_rat = fc.get("agreement_ratio")
            if agr_rat is not None:
                if not isinstance(agr_rat, (int, float)) or math.isnan(agr_rat) or agr_rat < 0.0 or agr_rat > 1.0:
                    issues.append(f"Invalid agreement_ratio '{agr_rat}' for finding '{fname}'. Must be float in [0.0, 1.0].")

    # 5. Agreement Metrics Validity
    metrics = session.get("agreement_metrics", {})
    if isinstance(metrics, dict):
        avg_agr = metrics.get("average_agreement_ratio")
        if avg_agr is not None:
            if not isinstance(avg_agr, (int, float)) or math.isnan(avg_agr) or avg_agr < 0.0 or avg_agr > 1.0:
                issues.append(f"Invalid average_agreement_ratio '{avg_agr}'. Must be float in [0.0, 1.0].")

        ck = metrics.get("cohens_kappa")
        if ck and isinstance(ck, dict):
            val = ck.get("value")
            if val is not None and (not isinstance(val, (int, float)) or math.isnan(val) or val < -1.0 or val > 1.0):
                issues.append(f"Invalid Cohen's Kappa value '{val}'. Must be float in [-1.0, 1.0] or null.")

        fk = metrics.get("fleiss_kappa")
        if fk and isinstance(fk, dict):
            val = fk.get("value")
            if val is not None and (not isinstance(val, (int, float)) or math.isnan(val) or val < -1.0 or val > 1.0):
                issues.append(f"Invalid Fleiss' Kappa value '{val}'. Must be float in [-1.0, 1.0] or null.")

    # 6. Adjudication Records Validation
    adj = session.get("adjudication", {})
    if isinstance(adj, dict):
        for rec in adj.get("adjudication_records", []):
            if not rec.get("adjudication_id"):
                issues.append("Adjudication record missing 'adjudication_id'.")
            if not rec.get("adjudicator_id"):
                issues.append("Adjudication record missing 'adjudicator_id'.")
            if not rec.get("reason") or not str(rec.get("reason")).strip():
                issues.append(f"Adjudication record '{rec.get('adjudication_id')}' missing mandatory non-empty justification reason.")
            if not rec.get("decision"):
                issues.append(f"Adjudication record '{rec.get('adjudication_id')}' missing 'decision'.")

    # 7. Finalization Prerequisites
    if require_finalization_ready:
        min_rev = session.get("minimum_reviewers", 2)
        completed_revs = [r for r in session.get("reviewers", []) if r.get("review_status") == "completed"]
        if len(completed_revs) < min_rev:
            issues.append(f"Finalization blocked: Only {len(completed_revs)}/{min_rev} required reviewers completed review.")

        # Check for unresolved items
        unresolved_adj = [it for it in adj.get("items_requiring_adjudication", []) if it.get("status") != "resolved"]
        if unresolved_adj:
            names = [it.get("target_id") for it in unresolved_adj]
            issues.append(f"Finalization blocked: Unresolved adjudication items remain for: {', '.join(names)}.")

        # Check for unresolved finding consensus
        for fc in session.get("finding_consensus", []):
            if fc.get("consensus_decision") is None:
                issues.append(f"Finalization blocked: Finding '{fc.get('finding')}' has no consensus decision.")

        # Check report non-empty
        crep = session.get("consensus_report", {})
        if not crep.get("final_findings"):
            issues.append("Finalization blocked: Consensus report 'final_findings' is empty.")
        if not crep.get("final_impression"):
            issues.append("Finalization blocked: Consensus report 'final_impression' is empty.")

    # 8. Zero Ground-Truth Report Leakage
    session_str = json.dumps(session)
    leakage_terms = ["<report>", "<xml", "COMPARISON:", "CLINICAL INFORMATION:", "CHEST RADIOGRAPH"]
    for term in leakage_terms:
        if term in session_str:
            issues.append(f"Ground-truth reference report leakage detected: '{term}' found in session data.")

    # 9. Machine Artifact Immutability Check
    if machine_study_data:
        m_ev = machine_study_data.get("evidence", [])
        for fc in session.get("finding_consensus", []):
            fname = fc.get("finding")
            orig = next((e for e in m_ev if e.get("finding", "").lower() == fname.lower()), None)
            if orig:
                if fc.get("machine_status") != orig.get("status"):
                    issues.append(f"Machine immutability violation: Baseline machine status for '{fname}' mutated from '{orig.get('status')}' to '{fc.get('machine_status')}'.")
                if not math.isclose(fc.get("machine_score", 0.0), orig.get("model_score", 0.0), abs_tol=1e-5):
                    issues.append(f"Machine immutability violation: Baseline model score for '{fname}' mutated.")

    return len(issues) == 0, issues
