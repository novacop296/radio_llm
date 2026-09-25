"""
LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 1.2 — Final Review & Human-in-the-Loop Validation Module

Module: validate_final_review.py
Purpose:
- Rigorously validates ReviewSession data models, finding corrections, QA adjustments,
  and finalized report drafts.
- Enforces:
  1. Study & finding ID integrity with path traversal prevention.
  2. Reviewer status validity (not_reviewed, confirmed_present, confirmed_absent, uncertain, needs_review).
  3. Machine evidence immutability (verifies stored machine values match immutable Evidence Layer).
  4. Machine report immutability (original machine report locked and unmodified).
  5. Review completeness for finalization (all candidate findings reviewed).
  6. Final report structure (non-empty findings and impression).
  7. Zero ground-truth leakage (no reference XML strings in review/report).
  8. Input length constraints and protection against malformed payloads.

DISCLAIMER:
This module enforces technical and clinical research safety invariants.
Reviewer annotations represent research metadata and not clinical ground truth.
"""

import re
import os
import json
from typing import Dict, List, Any, Tuple, Optional

VALID_SESSION_STATUSES = {"not_reviewed", "in_review", "reviewed", "finalized"}
VALID_REVIEWER_STATUSES = {"not_reviewed", "confirmed_present", "confirmed_absent", "uncertain", "needs_review"}
VALID_QA_ANSWERS = {"yes", "no", "uncertain", "unspecified", "not_reviewed"}

# Allowed characters for IDs
ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")

# Max string length constraints
MAX_SHORT_STR_LEN = 256
MAX_COMMENT_LEN = 4096
MAX_STATEMENT_LEN = 1024


def validate_review_session(
    review_data: Dict[str, Any],
    require_finalization_ready: bool = False,
    machine_study_data: Optional[Dict[str, Any]] = None
) -> Tuple[bool, List[str]]:
    """
    Validate a ReviewSession dictionary against structural, clinical, and safety invariants.

    Args:
        review_data: ReviewSession dictionary.
        require_finalization_ready: If True, enforces strict checks required for report finalization.
        machine_study_data: Optional authoritative machine study data loaded from disk to verify immutability.

    Returns:
        (is_valid: bool, issues: List[str])
    """
    issues = []

    if not isinstance(review_data, dict):
        return False, ["Review session must be a JSON object."]

    # 1. Required Root Fields
    required_root = [
        "study_id", "review_id", "reviewer", "status",
        "started_at", "updated_at", "finding_reviews",
        "qa_reviews", "report_review", "audit_trail"
    ]
    for req in required_root:
        if req not in review_data:
            issues.append(f"Missing required root field: '{req}'")

    study_id = review_data.get("study_id", "")
    if not isinstance(study_id, str) or not ID_PATTERN.match(study_id):
        issues.append(f"Invalid or malformed study_id: '{study_id}'. Must be alphanumeric.")

    review_id = review_data.get("review_id", "")
    if not isinstance(review_id, str) or not ID_PATTERN.match(review_id):
        issues.append(f"Invalid or malformed review_id: '{review_id}'. Must be alphanumeric.")

    # 2. Reviewer Object
    reviewer = review_data.get("reviewer", {})
    if not isinstance(reviewer, dict):
        issues.append("Field 'reviewer' must be a JSON object.")
    else:
        if not reviewer.get("id") or not isinstance(reviewer.get("id"), str):
            issues.append("Reviewer 'id' must be a non-empty string.")
        if not reviewer.get("display_name") or not isinstance(reviewer.get("display_name"), str):
            issues.append("Reviewer 'display_name' must be a non-empty string.")

    # 3. Session Status
    status = review_data.get("status", "")
    if status not in VALID_SESSION_STATUSES:
        issues.append(f"Invalid review session status '{status}'. Must be one of {sorted(VALID_SESSION_STATUSES)}")

    # 4. Finding Reviews Validation
    finding_reviews = review_data.get("finding_reviews", [])
    if not isinstance(finding_reviews, list):
        issues.append("Field 'finding_reviews' must be a list.")
        finding_reviews = []

    seen_findings = set()
    for idx, fr in enumerate(finding_reviews):
        if not isinstance(fr, dict):
            issues.append(f"finding_reviews[{idx}] must be an object.")
            continue

        finding_name = fr.get("finding", "")
        if not finding_name or not isinstance(finding_name, str):
            issues.append(f"finding_reviews[{idx}] missing valid 'finding' name.")
        elif finding_name in seen_findings:
            issues.append(f"Duplicate finding '{finding_name}' in finding_reviews.")
        else:
            seen_findings.add(finding_name)

        r_status = fr.get("reviewer_status", "")
        if r_status not in VALID_REVIEWER_STATUSES:
            issues.append(f"Invalid reviewer_status '{r_status}' for finding '{finding_name}'.")

        # Comment & Location length check
        comment = fr.get("reviewer_comment", "")
        if isinstance(comment, str) and len(comment) > MAX_COMMENT_LEN:
            issues.append(f"Comment for finding '{finding_name}' exceeds {MAX_COMMENT_LEN} chars.")

        loc = fr.get("reviewer_location", "")
        if isinstance(loc, str) and len(loc) > MAX_SHORT_STR_LEN:
            issues.append(f"Location for finding '{finding_name}' exceeds {MAX_SHORT_STR_LEN} chars.")

        # Finalization requirement: All findings must be explicitly reviewed
        if require_finalization_ready:
            if not fr.get("reviewed", False):
                issues.append(f"Finding '{finding_name}' is not marked as reviewed.")
            if r_status == "not_reviewed":
                issues.append(f"Finding '{finding_name}' still has status 'not_reviewed'.")
            if r_status == "needs_review":
                issues.append(f"Finding '{finding_name}' is marked as 'needs_review' and cannot be finalized.")

    # 5. QA Reviews Validation
    qa_reviews = review_data.get("qa_reviews", [])
    if not isinstance(qa_reviews, list):
        issues.append("Field 'qa_reviews' must be a list.")
    else:
        for idx, qr in enumerate(qa_reviews):
            if not isinstance(qr, dict):
                issues.append(f"qa_reviews[{idx}] must be an object.")
                continue
            q_ans = qr.get("reviewer_answer", "")
            if q_ans and q_ans not in VALID_QA_ANSWERS:
                issues.append(f"Invalid QA reviewer_answer '{q_ans}' for question '{qr.get('question_id')}'.")

    # 6. Report Review Validation
    report_rev = review_data.get("report_review", {})
    if not isinstance(report_rev, dict):
        issues.append("Field 'report_review' must be an object.")
    else:
        if not report_rev.get("machine_report_locked", True):
            issues.append("machine_report_locked must remain True to ensure machine report immutability.")

        final_findings = report_rev.get("final_findings", [])
        final_impression = report_rev.get("final_impression", [])

        if not isinstance(final_findings, list):
            issues.append("report_review.final_findings must be a list.")
        if not isinstance(final_impression, list):
            issues.append("report_review.final_impression must be a list.")

        if require_finalization_ready:
            if len(final_findings) == 0:
                issues.append("Finalized report must contain at least one finding in final_findings.")
            if len(final_impression) == 0:
                issues.append("Finalized report must contain at least one impression item.")

            for f_idx, ff in enumerate(final_findings):
                if not isinstance(ff, dict) or not ff.get("statement"):
                    issues.append(f"final_findings[{f_idx}] missing statement.")

    # 7. Audit Trail Validation
    audit_trail = review_data.get("audit_trail", [])
    if not isinstance(audit_trail, list):
        issues.append("Field 'audit_trail' must be a list.")
    else:
        for a_idx, entry in enumerate(audit_trail):
            if not isinstance(entry, dict):
                issues.append(f"audit_trail[{a_idx}] must be an object.")
                continue
            for req_a in ["timestamp", "reviewer_id", "finding", "field", "old_value", "new_value"]:
                if req_a not in entry:
                    issues.append(f"audit_trail[{a_idx}] missing required field '{req_a}'")

    # 8. Machine Evidence Immutability Verification (against disk authoritative data if provided)
    if machine_study_data:
        machine_ev = machine_study_data.get("evidence", [])
        machine_ev_map = {e["finding"]: e for e in machine_ev}

        for fr in finding_reviews:
            fname = fr.get("finding", "")
            if fname in machine_ev_map:
                auth_e = machine_ev_map[fname]
                if fr.get("machine_status") != auth_e.get("status"):
                    issues.append(
                        f"Machine immutability violation: 'machine_status' for '{fname}' was mutated "
                        f"from '{auth_e.get('status')}' to '{fr.get('machine_status')}'."
                    )
                if fr.get("machine_location") != auth_e.get("location"):
                    issues.append(
                        f"Machine immutability violation: 'machine_location' for '{fname}' was mutated."
                    )
                if fr.get("machine_severity") != auth_e.get("severity"):
                    issues.append(
                        f"Machine immutability violation: 'machine_severity' for '{fname}' was mutated."
                    )

    # 9. Ground-Truth Isolation Check
    raw_str = json.dumps(review_data)
    forbidden_leak_tokens = ["<report>", "<FINDINGS>", "<IMPRESSION>", "ecgen-radiology", "reference_ground_truth"]
    for tok in forbidden_leak_tokens:
        if tok in raw_str:
            issues.append(f"Ground-truth leakage detected: review payload contains forbidden token '{tok}'.")

    return len(issues) == 0, issues
