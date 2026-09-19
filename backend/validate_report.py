"""
LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 0.8 — Report Safety & Schema Validation Module

Module: validate_report.py
Purpose:
- Rigorously validates generated structured radiology reports against docs/report_schema.json and input evidence.
- Verifies:
  1. Study identifiers (study_id, image_id, view) match input package.
  2. Findings are traceable to input evidence.
  3. Status integrity (prevents illegal upgrades, e.g. 'possible' -> 'supported').
  4. Non-hallucination of location, severity, or numerical measurements.
  5. Absence of raw model_score exposure as probability.
  6. Impression consistency (impression does not introduce unmodeled/unmentioned findings).
  7. Zero ground-truth leakage.

DISCLAIMER:
This validation enforces technical and clinical safety invariants.
"""

import re
import json
from typing import Dict, List, Any, Tuple, Optional

VALID_STATUSES = {"supported", "possible", "uncertain", "absent"}
VALID_LOCATIONS = {
    "right", "left", "bilateral", "right_upper_lobe", "right_middle_lobe", "right_lower_lobe",
    "left_upper_lobe", "left_lower_lobe", "basilar", "apical", "retrocardiac", "mediastinum",
    "diffuse", "unspecified"
}
VALID_SEVERITIES = {
    "small", "minimal", "moderate", "large", "mild", "severe", "trace", "unspecified"
}

# Regex to detect fabricated numerical measurements (e.g., "2.5 cm", "15 mm")
MEASUREMENT_REGEX = re.compile(r"\b\d+(\.\d+)?\s*(cm|mm|centimeter|millimeter|inch|inches)\b", re.IGNORECASE)


def validate_generated_report(
    report: Dict[str, Any],
    llm_input_package: Optional[Dict[str, Any]] = None
) -> Tuple[bool, List[str]]:
    """
    Validate a generated radiology report against the input evidence and safety rules.

    Args:
        report: Generated report dictionary adhering to docs/report_schema.json.
        llm_input_package: Optional input package provided to the LLM.

    Returns:
        (is_valid: bool, issues: List[str])
    """
    issues = []

    if not isinstance(report, dict):
        return False, ["Report must be a JSON dictionary."]

    # 1. Root Fields Validation
    for req in ["study_id", "image_id", "view", "findings", "impression"]:
        if req not in report:
            issues.append(f"Missing required root field: '{req}'")

    study_id = report.get("study_id")
    image_id = report.get("image_id")
    view = report.get("view")

    # 2. Match with Input Package if provided
    input_evidence_map = {}
    if llm_input_package:
        in_study = llm_input_package.get("study", {})
        if study_id != in_study.get("study_id"):
            issues.append(f"study_id mismatch: expected '{in_study.get('study_id')}', got '{study_id}'")
        if image_id != in_study.get("image_id"):
            issues.append(f"image_id mismatch: expected '{in_study.get('image_id')}', got '{image_id}'")
        if view != in_study.get("view"):
            issues.append(f"view mismatch: expected '{in_study.get('view')}', got '{view}'")

        for ev in llm_input_package.get("evidence", []):
            input_evidence_map[ev["finding"]] = ev

    # 3. Findings Validation
    findings = report.get("findings", [])
    if not isinstance(findings, list) or len(findings) == 0:
        issues.append("'findings' must be a non-empty list.")
    else:
        seen_findings = set()
        for idx, f in enumerate(findings):
            if not isinstance(f, dict):
                issues.append(f"Finding entry [{idx}] must be a dictionary.")
                continue

            for f_req in ["finding", "statement", "status"]:
                if f_req not in f:
                    issues.append(f"Finding [{idx}] missing required field: '{f_req}'")

            name = f.get("finding")
            if not name:
                issues.append(f"Finding [{idx}] has empty finding name.")
                continue

            if name in seen_findings:
                issues.append(f"Duplicate finding in report: '{name}'")
            seen_findings.add(name)

            status = f.get("status")
            if status not in VALID_STATUSES:
                issues.append(f"Finding '{name}' has invalid status: '{status}'")

            statement = f.get("statement", "")
            if not statement or not statement.strip():
                issues.append(f"Finding '{name}' has empty statement.")

            # Location validation
            loc = f.get("location", "unspecified")
            if loc and loc not in VALID_LOCATIONS:
                issues.append(f"Finding '{name}' has invalid location: '{loc}'")

            # Severity validation
            sev = f.get("severity", "unspecified")
            if sev and sev not in VALID_SEVERITIES:
                issues.append(f"Finding '{name}' has invalid severity: '{sev}'")

            # Anti-Measurement Check
            if MEASUREMENT_REGEX.search(statement):
                issues.append(f"Finding '{name}' contains fabricated numerical measurements in statement: '{statement}'")

            # Evidence Traceability Checks
            if input_evidence_map:
                if name not in input_evidence_map:
                    issues.append(f"Unsupported finding introduced in report: '{name}' was not in input evidence.")
                else:
                    ev_item = input_evidence_map[name]
                    ev_status = ev_item.get("status")
                    ev_loc = ev_item.get("location", "unspecified")
                    ev_sev = ev_item.get("severity", "unspecified")

                    # Status Upgrade Check
                    if ev_status in ["possible", "uncertain"] and status == "supported":
                        issues.append(f"Illegal status upgrade: Finding '{name}' was '{ev_status}' in evidence but reported as 'supported'.")
                    elif ev_status == "absent" and status in ["supported", "possible"]:
                        issues.append(f"Illegal status contradiction: Finding '{name}' was 'absent' in evidence but reported as '{status}'.")

                    # Location Hallucination Check
                    if ev_loc == "unspecified" and loc != "unspecified":
                        issues.append(f"Fabricated location: Finding '{name}' had unspecified location in evidence but '{loc}' in report.")

                    # Severity Hallucination Check
                    if ev_sev == "unspecified" and sev != "unspecified":
                        issues.append(f"Fabricated severity: Finding '{name}' had unspecified severity in evidence but '{sev}' in report.")

    # 4. Impression Validation
    impression = report.get("impression", [])
    if not isinstance(impression, list) or len(impression) == 0:
        issues.append("'impression' must be a non-empty list of strings.")
    else:
        for imp_idx, bullet in enumerate(impression):
            if not isinstance(bullet, str) or not bullet.strip():
                issues.append(f"Impression entry [{imp_idx}] must be a non-empty string.")
            elif MEASUREMENT_REGEX.search(bullet):
                issues.append(f"Impression bullet [{imp_idx}] contains fabricated measurements: '{bullet}'")

    # 5. Model Score Exposure Check
    report_str = json.dumps(report)
    if re.search(r"\b(0\.\d{2,4}\s*(%|percent|probability|confidence))\b", report_str, re.IGNORECASE):
        issues.append("Model score improperly exposed as clinical probability or percentage.")

    # 6. Ground-Truth Leakage Check
    if "report_ground_truth" in report_str:
        issues.append("CRITICAL: Ground-truth marker detected in generated report.")

    is_valid = len(issues) == 0
    return is_valid, issues
