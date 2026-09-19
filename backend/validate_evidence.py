"""
LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 0.7 — Evidence Layer Validation Module

Module: validate_evidence.py
Purpose:
- Validates Evidence Layer output packages and LLM Input Packages.
- Enforces strict checks:
  1. Required fields exist in package and findings.
  2. Statuses are strictly in {'supported', 'possible', 'uncertain', 'absent'}.
  3. model_score is numeric, finite, and in [0.0, 1.0].
  4. Location and severity are valid strings and not hallucinated.
  5. Provenance sources are tracked and valid.
  6. No duplicate findings exist.
  7. Ground-truth data is STRICTLY ABSENT from the production LLM input package.
  8. Anti-hallucination constraints are present.

DISCLAIMER:
This validation enforces engineering safety and data integrity rules.
"""

import sys
import json
from typing import Dict, List, Any, Tuple

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


VALID_STATUSES = {"supported", "possible", "uncertain", "absent"}
VALID_SOURCES = {"vision_model", "diagnostic_qa", "rule_derived", "grounding_module", "report_ground_truth"}
VALID_LOCATIONS = {
    "right", "left", "bilateral", "right_upper_lobe", "right_middle_lobe", "right_lower_lobe",
    "left_upper_lobe", "left_lower_lobe", "basilar", "apical", "retrocardiac", "mediastinum",
    "diffuse", "unspecified"
}
VALID_SEVERITIES = {
    "small", "minimal", "moderate", "large", "mild", "severe", "trace", "unspecified"
}


def validate_evidence_package(package: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate an Evidence Layer output package for schema compliance and safety invariants.

    Returns:
        (is_valid: bool, issues: List[str])
    """
    issues = []

    # 1. Check Root Fields
    for root_field in ["study_id", "image_id", "view", "findings", "llm_input_package"]:
        if root_field not in package:
            issues.append(f"Missing root field: '{root_field}'")

    findings = package.get("findings", [])
    if not isinstance(findings, list):
        issues.append("'findings' must be a list.")
        return False, issues

    seen_findings = set()

    for idx, f in enumerate(findings):
        # 2. Required Finding Fields
        for req in ["finding", "status", "model_score", "location", "severity", "evidence_sources"]:
            if req not in f:
                issues.append(f"Finding [{idx}] missing required field: '{req}'")

        finding_name = f.get("finding")
        if finding_name:
            if finding_name in seen_findings:
                issues.append(f"Duplicate finding detected: '{finding_name}'")
            seen_findings.add(finding_name)

        # 3. Status Validation
        status = f.get("status")
        if status not in VALID_STATUSES:
            issues.append(f"Finding '{finding_name}' has invalid status '{status}'. Valid: {VALID_STATUSES}")

        # 4. Model Score Validation
        score = f.get("model_score")
        if score is None or not isinstance(score, (int, float)):
            issues.append(f"Finding '{finding_name}' model_score must be numeric, got: {type(score)}")
        elif not (0.0 <= score <= 1.0):
            issues.append(f"Finding '{finding_name}' model_score {score} out of bounds [0.0, 1.0].")

        # 5. Check 'confidence' is NOT present
        if "confidence" in f:
            issues.append(f"Finding '{finding_name}' illegally contains 'confidence' field. Use 'model_score'.")

        # 6. Location & Severity Validation
        loc = f.get("location", "").lower()
        if loc and loc not in VALID_LOCATIONS:
            issues.append(f"Finding '{finding_name}' location '{loc}' is invalid.")

        sev = f.get("severity", "").lower()
        if sev and sev not in VALID_SEVERITIES:
            issues.append(f"Finding '{finding_name}' severity '{sev}' is invalid.")

        # 7. Evidence Sources Validation
        sources = f.get("evidence_sources", [])
        if not sources or not isinstance(sources, list):
            issues.append(f"Finding '{finding_name}' evidence_sources must be a non-empty list.")
        else:
            for s in sources:
                if s not in VALID_SOURCES:
                    issues.append(f"Finding '{finding_name}' contains invalid evidence source '{s}'.")

    # 8. LLM Input Package Validation (Leakage Check)
    llm_pkg = package.get("llm_input_package", {})
    if not isinstance(llm_pkg, dict):
        issues.append("'llm_input_package' must be a dictionary.")
    else:
        for pkg_field in ["task", "study", "evidence", "constraints"]:
            if pkg_field not in llm_pkg:
                issues.append(f"llm_input_package missing field: '{pkg_field}'")

        # Ground-Truth Leakage Check
        pkg_str = json.dumps(llm_pkg)
        if "report_ground_truth" in pkg_str:
            issues.append("CRITICAL: 'report_ground_truth' leaked into llm_input_package!")
        if "ground_truth" in pkg_str.lower() and "evaluation" not in pkg_str.lower():
            issues.append("CRITICAL: Potential ground truth leakage detected in llm_input_package.")

        # Constraint Checks
        constraints = llm_pkg.get("constraints", {})
        for c in ["do_not_invent_findings", "do_not_invent_location", "do_not_invent_severity", "preserve_uncertainty"]:
            if not constraints.get(c):
                issues.append(f"llm_input_package missing mandatory safety constraint '{c}'=True")

    is_valid = len(issues) == 0
    return is_valid, issues


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_path = sys.argv[1]
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        valid, errs = validate_evidence_package(data)
        if valid:
            print(f"[SUCCESS] Evidence package {target_path} is valid.")
        else:
            print(f"[ERROR] Validation failed for {target_path}:")
            for err in errs:
                print(f"  - {err}")
            sys.exit(1)
    else:
        print("Usage: python validate_evidence.py <path_to_evidence_package.json>")
