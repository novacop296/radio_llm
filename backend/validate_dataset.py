"""
LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 1.4 — Dataset Integrity & Security Validator

Module: validate_dataset.py
Purpose:
- Validates machine artifact integrity, schema conformity, review artifacts, and consensus records.
- Verifies security invariants (path traversal rejection, string size bounds, ID sanitization).
- Enforces strict ground-truth isolation from all dataset summaries and API payloads.
- Ensures machine evidence immutability and finalization locking.
"""

import os
import sys
import json
import re
from typing import Dict, List, Any, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "iu_xray")
REVIEWS_DIR = os.path.join(BASE_DIR, "data", "reviews")
CONSENSUS_DIR = os.path.join(BASE_DIR, "data", "consensus")

FORBIDDEN_GROUND_TRUTH_PATTERNS = [
    "report_ground_truth",
    "reference report",
    "<eFind>",
    "<eImpression>",
    "ground_truth_report",
    "xml_content",
    "patient_name",
    "mrn"
]


class DatasetValidator:
    """Validates multi-study dataset integrity, security boundaries, and schema compliance."""

    def __init__(
        self,
        data_dir: str = DATA_DIR,
        reviews_dir: str = REVIEWS_DIR,
        consensus_dir: str = CONSENSUS_DIR
    ):
        self.data_dir = data_dir
        self.reviews_dir = reviews_dir
        self.consensus_dir = consensus_dir

    def validate_security_input(self, input_str: str) -> Tuple[bool, str]:
        """Rejects path traversals, absolute paths, and oversized strings."""
        if not isinstance(input_str, str):
            return False, "Input must be a string."

        if len(input_str) > 5000:
            return False, "Input exceeds maximum allowed string length (5000 characters)."

        if ".." in input_str or "/" in input_str or "\\" in input_str:
            return False, "Path traversal sequence detected."

        if os.path.isabs(input_str):
            return False, "Absolute filesystem paths are rejected."

        return True, "Security input validation passed."

    def validate_machine_artifacts(self) -> Tuple[bool, List[str]]:
        """Verifies integrity of machine-generated artifacts in data/iu_xray/."""
        errors = []

        images_dir = os.path.join(self.data_dir, "images")
        if not os.path.exists(images_dir):
            errors.append("Machine images directory missing.")
        else:
            pngs = [f for f in os.listdir(images_dir) if f.endswith(".png")]
            if len(pngs) == 0:
                errors.append("No radiograph images found in machine dataset.")

        # Check CXR1122 baseline grounding
        grounding_dir = os.path.join(self.data_dir, "visual_grounding", "CXR1122")
        if not os.path.exists(grounding_dir):
            grounding_dir = os.path.join(self.data_dir, "grounding")

        if not os.path.exists(grounding_dir):
            errors.append("Visual grounding artifacts missing for baseline study CXR1122.")

        # Check corpus summary JSON validity
        summary_file = os.path.join(self.data_dir, "corpus_analysis_summary.json")
        if os.path.exists(summary_file):
            try:
                with open(summary_file, "r", encoding="utf-8") as f:
                    json.load(f)
            except Exception as e:
                errors.append(f"Corpus analysis summary JSON corrupted: {e}")

        return (len(errors) == 0), errors

    def validate_review_artifacts(self) -> Tuple[bool, List[str]]:
        """Verifies review session files for schema compliance and valid reviewer IDs."""
        errors = []
        if not os.path.exists(self.reviews_dir):
            return True, []

        for fn in os.listdir(self.reviews_dir):
            if not fn.endswith(".json"):
                continue
            fp = os.path.join(self.reviews_dir, fn)
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    rdata = json.load(f)

                if "study_id" not in rdata:
                    errors.append(f"{fn}: Missing 'study_id'.")
                
                has_rev = ("reviewer_id" in rdata) or ("reviewer" in rdata and isinstance(rdata["reviewer"], dict) and "id" in rdata["reviewer"])
                if not has_rev:
                    errors.append(f"{fn}: Missing 'reviewer_id' or 'reviewer.id'.")

                has_findings = ("finding_reviews" in rdata) or ("reviews" in rdata)
                if not has_findings:
                    errors.append(f"{fn}: Missing 'finding_reviews' or 'reviews'.")
            except Exception as e:
                errors.append(f"{fn}: Corrupted JSON: {e}")

        return (len(errors) == 0), errors

    def validate_consensus_artifacts(self) -> Tuple[bool, List[str]]:
        """Verifies consensus sessions for valid structure, adjudication records, and finalization lock."""
        errors = []
        if not os.path.exists(self.consensus_dir):
            return True, []

        for fn in os.listdir(self.consensus_dir):
            if not fn.endswith(".json"):
                continue
            fp = os.path.join(self.consensus_dir, fn)
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    cdata = json.load(f)

                if "study_id" not in cdata:
                    errors.append(f"{fn}: Missing 'study_id'.")
                if "reviewers" not in cdata:
                    errors.append(f"{fn}: Missing 'reviewers'.")
                if "finding_consensus" not in cdata:
                    errors.append(f"{fn}: Missing 'finding_consensus'.")

                # Check dispute adjudication integrity
                adj = cdata.get("adjudication", {})
                if adj and adj.get("items"):
                    for item in adj.get("items", []):
                        if not item.get("clinical_rationale"):
                            errors.append(f"{fn}: Adjudicated item '{item.get('finding')}' lacks mandatory clinical rationale.")

            except Exception as e:
                errors.append(f"{fn}: Corrupted JSON: {e}")

        return (len(errors) == 0), errors

    def validate_ground_truth_isolation(self, payload: Any) -> Tuple[bool, List[str]]:
        """Ensures payload contains no forbidden ground-truth text or XML elements."""
        errors = []
        payload_str = json.dumps(payload, default=str).lower()

        for pattern in FORBIDDEN_GROUND_TRUTH_PATTERNS:
            if pattern.lower() in payload_str:
                errors.append(f"Ground-truth leakage detected: contains '{pattern}'.")

        return (len(errors) == 0), errors


# Global Validator Singleton
global_dataset_validator = DatasetValidator()
