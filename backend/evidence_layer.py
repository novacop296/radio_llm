"""
LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 0.7 — Evidence Layer & LLM Input Preparation

Module: evidence_layer.py
Purpose:
- Acts as a strict, intermediate Evidence Layer between Vision/QA modules and future LLM report generation.
- Resolves candidate findings into 4 explicit evidence statuses:
  1. 'supported' : Explicitly established by supporting QA evidence.
  2. 'possible'  : Suggested by vision model activations but not independently confirmed.
  3. 'uncertain' : Unresolved, ambiguous, or missing evidence.
  4. 'absent'    : Explicitly negated by evidence (e.g., QA answer 'no').
- Preserves raw 'model_score' (never represented as clinical confidence or probability).
- Prevents hallucination of location/severity (defaults to 'unspecified').
- Tracks provenance sources ('vision_model', 'diagnostic_qa', 'rule_derived').
- Produces clean, safe LLM Input Packages with anti-hallucination constraints.
- Enforces strict ground-truth separation (report ground-truth is NEVER passed to production LLM input).

DISCLAIMER:
This module is a research prototype. Model scores and evidence representations are
pattern activations and intermediate structured states, NOT clinical diagnoses or verified medical facts.
The Evidence Layer preserves uncertainty and does not treat model scores as clinical diagnoses.
"""

import sys
from typing import Dict, List, Optional, Any, Union

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


VALID_STATUSES = {"supported", "possible", "uncertain", "absent"}
VALID_SOURCES = {"vision_model", "diagnostic_qa", "rule_derived", "grounding_module", "report_ground_truth"}


class EvidenceLayer:
    """
    Evidence Layer for structured, safe chest X-ray report generation preparation.
    """

    def __init__(self):
        pass

    @staticmethod
    def validate_score(score: Any) -> float:
        """Validate and return model_score bounded in [0.0, 1.0]."""
        if score is None:
            return 0.0
        try:
            val = float(score)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid model_score '{score}': must be numeric.")
        if not (0.0 <= val <= 1.0):
            raise ValueError(f"model_score {val} out of bounds: must be in [0.0, 1.0].")
        return round(val, 4)

    def resolve_finding_status(
        self,
        finding_name: str,
        qa_finding: Optional[Dict[str, Any]] = None,
        vision_candidate: Optional[Dict[str, Any]] = None,
        is_evaluation_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Determine the strict evidence status and provenance for a single finding.

        Rules:
        1. If QA answered 'yes' -> 'supported'
        2. If QA answered 'no' -> 'absent'
        3. If QA answered 'uncertain' or unanswered:
           - If suggested by vision model -> 'possible' (unless marked explicit uncertain)
           - Otherwise -> 'uncertain'
        4. Ground-truth sources are restricted to evaluation mode.
        """
        evidence_sources = []
        model_score = 0.0

        if vision_candidate:
            raw_score = vision_candidate.get("model_score", 0.0)
            model_score = self.validate_score(raw_score)
            evidence_sources.append("vision_model")

        location = "unspecified"
        severity = "unspecified"
        presence = "uncertain"

        if qa_finding:
            presence = qa_finding.get("presence", "uncertain").lower()
            location = qa_finding.get("location", "unspecified") or "unspecified"
            severity = qa_finding.get("severity", "unspecified") or "unspecified"
            raw_score = qa_finding.get("model_score", model_score)
            model_score = self.validate_score(raw_score)
            
            source = qa_finding.get("evidence_source", "diagnostic_qa")
            if source in VALID_SOURCES:
                if source == "report_ground_truth" and not is_evaluation_mode:
                    # Ground-truth leakage prevention: do not carry into production
                    pass
                else:
                    if source not in evidence_sources:
                        evidence_sources.append(source)

        # Status resolution logic
        if presence == "yes":
            status = "supported"
        elif presence == "no":
            status = "absent"
        elif presence == "uncertain":
            if "vision_model" in evidence_sources and model_score > 0.0:
                status = "possible"
            else:
                status = "uncertain"
        else:
            status = "uncertain"

        if not evidence_sources:
            evidence_sources.append("rule_derived")

        return {
            "finding": finding_name,
            "status": status,
            "model_score": model_score,
            "location": location,
            "severity": severity,
            "evidence_sources": evidence_sources
        }

    def build_evidence_package(
        self,
        qa_output: Dict[str, Any],
        ground_truth: Optional[Dict[str, Any]] = None,
        is_evaluation_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Assemble the full structured Evidence Layer package.

        Args:
            qa_output: Output dictionary from DiagnosticQAEngine containing
                       'study_id', 'image_id', 'view', 'vision_candidates', 'qa_findings'.
            ground_truth: Optional ground truth dictionary from report_parser (evaluation only).
            is_evaluation_mode: Flag indicating whether this is an evaluation run.

        Returns:
            Dictionary compliant with docs/evidence_schema.json.
        """
        study_id = qa_output.get("study_id", "Unknown")
        image_id = qa_output.get("image_id", "Unknown")
        view = qa_output.get("view", "Unspecified")

        vision_map = {c["finding"]: c for c in qa_output.get("vision_candidates", [])}
        qa_map = {f["finding"]: f for f in qa_output.get("qa_findings", [])}

        all_findings = set(vision_map.keys()).union(set(qa_map.keys()))

        evidence_findings = []
        for name in all_findings:
            v_cand = vision_map.get(name)
            qa_f = qa_map.get(name)
            record = self.resolve_finding_status(
                finding_name=name,
                qa_finding=qa_f,
                vision_candidate=v_cand,
                is_evaluation_mode=is_evaluation_mode
            )
            evidence_findings.append(record)

        # Sort findings: supported first, then possible (by model_score desc), then absent, then uncertain
        status_priority = {"supported": 0, "possible": 1, "absent": 2, "uncertain": 3}
        evidence_findings.sort(key=lambda x: (status_priority.get(x["status"], 4), -x["model_score"], x["finding"]))

        # Build clean LLM input package (ground-truth is strictly excluded)
        llm_input_pkg = self.build_llm_input(study_id, image_id, view, evidence_findings)

        package = {
            "study_id": study_id,
            "image_id": image_id,
            "view": view,
            "findings": evidence_findings,
            "llm_input_package": llm_input_pkg
        }

        if is_evaluation_mode and ground_truth:
            package["ground_truth_evaluation"] = ground_truth

        return package

    def build_llm_input(
        self,
        study_id: str,
        image_id: str,
        view: str,
        evidence_findings: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Build a sanitized, safe LLM input package with explicit guardrails.
        Enforces strict absence of ground-truth data to prevent leakage.
        """
        sanitized_evidence = []
        for item in evidence_findings:
            # Strip internal provenance sources or any ground-truth artifacts
            clean_item = {
                "finding": item["finding"],
                "status": item["status"],
                "model_score": item["model_score"],
                "location": item.get("location", "unspecified"),
                "severity": item.get("severity", "unspecified")
            }
            sanitized_evidence.append(clean_item)

        llm_input = {
            "task": "radiology_report_generation",
            "study": {
                "study_id": study_id,
                "image_id": image_id,
                "view": view
            },
            "evidence": sanitized_evidence,
            "constraints": {
                "do_not_invent_findings": True,
                "do_not_invent_location": True,
                "do_not_invent_severity": True,
                "preserve_uncertainty": True
            }
        }

        # Assert data-leakage safeguard
        package_str = str(llm_input)
        if "report_ground_truth" in package_str:
            raise RuntimeError("[CRITICAL] Ground-truth leakage detected in LLM Input Package!")

        return llm_input
