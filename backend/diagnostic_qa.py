"""
LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 0.6 — Diagnostic QA Engine

Module: diagnostic_qa.py
Purpose:
- Deterministic, configurable Diagnostic Questioning Engine.
- Selects candidate radiological findings using a hybrid strategy:
  1. Routine baseline findings (Pneumothorax, Effusion, Consolidation, Cardiomegaly).
  2. Top-K additional findings based on model activation scores.
  3. Findings exceeding a configurable activation threshold (QA_THRESHOLD).
- Formulates multi-level conditional diagnostic questions (Presence -> Location -> Severity).
- Produces machine-readable structured findings adhering to docs/qa_schema.json.

DISCLAIMER:
This module is a research prototype. Model scores and structured findings are pattern activations
and QA representations, NOT clinical diagnoses or verified medical facts.
"""

import sys
import json
from typing import Dict, List, Optional, Any, Union

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


# Prototype engineering configuration defaults (NOT medically validated thresholds)
DEFAULT_BASELINE_FINDINGS = [
    "Pneumothorax",
    "Effusion",
    "Consolidation",
    "Cardiomegaly"
]

DEFAULT_TOP_K = 3
DEFAULT_QA_THRESHOLD = 0.35

# Deterministic Question Templates for 18 TorchXRayVision pathologies
QUESTION_TEMPLATES = {
    "Pneumothorax": {
        "level1": "Is there radiographic evidence of a pneumothorax?",
        "level2": "Where is the pneumothorax located (e.g., right, left, apical)?",
        "level3": "What is the extent of the pneumothorax (e.g., small, moderate, large)?"
    },
    "Effusion": {
        "level1": "Is there evidence of pleural effusion or costophrenic angle blunting?",
        "level2": "Where is the pleural effusion located (e.g., right, left, bilateral)?",
        "level3": "What is the volume/extent of the effusion (e.g., small, moderate, large, trace)?"
    },
    "Consolidation": {
        "level1": "Is there focal or dense airspace consolidation?",
        "level2": "Which lung lobe or region contains the consolidation?",
        "level3": "What is the extent/density of the consolidation?"
    },
    "Cardiomegaly": {
        "level1": "Is cardiomegaly or prominent cardiac enlargement present?",
        "level2": "Is the enlargement generalized or specific to cardiac chambers?",
        "level3": "What is the degree of cardiomegaly (e.g., mild, moderate, severe)?"
    },
    "Atelectasis": {
        "level1": "Is there evidence of linear, bandlike, or lobar atelectasis?",
        "level2": "Where is the atelectasis located (e.g., basilar, right_upper_lobe, left_lower_lobe)?",
        "level3": "What is the extent of the collapse (e.g., subsegmental, plate-like, lobar)?"
    },
    "Infiltration": {
        "level1": "Are there infiltrative or hazy airspace opacities?",
        "level2": "Where are the infiltrates distributed (e.g., right, left, bilateral)?",
        "level3": "What is the severity/extent of the infiltrate?"
    },
    "Edema": {
        "level1": "Is there radiographic evidence of pulmonary edema or vascular congestion?",
        "level2": "What is the distribution of edema (e.g., perihilar, bilateral, diffuse)?",
        "level3": "What is the severity of pulmonary edema (e.g., mild, moderate, severe)?"
    },
    "Emphysema": {
        "level1": "Are there signs of emphysema, hyperinflation, or flattened diaphragms?",
        "level2": "Is the hyperinflation bilateral or localized?",
        "level3": "What is the degree of emphysematous changes?"
    },
    "Fibrosis": {
        "level1": "Is there evidence of pulmonary fibrosis, reticular markings, or scarring?",
        "level2": "Where are the fibrotic markings located (e.g., apical, bibasilar, diffuse)?",
        "level3": "What is the extent of fibrotic change?"
    },
    "Pneumonia": {
        "level1": "Are there features suggestive of infectious pneumonia / bronchopneumonia?",
        "level2": "Where is the suspected pneumonia located?",
        "level3": "What is the extent of pulmonary involvement?"
    },
    "Pleural_Thickening": {
        "level1": "Is there evidence of pleural thickening or apical capping?",
        "level2": "Where is the pleural thickening located (e.g., right, left, apical)?",
        "level3": "Is the pleural thickening calcified or non-calcified?"
    },
    "Nodule": {
        "level1": "Is there evidence of pulmonary nodule(s) or calcified granuloma(s)?",
        "level2": "Where is the nodule located (e.g., right_upper_lobe, left_midlung)?",
        "level3": "What is the approximate size/characteristics (e.g., solitary, multiple, calcified)?"
    },
    "Mass": {
        "level1": "Is there a pulmonary or mediastinal mass lesion (>3 cm)?",
        "level2": "Where is the mass located (e.g., hilar, right_upper_lobe, mediastinum)?",
        "level3": "What are the mass characteristics and borders?"
    },
    "Hernia": {
        "level1": "Is there radiographic evidence of a hiatal hernia or diaphragmatic defect?",
        "level2": "Where is the hernia situated (e.g., retrocardiac space)?",
        "level3": "Does the hernia exhibit an air-fluid level?"
    },
    "Lung Lesion": {
        "level1": "Is there a focal parenchymal or cavitary lung lesion?",
        "level2": "Where is the lesion located?",
        "level3": "What are the structural features of the lesion?"
    },
    "Fracture": {
        "level1": "Is there evidence of acute or healed fractures (ribs, clavicle, spine)?",
        "level2": "Which bone is fractured (e.g., rib_right, clavicle_left, spine)?",
        "level3": "Is the fracture acute, displaced, or healing?"
    },
    "Lung Opacity": {
        "level1": "Are there patchy or streaky lung opacities present?",
        "level2": "Where are the opacities distributed (e.g., right, left, bibasilar)?",
        "level3": "What is the density and extent of the opacities?"
    },
    "Enlarged Cardiomediastinum": {
        "level1": "Is there widening of the cardiomediastinal silhouette or aortic tortuosity?",
        "level2": "Is the widening primarily cardiac, mediastinal, or aortic?",
        "level3": "What is the extent of mediastinal prominence?"
    }
}


class DiagnosticQAEngine:
    """
    Deterministic Diagnostic Questioning Engine for Chest X-Ray Analysis.
    """

    def __init__(
        self,
        baseline_findings: Optional[List[str]] = None,
        top_k: int = DEFAULT_TOP_K,
        qa_threshold: float = DEFAULT_QA_THRESHOLD
    ):
        """
        Initialize the QA engine with configurable prototype parameters.

        Args:
            baseline_findings: List of routine baseline findings to always evaluate.
            top_k: Number of highest-activation non-baseline findings to include.
            qa_threshold: Minimum activation threshold for triggering non-baseline findings.
        """
        self.baseline_findings = list(baseline_findings) if baseline_findings is not None else list(DEFAULT_BASELINE_FINDINGS)
        self.top_k = int(top_k)
        self.qa_threshold = float(qa_threshold)

    def select_candidates(
        self,
        pathology_scores: Union[Dict[str, float], List[tuple]],
        top_k: Optional[int] = None,
        qa_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Select candidate findings from the vision model's 18 pathology activation scores.
        Deduplicates selections and tracks selection reason.

        Returns:
            List of candidate dictionaries:
            [{"finding": str, "model_score": float, "is_baseline": bool, "selection_reason": str}]
        """
        k = self.top_k if top_k is None else int(top_k)
        thresh = self.qa_threshold if qa_threshold is None else float(qa_threshold)

        # Normalize input to dictionary
        if isinstance(pathology_scores, list):
            score_dict = {name: float(score) for name, score in pathology_scores}
        elif isinstance(pathology_scores, dict):
            score_dict = {name: float(score) for name, score in pathology_scores.items()}
        else:
            raise ValueError(f"Unsupported pathology_scores type: {type(pathology_scores)}")

        selected_map = {}

        # 1. Routine Baseline Inclusions
        for base_finding in self.baseline_findings:
            score = score_dict.get(base_finding, 0.0)
            selected_map[base_finding] = {
                "finding": base_finding,
                "model_score": round(score, 4),
                "is_baseline": True,
                "selection_reason": "baseline_routine"
            }

        # 2. Sort remaining non-baseline findings by score descending
        non_baseline = [
            (name, score) for name, score in score_dict.items()
            if name not in selected_map
        ]
        non_baseline.sort(key=lambda x: x[1], reverse=True)

        # 3. Add Top-K non-baseline candidates
        for name, score in non_baseline[:k]:
            if name not in selected_map:
                selected_map[name] = {
                    "finding": name,
                    "model_score": round(score, 4),
                    "is_baseline": False,
                    "selection_reason": f"top_{k}_activation"
                }

        # 4. Add any remaining candidates exceeding qa_threshold
        for name, score in non_baseline[k:]:
            if score >= thresh and name not in selected_map:
                selected_map[name] = {
                    "finding": name,
                    "model_score": round(score, 4),
                    "is_baseline": False,
                    "selection_reason": f"threshold_exceeded_>={thresh}"
                }

        return list(selected_map.values())

    def generate_questions_for_candidate(
        self,
        candidate: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate deterministic multi-level question definitions for a given candidate finding.
        """
        finding = candidate["finding"]
        score = candidate["model_score"]
        templates = QUESTION_TEMPLATES.get(finding, {
            "level1": f"Is there radiographic evidence of {finding}?",
            "level2": f"Where is the {finding} located?",
            "level3": f"What is the severity/extent of the {finding}?"
        })

        questions = [
            {
                "question_id": f"Q_{finding.upper()}_PRESENCE",
                "finding": finding,
                "level": 1,
                "question_text": templates["level1"],
                "options": ["yes", "no", "uncertain"],
                "model_score": score
            },
            {
                "question_id": f"Q_{finding.upper()}_LOCATION",
                "finding": finding,
                "level": 2,
                "question_text": templates["level2"],
                "options": [
                    "right", "left", "bilateral", "right_upper_lobe", "right_middle_lobe",
                    "right_lower_lobe", "left_upper_lobe", "left_lower_lobe", "basilar",
                    "apical", "retrocardiac", "mediastinum", "diffuse", "unspecified"
                ],
                "model_score": score
            },
            {
                "question_id": f"Q_{finding.upper()}_SEVERITY",
                "finding": finding,
                "level": 3,
                "question_text": templates["level3"],
                "options": [
                    "small", "minimal", "moderate", "large", "mild", "severe", "trace", "unspecified"
                ],
                "model_score": score
            }
        ]
        return questions

    def evaluate_study(
        self,
        study_id: str,
        image_id: str,
        view: str,
        pathology_scores: Union[Dict[str, float], List[tuple]],
        supplied_answers: Optional[Dict[str, Dict[str, str]]] = None,
        top_k: Optional[int] = None,
        qa_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Execute full diagnostic questioning evaluation on a study.

        Args:
            study_id: Unique identifier for the study (e.g., 'CXR1122').
            image_id: Unique identifier for the radiograph.
            view: Radiograph projection ('Frontal', 'Lateral', 'Unspecified').
            pathology_scores: 18 pathology activations from TorchXRayVision.
            supplied_answers: Optional explicit dictionary of answers per finding:
                {"Effusion": {"presence": "yes", "location": "right", "severity": "small"}}
            top_k: Override for top-k selection.
            qa_threshold: Override for threshold filtering.

        Returns:
            Structured dictionary compliant with docs/qa_schema.json.
        """
        candidates = self.select_candidates(pathology_scores, top_k=top_k, qa_threshold=qa_threshold)
        user_answers = supplied_answers or {}

        questions_evaluated = []
        qa_findings = []

        for cand in candidates:
            finding = cand["finding"]
            score = cand["model_score"]
            finding_answers = user_answers.get(finding, {})

            q_defs = self.generate_questions_for_candidate(cand)

            # Determine Level 1: Presence
            if "presence" in finding_answers:
                presence_ans = finding_answers["presence"].lower()
            else:
                # In prototype mode without human-in-the-loop:
                # If unsupplied, explicitly mark presence as uncertain unless user provided answer.
                presence_ans = "uncertain"

            q1_entry = dict(q_defs[0])
            q1_entry["answer"] = presence_ans
            questions_evaluated.append(q1_entry)

            # Conditional Level 2 & Level 3
            location_ans = "unspecified"
            severity_ans = "unspecified"

            if presence_ans in ["yes", "uncertain"]:
                # Evaluate Location (Level 2)
                if "location" in finding_answers:
                    location_ans = finding_answers["location"].lower()
                q2_entry = dict(q_defs[1])
                q2_entry["answer"] = location_ans
                questions_evaluated.append(q2_entry)

                # Evaluate Severity (Level 3) only if presence is yes
                if presence_ans == "yes":
                    if "severity" in finding_answers:
                        severity_ans = finding_answers["severity"].lower()
                    q3_entry = dict(q_defs[2])
                    q3_entry["answer"] = severity_ans
                    questions_evaluated.append(q3_entry)

            # Synthesize final structured finding record
            # Non-hallucination guarantee: location/severity default to 'unspecified'
            qa_findings.append({
                "finding": finding,
                "presence": presence_ans,
                "location": location_ans,
                "severity": severity_ans,
                "model_score": score,
                "evidence_source": "diagnostic_qa" if finding_answers else "vision_model"
            })

        structured_output = {
            "study_id": study_id,
            "image_id": image_id,
            "view": view,
            "vision_candidates": candidates,
            "questions_evaluated": questions_evaluated,
            "qa_findings": qa_findings
        }

        return structured_output
