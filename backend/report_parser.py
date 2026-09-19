"""
LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 0.6 — Report Parser & Ground Truth Extraction

Module: report_parser.py
Purpose:
- Parses IU X-Ray XML reports (NLM ecgen-radiology schema).
- Extracts study_id, associated parentImage identifiers, FINDINGS, IMPRESSION, INDICATION, COMPARISON.
- Converts unstructured report text into structured ground-truth finding representations.
- Distinguishes positive mentions, explicit negations ('no pneumothorax'), and ambiguous findings ('uncertain').
- Extracts anatomical location and severity descriptors when reliably documented.

DISCLAIMER:
Rule-based ground truth extraction from unstructured text is an NLP approximation.
Ambiguous findings are classified as 'uncertain' rather than guessed.
"""

import os
import sys
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Any

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


# Clinical finding pattern definitions for IU X-Ray report parsing
REPORT_FINDING_PATTERNS = {
    "Pneumothorax": {
        "positive": r"\b(pneumothorax|pneumothoraces)\b",
        "negation": r"\b(no|without|negative for|free of|no evidence of|no sign of)\s+([a-zA-Z\s]{0,30}\s+)?(pneumothorax|pneumothoraces)\b"
    },
    "Effusion": {
        "positive": r"\b(pleural effusion|effusion|blunting of (the )?costophrenic (angle|sulcus))\b",
        "negation": r"\b(no|without|negative for|free of|no evidence of|no radiographic evidence of)\s+([a-zA-Z\s]{0,30}\s+)?(pleural effusion|effusion|blunting)\b"
    },
    "Consolidation": {
        "positive": r"\b(consolidation|airspace disease|air space disease|focal consolidation)\b",
        "negation": r"\b(no|without|negative for|free of|no evidence of)\s+([a-zA-Z\s]{0,30}\s+)?(consolidation|airspace disease|air space disease)\b"
    },
    "Cardiomegaly": {
        "positive": r"\b(cardiomegaly|cardiac enlargement|enlarged heart|prominent cardiac silhouette|heart is enlarged)\b",
        "negation": r"\b(heart size (is )?normal|cardiac silhouette (and mediastinum )?(size )?(are )?within normal limits|normal heart size|no cardiomegaly)\b"
    },
    "Edema": {
        "positive": r"\b(pulmonary edema|edema|vascular congestion|interstitial edema)\b",
        "negation": r"\b(no|without|negative for|free of|no evidence of)\s+([a-zA-Z\s]{0,30}\s+)?(pulmonary edema|edema|vascular congestion|congestion)\b"
    },
    "Atelectasis": {
        "positive": r"\b(atelectasis|atelectatic|subsegmental atelectasis|linear opacity|bandlike opacity|platelike atelectasis)\b",
        "negation": r"\b(no|without|free of|no evidence of)\s+([a-zA-Z\s]{0,30}\s+)?(atelectasis|atelectatic)\b"
    },
    "Lung Opacity": {
        "positive": r"\b(opacity|opacities|density|densities|infiltrate|infiltrates|focal opacity)\b",
        "negation": r"\b(lungs are clear|clear of focal|no focal (air space )?opacit(y|ies)|no infiltrat(e|es)|clear lungs)\b"
    },
    "Nodule": {
        "positive": r"\b(nodule|nodules|calcified granuloma|granuloma|granulomas|nodular density)\b",
        "negation": r"\b(no|without|free of|no evidence of)\s+([a-zA-Z\s]{0,30}\s+)?(nodule|nodules|granuloma|granulomas)\b"
    },
    "Mass": {
        "positive": r"\b(lung mass|mass|masses|neoplasm|tumor)\b",
        "negation": r"\b(no|without|free of|no evidence of)\s+([a-zA-Z\s]{0,30}\s+)?(mass|masses)\b"
    },
    "Fracture": {
        "positive": r"\b(fracture|fractures|broken rib|rib fracture|compression deformity)\b",
        "negation": r"\b(no acute (bone|osseous|fracture)|no fracture|osseous structures appear intact|intact)\b"
    },
    "Hernia": {
        "positive": r"\b(hiatal hernia|hiatus hernia|retrocardiac density.*air)\b",
        "negation": r"\b(no (hiatal )?hernia)\b"
    },
    "Emphysema": {
        "positive": r"\b(emphysema|hyperinflation|hyperexpansion|flattened diaphragms)\b",
        "negation": r"\b(no emphysema|normal lung volumes)\b"
    },
    "Fibrosis": {
        "positive": r"\b(fibrosis|fibrotic changes|interstitial markings|reticular markings)\b",
        "negation": r"\b(no fibrosis|normal interstitial markings)\b"
    }
}

LOCATION_MATCHERS = [
    ("right_upper_lobe", r"\b(right upper (lobe|lung)|rul)\b"),
    ("right_middle_lobe", r"\b(right middle (lobe|lung)|rml)\b"),
    ("right_lower_lobe", r"\b(right lower (lobe|lung)|rll)\b"),
    ("left_upper_lobe", r"\b(left upper (lobe|lung)|lul)\b"),
    ("left_lower_lobe", r"\b(left lower (lobe|lung)|lll)\b"),
    ("basilar", r"\b(basilar|bases|lung base|costophrenic)\b"),
    ("apical", r"\b(apical|apex|apices)\b"),
    ("retrocardiac", r"\b(retrocardiac)\b"),
    ("bilateral", r"\b(bilateral|both lungs|both bases|diffuse)\b"),
    ("right", r"\b(right|right-sided|rt)\b"),
    ("left", r"\b(left|left-sided|lt)\b")
]

SEVERITY_MATCHERS = [
    ("small", r"\b(small|tiny|subtle|minor)\b"),
    ("minimal", r"\b(minimal|trace)\b"),
    ("moderate", r"\b(moderate|modest)\b"),
    ("large", r"\b(large|gross|marked|extensive|severe|dense)\b"),
    ("mild", r"\b(mild)\b")
]


def extract_location(text_snippet: str) -> str:
    """Extract location descriptor from text snippet or return 'unspecified'."""
    lower = text_snippet.lower()
    for loc_name, pattern in LOCATION_MATCHERS:
        if re.search(pattern, lower):
            return loc_name
    return "unspecified"


def extract_severity(text_snippet: str) -> str:
    """Extract severity descriptor from text snippet or return 'unspecified'."""
    lower = text_snippet.lower()
    for sev_name, pattern in SEVERITY_MATCHERS:
        if re.search(pattern, lower):
            return sev_name
    return "unspecified"


def parse_iu_report(xml_path: str) -> Dict[str, Any]:
    """
    Parse a single IU X-Ray XML report file and extract structured ground-truth findings.

    Args:
        xml_path: Absolute or relative path to the XML report.

    Returns:
        Dictionary containing study metadata, report sections, parent images, and ground-truth findings.
    """
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML report file not found at: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Extract Study ID
    uid_node = root.find("uId")
    study_id = uid_node.attrib.get("id") if uid_node is not None else os.path.splitext(os.path.basename(xml_path))[0]

    # Extract Parent Images
    parent_images = []
    for p_img in root.findall(".//parentImage"):
        img_id = p_img.attrib.get("id")
        if img_id:
            caption_elem = p_img.find("caption")
            caption_text = (caption_elem.text or "").strip() if caption_elem is not None else ""
            parent_images.append({
                "image_id": img_id,
                "caption": caption_text
            })

    # Extract Report Sections
    findings = ""
    impression = ""
    indication = ""
    comparison = ""

    for abs_text in root.iter("AbstractText"):
        label = (abs_text.attrib.get("Label") or "").upper().strip()
        text = (abs_text.text or "").strip()
        if label == "FINDINGS":
            findings = text
        elif label == "IMPRESSION":
            impression = text
        elif label == "INDICATION":
            indication = text
        elif label == "COMPARISON":
            comparison = text

    combined_text = f"{findings} {impression}".strip()

    # Rule-Based Ground-Truth Finding Extraction
    ground_truth_findings = []

    for finding_name, patterns in REPORT_FINDING_PATTERNS.items():
        pos_pat = patterns["positive"]
        neg_pat = patterns["negation"]

        # Check for explicit negation first
        neg_match = re.search(neg_pat, combined_text, re.IGNORECASE)
        if neg_match:
            ground_truth_findings.append({
                "finding": finding_name,
                "presence": "no",
                "location": "unspecified",
                "severity": "unspecified",
                "evidence_source": "report_ground_truth",
                "matched_text": neg_match.group(0)
            })
            continue

        # Check for positive mentions
        pos_matches = list(re.finditer(pos_pat, combined_text, re.IGNORECASE))
        if pos_matches:
            # Check context around match for location and severity
            m = pos_matches[0]
            start_idx = max(0, m.start() - 60)
            end_idx = min(len(combined_text), m.end() + 60)
            context_window = combined_text[start_idx:end_idx]

            # Check if there is ambiguity or uncertainty words (e.g., 'possible', 'cannot exclude', 'equivocal')
            is_uncertain = bool(re.search(r"\b(possible|may represent|cannot rule out|cannot exclude|borderline|equivocal|suggesting)\b", context_window, re.IGNORECASE))
            presence_val = "uncertain" if is_uncertain else "yes"

            loc_val = extract_location(context_window)
            sev_val = extract_severity(context_window)

            ground_truth_findings.append({
                "finding": finding_name,
                "presence": presence_val,
                "location": loc_val,
                "severity": sev_val,
                "evidence_source": "report_ground_truth",
                "matched_text": m.group(0)
            })

    # If lungs are explicitly reported clear/unremarkable, ensure normal profile
    if re.search(r"\b(lungs are clear|no acute cardiopulmonary|unremarkable|normal chest)\b", combined_text, re.IGNORECASE):
        # Explicitly verify core baseline presence
        existing_names = {f["finding"] for f in ground_truth_findings}
        for routine_check in ["Pneumothorax", "Effusion", "Consolidation"]:
            if routine_check not in existing_names:
                ground_truth_findings.append({
                    "finding": routine_check,
                    "presence": "no",
                    "location": "unspecified",
                    "severity": "unspecified",
                    "evidence_source": "report_ground_truth",
                    "matched_text": "Normal report synthesis"
                })

    report_data = {
        "study_id": study_id,
        "parent_images": parent_images,
        "sections": {
            "findings": findings,
            "impression": impression,
            "indication": indication,
            "comparison": comparison
        },
        "ground_truth_findings": ground_truth_findings
    }

    return report_data
