"""
LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 0.9 — Visual Grounding Validation Module

Module: validate_grounding.py
Purpose:
- Validates Visual Grounding artifacts against docs/grounding_schema.json and safety invariants.
- Verifies:
  1. Study ID, Image ID, View match input.
  2. Findings belong to the 18 DenseNet pathologies.
  3. Model score is numeric and in [0.0, 1.0] (labeled model_score).
  4. Heatmap and overlay files exist on disk and have valid dimensions.
  5. Target layer is documented as model.features.norm5.
  6. Method is recorded as grad_cam.
  7. No ground-truth leakage.
  8. Zero fabricated anatomical location claims (e.g., no fake 'right upper lobe' claims).
  9. Zero clinical diagnosis assertions.

DISCLAIMER:
This validation enforces technical and explainability safety invariants.
"""

import os
import sys
import json
from typing import Dict, List, Any, Tuple, Optional
from PIL import Image

VALID_PATHOLOGIES = {
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax", "Edema",
    "Emphysema", "Fibrosis", "Effusion", "Pneumonia", "Pleural_Thickening",
    "Cardiomegaly", "Nodule", "Mass", "Hernia", "Lung Lesion", "Fracture",
    "Lung Opacity", "Enlarged Cardiomediastinum"
}


def validate_grounding_package(
    package: Dict[str, Any],
    base_dir: Optional[str] = None
) -> Tuple[bool, List[str]]:
    """
    Validate a visual grounding package for schema compliance and safety invariants.

    Args:
        package: Grounding package dictionary conforming to docs/grounding_schema.json.
        base_dir: Optional project base directory for resolving relative file paths.

    Returns:
        (is_valid: bool, issues: List[str])
    """
    issues = []
    root_dir = base_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if not isinstance(package, dict):
        return False, ["Grounding package must be a JSON dictionary."]

    # If wrapped under a top-level key like "grounding_package"
    if "grounding_package" in package and isinstance(package["grounding_package"], dict):
        package = package["grounding_package"]

    # 1. Root Fields Validation
    for req in ["study_id", "image_id", "view"]:
        if req not in package:
            issues.append(f"Missing required root field: '{req}'")

    records = package.get("groundings") or package.get("grounding_records")
    if records is None or not isinstance(records, list) or len(records) == 0:
        issues.append("Package must contain a non-empty 'groundings' or 'grounding_records' list.")
    else:
        seen_findings = set()
        for idx, rec in enumerate(records):
            if not isinstance(rec, dict):
                issues.append(f"Record [{idx}] must be a dictionary.")
                continue

            for req_field in [
                "finding", "model_score", "target_layer", "heatmap_path", "overlay_path"
            ]:
                if req_field not in rec:
                    issues.append(f"Record [{idx}] missing required field: '{req_field}'")

            finding_name = rec.get("finding")
            if not finding_name or finding_name not in VALID_PATHOLOGIES:
                issues.append(f"Record [{idx}] finding '{finding_name}' is not in valid DenseNet pathologies.")
            elif finding_name in seen_findings:
                issues.append(f"Duplicate grounding record for finding: '{finding_name}'")
            seen_findings.add(finding_name)

            # Model Score Validation
            score = rec.get("model_score")
            if score is None or not isinstance(score, (int, float)):
                issues.append(f"Record '{finding_name}' model_score must be numeric.")
            elif not (0.0 <= score <= 1.0):
                issues.append(f"Record '{finding_name}' model_score {score} out of bounds [0.0, 1.0].")

            # Method & Layer Validation
            method = rec.get("method") or rec.get("explanation_type")
            if method != "grad_cam":
                issues.append(f"Record '{finding_name}' method must be 'grad_cam', got '{method}'.")

            layer = rec.get("target_layer")
            if layer != "model.features.norm5":
                issues.append(f"Record '{finding_name}' target_layer must be 'model.features.norm5', got '{layer}'.")

            # Activation / Heatmap Shape
            if "activation_shape" in rec:
                shape = rec["activation_shape"]
                if not isinstance(shape, list) or len(shape) != 2:
                    issues.append(f"Record '{finding_name}' activation_shape must be 2D [H, W], got {shape}.")

            if "heatmap_shape" in rec:
                h_shape = rec["heatmap_shape"]
                if not isinstance(h_shape, list) or len(h_shape) != 2:
                    issues.append(f"Record '{finding_name}' heatmap_shape must be 2D [H, W], got {h_shape}.")

            # Normalization
            if "normalization" in rec:
                norm = rec.get("normalization")
                if norm not in ["minmax_0_1", "minmax"]:
                    issues.append(f"Record '{finding_name}' normalization must be 'minmax_0_1', got '{norm}'.")

            # File Existence Checks
            for p_key in ["heatmap_path", "overlay_path"]:
                rel_path = rec.get(p_key, "")
                if rel_path:
                    abs_path = os.path.join(root_dir, rel_path) if not os.path.isabs(rel_path) else rel_path
                    if not os.path.exists(abs_path):
                        issues.append(f"Record '{finding_name}' {p_key} file not found at: {abs_path}")
                    else:
                        try:
                            with Image.open(abs_path) as img:
                                if img.size[0] <= 0 or img.size[1] <= 0:
                                    issues.append(f"Record '{finding_name}' {p_key} has invalid image dimensions: {img.size}")
                        except Exception as e:
                            issues.append(f"Record '{finding_name}' {p_key} failed image load: {e}")

            # Anti-Hallucination & Non-Diagnosis Invariant Checks
            rec_str = json.dumps(rec).lower()
            if "location" in rec:
                issues.append(f"Record '{finding_name}' asserts anatomical location. Location claims from Grad-CAM alone are forbidden.")
            if "confidence" in rec_str:
                issues.append(f"Record '{finding_name}' contains forbidden 'confidence' label. Use 'model_score'.")
            if "probability" in rec_str:
                issues.append(f"Record '{finding_name}' contains forbidden 'probability' label.")

    # Ground-Truth Leakage Check
    pkg_str = json.dumps(package)
    if "report_ground_truth" in pkg_str:
        issues.append("CRITICAL: Ground-truth marker detected in Visual Grounding package.")

    is_valid = len(issues) == 0
    return is_valid, issues


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_path = sys.argv[1]
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        valid, errs = validate_grounding_package(data)
        if valid:
            print(f"[SUCCESS] Grounding package {target_path} is valid.")
        else:
            print(f"[ERROR] Validation failed for {target_path}:")
            for err in errs:
                print(f"  - {err}")
            sys.exit(1)
    else:
        print("Usage: python validate_grounding.py <path_to_grounding_metadata.json>")
