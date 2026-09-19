"""
LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 0.7 — End-to-End Evidence Layer Pipeline Test

Executes the complete Phase 0.7 pipeline on a real IU X-Ray sample:
1. Real IU X-Ray image (CXR1122_IM-0080-1001-0002.png)
2. Preprocessing & DenseNet-121 Inference (18 pathology scores + 1024-D features)
3. Diagnostic QA Engine Question Selection & Evaluation (Top-K=3, QA_THRESHOLD=0.35, Baseline set)
4. Evidence Layer Status Resolution (supported, possible, uncertain, absent) & Provenance Tracking
5. Production LLM Input Package Generation (with Anti-Hallucination Constraints & Zero Ground-Truth Leakage)
6. Comprehensive Validation via validate_evidence_package()
7. Serialization to data/iu_xray/e2e_evidence_validation_result.json

DISCLAIMER:
This pipeline is a research prototype. Model scores and structured evidence are pattern activations
and intermediate representations, NOT clinical diagnoses or verified medical facts.
The Evidence Layer preserves uncertainty and does not treat model scores as clinical diagnoses.
"""

import os
import sys
import json
import torch
import torchvision
import skimage.io
import torchxrayvision as xrv

# Ensure local backend imports work
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from diagnostic_qa import DiagnosticQAEngine
from evidence_layer import EvidenceLayer
from report_parser import parse_iu_report
from validate_evidence import validate_evidence_package


def preprocess_image_tensor(image_path: str, device: torch.device) -> torch.Tensor:
    """Preprocess chest X-ray using TorchXRayVision utilities."""
    img = skimage.io.imread(image_path)
    if img.ndim == 3:
        if img.shape[2] >= 3:
            img = img[:, :, :3].mean(axis=2)
        else:
            img = img[:, :, 0]
    maxval = 65535 if img.max() > 255 else 255
    img = xrv.datasets.normalize(img, maxval=maxval)
    img_3d = img[None, ...]
    transform = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224)
    ])
    processed = transform(img_3d)
    return torch.from_numpy(processed).unsqueeze(0).float().to(device)


def run_e2e_evidence_pipeline():
    print("=" * 70)
    print("PHASE 0.7 — END-TO-END EVIDENCE LAYER & LLM INPUT PIPELINE TEST")
    print("=" * 70)

    image_path = os.path.join(BASE_DIR, "data", "iu_xray", "images", "CXR1122_IM-0080-1001-0002.png")
    report_path = os.path.join(BASE_DIR, "data", "iu_xray", "reports", "ecgen-radiology", "1122.xml")
    output_result_path = os.path.join(BASE_DIR, "data", "iu_xray", "e2e_evidence_validation_result.json")

    if not os.path.exists(image_path):
        print(f"[ERROR] Image file not found: {image_path}")
        sys.exit(1)
    if not os.path.exists(report_path):
        print(f"[ERROR] Report file not found: {report_path}")
        sys.exit(1)

    device = torch.device("cpu")
    print(f"Device Selected          : {device}")
    print(f"Image Path               : {os.path.relpath(image_path, BASE_DIR)}")
    print(f"Report Path (Ref only)   : {os.path.relpath(report_path, BASE_DIR)}\n")

    # Step 1: Load Vision Backbone
    print("1. Loading TorchXRayVision DenseNet-121...")
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval()
    model.to(device)
    print("   Model loaded successfully.")

    # Step 2: Preprocessing & Inference
    print("\n2. Preprocessing Radiograph & Extracting Activations...")
    input_tensor = preprocess_image_tensor(image_path, device)
    with torch.no_grad():
        features = model.features2(input_tensor)
        outputs = model(input_tensor)

    scores = outputs[0].detach().cpu().numpy()
    pathology_scores = {name: float(score) for name, score in zip(model.pathologies, scores)}
    print(f"   Input Tensor Shape    : {list(input_tensor.shape)}")
    print(f"   Visual Feature Shape  : {list(features.shape)} (1024-D)")
    print(f"   Evaluated Conditions  : {len(pathology_scores)}")

    # Step 3: Diagnostic QA Engine
    print("\n3. Executing Diagnostic QA Engine...")
    qa_engine = DiagnosticQAEngine(top_k=3, qa_threshold=0.35)
    qa_output = qa_engine.evaluate_study(
        study_id="CXR1122",
        image_id="CXR1122_IM-0080-1001-0002",
        view="Frontal",
        pathology_scores=pathology_scores
    )
    print(f"   Candidate Count       : {len(qa_output['vision_candidates'])}")
    print(f"   Questions Evaluated   : {len(qa_output['questions_evaluated'])}")

    # Step 4: Evidence Layer Status Resolution
    print("\n4. Building Evidence Layer Package & Status Resolution...")
    evidence_layer = EvidenceLayer()
    evidence_package = evidence_layer.build_evidence_package(qa_output)
    print(f"   Evidence Findings     : {len(evidence_package['findings'])}")

    print("\n   Evidence Status Breakdown:")
    for f in evidence_package["findings"]:
        print(f"   - {f['finding']:<24}: status = {f['status']:<10} | model_score = {f['model_score']:.4f} | sources = {f['evidence_sources']}")

    # Step 5: Clean LLM Input Package Inspection
    print("\n5. Inspecting Production LLM Input Package...")
    llm_pkg = evidence_package["llm_input_package"]
    print(f"   Task                  : {llm_pkg['task']}")
    print(f"   Study                 : {llm_pkg['study']['study_id']} ({llm_pkg['study']['view']})")
    print(f"   Evidence Item Count   : {len(llm_pkg['evidence'])}")
    print(f"   Constraints Enforced  : {list(llm_pkg['constraints'].keys())}")

    # Step 6: Schema & Safety Validation
    print("\n6. Running Schema & Safety Invariant Validation...")
    is_valid, validation_issues = validate_evidence_package(evidence_package)
    if not is_valid:
        print(f"[ERROR] Validation failed with errors:")
        for err in validation_issues:
            print(f"  - {err}")
        sys.exit(1)
    print("   Validation Result     : PASSED (Zero schema errors, zero ground-truth leakage)")

    # Step 7: Save Result JSON
    os.makedirs(os.path.dirname(output_result_path), exist_ok=True)
    with open(output_result_path, "w", encoding="utf-8") as f:
        json.dump(evidence_package, f, indent=4)
    print(f"\nSaved full end-to-end evidence package to: {output_result_path}")

    print("\n" + "=" * 70)
    print("PHASE 0.7 END-TO-END VALIDATION SUMMARY: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    run_e2e_evidence_pipeline()
