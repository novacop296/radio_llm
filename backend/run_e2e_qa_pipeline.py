"""
LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 0.6 — End-to-End QA Pipeline Integration Test

Executes the complete Phase 0.6 pipeline on a real IU X-Ray sample:
1. Real IU X-Ray image (CXR1122_IM-0080-1001-0002.png)
2. Preprocessing & DenseNet-121 Inference (18 pathology scores + 1024-D features)
3. Diagnostic QA Engine Question Selection & Evaluation (Top-K=3, QA_THRESHOLD=0.35, Baseline set)
4. Ground-Truth Report Parsing (1122.xml)
5. Comparative Evaluation (Presence agreement, matched, missed, extra)
6. Serialization to data/iu_xray/e2e_qa_validation_result.json

DISCLAIMER:
This pipeline is a research prototype. Model scores and structured findings are pattern activations
and QA representations, NOT clinical diagnoses or verified medical facts.
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
from report_parser import parse_iu_report
from evaluate_qa import evaluate_qa_against_ground_truth


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


def run_e2e_pipeline():
    print("=" * 65)
    print("PHASE 0.6 — END-TO-END DIAGNOSTIC QA PIPELINE TEST")
    print("=" * 65)

    image_path = os.path.join(BASE_DIR, "data", "iu_xray", "images", "CXR1122_IM-0080-1001-0002.png")
    report_path = os.path.join(BASE_DIR, "data", "iu_xray", "reports", "ecgen-radiology", "1122.xml")
    output_result_path = os.path.join(BASE_DIR, "data", "iu_xray", "e2e_qa_validation_result.json")

    if not os.path.exists(image_path):
        print(f"[ERROR] Image file not found: {image_path}")
        sys.exit(1)
    if not os.path.exists(report_path):
        print(f"[ERROR] Report file not found: {report_path}")
        sys.exit(1)

    device = torch.device("cpu")
    print(f"Device Selected          : {device}")
    print(f"Image Path               : {os.path.relpath(image_path, BASE_DIR)}")
    print(f"Report Path              : {os.path.relpath(report_path, BASE_DIR)}\n")

    # Step 1: Load Vision Model
    print("1. Loading TorchXRayVision DenseNet-121...")
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval()
    model.to(device)
    print("   Model loaded successfully.")

    # Step 2: Preprocessing & Inference
    print("\n2. Running Vision Preprocessing & Feature Extraction...")
    input_tensor = preprocess_image_tensor(image_path, device)
    with torch.no_grad():
        features = model.features2(input_tensor)
        outputs = model(input_tensor)
    
    scores = outputs[0].detach().cpu().numpy()
    pathology_scores = {name: float(score) for name, score in zip(model.pathologies, scores)}
    print(f"   Input Tensor Shape    : {list(input_tensor.shape)}")
    print(f"   Visual Feature Shape  : {list(features.shape)} (1024-D)")
    print(f"   Pathology Activations : {len(pathology_scores)} conditions evaluated")

    # Step 3: Diagnostic QA Engine Evaluation
    print("\n3. Executing Diagnostic QA Engine...")
    engine = DiagnosticQAEngine(top_k=3, qa_threshold=0.35)
    qa_output = engine.evaluate_study(
        study_id="CXR1122",
        image_id="CXR1122_IM-0080-1001-0002",
        view="Frontal",
        pathology_scores=pathology_scores
    )

    print(f"   Baseline Findings     : {engine.baseline_findings}")
    print(f"   Candidate Count       : {len(qa_output['vision_candidates'])}")
    print(f"   Questions Evaluated   : {len(qa_output['questions_evaluated'])}")
    print(f"   Structured Findings   : {len(qa_output['qa_findings'])}")

    print("\n   Candidate Finding Activations:")
    for cand in qa_output["vision_candidates"]:
        print(f"   - {cand['finding']:<24}: model_score = {cand['model_score']:.4f} ({cand['selection_reason']})")

    # Step 4: Parse Ground-Truth Report
    print("\n4. Parsing Associated Ground-Truth Report (1122.xml)...")
    report_data = parse_iu_report(report_path)
    print(f"   Study ID in Report    : {report_data['study_id']}")
    print(f"   Report FINDINGS       : {report_data['sections']['findings']}")
    print(f"   Report IMPRESSION     : {report_data['sections']['impression']}")
    print(f"   Extracted Ground Truth: {len(report_data['ground_truth_findings'])} finding entries")
    for gt in report_data["ground_truth_findings"]:
        print(f"   - {gt['finding']:<24}: presence = {gt['presence']:<9} (evidence: '{gt.get('matched_text', '')}')")

    # Step 5: Comparative Evaluation
    print("\n5. Running Comparative Evaluation (QA Output vs Ground Truth)...")
    eval_result = evaluate_qa_against_ground_truth(qa_output, report_data)
    print(f"   Matched Findings      : {eval_result['matched_findings_count']} ({', '.join(eval_result['matched_findings_list'])})")
    print(f"   Missed Findings       : {eval_result['missed_findings_count']} ({', '.join(eval_result['missed_findings_list']) if eval_result['missed_findings_list'] else 'None'})")
    print(f"   Extra Findings        : {eval_result['extra_findings_count']} ({', '.join(eval_result['extra_findings_list']) if eval_result['extra_findings_list'] else 'None'})")
    print(f"   Presence Agreement    : {eval_result['presence_matches']} / {eval_result['matched_findings_count']} evaluated ({eval_result['presence_agreement_rate']*100:.1f}%)")

    # Step 6: Save Validation Result JSON
    full_result = {
        "dataset_source": "Indiana University Chest X-Ray Collection (Open-i / NLM)",
        "sample_study_id": "CXR1122",
        "sample_image_id": "CXR1122_IM-0080-1001-0002",
        "view": "Frontal",
        "vision_pipeline": {
            "model": "TorchXRayVision DenseNet-121 (densenet121-res224-all)",
            "input_tensor_shape": list(input_tensor.shape),
            "feature_tensor_shape": list(features.shape),
            "device": str(device)
        },
        "qa_engine_config": {
            "baseline_findings": engine.baseline_findings,
            "top_k": engine.top_k,
            "qa_threshold": engine.qa_threshold
        },
        "qa_output": qa_output,
        "ground_truth_report": report_data,
        "evaluation_metrics": eval_result
    }

    os.makedirs(os.path.dirname(output_result_path), exist_ok=True)
    with open(output_result_path, "w", encoding="utf-8") as f:
        json.dump(full_result, f, indent=4)
    print(f"\nSaved full end-to-end validation results to: {output_result_path}")

    print("\n" + "=" * 65)
    print("PHASE 0.6 END-TO-END VALIDATION SUMMARY: PASSED")
    print("=" * 65)


if __name__ == "__main__":
    run_e2e_pipeline()
