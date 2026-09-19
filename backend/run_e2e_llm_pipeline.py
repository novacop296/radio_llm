"""
LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 0.8 — End-to-End LLM Report Generation Pipeline

Executes the complete Phase 0.8 pipeline on a real IU X-Ray sample:
1. Real IU X-Ray image (CXR1122_IM-0080-1001-0002.png)
2. Preprocessing & DenseNet-121 Inference (18 pathology scores + 1024-D features)
3. Diagnostic QA Engine Question Selection & Evaluation (Top-K=3, QA_THRESHOLD=0.35, Baseline set)
4. Evidence Layer Status Resolution (supported, possible, uncertain, absent)
5. Sanitized Production LLM Input Package Generation
6. LLM Report Generation (Mock or Configured Real Provider)
7. Report Safety & Schema Validation via validate_report.py
8. Evaluation against IU X-Ray Ground Truth (Reference Only)
9. Serialization to data/iu_xray/e2e_llm_validation_result.json

DISCLAIMER:
This pipeline is a research prototype. Generated reports, model scores, and findings are
pattern representations, NOT clinical diagnoses or verified medical facts.
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
from report_generator import RadiologyReportGenerator
from validate_report import validate_generated_report
from llm.factory import create_llm_provider
from llm.base import LLMConfig


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


def run_e2e_llm_pipeline():
    print("=" * 75)
    print("PHASE 0.8 — END-TO-END LLM RADIOLOGY REPORT GENERATION PIPELINE")
    print("=" * 75)

    image_path = os.path.join(BASE_DIR, "data", "iu_xray", "images", "CXR1122_IM-0080-1001-0002.png")
    report_path = os.path.join(BASE_DIR, "data", "iu_xray", "reports", "ecgen-radiology", "1122.xml")
    output_result_path = os.path.join(BASE_DIR, "data", "iu_xray", "e2e_llm_validation_result.json")

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
    llm_input_pkg = evidence_package["llm_input_package"]
    print(f"   Evidence Items        : {len(llm_input_pkg['evidence'])}")

    # Step 5: LLM Report Generation
    print("\n5. Invoking LLM Report Generator...")
    llm_provider = create_llm_provider()
    print(f"   Provider Selected     : {llm_provider.config.provider} (Model: {llm_provider.config.model})")
    generator = RadiologyReportGenerator(provider=llm_provider)
    generated_report, raw_llm_response, val_report = generator.generate_report(llm_input_pkg)

    print("\n   Generated Report Content:")
    print("   " + "-" * 60)
    print(f"   STUDY ID  : {generated_report['study_id']} ({generated_report['view']})")
    print("\n   FINDINGS  :")
    for f in generated_report["findings"]:
        print(f"   - [{f['status'].upper():<9}] {f['statement']}")
    print("\n   IMPRESSION:")
    for imp in generated_report["impression"]:
        print(f"   - {imp}")
    print("   " + "-" * 60)

    # Step 6: Report Validation
    print("\n6. Running Report Safety & Consistency Validation...")
    print(f"   Validation Status     : {'PASSED' if val_report['is_valid'] else 'FAILED'}")
    if not val_report["is_valid"]:
        for issue in val_report["issues"]:
            print(f"   - [VIOLATION] {issue}")
        sys.exit(1)

    # Step 7: Parse Ground-Truth Report (Reference & Benchmarking Only)
    print("\n7. Comparing with Reference Report Ground Truth (Evaluation Only)...")
    gt_report_data = parse_iu_report(report_path)
    gt_findings = {f["finding"]: f["presence"] for f in gt_report_data["ground_truth_findings"]}
    rep_findings = {f["finding"]: f["status"] for f in generated_report["findings"]}

    eval_breakdown = {
        "supported_generated": [f["finding"] for f in generated_report["findings"] if f["status"] == "supported"],
        "possible_generated": [f["finding"] for f in generated_report["findings"] if f["status"] == "possible"],
        "uncertain_generated": [f["finding"] for f in generated_report["findings"] if f["status"] == "uncertain"],
        "absent_generated": [f["finding"] for f in generated_report["findings"] if f["status"] == "absent"],
        "ground_truth_findings": gt_findings
    }

    print(f"   Ground Truth Summary  : {gt_findings}")
    print(f"   Generated Categories  : Supported={len(eval_breakdown['supported_generated'])}, Possible={len(eval_breakdown['possible_generated'])}, Absent={len(eval_breakdown['absent_generated'])}, Uncertain={len(eval_breakdown['uncertain_generated'])}")

    # Step 8: Save Final Artifact
    full_result = {
        "dataset_source": "Indiana University Chest X-Ray Collection (Open-i / NLM)",
        "sample_study_id": "CXR1122",
        "sample_image_id": "CXR1122_IM-0080-1001-0002",
        "view": "Frontal",
        "llm_input_package": llm_input_pkg,
        "generated_report": generated_report,
        "validation_report": val_report,
        "reference_ground_truth": gt_report_data,
        "evaluation_breakdown": eval_breakdown
    }

    os.makedirs(os.path.dirname(output_result_path), exist_ok=True)
    with open(output_result_path, "w", encoding="utf-8") as f:
        json.dump(full_result, f, indent=4)
    print(f"\nSaved full end-to-end LLM validation results to: {output_result_path}")

    print("\n" + "=" * 75)
    print("PHASE 0.8 END-TO-END VALIDATION SUMMARY: PASSED")
    print("=" * 75)


if __name__ == "__main__":
    run_e2e_llm_pipeline()
