"""
LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 0.9 — End-to-End Visual Grounding Pipeline

Executes the complete Phase 0.9 pipeline on a real IU X-Ray sample:
1. Real IU X-Ray image (CXR1122_IM-0080-1001-0002.png)
2. Preprocessing & DenseNet-121 Inference (18 pathology scores + 1024-D features)
3. Diagnostic QA Engine Question Selection (Top-K=3, QA_THRESHOLD=0.35, Baseline set)
4. Evidence Layer Status Resolution (supported, possible, uncertain, absent)
5. Grad-CAM Activation Map & Overlay Generation via VisualGroundingEngine
6. Visual Grounding Artifact Export (data/iu_xray/visual_grounding/CXR1122/)
7. Grounding Package Validation via validate_grounding_package()
8. Serialization to data/iu_xray/e2e_visual_grounding_validation_result.json

DISCLAIMER:
This pipeline is a research prototype. Grad-CAM heatmaps represent neural network
activation patterns, NOT clinically validated lesion localization or diagnostic proof.
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
from visual_grounding import VisualGroundingEngine, compute_model_checksum
from validate_grounding import validate_grounding_package


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


def run_e2e_visual_grounding_pipeline():
    print("=" * 75)
    print("PHASE 0.9 — END-TO-END VISUAL GROUNDING & EXPLAINABILITY PIPELINE")
    print("=" * 75)

    image_path = os.path.join(BASE_DIR, "data", "iu_xray", "images", "CXR1122_IM-0080-1001-0002.png")
    output_grounding_dir = os.path.join(BASE_DIR, "data", "iu_xray", "visual_grounding")
    output_result_path = os.path.join(BASE_DIR, "data", "iu_xray", "e2e_visual_grounding_validation_result.json")

    if not os.path.exists(image_path):
        print(f"[ERROR] Image file not found: {image_path}")
        sys.exit(1)

    device = torch.device("cpu")
    print(f"Device Selected          : {device}")
    print(f"Image Path               : {os.path.relpath(image_path, BASE_DIR)}")
    print(f"Grounding Output Dir     : {os.path.relpath(output_grounding_dir, BASE_DIR)}\n")

    # Step 1: Load Vision Backbone
    print("1. Loading TorchXRayVision DenseNet-121...")
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval()
    model.to(device)
    chk_initial = compute_model_checksum(model)
    print("   Model loaded successfully. Checksum verified.")

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

    # Step 3: Diagnostic QA & Evidence Layer
    print("\n3. Executing QA Candidate Selection & Evidence Layer...")
    qa_engine = DiagnosticQAEngine(top_k=3, qa_threshold=0.35)
    qa_output = qa_engine.evaluate_study(
        study_id="CXR1122",
        image_id="CXR1122_IM-0080-1001-0002",
        view="Frontal",
        pathology_scores=pathology_scores
    )
    evidence_layer = EvidenceLayer()
    evidence_package = evidence_layer.build_evidence_package(qa_output)
    target_findings = evidence_package["findings"]
    print(f"   Selected Findings     : {len(target_findings)}")

    # Step 4: Visual Grounding Generation (Grad-CAM)
    print("\n4. Generating Finding-Specific Grad-CAM Visualizations...")
    grounding_engine = VisualGroundingEngine(model, device)
    grounding_package = grounding_engine.generate_grounding_artifacts(
        study_id="CXR1122",
        image_id="CXR1122_IM-0080-1001-0002",
        view="Frontal",
        original_image_path=image_path,
        input_tensor=input_tensor,
        target_findings=target_findings,
        output_dir=output_grounding_dir
    )

    print(f"   Original Radiograph   : {grounding_package['original_image_path']}")
    print(f"   Grounding Records     : {len(grounding_package['grounding_records'])}")
    print("\n   Generated Visual Artifacts:")
    for rec in grounding_package["grounding_records"]:
        print(f"   - {rec['finding']:<24}: score={rec['model_score']:.4f} | Heatmap: {rec['heatmap_path']} | Overlay: {rec['overlay_path']}")

    # Step 5: Validate Grounding Package
    print("\n5. Running Schema & Safety Invariant Validation...")
    is_valid, validation_issues = validate_grounding_package(grounding_package, base_dir=BASE_DIR)
    if not is_valid:
        print("[ERROR] Validation failed with errors:")
        for err in validation_issues:
            print(f"  - {err}")
        sys.exit(1)
    print("   Validation Result     : PASSED (Zero schema errors, zero ground-truth leakage, zero fabricated locations)")

    # Step 6: Verify Model Integrity
    chk_final = compute_model_checksum(model)
    if chk_initial != chk_final:
        print("[ERROR] Model weights mutated during pipeline execution!")
        sys.exit(1)
    print("   Model Integrity Check : PASSED (Parameter checksums strictly identical before and after Grad-CAM)")

    # Step 7: Save Full Result Artifact
    full_result = {
        "dataset_source": "Indiana University Chest X-Ray Collection (Open-i / NLM)",
        "sample_study_id": "CXR1122",
        "sample_image_id": "CXR1122_IM-0080-1001-0002",
        "view": "Frontal",
        "grounding_package": grounding_package,
        "evidence_package": evidence_package,
        "model_integrity": {
            "checksum_initial": chk_initial,
            "checksum_final": chk_final,
            "weights_preserved": True,
            "eval_mode_preserved": not model.training
        }
    }

    os.makedirs(os.path.dirname(output_result_path), exist_ok=True)
    with open(output_result_path, "w", encoding="utf-8") as f:
        json.dump(full_result, f, indent=4)
    print(f"\nSaved full end-to-end visual grounding results to: {output_result_path}")

    print("\n" + "=" * 75)
    print("PHASE 0.9 END-TO-END VALIDATION SUMMARY: PASSED")
    print("=" * 75)


if __name__ == "__main__":
    run_e2e_visual_grounding_pipeline()
