"""
LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 0.9 — Visual Grounding Pipeline Execution Script

Module: run_grounding.py
Purpose:
- Loads CXR1122 and preprocessed [1, 1, 224, 224] tensor.
- Loads DenseNet-121 model.
- Obtains selected candidate findings from Diagnostic QA / Evidence Layer.
- Computes Grad-CAM activation maps, normalized heatmaps, and overlays.
- Saves visualizations into data/iu_xray/grounding/.
- Saves validation result artifact to data/iu_xray/e2e_grounding_validation_result.json.
- Validates the grounding package against grounding_schema.json.
- Prints a concise execution summary.
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
from visual_grounding import VisualGrounding, compute_model_checksum
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


def main():
    print("=" * 75)
    print("PHASE 0.9 — VISUAL GROUNDING & EXPLAINABILITY (run_grounding.py)")
    print("=" * 75)

    study_id = "CXR1122"
    image_filename = "CXR1122_IM-0080-1001-0002.png"
    image_id = "CXR1122_IM-0080-1001-0002"
    view = "Frontal"

    image_path = os.path.join(BASE_DIR, "data", "iu_xray", "images", image_filename)
    output_grounding_dir = os.path.join(BASE_DIR, "data", "iu_xray", "grounding")
    output_result_path = os.path.join(BASE_DIR, "data", "iu_xray", "e2e_grounding_validation_result.json")

    if not os.path.exists(image_path):
        print(f"[ERROR] Image file not found: {image_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device Selected          : {device}")
    print(f"Study ID                 : {study_id}")
    print(f"Image File               : {image_filename}")
    print(f"Output Directory         : {os.path.relpath(output_grounding_dir, BASE_DIR)}\n")

    # 1. Load Model
    print("1. Loading TorchXRayVision DenseNet-121 model...")
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval()
    model.to(device)
    chk_initial = compute_model_checksum(model)
    print("   [OK] Pretrained model loaded.")

    # 2. Preprocessing & Vision Scoring
    print("\n2. Preprocessing radiograph tensor [1, 1, 224, 224]...")
    input_tensor = preprocess_image_tensor(image_path, device)
    with torch.no_grad():
        features = model.features2(input_tensor)
        outputs = model(input_tensor)

    scores = outputs[0].detach().cpu().numpy()
    pathology_scores = {name: float(score) for name, score in zip(model.pathologies, scores)}
    print(f"   [OK] Preprocessed shape: {list(input_tensor.shape)}")
    print(f"   [OK] Feature vector shape: {list(features.shape)}")

    # 3. Diagnostic QA & Evidence Layer
    print("\n3. Obtaining candidate findings from Diagnostic QA & Evidence Layer...")
    qa_engine = DiagnosticQAEngine(top_k=3, qa_threshold=0.35)
    qa_output = qa_engine.evaluate_study(
        study_id=study_id,
        image_id=image_id,
        view=view,
        pathology_scores=pathology_scores
    )
    evidence_layer = EvidenceLayer()
    evidence_package = evidence_layer.build_evidence_package(qa_output)
    target_findings = evidence_package["findings"]
    print(f"   [OK] Candidate findings ({len(target_findings)}): {[f['finding'] for f in target_findings]}")

    # 4. Generate Visual Grounding Maps
    print("\n4. Generating Grad-CAM heatmaps and overlays...")
    grounding_engine = VisualGrounding(model=model, device=device)
    grounding_package = grounding_engine.generate_grounding_artifacts(
        study_id=study_id,
        image_id=image_id,
        view=view,
        original_image_path=image_path,
        input_tensor=input_tensor,
        target_findings=target_findings,
        output_dir=output_grounding_dir,
        flat_naming=True
    )

    print(f"   [OK] Preserved original: {grounding_package['original_image_path']}")
    for g in grounding_package["groundings"]:
        print(f"   - {g['finding']:<20}: score={g['model_score']:.4f} | heatmap={g['heatmap_path']} | overlay={g['overlay_path']}")

    # 5. Validate Grounding Package
    print("\n5. Validating visual grounding package against schema and invariants...")
    is_valid, validation_issues = validate_grounding_package(grounding_package, base_dir=BASE_DIR)
    if not is_valid:
        print("[ERROR] Validation failed:")
        for issue in validation_issues:
            print(f"  - {issue}")
        sys.exit(1)
    print("   [OK] Validation PASSED (0 errors, zero ground-truth leakage, zero anatomical hallucination).")

    # 6. Verify Model Weight Integrity
    chk_final = compute_model_checksum(model)
    if chk_initial != chk_final:
        print("[ERROR] Model weights were mutated during Grad-CAM backpropagation!")
        sys.exit(1)
    print("   [OK] Model weight integrity verified (SHA-256 parameter checksums match).")

    # 7. Save e2e JSON result
    result_data = {
        "status": "PASSED",
        "study_id": study_id,
        "image_id": image_id,
        "view": view,
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
        json.dump(result_data, f, indent=4)
    print(f"\nSaved grounding validation result to: {output_result_path}")

    print("\n" + "=" * 75)
    print("PHASE 0.9 GROUNDING PIPELINE COMPLETED SUCCESSFULLY.")
    print("=" * 75)


if __name__ == "__main__":
    main()
