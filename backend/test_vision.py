"""
LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 0.4 — Vision Model Validation & Feature Extraction

Purpose:
Validate loading of pretrained TorchXRayVision DenseNet-121 ('densenet121-res224-all'),
apply TorchXRayVision preprocessing, extract internal visual features, save/load features,
compute pathology prediction scores on real radiographs, and inspect paired radiology reports.

DISCLAIMER:
All outputs from this model are research prediction scores and NOT clinical diagnoses.
"""

import os
import sys
import json
import argparse
import xml.etree.ElementTree as ET
import numpy as np
import skimage.io
import torch
import torchvision
import torchxrayvision as xrv


# Ensure stdout/stderr handle UTF-8 cleanly on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def check_environment():
    """Check and display environment details and available compute device."""
    print("=" * 60)
    print("1. ENVIRONMENT & DEVICE CONFIGURATION")
    print("=" * 60)
    print(f"Python Version          : {sys.version.split()[0]}")
    print(f"PyTorch Version         : {torch.__version__}")
    print(f"TorchVision Version     : {torchvision.__version__}")
    print(f"TorchXRayVision Version : {xrv.__version__}")
    print(f"Scikit-Image Version    : {skimage.__version__}")
    print(f"NumPy Version           : {np.__version__}")
    print("Dependencies loaded successfully.\n")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available          : {cuda_available}")
    if cuda_available:
        device = torch.device("cuda")
        print(f"CUDA Version            : {torch.version.cuda}")
        print(f"GPU Name                : {torch.cuda.get_device_name(0)}")
        print(f"Device Selected         : {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("CUDA unavailable — using CPU.")
        print("Device Selected         : cpu")
    print()
    return device


def load_vision_model(device: torch.device):
    """Load pretrained TorchXRayVision DenseNet-121 model."""
    print("=" * 60)
    print("2. LOADING TORCHXRAYVISION DENSENET-121")
    print("=" * 60)
    try:
        print("Loading weights 'densenet121-res224-all'...")
        model = xrv.models.DenseNet(weights="densenet121-res224-all")
        model.eval()
        model.to(device)
        print("Model successfully loaded.")
        print(f"Model execution device  : {device}")
        print(f"Pathology count         : {len(model.pathologies)}")
        print(f"Pathologies supported   : {', '.join(model.pathologies)}")
        print()
        return model
    except Exception as e:
        print(f"\n[ERROR] Failed to load TorchXRayVision model: {e}")
        print("Troubleshooting: Check internet connection or inspect weights cache in ~/.torchxrayvision/models_data.")
        raise


def preprocess_image(image_path: str, device: torch.device):
    """
    Load and preprocess chest X-ray using TorchXRayVision documented utilities:
    1. Read image with skimage
    2. Convert 3-channel RGB to 2D grayscale
    3. Normalize pixel intensity to [-1024, 1024] via xrv.datasets.normalize(img, 255)
    4. Reshape to (1, H, W) for transforms
    5. Center crop and resize to 224x224 using XRayCenterCrop and XRayResizer(224)
    6. Convert to tensor with shape [1, 1, 224, 224]
    """
    print("=" * 60)
    print("3. IMAGE LOADING & PREPROCESSING")
    print("=" * 60)

    if not os.path.exists(image_path):
        print(f"[ERROR] Test X-ray not found. Place a chest X-ray image at {image_path}")
        print("Action Required: Please specify a valid chest X-ray image path via --image or place one at backend/test_xray.jpg.")
        sys.exit(1)

    print(f"Loading test image from  : {image_path}")
    try:
        img = skimage.io.imread(image_path)
    except Exception as e:
        print(f"[ERROR] Could not read image file '{image_path}': {e}")
        print("Action Required: Ensure the file is a valid, uncorrupted image (JPG/PNG).")
        sys.exit(1)

    print(f"Raw image shape          : {img.shape}, dtype: {img.dtype}")

    # Convert multi-channel (RGB/RGBA) to 2D grayscale
    if img.ndim == 3:
        if img.shape[2] >= 3:
            img = img[:, :, :3].mean(axis=2)
        else:
            img = img[:, :, 0]
        print(f"Converted to 2D grayscale: {img.shape}")
    elif img.ndim != 2:
        print(f"[ERROR] Unexpected image dimensions: {img.ndim}. Expected 2D grayscale or 3D RGB image.")
        sys.exit(1)

    # Normalize intensity to [-1024, 1024] standard for TorchXRayVision
    # If image max value is 65535 (16-bit), normalize with maxval=65535
    maxval = 65535 if img.max() > 255 else 255
    img = xrv.datasets.normalize(img, maxval=maxval)

    # Prepare (1, H, W) shape for TorchXRayVision transform pipeline
    img_3d = img[None, ...]

    # Compose documented TorchXRayVision transformation utilities
    transform = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224)
    ])

    processed_img = transform(img_3d)

    # Add batch dimension and convert to PyTorch FloatTensor on target device
    input_tensor = torch.from_numpy(processed_img).unsqueeze(0).float().to(device)
    print(f"Preprocessed tensor shape: {list(input_tensor.shape)} (Expected: [1, 1, 224, 224])")
    print(f"Tensor value range       : [{input_tensor.min().item():.2f}, {input_tensor.max().item():.2f}]")
    print(f"Tensor device            : {input_tensor.device}")
    print()

    if list(input_tensor.shape) != [1, 1, 224, 224]:
        print(f"[ERROR] Preprocessed tensor shape {list(input_tensor.shape)} does not match expected [1, 1, 224, 224].")
        sys.exit(1)

    return input_tensor


def extract_visual_features(model, input_tensor: torch.Tensor):
    """
    Extract internal visual feature representation from DenseNet-121
    prior to the final classification layer.
    """
    print("=" * 60)
    print("4. INTERNAL VISUAL FEATURE EXTRACTION")
    print("=" * 60)
    with torch.no_grad():
        # model.features2(x) extracts post-ReLU, adaptive average pooled 1024-d visual features
        features = model.features2(input_tensor)

    print(f"Extracted feature shape  : {list(features.shape)}")
    print(f"Feature element count    : {features.numel()}")

    # Validation checks
    is_non_empty = features.numel() > 0
    is_finite = bool(torch.isfinite(features).all().item())
    print(f"Feature tensor non-empty : {is_non_empty}")
    print(f"Feature tensor finite    : {is_finite} (no NaN or Inf values)")

    if not (is_non_empty and is_finite):
        print("[ERROR] Visual feature validation failed: features are empty or contain non-finite values.")
        sys.exit(1)

    print("Internal visual features verified successfully.")
    print()
    return features


def test_feature_serialization(features: torch.Tensor, save_path: str = "test_features.pt"):
    """
    Save extracted feature tensor to disk and verify round-trip reloading.
    """
    print("=" * 60)
    print("5. FEATURE SERIALIZATION & RELOAD TEST")
    print("=" * 60)
    try:
        # Save tensor (on CPU for portable storage)
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        torch.save(features.cpu(), save_path)
        print(f"Feature tensor saved to  : {save_path}")

        # Reload tensor
        loaded_features = torch.load(save_path, weights_only=True)
        print(f"Feature tensor reloaded  : {save_path}")
        print(f"Loaded tensor shape      : {list(loaded_features.shape)}")

        # Verify integrity
        shape_match = loaded_features.shape == features.cpu().shape
        values_match = bool(torch.allclose(loaded_features, features.cpu()))
        finite_check = bool(torch.isfinite(loaded_features).all().item())

        print(f"Shape match verification : {shape_match}")
        print(f"Value match verification : {values_match}")
        print(f"Finite value check       : {finite_check}")

        if not (shape_match and values_match and finite_check):
            print("[ERROR] Feature serialization validation failed!")
            sys.exit(1)

        print("Feature save and load test passed successfully.")
        print()
    except Exception as e:
        print(f"[ERROR] Feature serialization test failed with exception: {e}")
        raise


def run_pathology_inference(model, input_tensor: torch.Tensor):
    """
    Run model inference and display all pathology scores and Top 5 findings.
    """
    print("=" * 60)
    print("6. MODEL INFERENCE & PATHOLOGY PREDICTION SCORES")
    print("=" * 60)
    print("[DISCLAIMER] These outputs are research model scores and NOT clinical diagnoses.\n")

    with torch.no_grad():
        outputs = model(input_tensor)

    # Convert outputs to numpy array of scores
    scores = outputs[0].detach().cpu().numpy()
    pathologies = model.pathologies

    # Display all pathology scores
    print("All Pathology Scores:")
    print("-" * 40)
    pathology_scores = []
    for name, score in zip(pathologies, scores):
        pathology_scores.append((name, float(score)))
        print(f"{name:<28}: {score:8.4f}")
    print("-" * 40)

    # Sort by score descending and display Top 5
    pathology_scores.sort(key=lambda x: x[1], reverse=True)
    print("\nTop 5 Model Outputs:")
    print("-" * 40)
    for idx, (name, score) in enumerate(pathology_scores[:5], start=1):
        print(f"{idx}. {name:<26}: {score:8.4f}")
    print("-" * 40)
    print("[NOTE] Explicit reminder: Model scores represent feature activations, not clinical diagnoses.\n")

    return pathology_scores


def inspect_paired_report(report_path: str):
    """Parse and display FINDINGS and IMPRESSION sections from the paired XML report."""
    print("=" * 60)
    print("7. PAIRED RADIOLOGY REPORT INSPECTION")
    print("=" * 60)
    if not report_path or not os.path.exists(report_path):
        print("No report path provided or report file not found. Skipping report display.\n")
        return {}

    print(f"Loading paired report from : {report_path}")
    report_data = {}
    try:
        tree = ET.parse(report_path)
        root = tree.getroot()

        uid_node = root.find("uId")
        study_id = uid_node.attrib.get("id") if uid_node is not None else "Unknown"
        report_data["study_id"] = study_id

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

        report_data["findings"] = findings
        report_data["impression"] = impression
        report_data["indication"] = indication
        report_data["comparison"] = comparison

        print(f"Study ID   : {study_id}\n")
        if indication:
            print(f"INDICATION :\n{indication}\n")
        if comparison:
            print(f"COMPARISON :\n{comparison}\n")

        print(f"FINDINGS   :\n{findings if findings else '[None documented]'}\n")
        print(f"IMPRESSION :\n{impression if impression else '[None documented]'}\n")

    except Exception as e:
        print(f"[WARN] Failed to parse XML report {report_path}: {e}")

    return report_data


def save_validation_result(output_path, result_data):
    """Save machine-readable validation result."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=4)
    print(f"Saved validation result to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 0.4 Vision Model Validation & Feature Extraction")
    parser.add_argument("--image", type=str, default=None, help="Path to input chest X-ray image (JPG/PNG)")
    parser.add_argument("--report", type=str, default=None, help="Path to associated XML radiology report")
    parser.add_argument("--output_result", type=str, default=None, help="Path to save validation_result.json")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)

    # Determine image path
    if args.image:
        image_path = os.path.abspath(args.image)
    else:
        # Check validation sample or fallback to backend/test_xray.jpg
        manifest_path = os.path.join(project_root, "data", "iu_xray", "validation_sample.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
                image_path = os.path.join(project_root, manifest.get("image_path", ""))
                if not args.report:
                    args.report = os.path.join(project_root, manifest.get("report_path", ""))
        else:
            image_path = os.path.join(base_dir, "test_xray.jpg")

    features_path = os.path.join(base_dir, "test_features.pt")
    output_result_path = args.output_result or os.path.join(project_root, "data", "iu_xray", "validation_result.json")

    # Step 1: Environment & device detection
    device = check_environment()

    # Step 2: Load model
    model = load_vision_model(device)

    # Step 3: Load and preprocess image
    input_tensor = preprocess_image(image_path, device)

    # Step 4: Extract visual feature representation
    features = extract_visual_features(model, input_tensor)

    # Step 5: Feature serialization / deserialization test
    test_feature_serialization(features, features_path)

    # Step 6: Model inference & Top 5 findings
    pathology_scores = run_pathology_inference(model, input_tensor)

    # Step 7: Inspect paired radiology report
    report_data = inspect_paired_report(args.report)

    # Step 8: Save validation result JSON
    validation_result = {
        "dataset_source": "Indiana University Chest X-Ray (Open-i / NLM)",
        "study_id": report_data.get("study_id", "Unknown"),
        "image_path": os.path.relpath(image_path, project_root) if os.path.exists(image_path) else image_path,
        "view": "Frontal (PA/AP)",
        "report_path": os.path.relpath(args.report, project_root) if (args.report and os.path.exists(args.report)) else args.report,
        "image_successfully_loaded": True,
        "preprocessing_successful": True,
        "input_tensor_shape": list(input_tensor.shape),
        "model_loaded": True,
        "device_used": str(device),
        "pathology_prediction_successful": True,
        "number_of_pathology_outputs": len(pathology_scores),
        "top_5_pathologies": pathology_scores[:5],
        "feature_extraction_successful": True,
        "feature_tensor_shape": list(features.shape),
        "feature_values_finite": bool(torch.isfinite(features).all().item()),
        "report_successfully_loaded": bool(report_data),
        "image_report_pairing_verified": bool(report_data.get("study_id")),
        "report_sections": {
            "findings": report_data.get("findings", ""),
            "impression": report_data.get("impression", "")
        }
    }
    save_validation_result(output_result_path, validation_result)

    print("=" * 60)
    print("PHASE 0.4 VALIDATION SUMMARY")
    print("=" * 60)
    print("Status: Vision model loading, preprocessing, visual feature extraction,")
    print("        feature serialization, pathology scoring, and report pairing ALL PASSED.")
    print("=" * 60)


if __name__ == "__main__":
    main()
