"""
LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning
Phase 0.9 — Visual Grounding & Explainability via Grad-CAM

Module: visual_grounding.py
Purpose:
- Implements VisualGrounding (and VisualGroundingEngine) computing finding-specific Grad-CAM activation maps from DenseNet-121.
- Target Layer: model.features.norm5 (final convolutional feature representation yielding 1024 x 7 x 7 maps).
- Generates raw normalized heatmaps [0.0, 1.0], JET colormapped visualizations, and radiograph overlays.
- Preserves original radiograph and guarantees zero model weight mutation (parameter checksum verified).
- Produces structured grounding artifacts adhering to docs/grounding_schema.json.

DISCLAIMER:
Grad-CAM visualizations represent neural network feature activation patterns, NOT clinically validated
lesion localization or definitive diagnostic evidence.
"""

import os
import sys
import hashlib
import json
import time
from typing import Dict, List, Optional, Any, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import skimage.io
from PIL import Image
import torchxrayvision as xrv

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def compute_model_checksum(model: torch.nn.Module) -> str:
    """Compute SHA256 checksum across all model parameter tensors to verify zero weight mutation."""
    hasher = hashlib.sha256()
    for param in model.parameters():
        hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


def apply_jet_colormap(norm_map: np.ndarray) -> np.ndarray:
    """
    Apply standard JET colormap to a 2D float array in range [0.0, 1.0].
    Returns uint8 RGB array of shape (H, W, 3).
    """
    clamped = np.clip(norm_map, 0.0, 1.0)
    # 4 color bands: Blue -> Cyan -> Green -> Yellow -> Red
    r = np.clip(1.5 - np.abs(4.0 * clamped - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * clamped - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * clamped - 1.0), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


class VisualGrounding:
    """
    Grad-CAM Visual Grounding Engine for TorchXRayVision DenseNet-121.
    """

    TARGET_LAYER_NAME = "model.features.norm5"

    def __init__(self, model: Optional[torch.nn.Module] = None, device: Optional[torch.device] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        if model is None:
            self.model = xrv.models.DenseNet(weights="densenet121-res224-all")
        else:
            self.model = model

        self.model.eval()
        self.model.to(self.device)
        self.pathologies = list(self.model.pathologies)
        self.pathology_to_index = {p.lower(): idx for idx, p in enumerate(self.pathologies)}

    def get_pathology_index(self, finding_name: str) -> Optional[int]:
        """Resolves case-insensitive finding name to model output index."""
        cleaned = finding_name.strip().lower()
        if cleaned in self.pathology_to_index:
            return self.pathology_to_index[cleaned]
        # Match case-insensitively directly
        for idx, p in enumerate(self.pathologies):
            if p.lower() == cleaned:
                return idx
        return None

    def compute_gradcam(
        self,
        input_tensor: torch.Tensor,
        target_pathology: str
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Compute normalized 2D Grad-CAM activation map for a specific pathology.

        Args:
            input_tensor: PyTorch tensor with shape [1, 1, 224, 224].
            target_pathology: Name of the target pathology from model.pathologies.

        Returns:
            Tuple of:
            - raw_cam: 2D numpy array [7, 7] normalized to [0.0, 1.0]
            - model_score: Scalar model activation score for target pathology
            - activation_shape: Spatial shape of the convolutional feature map (7, 7)
        """
        target_idx = self.get_pathology_index(target_pathology)
        if target_idx is None:
            raise ValueError(
                f"Unknown pathology '{target_pathology}'. Supported pathologies: {self.pathologies}"
            )

        resolved_name = self.pathologies[target_idx]

        # Ensure tensor is on device
        x = input_tensor.to(self.device)

        # Verify parameter checksum before inference
        chk_before = compute_model_checksum(self.model)

        # Forward pass through features container up to norm5
        features = self.model.features(x)
        features.retain_grad()

        # Forward pass through post-feature ReLU, pooling, and classifier
        relu_features = F.relu(features)
        pooled = F.adaptive_avg_pool2d(relu_features, (1, 1)).view(x.size(0), -1)
        out = self.model.classifier(pooled)

        if hasattr(self.model, "op_threshs") and self.model.op_threshs is not None:
            out = torch.sigmoid(out)
            out = xrv.models.op_norm(out, self.model.op_threshs)
        elif hasattr(self.model, "apply_sigmoid") and self.model.apply_sigmoid:
            out = torch.sigmoid(out)

        model_score = float(out[0, target_idx].detach().cpu().item())

        # Backward pass on target pathology output
        self.model.zero_grad()
        if features.grad is not None:
            features.grad.zero_()

        out[0, target_idx].backward(retain_graph=False)

        # Extract gradients and compute channel weights via Global Average Pooling
        grads = features.grad
        if grads is None:
            raise RuntimeError(f"Failed to capture gradients for layer {self.TARGET_LAYER_NAME}")

        weights = grads.mean(dim=(2, 3), keepdim=True)

        # Weighted combination of forward activation maps followed by ReLU
        cam = F.relu((weights * features).sum(dim=1, keepdim=True))
        cam_np = cam[0, 0].detach().cpu().numpy()
        activation_shape = (cam_np.shape[0], cam_np.shape[1])

        # Min-Max Normalization to [0.0, 1.0]
        cam_min = float(cam_np.min())
        cam_max = float(cam_np.max())
        if cam_max - cam_min > 1e-8:
            norm_cam = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            norm_cam = np.zeros_like(cam_np)

        # Verify parameter integrity after backward pass
        chk_after = compute_model_checksum(self.model)
        if chk_before != chk_after:
            raise RuntimeError("[CRITICAL] Model weights mutated during Grad-CAM computation!")

        # Clean gradients
        self.model.zero_grad()

        return norm_cam, round(model_score, 4), activation_shape

    def generate_grounding_artifacts(
        self,
        study_id: str,
        image_id: str,
        view: str,
        original_image_path: str,
        input_tensor: torch.Tensor,
        target_findings: List[Dict[str, Any]],
        output_dir: str,
        alpha: float = 0.45,
        flat_naming: bool = False
    ) -> Dict[str, Any]:
        """
        Generate and save complete visual grounding package for a list of candidate findings.

        Args:
            study_id: Unique identifier for the study (e.g., 'CXR1122').
            image_id: Unique identifier for the radiograph.
            view: Radiograph projection ('Frontal', 'Lateral', 'Unspecified').
            original_image_path: Path to original radiograph image file.
            input_tensor: Preprocessed tensor [1, 1, 224, 224].
            target_findings: List of candidate finding dicts from Evidence Layer.
            output_dir: Base directory to save visualizations.
            alpha: Transparency factor for overlay blend (0.0 to 1.0).
            flat_naming: If True, uses CXR1122_<image_id>_<finding>_... naming in output_dir.

        Returns:
            Dictionary conforming to docs/grounding_schema.json.
        """
        if not os.path.exists(original_image_path):
            raise FileNotFoundError(f"Original image not found at: {original_image_path}")

        if flat_naming:
            study_out_dir = output_dir
        else:
            study_out_dir = os.path.join(output_dir, study_id)
        os.makedirs(study_out_dir, exist_ok=True)

        # Load and preserve original image
        raw_img = skimage.io.imread(original_image_path)
        if raw_img.ndim == 3:
            if raw_img.shape[2] >= 3:
                raw_gray = (raw_img[:, :, :3].mean(axis=2)).astype(np.uint8)
            else:
                raw_gray = raw_img[:, :, 0].astype(np.uint8)
        else:
            raw_gray = raw_img.astype(np.uint8)

        orig_h, orig_w = raw_gray.shape
        orig_rgb = np.stack([raw_gray, raw_gray, raw_gray], axis=-1)

        # Save preserved original copy
        clean_img_id = os.path.splitext(os.path.basename(image_id))[0]
        if flat_naming:
            preserved_orig_path = os.path.join(study_out_dir, f"{study_id}_{clean_img_id}_original.png")
        else:
            preserved_orig_path = os.path.join(study_out_dir, "original.png")
        Image.fromarray(orig_rgb).save(preserved_orig_path)

        grounding_records = []

        for item in target_findings:
            finding_name = item.get("finding", "")
            status = item.get("status", "possible")

            # Check if pathology exists in DenseNet
            target_idx = self.get_pathology_index(finding_name)
            if target_idx is None:
                continue
            canonical_name = self.pathologies[target_idx]

            # Compute Grad-CAM
            norm_cam, score, act_shape = self.compute_gradcam(input_tensor, canonical_name)

            # Resize raw CAM to original radiograph dimensions via bilinear PIL interpolation
            cam_pil = Image.fromarray((norm_cam * 255.0).astype(np.uint8))
            resized_cam_pil = cam_pil.resize((orig_w, orig_h), resample=Image.BILINEAR)
            resized_cam = np.array(resized_cam_pil).astype(np.float32) / 255.0

            # Generate JET Colormap Heatmap
            heatmap_rgb = apply_jet_colormap(resized_cam)

            # Generate Alpha-blended Overlay
            overlay_rgb = ((1.0 - alpha) * orig_rgb.astype(np.float32) + alpha * heatmap_rgb.astype(np.float32)).astype(np.uint8)

            # Sanitize finding name for filenames
            safe_name = canonical_name.lower().replace(" ", "_")
            if flat_naming:
                heatmap_rel_path = os.path.join(study_out_dir, f"{study_id}_{clean_img_id}_{safe_name}_heatmap.png")
                overlay_rel_path = os.path.join(study_out_dir, f"{study_id}_{clean_img_id}_{safe_name}_overlay.png")
            else:
                heatmap_rel_path = os.path.join(study_out_dir, f"{safe_name}_heatmap.png")
                overlay_rel_path = os.path.join(study_out_dir, f"{safe_name}_overlay.png")

            # Save visual artifacts
            Image.fromarray(heatmap_rgb).save(heatmap_rel_path)
            Image.fromarray(overlay_rgb).save(overlay_rel_path)

            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            record = {
                "finding": canonical_name,
                "model_score": score,
                "status": status,
                "method": "grad_cam",
                "explanation_type": "grad_cam",
                "target_layer": self.TARGET_LAYER_NAME,
                "activation_shape": list(act_shape),
                "heatmap_shape": [orig_h, orig_w],
                "normalization": "minmax_0_1",
                "heatmap_path": os.path.relpath(heatmap_rel_path, project_root).replace("\\", "/"),
                "overlay_path": os.path.relpath(overlay_rel_path, project_root).replace("\\", "/"),
                "notes": (
                    f"Grad-CAM visual activation map for {canonical_name}. "
                    "Represents neural network feature activations, not clinically verified lesion localization."
                )
            }
            grounding_records.append(record)

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        disclaimer_text = "Visual attribution map for research explainability; not a clinical localization or diagnosis."
        grounding_package = {
            "study_id": study_id,
            "image_id": image_id,
            "view": view,
            "disclaimer": disclaimer_text,
            "original_image_path": os.path.relpath(preserved_orig_path, project_root).replace("\\", "/"),
            "groundings": grounding_records,
            "grounding_records": grounding_records,
            "metadata": {
                "model": "TorchXRayVision DenseNet-121 (densenet121-res224-all)",
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "disclaimer": disclaimer_text
            }
        }

        # Save metadata.json in study folder
        metadata_json_path = os.path.join(study_out_dir, "grounding_metadata.json")
        with open(metadata_json_path, "w", encoding="utf-8") as f:
            json.dump(grounding_package, f, indent=4)

        return grounding_package


# Alias for backward compatibility
VisualGroundingEngine = VisualGrounding
