# Phase 0.9 — Visual Grounding & Explainability via Grad-CAM

## 1. Objective
Phase 0.9 introduces finding-specific visual grounding to explain the internal feature activations of the DenseNet-121 model. For each pathology identified by the upstream Diagnostic QA and Evidence Layer, this module generates a Grad-CAM (Gradient-weighted Class Activation Mapping) heatmap and an alpha-blended radiograph overlay.

> [!IMPORTANT]
> **Clinical Grounding Disclaimer**: Grad-CAM visualizations represent neural network feature activation patterns and must **NOT** be interpreted as clinically validated disease localization, lesion contours, or definitive diagnostic proof. This system remains an experimental research prototype.

---

## 2. Grad-CAM Explanation
Grad-CAM computes the gradient of a target pathology score $y^c$ with respect to the feature activation maps $A^k$ of a chosen convolutional layer:
$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$
The 2D class activation map is obtained by taking the rectified linear combination of forward activation maps weighted by their global-average-pooled gradient importance weights:
$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$

---

## 3. Target Layer Selection
- **Selected Layer**: `model.features.norm5`
- **Architecture Rationale**:
  - In TorchXRayVision DenseNet-121 (`xrv.models.DenseNet`), `model.features` terminates with `denseblock4` and `norm5` (BatchNorm2d).
  - Shape before spatial pooling and classification: `[1, 1024, 7, 7]`.
  - `norm5` preserves the final spatial feature grid ($7 \times 7$) before global average pooling collapses spatial dimensions into 1024-D vectors.

---

## 4. Finding-to-Index Mapping
Pathology scores are mapped directly to TorchXRayVision's 18 canonical output classes:
`Atelectasis (0)`, `Consolidation (1)`, `Infiltration (2)`, `Pneumothorax (3)`, `Edema (4)`, `Emphysema (5)`, `Fibrosis (6)`, `Effusion (7)`, `Pneumonia (8)`, `Pleural_Thickening (9)`, `Cardiomegaly (10)`, `Nodule (11)`, `Mass (12)`, `Hernia (13)`, `Lung Lesion (14)`, `Fracture (15)`, `Lung Opacity (16)`, `Enlarged Cardiomediastinum (17)`.

---

## 5. Normalization & Visualization Process
1. **Min-Max Normalization**: Raw Grad-CAM activations are normalized to $[0.0, 1.0]$.
2. **Bilinear Upsampling**: Scaled from $7 \times 7$ feature resolution to original radiograph dimensions (e.g., $512 \times 624$).
3. **Colormapping**: Mapped to standard JET colormap RGB without requiring external plotting libraries.
4. **Alpha-Blended Overlay**:
   $$I_{\text{overlay}} = 0.45 \times I_{\text{heatmap\_RGB}} + 0.55 \times I_{\text{original\_radiograph\_RGB}}$$
5. **Deterministic Artifact Preservation**:
   - `data/iu_xray/grounding/CXR1122_<image_id>_<finding>_original.png`
   - `data/iu_xray/grounding/CXR1122_<image_id>_<finding>_heatmap.png`
   - `data/iu_xray/grounding/CXR1122_<image_id>_<finding>_overlay.png`

---

## 6. Architecture & Ground-Truth Separation
Visual grounding operates downstream of the Evidence Layer and runs in parallel to report generation:
```
           Chest X-Ray
               ↓
          DenseNet-121
         ↙            ↘
  Pathology Scores    1024-D Features
         ↓
   Diagnostic QA
         ↓
   Evidence Layer
    ↙          ↘
LLM Report   Grad-CAM Grounding
```
- **No Feedback Loop**: Visual heatmaps do not alter the Evidence Layer findings or change the generated report.
- **Ground-Truth Separation**: Reference report ground truth is strictly excluded from target selection, backpropagation, and metadata generation.
- **No Hallucinated Anatomy**: Heatmap peaks are never converted into anatomical lobe coordinate claims.

---

## 7. Model Weight & Integrity Verification
- Verified by computing SHA-256 parameter checksums over all model parameter weights before and after Grad-CAM backward execution.
- Parameters remain 100% immutable (`checksum_initial == checksum_final`).
- Model remains in `.eval()` mode with zero parameter drift.

---

## 8. Limitations & Future Directions
1. **Coarse Spatial Resolution**: The $7 \times 7$ bottleneck in DenseNet-121 yields diffuse, smooth heatmaps upon interpolation.
2. **Correlation Overlap**: Multi-label pathologies in similar lung regions may display overlapping activation patterns.
3. **Phase 1.0 Recommendations**: Interactive UI visualization with multi-finding toggles and side-by-side comparison of frontal and lateral projections.
