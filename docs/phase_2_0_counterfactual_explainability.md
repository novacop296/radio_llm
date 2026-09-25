# Phase 2.0 — Interactive Counterfactual Explanations & Clinical Reasoning Synthesis

## 1. Executive Summary & Research Scope

Phase 2.0 introduces an interactive counterfactual reasoning and explainability platform built upon the Explainable Radiology Research Prototype (Phases 0.6–1.9).

> [!IMPORTANT]
> **MANDATORY RESEARCH DISCLAIMER**
> This system is an experimental **research prototype** and **NOT a certified medical device, clinical decision support system, or clinical diagnostic tool**.
>
> Counterfactual perturbations are controlled computational interventions designed to inspect model sensitivity.
> Changes in model activation, Grad-CAM attribution, Diagnostic QA responses, or generated report statements do not establish clinical causality, lesion boundaries, or diagnostic truth.
>
> - **Model Activation Score ≠ Clinical Probability**
> - **Grad-CAM Attribution ≠ Lesion Localization**
> - **Counterfactual Response Change ≠ Clinical Causality**

---

## 2. Architectural Design & Layer Isolation

To preserve strict architectural invariants, all counterfactual operations, configurations, derived image artifacts, and execution manifests are strictly isolated in an independent storage layer:

```
data/counterfactuals/
├── {study_id}/
│   ├── {counterfactual_id}.json       # Full structured experiment record & lifecycle
│   ├── artifacts/
│   │   ├── {cf_id}_baseline_gradcam.png
│   │   ├── {cf_id}_counterfactual_image.png
│   │   ├── {cf_id}_counterfactual_gradcam.png
│   │   └── {cf_id}_attribution_diff.png
│   └── manifests/
│       └── {cf_id}_reproducibility.json
```

### Invariants:
1. **Original Radiograph Immutability:** Underlying 5,399 raw files in `data/iu_xray/` remain 100% byte-identical (`SHA256_BEFORE == SHA256_AFTER`).
2. **Ground-Truth XML Report Isolation:** Reference clinical XML tags (`<eFind>`, `<eImpression>`, `<abstractModel>`) are never accessed or leaked into counterfactual records.
3. **Published Record Immutability:** Experiments transitioning to `PUBLISHED` state become permanently locked against modification.

---

## 3. Counterfactual Perturbation Engine

The perturbation engine (`backend/counterfactual_manager.py`) supports 8 deterministic, bounded operations:

| Perturbation Method | Description | Mathematical / Processing Operation |
| :--- | :--- | :--- |
| `REGION_MASK` | Neutral solid mask | $\text{Patch} \leftarrow \text{floor}(128 \times (1 - \min(s, 1.0)))$ |
| `REGION_OCCLUSION` | Contextual mean occlusion | $\text{Patch} \leftarrow \text{Blend}(\text{Patch}, \mu_{\text{patch}}, s)$ |
| `REGION_BLUR` | Gaussian blur | Gaussian filter with radius $r = \max(1.0, 8.0 \times s)$ |
| `REGION_NOISE` | Deterministic Gaussian noise | $\text{Patch} \leftarrow \text{Clip}(\text{Patch} + \mathcal{N}(0, 50 \times s), 0, 255)$ with seed |
| `REGION_CONTRAST` | Local contrast modulation | $\text{Patch} \leftarrow \text{Clip}((\text{Patch} - \mu) \times s + \mu, 0, 255)$ |
| `REGION_BRIGHTNESS` | Local brightness shift | $\text{Patch} \leftarrow \text{Clip}(\text{Patch} + (s - 1.0) \times 80, 0, 255)$ |
| `REMOVE_ATTRIBUTION_REGION` | Occlusion of top attribution | Replaces maximal Grad-CAM hotspot with median context |
| `RANDOM_CONTROL_REGION` | Non-overlapping control ROI | Applies identical perturbation to randomized control ROI |

---

## 4. Analytical Deltas & Impact Propagation

```
Original Image
      ↓
Baseline Inference & Grad-CAM
      ↓
Select ROI & Perturbation Method
      ↓
Generate Derived Counterfactual Image
      ↓
Counterfactual Inference & Grad-CAM
      ↓
Finding Activation Deltas (Absolute & Relative)
      ↓
Attribution Difference Heatmap & Energy Shift
      ↓
Diagnostic QA Impact (Level 1 Presence, Level 2 Severity)
      ↓
Machine Report Impact (Status Transitions & Statement Diffs)
      ↓
Reproducibility Verification (Deterministic Fingerprinting)
```

### Finding Activation Delta
- Absolute difference: $\Delta_{\text{abs}} = S_{\text{counterfactual}} - S_{\text{baseline}}$
- Safe relative difference: $\Delta_{\text{rel}} = \frac{\Delta_{\text{abs}}}{\max(S_{\text{baseline}}, 10^{-6})}$
- Direction: `INCREASED` if $\Delta_{\text{abs}} > 0.005$, `DECREASED` if $\Delta_{\text{abs}} < -0.005$, else `UNCHANGED`.

### Grad-CAM Attribution Metrics
- Mean Absolute Pixel Difference: $\frac{1}{HW} \sum_{x,y} |H_{\text{cf}}(x,y) - H_{\text{base}}(x,y)|$
- Attribution Overlap (Cosine Similarity): $\frac{\mathbf{H}_{\text{base}} \cdot \mathbf{H}_{\text{cf}}}{\|\mathbf{H}_{\text{base}}\| \|\mathbf{H}_{\text{cf}}\|}$
- ROI Attribution Fraction: $\frac{\sum_{\text{ROI}} H(x,y)}{\sum_{\text{Image}} H(x,y)}$

---

## 5. Experiment Lifecycle States

```
[DRAFT] ──> [CONFIGURED] ──> [RUNNING] ──> [COMPLETED] ──> [VALIDATED] ──> [PUBLISHED (Locked)]
                                                                    └──> [ARCHIVED]
```

---

## 6. REST API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/phase20/dashboard` | Aggregated dashboard summary statistics |
| `GET` | `/api/phase20/counterfactuals` | List all counterfactual experiments (optional `study_id`) |
| `POST` | `/api/phase20/counterfactuals` | Create new counterfactual draft |
| `GET` | `/api/phase20/counterfactuals/{id}` | Get full counterfactual experiment record |
| `POST` | `/api/phase20/counterfactuals/{id}/configure` | Configure perturbation method, ROI, strength, and seed |
| `POST` | `/api/phase20/counterfactuals/{id}/run` | Execute baseline vs counterfactual inference pipeline |
| `POST` | `/api/phase20/counterfactuals/{id}/validate` | Validate schema conformance and numeric invariants |
| `POST` | `/api/phase20/counterfactuals/{id}/finalize` | Finalize experiment for publication |
| `POST` | `/api/phase20/counterfactuals/{id}/publish` | Publish experiment to registry (immutable lock) |
| `POST` | `/api/phase20/counterfactuals/{id}/control` | Execute automated non-overlapping control ROI comparison |
| `POST` | `/api/phase20/counterfactuals/compare` | Compare two counterfactual experiments |
| `GET` | `/api/phase20/counterfactuals/{id}/baseline` | Retrieve baseline inference scores |
| `GET` | `/api/phase20/counterfactuals/{id}/counterfactual` | Retrieve counterfactual inference scores |
| `GET` | `/api/phase20/counterfactuals/{id}/comparison` | Retrieve finding activation deltas |
| `GET` | `/api/phase20/counterfactuals/{id}/attribution` | Retrieve Grad-CAM attribution metrics |
| `GET` | `/api/phase20/counterfactuals/{id}/qa-impact` | Retrieve diagnostic QA impact items |
| `GET` | `/api/phase20/counterfactuals/{id}/report-impact` | Retrieve machine report statement impact |
| `GET` | `/api/phase20/counterfactuals/{id}/reproducibility` | Retrieve reproducibility manifest |
| `GET` | `/api/phase20/counterfactuals/{id}/export` | Export structured experiment JSON |
| `GET` | `/api/phase20/safety-validation` | Run safety and invariant audit suite |

---

## 7. Verification & Test Coverage

- **Unit & Integration Suite (`tests/test_phase_2_0_counterfactual.py`):** 35 dedicated test cases covering schema validation, all 8 perturbation methods, lifecycle transitions, Grad-CAM diffing, QA/Report impact, immutability, zero-denominator safety, and REST API routes.
- **Full Project Regression Suite:** 407 tests passing (100% OK).
- **Phase 2.0 E2E Verification (`backend/run_e2e_phase_2_0.py`):** 20/20 stages passing (100% OK).
- **Phase 1.9 E2E Verification (`backend/run_e2e_phase_1_9.py`):** 21/21 stages passing (100% OK).
- **Phase 1.8 E2E Verification (`backend/run_e2e_phase_1_8.py`):** 28/28 stages passing (100% OK).
