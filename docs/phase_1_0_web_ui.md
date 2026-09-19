# Phase 1.0 — Web-Based Clinical Review & Explainability UI

## 1. Objective & Semantic Refinements
Phase 1.0 establishes an interactive, research-oriented clinical review web dashboard for the **LLM-Assisted Explainable Radiology Report Generation** prototype. The UI communicates the exact research pipeline and reinforces key semantic safety invariants:

$$\text{Chest X-Ray} \longrightarrow \text{DenseNet-121} \longrightarrow \text{Diagnostic QA} \longrightarrow \text{Evidence Layer} \longrightarrow \text{Grad-CAM} \longrightarrow \text{LLM Report} \longrightarrow \text{Human Review}$$

> [!IMPORTANT]
> **Research Prototype & Safety Notice**: This interface is an investigative explainability research tool. It is **NOT** a certified medical device or clinical diagnostic tool.
> - **Model Activation Scores** are continuous vision activations from DenseNet-121, **not** clinical probabilities, confidence values, or diagnostic certainties.
> - **Grad-CAM Attributions** highlight internal neural-network feature activation maps, **not** lesion boundaries, segmentations, or definitive anatomical localizations.
> - **LLM Generated Reports** represent synthesized research output that **strictly requires human clinician review**.

---

## 2. Refined UI Architecture & Components

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ APP HEADER: Title | Pipeline Status (✓ Vision ✓ QA ✓ Evidence ✓ Grounding ✓ LLM ✓ Validation) | Research Badge │
├───────────────────────────────────┬───────────────────────────────────┬─────────────────────────────────┤
│ COLUMN 1: IMAGE & ATTRIBUTION     │ COLUMN 2: FINDINGS & EVIDENCE     │ COLUMN 3: GENERATED REPORT      │
│ ┌───────────────────────────────┐ │ ┌───────────────────────────────┐ │ ┌─────────────────────────────┐ │
│ │ ⚠️ ATTRIBUTION SAFETY NOTICE │ │ │ 7 Finding Selector Pills      │ │ │ ⚠️ REQUIRES HUMAN REVIEW     │ │
│ │ Mode: Overlay/Original/Heatmap│ │ │ Active Finding Metrics Card   │ │ │ Findings (7 items, [STATUS])│ │
│ │ Opacity Slider (0% to 100%)   │ │ │ - Model Activation Score:0.475│ │ │ Structured Impression       │ │
│ │ HTML5 Canvas Viewport         │ │ │ - Status Interpretation Text  │ │ │ LLM Metadata & Backbone Info│ │
│ │ Zoom & Pan Navigation         │ │ │ "Why This Finding?" Accordion │ │ │ Safety Validation Badges    │ │
│ │ Layer: model.features.norm5   │ │ │ Diagnostic QA Accordion       │ │ │ Export: JSON / Formatted Text│ │
│ └───────────────────────────────┘ │ │ 6-Step Traceability Pipeline  │ │ └─────────────────────────────┘ │
│                                   │ └───────────────────────────────┘ │                                 │
└───────────────────────────────────┴───────────────────────────────────┴─────────────────────────────────┘
```

---

## 3. Evidence Status Semantics
The dashboard strictly adheres to established Evidence Layer semantics:
- **`supported`**: *"Sufficient explicit evidence supports this finding."*
- **`possible`**: *"Model or indirect evidence suggests this finding may warrant review, but it is not independently established."*
- **`uncertain`**: *"Available evidence is equivocal or insufficient to establish presence or absence."*
- **`absent`**: *"Available evidence supports absence of this finding."*

---

## 4. 6-Stage Evidence Traceability Pipeline
For each selected candidate finding, the UI renders a 6-stage audit trail:
1. **① Vision Model (DenseNet-121)**: Raw activation score extraction (e.g. `0.4750`) and 1024-D feature vector.
2. **② Diagnostic QA Engine**: Structured question evaluation preventing unwarranted certainty (Presence answer: `uncertain`).
3. **③ Evidence Layer Resolution**: Formal status assignment (`POSSIBLE`), preserving `unspecified` location and severity.
4. **④ Visual Grounding (Grad-CAM)**: Attribution target `model.features.norm5` ($7 \times 7$ feature grid).
5. **⑤ LLM Report Generation**: Structured statement synthesis (`"Possible infiltration cannot be excluded; correlation is advised."`).
6. **⑥ Safety & Schema Validation**: Automated verification (0 violations, zero ground-truth leakage).

---

## 5. Backend REST API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/studies` | `GET` | List available imaging studies |
| `/api/studies/{study_id}` | `GET` | Aggregated study package (Evidence, QA, Grounding, Report, Validation) |
| `/api/studies/{study_id}/image` | `GET` | Serves original radiograph image (PNG stream) |
| `/api/studies/{study_id}/evidence`| `GET` | Sanitized Evidence Layer findings with raw float model scores |
| `/api/studies/{study_id}/qa` | `GET` | Diagnostic QA candidate findings & 14 evaluated questions |
| `/api/studies/{study_id}/grounding`| `GET` | Finding-specific Grad-CAM grounding records & artifact URLs |
| `/api/studies/{study_id}/finding/{finding}/grounding` | `GET` | Detailed grounding metadata for a specific finding |
| `/api/studies/{study_id}/report` | `GET` | Structured radiology report (Findings + Impression) |
| `/api/studies/{study_id}/validation` | `GET` | System safety metrics, pipeline stages, and violations count |
| `/api/studies/{study_id}/export?format=json\|text` | `GET` | Export report with embedded research disclaimers |
| `/api/artifacts/...` | `GET` | Serves visual attribution heatmaps and overlays |
| `/`, `/styles.css`, `/app.js` | `GET` | Serves frontend static application assets |

---

## 6. Ground-Truth Isolation Verification
The REST API and client-facing JavaScript ensure strict isolation:
- Zero reference XML report text, matched ground-truth annotations, or presence labels are ever sent to client endpoints.
- Normal production and review views reflect only model-derived activations and evidence states.

---

## 7. Automated Test Suite
- **API & UI Refinement Suite**: 22 / 22 tests passing (`tests/test_phase_1_0_api.py`).
- **Complete Project Test Suite**: 92 / 92 tests passing across Phases 0.6–1.0.
