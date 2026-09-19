# Phase 0.7 — Evidence Layer & LLM Input Preparation Specification

**Project**: LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning  
**Document Version**: 1.0  
**Phase**: 0.7 (Evidence Layer & LLM Input Preparation)  
**Status**: IMPLEMENTED & VERIFIED  

---

## 1. Objective & Design Philosophy

The Evidence Layer acts as a strict, intermediate semantic representation bridging upstream vision models/diagnostic QA with downstream natural language generation.

Its core purpose is to prevent a future LLM from confusing:
1. **Confirmed evidence** (explicitly established by diagnostic questions / human answers),
2. **Possible / model-suggested findings** (signals from vision activations), and
3. **Unknown / unobserved information** (missing or ambiguous details).

```text
========================================================================================
                          PHASE 0.7 END-TO-END PIPELINE ARCHITECTURE
========================================================================================

Chest X-ray Image (e.g., CXR1122_IM-0080-1001-0002.png)
      ↓
TorchXRayVision DenseNet-121 ('densenet121-res224-all')
      ├── 1024-D Visual Feature Vector (model.features2)
      └── 18 Pathology Activations (model_score ∈ [0.0, 1.0])
      ↓
Diagnostic QA Engine (backend/diagnostic_qa.py)
      ├── Baseline Routine Inclusions (Pneumothorax, Effusion, Consolidation, Cardiomegaly)
      ├── Top-K Activation Selection (TOP_K = 3)
      └── Threshold Trigger (QA_THRESHOLD = 0.35)
      ↓
Evidence Layer (backend/evidence_layer.py)
      ├── Status Resolution (supported, possible, uncertain, absent)
      ├── Provenance Tracking (sources: vision_model, diagnostic_qa, rule_derived)
      ├── Non-Hallucination Guarantees (location='unspecified', severity='unspecified')
      └── Anti-Leakage Safeguard (Ground Truth is NEVER passed to Production Input)
      ↓
Production LLM Input Package (docs/evidence_schema.json)
      ├── Structured Findings (status, model_score, location, severity)
      └── Mandatory Generation Constraints (do_not_invent_findings, preserve_uncertainty)
      ↓
[Future Phase] LLM Report Generator
========================================================================================
```

> [!IMPORTANT]
> **Research Prototype Disclaimer**:  
> *This is a research prototype for structured diagnostic-question selection and report generation. Model scores and generated outputs are not clinical diagnoses. The Evidence Layer preserves uncertainty and does not treat model scores as clinical diagnoses.*

---

## 2. Evidence Status Definitions

Every finding passed to the Evidence Layer is resolved into one of four mutually exclusive statuses:

| Status | Exact Definition | Clinical / Algorithmic Trigger |
| :--- | :--- | :--- |
| **`supported`** | There is explicit, sufficient evidence from the QA process or confirmed answers to state the finding. | Explicit QA answer `"yes"` (with supporting location/severity where observed). |
| **`possible`** | The vision model suggests the finding via activation score, but it has not been independently established. | Vision candidate finding in prototype mode with unconfirmed/unanswered question. |
| **`uncertain`** | The available evidence is ambiguous, contradictory, or cannot determine presence. | Explicit QA answer `"uncertain"` or unresolvable finding. |
| **`absent`** | The available evidence explicitly supports the absence or negation of the finding. | Explicit QA answer `"no"` (e.g., routine negative checks). |

> [!CAUTION]
> **Strict Invariant**: A high DenseNet `model_score` alone **never** converts a finding into `supported`. It remains `possible` until explicit supporting evidence is provided.

---

## 3. Evidence Sources & Provenance Tracking

The Evidence Layer tracks data provenance through an explicit list of sources (`evidence_sources`):
- **`vision_model`**: Signal derived from TorchXRayVision DenseNet-121 activation scores.
- **`diagnostic_qa`**: Signal derived from structured question evaluation / answered prompts.
- **`rule_derived`**: Inferred via deterministic safety rules.
- **`grounding_module`**: (Reserved for future visual heatmap/bounding box grounding).
- **`report_ground_truth`**: Derived from original radiologist XML reports (**strictly restricted to evaluation mode; never passed to production LLM packages**).

---

## 4. Model Score & Data Representation Rules

1. **`model_score` $\neq$ Clinical Diagnosis**: Scores represent activation magnitude $[0.0, 1.0]$ used for candidate selection, NOT disease probability or diagnosis.
2. **`model_score` $\neq$ Confidence**: The term *"confidence"* is completely eliminated from the pipeline to prevent clinical misinterpretation.
3. **No Fabricated Location**: Location defaults to `"unspecified"` unless explicitly observed or supplied.
4. **No Fabricated Severity**: Severity defaults to `"unspecified"` unless explicitly observed or supplied.
5. **Preservation of Uncertainty**: Ambiguous or unstated details remain explicitly unstated rather than guessed.

---

## 5. Ground-Truth Data Leakage Safeguards

To ensure rigorous benchmarking, the pipeline separates production and evaluation workflows:

```text
Production Flow (Zero Leakage):
Vision Backbone → Diagnostic QA Engine → Evidence Layer → LLM Input Package

Evaluation Flow (Benchmarking Only):
Evidence Layer Output ──┐
                        ├──> backend/evaluate_qa.py ──> Alignment Metrics
Report Ground Truth ────┘
```

**Leakage Prevention Implementation**:
1. `EvidenceLayer.build_llm_input()` strips all internal ground-truth annotations.
2. Runtime assertion checks in `EvidenceLayer` throw a `RuntimeError` if `"report_ground_truth"` is detected in the LLM input payload.
3. Automated validator `validate_evidence_package()` explicitly verifies zero ground-truth leakage.

---

## 6. Schema & LLM Input Package Structure

Formalized in [evidence_schema.json](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/docs/evidence_schema.json):

```json
{
  "task": "radiology_report_generation",
  "study": {
    "study_id": "CXR1122",
    "image_id": "CXR1122_IM-0080-1001-0002",
    "view": "Frontal"
  },
  "evidence": [
    {
      "finding": "Infiltration",
      "status": "possible",
      "model_score": 0.4750,
      "location": "unspecified",
      "severity": "unspecified"
    },
    {
      "finding": "Pneumothorax",
      "status": "possible",
      "model_score": 0.2193,
      "location": "unspecified",
      "severity": "unspecified"
    },
    {
      "finding": "Consolidation",
      "status": "possible",
      "model_score": 0.2052,
      "location": "unspecified",
      "severity": "unspecified"
    },
    {
      "finding": "Fracture",
      "status": "possible",
      "model_score": 0.0845,
      "location": "unspecified",
      "severity": "unspecified"
    },
    {
      "finding": "Nodule",
      "status": "possible",
      "model_score": 0.0708,
      "location": "unspecified",
      "severity": "unspecified"
    },
    {
      "finding": "Effusion",
      "status": "possible",
      "model_score": 0.0423,
      "location": "unspecified",
      "severity": "unspecified"
    },
    {
      "finding": "Cardiomegaly",
      "status": "possible",
      "model_score": 0.0015,
      "location": "unspecified",
      "severity": "unspecified"
    }
  ],
  "constraints": {
    "do_not_invent_findings": true,
    "do_not_invent_location": true,
    "do_not_invent_severity": true,
    "preserve_uncertainty": true
  }
}
```

---

## 7. Verification & Test Results

### Automated Unit Test Suite (`tests/test_phase_0_7_evidence.py`):
14 out of 14 unit tests passed:

- **TEST 1**: Vision candidate becomes `possible` rather than automatically `supported`. (PASSED)
- **TEST 2**: Explicit QA answer `yes` produces `supported` status. (PASSED)
- **TEST 3**: Explicit QA answer `no` produces `absent` status. (PASSED)
- **TEST 4**: QA answer `uncertain` remains `uncertain` / `possible`. (PASSED)
- **TEST 5**: Missing answer remains `uncertain` / `possible` without fabricating `yes`. (PASSED)
- **TEST 6**: Model score is preserved as numeric `model_score`. (PASSED)
- **TEST 7**: No `confidence` field is generated anywhere. (PASSED)
- **TEST 8**: Location remains `unspecified` when no evidence exists. (PASSED)
- **TEST 9**: Severity remains `unspecified` when no evidence exists. (PASSED)
- **TEST 10**: Multiple evidence sources are preserved and tracked. (PASSED)
- **TEST 11**: Ground truth does NOT appear in production LLM input package. (PASSED)
- **TEST 12**: Invalid model scores are rejected. (PASSED)
- **TEST 13**: Duplicate findings are cleanly deduplicated. (PASSED)
- **TEST 14**: Evidence package passes schema validation. (PASSED)

### End-to-End Pipeline Execution (`backend/run_e2e_evidence_pipeline.py`):
- **Input**: Real IU X-Ray radiograph `CXR1122_IM-0080-1001-0002.png`.
- **Vision Inference**: 1024-D visual features + 18 pathology activation scores extracted.
- **QA Engine**: 7 candidates evaluated across 14 multi-level questions.
- **Evidence Layer**: Successfully resolved 7 findings into strict `possible` statuses (since unconfirmed by human radiologist), preserved model scores, and constructed sanitized LLM input package with safety constraints.
- **Validation**: `validate_evidence_package()` returned 100% compliance with zero errors and zero ground-truth leakage.
- **Artifact**: Saved to `data/iu_xray/e2e_evidence_validation_result.json`.

---

## 8. Documented Limitations

1. **Prototype Interactive Gap**: In the current automated test harness, absence of interactive radiologist answers places model candidates in the `possible` state with `unspecified` locations/severities.
2. **Deterministic Constraint Reliance**: The future LLM must be strictly prompted to adhere to the package's `constraints` dictionary to prevent hallucinating beyond the Evidence Layer.
3. **Single View Resolution**: Frontal radiographs do not provide lateral depth (e.g., retrocardiac space); findings in these regions naturally remain `unspecified` for depth.

---

## 9. Future LLM Integration Plan (Phase 0.8)

When Phase 0.8 is authorized:
1. Design structured prompting templates ingesting `llm_input_package`.
2. Map `supported` findings into positive assertions, `absent` findings into pertinent negatives, and `possible`/`uncertain` into explicit non-definitive statements in `FINDINGS` and `IMPRESSION`.
3. Integrate automated hallucination detection checking generated reports against the Evidence Layer.
