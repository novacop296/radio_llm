# Phase 0.6 — Diagnostic QA Engine Implementation Specification

**Project**: LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning  
**Document Version**: 1.0  
**Phase**: 0.6 (Implementation & Verification)  
**Status**: IMPLEMENTED & TESTED  

---

## 1. Architecture Overview

Phase 0.6 implements the deterministic, rule-grounded Diagnostic Questioning module. It forms the core intermediate reasoning layer between the continuous TorchXRayVision DenseNet-121 visual features/scores and downstream LLM report generation.

```text
========================================================================================
                          PHASE 0.6 PIPELINE ARCHITECTURE
========================================================================================

Chest X-ray Image (e.g., CXR1122_IM-0080-1001-0002.png)
      ↓
TorchXRayVision DenseNet-121 ('densenet121-res224-all')
      ├── 1024-D Visual Feature Vector (model.features2)
      └── 18 Pathology Activations (model_score ∈ [0.0, 1.0])
      ↓
Diagnostic QA Engine (backend/diagnostic_qa.py)
      ├── 1. Baseline Routine Inclusions (Pneumothorax, Effusion, Consolidation, Cardiomegaly)
      ├── 2. Top-K Selection (TOP_K = 3 highest non-baseline activations)
      └── 3. Threshold Trigger (QA_THRESHOLD = 0.35)
      ↓
Deterministic Question Generation
      ├── Level 1: Presence (yes / no / uncertain)
      ├── Level 2: Location (conditional on presence; defaults to "unspecified")
      └── Level 3: Severity (conditional on presence; defaults to "unspecified")
      ↓
Structured QA Findings JSON (docs/qa_schema.json)
      ↓
Comparative Evaluation (backend/evaluate_qa.py)
      ↑
Ground-Truth Extraction (backend/report_parser.py on IU X-Ray XML reports)
========================================================================================
```

> [!IMPORTANT]
> **Research Prototype Disclaimer**:
> This module is a research prototype for structured question selection and report representation. All model scores and generated finding records are pattern activations and representations, NOT clinical diagnoses or verified medical facts.

---

## 2. Implemented Modules

### A. `backend/diagnostic_qa.py`
The primary Diagnostic Questioning Engine class (`DiagnosticQAEngine`):
- **Candidate Selection**: Combines baseline routine checks with Top-K and threshold-based triggers. Deduplicates candidates and annotates selection reasons.
- **Question Generation**: Employs deterministic question templates for all 18 TorchXRayVision pathologies across 3 hierarchy levels (Presence $\rightarrow$ Location $\rightarrow$ Severity).
- **Study Evaluation**: Evaluates a study, incorporates explicit human/NLP answers when supplied, and ensures non-hallucinated `"unspecified"` defaults for location and severity.

### B. `backend/report_parser.py`
The XML report extraction module (`parse_iu_report`):
- Parses NLM `ecgen-radiology` XML schema.
- Extracts `study_id`, parent image identifiers, and sections (`FINDINGS`, `IMPRESSION`, `INDICATION`, `COMPARISON`).
- Converts unstructured report text into ground-truth finding records (`yes`, `no`, `uncertain`).
- Flags ambiguous findings (e.g., *"possible mass"*, *"cannot exclude"*) as `"uncertain"` rather than guessing.

### C. `backend/evaluate_qa.py`
The comparative evaluation module (`evaluate_qa_against_ground_truth`):
- Compares QA findings against report ground truth.
- Computes matched findings, missed findings, extra findings, and presence agreement rate.

### D. `backend/run_e2e_qa_pipeline.py`
Integrated runner connecting real image inference (`CXR1122`), QA candidate evaluation, report ground truth parsing (`1122.xml`), and JSON result export.

---

## 3. Question-Selection Logic & Configuration

The selection of questions to evaluate is controlled by three configurable parameters:

```python
# Prototype engineering defaults (NOT medically validated thresholds)
BASELINE_FINDINGS = ["Pneumothorax", "Effusion", "Consolidation", "Cardiomegaly"]
TOP_K = 3
QA_THRESHOLD = 0.35
```

### Selection Steps:
1. **Baseline Inclusions**: Always include `BASELINE_FINDINGS` to reflect clinical standard of care (verifying absence of acute emergency conditions).
2. **Top-K Non-Baseline Findings**: Sort all remaining findings by `model_score` descending and select the top $K$ (default: 3).
3. **Threshold Inclusions**: Include any additional non-baseline findings whose `model_score` $\ge 0.35$.
4. **Deduplication**: Produce an ordered list of 4 to 7 candidate finding objects.

---

## 4. Structured Output Format

Adheres strictly to [qa_schema.json](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/docs/qa_schema.json):

```json
{
  "study_id": "CXR1122",
  "image_id": "CXR1122_IM-0080-1001-0002",
  "view": "Frontal",
  "vision_candidates": [
    {
      "finding": "Pneumothorax",
      "model_score": 0.2193,
      "is_baseline": true,
      "selection_reason": "baseline_routine"
    },
    {
      "finding": "Infiltration",
      "model_score": 0.4750,
      "is_baseline": false,
      "selection_reason": "top_3_activation"
    }
  ],
  "questions_evaluated": [
    {
      "question_id": "Q_PNEUMOTHORAX_PRESENCE",
      "finding": "Pneumothorax",
      "level": 1,
      "question_text": "Is there radiographic evidence of a pneumothorax?",
      "options": ["yes", "no", "uncertain"],
      "model_score": 0.2193,
      "answer": "uncertain"
    }
  ],
  "qa_findings": [
    {
      "finding": "Pneumothorax",
      "presence": "uncertain",
      "location": "unspecified",
      "severity": "unspecified",
      "model_score": 0.2193,
      "evidence_source": "vision_model"
    }
  ]
}
```

---

## 5. Verification & Test Results

All 13 unit tests in [test_phase_0_6_qa.py](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/tests/test_phase_0_6_qa.py) passed:

| Test ID | Test Name | Status |
| :--- | :--- | :---: |
| TEST 1 | Baseline findings are always selected | **PASSED** |
| TEST 2 | Top-K non-baseline findings are selected correctly | **PASSED** |
| TEST 3 | Threshold filtering works | **PASSED** |
| TEST 4 | TOP_K and QA_THRESHOLD are configurable | **PASSED** |
| TEST 5 | No duplicate findings are generated | **PASSED** |
| TEST 6 | Conditional location questions created only when appropriate | **PASSED** |
| TEST 7 | Severity remains 'unspecified' when unsupported | **PASSED** |
| TEST 8 | Vision model score stored as 'model_score', never 'confidence' | **PASSED** |
| TEST 9 | XML report parsing works | **PASSED** |
| TEST 10 | Image/report study IDs remain correctly associated | **PASSED** |
| TEST 11 | Normal report converts into negative ground-truth findings | **PASSED** |
| TEST 12 | Single abnormal finding extracted with location/severity | **PASSED** |
| TEST 13 | Multiple abnormal findings extracted correctly from complex report | **PASSED** |

### End-to-End Pipeline Execution (`backend/run_e2e_qa_pipeline.py`):
- **Study ID**: `CXR1122`
- **Image ID**: `CXR1122_IM-0080-1001-0002` (Frontal view)
- **Input Tensor**: `[1, 1, 224, 224]`
- **Visual Features**: `[1, 1024]`
- **Candidate Findings Evaluated**: 7 findings (`Pneumothorax`, `Effusion`, `Consolidation`, `Cardiomegaly`, `Infiltration`, `Fracture`, `Nodule`).
- **Ground-Truth Findings Extracted**: 4 findings (`Pneumothorax: no`, `Effusion: no`, `Consolidation: no`, `Fracture: no`).
- **Evaluation Result**: Matched 4 findings; 0 missed ground-truth findings; 3 extra exploratory findings.
- **Machine-Readable Output**: Saved to `data/iu_xray/e2e_qa_validation_result.json`.

---

## 6. Documented Limitations

1. **Absence of Interactive Human Answers**: In prototype mode, without a human radiologist or conversational agent answering questions in real-time, questions without supplied answers default to `"uncertain"` with non-hallucinated `"unspecified"` locations and severities.
2. **Rule-Based NLP Approximation**: Report ground truth is extracted via clinical regular expressions. Complex sentence syntax or unmodeled entities are classified as `"uncertain"`.
3. **Threshold Sensitivity**: The parameters `TOP_K=3` and `QA_THRESHOLD=0.35` are prototype design choices and require validation against clinical utility in subsequent phases.
