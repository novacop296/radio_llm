# Phase 0.8 — LLM Integration & Structured Radiology Report Generation Specification

**Project**: LLM-Assisted Explainable Radiology Report Generation Using Diagnostic Questioning  
**Document Version**: 1.0  
**Phase**: 0.8 (LLM Integration & Report Generation)  
**Status**: IMPLEMENTED & VERIFIED  

---

## 1. Objective

Phase 0.8 integrates a structured Language Model (LLM) report generation subsystem into the research pipeline. The LLM consumes sanitized Evidence Layer input packages and translates intermediate findings into structured, clinically coherent `FINDINGS` and `IMPRESSION` sections conforming to `docs/report_schema.json`.

```text
========================================================================================
                          END-TO-END VALIDATED PIPELINE
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
      ├── Strict Status Resolution (supported, possible, uncertain, absent)
      ├── Non-Hallucination Guarantees (location='unspecified', severity='unspecified')
      └── Anti-Leakage Safeguard (Ground Truth strictly blocked)
      ↓
Sanitized LLM Input Package (docs/evidence_schema.json)
      ↓
LLM Report Generator (backend/report_generator.py + backend/llm/)
      ├── Deterministic Prompt Builder (12 Safety Constraints)
      ├── LLM Provider Abstraction (Mock / OpenAI)
      └── JSON Parser
      ↓
Report Safety & Schema Validator (backend/validate_report.py)
      ├── Traceability & Status Upgrade Checks
      ├── Measurement & Entity Anti-Hallucination Checks
      └── Impression Consistency Checks
      ↓
Validated Structured Report JSON (docs/report_schema.json)
========================================================================================
```

> [!IMPORTANT]
> **Research Prototype Disclaimer**:  
> *This is a research prototype for structured diagnostic-question selection and report generation. Model scores and generated outputs are not clinical diagnoses. The Evidence Layer and LLM generator preserve uncertainty and do not treat model scores as clinical diagnoses.*

---

## 2. Pluggable LLM Provider Architecture

A provider abstraction isolates the core report generation logic from any specific model vendor:

```text
               LLMProvider (Abstract Base Class)
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
   MockLLMProvider             OpenAIProvider
 (Deterministic Offline)     (Live API Endpoint)
```

- **`LLMConfig`**: Dataclass managing configuration loaded safely from environment variables (`LLM_PROVIDER`, `LLM_MODEL`, `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`, `LLM_TIMEOUT`, `LLM_RETRY_COUNT`, `LLM_API_KEY`).
- **`MockLLMProvider`**: Deterministic provider that processes the structured evidence and outputs compliant report JSON without external network access or API credentials.
- **`OpenAIProvider`**: Production-ready provider supporting OpenAI-compatible chat completion APIs with exponential retry, temperature=0, and JSON mode.

---

## 3. Prompt Design & Anti-Hallucination Constraints

The system prompt (`backend/llm/prompts.py`) enforces **12 deterministic safety rules**:

1. **Do Not Invent Findings**: Mention ONLY concepts provided in the input evidence.
2. **Preserve Status Distinctions**:
   - `supported` $\rightarrow$ Affirmative observed finding (*"Pleural effusion is present."*).
   - `possible` $\rightarrow$ Cautious non-assertive finding (*"Possible infiltration cannot be excluded; correlation advised."*). Never upgrade to `supported`.
   - `uncertain` $\rightarrow$ Equivocal / indeterminate finding.
   - `absent` $\rightarrow$ Explicit pertinent negative (*"No radiographic evidence of pneumothorax."*).
3. **Do Not Invent Location**: If location is `"unspecified"`, do not state side, lobe, or anatomical zone.
4. **Do Not Invent Severity**: If severity is `"unspecified"`, do not state degree/extent.
5. **Do Not Invent Measurements**: Strictly forbidden from outputting fabricated millimeter/centimeter numbers.
6. **Do Not Mention Model Scores**: Raw scores (e.g. 0.4750) are internal triggers and must not appear in narrative prose.
7. **No Clinical Diagnoses**: Observations remain radiological patterns, not definitive clinical etiologies.
8. **Impression Consistency**: The `IMPRESSION` section must ONLY summarize findings documented in `FINDINGS`.
9. **No External / Ground-Truth Knowledge**: Ground-truth reports are strictly excluded from the prompt.
10. **Output Format**: Enforces strict JSON matching `docs/report_schema.json`.

---

## 4. Output Report Schema

Formalized in [report_schema.json](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/docs/report_schema.json):

```json
{
  "study_id": "CXR1122",
  "image_id": "CXR1122_IM-0080-1001-0002",
  "view": "Frontal",
  "findings": [
    {
      "finding": "Infiltration",
      "statement": "Possible infiltration cannot be excluded; correlation is advised.",
      "status": "possible",
      "location": "unspecified",
      "severity": "unspecified"
    },
    {
      "finding": "Pneumothorax",
      "statement": "No radiographic evidence of pneumothorax.",
      "status": "absent",
      "location": "unspecified",
      "severity": "unspecified"
    }
  ],
  "impression": [
    "1. Possible infiltration; recommend clinical correlation.",
    "2. No acute pneumothorax."
  ],
  "metadata": {
    "provider": "mock",
    "model": "mock-radiology-llm",
    "generated_at": "2026-09-19T06:28:13Z"
  }
}
```

---

## 5. Report Safety & Schema Validation Rules

Implemented in `backend/validate_report.py`:

| Check | Rule Description | Enforcement |
| :--- | :--- | :---: |
| **Identifier Invariant** | `study_id`, `image_id`, `view` must strictly match the input package. | Fatal Error |
| **Evidence Traceability** | Every finding in the report must exist in the input evidence. | Fatal Error |
| **Status Upgrade Check** | `possible` or `uncertain` cannot be reported as `supported`. | Fatal Error |
| **Contradiction Check** | `absent` findings cannot be reported as `supported` or `possible`. | Fatal Error |
| **Location Integrity** | Unspecified location in evidence cannot be localized in report. | Fatal Error |
| **Severity Integrity** | Unspecified severity in evidence cannot be assigned degree in report. | Fatal Error |
| **Anti-Measurement Check** | Numerical size strings (e.g., `\d+ cm`) are rejected. | Fatal Error |
| **Impression Consistency** | Impression cannot introduce unmodeled findings. | Fatal Error |
| **Ground-Truth Leakage** | Detects and blocks any ground-truth report tokens. | Fatal Error |

---

## 6. Verification & Test Results

### Automated Unit Test Suite (`tests/test_phase_0_8_llm.py`):
All 22 unit tests passed:
- **TEST 1**: Mock provider works without API key. (PASSED)
- **TEST 2**: LLM input package is accepted. (PASSED)
- **TEST 3**: Required system constraints are present in prompt. (PASSED)
- **TEST 4**: Structured JSON output is parsed. (PASSED)
- **TEST 5**: Invalid JSON is rejected. (PASSED)
- **TEST 6**: Invalid schema is rejected. (PASSED)
- **TEST 7**: Missing required fields are rejected. (PASSED)
- **TEST 8**: Study ID cannot change. (PASSED)
- **TEST 9**: Image ID cannot change. (PASSED)
- **TEST 10**: Possible finding remains possible. (PASSED)
- **TEST 11**: Uncertain finding remains uncertain. (PASSED)
- **TEST 12**: Absent finding remains absent. (PASSED)
- **TEST 13**: LLM cannot introduce an unsupported finding. (PASSED)
- **TEST 14**: LLM cannot invent location. (PASSED)
- **TEST 15**: LLM cannot invent severity. (PASSED)
- **TEST 16**: LLM cannot introduce unsupported measurements. (PASSED)
- **TEST 17**: Ground truth is absent from LLM input. (PASSED)
- **TEST 18**: Model score is not interpreted as clinical probability. (PASSED)
- **TEST 19**: Impression cannot introduce unsupported findings. (PASSED)
- **TEST 20**: API errors are handled gracefully. (PASSED)
- **TEST 21**: Missing API key is handled gracefully. (PASSED)
- **TEST 22**: End-to-end Mock LLM pipeline succeeds. (PASSED)

**Total Project Tests Passing**: **49 / 49 (100%)** across Phases 0.6, 0.7, and 0.8.

### End-to-End Pipeline Execution (`backend/run_e2e_llm_pipeline.py`):
- **Sample**: Real radiograph `CXR1122_IM-0080-1001-0002.png` + reference report `1122.xml`.
- **Pipeline Execution**: DenseNet-121 $\rightarrow$ Diagnostic QA $\rightarrow$ Evidence Layer $\rightarrow$ LLM Report Generator $\rightarrow$ Validator.
- **Validation**: Schema and safety validator passed with 0 violations.
- **Artifact**: Saved to `data/iu_xray/e2e_llm_validation_result.json`.

---

## 7. Documented Research Limitations

1. **Non-Clinical Validation**: The generated text is a research demonstration of constrained language generation, not a clinically approved radiology report.
2. **Vision Model Uncertainty**: The vision backbone may exhibit false positives/negatives; the Evidence Layer prevents over-confidence by strictly bounding activations to `possible` unless confirmed.
3. **No Visual Grounding Yet**: Statements are currently linked to global image activations rather than localized heatmaps/bounding boxes (reserved for Phase 0.9).
