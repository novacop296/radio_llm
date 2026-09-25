# Phase 1.2 — Human-in-the-Loop Review, Report Correction & Finalization

> **Research Prototype Notice**: Reviewer annotations, decisions, and finalized reports are research metadata only. They are NOT certified clinical ground truth or automated clinical diagnoses. All machine evidence, vision backbone activations, Diagnostic QA answers, Grad-CAM attributions, and machine reports remain strictly immutable.

---

## 1. Overview & Architecture

Phase 1.2 introduces a **Human-in-the-Loop Review & Finalization Layer** on top of the validated 6-stage explainable radiology report generation pipeline (Vision Backbone $\rightarrow$ Diagnostic QA Engine $\rightarrow$ Evidence Layer $\rightarrow$ Grad-CAM Grounding $\rightarrow$ LLM Generator $\rightarrow$ Safety Validator).

```
IU Chest X-Ray (CXR1122)
          ↓
TorchXRayVision DenseNet-121 (Immutable)
          ↓
Diagnostic QA Engine (14 Evaluated Questions) (Immutable)
          ↓
Evidence Layer (7 Candidate Findings) (Immutable)
          ↓
Grad-CAM Visual Grounding (norm5 Activation Maps) (Immutable)
          ↓
Machine-Generated Report (Findings + Impression) (Immutable)
          ↓
========================================================================
            PHASE 1.2 HUMAN-IN-THE-LOOP REVIEW LAYER
========================================================================
Review Session Init (`data/reviews/{study_id}.json`)
          ├── Finding Decisions: confirmed_present / confirmed_absent / uncertain / needs_review
          ├── QA Corrections: Reviewer Answer vs Immutable Machine Answer
          ├── Structured Report Draft: Editable findings, statements, locations, impression
          ├── Append-Only Audit Trail: Chronological change logging with old/new values
          ├── Finalization Validation: Completeness & integrity verification
          └── Finalized Review Package: Immutable locked review & JSON/Text export
```

---

## 2. Review Data Model & Schema

All reviewer modifications exist exclusively in an isolated review storage layer: `data/reviews/{study_id}.json`.

Formal JSON Schema definition is maintained in [`docs/review_schema.json`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/docs/review_schema.json).

### Root Schema (`ReviewSession`):
```json
{
  "study_id": "CXR1122",
  "review_id": "rev_CXR1122_1790031717",
  "reviewer": {
    "id": "dr_lead_reviewer",
    "display_name": "Dr. Lead Reviewer, MD",
    "role": "Research Reviewer"
  },
  "status": "in_review",
  "started_at": "2026-09-21T23:01:57Z",
  "updated_at": "2026-09-21T23:01:57Z",
  "completed_at": null,
  "finding_reviews": [
    {
      "finding": "Infiltration",
      "machine_status": "possible",
      "reviewer_status": "confirmed_present",
      "machine_location": "unspecified",
      "reviewer_location": "right_lower_lobe",
      "machine_severity": "unspecified",
      "reviewer_severity": "moderate",
      "reviewer_comment": "Faint patchy airspace opacity verified.",
      "reviewed": true
    }
  ],
  "qa_reviews": [
    {
      "question_id": "Q_INFILTRATION_PRESENCE",
      "finding": "Infiltration",
      "level": 1,
      "machine_answer": "uncertain",
      "reviewer_answer": "yes",
      "reviewer_comment": "Reviewer verified right basilar infiltrate presence manually."
    }
  ],
  "report_review": {
    "machine_report_locked": true,
    "final_findings": [],
    "final_impression": [],
    "reviewer_comment": "",
    "finalized": false
  },
  "audit_trail": [
    {
      "timestamp": "2026-09-21T23:01:57Z",
      "reviewer_id": "dr_lead_reviewer",
      "finding": "Infiltration",
      "field": "reviewer_status",
      "old_value": "not_reviewed",
      "new_value": "confirmed_present"
    }
  ]
}
```

### Permitted Status Values:
- **Session Status**: `not_reviewed`, `in_review`, `reviewed`, `finalized`
- **Reviewer Finding Status**: `not_reviewed`, `confirmed_present`, `confirmed_absent`, `uncertain`, `needs_review`

---

## 3. Strict Immutability Invariant

The review layer is completely decoupled from the machine evidence layer:
1. When a reviewer confirms a finding as `confirmed_present`, `machine_status` remains `possible`.
2. Machine location and machine severity are never overwritten.
3. Original Diagnostic QA artifacts in `data/iu_xray/e2e_qa_validation_result.json` remain byte-identical.
4. Model weights, activations, and Grad-CAM PNG heatmaps are never touched.
5. Machine-generated report in `data/iu_xray/e2e_llm_validation_result.json` remains immutable.

Verification: Before and after finalization, machine artifacts are re-read and compared byte-for-byte against their initial state.

---

## 4. REST API Endpoints (Phase 1.2)

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/studies/{study_id}/review` | Get or automatically create the review session |
| `POST` | `/api/studies/{study_id}/review` | Create a review session with reviewer metadata |
| `POST` | `/api/studies/{study_id}/review/finding` | Update finding reviewer status, location, severity, notes |
| `GET` | `/api/studies/{study_id}/review/finding/{finding}` | Retrieve specific finding review state |
| `POST` | `/api/studies/{study_id}/review/qa` | Update reviewer QA answer and comments |
| `POST` | `/api/studies/{study_id}/review/report` | Save editable structured report draft |
| `POST` | `/api/studies/{study_id}/review/report/reset` | Reset report draft back to machine baseline |
| `POST` | `/api/studies/{study_id}/review/finalize` | Run finalization validation and lock session |
| `GET` | `/api/studies/{study_id}/review/audit` | Retrieve append-only chronological audit trail |
| `GET` | `/api/studies/{study_id}/review/export?format=json` | Export finalized review as JSON |
| `GET` | `/api/studies/{study_id}/review/export?format=text` | Export finalized review as structured text report |

All 10 previous Phase 1.0 & Phase 1.1 endpoints remain 100% backward compatible.

---

## 5. Finalization & Locking Rules

Enforced deterministically by [`backend/validate_final_review.py`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/backend/validate_final_review.py):
1. **Completeness**: Every candidate finding must be reviewed (`reviewed == true` and status $\in$ {`confirmed_present`, `confirmed_absent`, `uncertain`}). Any finding left as `not_reviewed` or `needs_review` blocks finalization.
2. **Report Structure**: Structured findings and impression statements must be non-empty and well-formed.
3. **Reviewer Attribution**: Valid reviewer ID and name required.
4. **Machine Integrity**: Machine baseline matches disk artifacts.
5. **Zero Ground-Truth Leakage**: Reference XML reports are absent from review structures.
6. **Post-Finalization Immutability**: Once finalized (`status == 'finalized'`), all subsequent POST requests to `/finding`, `/qa`, `/report`, `/report/reset`, or `/finalize` return `400 Bad Request` with an immutable locking error.

---

## 6. Audit Trail System

Every reviewer interaction creates an immutable chronological audit record:
- **Timestamp**: ISO 8601 UTC timestamp
- **Reviewer ID**: Identifier of the acting clinician/researcher
- **Finding**: Target pathology or session scope (`*SESSION*`)
- **Field**: Attribute modified (`reviewer_status`, `reviewer_location`, `reviewer_severity`, `qa_answer`, `report_draft`)
- **Old Value**: Previous value before mutation
- **New Value**: Updated value

Audit records are append-only and cannot be cleared or pruned once written.

---

## 7. Web Dashboard UI (Phase 1.2)

The web dashboard (`frontend/index.html`, `styles.css`, `app.js`) provides:
1. **Review Progress Tracker**:
   - `[1. Evidence Inspected]` $\rightarrow$ `[2. Findings Reviewed (7/7)]` $\rightarrow$ `[3. Report Draft]` $\rightarrow$ `[4. Validation (Valid)]` $\rightarrow$ `[5. Finalization (FINALIZED)]`
2. **Distinct Visual Boundaries**:
   - Machine Evidence Card styled in neutral slate (`MACHINE EVIDENCE (IMMUTABLE)`).
   - Reviewer Decision Card styled in glowing blue (`HUMAN REVIEWER DECISION`).
3. **Interactive Decision Buttons**:
   - `Confirm Present` (Emerald), `Confirm Absent` (Slate), `Uncertain` (Purple), `Needs Review` (Rose).
   - Real-time Machine vs Reviewer Disagreement Callout Banner.
4. **Diagnostic QA Review**:
   - Side-by-side display of immutable machine answer vs interactive reviewer correction with difference tags.
5. **Report Editor**:
   - Live editable finding statements, status selectors, multiline impression, and synthesis notes.
   - Quick action buttons: `Reset to Machine Report`, `Save Draft`, `Validate Review`, `Finalize Report`.
6. **Comparison Metrics Dashboard**:
   - Total Candidates, Reviewed, Not Reviewed, Present, Absent, Uncertain, Needs Review, Disagreements.
7. **Audit Trail Viewer**:
   - Expandable chronological log table.

---

## 8. Test Execution & Verification

### Unit & Regression Test Suite:
```bash
backend\venv\Scripts\python -m unittest discover -s tests -v
```
- **Total Tests**: 135 tests executed across Phases 0.6–1.2
- **Pass Rate**: 135 / 135 (100% PASS)
- **Phase 1.2 Tests**: 23 dedicated invariant tests in [`tests/test_phase_1_2_review.py`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/tests/test_phase_1_2_review.py)

### End-to-End Review Pipeline:
```bash
backend\venv\Scripts\python backend/run_e2e_review_pipeline.py
```
- **Total Steps**: 16 verification stages
- **Status**: ALL 16 PASS

### Phase 1.1 Regression Pipeline:
```bash
backend\venv\Scripts\python backend/run_e2e_phase_1_1.py
```
- **Total Steps**: 14 verification stages
- **Status**: ALL 14 PASS

---

## 9. Limitations & Research Disclaimer

- Reviewer annotations and finalized reports are exploratory research metadata only.
- Model scores from DenseNet-121 reflect feature activations, not calibrated clinical probabilities.
- Grad-CAM highlights attention regions, not anatomic lesion boundaries or segmentation masks.
- This prototype is strictly for research and explainability experimentation, not patient diagnostic workflows.
