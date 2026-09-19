# Phase 1.1 — Multi-Study Browsing, Dual-View Radiographs, Reviewer Workflow & Batch Processing

## Research Prototype Disclaimer
> **IMPORTANT NOTICE**: This software is an explainability and research prototype designed to investigate diagnostic questioning and visual attribution methods for LLM-assisted radiology report generation. It is **NOT** a certified medical device, clinical decision support system, or diagnostic tool. Model activation scores are non-probabilistic continuous signals, and Grad-CAM maps reflect internal neural network feature attributions rather than definitive anatomical segmentations or lesion boundaries. All outputs require licensed clinician review.

---

## 1. Architecture Overview

Phase 1.1 extends the validated Phase 0.6–1.0 architecture by introducing multi-study indexing, synchronized dual-view radiograph examination, persistent clinician/reviewer annotation layers, and a decoupled 6-stage batch study runner.

```
+-----------------------------------------------------------------------------------+
|                                 CLIENT WEB UI                                     |
|  +--------------------+  +--------------------+  +-----------------------------+  |
|  | Multi-Study Search |  | Dual-View Viewport |  | Reviewer Annotation Layer   |  |
|  | & View Filtering   |  | (Frontal + Lateral)|  | (Separated from Machine EV) |  |
|  +--------------------+  +--------------------+  +-----------------------------+  |
+-----------------------------------------+-----------------------------------------+
                                          | REST API (HTTP Server)
                                          v
+-----------------------------------------------------------------------------------+
|                                BACKEND PIPELINE                                   |
|  +-----------------------------------------------------------------------------+  |
|  | 1. Vision Backbone (TorchXRayVision DenseNet-121, 18 Activations, 1024-D)   |  |
|  | 2. Diagnostic QA Engine (Presence, Location, Severity Hierarchical QA)      |  |
|  | 3. Immutable Machine Evidence Layer (Strict Invariant Protection)            |  |
|  | 4. Grad-CAM Visual Grounding (Layer norm5 7x7 Feature Map Attributions)    |  |
|  | 5. LLM Report Generation (Evidence-Sanitized Prompting)                     |  |
|  | 6. Safety & Schema Validator (Zero Ground-Truth Leakage Guarantee)          |  |
|  +-----------------------------------------------------------------------------+  |
|  +------------------------------------+   +------------------------------------+  |
|  | Reviewer Annotation Store (JSON)   |   | Thread-Safe Batch Processor        |  |
|  | data/reviewer_annotations/*.json   |   | 6-Stage Job Lifecycle Manager      |  |
|  +------------------------------------+   +------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

---

## 2. API Additions & Backward Compatibility

All 10 existing Phase 1.0 endpoints remain backward compatible. The following new endpoints have been introduced in Phase 1.1:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/studies` | Enhanced multi-study index containing available views, image counts, report availability, and validation status. |
| `GET` | `/api/studies/{study_id}/images` | Returns image metadata, view designations (`Frontal`, `Lateral`), and visual grounding availability for dual-view viewing. |
| `GET` | `/api/studies/{study_id}/image/{image_id}` | Serves a specific radiograph image binary from the study's image collection. |
| `GET` | `/api/studies/{study_id}/reviews` | Retrieves saved reviewer annotations for the study. |
| `POST` | `/api/studies/{study_id}/reviews` | Saves or updates a reviewer annotation for a specific candidate finding. |
| `DELETE` | `/api/studies/{study_id}/reviews/{finding}` | Removes a reviewer annotation for a specific candidate finding. |
| `POST` | `/api/batch/run` | Triggers a background batch processing job across a list of study IDs. |
| `GET` | `/api/batch/status` | Queries current/latest batch job execution status and stage progress. |
| `GET` | `/api/batch/status/{job_id}` | Queries a specific batch job's status and detailed per-study execution log. |

---

## 3. Dual-View Radiograph Support & Synchronization

1. **Dual-View Configuration**:
   - Detects when a study has multiple radiographic projections (e.g. `Frontal` / `PA` / `AP` and `Lateral`).
   - Dynamically expands the UI into a dual-viewport configuration (Primary on the left, Secondary on the right).
   - Single-view studies automatically collapse back to the compact single-viewer layout.
2. **Synchronized Controls**:
   - **`SYNC VIEWS` Switch**: When active (default for multi-view studies), zoom adjustments, pan mouse dragging, and Fit View operations are synchronized across both viewports.
   - **Finding Attribution Propagation**: Selecting a candidate finding updates the Grad-CAM attribution overlay on both viewports simultaneously.
   - **Independent State Isolation**: If the reviewer toggles `SYNC VIEWS` off, pan and zoom transforms operate independently per viewport without state corruption.
3. **Missing View & Grounding Handling**:
   - If visual attribution is unavailable for a secondary view, the interface explicitly displays `"Visual attribution unavailable for this view."` rather than fabricating synthetic heatmaps.

---

## 4. Clinician / Reviewer Annotation System

1. **Two Distinct Layers**:
   - **MACHINE EVIDENCE**: Immutable machine-generated findings derived from DenseNet-121 activations and Diagnostic QA.
   - **RESEARCH REVIEWER ANNOTATION**: Research metadata recorded by reviewing clinicians.
2. **Reviewer Status Taxonomy**:
   - `[ Not Reviewed ]`
   - `[ Confirmed Present ]`
   - `[ Confirmed Absent ]`
   - `[ Uncertain ]`
3. **Machine vs. Reviewer Comparison Card**:
   - The UI directly juxtaposes machine status against reviewer status (e.g., `Machine: POSSIBLE` vs. `Reviewer: UNCERTAIN`).
   - Explains that divergence is expected during exploratory model evaluation.
4. **JSON-Based Persistent Storage**:
   - Stored in `data/reviewer_annotations/{study_id}.json`.
   - Includes `study_id`, `image_id`, `finding`, `reviewer_status`, `location`, `severity`, `notes`, `timestamp`, and `schema_version`.

---

## 5. Batch Study Processing Pipeline

1. **6-Stage Lifecycle Tracking**:
   - Tracks study execution through `Vision` $\rightarrow$ `Diagnostic QA` $\rightarrow$ `Evidence` $\rightarrow$ `Grounding` $\rightarrow$ `LLM` $\rightarrow$ `Validation`.
2. **Lifecycle States**:
   - `QUEUED` $\rightarrow$ `RUNNING` $\rightarrow$ `COMPLETED` / `FAILED`.
3. **Ground-Truth Isolation**:
   - Batch runner executes model inference and report generation without accessing ground-truth XML files, preserving benchmark validity.

---

## 6. Safety Guarantees & Invariant Protections

- **No Ground-Truth Leakage**: Reference XML reports are never loaded into prompt contexts, API payloads, or client bundles.
- **Model Score Fidelity**: Activation values are preserved as continuous floating-point numbers and never misrepresented as probabilities or confidence percentages.
- **Visual Grounding Limitations**: Grad-CAM is strictly designated as an internal activation attribution signal, never as lesion segmentation or localization.
- **Evidence Immutability**: Reviewer annotations do not modify underlying machine evidence JSON artifacts.
- **No Synthetic Hallucinations**: Missing views and missing heatmaps are explicitly indicated and never fabricated.

---

## 7. Verification Summary

- **Total Test Suite**: 112 automated unit and integration tests passing (`112 / 112 PASS`).
- **End-to-End Suite**: 14/14 automated E2E steps verified in `backend/run_e2e_phase_1_1.py`.
