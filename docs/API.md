# REST API Reference Documentation

**LLM-Assisted Explainable Radiology Report Generation (Phases 1.0–1.1)**

> **RESEARCH DISCLAIMER**: This API is a research prototype for explainability and evaluation of intermediate diagnostic questioning. It is **NOT** a certified medical device, clinical decision support system, or diagnostic tool. Model activation scores are non-probabilistic continuous signals, and Grad-CAM maps represent internal convolutional feature attributions rather than lesion segmentations or boundaries.

---

## 1. Overview & Server Information

- **Default Base URL**: `http://127.0.0.1:8000`
- **Protocol**: HTTP/1.1 (Standard Library `http.server`)
- **Data Format**: `application/json; charset=utf-8`
- **CORS Support**: Enabled (`Access-Control-Allow-Origin: *`, `Methods: GET, POST, DELETE, OPTIONS`)
- **Static Frontend**: Serving `index.html`, `styles.css`, `app.js` at root `/`

---

## 2. Study Discovery & Dual-View Endpoints

### 2.1 List Discovered Studies
`GET /api/studies`

Returns the multi-study discovery index dynamically scanned from the dataset directory.

#### Response `200 OK`
```json
{
  "total": 1114,
  "studies": [
    {
      "study_id": "CXR1122",
      "image_ids": [
        "CXR1122_IM-0080-1001-0002"
      ],
      "images_count": 1,
      "available_views": [
        "Frontal"
      ],
      "views": [
        "Frontal"
      ],
      "has_report": true,
      "report_available": true,
      "has_evidence": true,
      "evidence_available": true,
      "has_grounding": true,
      "grounding_available": true,
      "validation_status": "PASS",
      "model": "TorchXRayVision DenseNet-121",
      "findings_count": 7,
      "is_active": true
    }
  ]
}
```

---

### 2.2 Get Complete Aggregated Study Data
`GET /api/studies/{study_id}`

Retrieves the complete aggregated study package combining evidence, QA, grounding, structured report, and validation metadata.

#### Parameters
- `study_id` (path): Study identifier (e.g., `CXR1122`).

#### Response `200 OK`
```json
{
  "study_id": "CXR1122",
  "image_id": "CXR1122_IM-0080-1001-0002",
  "view": "Frontal",
  "original_image_url": "/api/studies/CXR1122/image",
  "evidence": [
    {
      "finding": "Infiltration",
      "status": "possible",
      "model_score": 0.475,
      "location": "unspecified",
      "severity": "unspecified",
      "evidence_source": "vision_model"
    }
  ],
  "qa_candidates": [...],
  "qa_questions": [...],
  "groundings": [...],
  "report": {
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
      }
    ],
    "impression": [
      "1. Possible infiltration... recommend clinical correlation."
    ],
    "metadata": {
      "provider": "mock",
      "model": "mock-radiology-llm"
    }
  },
  "validation": {
    "schema_validation": "PASS",
    "evidence_validation": "PASS",
    "report_validation": "PASS",
    "ground_truth_isolation": "PASS",
    "grounding_validation": "PASS",
    "safety_violations_count": 0,
    "pipeline_stages": {
      "vision": true,
      "qa": true,
      "evidence": true,
      "grounding": true,
      "llm": true,
      "validation": true
    }
  },
  "disclaimer": "Research Prototype — Visual attribution maps and report suggestions are for investigative explainability and not for clinical diagnostic decision-making."
}
```

---

### 2.3 Get Dual-View Image Metadata
`GET /api/studies/{study_id}/images`

Returns image metadata, view classifications (`Frontal`, `Lateral`), and visual grounding availability for dual-view viewing.

#### Response `200 OK`
```json
{
  "study_id": "CXR1122",
  "total_images": 1,
  "has_dual_view": false,
  "images": [
    {
      "image_id": "CXR1122_IM-0080-1001-0002",
      "view": "Frontal",
      "available": true,
      "is_primary": true,
      "image_url": "/api/studies/CXR1122/image/CXR1122_IM-0080-1001-0002",
      "grounding_available": true
    }
  ]
}
```

---

### 2.4 Get Primary Radiograph Binary
`GET /api/studies/{study_id}/image`

Serves the primary radiograph image binary stream (`image/png`).

---

### 2.5 Get Specific Radiograph Binary by Image ID
`GET /api/studies/{study_id}/image/{image_id}`

Serves a specific radiograph image binary stream (`image/png`).

---

## 3. Evidence, QA & Grounding Sub-Resources

### 3.1 Get Machine Evidence Layer
`GET /api/studies/{study_id}/evidence`

#### Response `200 OK`
```json
{
  "study_id": "CXR1122",
  "image_id": "CXR1122_IM-0080-1001-0002",
  "evidence": [
    {
      "finding": "Infiltration",
      "status": "possible",
      "model_score": 0.475,
      "location": "unspecified",
      "severity": "unspecified"
    }
  ]
}
```

---

### 3.2 Get Diagnostic QA Questions
`GET /api/studies/{study_id}/qa`

#### Response `200 OK`
```json
{
  "study_id": "CXR1122",
  "image_id": "CXR1122_IM-0080-1001-0002",
  "candidates": [...],
  "questions": [
    {
      "question_id": "Q1_Infiltration",
      "finding": "Infiltration",
      "level": 1,
      "question_text": "Is infiltration present in the radiograph?",
      "answer": "uncertain",
      "rationale": "Vision activation score 0.4750 indicates borderline presence."
    }
  ]
}
```

---

### 3.3 Get Visual Grounding Package
`GET /api/studies/{study_id}/grounding`

#### Response `200 OK`
```json
{
  "study_id": "CXR1122",
  "image_id": "CXR1122_IM-0080-1001-0002",
  "groundings": [
    {
      "finding": "Infiltration",
      "model_score": 0.475,
      "status": "possible",
      "target_layer": "model.features.norm5",
      "activation_shape": [7, 7],
      "heatmap_shape": [512, 624],
      "normalization": "minmax_0_1",
      "heatmap_url": "/api/artifacts/grounding/CXR1122_CXR1122_IM-0080-1001-0002_infiltration_heatmap.png",
      "overlay_url": "/api/artifacts/grounding/CXR1122_CXR1122_IM-0080-1001-0002_infiltration_overlay.png",
      "notes": "Grad-CAM visual attribution map."
    }
  ],
  "disclaimer": "Research Prototype — Visual attribution maps and report suggestions are for investigative explainability and not for clinical diagnostic decision-making."
}
```

---

### 3.4 Get Single Finding Grounding
`GET /api/studies/{study_id}/finding/{finding}/grounding`

#### Response `200 OK`
```json
{
  "finding": "Infiltration",
  "model_score": 0.475,
  "status": "possible",
  "target_layer": "model.features.norm5",
  "activation_shape": [7, 7],
  "heatmap_shape": [512, 624],
  "normalization": "minmax_0_1",
  "heatmap_url": "/api/artifacts/grounding/CXR1122_CXR1122_IM-0080-1001-0002_infiltration_heatmap.png",
  "overlay_url": "/api/artifacts/grounding/CXR1122_CXR1122_IM-0080-1001-0002_infiltration_overlay.png",
  "notes": "Grad-CAM visual attribution map."
}
```

---

## 4. Structured Report & Validation Endpoints

### 4.1 Get Generated Report
`GET /api/studies/{study_id}/report`

#### Response `200 OK`
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
    }
  ],
  "impression": [
    "1. Possible infiltration, Possible pneumothorax... recommend clinical correlation."
  ],
  "metadata": {
    "provider": "mock",
    "model": "mock-radiology-llm",
    "generated_at": "2026-09-19T01:23:24Z"
  }
}
```

---

### 4.2 Get Validation & Safety Invariants
`GET /api/studies/{study_id}/validation`

#### Response `200 OK`
```json
{
  "schema_validation": "PASS",
  "evidence_validation": "PASS",
  "report_validation": "PASS",
  "ground_truth_isolation": "PASS",
  "grounding_validation": "PASS",
  "clinical_validation": "NOT PERFORMED (Research Prototype Only)",
  "safety_violations_count": 0,
  "pipeline_stages": {
    "vision": true,
    "qa": true,
    "evidence": true,
    "grounding": true,
    "llm": true,
    "validation": true
  }
}
```

---

### 4.3 Export Structured Report
`GET /api/studies/{study_id}/export?format=json|text`

#### Parameters
- `format` (query): `json` (default) or `text`.

#### Response `200 OK` (JSON format)
```json
{
  "disclaimer": "Research Prototype — ...",
  "research_disclaimer": "Research Prototype — ...",
  "study_id": "CXR1122",
  "image_id": "CXR1122_IM-0080-1001-0002",
  "view": "Frontal",
  "report": {...},
  "evidence": [...],
  "validation": {...},
  "reviewer_annotations": [...]
}
```

#### Response `200 OK` (Text format)
`Content-Type: text/plain; charset=utf-8`
```text
=================================================================
EXPLAINABLE RADIOLOGY REPORT — RESEARCH PROTOTYPE EXPORT
RESEARCH PROTOTYPE — NOT FOR CLINICAL USE
DISCLAIMER: For research/investigative use only; not a diagnostic tool.
RESEARCH OUTPUT — REQUIRES HUMAN REVIEW. NOT A CLINICAL DIAGNOSIS.
=================================================================
STUDY ID : CXR1122
IMAGE ID : CXR1122_IM-0080-1001-0002
VIEW     : Frontal
PROVIDER : mock

FINDINGS:
- [POSSIBLE  ] Possible infiltration cannot be excluded; correlation is advised.
...
```

---

## 5. Clinician / Reviewer Annotation Endpoints

### 5.1 Get Study Reviewer Annotations
`GET /api/studies/{study_id}/reviews`

#### Response `200 OK`
```json
{
  "study_id": "CXR1122",
  "schema_version": "1.0",
  "reviews": {
    "Infiltration": {
      "finding": "Infiltration",
      "reviewer_status": "confirmed_present",
      "location": "right_lower_lobe",
      "severity": "mild",
      "notes": "Reviewer confirmation test note.",
      "timestamp": "2026-09-19T04:33:35Z",
      "schema_version": "1.0"
    }
  },
  "annotations": [
    {
      "finding": "Infiltration",
      "reviewer_status": "confirmed_present",
      "location": "right_lower_lobe",
      "severity": "mild",
      "notes": "Reviewer confirmation test note.",
      "timestamp": "2026-09-19T04:33:35Z",
      "schema_version": "1.0"
    }
  ],
  "disclaimer": "Research Reviewer Annotations are exploratory metadata and do not constitute clinical ground truth."
}
```

---

### 5.2 Save / Update Reviewer Annotation
`POST /api/studies/{study_id}/reviews`

#### Request Body
```json
{
  "study_id": "CXR1122",
  "image_id": "CXR1122_IM-0080-1001-0002",
  "finding": "Infiltration",
  "reviewer_status": "confirmed_present",
  "location": "right_lower_lobe",
  "severity": "mild",
  "notes": "Clinical review notes."
}
```

#### Status Codes
- `200 OK`: Annotation saved successfully.
- `400 Bad Request`: Missing finding or invalid status value (allowed: `not_reviewed`, `confirmed_present`, `confirmed_absent`, `uncertain`).

---

### 5.3 Delete Reviewer Annotation
`DELETE /api/studies/{study_id}/reviews/{finding}`

#### Status Codes
- `200 OK`: Annotation deleted.
- `404 Not Found`: No annotation found for finding.

---

## 6. Batch Study Runner Endpoints

### 6.1 Trigger Batch Processing Job
`POST /api/batch/run`

#### Request Body
```json
{
  "study_ids": ["CXR1122", "CXR1007", "CXR1009"]
}
```

#### Response `200 OK`
```json
{
  "job_id": "batch_2a7b2898",
  "status": "QUEUED",
  "study_ids": ["CXR1122", "CXR1007", "CXR1009"],
  "total_studies": 3,
  "message": "Batch processing job queued successfully."
}
```

---

### 6.2 Get Batch Runner Status
`GET /api/batch/status` or `GET /api/batch/status/{job_id}`

#### Response `200 OK`
```json
{
  "job_id": "batch_2a7b2898",
  "status": "COMPLETED",
  "total_studies": 1,
  "current_index": 1,
  "current_study_id": "CXR1122",
  "current_stage": "Validation",
  "results": [
    {
      "study_id": "CXR1122",
      "status": "COMPLETED",
      "stage": "Validation",
      "validation_status": "PASS",
      "error": null
    }
  ],
  "created_at": "2026-09-19T04:37:47Z",
  "updated_at": "2026-09-19T04:37:48Z"
}
```
