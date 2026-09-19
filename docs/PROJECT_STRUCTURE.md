# Project Structure & Architecture Guide

This document describes the directory hierarchy, core modules, data flow, and responsibility boundaries of the **Explainable Radiology Report Generation** research prototype (Phases 0.6–1.1).

---

## 1. Top-Level Repository Overview

```
radio-llm/
├── .env.example                 # Environment configuration template
├── .gitignore                   # Git exclusion rules for large datasets & caches
├── README.md                    # Main project documentation & quick start
├── TROUBLESHOOTING.md           # Troubleshooting guide for common local setup issues
├── CONTRIBUTING.md              # Team collaboration, git workflow, and PR standards
├── requirements.txt             # Primary Python dependencies
├── backend/                     # Python backend services, ML models, and API
├── data/                        # Dataset schemas, fixtures, and local storage guide
├── docs/                        # Specifications, API reference, and phase documentation
├── frontend/                    # Vanilla HTML5/CSS3/JavaScript web interface
├── scripts/                     # Dataset acquisition and automated setup scripts
└── tests/                       # Automated unit and integration test suite
```

---

## 2. Backend Modules (`backend/`)

The backend is built with Python 3.11 using standard library services (`http.server`, `urllib.request`) and PyTorch/TorchXRayVision for deep learning.

### 2.1 API & Background Services
- [`api.py`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/backend/api.py): REST API server handling static frontend hosting, multi-study discovery, dual-view radiograph serving, reviewer annotation persistence, and report exports.
- [`batch_processor.py`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/backend/batch_processor.py): Decoupled asynchronous batch execution manager tracking studies across the 6 pipeline stages.

### 2.2 Deep Learning & Diagnostic QA
- [`test_vision.py`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/backend/test_vision.py): Vision pipeline runner executing TorchXRayVision DenseNet-121 (`densenet121-res224-all`), producing 18 pathology activation scores and a 1024-dimensional feature vector.
- [`diagnostic_qa.py`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/backend/diagnostic_qa.py): Diagnostic Questioning engine formulating Level 1 (presence) and Level 2 (location/severity) questions to prevent over-certainty.
- [`evaluate_qa.py`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/backend/evaluate_qa.py): Evaluates candidate questions against model activation signals and clinical thresholds.
- [`report_parser.py`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/backend/report_parser.py): Offline ground-truth XML report parser reserved strictly for evaluation/benchmarking.

### 2.3 Evidence Layer & Validation
- [`evidence_layer.py`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/backend/evidence_layer.py): Resolves QA answers into an immutable machine evidence bundle (`supported`, `possible`, `uncertain`, `absent`).
- [`validate_evidence.py`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/backend/validate_evidence.py): Strict schema and semantic invariant validator enforcing evidence integrity.

### 2.4 Visual Grounding & Explainability
- [`visual_grounding.py`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/backend/visual_grounding.py): Implements Grad-CAM visual attribution on layer `model.features.norm5` ($7 \times 7$ feature map) and generates color-mapped overlays.
- [`run_grounding.py`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/backend/run_grounding.py): CLI grounding runner generating heatmaps for candidate findings.
- [`validate_grounding.py`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/backend/validate_grounding.py): Verifies Grad-CAM heatmap dimensions, normalization bounds ($[0, 1]$), and layer immutability.

### 2.5 LLM Synthesis & Providers (`backend/llm/`)
- [`backend/llm/base.py`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/backend/llm/base.py): Abstract `LLMProvider` interface and `LLMConfig` dataclass with environment-variable loading.
- [`backend/llm/mock_provider.py`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/backend/llm/mock_provider.py): Deterministic offline mock LLM generator preserving clinical phrasing and uncertainty semantics.
- [`backend/llm/openai_provider.py`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/backend/llm/openai_provider.py): Standard-library `urllib.request` client for OpenAI-compatible chat completion APIs.
- [`backend/llm/prompts.py`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/backend/llm/prompts.py): System prompts and sanitized evidence formatting templates.
- [`backend/llm/factory.py`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/backend/llm/factory.py): Provider factory instantiating configured LLM backends.
- [`report_generator.py`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/backend/report_generator.py): Top-level coordinator orchestrating LLM synthesis from evidence packages.
- [`validate_report.py`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/backend/validate_report.py): Invariant validator enforcing that every report statement traces back to evidence.

### 2.6 End-to-End Verification Pipeline Scripts
- [`run_e2e_phase_1_1.py`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/backend/run_e2e_phase_1_1.py): Complete 14-step verification of multi-study, dual-view, reviewer CRUD, and batch capabilities.
- [`run_e2e_ui_pipeline.py`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/backend/run_e2e_ui_pipeline.py): Phase 1.0 UI integration and REST contract verification.
- [`run_e2e_visual_grounding.py`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/backend/run_e2e_visual_grounding.py): Phase 0.9 Grad-CAM verification.
- [`run_e2e_llm_pipeline.py`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/backend/run_e2e_llm_pipeline.py): Phase 0.8 LLM report generation verification.
- [`run_e2e_evidence_pipeline.py`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/backend/run_e2e_evidence_pipeline.py): Phase 0.7 Evidence Layer verification.
- [`run_e2e_qa_pipeline.py`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/backend/run_e2e_qa_pipeline.py): Phase 0.6 Diagnostic QA verification.

---

## 3. Frontend Architecture (`frontend/`)

Built with standard web technologies (HTML5, CSS3, ES6+ JavaScript) without heavy framework dependencies.

- [`index.html`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/frontend/index.html): Semantic layout containing:
  - Multi-study selector, search, and view filtering toolbar.
  - Dual-viewport radiograph canvas grid with `SYNC VIEWS` switch.
  - Candidate finding pills and continuous model activation score cards.
  - Research Reviewer Annotation form and `MACHINE vs REVIEWER` comparison box.
  - 6-Stage evidence traceability chain.
  - Structured Findings and Impression sections with JSON/Text export.
  - Batch Study Runner modal dialog.
- [`styles.css`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/frontend/styles.css): Dark medical-tech design system with glassmorphism, responsive grid, status badges, and accessible typography.
- [`app.js`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/frontend/app.js): Asynchronous client state manager handling canvas blending, pan/zoom synchronization, reviewer CRUD, and batch polling.

---

## 4. Test Suite (`tests/`)

Automated test suite using Python standard `unittest`:
- [`test_phase_0_6_qa.py`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/tests/test_phase_0_6_qa.py): 20 tests verifying question selection, answer resolution, and absence handling.
- [`test_phase_0_7_evidence.py`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/tests/test_phase_0_7_evidence.py): 20 tests verifying evidence schemas, uncertainty preservation, and zero ground-truth leakage.
- [`test_phase_0_8_llm.py`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/tests/test_phase_0_8_llm.py): 20 tests verifying LLM input sanitization, mock provider determinism, and report structure.
- [`test_phase_0_9_grounding.py`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/tests/test_phase_0_9_grounding.py): 10 tests verifying Grad-CAM target layers, heatmap normalization ($[0, 1]$), and overlay dimensions.
- [`test_phase_1_0_api.py`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/tests/test_phase_1_0_api.py): 22 tests verifying REST endpoints, static serving, CORS headers, and safety disclaimers.
- [`test_phase_1_1_multistudy.py`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/tests/test_phase_1_1_multistudy.py): 20 tests verifying multi-study indexing, dual-view metadata, reviewer CRUD persistence, and batch processing.

**Total Automated Coverage**: 112 / 112 passing tests.

---

## 5. Documentation (`docs/`)

- [`API.md`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/docs/API.md): Full REST API endpoint reference with JSON payloads.
- [`PROJECT_STRUCTURE.md`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/docs/PROJECT_STRUCTURE.md): This file.
- [`phase_1_1_multistudy_review.md`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/docs/phase_1_1_multistudy_review.md): Phase 1.1 detailed technical specifications.
- [`phase_1_0_web_ui.md`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/docs/phase_1_0_web_ui.md): Phase 1.0 web interface and safety communication guide.
- [`phase_0_9_visual_grounding.md`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/docs/phase_0_9_visual_grounding.md): Grad-CAM math, target layer selection, and visual grounding design.
- [`phase_0_8_llm_integration.md`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/docs/phase_0_8_llm_integration.md): LLM provider architecture and report validation rules.
- [`phase_0_7_evidence_layer.md`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/docs/phase_0_7_evidence_layer.md): Evidence Layer intermediate representation and ground-truth isolation.
- [`phase_0_6_diagnostic_qa_implementation.md`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/docs/phase_0_6_diagnostic_qa_implementation.md): Diagnostic QA engine rules and question hierarchies.
- [`evidence_schema.json`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/docs/evidence_schema.json), [`qa_schema.json`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/docs/qa_schema.json), [`report_schema.json`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/docs/report_schema.json), [`grounding_schema.json`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/docs/grounding_schema.json): Formal JSON schema definitions.
