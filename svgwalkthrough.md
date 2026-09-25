# Explainable Radiology Research Prototype — Architecture & Phase 2.0 Walkthrough

## Complete System Architecture Flow (Phases 0.6 – 2.0)

```
                      CHEST X-RAY (Multi-Study IU X-Ray Dataset)
                                          ↓
                         TorchXRayVision DenseNet-121
                           ↙                        ↘
                 18 Pathology Activations      1024-D Visual Features
                           ↓
                Diagnostic QA Engine (14 Hierarchical Questions)
                           ↓
                  Immutable Machine Evidence Layer (7 Findings)
                           ↙                                    ↘
           Grad-CAM Visual Grounding                    LLM Report Generation
         (model.features.norm5, [7, 7])                (Findings + Impression)
                           ↘                                    ↙
  ─────────────────────────────────────────────────────────────────────────────────
                     REST API Layer (backend/api.py & Core Managers)
  ─────────────────────────────────────────────────────────────────────────────────
                                          ↓
             Interactive Web Review & Research Dashboard (frontend/)
       ├── Multi-Study Search & Filter Bar (1,114 Discovered Studies)
       ├── Dual-View Radiograph Viewport (Frontal & Lateral Projections)
       ├── Synchronized Controls Switch (Zoom, Pan, Fit View, Attribution)
       ├── Finding-Specific Grad-CAM Overlays (0–100% Opacity)
       ├── 5-Stage Review Workflow Progress Tracker (Phase 1.2)
       ├── Multi-Reviewer Consensus & Adjudication Engine (Phase 1.3)
       ├── Dataset Management & Study Review Queue (Phase 1.4)
       ├── Research Evaluation & Experiment Tracking Engine (Phase 1.5)
       ├── Research Evaluation Dashboard & Statistical Analysis (Phase 1.6)
       ├── Reproducible Experiment Registry & Model Versioning (Phase 1.7)
       ├── Multi-Modal Experiment Benchmarking & Test Portability (Phase 1.8)
       ├── Research Experiment Orchestration & Reporting (Phase 1.9)
       └── 🔬 Phase 2.0 Interactive Counterfactual Explainability
             ├── Counterfactual Perturbation Engine (8 Methods, Bounded & Deterministic)
             ├── Synchronized 3-Panel Viewer (Original | Counterfactual | Attribution Diff)
             ├── Finding Activation Deltas (Safe Relative Deltas, Zero-Denominator Safe)
             ├── Grad-CAM Heatmap Difference Map & Energy Shift Metrics
             ├── Diagnostic QA Impact Propagation (Level 1 Presence, Level 2 Severity)
             ├── Machine Report Statement & Status Transition Diffing
             ├── Random Non-Overlapping Control ROI Comparator & Bootstrap CI
             ├── 7-State Immutable Experiment Lifecycle (DRAFT → PUBLISHED Lock)
             ├── Deterministic Fingerprinting & Runtime Drift Detection
             └── Strict Research Disclaimers & Zero Ground-Truth Leakage Invariants
```

## Summary of Results (Phase 2.0)
- **Files Created**:
  - `docs/counterfactual_schema.json`
  - `backend/counterfactual_manager.py`
  - `backend/counterfactual_comparator.py`
  - `backend/validate_phase_2_0.py`
  - `tests/test_phase_2_0_counterfactual.py`
  - `backend/run_e2e_phase_2_0.py`
  - `docs/phase_2_0_counterfactual_explainability.md`
- **Files Modified**:
  - `backend/api.py`
  - `frontend/index.html`
  - `frontend/app.js`
  - `README.md`
  - `walkthrough.md`
  - `svgwalkthrough.md`
- **Tests Executed & Passed**:
  - `python -m unittest discover -s tests -v`: **407 / 407 passed (100%)** (35 dedicated Phase 2.0 unit & integration tests).
- **End-to-End Pipelines**:
  - `python backend/run_e2e_phase_2_0.py`: **20 / 20 verification stages passed (100%)**.
  - `python backend/run_e2e_phase_1_9.py`: **21 / 21 verification stages passed (100%)**.
  - `python backend/run_e2e_phase_1_8.py`: **28 / 28 verification stages passed (100%)**.
- **Invariants Verified**:
  - All 8 perturbation operations execute deterministically without mutating disk images.
  - Zero XML ground-truth reference report leakage (`<eFind>`, `<eImpression>`).
  - Byte-for-byte machine evidence immutability (`data/iu_xray/` 5,399 files unchanged).
  - Permanent immutable locking for published counterfactual experiments.
  - Strict non-clinical research framing across all UI components and API responses.
