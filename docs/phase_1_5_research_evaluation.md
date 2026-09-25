# Phase 1.5 — Research Evaluation, Experiment Tracking & Dataset-Level Comparison

## 1. Overview & Research Prototype Disclaimer

> ### ⚠️ RESEARCH EVALUATION ONLY — NOT CLINICAL PERFORMANCE
> - All experiment tracking records, dataset snapshots, configuration fingerprints, and comparison metrics are academic research artifacts.
> - They are strictly non-evaluative and must NOT be interpreted as certified medical efficacy, diagnostic accuracy, or clinical performance rankings.
> - Zero clinician evaluation, doctor scoring, or patient diagnostic risk stratification is performed.

Phase 1.5 turns the Explainable Radiology Research Prototype into a structured research evaluation and experiment tracking platform. It enables reproducible experiments across immutable dataset snapshots, computes standardized non-evaluative research metrics, tracks complete 11-stage provenance pipelines, and allows side-by-side comparison of different model/pipeline configurations.

---

## 2. Multi-Layer Storage Isolation

The architecture enforces complete multi-layer storage isolation:

```
data/
├── iu_xray/       <- Layer 1: Immutable machine artifacts (DenseNet-121, QA, Grad-CAM, LLM reports) [READ-ONLY]
├── reviews/       <- Layer 2: Individual clinician review sessions ({study_id}_{reviewer_id}.json)
├── consensus/     <- Layer 3: Multi-reviewer consensus sessions and adjudication records ({study_id}.json)
├── snapshots/     <- Layer 4: Immutable dataset snapshot manifests ({snapshot_id}.json)
└── experiments/   <- Layer 5: Reproducible research experiments ({experiment_id}/)
    └── exp_YYYYMMDD_HHMMSS_HASH/
        ├── metadata.json       (Complete experiment definition & status)
        ├── configuration.json  (Canonical fingerprint & hyperparameters)
        ├── metrics.json        (Descriptive research evaluation metrics)
        ├── provenance.json     (11-stage provenance audit chain)
        ├── validation.json     (Invariant validation results)
        └── exports/            (Sanitized JSON and plain-text export packages)
```

---

## 3. Core Components

### 3.1 Immutable Dataset Snapshots (`DatasetSnapshotManager`)
- **Location**: `backend/dataset_snapshot_manager.py`
- **Storage**: `data/snapshots/{snapshot_id}.json`
- **Mechanism**:
  - Deterministic SHA-256 manifest hashing over alphabetically sorted study ID lists.
  - Source artifact checksum recording across baseline machine evidence.
  - View distribution profiling (`Frontal`, `Lateral`).
  - Strict immutability: once created, snapshots cannot be modified.

### 3.2 Configuration Fingerprinting (`ExperimentManager`)
- **Location**: `backend/experiment_manager.py`
- **Fingerprint Hash**: Deterministic SHA-256 hash over canonical JSON representation of pipeline hyperparameters:
  - `model_name`: e.g. `TorchXRayVision DenseNet-121`
  - `model_version`: e.g. `densenet121-res224-all`
  - `target_layer`: e.g. `model.features.norm5`
  - `preprocessing_resolution`: `[224, 224]`
  - `normalization`: `minmax_0_1`
  - `device`: `cpu`
  - `qa_threshold`: `0.15`
  - `qa_top_k`: `5`
  - `llm_provider`: `mock`
  - `llm_model`: `mock-radiology-llm`
- **Security & Privacy**: Automatically strips any passwords, tokens, or API keys from fingerprint generation and storage.

### 3.3 Standardized Research Metrics (`ResearchMetricsCalculator`)
- **Location**: `backend/research_metrics.py`
- **Metric Categories**:
  1. **Review Coverage**: Total studies, reviewed studies, finalized studies, review completion ratio ($[0.0, 1.0]$).
  2. **Consensus Coverage**: Unanimous studies, majority studies, adjudication required studies, finalized consensus sessions.
  3. **Machine–Reviewer Agreement**: Overall concordance ratio and per-finding concordance ratios without evaluative scoring.
  4. **Inter-Rater Reliability**: Cohen's Kappa for 2 reviewers (sample size $N$, average $\kappa \in [-1.0, 1.0]$) and Fleiss' Kappa for $\ge 3$ reviewers.
  5. **Explainability Coverage**: Grounded findings count, Grad-CAM generation success rate ($[0.0, 1.0]$).
  6. **Human Workflow Corrections**: Counts of reviewer modifications to status, location, severity, and report text.
  7. **Mathematical Safety**: Bounded numerical ranges; zero `NaN`, `Infinity`, or `-Infinity` values.

### 3.4 11-Stage Experiment Provenance Pipeline (`ExperimentRunner`)
- **Location**: `backend/experiment_runner.py`
- **11 Provenance Stages**:
  1. `Immutable Dataset Snapshot`: Bound snapshot ID and manifest hash.
  2. `Study Cohort Selection`: Total studies evaluated and deterministic ordering.
  3. `Vision Backbone Model`: DenseNet-121 architecture and weights version.
  4. `Image Preprocessing & Normalization`: Target resolution ($224 \times 224$) and normalization range.
  5. `Diagnostic QA Engine`: Candidate selection thresholds and top-$k$ limit.
  6. `Immutable Evidence Layer`: Categorical evidence status derivation.
  7. `Grad-CAM Visual Grounding`: Feature attribution layer and heatmap resolution.
  8. `Independent Human Review Tracking`: Clinician review coverage and participation.
  9. `Multi-Reviewer Consensus Synthesis`: Agreement classification and dispute states.
  10. `Research Metrics & Inter-Rater Reliability`: Statistical concordance and Kappa scores.
  11. `Finalized Reproducible Experiment Artifact`: Permanent lock and export verification.

### 3.5 Side-by-Side Experiment Comparison (`ExperimentComparator`)
- **Location**: `backend/experiment_comparator.py`
- **Features**:
  - Compares 2 or more completed experiment runs side-by-side.
  - Automatically identifies configuration differences (e.g. `target_layer`, `qa_threshold`).
  - Contrasts observed research metrics across review coverage, inter-rater reliability, machine agreement, explainability coverage, and workflow edits.
  - **Strictly Non-Evaluative**: Zero "winner", "superior", or "ranking" terminology.

### 3.6 Invariant & Security Validator (`ExperimentValidator`)
- **Location**: `backend/validate_experiment.py`
- **Checks**:
  - Schema validity against `docs/experiment_schema.json`.
  - SHA-256 configuration fingerprint verification.
  - Metric mathematical bounds ($[-1.0, 1.0]$ for Kappa, $[0.0, 1.0]$ for completion ratios).
  - Snapshot cryptographic integrity.
  - Zero ground-truth reference report leakage (`<eFind>`, `<eImpression>`, etc.).
  - Zero private tokens, passwords, or API keys.
  - Zero path traversal in experiment identifiers.

---

## 4. REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/experiments/snapshots` | List all immutable dataset snapshots |
| `POST` | `/api/experiments/snapshots` | Create a new immutable dataset snapshot |
| `GET` | `/api/experiments/snapshots/{id}` | Retrieve specific snapshot details and manifest hash |
| `GET` | `/api/experiments` | List all registered research experiments |
| `POST` | `/api/experiments` | Register a new research experiment record |
| `GET` | `/api/experiments/{id}` | Retrieve experiment metadata, configuration, and metrics |
| `POST` | `/api/experiments/{id}/run` | Execute experiment across bound snapshot and compute provenance |
| `GET` | `/api/experiments/{id}/metrics` | Fetch descriptive research metrics for experiment |
| `GET` | `/api/experiments/{id}/provenance` | Fetch 11-stage provenance pipeline audit trail |
| `GET` | `/api/experiments/{id}/validation` | Run automated invariant validation against experiment |
| `POST` | `/api/experiments/{id}/archive` | Archive completed experiment and lock state permanently |
| `POST` | `/api/experiments/compare` | Compare 2+ experiments side-by-side (config diffs & metrics) |
| `GET` | `/api/experiments/{id}/export` | Export sanitized JSON or formatted text experiment package |

---

## 5. Automated Verification

- **Unit & Integration Test Suite**: `tests/test_phase_1_5_experiments.py` (31 tests covering all Phase 1.5 capabilities).
- **20-Stage E2E Pipeline**: `backend/run_e2e_phase_1_5.py` (Full automated end-to-end execution of snapshots, experiments, provenance, comparison, exports, and byte immutability verification).
