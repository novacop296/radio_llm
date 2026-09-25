# Phase 1.8: Multi-Modal Experiment Benchmarking & External Test Set Portability

## Overview
Phase 1.8 extends the Explainable Radiology Research Prototype with multi-modal benchmarking capabilities, external dataset support, multi-view CXR analysis, and portable experiment bundle packaging for air-gapped reproducibility verification.

## Core Architectural Components

### 1. External Dataset Registry (`backend/external_dataset_manager.py`)
- **Schema**: `docs/external_dataset_schema.json`
- **Capabilities**:
  - Registers external chest X-ray datasets (e.g. MIMIC-CXR, CheXpert, PadChest subsets).
  - Multi-view pairing and projection normalization (`PA`, `AP`, `LATERAL`, `OBLIQUE`).
  - Strict cohort split tracking (`train`, `val`, `test`, `external_validation`).
  - Immutable manifest generation with SHA-256 integrity verification.
  - Lifecycle state machine: `REGISTERED` $\rightarrow$ `VALIDATING` $\rightarrow$ `VALIDATED` $\rightarrow$ `FINALIZED` $\rightarrow$ `ARCHIVED`.

### 2. Portable Experiment Bundles (`backend/experiment_bundle.py`)
- **Schema**: `docs/experiment_bundle_schema.json`
- **Capabilities**:
  - Bundles experiment configuration, model architecture metadata, weights digest, dataset manifests, evaluation metrics, and standalone reproduction scripts.
  - Self-contained directory layout:
    ```
    bundle_<id>/
    ├── manifest.json
    ├── model/
    ├── dataset/
    ├── experiment/
    ├── evaluation/
    ├── reports/
    ├── provenance/
    ├── scripts/
    │   └── reproduce.py
    └── README.md
    ```
  - Full `.zip` export and archive ingest with schema validation.

### 3. Air-Gapped Verification Runner (`backend/portable_runner.py`)
- **Schema**: `backend/validate_portable_bundle.py`
- **Capabilities**:
  - Validates bundle structure, directory boundaries, relative paths, and zero secret leakage.
  - Computes canonical experiment fingerprints offline without network dependencies.
  - Returns structured verification verdicts: `VERIFIED`, `MISMATCH`, `INCOMPLETE`, `INVALID`.

### 4. Multi-View Benchmarking Engine (`backend/benchmark_manager.py`)
- **Schema**: `docs/benchmark_schema.json`
- **Capabilities**:
  - Supports `SINGLE_VIEW`, `FRONTAL_ONLY`, `LATERAL_ONLY`, and `MULTI_VIEW` representations.
  - Generates classification metrics (AUC, sensitivity, specificity, F1, accuracy) with empirical bootstrap confidence intervals (1,000 iterations).
  - Supports benchmark execution on internal datasets (`IU-Xray`) and external datasets (`MIMIC-CXR`, `CheXpert`).
  - Immutability locking: finalized benchmarks cannot be overwritten or modified.

### 5. Cross-Dataset Performance Comparison (`backend/experiment_comparator.py`)
- **Capabilities**:
  - Compares model performance across distinct datasets and multi-view setups.
  - Computes absolute difference ($\Delta_{\text{abs}}$) and relative difference ($\Delta_{\text{rel}}$).
  - Employs strictly neutral research framing without normative labels ("superior", "best", "winner").

### 6. Phase 1.8 Invariant Validator (`backend/validate_phase_1_8.py`)
- Verifies:
  1. Base IU X-ray dataset integrity (5,399 files unchanged).
  2. Ground-truth XML report isolation (zero `<eFind>` or `<eImpression>` leakage into benchmarks).
  3. Finite metrics (all numbers finite, no NaN / Infinity).
  4. Safe file paths (no absolute path escaping, no directory traversal).
  5. Zero secret or credential leakage.
  6. Immutability of finalized external datasets and benchmarks.

---

## REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/phase18/datasets` | List registered external datasets |
| `POST` | `/api/phase18/datasets` | Register a new external dataset |
| `GET` | `/api/phase18/datasets/<id>` | Retrieve external dataset details |
| `POST` | `/api/phase18/datasets/<id>/validate` | Validate external dataset manifest & view pairs |
| `POST` | `/api/phase18/datasets/<id>/finalize` | Finalize and lock external dataset |
| `GET` | `/api/phase18/benchmarks` | List multi-modal benchmark records |
| `POST` | `/api/phase18/benchmarks/run` | Execute a benchmark evaluation |
| `GET` | `/api/phase18/benchmarks/<id>` | Retrieve benchmark details |
| `POST` | `/api/phase18/benchmarks/<id>/finalize` | Finalize and lock benchmark run |
| `POST` | `/api/phase18/bundles/export` | Export portable experiment bundle |
| `GET` | `/api/phase18/bundles` | List available portable bundles |
| `GET` | `/api/phase18/bundles/<id>` | Retrieve bundle manifest & verification state |
| `POST` | `/api/phase18/bundles/<id>/verify` | Run air-gapped reproducibility verification |
| `POST` | `/api/phase18/benchmarks/compare` | Compare benchmarks across datasets/views |
| `GET` | `/api/phase18/dashboard` | Aggregated dashboard summary statistics |

---

## Verification & Compliance
- **Unit & Integration Tests**: `tests/test_phase_1_8_external_benchmarking.py` (35/35 passing, 337/337 total test suite).
- **E2E Verification**: `backend/run_e2e_phase_1_8.py` (28/28 stages passing).
- **Safety Invariants**: Confirmed 100% compliant with zero ground-truth leakage and full immutability.
