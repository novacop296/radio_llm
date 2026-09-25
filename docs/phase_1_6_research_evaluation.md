# Phase 1.6: Research Evaluation Dashboard, Benchmarking & Statistical Analysis

## 1. Executive Summary

Phase 1.6 extends the Explainable Radiology Research Prototype into an end-to-end **Research Evaluation, Benchmarking & Statistical Analysis Framework**. It allows researchers to evaluate completed model experiments, compute classification and agreement metrics across isolated evaluation datasets, calculate descriptive metric distributions and 95% bootstrap confidence intervals, inspect per-finding confusion matrices and error breakdowns, and generate reproducible research reports.

> **CRITICAL RESEARCH PROTOTYPE NOTICE**
> 
> This is an academic research prototype for explainability and evaluation research. Machine activation scores, Grad-CAM visual heatmaps, and benchmark metrics are mathematical model outputs and research evaluation measurements, NOT clinical probabilities, diagnostic certainty, or clinical efficacy claims. No medical or clinical decisions should ever be made based on these outputs. The system strictly employs neutral research terminology (e.g., "observed metric", "configuration difference", "observed agreement") without clinical superiority or ranking claims.

---

## 2. Architectural Separation & Multi-Layer Isolation

```
Vision Backbone (DenseNet-121)
         ↓
Diagnostic QA Engine
         ↓
Evidence Layer (Immutable: data/iu_xray/)
         ↓
Grad-CAM Visual Grounding
         ↓
LLM Structured Report
         ↓
Web Review Dashboard
         ↓
Individual Human Review (data/reviews/)
         ↓
Multi-Reviewer Consensus & Adjudication (data/consensus/)
         ↓
Dataset Management & Review Queue (study_manager.py, review_queue.py)
         ↓
Experiment Tracking & Snapshots (data/experiments/, data/snapshots/)
         ↓
Phase 1.6: Research Evaluation & Benchmarking
   ├── Isolated Evaluation Dataset (data/evaluation_dataset/)
   ├── Evaluation Engine (backend/evaluation_engine.py)
   ├── Statistical Analysis & Bootstrapping (backend/statistical_analysis.py)
   ├── Error Analysis & Confusion Matrices (backend/error_analysis.py)
   ├── Evaluation Lifecycle & Immutability (backend/evaluation_manager.py)
   ├── Reproducibility Fingerprint (SHA-256)
   ├── Structured Research Report Generator (backend/evaluation_report.py)
   ├── REST API Extensions (backend/api.py)
   └── Interactive Research Dashboard (frontend/index.html, styles.css, app.js)
```

### Storage Isolation Layers

| Layer | Directory / File | Description | Mutability |
| :--- | :--- | :--- | :--- |
| **Machine Evidence** | `data/iu_xray/` | Raw radiograph images, vision activations, Grad-CAM overlays | Permanently Read-Only (`SHA256_BEFORE == SHA256_AFTER`) |
| **Reviewer Decisions** | `data/reviews/` | Individual clinician review sessions | Mutable until finalized |
| **Consensus & Disputes** | `data/consensus/` | Multi-reviewer consensus sessions, adjudications | Mutable until finalized |
| **Dataset Snapshots** | `data/snapshots/` | Immutable dataset manifests with SHA-256 digests | Permanently Read-Only |
| **Experiment Records** | `data/experiments/` | Model configurations, fingerprints, provenance trails | Immutable once COMPLETED/ARCHIVED |
| **Isolated Evaluation Datasets** | `data/evaluation_dataset/` | Reference annotations and cryptographic manifests | Read-Only |
| **Evaluation Runs** | `data/evaluations/` | Evaluation runs, confusion matrices, statistical summaries | Immutable once COMPLETED/VALIDATED/ARCHIVED |

---

## 3. Metrics & Statistical Methodology

### A. Classification Metrics
- **Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$
- **Precision**: $\frac{TP}{TP + FP}$
- **Recall / Sensitivity**: $\frac{TP}{TP + FN}$
- **Specificity**: $\frac{TN}{TN + FP}$
- **F1 Score**: $\frac{2 \cdot Precision \cdot Recall}{Precision + Recall}$
- **Balanced Accuracy**: $\frac{Sensitivity + Specificity}{2}$

### B. Agreement Metrics
- **Observed Agreement**: Proportion of concordant decisions across raters.
- **Cohen's Kappa ($\kappa$)**: Chance-corrected agreement for 2-rater pairs bounded in $[-1.0, 1.0]$.
- **Fleiss' Kappa ($\kappa$)**: Multi-rater inter-rater reliability for 3+ reviewers bounded in $[-1.0, 1.0]$.

### C. Descriptive Statistics & Bootstrapping
- **Distributions**: Mean, Median, Sample Standard Deviation, Minimum, Maximum, 25th Quartile ($Q_1$), 75th Quartile ($Q_3$), Interquartile Range (IQR).
- **Uncertainty Estimation**: Non-parametric bootstrap percentile method ($B=1000$ resamples) generating 95% statistical uncertainty intervals (clearly distinguished from clinical confidence).
- **Mathematical Robustness**: Bounded floating-point operations guarded against zero denominators, empty collections, NaN, and Infinity.

---

## 4. Evaluation Run Lifecycle

```
CREATED  ──►  PREPARING  ──►  RUNNING  ──►  COMPLETED  ──►  VALIDATED  ──►  ARCHIVED
   │                                           │               │               │
 (mutable)                                (immutable)     (immutable)     (immutable)
```

- **CREATED**: Mutable metadata setup (title, hypothesis, target experiment, evaluation dataset).
- **RUNNING**: Active execution of evaluation engine over isolated reference dataset.
- **COMPLETED**: Benchmark metrics, confusion matrices, and distributions calculated. Permanently locked.
- **VALIDATED**: Invariants (metric bounds, zero secrets, zero ground truth leakage, byte immutability) verified.
- **ARCHIVED**: Research record archived for long-term auditability.

---

## 5. REST API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/evaluations` | List all evaluation runs |
| `GET` | `/api/evaluations/{id}` | Get evaluation details |
| `POST` | `/api/evaluations` | Create a new evaluation run |
| `POST` | `/api/evaluations/{id}/run` | Execute evaluation run |
| `POST` | `/api/evaluations/{id}/validate` | Validate evaluation run |
| `POST` | `/api/evaluations/{id}/finalize` | Finalize evaluation run |
| `POST` | `/api/evaluations/{id}/archive` | Archive evaluation run |
| `GET` | `/api/evaluations/{id}/metrics` | Retrieve aggregate and finding metrics |
| `GET` | `/api/evaluations/{id}/errors` | Retrieve error analysis and confusion matrix |
| `GET` | `/api/evaluations/{id}/statistics` | Retrieve metric distributions and 95% bootstrap CIs |
| `GET` | `/api/evaluations/{id}/comparison?with={other_id}` | Compare evaluation with another run |
| `GET` | `/api/evaluations/{id}/provenance` | Get evaluation lifecycle provenance trail |
| `GET` | `/api/evaluations/{id}/export?format=json\|text` | Export research evaluation report |
| `GET` | `/api/evaluation-datasets` | List isolated evaluation datasets |
| `GET` | `/api/evaluation-datasets/{id}` | Get evaluation dataset metadata (sanitized) |
| `POST` | `/api/evaluation-datasets` | Create isolated evaluation dataset |
| `GET` | `/api/evaluation-dashboard` | Aggregate dashboard statistics |

---

## 6. Verification Matrix

- **Unit & Integration Suite**: 267 / 267 Tests Passing (100% Green).
- **Phase 1.6 26-Stage E2E Verification**: 26 / 26 Stages Passed (100% Green).
- **Legacy Pipelines (1.5, 1.4, 1.3, 1.2, 1.1)**: 100% Passed with zero regressions.
- **Byte-Level Immutability**: All 5,399 files under `data/iu_xray/` confirmed byte-identical (`SHA256_BEFORE == SHA256_AFTER`).
