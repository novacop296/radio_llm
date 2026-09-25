# Explainable Radiology Report Generation Using Diagnostic Questioning

A modular research prototype exploring explainable chest X-ray report generation through intermediate diagnostic questioning, visual feature attribution (Grad-CAM), immutable evidence representations, and synchronized multi-view clinical dashboards.

---

> ### ⚠️ RESEARCH PROTOTYPE — NOT FOR CLINICAL DIAGNOSIS
> This software is an experimental research prototype intended solely for academic evaluation and explainability research.
> - **Model activation scores are continuous mathematical activations**, NOT clinical probabilities or disease likelihoods.
> - **Grad-CAM visual grounding maps highlight convolutional neural network feature attention**, NOT validated lesion boundaries or anatomical segmentations.
> - **Generated reports require review and validation by a licensed radiologist/physician**.
> - Do not use this system for real-world diagnostic decision-making or patient management.

---

## 1. Features & Capabilities

- **Vision Backbone**: Pre-trained TorchXRayVision DenseNet-121 (`densenet121-res224-all`) generating 18 pathology activation scores and a 1024-dimensional visual feature vector.
- **Diagnostic QA Engine**: Hierarchical questioning (Level 1 presence, Level 2 location/severity) designed to eliminate over-certainty on borderline model activations.
- **Immutable Evidence Layer**: Intermediate representation preserving explicit uncertainty (`supported`, `possible`, `uncertain`, `absent`).
- **Structured LLM Report Generation**: Synthesizes structured **FINDINGS** and **IMPRESSION** sections strictly constrained by the sanitized Evidence Layer.
- **Grad-CAM Visual Grounding**: Generates finding-specific spatial attribution heatmaps from layer `model.features.norm5` ($7 \times 7$ feature activation maps) and alpha-blended overlays ($0\text{--}100\%$ opacity).
- **Multi-Study Browsing**: Dynamic dataset discovery indexing 1,114 IU X-Ray studies with live search, view filtering, and validation status indicators.
- **Dual-View Radiograph Viewing**: Side-by-side radiograph inspection for studies containing multiple projections (`Frontal` and `Lateral`).
- **Synchronized Viewport Interaction**: Optional synchronized pan, zoom, fit-view, opacity, and finding attribution across both viewers.
- **Research Reviewer Annotations**: Clinician annotation layer (`confirmed_present`, `confirmed_absent`, `uncertain`, `needs_review`) with persistent JSON storage (`data/reviews/`) strictly separated from immutable machine evidence.
- **Human-in-the-Loop Review & Finalization (Phase 1.2)**: Comprehensive review session manager with interactive finding decisions, diagnostic QA corrections, structured report drafting, reset to machine baseline, pre-finalization validation, immutable finalization locking, and append-only audit trail logging.
- **Multi-Reviewer Consensus & Adjudication (Phase 1.3)**: Isolated multi-reviewer evaluation sessions (`data/reviews/{study_id}_{reviewer_id}.json`), deterministic agreement aggregation (`unanimous`, `majority`, `adjudication_required`), inter-rater reliability synthesis (Average Agreement Ratio, Cohen's Kappa for $N=2$, Fleiss' Kappa for $N \ge 3$), binding dispute adjudication with mandatory clinical justification, and finalized consensus reporting.
- **Multi-Study Dataset Management & Review Queue (Phase 1.4)**: Dynamic multi-study discovery, 6-state lifecycle tracking (`UNREVIEWED`, `IN_REVIEW`, `AWAITING_CONSENSUS`, `ADJUDICATION_REQUIRED`, `READY_FOR_FINALIZATION`, `FINALIZED`), deterministic review queue filtering/sorting without subjective clinical ranking, dataset evaluation analytics, non-evaluative reviewer workflow activity monitoring, and 8-stage study provenance audit trail.
- **Research Evaluation & Experiment Tracking (Phase 1.5)**: Immutable dataset snapshots with SHA-256 manifest hashing, deterministic configuration fingerprinting (excluding secrets/keys), experiment lifecycle management (`CREATED` $\to$ `RUNNING` $\to$ `COMPLETED` $\to$ `ARCHIVED`), descriptive research metrics (review coverage, consensus coverage, machine-reviewer agreement, Cohen's & Fleiss' Kappa, explainability coverage, human corrections), 11-stage provenance audit chain, side-by-side experiment comparison with configuration difference detection, and full JSON/text export.
- **Research Evaluation Dashboard & Statistical Analysis (Phase 1.6)**: Isolated evaluation dataset management (`data/evaluation_dataset/`), formal evaluation schemas (`docs/evaluation_schema.json`), evaluation lifecycle (`CREATED` $\to$ `PREPARING` $\to$ `RUNNING` $\to$ `COMPLETED` $\to$ `VALIDATED` $\to$ `ARCHIVED`), classification benchmarks (Accuracy, Precision, Recall, F1, Specificity, Sensitivity, Balanced Accuracy), inter-rater agreement (Observed Agreement, Cohen's & Fleiss' Kappa), confusion matrix statistics, summary distribution metrics (mean, median, std, quartiles, IQR), 95% bootstrap confidence intervals (labeled strictly as statistical uncertainty intervals), finding-level error analysis, neutral experiment comparison, structured evaluation report generation (JSON/Text), machine artifact SHA-256 byte immutability, and zero ground-truth report leakage protection.
- **Reproducible Experiment Registry & Model Versioning (Phase 1.7)**: Cryptographic model registry (`data/model_registry/`), dataset versioning manager (`data/dataset_versions/`), 8-state experiment registry lifecycle (`REGISTERED` $\to$ `CONFIGURED` $\to$ `READY` $\to$ `RUNNING` $\to$ `COMPLETED` $\to$ `VALIDATED` $\to$ `FINALIZED` $\to$ `ARCHIVED`), deterministic configuration fingerprinting (SHA-256, excluding secrets/keys), immutable finalized snapshots (`data/experiment_snapshots/`), longitudinal history event timelines, non-evaluative multi-experiment comparison (safe zero-division absolute/relative deltas), and zero ground-truth leakage.
- **Machine vs. Reviewer Comparison**: Live juxtaposition highlighting agreement/divergence between model suggestions and clinician reviews.
- **Batch Processing Runner**: Background study processing manager tracking execution across all 6 pipeline stages.
- **Report & Review Export**: Structured export in JSON and formatted plain-text formats with complete research disclaimers.
- **Strict Invariant Validation**: Automated verification guaranteeing schema compliance, evidence traceability, machine artifact immutability, and **zero ground-truth report leakage**.

---

## 2. Nine-Stage Pipeline & Evaluation Architecture

```
Stage 1: Vision Backbone (DenseNet-121 Activation Scores & 1024-D Visual Features)
   ↓
Stage 2: Diagnostic QA Engine (Hierarchical Level 1 & Level 2 Question Resolution)
   ↓
Stage 3: Evidence Layer (Immutable Grounded Candidate Evidence Package)
   ↓
Stage 4: Visual Grounding (Grad-CAM Spatial Feature Attribution on Layer norm5)
   ↓
Stage 5: LLM Report Generation (Evidence-Sanitized Structured Findings & Impression)
   ↓
Stage 6: Individual Human Review (Interactive Finding Decisions, QA Corrections, Audit Trail)
   ↓
Stage 7: Multi-Reviewer Consensus & Adjudication (Inter-Rater Reliability, Adjudication, Consensus Finalization)
   ↓
Stage 8: Research Evaluation & Statistical Analysis (Dataset Isolation, Benchmarks, Bootstrap CI, Neutral Comparison)
   ↓
Stage 9: Experiment Registry & Versioning (Model Registry, Dataset Versions, Longitudinal Tracking, Immutable Snapshots)
```

```
+---------------------------------------------------------------------------------------------+
|                                    IU X-RAY DATASET                                         |
|                                (1,114 Discovered Studies)                                   |
+----------------------------------------------+----------------------------------------------+
                                               |
                                               v
+---------------------------------------------------------------------------------------------+
|                                  DENSENET-121 BACKBONE                                      |
|               18 Pathology Activation Scores   |   1024-D Visual Feature Vector             |
+----------------------------------------------+----------------------------------------------+
                                               |
                                               v
+---------------------------------------------------------------------------------------------+
|                                DIAGNOSTIC QA ENGINE                                         |
|             Level 1: Presence Questioning  |  Level 2: Location/Severity Questioning        |
+----------------------------------------------+----------------------------------------------+
                                               |
                                               v
+---------------------------------------------------------------------------------------------+
|                                IMMUTABLE EVIDENCE LAYER                                     |
|               (Supported | Possible | Uncertain | Absent Candidate Findings)                |
+--------------------------------------+------------------------------------------------------+
                                       |
                   +-------------------+-------------------+
                   |                                       |
                   v                                       v
+-------------------------------------+ +-----------------------------------------------------+
|        GRAD-CAM GROUNDING           | |                LLM REPORT GENERATOR                 |
| (model.features.norm5 Feature Maps) | |         (Structured FINDINGS + IMPRESSION)          |
+-------------------------------------+ +-----------------------------------------------------+
                                       |
                                       v
+---------------------------------------------------------------------------------------------+
|                    PHASE 1.2: INDIVIDUAL CLINICAL REVIEW SESSIONS                           |
|       (Isolated per reviewer: data/reviews/{study_id}_{reviewer_id}.json)                   |
+----------------------------------------------+----------------------------------------------+
                                               |
                                               v
+---------------------------------------------------------------------------------------------+
|               PHASE 1.3: MULTI-REVIEWER CONSENSUS & ADJUDICATION                            |
|       - Deterministic Consensus Rules: Unanimous, Majority, Adjudication Required          |
|       - Inter-Rater Reliability Metrics: Average Agreement, Cohen's & Fleiss' Kappa         |
|       - Mandatory Clinical Dispute Adjudication Rationales                                  |
+----------------------------------------------+----------------------------------------------+
                                               |
                                               v
+---------------------------------------------------------------------------------------------+
|           PHASE 1.4: DATASET MANAGEMENT, REVIEW QUEUE & STUDY PROVENANCE                    |
|       - 6-State Lifecycle Tracking & Deterministic Review Queue Filtering                   |
|       - 8-Stage Study Provenance Pipeline & Audit Trail Verification                        |
+----------------------------------------------+----------------------------------------------+
                                               |
                                               v
+---------------------------------------------------------------------------------------------+
|            PHASE 1.6: RESEARCH EVALUATION & BENCHMARKING DASHBOARD                          |
|       - Isolated Permitted Reference Annotations (data/evaluation_dataset/)                 |
|       - Classification Metrics (Accuracy, Precision, Recall, F1, Specificity, Balanced Acc) |
|       - 95% Bootstrap Uncertainty Intervals & Finding-Level Error Analysis                  |
+----------------------------------------------+----------------------------------------------+
                                               |
                                               v
+---------------------------------------------------------------------------------------------+
|            PHASE 1.7: EXPERIMENT REGISTRY, MODEL VERSIONING & SNAPSHOTS                     |
|       - Cryptographic Model Registry & Weight Digests (data/model_registry/)                |
|       - Versioned Study Cohorts & Manifest Hashes (data/dataset_versions/)                  |
|       - Deterministic Experiment Fingerprinting & 8-State Lifecycle (data/experiments/)     |
|       - Immutable Snapshots & Longitudinal Event History (data/experiment_snapshots/)       |
+----------------------------------------------+----------------------------------------------+
                                               |
                                               v
+---------------------------------------------------------------------------------------------+
|                                SAFETY & SCHEMA VALIDATOR                                    |
|         (Traceability Verification | Invariant Checks | Zero Ground-Truth Leakage)          |
+----------------------------------------------+----------------------------------------------+
                                               |
                                               v
+---------------------------------------------------------------------------------------------+
|                                REST API (backend/api.py)                                    |
|          Static Server | Study Index | Dual-View | Grounding | Batch | Export               |
+----------------------------------------------+----------------------------------------------+
                                               |
                                               v
+---------------------------------------------------------------------------------------------+
|                              WEB DASHBOARD (frontend/)                                      |
|  Multi-Study Search | Dual-View Viewports | Sync Views | Attributions | Report Review       |
+----------------------------------------------+----------------------------------------------+
                                               ^
                                               | (Independent Persistent Layer)
+----------------------------------------------+----------------------------------------------+
|                         RESEARCH REVIEWER ANNOTATIONS                                       |
|               (Stored separately in data/reviewer_annotations/{study_id}.json)              |
+---------------------------------------------------------------------------------------------+
```

---

## 3. Technology Stack

- **Deep Learning Vision Model**: PyTorch, TorchXRayVision (DenseNet-121 pre-trained on chest radiographs)
- **Grad-CAM Grounding**: Custom PyTorch hook implementation on `model.features.norm5` with Bilinear Interpolation & Jet Colormap Synthesis
- **Diagnostic Questioning**: Rule-driven, threshold-grounded hierarchical question tree
- **Language Model**: Pluggable architecture supporting zero-cost deterministic offline Mock LLM (default) or OpenAI API (`gpt-4o-mini`)
- **Backend API**: Python standard library `http.server` (Zero heavy web-framework dependencies)
- **Frontend Dashboard**: Vanilla HTML5, modern CSS3 (glassmorphism design system, CSS grid/flexbox), Vanilla ES6+ JavaScript (zero external CDN or bundle dependencies)
- **Testing & Verification**: Python standard library `unittest` (267 tests across 11 test suites) and multi-stage end-to-end verification pipelines

---

## 4. Architectural Boundaries & Data Isolation

Strict storage and permission isolation is maintained across all phases:

```
data/
├── iu_xray/                 # Phase 0.5-1.0: Machine Evidence (READ-ONLY, SHA-256 IMMUTABLE)
│   ├── images/              # Radiographs (.png)
│   ├── reports/             # XML Reference reports (OFFLINE ISOLATED ONLY)
│   └── grounding/           # Grad-CAM attribution heatmaps
├── reviews/                 # Phase 1.2: Human Review Sessions (data/reviews/{study_id}_{reviewer_id}.json)
├── consensus/               # Phase 1.3: Consensus & Adjudication (data/consensus/{study_id}_consensus.json)
├── snapshots/               # Phase 1.5: Immutable Dataset Snapshots (manifest.json with SHA-256)
├── experiments/             # Phase 1.5: Experiment Records & Provenance (metadata.json, metrics.json)
├── evaluation_dataset/      # Phase 1.6: Isolated Evaluation Dataset Manifests (manifest.json)
└── evaluations/             # Phase 1.6: Evaluation Runs & Reports (metadata.json, metrics.json, report.json)
```

---

## 5. System Requirements

- **Operating System**: Windows 10/11, Linux (Ubuntu 20.04+), or macOS (12+)
- **Python**: `Python 3.11` (tested with 3.11.9; compatible with 3.10+)
- **Git**: Git 2.30+
- **Hardware**:
  - **CPU**: Compatible with any modern x86_64 / ARM64 processor (Default mode)
  - **GPU (Optional)**: NVIDIA GPU with CUDA 11.8 or 12.1 for accelerated inference
- **Disk Space**: ~1.5 GB if downloading the full IU X-Ray radiograph collection

---

## 6. Quick Start & Local Installation

### 6.1 Clone Repository
```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd radio-llm
```

### 6.2 Windows Setup

1. **Create and activate a virtual environment**:
   ```powershell
   python -m venv backend\venv
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   backend\venv\Scripts\activate
   ```
2. **Install dependencies**:
   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. **Configure environment**:
   ```powershell
   copy .env.example .env
   ```
*(Optional: Run the automated setup script `powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1`)*

---

### 6.3 Linux / macOS Setup

1. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv backend/venv
   source backend/venv/bin/activate
   ```
2. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. **Configure environment**:
   ```bash
   cp .env.example .env
   ```
*(Optional: Run the automated setup script `bash scripts/setup_linux.sh`)*

---

## 7. Dataset Setup

The project integrates with the open-access **Indiana University Chest X-Ray Collection (Open-i / NLM)**.

To download and extract the dataset automatically into `data/iu_xray/`:

```bash
# Windows
backend\venv\Scripts\python scripts/download_iu_xray.py

# Linux / macOS
python scripts/download_iu_xray.py
```

### Dataset Structure
```
data/
└── iu_xray/
    ├── images/      # PNG chest X-rays (7,470 images, ~1.27 GB) [Excluded from Git]
    ├── reports/     # XML reference reports (Offline benchmarking only) [Excluded from Git]
    ├── grounding/   # Generated Grad-CAM attribution heatmaps [Excluded from Git]
    └── *.json       # Tracked baseline test fixtures and schema summaries
```

> **Data Policy**: Raw image archives and ground-truth XML files are excluded from Git via `.gitignore`. Baseline JSON fixtures for `CXR1122` are tracked, enabling tests to run immediately without a full dataset download.

---

## 8. Model Weights

The DenseNet-121 model weights (`densenet121-res224-all`) are provided by `torchxrayvision` and **downloaded automatically on first use** to your local cache:
- **Windows**: `C:\Users\<User>\.torchxrayvision\models_data`
- **Linux/macOS**: `~/.torchxrayvision/models_data`

No manual checkpoint downloads or weights placement are required.

---

## 9. Running the Web Application

Start the unified backend server (which serves both the REST API and the static web frontend):

```bash
# Windows
backend\venv\Scripts\python backend/api.py

# Linux / macOS
python backend/api.py
```

Open your browser and navigate to:
```
http://127.0.0.1:8000/
```

- **Frontend UI**: `http://127.0.0.1:8000/`
- **API Endpoints**: `http://127.0.0.1:8000/api/...`

---

## 10. Verification & Testing

### 10.1 Run the Full Test Suite
The automated test suite runs via Python's standard `unittest` framework:

```bash
# Windows
backend\venv\Scripts\python -m unittest discover -s tests -v

# Linux / macOS
python -m unittest discover -s tests -v
```

**Expected Result**:
```
Ran 267 tests in ~35s
OK
```

### Test Suite Breakdown:
| Test Module | Phase Covered | Tests | Status |
| :--- | :--- | :--- | :--- |
| `test_phase_0_6_qa.py` | Phase 0.6 Diagnostic QA Engine | 20 | ✅ PASS |
| `test_phase_0_7_evidence.py` | Phase 0.7 Evidence Layer & Isolation | 20 | ✅ PASS |
| `test_phase_0_8_llm.py` | Phase 0.8 LLM Integration & Schema | 20 | ✅ PASS |
| `test_phase_0_9_grounding.py` | Phase 0.9 Grad-CAM Visual Grounding | 21 | ✅ PASS |
| `test_phase_1_0_api.py` | Phase 1.0 Web API & Safety Badges | 22 | ✅ PASS |
| `test_phase_1_1_multistudy.py` | Phase 1.1 Multi-Study, Dual-View & Reviewer | 20 | ✅ PASS |
| `test_phase_1_2_review.py` | Phase 1.2 Human-in-the-Loop Review & Finalization | 23 | ✅ PASS |
| `test_phase_1_3_consensus.py` | Phase 1.3 Multi-Reviewer Consensus & Adjudication | 33 | ✅ PASS |
| `test_phase_1_4_dataset.py` | Phase 1.4 Dataset Management, Queue & Analytics | 22 | ✅ PASS |
| `test_phase_1_5_experiments.py` | Phase 1.5 Research Evaluation & Experiment Tracking | 31 | ✅ PASS |
| `test_phase_1_6_evaluation.py` | Phase 1.6 Research Evaluation, Benchmarking & Stats | 35 | ✅ PASS |
| **Total** | | **267** | **267 / 267 PASS (100%)** |

---

### 10.2 Run End-to-End Verification
To verify the complete test suite and end-to-end pipelines:

```bash
# Run all unit and regression tests (Phases 0.6 - 1.6: 267 tests)
backend\venv\Scripts\python -m unittest discover -s tests -v

# Run Phase 1.6 End-to-End Evaluation & Statistical Analysis Pipeline (26 verification stages)
backend\venv\Scripts\python backend/run_e2e_phase_1_6.py

# Run Phase 1.5 End-to-End Research Evaluation Pipeline (20 verification stages)
backend\venv\Scripts\python backend/run_e2e_phase_1_5.py

# Run Phase 1.4 End-to-End Dataset Management Pipeline (16 verification stages)
backend\venv\Scripts\python backend/run_e2e_phase_1_4.py

# Run Phase 1.3 End-to-End Multi-Reviewer Consensus Pipeline (9 verification stages)
backend\venv\Scripts\python backend/run_e2e_consensus_pipeline.py

# Run Phase 1.2 End-to-End Human Review Pipeline (16 verification steps)
backend\venv\Scripts\python backend/run_e2e_review_pipeline.py

# Run Phase 1.1 End-to-End Multi-Study Verification Pipeline (14 verification steps)
backend\venv\Scripts\python backend/run_e2e_phase_1_1.py
```

---

## 11. CPU & GPU Configuration

The system is configured with automatic device detection:
- **CUDA GPU**: Detected automatically if compatible NVIDIA drivers and PyTorch CUDA packages are present.
- **CPU Fallback**: Used automatically if CUDA is not available.

To force CPU-only execution, set the environment variable:
```bash
set CUDA_VISIBLE_DEVICES=
```

---

## 12. LLM Provider Modes

Configure the language model backend via `.env` or environment variables:

1. **`LLM_PROVIDER=mock` (Default / Recommended)**:
   - Zero-cost, 100% offline, deterministic report synthesis.
   - Requires no API keys.
   - Enforces structured radiology output schema.
2. **`LLM_PROVIDER=openai` (Optional)**:
   - Integrates with OpenAI chat completion models (e.g. `gpt-4o-mini`).
   - Requires setting `LLM_API_KEY=your_key_here`.

---

## 13. Documentation Links

- [`docs/batch_processing.md`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/docs/batch_processing.md): Multi-Study Batch Processing Architecture, 6-Stage Pipeline Lifecycle, REST API endpoints, Race-Condition Protection, and Artifact Persistence.
- [`docs/phase_2_0_counterfactual_explainability.md`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/docs/phase_2_0_counterfactual_explainability.md): Phase 2.0 Interactive Counterfactual Explanations, Perturbation Engine, Synchronized Viewer, Multi-Level Impact Analysis, Control Comparisons, and Invariant Verification.
- [`docs/counterfactual_schema.json`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/docs/counterfactual_schema.json): Formal JSON schema definition for Counterfactual Experiments, Perturbations, Inference Deltas, Attribution Maps, QA/Report Impact, and Reproducibility Manifests.
- [`docs/phase_1_9_research_experiments.md`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/docs/phase_1_9_research_experiments.md): Phase 1.9 Research Experiment Orchestration, Statistical Summaries, Bootstrap Confidence Intervals, Neutral Comparison, and 16-Section Reports.
- [`docs/phase_1_8_external_benchmarking.md`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/docs/phase_1_8_external_benchmarking.md): Phase 1.8 External Benchmarking, Multi-Modal Evaluation, and Air-Gapped Portable Bundles.
- [`docs/phase_1_7_experiment_registry.md`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/docs/phase_1_7_experiment_registry.md): Phase 1.7 Experiment Registry, Cryptographic Model Registry, Dataset Versioning, and Longitudinal History.
- [`docs/phase_1_6_research_evaluation.md`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/docs/phase_1_6_research_evaluation.md): Phase 1.6 Research Evaluation Dashboard, Benchmarking, Bootstrap Statistical Summaries, Error Analysis, Neutral Experiment Comparison, and Immutability.
- [`docs/evaluation_schema.json`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/docs/evaluation_schema.json): Formal JSON schema definition for Evaluation Runs, Evaluation Datasets, Metric Results, Finding Evaluations, Agreement Evaluations, Error Analyses, and Evaluation Reports.
- [`docs/phase_1_5_research_evaluation.md`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/docs/phase_1_5_research_evaluation.md): Phase 1.5 Research Evaluation, Experiment Tracking, Snapshots, Fingerprinting, and Provenance.
- [`docs/experiment_schema.json`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/docs/experiment_schema.json): Formal JSON schema definition for experiments, snapshots, comparisons, and provenance.
- [`docs/phase_1_4_dataset_management.md`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/docs/phase_1_4_dataset_management.md): Phase 1.4 Multi-Study dataset discovery, review queue lifecycle, evaluation analytics, inter-rater reliability, reviewer workflow monitoring, and provenance.
- [`docs/study_schema.json`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/docs/study_schema.json): Formal JSON schema definition for study summaries, review queue items, dataset statistics, and study provenance.
- [`docs/phase_1_3_multi_reviewer_consensus.md`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/docs/phase_1_3_multi_reviewer_consensus.md): Phase 1.3 Multi-Reviewer consensus architecture, inter-rater reliability metrics, adjudication workflows, schemas, and APIs.
- [`docs/consensus_schema.json`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/docs/consensus_schema.json): Formal JSON schema definition for consensus sessions, agreement metrics, and adjudication records.
- [`docs/phase_1_2_human_review.md`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/docs/phase_1_2_human_review.md): Phase 1.2 Human-in-the-Loop review architecture, schema, APIs, immutability model, and validation.
- [`docs/review_schema.json`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/docs/review_schema.json): Formal JSON schema definition for review sessions and audit trails.
- [`docs/API.md`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/docs/API.md): Complete REST API endpoint reference with request/response schemas.
- [`docs/PROJECT_STRUCTURE.md`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/docs/PROJECT_STRUCTURE.md): Comprehensive module and file responsibility guide.
- [`TROUBLESHOOTING.md`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/TROUBLESHOOTING.md): Solutions for setup, environment, port, and dependency errors.
- [`CONTRIBUTING.md`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/CONTRIBUTING.md): Branching strategy, PR guidelines, and safety invariants.
- [`data/README.md`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/data/README.md): Dataset organization and download instructions.

---

## 14. Suggested First Commit (GitHub Preparation)

To push this repository to GitHub for team collaboration:

```bash
# Initialize git in project directory (if not already initialized)
git init

# Verify gitignore rules
git status

# Stage all tracked documentation, code, and fixtures
git add .

# Commit
git commit -m "docs: prepare explainable radiology prototype for GitHub collaboration"

# Set main branch and remote
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL

# Push to GitHub
git push -u origin main
```
