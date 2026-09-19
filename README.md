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
- **Research Reviewer Annotations**: Clinician annotation layer (`confirmed_present`, `confirmed_absent`, `uncertain`) with persistent JSON storage (`data/reviewer_annotations/`) strictly separated from immutable machine evidence.
- **Machine vs. Reviewer Comparison**: Live juxtaposition highlighting agreement/divergence between model suggestions and clinician reviews.
- **Batch Processing Runner**: Background study processing manager tracking execution across all 6 pipeline stages.
- **Report Export**: Structured export in JSON and formatted plain-text formats with research disclaimers.
- **Strict Invariant Validation**: Automated verification guaranteeing schema compliance, evidence traceability, and **zero ground-truth report leakage**.

---

## 2. Six-Stage Pipeline Architecture

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
Stage 6: Safety & Schema Validation (Traceability Verification & Zero Ground-Truth Leakage)
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
+------------------+------------------+ +--------------------------+--------------------------+
                   |                                               |
                   +-------------------+---------------------------+
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

## 3. System Requirements

- **Operating System**: Windows 10/11, Linux (Ubuntu 20.04+), or macOS (12+)
- **Python**: `Python 3.11` (tested with 3.11.9; compatible with 3.10+)
- **Git**: Git 2.30+
- **Hardware**:
  - **CPU**: Compatible with any modern x86_64 / ARM64 processor (Default mode)
  - **GPU (Optional)**: NVIDIA GPU with CUDA 11.8 or 12.1 for accelerated inference
- **Disk Space**: ~1.5 GB if downloading the full IU X-Ray radiograph collection

---

## 4. Quick Start & Local Installation

### 4.1 Clone Repository
```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd radio-llm
```

### 4.2 Windows Setup

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

### 4.3 Linux / macOS Setup

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

## 5. Dataset Setup

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

## 6. Model Weights

The DenseNet-121 model weights (`densenet121-res224-all`) are provided by `torchxrayvision` and **downloaded automatically on first use** to your local cache:
- **Windows**: `C:\Users\<User>\.torchxrayvision\models_data`
- **Linux/macOS**: `~/.torchxrayvision/models_data`

No manual checkpoint downloads or weights placement are required.

---

## 7. Running the Web Application

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

## 8. Verification & Testing

### 8.1 Run the Full Test Suite
The automated test suite runs via Python's standard `unittest` framework:

```bash
# Windows
backend\venv\Scripts\python -m unittest discover -s tests -v

# Linux / macOS
python -m unittest discover -s tests -v
```

**Expected Result**:
```
Ran 112 tests in ~4.5s
OK
```

### Test Suite Breakdown:
| Test Module | Phase Covered | Tests | Status |
| :--- | :--- | :--- | :--- |
| `test_phase_0_6_qa.py` | Phase 0.6 Diagnostic QA Engine | 20 | ✅ PASS |
| `test_phase_0_7_evidence.py` | Phase 0.7 Evidence Layer & Isolation | 20 | ✅ PASS |
| `test_phase_0_8_llm.py` | Phase 0.8 LLM Integration & Schema | 20 | ✅ PASS |
| `test_phase_0_9_grounding.py` | Phase 0.9 Grad-CAM Visual Grounding | 10 | ✅ PASS |
| `test_phase_1_0_api.py` | Phase 1.0 Web API & Safety Badges | 22 | ✅ PASS |
| `test_phase_1_1_multistudy.py` | Phase 1.1 Multi-Study, Dual-View & Reviewer | 20 | ✅ PASS |
| **Total** | | **112** | **112 / 112 PASS (100%)** |

---

### 8.2 Run End-to-End Verification
To verify the complete 14-step Phase 1.1 pipeline:

```bash
# Windows
backend\venv\Scripts\python backend/run_e2e_phase_1_1.py

# Linux / macOS
python backend/run_e2e_phase_1_1.py
```

---

## 9. CPU & GPU Configuration

The system is configured with automatic device detection:
- **CUDA GPU**: Detected automatically if compatible NVIDIA drivers and PyTorch CUDA packages are present.
- **CPU Fallback**: Used automatically if CUDA is not available.

To force CPU-only execution, set the environment variable:
```bash
set CUDA_VISIBLE_DEVICES=
```

---

## 10. LLM Provider Modes

Configure the language model backend via `.env` or environment variables:

1. **`LLM_PROVIDER=mock` (Default / Recommended)**:
   - Zero-cost, 100% offline, deterministic report synthesis.
   - Requires no API keys.
   - Enforces structured radiology output schema.
2. **`LLM_PROVIDER=openai` (Optional)**:
   - Integrates with OpenAI chat completion models (e.g. `gpt-4o-mini`).
   - Requires setting `LLM_API_KEY=your_key_here`.

---

## 11. Documentation Links

- [`docs/API.md`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/docs/API.md): Complete REST API endpoint reference with request/response schemas.
- [`docs/PROJECT_STRUCTURE.md`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/docs/PROJECT_STRUCTURE.md): Comprehensive module and file responsibility guide.
- [`TROUBLESHOOTING.md`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/TROUBLESHOOTING.md): Solutions for setup, environment, port, and dependency errors.
- [`CONTRIBUTING.md`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/CONTRIBUTING.md): Branching strategy, PR guidelines, and safety invariants.
- [`data/README.md`](file:///c:/Users/ankit/OneDrive/Desktop/radio-llm/data/README.md): Dataset organization and download instructions.

---

## 12. Suggested First Commit (GitHub Preparation)

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
