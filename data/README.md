# Dataset Directory Guide: Indiana University Chest X-Ray Collection (IU X-Ray / Open-i)

This directory contains dataset resources, metadata schemas, and baseline validated evaluation fixtures for the **Explainable Radiology Report Generation** research prototype.

> **RESEARCH DISCLAIMER**: Dataset resources and annotations are used solely for academic research and algorithm benchmarking. They do not constitute clinical guidance.

---

## 1. Directory Structure

```
data/
├── README.md                           # This guide
├── .gitkeep                            # Keeps data directory tracked in Git
├── reviewer_annotations/               # Local JSON storage for clinician review metadata
│   └── CXR1122.json                    # Per-study reviewer annotations (ignored in Git)
└── iu_xray/                            # IU X-Ray dataset directory
    ├── images/                         # Raw PNG chest radiographs (7,470 images, ~1.27 GB)
    │   ├── CXR1122_IM-0080-1001-0002.png
    │   └── ...
    ├── reports/                        # Raw XML radiology reports (~3,955 reports, ~1.1 MB)
    │   └── ecgen-radiology/
    │       ├── 1122.xml
    │       └── ...
    ├── grounding/                      # Generated Grad-CAM heatmaps and PNG overlays
    │   ├── CXR1122_..._infiltration_heatmap.png
    │   └── CXR1122_..._infiltration_overlay.png
    ├── dataset_summary.json            # Dataset summary statistics
    ├── corpus_analysis_summary.json    # Pathology frequency and vocabulary distribution
    ├── e2e_qa_validation_result.json   # Validated Phase 0.6 QA baseline artifact
    ├── e2e_evidence_validation_result.json # Validated Phase 0.7 Evidence Layer artifact
    ├── e2e_llm_validation_result.json  # Validated Phase 0.8 LLM synthesis artifact
    ├── e2e_grounding_validation_result.json # Validated Phase 0.9 Grad-CAM artifact
    └── validation_sample.json          # Pipeline sample metadata
```

---

## 2. Dataset Acquisition

The Indiana University Chest X-Ray Collection is an open-access public dataset hosted by the National Library of Medicine (NLM / NIH) Open-i service.

To acquire and extract the dataset automatically, run:

```bash
# Windows
backend\venv\Scripts\python scripts/download_iu_xray.py

# Linux / macOS
python scripts/download_iu_xray.py
```

### Official Archive Sources:
- **Radiology Reports**: `https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz` (~1.1 MB, ~3,955 XML reports)
- **Chest Radiographs**: `https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz` (~1.27 GB, ~7,470 PNG images)

---

## 3. Data Usage Policies & Safety Invariants

1. **Inference vs. Benchmarking Isolation**:
   - **Inference (`data/iu_xray/images/`)**: Only radiograph images are loaded during vision inference and report generation.
   - **Benchmarking (`data/iu_xray/reports/`)**: Reference XML reports are reserved exclusively for offline evaluation, parser validation, and QA ground-truth comparison.
2. **Zero Ground-Truth Leakage Guarantee**:
   - The production pipeline and REST API strictly prohibit passing ground-truth XML content or reference report text into the LLM prompt or client responses.
3. **Git Tracking Policy**:
   - Large raw image archives (`images/`, `reports/`, `archives/`) and generated PNG heatmaps (`grounding/*.png`) are excluded from Git via `.gitignore`.
   - Small JSON schema definitions, baseline validation fixtures (`e2e_*_validation_result.json`), and dataset summaries are tracked in Git to allow unit tests and CI pipelines to execute immediately after cloning.
