# Contributing Guidelines

Thank you for contributing to the **Explainable Radiology Report Generation** research prototype. This guide outlines the development standards, git branching strategy, safety invariants, and pull request requirements.

> **RESEARCH PROTOTYPE NOTICE**: This project is for academic explainability research. Code contributions must never present model outputs or Grad-CAM heatmaps as definitive clinical diagnoses or certified medical advice.

---

## 1. Safety & Architecture Invariants

Every code contribution must uphold these mandatory safety guarantees:

1. **Zero Ground-Truth Leakage**: Reference XML reports and ground-truth labels must never be passed into model inference contexts, LLM prompt inputs, or client API responses.
2. **Model Activation Score Fidelity**: Vision model outputs are continuous activation floats (e.g. `0.4750`) and must never be converted to percentages or labeled as "clinical confidence" or "probability of disease".
3. **Grad-CAM Attribution Scope**: Grad-CAM maps represent internal convolutional feature attributions; they must not be referred to as lesion segmentations or localization boundaries.
4. **Machine Evidence Immutability**: Clinician reviewer annotations are stored in a separate layer and must never modify underlying machine-generated evidence artifacts.
5. **No Synthetic Fabrication**: Missing views and missing heatmaps must be explicitly indicated as unavailable and never replaced with faked images or synthetic findings.

---

## 2. Git Workflow & Collaboration

### Step 1: Sync with `main`
```bash
git checkout main
git pull origin main
```

### Step 2: Create a Feature Branch
Use standardized branch naming conventions:
- `feature/<short-description>` (e.g., `feature/dual-view-sync`)
- `fix/<issue-description>` (e.g., `fix/opacity-slider-clamp`)
- `docs/<doc-update>` (e.g., `docs/api-update`)
- `test/<test-suite>` (e.g., `test/batch-processor-edge-cases`)

```bash
git checkout -b feature/your-feature-name
```

### Step 3: Implement Changes & Adhere to Coding Standards
- Python code should follow PEP 8 and include descriptive docstrings and type hints.
- Frontend code should use semantic HTML5, CSS custom properties, and standard ES6+ JavaScript.
- Avoid introducing unnecessary heavy external dependencies.

### Step 4: Run Automated Verification
Before committing, ensure all 112 automated unit tests and E2E verification scripts pass:

```bash
# Windows
backend\venv\Scripts\python -m unittest discover -s tests -v
backend\venv\Scripts\python backend/run_e2e_phase_1_1.py

# Linux / macOS
python -m unittest discover -s tests -v
python backend/run_e2e_phase_1_1.py
```

### Step 5: Commit Changes
Use conventional commit prefixes:
- `feat:` New capability or feature
- `fix:` Bug fix or invariant correction
- `docs:` Documentation updates
- `test:` Unit or integration test additions
- `refactor:` Code reorganization without behavioral changes
- `perf:` Performance improvements

```bash
git commit -m "feat(viewer): add synchronized dual-view pan controls"
```

### Step 6: Push & Open a Pull Request
```bash
git push -u origin feature/your-feature-name
```

---

## 3. Pull Request Requirements

When submitting a PR, include:
1. **Summary of Changes**: High-level explanation of what was modified or added.
2. **Rationale**: Architectural motivation or bug explanation.
3. **Testing Performed**: Exact test command outputs and verification logs.
4. **Safety Confirmation**: Explicit affirmation that no ground-truth leakage or safety regressions were introduced.
5. **UI Screenshots / Demos**: For visual frontend modifications.

---

## 4. What NEVER to Commit

- ❌ `.env` files or API keys / secrets
- ❌ Virtual environment directories (`venv/`, `.venv/`)
- ❌ Large raw dataset archives or images (`data/iu_xray/images/`, `*.tgz`)
- ❌ PyTorch checkpoint weights (`*.pt`, `*.pth`, `*.bin`, `*.safetensors`)
- ❌ Python cache files (`__pycache__/`, `*.pyc`)
- ❌ IDE-specific settings (`.vscode/`, `.idea/`)
