# Troubleshooting Guide

This guide provides solutions for common issues when setting up and running the **Explainable Radiology Report Generation** research prototype.

---

## 1. Environment & Python Setup

### Problem 1: `python` command is not recognized or wrong version is used
- **Symptom**: `'python' is not recognized as an internal or external command` or `Python 3.8 / 3.9 detected`.
- **Cause**: Python is not on your system `PATH`, or an incompatible Python version is active.
- **Solution**:
  - Install **Python 3.11** (or 3.10+) from [python.org](https://www.python.org/).
  - Ensure **"Add Python to PATH"** was checked during installation.
  - On Windows, verify with `py -3.11 --version` or `python --version`.

---

### Problem 2: PowerShell script execution error on activating `venv`
- **Symptom**: `backend\venv\Scripts\Activate.ps1 cannot be loaded because running scripts is disabled on this system.`
- **Cause**: Windows PowerShell execution policy restricts running unsigned scripts by default.
- **Solution**:
  - Run the following in your current PowerShell terminal:
    ```powershell
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    backend\venv\Scripts\activate
    ```
  - Alternatively, use the standard Command Prompt (`cmd.exe`):
    ```cmd
    backend\venv\Scripts\activate.bat
    ```

---

## 2. Dependencies & PyTorch

### Problem 3: PyTorch installation fails or CUDA is not detected
- **Symptom**: `torch.cuda.is_available()` returns `False` despite having an NVIDIA GPU.
- **Solution**:
  - **CPU-Only Setup (Default / Compatible)**:
    ```bash
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install -r requirements.txt
    ```
  - **GPU Setup (CUDA 11.8 / 12.1)**:
    ```bash
    # For CUDA 12.1
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt
    ```
  - **Note**: The entire codebase automatically falls back to CPU if CUDA is unavailable.

---

### Problem 4: TorchXRayVision model weight download failure
- **Symptom**: `Failed to load TorchXRayVision model: HTTP Error` or timeout during first run.
- **Cause**: TorchXRayVision downloads pre-trained DenseNet-121 weights on first instantiation. A firewall or interrupted connection may block the download.
- **Solution**:
  - Ensure an active internet connection on the first run.
  - Check the local weights cache directory:
    - **Windows**: `C:\Users\<User>\.torchxrayvision\models_data`
    - **Linux/macOS**: `~/.torchxrayvision/models_data`
  - If a download was interrupted, delete any partial `.pt` files in that directory and rerun the test command:
    ```bash
    backend\venv\Scripts\python backend/test_vision.py
    ```

---

## 3. Dataset & Image Discovery

### Problem 5: No studies found or `data/iu_xray/images` is empty
- **Symptom**: `GET /api/studies` returns 0 studies or warnings about missing radiograph images.
- **Cause**: The IU X-Ray dataset has not been downloaded yet.
- **Solution**:
  - Execute the dataset acquisition script:
    ```bash
    backend\venv\Scripts\python scripts/download_iu_xray.py
    ```
  - Verify that PNG images are present in `data/iu_xray/images/` (e.g. `CXR1122_IM-0080-1001-0002.png`).

---

## 4. Server & Web Dashboard

### Problem 6: Port 8000 already in use
- **Symptom**: `OSError: [Errno 98] Address already in use` or `WinError 10048`.
- **Cause**: Another instance of the API server or a different service is bound to port 8000.
- **Solution**:
  - **Windows**: Find and terminate the process using port 8000:
    ```powershell
    netstat -ano | findstr :8000
    taskkill /PID <PID_NUMBER> /F
    ```
  - **Linux/macOS**:
    ```bash
    lsof -i :8000
    kill -9 <PID>
    ```
  - Alternatively, pass a custom port to `api.py`:
    ```bash
    backend\venv\Scripts\python backend/api.py --port 8080
    ```

---

### Problem 7: UI loads but radiograph images or heatmaps are blank
- **Symptom**: Canvas area remains dark or displays "Loading radiograph...".
- **Cause**: Browser cache or CORS issue when opening `index.html` directly from the filesystem (`file:///`).
- **Solution**:
  - **Do NOT** open `frontend/index.html` by double-clicking it.
  - Always start `backend/api.py` and navigate to:
    ```
    http://127.0.0.1:8000/
    ```
  - Hard-refresh your browser (`Ctrl + F5` on Windows / `Cmd + Shift + R` on Mac) to clear stale JavaScript or CSS.

---

## 5. LLM Synthesis & API Keys

### Problem 8: OpenAI provider throws `API key is missing` or rate limits
- **Symptom**: `ValueError: [ERROR] OpenAI API key is missing.`
- **Cause**: `LLM_PROVIDER=openai` is set without specifying `LLM_API_KEY`.
- **Solution**:
  - **For offline, deterministic testing (Recommended)**: Set `LLM_PROVIDER=mock` in your `.env` file or environment:
    ```bash
    set LLM_PROVIDER=mock
    ```
  - **For OpenAI integration**: Provide a valid API key in `.env`:
    ```ini
    LLM_PROVIDER=openai
    LLM_MODEL=gpt-4o-mini
    LLM_API_KEY=your_actual_key_here
    ```

---

## 6. Running Tests

### Problem 9: Unit tests fail on import or path resolution
- **Symptom**: `ModuleNotFoundError: No module named 'diagnostic_qa'`.
- **Cause**: Running `unittest` from inside the `tests/` subdirectory instead of the project root.
- **Solution**:
  - Always execute test commands from the **root directory** of the repository:
    ```bash
    # Windows
    backend\venv\Scripts\python -m unittest discover -s tests -v

    # Linux / macOS
    python -m unittest discover -s tests -v
    ```
