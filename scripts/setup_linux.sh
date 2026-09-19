#!/usr/bin/env bash
# ==============================================================================
# Linux / macOS Automated Setup Script
# LLM-Assisted Explainable Radiology Report Generation
# ==============================================================================

set -e

echo "================================================================="
echo "LLM-Assisted Explainable Radiology — Linux / macOS Setup"
echo "================================================================="

# 1. Check Python
PYTHON_CMD="python3"
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo "[ERROR] python3 could not be found. Please install Python 3.11."
    exit 1
fi
echo "[1/5] Detected Python: $($PYTHON_CMD --version)"

# 2. Virtual Environment
VENV_PATH="backend/venv"
if [ ! -d "$VENV_PATH" ]; then
    echo "[2/5] Creating virtual environment at $VENV_PATH..."
    $PYTHON_CMD -m venv "$VENV_PATH"
else
    echo "[2/5] Virtual environment already exists at $VENV_PATH."
fi

# 3. Dependencies
source "$VENV_PATH/bin/activate"
echo "[3/5] Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Environment file
if [ ! -f ".env" ]; then
    echo "[4/5] Creating .env from .env.example..."
    cp .env.example .env
else
    echo "[4/5] .env configuration already present."
fi

# 5. Run tests
echo "[5/5] Running automated verification test suite..."
python -m unittest discover -s tests -v

echo "================================================================="
echo "SETUP COMPLETE!"
echo "To start the application:"
echo "  source backend/venv/bin/activate"
echo "  python backend/api.py"
echo "Then open: http://127.0.0.1:8000/"
echo "================================================================="
