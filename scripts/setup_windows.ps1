# ==============================================================================
# Windows Automated Setup Script (PowerShell)
# LLM-Assisted Explainable Radiology Report Generation
# ==============================================================================

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "LLM-Assisted Explainable Radiology — Windows Setup" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan

# 1. Check Python installation
$PythonCmd = "python"
try {
    $PyVersion = & $PythonCmd --version 2>&1
    Write-Host "[1/5] Detected Python: $PyVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python was not found on PATH. Please install Python 3.11 from https://python.org" -ForegroundColor Red
    Exit 1
}

# 2. Setup Virtual Environment
$VenvPath = "backend\venv"
if (-not (Test-Path $VenvPath)) {
    Write-Host "[2/5] Creating Python virtual environment at $VenvPath..." -ForegroundColor Yellow
    & $PythonCmd -m venv $VenvPath
} else {
    Write-Host "[2/5] Virtual environment already exists at $VenvPath." -ForegroundColor Green
}

# 3. Upgrade pip and install dependencies
$PipCmd = "$VenvPath\Scripts\pip.exe"
Write-Host "[3/5] Installing dependencies from requirements.txt..." -ForegroundColor Yellow
& $PipCmd install --upgrade pip
& $PipCmd install -r requirements.txt

# 4. Copy .env.example to .env if not present
if (-not (Test-Path ".env")) {
    Write-Host "[4/5] Creating .env from .env.example..." -ForegroundColor Yellow
    Copy-Item ".env.example" ".env"
} else {
    Write-Host "[4/5] .env configuration already present." -ForegroundColor Green
}

# 5. Run test suite verification
$PythonVenvCmd = "$VenvPath\Scripts\python.exe"
Write-Host "[5/5] Running automated verification test suite..." -ForegroundColor Yellow
& $PythonVenvCmd -m unittest discover -s tests -v

Write-Host "`n=================================================================" -ForegroundColor Cyan
Write-Host "SETUP COMPLETE!" -ForegroundColor Green
Write-Host "To start the application:" -ForegroundColor White
Write-Host "  $PythonVenvCmd backend/api.py" -ForegroundColor Cyan
Write-Host "Then open your browser at: http://127.0.0.1:8000/" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan
