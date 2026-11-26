<#
.SYNOPSIS
Setup otomatis environment Python + Frontend untuk perpindahan ke PC/Laptop Windows baru.

.DESCRIPTION
Menyiapkan virtualenv lokal (.venv) atau Conda (environment.yml), menginstal PyTorch CUDA
beserta dependensi dari requirements-windows.txt, menyalin .env dari .env.example (jika belum ada),
menginstal dependensi Node/Frontend (FE/), dan memverifikasi import serta kesiapan GPU.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 -UseConda
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 -CheckOnly
#>
[CmdletBinding()]
param(
    [switch]$UseConda,
    [string]$VenvDir = ".venv",
    [switch]$CpuOnly,
    [switch]$SkipFrontend,
    [switch]$ExportTensorRT,
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  SETUP ENVIRONMENT WINDOWS - AI SLAT RIB CONVEYOR SYSTEM" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "[*] Direktori Proyek : $ProjectRoot"

function Invoke-Checked {
    param([string]$FilePath, [string[]]$ArgumentList)
    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "Perintah gagal ($LASTEXITCODE): $FilePath $($ArgumentList -join ' ')"
    }
}

$PythonExe = "python"

if (-not $CheckOnly) {
    if ($UseConda) {
        Write-Host "`n[1/5] Menyiapkan Conda environment 'ai-conveyor'..." -ForegroundColor Yellow
        if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
            throw "Conda tidak ditemukan di PATH. Jalankan tanpa -UseConda untuk memakai .venv."
        }
        $envList = (& conda env list | Out-String)
        if ($envList -match "(?m)^ai-conveyor\s") {
            Invoke-Checked "conda" @("env", "update", "-n", "ai-conveyor", "-f", "environment.yml", "--prune")
        } else {
            Invoke-Checked "conda" @("env", "create", "-f", "environment.yml")
        }
        $condaPython = (& conda run -n ai-conveyor python -c "import sys; print(sys.executable)" | Select-Object -Last 1).Trim()
        if ($condaPython -and (Test-Path $condaPython)) {
            $PythonExe = $condaPython
        }
    } else {
        Write-Host "`n[1/5] Menyiapkan Python Virtual Environment ($VenvDir)..." -ForegroundColor Yellow
        $venvPython = Join-Path $ProjectRoot "$VenvDir\Scripts\python.exe"
        if (-not (Test-Path $venvPython)) {
            Invoke-Checked "python" @("-m", "venv", $VenvDir)
        }
        $PythonExe = $venvPython

        Write-Host "`n[2/5] Memperbarui pip & menginstal dependensi Python..." -ForegroundColor Yellow
        Invoke-Checked $PythonExe @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")
        if ($CpuOnly) {
            Invoke-Checked $PythonExe @("-m", "pip", "install", "torch", "torchvision")
        } else {
            Invoke-Checked $PythonExe @("-m", "pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu124")
        }
        Invoke-Checked $PythonExe @("-m", "pip", "install", "-r", "requirements-windows.txt")
    }

    Write-Host "`n[3/5] Memeriksa file .env lokal..." -ForegroundColor Yellow
    if (-not (Test-Path ".env") -and (Test-Path ".env.example")) {
        Copy-Item ".env.example" ".env"
        Write-Host "    [OK] .env dibuat dari .env.example. Sesuaikan source kamera/video Anda." -ForegroundColor Green
    } else {
        Write-Host "    [OK] .env sudah tersedia (tidak ditimpa)." -ForegroundColor Green
    }

    if (-not $SkipFrontend) {
        Write-Host "`n[4/5] Menyiapkan Frontend Dashboard (FE/)..." -ForegroundColor Yellow
        if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
            Write-Host "    [WARN] npm/Node.js tidak ditemukan di PATH. Lewati setup FE." -ForegroundColor DarkYellow
        } else {
            Push-Location (Join-Path $ProjectRoot "FE")
            try {
                if (Test-Path "package-lock.json") {
                    Invoke-Checked "npm" @("ci")
                } else {
                    Invoke-Checked "npm" @("install")
                }
                Invoke-Checked "npm" @("run", "build")
            } finally {
                Pop-Location
            }
        }
    }

    if ($ExportTensorRT) {
        Write-Host "`n[*] Mengekspor model TensorRT (.engine) untuk GPU device ini..." -ForegroundColor Yellow
        Invoke-Checked $PythonExe @("scripts\export_tensorrt.py")
    }
} else {
    $venvCandidate = Join-Path $ProjectRoot "$VenvDir\Scripts\python.exe"
    if (Test-Path $venvCandidate) {
        $PythonExe = $venvCandidate
    }
    Write-Host "[*] Mode -CheckOnly aktif (memakai Python: $PythonExe)" -ForegroundColor Yellow
}

Write-Host "`n[5/5] Verifikasi modul Python, CUDA, dan WebSocket..." -ForegroundColor Yellow
$verifyCode = @'
import importlib.metadata as md
import sys
print("Python       :", sys.version.split()[0])
for name in ("numpy", "cv2", "ultralytics", "websockets", "flask", "ws_stream_server"):
    mod = __import__(name)
    try:
        ver = md.version(name)
    except Exception:
        ver = getattr(mod, "__version__", "module")
    print(f"{name:13s}: OK ({ver})")
import torch
cuda_ok = torch.cuda.is_available()
dev = torch.cuda.get_device_name(0) if cuda_ok else "CPU only"
print(f"torch        : {torch.__version__} | CUDA={cuda_ok} ({dev})")
'@
$verifyCode | & $PythonExe -
if ($LASTEXITCODE -ne 0) {
    throw "Verifikasi environment Python gagal ($LASTEXITCODE)."
}

Write-Host "`n======================================================================" -ForegroundColor Green
Write-Host "  ENVIRONMENT SIAP DIGUNAKAN!" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
Write-Host "Langkah berikutnya di device baru:"
Write-Host "  1. Pastikan .env sudah sesuai kamera/video lokal"
Write-Host "  2. Jika GPU berbeda, export ulang TensorRT: $PythonExe scripts\export_tensorrt.py"
Write-Host "  3. Jalankan Web Mode: $PythonExe run_multi_camera_benchmark.py --mode web"
