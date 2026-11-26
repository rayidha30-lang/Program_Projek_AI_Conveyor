$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$PythonCmd = if (Test-Path $VenvPython) { $VenvPython } else { "python" }

# Jalankan Backend API di window terpisah
Write-Host 'Menjalankan Backend API...' -ForegroundColor Cyan
Start-Process powershell -ArgumentList '-NoExit', '-Command', "Set-Location '$ProjectRoot\backend_api'; & '$PythonCmd' api.py"

Start-Sleep -Seconds 2

# Jalankan Frontend Web Dashboard
Write-Host 'Menjalankan Frontend Web Dashboard...' -ForegroundColor Green
Set-Location "$ProjectRoot\FE"
npm run dev -- --open
