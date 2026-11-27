$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$PythonCmd = $null
$NpmCmd = "npm"

if (Test-Path $VenvPython) {
    $PythonCmd = $VenvPython
} elseif (Get-Command conda -ErrorAction SilentlyContinue) {
    $condaPython = (& conda run -n ai-conveyor python -c "import sys; print(sys.executable)" 2>$null | Select-Object -Last 1)
    if ($condaPython) {
        $condaPython = $condaPython.Trim()
        if (Test-Path $condaPython) {
            $PythonCmd = $condaPython
            $condaDir = Split-Path -Parent $condaPython
            $condaNpm = Join-Path $condaDir "npm.cmd"
            if (Test-Path $condaNpm) {
                $env:PATH = "$condaDir;$env:PATH"
                $NpmCmd = $condaNpm
            }
        }
    }
}
if (-not $PythonCmd) {
    $PythonCmd = "python"
}

Write-Host "[*] Python : $PythonCmd" -ForegroundColor DarkGray
Write-Host "[*] NPM    : $NpmCmd" -ForegroundColor DarkGray

# Jalankan Backend API di window terpisah
Write-Host 'Menjalankan Backend API...' -ForegroundColor Cyan
Start-Process powershell -ArgumentList '-NoExit', '-Command', "Set-Location '$ProjectRoot\backend_api'; & '$PythonCmd' api.py"

Start-Sleep -Seconds 2

# Jalankan Frontend Web Dashboard
Write-Host 'Menjalankan Frontend Web Dashboard...' -ForegroundColor Green
Set-Location "$ProjectRoot\FE"
& $NpmCmd run dev -- --open
