# Jalankan Backend API di window terpisah
Write-Host 'Menjalankan Backend API...' -ForegroundColor Cyan
Start-Process powershell -ArgumentList '-NoExit', '-Command', "Set-Location '$PSScriptRoot\backend_api'; python api.py"

Start-Sleep -Seconds 2

# Jalankan Frontend Web Dashboard
Write-Host 'Menjalankan Frontend Web Dashboard...' -ForegroundColor Green
Set-Location "$PSScriptRoot\FE"
npm run dev -- --open
