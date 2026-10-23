# Script Perbaikan Kamera Windows 11 & HP Wide Vision HD Camera
# Dijalankan dengan hak Administrator (Elevated)

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " MEMULAI DIAGNOSTIK & PERBAIKAN SERVICE KAMERA WINDOWS " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Hentikan aplikasi yang mungkin mengunci kamera
Write-Host "`n[1] Menutup proses aplikasi kamera yang mungkin menggantung..." -ForegroundColor Yellow
$procList = @("WindowsCamera", "obs64", "obs32", "Teams", "Discord", "Zoom")
foreach ($p in $procList) {
    $proc = Get-Process -Name $p -ErrorAction SilentlyContinue
    if ($proc) {
        Write-Host "  -> Mematikan proses: $p" -ForegroundColor Gray
        Stop-Process -Name $p -Force -ErrorAction SilentlyContinue
    }
}

# 2. Perbaiki Registry Windows Media Foundation FrameServer
# Bug terkenal di Windows 11 24H2: FrameServer crash/looping saat decode MJPG/YUY2 dari webcam laptop
Write-Host "`n[2] Menerapkan Fix Registry EnableFrameServerMode..." -ForegroundColor Yellow
$regPaths = @(
    "HKLM:\SOFTWARE\Microsoft\Windows Media Foundation\Platform",
    "HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows Media Foundation\Platform"
)
foreach ($path in $regPaths) {
    if (-not (Test-Path $path)) {
        New-Item -Path $path -Force | Out-Null
    }
    Set-ItemProperty -Path $path -Name "EnableFrameServerMode" -Value 0 -Type DWord -Force
    Write-Host "  -> Registry set: $path\EnableFrameServerMode = 0" -ForegroundColor Green
}

# 3. Konfigurasi dan Nyalakan Windows Camera Frame Server (FrameServer)
Write-Host "`n[3] Memeriksa & Mengaktifkan Service Windows Camera Frame Server (FrameServer)..." -ForegroundColor Yellow
try {
    Set-Service -Name "FrameServer" -StartupType Automatic
    Stop-Service -Name "FrameServer" -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1
    Start-Service -Name "FrameServer"
    $status = (Get-Service -Name "FrameServer").Status
    Write-Host "  -> Service FrameServer berhasil dijalankan! Status: $status" -ForegroundColor Green
} catch {
    Write-Host "  [!] Gagal mengontrol service FrameServer: $_" -ForegroundColor Red
}

# 4. Reset USB Device HP Wide Vision HD Camera
Write-Host "`n[4] Merestart Hardware Driver Kamera (PnP Reset)..." -ForegroundColor Yellow
$camInstance = "USB\VID_30C9&PID_0069&MI_00\6&2000b69f&0&0000"
try {
    $pnpOut = pnputil /restart-device "$camInstance"
    Write-Host "  -> pnputil output: $pnpOut" -ForegroundColor Green
} catch {
    Write-Host "  [!] pnputil error: $_" -ForegroundColor Red
}

# 5. Reset Windows Camera App Package
Write-Host "`n[5] Me-reset cache aplikasi Windows Camera..." -ForegroundColor Yellow
try {
    Get-AppxPackage *windowscamera* | Reset-AppxPackage -ErrorAction SilentlyContinue
    Write-Host "  -> Reset Windows Camera app selesai." -ForegroundColor Green
} catch {
    Write-Host "  [!] Skip reset appx: $_" -ForegroundColor Gray
}

Write-Host "`n==========================================================" -ForegroundColor Cyan
Write-Host " PERBAIKAN SELESAI DILAKUKAN " -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Cyan
