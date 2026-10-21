@echo off
title Multi-Camera Inspection Benchmark Runner
cd /d "%~dp0"

echo ======================================================================
echo   🚀 BENCHMARK MULTI-KAMERA: TOP (SLAT & RIB) + SIDE (BOLT & PIN)
echo ======================================================================
echo.
echo Pilih mode pengujian yang ingin dijalankan:
echo  [1] Mode Dual-Window (Threading - Hemat RAM/VRAM Laptop 8GB) [DEFAULT]
echo  [2] Mode Dual-Window Multiprocessing (Multi-Core Terisolasi - Bebas GIL)
echo  [3] Mode Dual-Window + Database ^& Notifikasi Teams AKTIF (Test Dynamic Card)
echo  [4] Mode Dual-Window + Pilih File Video Sendiri (File Explorer Dialog)
echo  [5] Mode Headless Throughput Tinggi (Simulasi Jetson / Tanpa GUI Desktop)
echo  [6] Mode Top Inspection Saja (CAM-01: Plat Slat + Bilah Rib)
echo  [7] Mode Side Inspection Saja (CAM-02: Baut + Pin + Bracket)
echo  [8] Pengujian 60 Detik + Rekam Hasil ke CSV (benchmark_report.csv)
echo  [9] Uji Langsung Notifikasi Dynamic Card ke Node-RED / MS Teams
echo  [10] Ekspor Seluruh Model YOLO ke TensorRT FP16 (.engine)
echo.
set /p opt="Pilihan Anda (1-10, Enter untuk Default 1): "

if "%opt%"=="2" (
    echo.
    echo [*] Menjalankan Mode Dual-Window Multiprocessing (Multi-Core Terisolasi)...
    python run_multi_camera_benchmark.py --engine process --mode dual
) else if "%opt%"=="3" (
    echo.
    echo [*] Menjalankan Mode Dual-Window dengan DATABASE ^& TEAMS NOTIFIER AKTIF...
    python run_multi_camera_benchmark.py --mode dual --enable-db
) else if "%opt%"=="4" (
    echo.
    echo [*] Membuka File Dialog untuk Memilih Video CAM-01 Top ^& CAM-02 Side...
    python run_multi_camera_benchmark.py --mode dual --browse
) else if "%opt%"=="5" (
    echo.
    echo [*] Menjalankan Mode Headless Throughput Tinggi (Multiprocessing)...
    python run_multi_camera_benchmark.py --engine process --mode headless
) else if "%opt%"=="6" (
    echo.
    echo [*] Menjalankan Mode Top Inspection Saja (CAM-01)...
    python run_multi_camera_benchmark.py --mode top-only --enable-db
) else if "%opt%"=="7" (
    echo.
    echo [*] Menjalankan Mode Side Inspection Saja (CAM-02)...
    python run_multi_camera_benchmark.py --mode side-only --enable-db
) else if "%opt%"=="8" (
    echo.
    echo [*] Menjalankan Pengujian 60 Detik dengan Rekam CSV...
    python run_multi_camera_benchmark.py --mode dual --duration 60 --record-csv benchmark_report.csv
) else if "%opt%"=="9" (
    echo.
    echo ======================================================================
    echo   PILIH SKENARIO PENGUJIAN ADAPTIVE CARD NOTIFICATION:
    echo  [A] Single Defect: HANYA Rib Bending (Bilah Sirip)
    echo  [B] Single Defect: HANYA Plat Mangap (Center Void)
    echo  [C] Single Defect: HANYA Side Chain (Baut Kendor ^& Pin Geser)
    echo  [D] Multi Defect : Top Inspection (Rib Bengkok + Plat Void Mangap)
    echo  [E] Multi Defect : Gabungan Seluruh Kamera (Rib + Plat + Pin + Bolt)
    echo ======================================================================
    set /p sc="Pilihan Skenario (A-E, Enter untuk A): "
    if /i "%sc%"=="B" (
        python teams_notifier.py --scenario slat-only
    ) else if /i "%sc%"=="C" (
        python teams_notifier.py --scenario side-only
    ) else if /i "%sc%"=="D" (
        python teams_notifier.py --scenario multi-top
    ) else if /i "%sc%"=="E" (
        python teams_notifier.py --scenario multi-all
    ) else (
        python teams_notifier.py --scenario rib-only
    )
) else if "%opt%"=="10" (
    echo.
    echo [*] Menjalankan Ekspor Model ke TensorRT FP16 (.engine)...
    python export_tensorrt.py
) else (
    echo.
    echo [*] Menjalankan Mode Dual-Window (Kiri: CAM-01 Top, Kanan: CAM-02 Side)...
    python run_multi_camera_benchmark.py --mode dual
)

echo.
pause
