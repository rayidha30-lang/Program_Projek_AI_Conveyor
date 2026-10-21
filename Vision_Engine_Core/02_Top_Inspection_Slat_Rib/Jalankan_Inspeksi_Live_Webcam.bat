@echo off
chcp 65001 > nul
echo ==============================================================================
echo   INSPEKSI LIVE WEBCAM - HOUGH LINE STITCHING CONVEYOR
echo ==============================================================================
echo Membuka kamera webcam (default: Kamera #0)...
echo Catatan: Tekan [W] di layar untuk berganti kamera atau kembali ke video.
echo.
python "%~dp0src\hough_stitching_inspector.py" --cam 0
pause
