@echo off
chcp 65001 > nul
echo ==============================================================================
echo   INSPEKSI LIVE WEBCAM - HOUGH LINE STITCHING CONVEYOR
echo ==============================================================================
echo Membuka kamera webcam (default: Kamera #0)...
echo Catatan: Tekan [W] di layar untuk berganti kamera atau kembali ke video.
echo.
"C:\Users\Victus\miniconda3\python.exe" "c:\File Rayidha\Internship\Project\5. Project AI Slat Rib Conveyor\Program\Vision_Engine_Core\02_Top_Inspection_Slat_Rib\src\hough_stitching_inspector.py" --cam 0
pause
