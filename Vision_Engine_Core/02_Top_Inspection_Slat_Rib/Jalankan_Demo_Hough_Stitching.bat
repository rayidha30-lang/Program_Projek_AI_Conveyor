@echo off
chcp 65001 > nul
echo ==============================================================================
echo   DEMO INTERAKTIF HOUGH LINE STITCHING (PAPER SENSORS 2025)
echo ==============================================================================
echo Memulai viewer interaktif...
python "%~dp0src\hough_stitching_inspector.py"
pause
