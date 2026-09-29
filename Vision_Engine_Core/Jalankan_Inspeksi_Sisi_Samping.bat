@echo off
title Conveyor Side Inspector (Bolt + Pin + Lekukan)
echo ============================================================
echo   AI Slat Conveyor - Side Inspection (CAM-01 / CAM-02)
echo   Komponen: Bolt, Pin, & Lekukan (Hybrid YOLO + PatchCore)
echo ============================================================
echo.

set PYTHON_SCRIPT=01_Side_Inspection_Bolt_Pin_Lekukan\src\side_inspector_pipeline.py

python "%PYTHON_SCRIPT%" %*

pause

