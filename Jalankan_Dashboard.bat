@echo off
title Web Dashboard - Slat Rib Conveyor
echo =======================================================
echo   Menjalankan Sistem Web Dashboard Slat Rib Conveyor
echo =======================================================
echo.

echo [1/2] Menjalankan Backend REST API Server (Python Flask :5000)...
start "Backend API - Conveyor Inspection" python "%~dp0Program AI\api.py"

echo.
echo [2/2] Menjalankan Frontend Web Dashboard (React Vite :5173)...
cd /d "%~dp0FE"
npm run dev -- --open

pause
