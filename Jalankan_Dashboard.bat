@echo off
title Web Dashboard - Slat Rib Conveyor
echo =======================================================
echo   Menjalankan Sistem Web Dashboard Slat Rib Conveyor
echo =======================================================
echo.

:: 1. Pastikan bekerja di folder root Program
cd /d "%~dp0"

:: 2. Menjalankan Backend REST API Server di window cmd terpisah (cmd /k agar jendela tetap terbuka)
echo [1/2] Menjalankan Backend REST API Server (Python Flask :5000)...
start "Backend API - Conveyor Inspection" cmd /k "cd /d "%~dp0backend_api" && python api.py"

:: Beri jeda 2 detik agar server API siap
timeout /t 2 /nobreak >nul

:: 3. Menjalankan Frontend Web Dashboard menggunakan 'call'
echo.
echo [2/2] Menjalankan Frontend Web Dashboard (React Vite :5173)...
cd /d "%~dp0FE"
call npm run dev -- --open

pause
