@echo off
title Web Dashboard - Slat Rib Conveyor
echo =======================================================
echo   Menjalankan Web Dashboard Slat Rib Conveyor...
echo =======================================================
echo.
cd /d "%~dp0FE"
echo Membuka server Vite dan web browser...
npm run dev -- --open
pause
