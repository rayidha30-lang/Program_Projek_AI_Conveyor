@echo off
title Perbaikan Kamera Windows 11 - HP Victus
echo ========================================================
echo   MENJALANKAN PERBAIKAN SERVICE KAMERA (ADMINISTRATOR)
echo ========================================================
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0repair_camera.ps1"
echo.
echo ========================================================
echo Perbaikan selesai. Tekan sembarang tombol untuk keluar...
pause >nul
