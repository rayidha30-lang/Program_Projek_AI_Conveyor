#!/usr/bin/env bash
# ==============================================================================
# SCRIPT RUNNER PRODUKSI - JETSON ORIN NANO (UBUNTU LINUX)
# Multi-Camera AI Slat Rib Conveyor System
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================================"
echo "🚀 MENJALANKAN SISTEM INSPEKSI PRODUKSI DI JETSON ORIN NANO"
echo "======================================================================"

# 1. Kunci Clock & Performa Daya Maksimum
if command -v jetson_clocks &> /dev/null; then
    echo "[*] Mengaktifkan Performa Maksimum CPU/GPU (jetson_clocks)..."
    sudo jetson_clocks || true
fi

# 2. Cek Mode Eksekusi
MODE="${1:-headless}"
ENGINE="${2:-process}"

echo "[*] Mode Operasional : $MODE"
echo "[*] Engine Runner    : $ENGINE (Multi-Core Process Isolation)"
echo "[*] Database Logging : AKTIF (SQLite + Teams Alert)"
echo "----------------------------------------------------------------------"

# 3. Jalankan Runner Multi-Kamera
python3 run_multi_camera_benchmark.py \
    --engine "$ENGINE" \
    --mode "$MODE" \
    --enable-db \
    --top-source "${TOP_CAM:-0}" \
    --side-source "${SIDE_CAM:-1}"
