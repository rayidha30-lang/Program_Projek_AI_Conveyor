#!/usr/bin/env bash
# ==============================================================================
# SCRIPT RUNNER PRODUKSI - JETSON ORIN NANO (UBUNTU LINUX)
# Multi-Camera AI Slat Rib Conveyor System
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

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
if [ -n "${2:-}" ]; then
    ENGINE="$2"
elif [ "$MODE" = "web" ]; then
    ENGINE="hybrid"
else
    ENGINE="process"
fi

echo "[*] Mode Operasional : $MODE"
echo "[*] Engine Runner    : $ENGINE (web -> hybrid WS 3-kam, headless/dual -> isolated)"
echo "[*] Database Logging : AKTIF (SQLite + Teams Alert)"
echo "----------------------------------------------------------------------"

# 3. Jalankan Runner Multi-Kamera
python3 run_multi_camera_benchmark.py \
    --engine "$ENGINE" \
    --mode "$MODE" \
    --enable-db \
    --top-source "${TOP_CAM:-0}" \
    --side-source "${SIDE_CAM:-1}"
