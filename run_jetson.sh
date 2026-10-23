#!/usr/bin/env bash
# ==============================================================================
# AI Slat & Rib Conveyor Inspection - Universal Jetson Launcher
# ==============================================================================
cd "$(dirname "$0")"

echo "======================================================================"
echo "🚀 Menjalankan Multi-Camera Inspection di NVIDIA Jetson Orin Nano"
echo "   Engine: Multi-Process Core | Akselerasi: CUDA / TensorRT FP16"
echo "======================================================================"

python3 run_multi_camera_benchmark.py --engine process --mode dual "$@"
