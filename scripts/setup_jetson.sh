#!/usr/bin/env bash
# ==============================================================================
# SCRIPT SETUP & INSTALASI OTOMATIS - JETSON ORIN NANO (UBUNTU LINUX)
# Sistem Inspeksi AI Slat Rib Conveyor
# ==============================================================================

set -e

echo "======================================================================"
echo "🚀 SETUP ENVIRONMENT AI CONVEYOR DI UBUNTU JETSON ORIN NANO"
echo "======================================================================"

# 1. Cek Arsitektur & Perangkat
ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ]; then
    echo "⚠️ Peringatan: Script ini dioptimalkan untuk arsitektur aarch64 (ARM64 Jetson)."
    echo "   Arsitektur sistem Anda terdeteksi: $ARCH"
fi

if [ -f "/etc/nv_tegra_release" ]; then
    echo "✅ Terdeteksi NVIDIA Jetson Platform:"
    cat /etc/nv_tegra_release
else
    echo "ℹ️ Catatan: File /etc/nv_tegra_release tidak ditemukan (kemungkinan container atau VM)."
fi

# Cek Versi Ubuntu
UBUNTU_VER=$(lsb_release -rs 2>/dev/null || echo "unknown")
echo "📦 Versi Ubuntu: $UBUNTU_VER"

# 2. Update APT & Instal Dependensi Sistem
echo ""
echo "----------------------------------------------------------------------"
echo "[1/5] Menginstal Dependensi Sistem (APT Packages)..."
echo "----------------------------------------------------------------------"
sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    python3-tk \
    build-essential \
    libopenblas-dev \
    libopenmpi-dev \
    libjpeg-dev \
    zlib1g-dev \
    unixodbc \
    unixodbc-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    curl \
    git \
    v4l-utils

# 3. Persiapan Python & Pip
echo ""
echo "----------------------------------------------------------------------"
echo "[2/5] Memperbarui pip, setuptools, dan wheel..."
echo "----------------------------------------------------------------------"
python3 -m pip install --upgrade pip setuptools wheel

# 4. Instalasi PyTorch Khusus NVIDIA Jetson (CUDA Accelerated)
echo ""
echo "----------------------------------------------------------------------"
echo "[3/5] Memeriksa Instalasi PyTorch CUDA untuk Jetson..."
echo "----------------------------------------------------------------------"

HAS_TORCH_CUDA=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")

if [ "$HAS_TORCH_CUDA" = "True" ]; then
    echo "✅ PyTorch dengan dukungan CUDA sudah terpasang dan aktif!"
    python3 -c "import torch; print('   Torch:', torch.__version__, '| Device:', torch.cuda.get_device_name(0))"
else
    echo "⚠️ PyTorch CUDA belum terdeteksi. Menginstal PyTorch resmi NVIDIA Jetson..."
    
    # Deteksi JetPack / Ubuntu version
    if [ "$UBUNTU_VER" = "20.04" ]; then
        echo "📥 Mengunduh PyTorch untuk JetPack 5 (Ubuntu 20.04 - Python 3.8)..."
        pip3 install --no-cache-dir https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
        # Torchvision v0.16.0 untuk PyTorch 2.1
        sudo apt-get install -y libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
        pip3 install --no-cache-dir --no-deps torchvision==0.16.1
    elif [ "$UBUNTU_VER" = "22.04" ]; then
        echo "📥 Mengunduh PyTorch untuk JetPack 6 (Ubuntu 22.04 - Python 3.10)..."
        pip3 install --no-cache-dir https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.3.0+nv24.05-cp310-cp310-linux_aarch64.whl || \
        pip3 install --no-cache-dir torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
        pip3 install --no-cache-dir --no-deps torchvision==0.18.0
    else
        echo "ℹ️ Mengunduh wheel PyTorch universal ARM64..."
        pip3 install --no-cache-dir torch torchvision
    fi
fi

# 5. Instalasi Dependensi Python dari requirements.txt
echo ""
echo "----------------------------------------------------------------------"
echo "[4/5] Menginstal Paket Python dari requirements.txt..."
echo "----------------------------------------------------------------------"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Kunci numpy < 2.0.0 terlebih dahulu agar tidak terjadi crash ABI dengan PyTorch / OpenCV Jetson
pip3 install "numpy>=1.23.5,<2.0.0"

# Install seluruh requirements
pip3 install -r "$SCRIPT_DIR/requirements.txt"

# 6. Verifikasi Akhir
echo ""
echo "----------------------------------------------------------------------"
echo "[5/5] Memverifikasi Lingkungan Runtime..."
echo "----------------------------------------------------------------------"
python3 -c "
import sys
print(f'Python Version      : {sys.version.split()[0]}')

try:
    import numpy as np
    print(f'NumPy               : {np.__version__} (OK)')
except Exception as e:
    print(f'NumPy               : GAGAL ({e})')

try:
    import cv2
    print(f'OpenCV              : {cv2.__version__} (OK)')
except Exception as e:
    print(f'OpenCV              : GAGAL ({e})')

try:
    import torch
    cuda_ok = torch.cuda.is_available()
    dev_name = torch.cuda.get_device_name(0) if cuda_ok else 'CPU Only'
    print(f'PyTorch             : {torch.__version__} | CUDA: {cuda_ok} ({dev_name})')
except Exception as e:
    print(f'PyTorch             : GAGAL ({e})')

try:
    import ultralytics
    print(f'Ultralytics (YOLO)  : {ultralytics.__version__} (OK)')
except Exception as e:
    print(f'Ultralytics         : GAGAL ({e})')

try:
    import sklearn
    print(f'Scikit-Learn        : {sklearn.__version__} (OK)')
except Exception as e:
    print(f'Scikit-Learn        : GAGAL ({e})')

try:
    import flask
    print(f'Flask Backend API   : {flask.__version__} (OK)')
except Exception as e:
    print(f'Flask Backend API   : GAGAL ({e})')
"

echo ""
echo "======================================================================"
echo "🎉 SETUP SELESAI! Sistem siap dijalankan di Jetson Orin Nano."
echo "======================================================================"
echo "Langkah Selanjutnya:"
echo "  1. Konfigurasi .env : bash configure_jetson_env.sh"
echo "  2. Multi-Cam Runner : python3 run_multi_camera_benchmark.py"
echo "  3. Production Mode  : bash run_jetson_production.sh"
echo "======================================================================"
