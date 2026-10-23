#!/usr/bin/env bash
# ==============================================================================
# AI Slat & Rib Conveyor Inspection - Jetson Camera & Process Recovery Utility
# ==============================================================================
# Gunakan skrip ini jika kamera Jetson terkunci (busy), proses Python hang,
# atau setelah terjadi force close pada benchmark.
# ==============================================================================

echo "=================================================================="
echo "🛠️  MEMULAI PEMULIHAN KAMERA & PROSES NVIDIA JETSON ORIN NANO"
echo "=================================================================="

# 1. Hentikan seluruh proses Python benchmark yang masih menggantung
echo -e "\n[*] [1/4] Menghentikan sisa proses Python benchmark lama..."
pkill -9 -f run_multi_camera_benchmark.py 2>/dev/null || true
pkill -9 -f combined_top_inspector.py 2>/dev/null || true
pkill -9 -f side_inspector_pipeline.py 2>/dev/null || true
sleep 1
echo "    ✅ Sisa proses berhasil dibersihkan."

# 2. Reset daemon kamera Jetson (Argus / CSI / V4L2)
echo -e "\n[*] [2/4] Me-reset daemon kamera Jetson (nvargus-daemon)..."
if command -v systemctl &> /dev/null; then
    sudo systemctl restart nvargus-daemon 2>/dev/null || true
    echo "    ✅ nvargus-daemon di-restart."
else
    echo "    ℹ️  systemctl tidak tersedia, melewati langkah ini."
fi

# 3. Pulihkan hak akses node /dev/video*
echo -e "\n[*] [3/4] Memeriksa node kamera V4L2 (/dev/video*)..."
if ls /dev/video* 1> /dev/null 2>&1; then
    sudo chmod 666 /dev/video* 2>/dev/null || true
    echo "    ✅ Perangkat video terdeteksi:"
    ls -l /dev/video*
else
    echo "    ⚠️ Tidak ada perangkat /dev/video* yang terdeteksi saat ini."
fi

# 4. Verifikasi pustaka GUI Tkinter
echo -e "\n[*] [4/4] Memeriksa ketersediaan pustaka GUI Tkinter..."
if python3 -c "import tkinter" 2>/dev/null; then
    echo "    ✅ Pustaka Tkinter (python3-tk) siap digunakan."
else
    echo "    ⚠️ Pustaka Tkinter belum terpasang. Memasang otomatis..."
    sudo apt-get update && sudo apt-get install -y python3-tk
fi

echo -e "\n=================================================================="
echo "✅ PEMULIHAN SELESAI. Anda dapat menjalankan kembali benchmark:"
echo "   python3 run_multi_camera_benchmark.py"
echo "=================================================================="
