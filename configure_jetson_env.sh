#!/usr/bin/env bash
# ==============================================================================
# 🛠️ NVIDIA JETSON ENVIRONMENT AUTO-CONFIGURATOR (.env GENERATOR)
# Sistem Inspeksi AI Slat & Rib Conveyor
# Target Platform: NVIDIA Jetson Orin Nano / Orin NX / Xavier (Ubuntu aarch64)
# ==============================================================================
# Script ini secara cerdas mendeteksi hardware Jetson, kamera yang terpasang,
# dan model TensorRT, kemudian membuat file '.env' yang 100% selaras dengan
# sistem inspeksi conveyor.
#
# Penggunaan:
#   1. Mode Interaktif (Wizard Tanya-Jawab):
#      bash configure_jetson_env.sh
#
#   2. Mode Otomatis (Auto-Detect Hardware):
#      bash configure_jetson_env.sh --auto
# ==============================================================================

set -e

# Warna Terminal
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${CYAN}======================================================================${NC}"
echo -e "${BOLD}🚀 NVIDIA JETSON .env GENERATOR & HARDWARE DETECTOR${NC}"
echo -e "${CYAN}======================================================================${NC}"

# 1. Deteksi Platform & Arsitektur
ARCH=$(uname -m)
IS_JETSON=false
if [ -f "/etc/nv_tegra_release" ] || [ "$ARCH" = "aarch64" ]; then
    IS_JETSON=true
    echo -e "${GREEN}✅ Terdeteksi Platform NVIDIA Jetson (${ARCH})${NC}"
    if [ -f "/etc/nv_tegra_release" ]; then
        head -n 1 /etc/nv_tegra_release
    fi
else
    echo -e "${YELLOW}ℹ️ Berjalan di perangkat non-Jetson / Host PC (${ARCH})${NC}"
fi

# 2. Backup file .env lama jika ada
if [ -f ".env" ]; then
    BACKUP_NAME=".env.backup_$(date +%Y%m%d_%H%M%S)"
    cp .env "$BACKUP_NAME"
    echo -e "${YELLOW}📦 File .env lama dibackup ke: ${BACKUP_NAME}${NC}"
fi

AUTO_MODE=false
if [ "$1" = "--auto" ] || [ "$1" = "-y" ]; then
    AUTO_MODE=true
    echo -e "${CYAN}[*] Berjalan dalam Mode Otomatis (--auto)...${NC}"
fi

# 3. Deteksi Kamera Hardware (/dev/video*)
echo ""
echo -e "${BOLD}[1/5] Memindai Kamera Hardware...${NC}"
AVAILABLE_V4L2=()
if compgen -G "/dev/video*" > /dev/null; then
    for v in /dev/video*; do
        AVAILABLE_V4L2+=("$v")
    done
    echo -e "${GREEN}✅ Ditemukan ${#AVAILABLE_V4L2[@]} port video: ${AVAILABLE_V4L2[*]}${NC}"
else
    echo -e "${YELLOW}⚠️ Tidak ada kamera USB/CSI (/dev/video*) yang terdeteksi.${NC}"
fi

# Tentukan Default Kamera Top & Side
DEFAULT_TOP_CAM="Video_Testing/OK Semua.mp4"
DEFAULT_SIDE_CAM="Video_Testing/Deform Sedikit.mp4"

if [ ${#AVAILABLE_V4L2[@]} -ge 2 ]; then
    DEFAULT_TOP_CAM="0"
    DEFAULT_SIDE_CAM="1"
elif [ ${#AVAILABLE_V4L2[@]} -eq 1 ]; then
    DEFAULT_TOP_CAM="0"
    DEFAULT_SIDE_CAM="Video_Testing/Deform Sedikit.mp4"
fi

SELECTED_TOP="$DEFAULT_TOP_CAM"
SELECTED_SIDE="$DEFAULT_SIDE_CAM"

if [ "$AUTO_MODE" = false ]; then
    echo ""
    echo -e "${BOLD}Pilih Sumber Input CAM-01 (Kamera Atas / Slat & Rib):${NC}"
    echo "  1) Video File Offline : Video_Testing/OK Semua.mp4"
    if [ ${#AVAILABLE_V4L2[@]} -gt 0 ]; then
        echo "  2) Kamera USB / CSI   : Index 0 (/dev/video0)"
    fi
    if [ ${#AVAILABLE_V4L2[@]} -gt 1 ]; then
        echo "  3) Kamera USB / CSI   : Index 1 (/dev/video1)"
    fi
    echo "  4) CSI GStreamer Pipeline (nvarguscamerasrc)"
    echo "  5) Input Custom / RTSP URL"
    read -p "Pilihan Anda [Default: $DEFAULT_TOP_CAM]: " top_choice

    case "$top_choice" in
        1) SELECTED_TOP="Video_Testing/OK Semua.mp4" ;;
        2) SELECTED_TOP="0" ;;
        3) SELECTED_TOP="1" ;;
        4) SELECTED_TOP="nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=1920, height=1080 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! appsink" ;;
        5) read -p "Masukkan Path Video / RTSP URL: " custom_top; SELECTED_TOP="$custom_top" ;;
        *) SELECTED_TOP="$DEFAULT_TOP_CAM" ;;
    esac

    echo ""
    echo -e "${BOLD}Pilih Sumber Input CAM-02 (Kamera Samping / Baut & Pin):${NC}"
    echo "  1) Video File Offline : Video_Testing/Deform Sedikit.mp4"
    if [ ${#AVAILABLE_V4L2[@]} -gt 1 ]; then
        echo "  2) Kamera USB / CSI   : Index 1 (/dev/video1)"
    fi
    if [ ${#AVAILABLE_V4L2[@]} -gt 0 ]; then
        echo "  3) Kamera USB / CSI   : Index 0 (/dev/video0)"
    fi
    echo "  4) Input Custom / RTSP URL"
    read -p "Pilihan Anda [Default: $DEFAULT_SIDE_CAM]: " side_choice

    case "$side_choice" in
        1) SELECTED_SIDE="Video_Testing/Deform Sedikit.mp4" ;;
        2) SELECTED_SIDE="1" ;;
        3) SELECTED_SIDE="0" ;;
        4) read -p "Masukkan Path Video / RTSP URL: " custom_side; SELECTED_SIDE="$custom_side" ;;
        *) SELECTED_SIDE="$DEFAULT_SIDE_CAM" ;;
    esac
fi

# 4. Deteksi Model AI (TensorRT .engine vs PyTorch .pt)
echo ""
echo -e "${BOLD}[2/5] Memeriksa Ketersediaan Bobot Model AI...${NC}"
TOP_MODEL="Dataset Yolo/best_v2.pt"
SIDE_MODEL="Dataset Yolo/best_v1.pt"

# Cek apakah ada TensorRT engine hasil konversi di Jetson
if [ -f "Jetson_Vision_Engine/02_Top_Inspection_Slat_Rib/models/best_rib.engine" ]; then
    TOP_MODEL="Jetson_Vision_Engine/02_Top_Inspection_Slat_Rib/models/best_rib.engine"
    echo -e "${GREEN}✅ Ditemukan TensorRT Top Model: $TOP_MODEL${NC}"
elif [ -f "02_Top_Inspection_Slat_Rib/models/best_rib.engine" ]; then
    TOP_MODEL="02_Top_Inspection_Slat_Rib/models/best_rib.engine"
    echo -e "${GREEN}✅ Ditemukan TensorRT Top Model: $TOP_MODEL${NC}"
fi

if [ -f "Jetson_Vision_Engine/01_Side_Inspection_Bolt_Pin/models/best_v2.engine" ]; then
    SIDE_MODEL="Jetson_Vision_Engine/01_Side_Inspection_Bolt_Pin/models/best_v2.engine"
    echo -e "${GREEN}✅ Ditemukan TensorRT Side Model: $SIDE_MODEL${NC}"
elif [ -f "01_Side_Inspection_Bolt_Pin/models/best_v2.engine" ]; then
    SIDE_MODEL="01_Side_Inspection_Bolt_Pin/models/best_v2.engine"
    echo -e "${GREEN}✅ Ditemukan TensorRT Side Model: $SIDE_MODEL${NC}"
fi

# 5. Rekomendasi Engine & Display Mode
echo ""
echo -e "${BOLD}[3/5] Menentukan Mode Benchmark & Engine...${NC}"

# Di Jetson Orin Nano, 'process' lebih unggul karena memanfaatkan multi-core CPU Cortex-A78AE
if [ "$IS_JETSON" = true ]; then
    SELECTED_ENGINE="process"
else
    SELECTED_ENGINE="threads"
fi

# Cek apakah terhubung ke Layar Desktop GUI
if [ -n "$DISPLAY" ]; then
    SELECTED_MODE="dual"
    echo -e "${GREEN}✅ Display X11 terdeteksi ($DISPLAY) -> Default Mode: DUAL GUI${NC}"
else
    SELECTED_MODE="headless"
    echo -e "${YELLOW}ℹ️ Tidak ada Display X11 (Headless / SSH) -> Default Mode: HEADLESS (Server Mode)${NC}"
fi

if [ "$AUTO_MODE" = false ]; then
    read -p "Pilih Tampilan GUI ('dual' untuk 2 jendela OpenCV, 'headless' tanpa jendela) [Default: $SELECTED_MODE]: " mode_in
    if [ -n "$mode_in" ]; then
        SELECTED_MODE="$mode_in"
    fi
fi

# 6. Menulis File .env Resmi
echo ""
echo -e "${BOLD}[4/5] Menghasilkan File .env Baru...${NC}"

cat <<EOF > .env
# ==============================================================================
# AI SLAT & RIB CONVEYOR INSPECTION - JETSON ENVIRONMENT (.env)
# Dibuat otomatis oleh: configure_jetson_env.sh pada $(date)
# Platform: $(uname -s) $(uname -m)
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. INPUT VIDEO & SUMBER KAMERA (JETSON CAMERA SOURCES)
# ------------------------------------------------------------------------------
# Sumber input CAM-01 (Kamera Atas / Slat & Rib)
TOP_CAMERA_SOURCE=$SELECTED_TOP

# Sumber input CAM-02 / CAM-03 (Kamera Samping / Baut & Pin)
SIDE_CAMERA_SOURCE=$SELECTED_SIDE

# Identitas Kamera Samping
SIDE_CAMERA_NAME=RIGHT
SIDE_CAMERA_ID=CAM-02

# ------------------------------------------------------------------------------
# 2. MODEL AI & ENGINE INFERENSI (TENSORRT / JETSON ENGINE)
# ------------------------------------------------------------------------------
# Arsitektur: 'process' (Multi-process isolation multi-core Jetson) atau 'threads'
BENCHMARK_ENGINE=$SELECTED_ENGINE

# Komputasi Hardware GPU NVIDIA Jetson
INFERENCE_DEVICE=cuda

# Mode deteksi sirip rib
RIB_DETECTION_MODE=HYBRID

# Path bobot model AI (Gunakan .engine untuk akselerasi maksimal TensorRT FP16)
MODEL_TOP_PATH=$TOP_MODEL
MODEL_SIDE_PATH=$SIDE_MODEL

# ------------------------------------------------------------------------------
# 3. DURASI & MODE EKSEKUSI
# ------------------------------------------------------------------------------
# Mode eksekusi: 'headless' (Cocok untuk Jetson / Web Dashboard) atau 'dual' (GUI)
BENCHMARK_MODE=$SELECTED_MODE

# Durasi benchmark (detik): 0 = berjalan terus menerus (produksi)
BENCHMARK_DURATION=0

# Log CSV telemetri performa
RECORD_CSV=jetson_benchmark_fps.csv

# ------------------------------------------------------------------------------
# 4. DATABASE & LOGGING BUKTI CACAT (SQLITE & STORAGE)
# ------------------------------------------------------------------------------
ENABLE_DB_LOGGING=false
SQLITE_DB_PATH=AI_Slat_Rib_Conveyor.db
CAPTURES_DIR=captures

# ------------------------------------------------------------------------------
# 5. INTEGRASI NOTIFIKASI OTOMASI INDUSTRI
# ------------------------------------------------------------------------------
NODE_RED_ENDPOINT=http://127.0.0.1:1880/ai-inspection
NOTIFICATION_COOLDOWN_SEC=3.0

# ------------------------------------------------------------------------------
# 6. PARAMETER KALIBRASI FISIK KONVEYOR
# ------------------------------------------------------------------------------
SCALE_FACTOR_MM_PER_PX=0.82
TARGET_MIN_FPS=20.0
EOF

# Sinkronkan juga ke dalam folder Jetson_Vision_Engine jika folder tersebut ada
if [ -d "Jetson_Vision_Engine" ]; then
    cp .env Jetson_Vision_Engine/.env
    echo -e "${GREEN}✅ Disalin ke Jetson_Vision_Engine/.env${NC}"
fi

echo -e "${GREEN}🎉 File .env berhasil dibuat dan dikonfigurasi!${NC}"
echo "----------------------------------------------------------------------"
cat .env
echo "----------------------------------------------------------------------"

# 7. Tes Singkat Verifikasi
echo ""
echo -e "${BOLD}[5/5] Uji Coba Cepat (Sanity Check)...${NC}"
if command -v python3 &> /dev/null; then
    echo "[*] Menjalankan benchmark verifikasi 2 detik dalam mode headless..."
    python3 run_multi_camera_benchmark.py --mode headless --duration 2 || {
        echo -e "${YELLOW}⚠️ Uji coba benchmark selesai dengan peringatan (periksa log di atas).${NC}"
    }
fi

echo ""
echo -e "${GREEN}======================================================================${NC}"
echo -e "${BOLD}✅ KONFIGURASI JETSON SELESAI & SIAP DIJALANKAN!${NC}"
echo -e "${GREEN}======================================================================${NC}"
echo "Untuk menjalankan sistem inspeksi di Jetson:"
echo "  1. Live Benchmark   : python3 run_multi_camera_benchmark.py"
echo "  2. Mode Headless    : python3 run_multi_camera_benchmark.py --mode headless"
echo "  3. Production Mode  : bash run_jetson_production.sh"
echo -e "${CYAN}======================================================================${NC}"
