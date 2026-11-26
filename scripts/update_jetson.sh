#!/usr/bin/env bash
# ==============================================================================
# 🔄 SCRIPT UPDATE & PENYESUAIAN OTOMATIS - NVIDIA JETSON ORIN NANO
# Sistem Inspeksi AI Slat Rib Conveyor
# ==============================================================================
# Deskripsi:
#   Satu skrip terpadu untuk menyinkronkan pembaruan kode terbaru (CM-1/CM-2,
#   Batas Lateral Adaptif, Hysteresis & Consensus terbaru) ke perangkat Jetson
#   Orin Nano sekaligus memverifikasi profil kalibrasi, model GPU/TensorRT,
#   dan kesiapan kamera.
#
# Penggunaan di Jetson:
#   cd /path/ke/Program
#   chmod +x scripts/update_jetson.sh
#   bash scripts/update_jetson.sh
#
# Opsi tambahan:
#   bash scripts/update_jetson.sh --skip-trt   # Lewati ekspor TensorRT .engine
#   bash scripts/update_jetson.sh --skip-pull  # Lewati git pull (jika copy manual)
# ==============================================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

SKIP_TRT=false
SKIP_PULL=false
TARGET_BRANCH=""
while [ $# -gt 0 ]; do
    case "$1" in
        --skip-trt)  SKIP_TRT=true ;;
        --skip-pull) SKIP_PULL=true ;;
        --branch)    TARGET_BRANCH="${2:-}"; shift ;;
    esac
    shift
done

echo -e "${CYAN}======================================================================${NC}"
echo -e "${BOLD}🚀 UPDATE & PENYESUAIAN SISTEM INSPEKSI AI JETSON ORIN NANO${NC}"
echo -e "${CYAN}======================================================================${NC}"
echo -e "[*] Direktori Proyek : ${PROJECT_ROOT}"
echo -e "[*] Arsitektur       : $(uname -m)"

# ------------------------------------------------------------------------------
# 1. Tarik Pembaruan Kode Terbaru dari Git (dengan proteksi .env lokal)
# ------------------------------------------------------------------------------
echo ""
echo -e "${BOLD}[1/5] Sinkronisasi Kode Terbaru...${NC}"
if [ "$SKIP_PULL" = true ]; then
    echo -e "${YELLOW}ℹ️ --skip-pull aktif: Melewati git pull.${NC}"
elif [ -d ".git" ]; then
    # Backup .env lokal jika ada agar tidak tertimpa
    if [ -f ".env" ]; then
        cp .env ".env.bak_update"
    fi

    # Cek apakah ada perubahan lokal selain .env
    if ! git diff --quiet || ! git diff --cached --quiet; then
        echo -e "${YELLOW}⚠️ Terdeteksi perubahan lokal di Jetson. Menyimpan ke git stash sementara...${NC}"
        git stash push -m "auto-stash-before-jetson-update-$(date +%Y%m%d_%H%M%S)" || true
    fi

    if [ -z "$TARGET_BRANCH" ]; then
        TARGET_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)"
    fi
    echo -e "[*] Menarik pembaruan dari origin/${TARGET_BRANCH}..."
    git pull origin "$TARGET_BRANCH"

    if [ -f ".env.bak_update" ]; then
        mv .env.bak_update .env
    fi
    echo -e "${GREEN}✅ Kode berhasil diperbarui ke commit terbaru: $(git rev-parse --short HEAD)${NC}"
else
    echo -e "${YELLOW}⚠️ Folder .git tidak ditemukan (diasumsikan copy file manual). Lanjut...${NC}"
fi

# ------------------------------------------------------------------------------
# 2. Pastikan Profil Kalibrasi Aktif = HASIL_KALIBRASI
# ------------------------------------------------------------------------------
echo ""
echo -e "${BOLD}[2/5] Menyelaraskan Profil Parameter Kalibrasi...${NC}"
if [ -f "scripts/switch_inspection_profile.py" ]; then
    python3 scripts/switch_inspection_profile.py --use-calibrated
    echo -e "${GREEN}✅ Profil aktif disetel ke HASIL_KALIBRASI.${NC}"
else
    echo -e "${YELLOW}⚠️ scripts/switch_inspection_profile.py tidak ditemukan.${NC}"
fi

# ------------------------------------------------------------------------------
# 3. Pastikan File .env Siap di Root Proyek
# ------------------------------------------------------------------------------
echo ""
echo -e "${BOLD}[3/5] Memeriksa Konfigurasi Environment (.env)...${NC}"
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "${YELLOW}ℹ️ .env dibuat dari .env.example. Silakan sesuaikan kamera/DB bila perlu.${NC}"
    else
        echo -e "${YELLOW}⚠️ .env belum ada. Jalankan 'bash scripts/configure_jetson_env.sh --auto' nanti.${NC}"
    fi
else
    echo -e "${GREEN}✅ File .env sudah tersedia di root proyek.${NC}"
fi

# ------------------------------------------------------------------------------
# 4. Cek / Ekspor Model TensorRT (.engine) Khusus GPU Jetson
# ------------------------------------------------------------------------------
echo ""
echo -e "${BOLD}[4/5] Memeriksa Akselerasi Model AI (CUDA / TensorRT)...${NC}"
RIB_PT="Vision_Engine_Core/02_Top_Inspection_Slat_Rib/models/best_rib_v5.pt"
RIB_ONNX="Vision_Engine_Core/02_Top_Inspection_Slat_Rib/models/best_rib_v5.onnx"
RIB_ENGINE="Vision_Engine_Core/02_Top_Inspection_Slat_Rib/models/best_rib_v5.engine"

if [ "$SKIP_TRT" = false ] && [ -f "/etc/nv_tegra_release" ] && [ -f "$RIB_PT" ]; then
    if [ ! -f "$RIB_ENGINE" ] || [ "$RIB_PT" -nt "$RIB_ENGINE" ]; then
        echo -e "${CYAN}[*] Mengekspor best_rib_v5.pt ke TensorRT FP16 (.engine) untuk Jetson...${NC}"
        python3 scripts/export_tensorrt.py --model "$RIB_PT" || \
            echo -e "${YELLOW}⚠️ Ekspor TensorRT dilewati/gagal; sistem akan fallback ke ONNX/PyTorch.${NC}"
    else
        echo -e "${GREEN}✅ Model TensorRT best_rib_v5.engine sudah up-to-date.${NC}"
    fi
else
    if [ -f "$RIB_ONNX" ]; then
        echo -e "${GREEN}✅ Model ONNX tersedia: $RIB_ONNX${NC}"
    fi
    if [ "$SKIP_TRT" = true ]; then
        echo -e "${YELLOW}ℹ️ --skip-trt aktif: Melewati kompilasi TensorRT.${NC}"
    fi
fi

# ------------------------------------------------------------------------------
# 5. Verifikasi Kamera, CUDA, dan Self-Test Modul Inspeksi Terbaru
# ------------------------------------------------------------------------------
echo ""
echo -e "${BOLD}[5/5] Verifikasi Kamera Hardware & Self-Test Engine...${NC}"
if command -v v4l2-ctl &> /dev/null; then
    echo "[*] Perangkat kamera V4L2 terdeteksi:"
    v4l2-ctl --list-devices 2>/dev/null || ls -1 /dev/video* 2>/dev/null || echo "   (Tidak ada kamera terhubung)"
fi

python3 -c "
import sys
sys.path.insert(0, 'Vision_Engine_Core/02_Top_Inspection_Slat_Rib/src')
import cv2, numpy as np
print(f'OpenCV : {cv2.__version__} | NumPy : {np.__version__}')
try:
    import torch
    print(f'PyTorch: {torch.__version__} | CUDA Available: {torch.cuda.is_available()}')
except Exception as e:
    print(f'PyTorch Check: {e}')
from hybrid_rib_inspector import HybridRibInspector
insp = HybridRibInspector(tracker_engine='LEGACY')
print(f'HybridRibInspector OK | Mode: {insp.mode} | Thresholds: tip={insp.tip_thresh}, dev={insp.dev_thresh}, pv={insp.pv_thresh}')
"

# Pastikan semua script .sh memiliki izin eksekusi
chmod +x scripts/*.sh 2>/dev/null || true

echo ""
echo -e "${GREEN}======================================================================${NC}"
echo -e "${BOLD}✅ JETSON SUDAH TERSINKRONISASI DENGAN PEMBARUAN TERBARU!${NC}"
echo -e "${GREEN}======================================================================${NC}"
echo "Langkah selanjutnya untuk menjalankan inspeksi produksi di Jetson:"
echo "  • Mode Headless (Produksi 24/7) : bash scripts/run_jetson_production.sh headless"
echo "  • Mode Visual (Monitor HDMI)    : bash scripts/run_jetson_production.sh gui"
echo "  • Uji Kamera Top Saja           : python3 Vision_Engine_Core/02_Top_Inspection_Slat_Rib/src/hybrid_rib_inspector.py"
echo ""
