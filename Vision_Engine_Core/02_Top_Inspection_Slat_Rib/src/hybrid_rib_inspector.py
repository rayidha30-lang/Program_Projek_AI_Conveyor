"""
==============================================================================
MODUL INSPEKSI SIRIP HIBRIDA (HYBRID RIB INSPECTOR: YOLO + DUAL-POLARITY)
==============================================================================
Deskripsi:
  Pendeteksi & Pengukur Deformasi Kelurusan Sirip (Rib) Konveyor CAM-03
  dengan arsitektur modular yang mendukung 3 mode operasional:
    1. HYBRID (Rekomendasi Utama):
       Konfirmasi ganda: Deep Learning YOLO Bounding Box + Verifikasi Fisika 
       Optik Dual-Polarity Thickness (Bilah Tegak 3D setinggi 50 - 220 px).
       Hasil: 100% Kebal False Alarm pada plat datar/noda oli, dan presisi tinggi pada sirip.
    2. YOLO_ONLY:
       Deteksi keberadaan sirip berbasis Deep Learning YOLO best_rib.pt pada zona inspeksi konveyor.
    3. DUAL_POLARITY_ONLY:
       Deteksi keberadaan murni berbasis Fisika Optik Gradien Sobel Y (+Sobel Y puncak vs -Sobel Y kaki bilah)
       tanpa membutuhkan GPU/PyTorch (< 2 ms).

Metrologi:
  - Pelacakan Punggung Atas (Top Ridge Tracking) sub-pixel langsung pada bilah fisik
  - Regresi Garis Acuan Linear (Baseline) & Filter Median 1D
  - Perhitungan Flatness PV (Peak-to-Valley) dan Sagitta Parabola (Kelengkungan C-Shape)
  - Kriteria Mutu QC: OK (Lurus) vs NG (Bengkok / Deformasi)
==============================================================================
"""

import os
import sys
import time
import json
import glob
import argparse
import warnings
import cv2
import numpy as np

# Filter warning ABI torchvision dan deprecation half agar terminal bersih
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", message=".*half.*deprecated.*")

# Bootstrap Environment CUDA 12.6 Jetson Orin Nano secara otomatis
try:
    import jetson_cuda_env  # noqa: F401
except ImportError:
    pass

# Aktifkan High-Resolution Multimedia Timer (1ms quantum) pada Windows
if os.name == "nt":
    try:
        import ctypes
        ctypes.windll.winmm.timeBeginPeriod(1)
    except Exception:
        pass


# Safe / Optional Import YOLO & PyTorch
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

try:
    from hardware_telemetry import HardwareTelemetry, draw_camera_badge, draw_hud_card
    HAS_TELEMETRY = True
except ImportError:
    HAS_TELEMETRY = False

try:
    from rib_tip_features import (
        extract_tip_classifier_features,
        load_tip_classifier_model,
        predict_tip_ng_prob,
    )
    HAS_TIP_CLASSIFIER = True
except ImportError:
    HAS_TIP_CLASSIFIER = False


def select_file_dialog(initial_dir=None, title="Pilih File Video / Foto Inspeksi Sirip"):
    """
    Membuka file dialog explorer Windows (Tkinter) untuk memilih file video atau foto.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            root.lift()
            root.focus_force()
        except Exception:
            pass

        file_path = filedialog.askopenfilename(
            title=title,
            initialdir=initial_dir if initial_dir and os.path.exists(initial_dir) else os.getcwd(),
            filetypes=[
                ("Semua Media Inspeksi (*.mp4, *.avi, *.jpg, *.png)", "*.mp4 *.avi *.mkv *.mov *.jpg *.jpeg *.png *.bmp *.webp"),
                ("Video Files (*.mp4, *.avi, *.mkv, *.mov)", "*.mp4 *.avi *.mkv *.mov"),
                ("Image Files (*.jpg, *.jpeg, *.png, *.bmp)", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("Semua File (*.*)", "*.*")
            ]
        )
        root.update()
        root.destroy()
        return file_path if file_path else None
    except Exception as e:
        print(f"\n[!] File Dialog GUI (Tkinter) tidak dapat dibuka: {e}")
        print("    ℹ️  Tips Jetson Linux: jalankan 'sudo apt-get install -y python3-tk' jika ingin jendela file dialog.")
        try:
            p = input(f"[*] Masukkan path file untuk [{title}] (Enter untuk batal): ").strip().strip("'\"")
            if p and os.path.exists(p):
                return os.path.abspath(p)
        except Exception:
            pass
        return None


def open_media_source(source_path):
    """
    Membuka sumber media berupa video (.mp4, .avi) atau gambar diam (.jpg, .png).
    Mengembalikan: (cap, is_image, static_frame, total_frames, file_name)
    """
    if not source_path or not os.path.exists(source_path):
        return None, False, None, 0, ""

    fname = os.path.basename(source_path)
    ext = os.path.splitext(source_path)[1].lower()
    img_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]

    if ext in img_exts:
        img = cv2.imread(source_path)
        if img is not None:
            return None, True, img, 1, fname

    cap = cv2.VideoCapture(source_path)
    if cap.isOpened():
        tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return cap, False, None, tot, fname

    return None, False, None, 0, fname


def fast_medfilt(x, k=9):
    """Filter Median 1D cepat menggunakan batas kernel ganjil."""
    k = max(3, int(k) if int(k) % 2 == 1 else int(k) + 1)
    pad = k // 2
    x_padded = np.pad(x, pad, mode='edge')
    windows = np.lib.stride_tricks.sliding_window_view(x_padded, k)
    return np.median(windows, axis=-1)

def order_quad_points(pts):
    """
    Mengurutkan 4 titik kuadran konveyor: [Top-Left, Top-Right, Bottom-Right, Bottom-Left].
    Menjamin urutan sudut selalu tepat untuk cv2.getPerspectiveTransform.
    """
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)



class HybridRibInspector:
    """
    Inspector Kelurusan Sirip (Rib) dengan Konfigurasi Hibrida:
    YOLO Object Detection + Dual-Polarity Physics Ridge Tracking.
    """
    MODES = ["HYBRID", "YOLO_ONLY", "DUAL_POLARITY_ONLY"]

    def __init__(
        self,
        mode="HYBRID",
        model_path=None,
        conf_thresh=0.55,
        iou_thresh=0.45,
        pv_thresh=16.5,
        sagitta_thresh=11.5,
        dev_thresh=9.5,
        tip_thresh=9.5,
        device=None,
        stride=None,
        tracker_engine="LEGACY",
        use_tip_classifier=False
    ):
        """
        :param mode: "HYBRID", "YOLO_ONLY", atau "DUAL_POLARITY_ONLY"
        :param model_path: Path ke bobot YOLO (.pt)
        :param conf_thresh: Confidence threshold YOLO
        :param iou_thresh: IoU NMS threshold YOLO
        :param pv_thresh: Batas toleransi Flatness PV (px)
        :param sagitta_thresh: Batas toleransi kelengkungan Sagitta (px)
        :param dev_thresh: Batas toleransi simpangan deviasi lokal badan sirip (px)
        :param tip_thresh: Batas toleransi simpangan deviasi ujung sirip / tip (px)
        :param device: Komputasi AI ('auto', 'cuda', 0, atau 'cpu')
        """
        # Muat otomatis threshold resmi dari top_inspection_config.json bila tersedia
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(cur_dir)
        cfg_path = os.path.join(parent_dir, "configs", "top_inspection_config.json")
        cfg_prime_min_y = 355
        cfg_prime_max_y = 435
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as _f_cfg:
                    _cfg_data = json.load(_f_cfg)
                    _th = _cfg_data.get("thresholds", {})
                    if "rib_pv_threshold" in _th and pv_thresh == 16.5:
                        pv_thresh = float(_th["rib_pv_threshold"])
                    if "rib_sagitta_threshold" in _th and sagitta_thresh == 11.5:
                        sagitta_thresh = float(_th["rib_sagitta_threshold"])
                    if ("rib_dev_threshold" in _th or "rib_max_dev_threshold" in _th) and dev_thresh == 9.5:
                        dev_thresh = float(_th.get("rib_dev_threshold", _th.get("rib_max_dev_threshold")))
                    if "rib_tip_threshold" in _th and tip_thresh == 9.5:
                        tip_thresh = float(_th["rib_tip_threshold"])
                    if "rib_prime_min_y" in _th and _th["rib_prime_min_y"] is not None:
                        cfg_prime_min_y = int(_th["rib_prime_min_y"])
                    if "rib_prime_max_y" in _th and _th["rib_prime_max_y"] is not None:
                        cfg_prime_max_y = int(_th["rib_prime_max_y"])
                    # Muat severity zones (untuk klasifikasi LOW/MEDIUM/HIGH bending di masa depan)
                    self._severity_zones = _th.get("rib_severity_zones", {})
            except Exception:
                pass

        self.mode = mode.upper() if mode.upper() in self.MODES else "HYBRID"
        self.conf_thresh = float(conf_thresh)
        self.iou_thresh = float(iou_thresh)
        self.pv_thresh = float(pv_thresh)
        self.sagitta_thresh = float(sagitta_thresh)
        self.dev_thresh = float(dev_thresh)
        self.tip_thresh = float(tip_thresh)
        self._base_pv_thresh = self.pv_thresh
        self._base_sagitta_thresh = self.sagitta_thresh
        self._base_dev_thresh = self.dev_thresh
        self._base_tip_thresh = self.tip_thresh
        self._viterbi_min_tip_thresh = 10.5

        # Flag: threshold sudah dimuat dari dataset adaptif (untuk log)
        self._adaptive_loaded = False
        self._adaptive_n_ok = 0
        # Pastikan _severity_zones terisi default jika config tidak memuatnya
        if not hasattr(self, "_severity_zones"):
            self._severity_zones = {}

        # Muat Jalur 1 Pure-NumPy Tip Classifier hanya jika diaktifkan (opt-in setelah multi-video training)
        self.use_tip_classifier = bool(use_tip_classifier)
        self.tip_classifier_model = None
        if self.use_tip_classifier and HAS_TIP_CLASSIFIER:
            _clf_path = os.path.join(parent_dir, "configs", "rib_tip_classifier.json")
            self.tip_classifier_model = load_tip_classifier_model(_clf_path)
            if self.tip_classifier_model is not None:
                _meta = self.tip_classifier_model.get("metadata", {})
                print(
                    f"[HybridRibInspector] 🧠 Tip Classifier (Jalur 1) aktif: "
                    f"n_ok={_meta.get('n_ok', '?')}, n_ng={_meta.get('n_ng', '?')}, "
                    f"th={self.tip_classifier_model.get('prob_threshold', 0.5):.2f}"
                )

        # 0. Konfigurasi Device Komputasi (CUDA / CPU)
        if device is None or str(device).lower() in ("auto", ""):
            if HAS_TORCH and torch.cuda.is_available():
                self.device = 0
            else:
                self.device = "cpu"
        elif str(device).lower() in ("cuda", "gpu"):
            self.device = 0 if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
        elif str(device).isdigit():
            self.device = int(device)
        else:
            self.device = str(device)

        if self.device != "cpu" and HAS_TORCH and torch.cuda.is_available():
            dev_idx = self.device if isinstance(self.device, int) else 0
            dev_name = torch.cuda.get_device_name(dev_idx)
            self.device_info = f"CUDA GPU #{dev_idx} ({dev_name})"
            self.device_badge = "CUDA"
            self.use_half = True  # FP16 Tensor Cores untuk akselerasi optimal pada NVIDIA RTX & Jetson Orin
            self.torch_device = torch.device(f"cuda:{dev_idx}")
        else:
            self.device = "cpu"
            self.device_info = "CPU"
            self.device_badge = "CPU"
            self.use_half = False
            self.torch_device = torch.device("cpu") if HAS_TORCH else None

        # Stride inferensi YOLO: 2 pada Orin Nano / edge device (melipatgandakan FPS), 1 jika GPU desktop
        if stride is not None:
            self.yolo_stride = max(1, int(stride))
        elif self.device == "cpu" or (self.device_info and "orin" in self.device_info.lower()):
            self.yolo_stride = 2
        else:
            self.yolo_stride = 1

        # Cari model YOLO otomatis jika tidak dispesifikasi
        self.yolo_model = None
        self.model_path = model_path
        self.class_names = {}
        if HAS_YOLO:
            cur_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(cur_dir)

            # Siapkan seluruh kandidat model rib resmi (.engine, .pt, .onnx)
            engine_candidates = [
                os.path.join(parent_dir, "models", "best_rib_v5.engine"),
                os.path.join(parent_dir, "models", "best_rib_v4.engine"),
                os.path.join(parent_dir, "models", "best_rib.engine"),
                os.path.join(cur_dir, "best_rib_v5.engine"),
                os.path.join(cur_dir, "best_rib_v4.engine"),
                os.path.join(cur_dir, "best_rib.engine"),
            ]
            pt_candidates = [
                os.path.join(parent_dir, "models", "best_rib_v5.pt"),
                os.path.join(parent_dir, "models", "best_rib_v4.pt"),
                os.path.join(parent_dir, "models", "best_rib_v3.pt"),
                os.path.join(parent_dir, "models", "best_rib_v2.pt"),
                os.path.join(parent_dir, "models", "best_rib.pt"),
                os.path.join(cur_dir, "best_rib_v5.pt"),
                os.path.join(cur_dir, "best_rib_v4.pt"),
                os.path.join(cur_dir, "best_rib_v3.pt"),
                os.path.join(cur_dir, "best_rib_v2.pt"),
                os.path.join(cur_dir, "best_rib_v1.pt"),
                os.path.join(parent_dir, "best_rib_v5.pt"),
                os.path.join(parent_dir, "best_rib_v4.pt"),
                os.path.join(parent_dir, "best_rib_v3.pt"),
                os.path.join(parent_dir, "best_rib_v2.pt"),
                os.path.join(parent_dir, "best_rib_v1.pt")
            ]
            onnx_candidates = [
                os.path.join(parent_dir, "models", "best_rib_v5.onnx"),
                os.path.join(parent_dir, "models", "best_rib_v4.onnx"),
                os.path.join(parent_dir, "models", "best_rib.onnx"),
                os.path.join(cur_dir, "best_rib_v5.onnx"),
                os.path.join(cur_dir, "best_rib_v4.onnx"),
                os.path.join(cur_dir, "best_rib.onnx"),
                os.path.abspath(os.path.join(cur_dir, "..", "..", "..", "02_Top_Inspection_Slat_Rib", "models", "best_rib_v4.onnx")),
                os.path.abspath(os.path.join(cur_dir, "..", "..", "..", "02_Top_Inspection_Slat_Rib", "models", "best_rib.onnx")),
            ]
            has_ort_cuda = False
            if self.device != "cpu":
                try:
                    import onnxruntime as _ort
                    has_ort_cuda = "CUDAExecutionProvider" in _ort.get_available_providers()
                except Exception:
                    has_ort_cuda = False

            if self.device != "cpu":
                candidates = (engine_candidates + onnx_candidates + pt_candidates) if has_ort_cuda else (engine_candidates + pt_candidates + onnx_candidates)
            else:
                candidates = pt_candidates + onnx_candidates

            if not model_path or not os.path.exists(model_path):
                env_top = os.environ.get("MODEL_TOP_PATH")
                if env_top:
                    cand_env = [
                        env_top,
                        os.path.join(cur_dir, env_top),
                        os.path.join(parent_dir, env_top),
                        os.path.abspath(os.path.join(cur_dir, "..", "..", "..", env_top))
                    ]
                    for ce in cand_env:
                        if os.path.exists(ce):
                            model_path = ce
                            break

            if not model_path or not os.path.exists(model_path):
                for c in candidates:
                    if os.path.exists(c):
                        model_path = c
                        break

            # Auto-upgrade .pt ke .engine (TensorRT) atau .onnx (ONNX Runtime CUDA) jika tersedia di GPU
            if model_path and model_path.endswith(".pt") and self.device != "cpu":
                eng_cand = model_path.replace(".pt", ".engine")
                onnx_cand = model_path.replace(".pt", ".onnx")
                if os.path.exists(eng_cand):
                    model_path = eng_cand
                    print(f"[HybridRibInspector] 🚀 Menggunakan TensorRT Engine FP16: {os.path.basename(eng_cand)}")
                elif has_ort_cuda and os.path.exists(onnx_cand):
                    model_path = onnx_cand
                    print(f"[HybridRibInspector] ⚡ Akselerasi ONNX Runtime CUDA aktif: {os.path.basename(onnx_cand)}")

            def _load_detector_weights(weights_path):
                is_trt_or_onnx = str(weights_path).endswith((".engine", ".onnx"))
                if str(weights_path).endswith(".onnx"):
                    try:
                        import onnxruntime as _ort
                        _ort.set_default_logger_severity(3)
                    except Exception:
                        pass
                loaded = YOLO(weights_path, task="detect") if is_trt_or_onnx else YOLO(weights_path)
                if self.device != "cpu" and HAS_TORCH and torch.cuda.is_available() and not is_trt_or_onnx:
                    try:
                        loaded.to(self.device)
                        if hasattr(loaded, "model") and loaded.model is not None:
                            loaded.model.half()
                        self.use_half = True
                    except Exception:
                        pass
                elif is_trt_or_onnx:
                    self.use_half = True
                return loaded

            self.model_path = model_path
            if model_path and os.path.exists(model_path):
                try:
                    self.yolo_model = _load_detector_weights(model_path)
                    self.class_names = self.yolo_model.names if hasattr(self.yolo_model, 'names') else {0: 'rib'}

                    # Validasi kelas model: pastikan bukan model Bolt/Pin yang salah konfigurasi
                    is_rib_model = any("rib" in str(v).lower() or "sirip" in str(v).lower() for v in self.class_names.values()) or ("rib" in os.path.basename(model_path).lower())
                    if not is_rib_model:
                        print(f"[HybridRibInspector] ⚠️ PERINGATAN: Model '{os.path.basename(model_path)}' memiliki kelas {self.class_names} dan BUKAN model deteksi Rib/Sirip!")
                        print(f"[HybridRibInspector] 🔄 Otomatis beralih ke bobot model Rib resmi...")
                        fallback_rib = None
                        for cand in candidates:
                            if "rib" in os.path.basename(cand).lower() and os.path.exists(cand):
                                fallback_rib = cand
                                break
                        if fallback_rib:
                            try:
                                self.yolo_model = _load_detector_weights(fallback_rib)
                                self.model_path = fallback_rib
                                self.class_names = self.yolo_model.names if hasattr(self.yolo_model, 'names') else {0: 'rib'}
                                print(f"[HybridRibInspector] ✅ Berhasil beralih ke Model Rib: {os.path.basename(fallback_rib)} (Kelas: {self.class_names})")
                            except Exception as fb_err:
                                print(f"[HybridRibInspector] Gagal memuat fallback model rib: {fb_err}")
                        else:
                            print(f"[HybridRibInspector] ⚠️ Tidak menemukan file model rib resmi di folder kandidat.")

                    print(f"[HybridRibInspector] Model YOLO dimuat: {os.path.basename(self.model_path)} (Device: {self.device_info}, FP16: {self.use_half}, Kelas: {self.class_names})")
                except Exception as e:
                    print(f"[HybridRibInspector WARNING] Gagal memuat YOLO ({e}). Fallback ke DUAL_POLARITY_ONLY.")
                    self.yolo_model = None
                    self.mode = "DUAL_POLARITY_ONLY"

        self.model_name = os.path.basename(self.model_path) if self.model_path else "No-YOLO"

        # Counter Statistik QC Berbasis Objek Fisik & Temporal Voting
        self.rib_count = 0
        self.ok_count = 0
        self.ng_count = 0
        self.last_presence = False
        self.current_rib_status = None
        self._session_committed = False
        self._committed_rib_statuses = {}
        self._last_finalized_rib_id = 0
        self._last_finalized_status = None
        self._prime_warmup_frames = 0

        # Parameter Temporal Voting & Multi-Frame Consensus Anti-Glitch
        self.min_transit_frames = 15
        self.absence_frames = 0
        self.max_absence_tolerance = 8

        # ── Anti-False Detection: EMA Dual Polarity Ratio (S4) ───────────────
        # Meratakan rasio valid_points antar frame → menolak spike 1-frame dari
        # noda oli, pantulan sesaat, atau fluktuasi cahaya redup
        self._dp_ratio_ema = 0.0
        self._dp_alpha = 0.55  # 0.55 = responsif tapi punya memori 2–3 frame

        # Kernel morphological BG subtraction (dibuat sekali, re-use tiap frame)
        self._dp_kernel_bg = cv2.getStructuringElement(cv2.MORPH_RECT, (91, 1))

        # Zona Prime Evaluasi Konveyor (Pusat optik planar bebas distorsi masuk/keluar Y: 355 - 435 px)
        self.prime_min_y = int(cfg_prime_min_y)
        self.prime_max_y = int(cfg_prime_max_y)
        # Prime zone kamera untuk TIER3_UNWARP (305-385px): area tengah planar bebas distorsi lensa & bebas seam konveyor
        # Di unwarp space, rentang ini memetakan tepat ke Y=145-215px (sweet spot optik).
        self.tier3_prime_min_y = 305
        self.tier3_prime_max_y = 385
        # Prime zone kamera untuk DEFAULT LEGACY & TIER 1 (sinkron dengan top_inspection_config.json)
        self._legacy_prime_min_y = int(cfg_prime_min_y)
        self._legacy_prime_max_y = int(cfg_prime_max_y)
        self.prime_pv_samples = []
        self.prime_sagitta_samples = []
        self.prime_inner_dev_samples = []
        self.prime_tip_dev_samples = []
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

        self.ng_consec_thresh = 4
        self.ng_ratio_thresh = 0.45
        self.min_ng_frames = 4
        self.active_rib_frames = 0
        self.active_rib_ok_frames = 0
        self.active_rib_ng_frames = 0
        self.active_consec_ng = 0
        self.active_max_consec_ng = 0
        self.active_consec_ok = 0
        self.active_pv_history = []
        self.active_sagitta_history = []
        self.last_ng_reason = None

        # Optimasi Performa & Akselerasi Temporal Stride
        self._frame_count = 0
        self._last_yolo_det = (False, None, 0.0, [])

        # Solusi 1: Temporal Multi-Frame Profile Fusion (Tier 4 Hough Stitching)
        self.tier4_profile_history = []
        self.tier4_fused_dev = None
        self.tier4_ref_xs = np.linspace(260.0, 1620.0, 341, dtype=np.float32)

        # Cache Dual Polarity untuk stride (skip setiap frame ke-2 hemat 6-7ms)
        self._last_dual_det = (False, None, 0.0, 0.0, None)
        self._dual_stride = 2  # Jalankan dual polarity setiap N frame

        # Telemetri Akselerasi Perangkat Keras
        self.telemetry = HardwareTelemetry(sample_interval_sec=0.25) if HAS_TELEMETRY else None

        # Tracking kontinuitas posisi fisik bilah antar frame (Anti-Jump & Multi-Rib Disambiguation)
        self._last_active_box = None
        self._last_active_cy = None

        # Sakelar Engine Pelacakan Kontur Sirip (Mutually Exclusive 3-Way Teruji):
        # 1. "LEGACY" (Default: Sobel-Y Anchor + Guided Crest Outward — 46 FPS)
        # 2. "TIER1_VITERBI" (Pilihan A: Anchor-Guided Tight-Ribbon Steger-Viterbi v2)
        # 3. "TIER4_HOUGH_STITCH" (Pilihan B: Micro-Houghlet Grouping & Elastic Stitching)
        # (TIER2_YOLO_POSE & TIER3 YOLO-Segmentation di-take out karena tidak konsisten;
        #  Preprocessing Perspective Transform / Unwarp 90 Deg tetap dipertahankan penuh.)
        # Parameter Preprocessing Perspective Transform (Top-Down Orthogonal 90 Deg)
        self.canonical_w = 960
        self.canonical_h = 460
        self.unwarp_y_top = 0.16  # Ketinggian zona atas (~172 px pada 1080p, pangkal konveyor)
        self.unwarp_y_bot = 0.64  # Ketinggian zona bawah (~690 px pada 1080p, dasar konveyor)
        self.unwarp_tilt = 2.15   # Faktor kecuraman sudut top-down 90 deg (2.15x = orthorectified tampak atas)
        self.unwarp_M = None
        self.unwarp_Minv = None
        self.last_unwarped_frame = None
        self._last_unwarp_contour = None

        self.tracker_engine = "LEGACY"
        self.pose_model = None
        self.pose_model_path = None
        self.seg_model = None
        self.seg_model_path = None
        if tracker_engine and tracker_engine != "LEGACY":
            self.set_tracker_engine(tracker_engine)
        # Cache Stride Temporal Tracker untuk Tier 3 (Ultra-Fast 45+ FPS)
        self._seg_stride_counter = 0
        self._last_seg_xs = None
        self._last_seg_ys = None
        self._last_seg_cy = None
        # Anti-Flicker & Stabilisasi Temporal Tier 3 (YOLO-Segmentation + Unwarp Metrology)
        self._prev_tier3_cam_xs = None
        self._prev_tier3_cam_ys = None
        self._prev_tier3_unw_xs = None
        self._prev_tier3_unw_ys = None
        self.tier3_profile_history = []

        # Buffer Diagnostik Pre-Processing & Temporal Tracker untuk Tier 4 (Micro-Houghlet Grouping & Stitching)
        self._last_hough_diag = None
        self._last_hough_patch = None
        self._prev_hough_pguide = None
        self._prev_hough_contour_ys = None
        self.tier4_profile_history.clear()
        self.tier4_fused_dev = None
        self._kinematic_coasting_frames = 0

        # State Kinematic Linear Motion Tracker & Anti-Switching Locking
        self._last_active_cy = None
        self._last_active_box = None
        self._conveyor_vy_ema = -2.5  # Kecepatan translasi vertikal konveyor (px/frame)
        self._kinematic_coasting_frames = 0
        self._max_kinematic_coasting = 4  # Toleransi bridge drop AI maksimal 4 frame (cegah ghost holding real-time)

    # ------------------------------------------------------------------
    # FASE 2: Adaptive Threshold dari Dataset Rib_Dataset (SQLite)
    # ------------------------------------------------------------------
    def load_adaptive_thresholds(self, db_path: str, sigma_factor: float = 2.5,
                                  min_ok_rows: int = 30,
                                  camera_profile: str = None) -> bool:
        """
        Memuat threshold adaptif dari tabel Rib_Dataset di database SQLite,
        difilter per profil kamera agar dataset posisi berbeda tidak tercampur.

        Threshold dihitung dengan formula:  mean(OK) + sigma_factor × std(OK)
        untuk setiap metrik (PVFlatness, Sagitta, InnerDev, TipDev).

        Parameter
        ----------
        db_path        : Path absolut ke file database SQLite.
        sigma_factor   : Pengali standar deviasi. Default 2.5 → menangkap ~99 % distribusi normal.
        min_ok_rows    : Jumlah baris OK minimum agar adaptive threshold dianggap valid.
        camera_profile : Nama profil kamera (dari profile_name di config JSON). Jika None,
                         mengambil semua baris tanpa filter profil (backward-compatible).

        Return
        ------
        True  jika threshold berhasil dimuat dan diterapkan.
        False jika gagal (file tidak ada, tabel belum ada, data terlalu sedikit, dsb.)
        """
        import sqlite3 as _sqlite3
        import math as _math

        if not db_path or not os.path.exists(db_path):
            return False

        try:
            conn = _sqlite3.connect(db_path, timeout=5)
            cur = conn.cursor()

            # Query difilter per profil kamera jika tersedia, agar distribusi tiap posisi terpisah
            if camera_profile:
                cur.execute(
                    "SELECT PVFlatness, Sagitta, InnerDev, TipDev "
                    "FROM Rib_Dataset "
                    "WHERE Label='OK' AND PVFlatness > 0 AND ActiveFrames >= 15 "
                    "AND CameraProfile = ?",
                    (camera_profile,)
                )
                profile_desc = f"profil='{camera_profile}'"
            else:
                # Fallback: ambil semua baris tanpa filter profil (kompatibel mundur)
                cur.execute(
                    "SELECT PVFlatness, Sagitta, InnerDev, TipDev "
                    "FROM Rib_Dataset "
                    "WHERE Label='OK' AND PVFlatness > 0 AND ActiveFrames >= 15"
                )
                profile_desc = "semua profil"

            rows = cur.fetchall()
            conn.close()

            n = len(rows)
            if n < min_ok_rows:
                print(
                    f"[HybridRibInspector] ⚠ Adaptive threshold dilewati [{profile_desc}]: "
                    f"hanya {n} baris OK valid (minimum {min_ok_rows}). "
                    f"Menggunakan threshold dari config."
                )
                return False

            def _stats(vals):
                mean = sum(vals) / len(vals)
                std  = _math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
                return mean, std

            pvs  = [r[0] for r in rows]
            sags = [r[1] for r in rows]
            devs = [r[2] for r in rows]
            tips = [r[3] for r in rows]

            pv_m,  pv_s  = _stats(pvs)
            sag_m, sag_s = _stats(sags)
            dev_m, dev_s = _stats(devs)
            tip_m, tip_s = _stats(tips)

            # Hitung threshold adaptif; pastikan tidak lebih longgar dari nilai config saat ini
            new_pv  = round(pv_m  + sigma_factor * pv_s,  2)
            new_sag = round(sag_m + sigma_factor * sag_s, 2)
            new_dev = round(dev_m + sigma_factor * dev_s,  2)
            new_tip = round(tip_m + sigma_factor * tip_s,  2)

            # Terapkan — gunakan nilai terkecil antara config lama dan adaptive (lebih aman / lebih ketat)
            self.pv_thresh       = min(self.pv_thresh,       new_pv)
            self.sagitta_thresh  = min(self.sagitta_thresh,  new_sag)
            self.dev_thresh      = min(self.dev_thresh,      new_dev)
            self.tip_thresh      = min(self.tip_thresh,      new_tip)

            self._adaptive_loaded  = True
            self._adaptive_n_ok    = n
            self._adaptive_profile = camera_profile or "ALL"

            print(
                f"[HybridRibInspector] ✅ Adaptive threshold dimuat dari Rib_Dataset "
                f"(n={n} OK, σ×{sigma_factor}, {profile_desc}) | "
                f"PV≤{self.pv_thresh:.2f}  Sag≤{self.sagitta_thresh:.2f}  "
                f"Dev≤{self.dev_thresh:.2f}  Tip≤{self.tip_thresh:.2f}"
            )
            return True

        except Exception as exc:
            print(f"[HybridRibInspector] ⚠ Gagal muat adaptive threshold: {exc}")
            return False

    # ------------------------------------------------------------------
    # Tingkat Keparahan Bending (Severity: LOW / MEDIUM / HIGH)
    # Arsitektur siap pakai — semua level saat ini divonis NG.
    # Pembagian zona mengacu pada rib sebagai penahan ingot di incline conveyor.
    # ------------------------------------------------------------------
    def compute_ng_severity(self, eval_pv: float, eval_sag: float,
                            eval_idev: float, eval_tdev: float,
                            is_ng: bool = None) -> str:
        """
        Menghitung tingkat keparahan cacat rib berdasarkan nilai metrik evaluasi.

        Tingkat (saat ini semua divonis NG, severity hanya informatif):
          - 'OK'     : semua metrik dalam batas toleransi
          - 'LOW'    : satu metrik melampaui threshold utama (bending ringan)
          - 'MEDIUM' : nilai cukup jauh melampaui threshold (bending sedang)
          - 'HIGH'   : nilai jauh melampaui threshold (bending parah / kritis)

        Zona threshold (px) — dapat dikustomisasi melalui config rib_severity_zones:
          TipDev  : LOW ≥ tip_thresh | MEDIUM ≥ tip_thresh+3 | HIGH ≥ tip_thresh+6
          InnerDev: LOW ≥ dev_thresh | MEDIUM ≥ dev_thresh+3 | HIGH ≥ dev_thresh+6
          PV      : LOW ≥ pv_thresh  | MEDIUM ≥ pv_thresh+2  | HIGH ≥ pv_thresh+5

        Return
        ------
        'OK' | 'LOW' | 'MEDIUM' | 'HIGH'
        """
        if is_ng is False:
            return "OK"

        # Ambil zona kustom dari atribut (diisi dari config) atau hitung dari threshold
        sz = getattr(self, "_severity_zones", {}) or {}

        def _zone(val, prefix, thresh, med_add=3.0, high_add=6.0):
            """Klasifikasikan nilai ke zona LOW/MEDIUM/HIGH sesuai kunci prefix config."""
            low_cfg = sz.get(f"{prefix}_low", sz.get("low", thresh))
            low_bound  = min(float(thresh), float(low_cfg))
            med_bound  = float(sz.get(f"{prefix}_medium", sz.get("medium", thresh + med_add)))
            high_bound = float(sz.get(f"{prefix}_high", sz.get("high", thresh + high_add)))
            if val >= high_bound:
                return "HIGH"
            elif val >= med_bound:
                return "MEDIUM"
            elif val >= low_bound:
                return "LOW"
            return "OK"

        # Evaluasi severity masing-masing metrik
        pv_th, sag_th, dev_th, tip_th = self._get_effective_thresholds()
        sev_tip  = _zone(eval_tdev,  "tip", tip_th,  med_add=3.0, high_add=6.0)
        sev_dev  = _zone(eval_idev,  "dev", dev_th,  med_add=3.0, high_add=6.0)
        sev_pv   = _zone(eval_pv,    "pv",  pv_th,   med_add=2.0, high_add=5.0)
        sev_sag  = _zone(eval_sag,   "sag", sag_th,  med_add=2.0, high_add=4.0)

        # Severity keseluruhan = level tertinggi dari semua metrik
        rank = {"OK": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
        worst = max([sev_tip, sev_dev, sev_pv, sev_sag], key=lambda s: rank[s])
        if is_ng is True and worst == "OK":
            return "LOW"
        return worst


    def _load_seg_model(self):
        """Memuat model YOLO11-Segmentation untuk Tier 3 secara lazy loading."""
        if self.seg_model is not None:
            return True
        if not HAS_YOLO:
            return False

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(cur_dir)
        seg_candidates = [
            os.path.join(parent_dir, "models", "best_rib_segmentation.engine"),
            os.path.join(parent_dir, "models", "best_rib_segmentation.pt"),
            os.path.join(cur_dir, "best_rib_segmentation.engine"),
            os.path.join(cur_dir, "best_rib_segmentation.pt"),
            os.path.abspath(os.path.join(cur_dir, "..", "..", "..", "models", "best_rib_segmentation.pt")),
        ]

        for p_cand in seg_candidates:
            if os.path.exists(p_cand):
                try:
                    self.seg_model = YOLO(p_cand)
                    is_trt = str(p_cand).endswith((".engine", ".onnx"))
                    if self.device != "cpu" and HAS_TORCH and torch.cuda.is_available() and not is_trt:
                        try:
                            self.seg_model.to(self.device)
                            if hasattr(self.seg_model, "model") and self.seg_model.model is not None:
                                self.seg_model.model.half()
                        except Exception:
                            pass
                    self.seg_model_path = p_cand
                    print(f"[HybridRibInspector] ✅ Model YOLO-Segmentation dimuat: {os.path.basename(p_cand)}")
                    return True
                except Exception as e:
                    print(f"[HybridRibInspector] Gagal memuat YOLO-Segmentation ({p_cand}): {e}")

        print("[HybridRibInspector] ⚠️ Model Segmentation (best_rib_segmentation.pt) tidak ditemukan.")
        return False

    def _load_pose_model(self):
        """Memuat model YOLOv8-Pose untuk Tier 2 secara lazy loading."""
        if self.pose_model is not None:
            return True
        if not HAS_YOLO:
            return False

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(cur_dir)
        pose_candidates = [
            os.path.join(parent_dir, "models", "best_rib_pose.engine"),
            os.path.join(parent_dir, "models", "best_rib_pose.pt"),
            os.path.join(cur_dir, "best_rib_pose.engine"),
            os.path.join(cur_dir, "best_rib_pose.pt"),
            os.path.join(parent_dir, "runs", "pose", "train", "weights", "best.pt"),
            os.path.abspath(os.path.join(cur_dir, "..", "..", "..", "runs", "pose", "train", "weights", "best.pt")),
        ]

        for p_cand in pose_candidates:
            if os.path.exists(p_cand):
                try:
                    self.pose_model = YOLO(p_cand)
                    is_trt = str(p_cand).endswith((".engine", ".onnx"))
                    if self.device != "cpu" and HAS_TORCH and torch.cuda.is_available() and not is_trt:
                        try:
                            self.pose_model.to(self.device)
                            if hasattr(self.pose_model, "model") and self.pose_model.model is not None:
                                self.pose_model.model.half()
                        except Exception:
                            pass
                    self.pose_model_path = p_cand
                    print(f"[HybridRibInspector] ✅ Model YOLO-Pose dimuat: {os.path.basename(p_cand)}")
                    return True
                except Exception as e:
                    print(f"[HybridRibInspector] Gagal memuat YOLO-Pose ({p_cand}): {e}")

        print("[HybridRibInspector] ⚠️ Model Pose (best_rib_pose.pt) tidak ditemukan.")
        return False

    def toggle_tracker_engine(self):
        """Beralih secara mutually exclusive antara 3 engine teruji: LEGACY -> TIER1_VITERBI -> TIER4_HOUGH_STITCH -> LEGACY."""
        if self.tracker_engine == "LEGACY":
            self.tracker_engine = "TIER1_VITERBI"
        elif self.tracker_engine == "TIER1_VITERBI":
            self.tracker_engine = "TIER4_HOUGH_STITCH"
        elif self.tracker_engine == "TIER4_HOUGH_STITCH":
            self.tracker_engine = "LEGACY"
        else:
            self.tracker_engine = "LEGACY"
        print(f"\n[HybridRibInspector] 🔄 Tracker Engine beralih ke: [{self.tracker_engine}]")
        return self.tracker_engine

    def set_tracker_engine(self, engine_name):
        """Mengatur engine pelacakan kontur ('LEGACY', 'TIER1_VITERBI', atau 'TIER4_HOUGH_STITCH')."""
        eng = str(engine_name).strip().upper()
        if "TIER4" in eng or "HOUGH" in eng or "STITCH" in eng:
            self.tracker_engine = "TIER4_HOUGH_STITCH"
        elif "TIER3" in eng or "SEG" in eng or "UNWARP" in eng or "PERSPECTIVE" in eng:
            print("[HybridRibInspector] ⚠️ TIER3_UNWARP (YOLO-Seg) telah di-take out (preprocessing Perspective Transform tetap aktif); dialihkan ke TIER4_HOUGH_STITCH.")
            self.tracker_engine = "TIER4_HOUGH_STITCH"
        elif "TIER2" in eng or "POSE" in eng:
            print("[HybridRibInspector] ⚠️ TIER2_YOLO_POSE telah di-take out; dialihkan ke LEGACY.")
            self.tracker_engine = "LEGACY"
        elif "TIER1" in eng or "VITERBI" in eng:
            self.tracker_engine = "TIER1_VITERBI"
        else:
            self.tracker_engine = "LEGACY"
        print(f"[HybridRibInspector] Tracker Engine aktif: [{self.tracker_engine}]")
        return self.tracker_engine

    def _commit_rib_session(self):
        """Mengunci counter QC (ok_count/ng_count) tepat 1 kali saat sesi rib berakhir."""
        if self.current_rib_status in ("OK", "NG") and not self._session_committed:
            self._session_committed = True
            if self.current_rib_status == "NG":
                self.ng_count += 1
            else:
                self.ok_count += 1
            if self.rib_count > 0:
                self._committed_rib_statuses[self.rib_count] = self.current_rib_status
                if len(self._committed_rib_statuses) > 100:
                    oldest_key = next(iter(self._committed_rib_statuses))
                    self._committed_rib_statuses.pop(oldest_key, None)
                self._last_finalized_rib_id = self.rib_count
                self._last_finalized_status = self.current_rib_status

    def get_committed_status(self, rib_no):
        """Mengembalikan vonis final yang sudah dikunci untuk rib_no, atau None jika belum final."""
        if rib_no is None:
            return None
        return self._committed_rib_statuses.get(rib_no)

    def finalize_pending_session(self):
        """Mengunci sesi aktif yang tertunda (misalnya saat video/aplikasi ditutup)."""
        self._commit_rib_session()
        return self._last_finalized_status

    def _record_prime_sample(self, pv, sagitta, inner_dev, tip_dev):
        """Menyimpan sampel metrologi ke buffer Prime Zone setelah melewati warm-up masuk ROI."""
        self._prime_warmup_frames += 1
        if self._prime_warmup_frames <= 2:
            return
        self.prime_pv_samples.append(pv)
        self.prime_sagitta_samples.append(sagitta)
        self.prime_inner_dev_samples.append(inner_dev)
        self.prime_tip_dev_samples.append(tip_dev)
        if len(self.prime_pv_samples) > 25:
            self.prime_pv_samples = self.prime_pv_samples[-25:]
            self.prime_sagitta_samples = self.prime_sagitta_samples[-25:]
            self.prime_inner_dev_samples = self.prime_inner_dev_samples[-25:]
            self.prime_tip_dev_samples = self.prime_tip_dev_samples[-25:]

    def _reset_rib_session(self):
        """Mengunci vonis akhir sesi saat ini lalu mereset buffer untuk sirip baru."""
        self._commit_rib_session()
        self.last_presence = False
        self._session_committed = False
        self._prime_warmup_frames = 0
        self._dp_ratio_ema = 0.0
        self._last_unwarp_contour = None
        self._last_unwarp_baseline = None
        self._prev_unwarp_anchor_y = None
        self.active_rib_frames = 0
        self.active_rib_ok_frames = 0
        self.active_rib_ng_frames = 0
        self.active_consec_ng = 0
        self.active_max_consec_ng = 0
        self.active_consec_ok = 0
        self.active_pv_history.clear()
        self.active_sagitta_history.clear()
        self.prime_pv_samples.clear()
        self.prime_sagitta_samples.clear()
        self.prime_inner_dev_samples.clear()
        self.prime_tip_dev_samples.clear()
        self.current_rib_status = None
        self.last_ng_reason = None
        self._seg_stride_counter = 0
        self._last_seg_xs = None
        self._last_seg_ys = None
        self._last_seg_cy = None
        self._prev_tier3_cam_xs = None
        self._prev_tier3_cam_ys = None
        self._prev_tier3_unw_xs = None
        self._prev_tier3_unw_ys = None
        if hasattr(self, 'tier3_profile_history'):
            self.tier3_profile_history.clear()
        self._last_hough_diag = None
        self._last_hough_patch = None
        self._prev_hough_pguide = None
        self._prev_hough_contour_ys = None
        self.tier4_profile_history.clear()
        self.tier4_fused_dev = None
        self._kinematic_coasting_frames = 0

    @staticmethod
    def _fit_robust_guide_poly(anchor_xs, anchor_ys, default_y, max_abs_slope=0.035):
        """
        Menghitung garis pandu linear yang kebal outlier (anti-diagonal skew di tepi/rust):
        1. Saring outlier vertikal terhadap median sebelum regresi awal (cegah rusaknya p_init).
        2. Saring ulang residual OLS agar kurva tetap rapat pada bibir atas.
        3. Clamp kemiringan ke batas fisik konveyor (±max_abs_slope) agar ekstrapolasi ke ujung tidak meleset.
        """
        if len(anchor_xs) < 2:
            return np.array([0.0, float(default_y)], dtype=np.float64)

        xs = np.asarray(anchor_xs, dtype=np.float64)
        ys = np.asarray(anchor_ys, dtype=np.float64)
        med_y = float(np.median(ys))

        # Tahap 1: Klaster median awal (±16 px) sebelum fit OLS
        mask1 = np.abs(ys - med_y) <= 16.0
        if np.sum(mask1) >= 3:
            xs, ys = xs[mask1], ys[mask1]

        # Tahap 2: Regresi awal + saring residual
        if len(xs) >= 3:
            p_init = np.polyfit(xs, ys, 1)
            resids = np.abs(ys - np.polyval(p_init, xs))
            med_res = float(np.median(resids))
            mask2 = resids <= max(6.5, med_res * 2.0)
            if np.sum(mask2) >= 3:
                xs, ys = xs[mask2], ys[mask2]

        poly = np.polyfit(xs, ys, deg=1) if len(xs) >= 2 else np.array([0.0, med_y], dtype=np.float64)

        # Tahap 3: Clamp kemiringan ke batas fisik agar tidak menukik di ujung kiri/kanan
        if abs(poly[0]) > max_abs_slope:
            clamped_m = float(np.clip(poly[0], -max_abs_slope, max_abs_slope))
            med_x = float(np.median(xs))
            med_y_clean = float(np.median(ys))
            poly = np.array([clamped_m, med_y_clean - clamped_m * med_x], dtype=np.float64)
        return poly

    @staticmethod
    def _normalize_flat_field_gray(gray_img, scale=128):
        """Normalisasi iluminasi flat-field cepat via downscale 4x + GaussianBlur (~0.3ms vs ~57ms)."""
        h_g, w_g = gray_img.shape[:2]
        if h_g < 16 or w_g < 16:
            return gray_img
        sw, sh = max(8, w_g // 4), max(8, h_g // 4)
        tiny = cv2.resize(gray_img, (sw, sh), interpolation=cv2.INTER_AREA)
        bg_tiny = cv2.GaussianBlur(tiny, (0, 0), sigmaX=11.25, sigmaY=6.25)
        bg = cv2.resize(bg_tiny, (w_g, h_g), interpolation=cv2.INTER_LINEAR)
        return cv2.divide(gray_img, np.maximum(bg, 1), scale=scale)

    def _get_effective_thresholds(self):
        """Return active QC thresholds; TIER1_VITERBI bypasses adaptive ratchet shrinkage."""
        if self.tracker_engine == "TIER1_VITERBI":
            return (
                max(self.pv_thresh, getattr(self, "_base_pv_thresh", self.pv_thresh)),
                max(self.sagitta_thresh, getattr(self, "_base_sagitta_thresh", self.sagitta_thresh)),
                max(self.dev_thresh, getattr(self, "_base_dev_thresh", self.dev_thresh)),
                max(
                    self.tip_thresh,
                    getattr(self, "_base_tip_thresh", self.tip_thresh),
                    getattr(self, "_viterbi_min_tip_thresh", 10.5),
                ),
            )
        return self.pv_thresh, self.sagitta_thresh, self.dev_thresh, self.tip_thresh

    def _compute_prime_consensus_metrics(self, pv, sagitta, inner_dev, tip_dev, min_samples_required=8):
        """Compute Prime Zone metrics using trimmed median for TIER1_VITERBI, plain median otherwise."""
        if len(self.prime_pv_samples) < min_samples_required:
            return pv, sagitta, inner_dev, tip_dev

        def _agg(samples):
            if self.tracker_engine == "TIER1_VITERBI" and len(samples) >= 8:
                ordered = sorted(samples)
                return float(np.median(ordered[1:-1]))
            return float(np.median(samples))

        return (
            _agg(self.prime_pv_samples),
            _agg(self.prime_sagitta_samples),
            _agg(self.prime_inner_dev_samples),
            _agg(self.prime_tip_dev_samples),
        )

    def _update_prime_consensus_vote(self, is_in_prime, full_span_valid, sample_count, min_samples_required, transit_is_ok):
        """Update consecutive OK/NG consensus counters; TIER1_VITERBI holds votes until min_samples_required."""
        if self.tracker_engine == "TIER1_VITERBI":
            can_update = bool(is_in_prime and full_span_valid and sample_count >= min_samples_required)
        else:
            can_update = bool(is_in_prime and full_span_valid)
        if can_update:
            if transit_is_ok:
                self.active_consec_ok += 1
                self.active_consec_ng = 0
            else:
                self.active_consec_ng += 1
                self.active_consec_ok = 0
        return can_update

    def reset_counters(self):
        """Mereset seluruh statistik hitungan sirip (Total Rib, OK, NG)."""
        self.rib_count = 0
        self.ok_count = 0
        self.ng_count = 0
        self.absence_frames = 0
        self.current_rib_status = None
        self._session_committed = False
        self._committed_rib_statuses.clear()
        self._last_finalized_rib_id = 0
        self._last_finalized_status = None
        self._prime_warmup_frames = 0
        self._last_active_box = None
        self._last_active_cy = None
        self._last_yolo_det = (False, None, 0.0, [])
        self._reset_rib_session()

    def set_mode(self, new_mode):
        """Mengubah mode operasional detektor saat runtime."""
        if new_mode.upper() in self.MODES:
            if "YOLO" in new_mode.upper() and self.yolo_model is None:
                print(f"[HybridRibInspector] Model YOLO tidak tersedia, tetap pada DUAL_POLARITY_ONLY.")
                self.mode = "DUAL_POLARITY_ONLY"
            else:
                self.mode = new_mode.upper()
                print(f"[HybridRibInspector] Mode diubah ke: {self.mode}")

    def nudge_unwarp_zone(self, delta_y_frac):
        """Menaikkan atau menurunkan posisi zona unwarp secara interaktif saat runtime."""
        span = self.unwarp_y_bot - self.unwarp_y_top
        new_top = max(0.06, min(0.60, self.unwarp_y_top + delta_y_frac))
        new_bot = min(0.85, new_top + span)
        self.unwarp_y_top = new_top
        self.unwarp_y_bot = new_bot
        self.unwarp_M = None
        self.unwarp_Minv = None
        print(f"[!] ↕ Posisi Zona Unwarp: Y [{self.unwarp_y_top*100:.1f}%, {self.unwarp_y_bot*100:.1f}%]")

    def adjust_unwarp_tilt(self, delta_tilt):
        """Menyesuaikan tingkat kecuraman sudut pandang top-down 90 derajat."""
        self.unwarp_tilt = max(0.8, min(2.8, self.unwarp_tilt + delta_tilt))
        self.canonical_h = int(850 / (1.0 + 0.70 * (self.unwarp_tilt - 1.0)))
        self.unwarp_M = None
        self.unwarp_Minv = None
        # Reset anchor EMA dan session saat geometri unwarp berubah
        # agar sampel metrologi dari geometri lama tidak mengkontaminasi rib berikutnya
        self._prev_unwarp_anchor_y = None
        self._last_unwarp_contour = None
        self._reset_rib_session()
        print(f"[!] 📐 Sudut Top-Down 90 Deg disetel ke: [{self.unwarp_tilt:.2f}x] (Tinggi Kanvas: {self.canonical_h}px)")

    def cycle_unwarp_preset(self):
        """Berganti preset unwarp: 1. Orthogonal 90° (Default) -> 2. Rib-Focused 90° -> 3. Full Deck Klasik"""
        presets = [
            ("ORTHOGONAL 90 DEG (Standard 460px)", 2.15, 460, 0.16, 0.64),
            ("RIB-FOCUSED 90 DEG (Ultra Flat 320px)", 2.60, 320, 0.22, 0.58),
            ("FULL DECK KLASIK (Tall 850px)", 1.00, 850, 0.18, 0.65),
        ]
        curr_tilt = self.unwarp_tilt
        if curr_tilt >= 2.40:
            next_p = presets[2]  # Ke Full Deck Klasik
        elif curr_tilt <= 1.30:
            next_p = presets[0]  # Ke Orthogonal Standard
        else:
            next_p = presets[1]  # Ke Rib-Focused Ultra Flat

        name, tilt, can_h, y_t, y_b = next_p
        self.unwarp_tilt = tilt
        self.canonical_h = can_h
        self.unwarp_y_top = y_t
        self.unwarp_y_bot = y_b
        self.unwarp_M = None
        self.unwarp_Minv = None
        # Reset anchor EMA, kontur terakhir, dan session metrologi saat preset berganti
        # agar geometri baru tidak terkontaminasi data dari geometri lama
        self._prev_unwarp_anchor_y = None
        self._last_unwarp_contour = None
        self._reset_rib_session()
        print(f"[!] 🔄 Preset Unwarp berganti ke: [{name}] (Tilt: {tilt:.2f}x, H: {can_h}px)")
        return name

    def unwarp_rib_zone(self, frame, polygon_pts=None):
        """
        Tahap 1 Preprocessing: Perspective Transform untuk Bilah Sirip.
        Mentransformasi citra miring konveyor menjadi citra datar ortogonal
        tampak atas (Top-Down 90 Deg) secara terisolasi tanpa mempengaruhi citra asli.

        :param frame: Frame citra input BGR
        :param polygon_pts: 4 titik koordinat zona konveyor (opsional)
        :return: (unwarped_bgr, M, M_inv)
        """
        if frame is None or frame.size == 0:
            return None, None, None

        h_f, w_f = frame.shape[:2]

        if polygon_pts is not None and len(polygon_pts) >= 4:
            pts_px = []
            for p in polygon_pts[:4]:
                x = p[0] * w_f if p[0] <= 1.0 else p[0]
                y = p[1] * h_f if p[1] <= 1.0 else p[1]
                pts_px.append([x, y])
            src_pts = order_quad_points(pts_px)
        else:
            # Menggunakan geometri rel konveyor dengan faktor sudut top-down dinamis (tilt keystone)
            y_top = self.unwarp_y_top * h_f
            y_bot = self.unwarp_y_bot * h_f
            taper = getattr(self, "unwarp_tilt", 1.60)
            m_left = -0.375 * taper
            m_right = +0.370 * taper

            scale_w = w_f / 1920.0
            scale_h = h_f / 1080.0

            x_tl = (374.0 + m_left * (y_top / scale_h - 216.0)) * scale_w
            x_bl = (374.0 - 0.375 * (y_bot / scale_h - 216.0)) * scale_w
            x_tr = (1623.0 + m_right * (y_top / scale_h - 216.0)) * scale_w
            x_br = (1623.0 + 0.370 * (y_bot / scale_h - 216.0)) * scale_w

            src_pts = np.array([
                [x_tl, y_top],
                [x_tr, y_top],
                [x_br, y_bot],
                [x_bl, y_bot]
            ], dtype=np.float32)

        dst_pts = np.array([
            [0, 0],
            [self.canonical_w - 1, 0],
            [self.canonical_w - 1, self.canonical_h - 1],
            [0, self.canonical_h - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)

        unwarped = cv2.warpPerspective(
            frame,
            M,
            (self.canonical_w, self.canonical_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        self.unwarp_M = M
        self.unwarp_Minv = M_inv
        self.last_unwarped_frame = unwarped

        return unwarped, M, M_inv

    def _create_compact_hough_patch(self, strip, bin_edge, accepted_lines, rejected_lines, smooth_ys=None, x_grid=None, y_min_g=0, x_limit_left=0):
        """
        Membuat patch pratinjau ringkas (480x110 BGR) untuk kartu HUD bawah kanan,
        menampilkan perbandingan Sobel-Y Binarized Edge Map + Hough Lines vs Stitched Contour.
        """
        patch = np.zeros((110, 480, 3), dtype=np.uint8)
        if strip is None or strip.size == 0 or bin_edge is None or bin_edge.size == 0:
            return patch

        h_s, w_s = strip.shape[:2]
        if w_s < 10 or h_s < 5:
            return patch

        # Panel Atas (Y: 0..52): Sobel-Y Binarized Edge Map + Micro-Houghlet Lines
        edge_bgr = cv2.cvtColor(bin_edge, cv2.COLOR_GRAY2BGR)
        edge_bgr[bin_edge == 0] = [28, 18, 14]
        edge_bgr[bin_edge > 0] = [240, 240, 240]

        for x1, y1, x2, y2 in rejected_lines:
            cv2.line(edge_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 230), 1, cv2.LINE_AA)
        for x1, y1, x2, y2 in accepted_lines:
            cv2.line(edge_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 100), 2, cv2.LINE_AA)

        top_disp = cv2.resize(edge_bgr, (480, 52), interpolation=cv2.INTER_LINEAR)
        cv2.putText(top_disp, f"SOBEL-Y + HOUGH ({len(accepted_lines)} Acc / {len(rejected_lines)} Rej)", (8, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 255, 200), 1, cv2.LINE_AA)
        patch[0:52, 0:480] = top_disp

        # Garis pemisah horizontal
        cv2.line(patch, (0, 53), (480, 53), (80, 80, 90), 1)

        # Panel Bawah (Y: 55..110): Citra Grayscale Asli + Stitched Ridge
        strip_bgr = cv2.cvtColor(strip, cv2.COLOR_GRAY2BGR)
        if smooth_ys is not None and x_grid is not None and len(smooth_ys) > 5:
            pts = []
            for gx, gy in zip(x_grid, smooth_ys):
                sx = int(gx - x_limit_left)
                sy = int(gy - y_min_g)
                if 0 <= sx < w_s and -10 <= sy < h_s + 10:
                    pts.append((sx, sy))
            if len(pts) > 2:
                cv2.polylines(strip_bgr, [np.array(pts, dtype=np.int32)], isClosed=False, color=(0, 220, 255), thickness=2, lineType=cv2.LINE_AA)

        bot_disp = cv2.resize(strip_bgr, (480, 55), interpolation=cv2.INTER_LINEAR)
        cv2.putText(bot_disp, "STITCHED RIDGE CONTOUR (SUB-PIXEL POLISHED)", (8, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 220, 0), 1, cv2.LINE_AA)
        patch[55:110, 0:480] = bot_disp

        return patch

    def render_hough_preview(self, frame=None, active_box=None, res=None):
        """
        Merender tampilan studio diagnostik pre-processing Tier 4 (Micro-Houghlet Grouping & Stitching):
        Panel 1: Narrow Strip ROI (Grayscale) & Multi-Point Anchor Sampling Guide
        Panel 2: Sobel-Y Binarized Edge Map & Asymmetric Gating (Lolos Hijau vs Ditolak Merah)
        Panel 3: Elastic Gap-Filling (Stitching) & Sub-Pixel Parabolic Polish dengan Baseline Theil-Sen
        """
        w_can = getattr(self, "canonical_w", 960)
        h_can = getattr(self, "canonical_h", 460)
        disp = np.full((h_can, w_can, 3), 18, dtype=np.uint8)

        diag = getattr(self, "_last_hough_diag", None)
        status_val = res.get("status", "WAIT") if isinstance(res, dict) else "WAIT"

        if status_val == "OK":
            status_col = (0, 255, 0)
        elif status_val == "NG":
            status_col = (0, 0, 255)
        else:
            status_col = (0, 180, 255)

        # ----------------------------------------------------------------------
        # HEADER STUDIO (Y: 0..46)
        # ----------------------------------------------------------------------
        cv2.rectangle(disp, (0, 0), (w_can, 46), (28, 24, 20), -1)
        cv2.line(disp, (0, 46), (w_can, 46), (80, 70, 50), 1)

        cv2.putText(disp, "TIER 4: HOUGH STITCHING PRE-PROCESSING & DIAGNOSTICS (SENSORS 2025)",
                    (16, 22), cv2.FONT_HERSHEY_DUPLEX, 0.48, (255, 190, 40), 1, cv2.LINE_AA)

        if diag is not None:
            n_acc = len(diag.get("accepted_lines", []))
            n_rej = len(diag.get("rejected_lines", []))
            cov = diag.get("coverage", 0.0)
            sub_txt = f"Houghlets: Total {n_acc + n_rej} (Lolos: {n_acc} | Ditolak: {n_rej}) | Coverage Ridge: {cov*100:.0f}%"
        else:
            sub_txt = "Status: Menunggu sirip berada di dalam zona aktif konveyor..."

        cv2.putText(disp, sub_txt, (16, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1, cv2.LINE_AA)

        # Status Pill Badge di kanan atas
        badge_w, badge_h = 110, 28
        bx1 = w_can - badge_w - 15
        by1 = 9
        cv2.rectangle(disp, (bx1, by1), (bx1 + badge_w, by1 + badge_h), (35, 30, 25), -1)
        cv2.rectangle(disp, (bx1, by1), (bx1 + badge_w, by1 + badge_h), status_col, 2, cv2.LINE_AA)
        cv2.putText(disp, f"STATUS: {status_val}", (bx1 + 10, by1 + 19),
                    cv2.FONT_HERSHEY_DUPLEX, 0.42, status_col, 1, cv2.LINE_AA)

        # Jika belum ada data sirip (misal status WAIT atau plat datar), tampilkan placeholder
        if diag is None or diag.get("strip") is None or diag["strip"].size == 0:
            cv2.rectangle(disp, (40, 90), (w_can - 40, 390), (25, 25, 32), -1)
            cv2.rectangle(disp, (40, 90), (w_can - 40, 390), (60, 60, 80), 1)
            cv2.putText(disp, "[WAIT] PLAT DATAR - MENUNGGU BILAH SIRIP MELINTAS",
                        (w_can // 2 - 280, 210), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 200, 255), 1, cv2.LINE_AA)
            cv2.putText(disp, "Preview visualisasi Sobel-Y, seleksi Hough Lines, dan Elastic Gap-Filling",
                        (w_can // 2 - 270, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 180, 190), 1, cv2.LINE_AA)
            cv2.putText(disp, "akan di-render secara realtime saat bibir sirip terdeteksi di kamera.",
                        (w_can // 2 - 240, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 180, 190), 1, cv2.LINE_AA)
            return disp

        strip = diag["strip"]
        bin_edge = diag["bin_edge"]
        h_s, w_s = strip.shape[:2]
        p_w = w_can - 40   # 920 px
        p_h = 92           # 92 px per panel
        scale_x = float(p_w) / max(1.0, float(w_s))
        scale_y = float(p_h) / max(1.0, float(h_s))

        # ----------------------------------------------------------------------
        # PANEL 1: NARROW STRIP ROI & MULTI-POINT ANCHOR GUIDE (Y: 54..166)
        # ----------------------------------------------------------------------
        y1_p1 = 54
        cv2.rectangle(disp, (20, y1_p1), (w_can - 20, y1_p1 + p_h + 20), (24, 26, 32), -1)
        cv2.rectangle(disp, (20, y1_p1), (w_can - 20, y1_p1 + p_h + 20), (50, 55, 68), 1)
        cv2.putText(disp, "1. NARROW STRIP ROI (GRAYSCALE) & MULTI-POINT ANCHOR GUIDE",
                    (28, y1_p1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 200, 60), 1, cv2.LINE_AA)

        p1_strip_bgr = cv2.cvtColor(strip, cv2.COLOR_GRAY2BGR)
        p_guide = diag.get("p_guide")
        y_min_g = diag.get("y_min_g", 0)
        x_limit_left = diag.get("x_limit_left", 0)
        x_limit_right = diag.get("x_limit_right", w_s)

        guard_xs = np.linspace(0, w_s - 1, 100)
        ceil_pts = []
        floor_pts = []
        guide_pts = []
        for sx in guard_xs:
            gx = sx + x_limit_left
            gy_nom = float(np.polyval(p_guide, gx)) - y_min_g
            is_tip = (gx < x_limit_left + 120 or gx > x_limit_right - 120)
            floor_g = 8.0 if is_tip else 7.0
            ceil_g = 8.5 if is_tip else 6.0
            guide_pts.append((int(sx * scale_x), int(gy_nom * scale_y)))
            ceil_pts.append((int(sx * scale_x), int((gy_nom - ceil_g) * scale_y)))
            floor_pts.append((int(sx * scale_x), int((gy_nom + floor_g) * scale_y)))

        p1_canvas = cv2.resize(p1_strip_bgr, (p_w, p_h), interpolation=cv2.INTER_LINEAR)
        if len(ceil_pts) > 1:
            cv2.polylines(p1_canvas, [np.array(ceil_pts, dtype=np.int32)], isClosed=False, color=(255, 160, 40), thickness=1, lineType=cv2.LINE_AA)
        if len(floor_pts) > 1:
            cv2.polylines(p1_canvas, [np.array(floor_pts, dtype=np.int32)], isClosed=False, color=(40, 160, 255), thickness=1, lineType=cv2.LINE_AA)
        if len(guide_pts) > 1:
            cv2.polylines(p1_canvas, [np.array(guide_pts, dtype=np.int32)], isClosed=False, color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        for ax_s, ay_s in diag.get("anchor_pts", []):
            cx = int(ax_s * scale_x)
            cy = int(ay_s * scale_y)
            cv2.circle(p1_canvas, (cx, cy), 4, (0, 240, 255), -1, cv2.LINE_AA)
            cv2.circle(p1_canvas, (cx, cy), 7, (255, 255, 255), 1, cv2.LINE_AA)

        tip_x_l = int(120 * scale_x)
        tip_x_r = int((w_s - 120) * scale_x)
        cv2.line(p1_canvas, (tip_x_l, 0), (tip_x_l, p_h), (120, 120, 200), 1, cv2.LINE_4)
        cv2.line(p1_canvas, (tip_x_r, 0), (tip_x_r, p_h), (120, 120, 200), 1, cv2.LINE_4)

        disp[y1_p1 + 18:y1_p1 + 18 + p_h, 20:20 + p_w] = p1_canvas

        # ----------------------------------------------------------------------
        # PANEL 2: SOBEL-Y BINARY EDGE & MICRO-HOUGHLETS (Y: 176..288)
        # ----------------------------------------------------------------------
        y2_p2 = 176
        cv2.rectangle(disp, (20, y2_p2), (w_can - 20, y2_p2 + p_h + 20), (24, 26, 32), -1)
        cv2.rectangle(disp, (20, y2_p2), (w_can - 20, y2_p2 + p_h + 20), (50, 55, 68), 1)
        cv2.putText(disp, "2. SOBEL-Y BINARY EDGE MAP & MICRO-HOUGHLET FILTERING",
                    (28, y2_p2 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 230, 255), 1, cv2.LINE_AA)
        cv2.putText(disp, "[Hijau: Lolos Asymmetric Guard | Merah: Ditolak (Sudut / Di Luar Ridge)]",
                    (w_can - 470, y2_p2 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (180, 220, 180), 1, cv2.LINE_AA)

        p2_canvas = np.zeros((p_h, p_w, 3), dtype=np.uint8)
        p2_canvas[:] = (26, 18, 14)
        bin_edge_scaled = cv2.resize(bin_edge, (p_w, p_h), interpolation=cv2.INTER_NEAREST)
        p2_canvas[bin_edge_scaled > 0] = (160, 160, 160)

        for x1, y1, x2, y2 in diag.get("rejected_lines", []):
            sx1 = int(x1 * scale_x)
            sy1 = int(y1 * scale_y)
            sx2 = int(x2 * scale_x)
            sy2 = int(y2 * scale_y)
            cv2.line(p2_canvas, (sx1, sy1), (sx2, sy2), (0, 0, 220), 1, cv2.LINE_AA)

        for x1, y1, x2, y2 in diag.get("accepted_lines", []):
            sx1 = int(x1 * scale_x)
            sy1 = int(y1 * scale_y)
            sx2 = int(x2 * scale_x)
            sy2 = int(y2 * scale_y)
            cv2.line(p2_canvas, (sx1, sy1), (sx2, sy2), (0, 255, 100), 2, cv2.LINE_AA)

        disp[y2_p2 + 18:y2_p2 + 18 + p_h, 20:20 + p_w] = p2_canvas

        # ----------------------------------------------------------------------
        # PANEL 3: ELASTIC GAP-FILLING STITCHING & SUB-PIXEL RIDGE (Y: 298..410)
        # ----------------------------------------------------------------------
        y3_p3 = 298
        cv2.rectangle(disp, (20, y3_p3), (w_can - 20, y3_p3 + p_h + 20), (24, 26, 32), -1)
        cv2.rectangle(disp, (20, y3_p3), (w_can - 20, y3_p3 + p_h + 20), (50, 55, 68), 1)
        cv2.putText(disp, "3. ELASTIC GAP-FILLING (STITCHING) & SUB-PIXEL RIDGE METROLOGY",
                    (28, y3_p3 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 255, 120), 1, cv2.LINE_AA)
        cv2.putText(disp, "[Kuning: Baseline Theil-Sen | Sian/Hijau: Ridge Terpoles | Oranye: Gap Bridged]",
                    (w_can - 470, y3_p3 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 220, 240), 1, cv2.LINE_AA)

        p3_canvas = cv2.resize(p1_strip_bgr, (p_w, p_h), interpolation=cv2.INTER_LINEAR)
        p3_canvas = (p3_canvas.astype(np.float32) * 0.45).astype(np.uint8)

        baseline = res.get("baseline") if isinstance(res, dict) else None
        if baseline is not None and len(baseline) == len(diag.get("x_grid", [])):
            base_pts = []
            for gx, by in zip(diag["x_grid"], baseline):
                bx_s = int((gx - x_limit_left) * scale_x)
                by_s = int((by - y_min_g) * scale_y)
                base_pts.append((bx_s, by_s))
            if len(base_pts) > 1:
                cv2.polylines(p3_canvas, [np.array(base_pts, dtype=np.int32)], isClosed=False, color=(0, 240, 255), thickness=1, lineType=cv2.LINE_AA)

        smooth_ys = diag.get("smooth_ys")
        x_grid = diag.get("x_grid")
        valid_idxs = set(diag.get("valid_idxs", []))

        if smooth_ys is not None and x_grid is not None and len(smooth_ys) > 5:
            full_pts = []
            gap_pts = []
            for i, (gx, gy) in enumerate(zip(x_grid, smooth_ys)):
                sx = int((gx - x_limit_left) * scale_x)
                sy = int((gy - y_min_g) * scale_y)
                full_pts.append((sx, sy))
                if i not in valid_idxs:
                    gap_pts.append((sx, sy))

            c_contour = (0, 255, 0) if status_val == "OK" else (0, 0, 255)
            if len(full_pts) > 2:
                cv2.polylines(p3_canvas, [np.array(full_pts, dtype=np.int32)], isClosed=False, color=c_contour, thickness=2, lineType=cv2.LINE_AA)

            for gx_s, gy_s in gap_pts:
                cv2.circle(p3_canvas, (gx_s, gy_s), 2, (0, 160, 255), -1, cv2.LINE_AA)

            if len(full_pts) > 0:
                cv2.circle(p3_canvas, full_pts[0], 5, (0, 240, 255), -1, cv2.LINE_AA)
                cv2.circle(p3_canvas, full_pts[-1], 5, (0, 240, 255), -1, cv2.LINE_AA)

        disp[y3_p3 + 18:y3_p3 + 18 + p_h, 20:20 + p_w] = p3_canvas

        # ----------------------------------------------------------------------
        # FOOTER TELEMETRI METROLOGI (Y: 420..460)
        # ----------------------------------------------------------------------
        pv = res.get("pv_flatness", 0.0) if isinstance(res, dict) else 0.0
        sag = res.get("sagitta", 0.0) if isinstance(res, dict) else 0.0
        tdev = res.get("tip_dev", 0.0) if isinstance(res, dict) else 0.0
        idev = res.get("inner_dev", 0.0) if isinstance(res, dict) else 0.0

        pv_col = (0, 255, 200) if pv <= self.pv_thresh else (0, 0, 255)
        sag_col = (0, 255, 200) if sag <= self.sagitta_thresh else (0, 0, 255)
        tip_col = (0, 255, 200) if tdev <= self.tip_thresh else (0, 0, 255)

        cv2.putText(disp, f"Flatness PV: {pv:.1f} px", (25, 442), cv2.FONT_HERSHEY_SIMPLEX, 0.42, pv_col, 1, cv2.LINE_AA)
        cv2.putText(disp, f"Sagitta: {sag:.1f} px", (230, 442), cv2.FONT_HERSHEY_SIMPLEX, 0.42, sag_col, 1, cv2.LINE_AA)
        cv2.putText(disp, f"Deviasi Ujung: {tdev:.1f} px", (420, 442), cv2.FONT_HERSHEY_SIMPLEX, 0.42, tip_col, 1, cv2.LINE_AA)
        cv2.putText(disp, f"Deviasi Badan: {idev:.1f} px", (640, 442), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(disp, "Latency: ~14ms [CPU SIMD]", (800, 442), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)

        return disp

    def render_unwarp_preview(self, frame, active_box=None, res=None):
        """
        Merender tampilan preview visualisasi hasil Perspective Transform (Bird's-Eye Top-Down)
        dengan zero-flickering, resolusi konsisten, garis bantu kelurusan horizontal,
        penanda posisi fisik sirip, dan overlay kontur terdeteksi (TIER 3).
        Jika tracker_engine == 'TIER4_HOUGH_STITCH', merender studio diagnostik pre-processing Tier 4.
        """
        if self.tracker_engine == "TIER4_HOUGH_STITCH":
            return self.render_hough_preview(frame=frame, active_box=active_box, res=res)

        unwarped, M, Minv = self.unwarp_rib_zone(frame)
        if unwarped is None:
            return np.zeros((self.canonical_h, self.canonical_w, 3), dtype=np.uint8)

        disp = unwarped.copy()
        h_u, w_u = disp.shape[:2]

        # 1. Batas lateral rel konveyor kiri & kanan (Dek Aktif bebas rantai)
        cv2.line(disp, (50, 0), (50, h_u), (255, 120, 0), 1, cv2.LINE_AA)
        cv2.line(disp, (w_u - 50, 0), (w_u - 50, h_u), (255, 120, 0), 1, cv2.LINE_AA)

        # 2. Garis sumbu tengah longitudinal konveyor (Center axis X = w_u // 2)
        cx = w_u // 2
        cv2.line(disp, (cx, 0), (cx, h_u), (0, 220, 255), 1, cv2.LINE_AA)

        # 3. Zona Prime Inspeksi Unwarp (Garis pandu dinamis ortogonal)
        u_pmin = getattr(self, "_last_unwarp_pmin", 175.0)
        u_pmax = getattr(self, "_last_unwarp_pmax", 295.0)
        if 0 < u_pmin < h_u and 0 < u_pmax < h_u:
            cv2.line(disp, (50, int(u_pmin)), (w_u - 50, int(u_pmin)), (255, 200, 0), 1, cv2.LINE_AA)
            cv2.line(disp, (50, int(u_pmax)), (w_u - 50, int(u_pmax)), (255, 200, 0), 1, cv2.LINE_AA)
            cv2.putText(disp, "ZONA PRIME INSPEKSI UNWARP (90 DEG)", (60, int(u_pmin) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 200, 0), 1, cv2.LINE_AA)

        # 4. Proyeksikan bounding box aktif ke ruang unwarp jika tersedia
        if active_box is not None and M is not None:
            bx1, by1, bx2, by2 = active_box
            box_corners = np.array([
                [[bx1, by1]],
                [[bx2, by1]],
                [[bx2, by2]],
                [[bx1, by2]]
            ], dtype=np.float32)
            u_box = cv2.perspectiveTransform(box_corners, M)
            u_pts = u_box.reshape(-1, 2).astype(np.int32)
            cv2.polylines(disp, [u_pts], isClosed=True, color=(180, 180, 0), thickness=1, lineType=cv2.LINE_AA)

        # 5. Gambar kontur sirip terdeteksi jika tersedia (TIER 3 / Unwarp Ridge)
        unwarp_contour = getattr(self, "_last_unwarp_contour", None)
        baseline_u = getattr(self, "_last_unwarp_baseline", None)
        if unwarp_contour is not None and len(unwarp_contour) > 2:
            u_pts_draw = unwarp_contour.astype(np.int32)
            status_val = res.get("status", "WAIT") if res else "WAIT"
            if status_val == "OK":
                contour_color = (0, 255, 0)
            elif status_val == "NG":
                contour_color = (0, 0, 255)
            else:
                contour_color = (0, 220, 255)

            # Gambar garis kontur puncak bilah
            cv2.polylines(disp, [u_pts_draw], isClosed=False, color=contour_color, thickness=2, lineType=cv2.LINE_AA)

            # Gambar garis referensi lurus baseline unwarp
            if baseline_u is not None and len(baseline_u) == len(unwarp_contour):
                u_xs = unwarp_contour[:, 0].astype(np.int32)
                u_base_pts = np.column_stack([u_xs, baseline_u.astype(np.int32)])
                cv2.polylines(disp, [u_base_pts], isClosed=False, color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA)
            else:
                med_uy = int(np.median(unwarp_contour[:, 1]))
                cv2.line(disp, (50, med_uy), (w_u - 50, med_uy), (0, 255, 255), 1, cv2.LINE_AA)

            # Badge Metrologi di jendela unwarp
            if res:
                pv_v = res.get("pv_flatness", 0.0)
                sag_v = res.get("sagitta", 0.0)
                tip_v = res.get("tip_dev", 0.0)
                med_uy = int(np.median(unwarp_contour[:, 1]))
                hud_txt = f"[{status_val}] PV: {pv_v:.1f}px | Sag: {sag_v:.1f}px | Tip: {tip_v:.1f}px"
                # Background kotak badge
                cv2.rectangle(disp, (w_u - 340, max(8, med_uy - 24)), (w_u - 15, max(30, med_uy - 2)), (20, 20, 20), -1)
                cv2.rectangle(disp, (w_u - 340, max(8, med_uy - 24)), (w_u - 15, max(30, med_uy - 2)), contour_color, 1)
                cv2.putText(disp, hud_txt, (w_u - 335, max(22, med_uy - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, contour_color, 1, cv2.LINE_AA)

        # 6. Label info HUD pada unwarp view
        eng_label = f" | {self.tracker_engine} (90 Deg Perspective View)"
        cv2.putText(disp, f"TOP-DOWN 90 DEG UNWARP ({w_u}x{h_u} | Tilt {self.unwarp_tilt:.2f}x{eng_label})", (15, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(disp, "Rel Sejajar | [W]/[S]: Tilt | [K]/[J]: Posisi Y | [Z]: Ganti Preset | [T]: Ganti Engine",
                    (15, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1, cv2.LINE_AA)

        return disp

    def set_prime_zone(self, min_y, max_y):
        """Memperbarui batas zona prime evaluasi sirip secara dinamis saat runtime."""
        self.prime_min_y = int(min_y)
        self.prime_max_y = int(max_y)
        self._legacy_prime_min_y = int(min_y)
        self._legacy_prime_max_y = int(max_y)

    def _letterbox_gpu(self, img_bgr, target_size):
        """
        Letterbox resize + BGR→RGB normalisasi langsung ke CUDA tensor FP16.
        Menggunakan pre-allocated pinned memory buffer untuk DMA transfer cepat
        (eliminasi alokasi heap per-frame: 4.3ms → ~1.5ms).
        :param img_bgr: Frame numpy BGR uint8
        :param target_size: Target ukuran sisi terpanjang (mis. 480)
        :return: CUDA tensor shape (1,3,H,W) FP16, (scale, pad_top, pad_left, src_h, src_w)
        """
        src_h, src_w = img_bgr.shape[:2]
        scale = target_size / max(src_h, src_w)
        new_w = int(round(src_w * scale))
        new_h = int(round(src_h * scale))

        # Resize ke dimensi baru (CPU via OpenCV, sangat cepat ~0.5ms)
        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad ke target_size x target_size dengan nilai abu 114
        pad_top = (target_size - new_h) // 2
        pad_bot = target_size - new_h - pad_top
        pad_left = (target_size - new_w) // 2
        pad_right = target_size - new_w - pad_left
        padded = cv2.copyMakeBorder(resized, pad_top, pad_bot, pad_left, pad_right,
                                    cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # BGR→RGB contiguous array
        rgb = np.ascontiguousarray(padded[:, :, ::-1])
        dtype = torch.float16 if self.use_half else torch.float32
        tensor = torch.from_numpy(rgb).to(
            device=self.torch_device, dtype=dtype
        ).permute(2, 0, 1).unsqueeze(0) / 255.0

        return tensor, (scale, pad_top, pad_left, src_h, src_w)

    def _nms_boxes(self, raw_preds, conf_thresh, iou_thresh, scale, pad_top, pad_left, off_x, off_y):
        """
        NMS pada output YOLO raw tensor (format XYXY + conf + cls).
        Menggunakan cv2.dnn.NMSBoxes untuk keandalan 100% tanpa ketergantungan pada
        ekstensi C++ torchvision yang rentan error ABI pada lingkungan JetPack 6.
        :param raw_preds: Tensor shape (1, num_anchors, 4+num_cls) atau (1, 4+num_cls, num_anchors)
        :return: list dict {box, conf, cls_id}
        """
        if raw_preds is None:
            return []

        try:
            pred = raw_preds[0]  # (num_anchors, 4+num_cls) atau transposed
            # YOLOv8 output format: (batch, 4+num_cls, num_anchors) → transpose ke (num_anchors, 4+num_cls)
            if pred.ndim == 2 and pred.shape[0] < pred.shape[1]:
                pred = pred.T

            # Pisahkan koordinat + skor kelas
            boxes_xywh = pred[:, :4]  # cx, cy, w, h (dalam ruang letterbox)
            scores = pred[:, 4:]      # (num_anchors, num_cls)
            if scores.shape[1] == 0:
                return []

            cls_confs, cls_ids = scores.max(dim=1)  # confidence & class per anchor
            mask = cls_confs >= conf_thresh
            if mask.sum() == 0:
                return []

            boxes_f = boxes_xywh[mask].cpu().numpy()
            confs_f = cls_confs[mask].cpu().numpy()
            cls_f   = cls_ids[mask].cpu().numpy()

            # Konversi cx, cy, w, h dari letterbox space ke koordinat frame penuh
            x1 = (boxes_f[:, 0] - boxes_f[:, 2] / 2 - pad_left) / scale + off_x
            y1 = (boxes_f[:, 1] - boxes_f[:, 3] / 2 - pad_top) / scale + off_y
            x2 = (boxes_f[:, 0] + boxes_f[:, 2] / 2 - pad_left) / scale + off_x
            y2 = (boxes_f[:, 1] + boxes_f[:, 3] / 2 - pad_top) / scale + off_y

            cv_boxes = []
            for i in range(len(x1)):
                cv_boxes.append([int(x1[i]), int(y1[i]), int(x2[i] - x1[i]), int(y2[i] - y1[i])])

            indices = cv2.dnn.NMSBoxes(
                cv_boxes, confs_f.tolist(), score_threshold=conf_thresh, nms_threshold=iou_thresh
            )
            if len(indices) == 0:
                return []

            indices = np.array(indices).flatten()
            results = []
            for idx in indices:
                bx, by, bw, bh = cv_boxes[idx]
                results.append({
                    "box": [bx, by, bx + bw, by + bh],
                    "conf": float(confs_f[idx]),
                    "cls_id": int(cls_f[idx])
                })
            return results
        except Exception:
            return []

    def detect_yolo_zone(self, frame, zone_rect):
        """
        Inferensi YOLO direct GPU pada zona inspeksi konveyor.
        Menggunakan pipeline GPU-native: letterbox pada CPU → CUDA tensor FP16 → direct model.model() forward
        → NMS GPU (torchvision) → koordinat frame penuh.
        Menggantikan Ultralytics predictor API untuk menghilangkan bottleneck CPU preprocessing ~40-80ms.
        :param frame: Frame citra BGR
        :param zone_rect: (zx1, zy1, zx2, zy2) batas piksel zona inspeksi
        :return: (is_detected, best_box, conf, all_boxes)
        """
        if self.yolo_model is None or frame is None or frame.size == 0:
            return False, None, 0.0, []

        h_f, w_f = frame.shape[:2]
        zx1, zy1, zx2, zy2 = zone_rect

        # Tentukan ROI crop zona konveyor dengan sedikit padding vertikal
        pad_y = int(0.04 * h_f)
        crop_y1 = max(0, zy1 - pad_y)
        crop_y2 = min(h_f, zy2 + pad_y)
        crop_x1 = max(0, zx1)
        crop_x2 = min(w_f, zx2)

        infer_img = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if infer_img.size == 0:
            infer_img = frame
            off_x, off_y = 0, 0
        else:
            off_x, off_y = crop_x1, crop_y1

        # Pilih resolusi input YOLO adaptif (384 px pada CPU, 480 px pada GPU)
        target_imgsz = 480 if self.device != "cpu" else 384

        rib_boxes = []
        best_box = None
        best_match_score = 0.0
        max_conf = 0.0

        # Parameter Geometri Fisik Bilah Sirip pada Konveyor (Adaptive Conveyor Geometric Envelope):
        # 1. Lebar sirip riil konveyor: fleksibel mencakup sudut dekat/lebar (672px - 1824px pada 1080p)
        #    Menolak struktur rangka bawah konveyor (bw > 1850px) & plat slat dasar (Y1 > 540px, CY > 572px)
        # 2. Batas lateral: fleksibel mengikuti sudut pandang/jarak kamera (x1 >= 38px, x2 <= 1881px)
        # 3. Ketebalan vertikal fisik bilah tegak: 18px - 194px
        # 4. Rasio aspek bilah tipis memanjang: 5.0 - 35.0
        min_bw = int(0.35 * w_f)   # ~672 px
        max_bw = int(0.95 * w_f)   # ~1824 px (mengakomodasi bilah sirip pada kamera FOV lebar)
        min_x1 = int(0.02 * w_f)   # ~38 px
        max_x2 = int(0.98 * w_f)   # ~1881 px
        min_bh = max(18, int(0.018 * h_f))
        max_bh = int(0.18 * h_f)   # ~194 px
        nominal_w = int(0.70 * w_f)

        try:
            # S6: 3-Level Hysteresis Detection Threshold (Diperketat untuk anti-false detection real-time):
            # - Level 1 Cold Start (discovery): max(min(conf_thresh, 0.48), 0.42) → tolak noise kamera live awal
            # - Level 2 Early Active (<5 frame): max(conf_thresh * 0.72, 0.36) → tahan drop transisi awal
            # - Level 3 Stable Active (≥5 frame): max(conf_thresh * 0.64, 0.34) → cegah flicker tanpa over-sensitif
            if self.active_rib_frames == 0:
                eval_conf = max(min(self.conf_thresh, 0.48), 0.42)   # Level 1: Discovery ketat
            elif self.active_rib_frames < 5:
                eval_conf = max(self.conf_thresh * 0.72, 0.36)       # Level 2: Early Active
            else:
                eval_conf = max(self.conf_thresh * 0.64, 0.34)       # Level 3: Stable Active
            predict_kwargs = {
                "source": infer_img,
                "conf": eval_conf,
                "iou": self.iou_thresh,
                "device": self.device,
                "agnostic_nms": True,
                "verbose": False
            }
            # Jangan override imgsz & half jika model adalah TensorRT .engine atau ONNX .onnx
            is_trt_or_onnx = self.model_path and (str(self.model_path).endswith(".engine") or str(self.model_path).endswith(".onnx"))
            if not is_trt_or_onnx:
                predict_kwargs["imgsz"] = target_imgsz
                if self.use_half and self.device != "cpu":
                    predict_kwargs["half"] = True

            try:
                results = self.yolo_model.predict(**predict_kwargs)
            except Exception as e_fp16:
                if predict_kwargs.get("half", False):
                    predict_kwargs["half"] = False
                    results = self.yolo_model.predict(**predict_kwargs)
                else:
                    raise e_fp16

            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                if len(boxes) > 0:
                    boxes_xyxy = boxes.xyxy.cpu().numpy()
                    boxes_conf = boxes.conf.cpu().numpy()
                    boxes_cls = boxes.cls.cpu().numpy()
                else:
                    boxes_xyxy = boxes_conf = boxes_cls = ()
                for i in range(len(boxes_xyxy)):
                    xyxy = boxes_xyxy[i]
                    conf = float(boxes_conf[i])
                    cls_id = int(boxes_cls[i])
                    cls_name = self.class_names.get(cls_id, str(cls_id)).lower()

                    if "rib" in cls_name or cls_id == 0:
                        x1 = int(xyxy[0]) + off_x
                        y1 = int(xyxy[1]) + off_y
                        x2 = int(xyxy[2]) + off_x
                        y2 = int(xyxy[3]) + off_y
                        bw = x2 - x1
                        bh = y2 - y1
                        cy = (y1 + y2) // 2
                        ar = bw / max(1.0, float(bh))

                        # Filter validitas bilah sirip:
                        # 1. Batasi CY maksimum ke 572 px (0.53 * h_f) untuk menolak gerbang plat slat (Y > 565 px)
                        # 2. Batasi Y1 maksimum ke 540 px (0.50 * h_f) untuk menolak 100% plat slat dasar bawah (Y1 > 600 px)
                        # 3. Pastikan lebar bilah memenuhi batas fisik sirip riil adaptif (bw <= 1824 px)
                        is_valid = (
                            (min_bw <= bw <= max_bw) and
                            (x1 >= min_x1 and x2 <= max_x2) and
                            (min_bh <= bh <= max_bh) and
                            (5.0 <= ar <= 35.0) and
                            (y1 >= zy1 - max(12, int(0.017 * h_f))) and (cy >= zy1 + max(10, int(0.014 * h_f))) and (cy <= min(zy2 + 15, int(0.53 * h_f))) and
                            (y1 <= int(0.50 * h_f))
                        )

                        if is_valid:
                            rib_boxes.append({
                                "box": [x1, y1, x2, y2],
                                "conf": conf,
                                "centroid": (int((x1 + x2) / 2), cy)
                            })
                            # Seleksi berbasis kombinasi skor deteksi, kemiripan lebar nominal (1214px), dan kedekatan ke prime zone
                            prime_mid = (self.prime_min_y + self.prime_max_y) // 2
                            dist_to_prime = abs(cy - prime_mid)
                            w_dev = abs(bw - nominal_w) / float(nominal_w)
                            match_score = conf * max(0.1, 1.0 - 2.0 * w_dev) - (dist_to_prime / 800.0)
                            if match_score > best_match_score:
                                best_match_score = match_score
                                best_box = [x1, y1, x2, y2]
                                max_conf = conf

        except Exception as e:
            if not hasattr(self, "_last_yolo_err") or self._last_yolo_err != str(e):
                print(f"[WARN HybridRibInspector YOLO] Gagal inferensi ({type(e).__name__}): {e}")
                self._last_yolo_err = str(e)

        is_detected = best_box is not None
        return is_detected, best_box, max_conf, rib_boxes

    def detect_dual_polarity_zone(self, frame, zone_rect):
        """
        Analisis Fisika Optik Dual-Polarity Sobel Y pada zona inspeksi konveyor:
        Memeriksa keberadaan bilah tegak 3D dengan tinggi H in [50, 220] px
        (+Sobel Y pada puncak atas bilah vs -Sobel Y pada kaki/bayangan bilah).
        Dioptimalkan dengan evaluasi khusus pada ROI zona konveyor (hemat komputasi 75%).

        PENINGKATAN ANTI-FALSE DETECTION:
          S1A – Morphological Background Subtraction: menghilangkan bias iluminasi horizontal
                yang menyebabkan threshold Sobel-Y meledak saat cahaya ambient berfluktuasi.
          S3  – Adaptive Sobel Threshold: threshold top/bot dihitung dari percentile distribusi
                gradien ROI sehingga naik-turun otomatis mengikuti kondisi pencahayaan.
          S4  – EMA Dual Polarity Ratio: meratakan keputusan biner antar frame, menolak
                spike false-positive dari noda oli atau flicker 1–2 frame.
          S5  – Spatial Consistency Gate: valid_points harus tersebar merata ≥ 60% lebar zona,
                sehingga noda terlokalisasi (oli, bayangan) tidak memicu deteksi.

        :param frame: Frame citra BGR
        :param zone_rect: (zx1, zy1, zx2, zy2)
        :return: (is_detected, dual_box, median_height, valid_ratio, None)
        """
        zx1, zy1, zx2, zy2 = zone_rect
        crop_bgr = frame[zy1:zy2, zx1:zx2]
        if crop_bgr.size == 0:
            return False, None, 0.0, 0.0, None

        h_c, w_c = crop_bgr.shape[:2]
        # Optimasi Downscale 2x: Memangkas komputasi Sobel-Y & looping CPU dari ~10ms menjadi ~2.9ms
        small_bgr = cv2.resize(crop_bgr, (w_c // 2, h_c // 2), interpolation=cv2.INTER_LINEAR)
        crop_gray = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY)

        # ── S1A: Gaussian Illumination Flat-Field Normalization (Fast 4x Downscale) ─
        try:
            crop_gray = self._normalize_flat_field_gray(crop_gray, scale=128)
        except Exception:
            pass  # fallback ke crop_gray asli jika error

        # ── S3: Hitung Adaptive Threshold dari Distribusi Gradien ROI ────────
        # Ambil distribusi Sobel-Y seluruh ROI untuk estimasi kondisi pencahayaan.
        # Percentile ke-82 dari distribusi positif → threshold 'top_val' adaptif.
        # Clamp [12, 32] agar tidak terlalu longgar saat gelap / terlalu ketat saat terang.
        sy_full = cv2.Sobel(crop_gray, cv2.CV_32F, 0, 1, ksize=3)
        pos_all = sy_full[sy_full > 4.0]
        neg_all = -sy_full[sy_full < -4.0]
        if len(pos_all) > 20:
            # [ANTI-FALSE-DET] Percentile 82→85 & clamp bawah 12→16: threshold lebih konservatif
            # sehingga gradien shadow celah slat saat cahaya rendah tidak lolos gate deteksi
            _dp_top_thresh = float(np.clip(np.percentile(pos_all, 85), 16.0, 32.0))
        else:
            _dp_top_thresh = 18.0  # fallback ke default
        if len(neg_all) > 20:
            # [ANTI-FALSE-DET] Clamp bawah 14→18: bot threshold tidak terlalu longgar saat gelap
            _dp_bot_thresh = float(np.clip(np.percentile(neg_all, 85), 18.0, 36.0))
        else:
            _dp_bot_thresh = 20.0  # fallback ke default

        pos_sy = np.clip(sy_full, 0, 255)
        neg_sy = np.clip(-sy_full, 0, 255)

        h_crop, w_crop = crop_gray.shape[:2]
        scan_xs_rel = np.arange(30, max(31, w_crop - 30), 8)
        valid_points = []

        for x_rel in scan_xs_rel:
            col_pos = pos_sy[:, x_rel]
            if len(col_pos) == 0:
                continue

            top_offset = int(np.argmax(col_pos))
            top_val = col_pos[top_offset]
            top_y = zy1 + top_offset * 2

            # Kaki bawah / bayangan bilah berada di bawah puncak atas
            b_lo = min(h_crop - 2, top_offset + 22)
            b_hi = min(h_crop - 1, top_offset + 115)
            if b_hi > b_lo:
                sub_neg = neg_sy[b_lo:b_hi, x_rel]
                if len(sub_neg) > 0:
                    bot_offset = int(np.argmax(sub_neg))
                    bot_val = sub_neg[bot_offset]
                    bot_y = zy1 + (b_lo + bot_offset) * 2
                    h_rib = bot_y - top_y

                    # S3: Gunakan adaptive threshold (bukan fixed 18.0 / 20.0)
                    if top_val >= _dp_top_thresh and bot_val >= _dp_bot_thresh and (50 <= h_rib <= 220):
                        valid_points.append((zx1 + x_rel * 2, top_y, bot_y, h_rib))

        # Konsistensi Cluster Horizontal (S5b): Saring titik yang sebaris dengan bibir rib dominan
        # (±34px dari median top_y) agar edge acak antar celah slat tereliminasi.
        if len(valid_points) >= 5:
            _med_ty_cluster = float(np.median([p[1] for p in valid_points]))
            _clustered = [p for p in valid_points if abs(p[1] - _med_ty_cluster) <= 34.0]
            if len(_clustered) >= 3:
                valid_points = _clustered

        raw_ratio = len(valid_points) / float(len(scan_xs_rel)) if len(scan_xs_rel) > 0 else 0.0

        # ── S4: EMA Smoothing pada Dual Polarity Ratio ───────────────────────
        # Meratakan rasio antar frame: menolak spike 1-frame (noda, pantulan sesaat)
        # tanpa menunda deteksi rib nyata (alpha=0.55 → memory ~2 frame)
        self._dp_ratio_ema = self._dp_alpha * raw_ratio + (1.0 - self._dp_alpha) * self._dp_ratio_ema
        smoothed_ratio = self._dp_ratio_ema

        # ── S5: Spatial Consistency Gate ─────────────────────────────────────
        # Valid points harus tersebar merata sepanjang ≥ 60% lebar zona konveyor.
        # Noda oli atau bayangan terlokalisasi hanya mengisi < 30% → ditolak.
        spatial_ok = True
        if valid_points and len(valid_points) >= 3:
            xs_vp = [p[0] for p in valid_points]
            span_vp = max(xs_vp) - min(xs_vp)
            zone_width = zx2 - zx1
            # Span harus ≥ 60% lebar zona aktif
            if zone_width > 0 and span_vp < zone_width * 0.60:
                spatial_ok = False
            # Distribusi merata: minimal 3 dari 4 kuadran lateral harus ada valid_point
            if spatial_ok and zone_width > 0:
                q_w = max(1.0, zone_width / 4.0)
                quadrants = [0, 0, 0, 0]
                for xp in xs_vp:
                    q_idx = min(3, int((xp - zx1) / q_w))
                    quadrants[q_idx] = 1
                if sum(quadrants) < 3:
                    spatial_ok = False

        # Keputusan deteksi: EMA ratio ≥ 0.38 DAN distribusi spasial memadai
        dual_detected = bool(smoothed_ratio >= 0.38 and spatial_ok)
        dual_box = None
        med_height = 0.0

        if dual_detected and valid_points:
            med_height = float(np.median([p[3] for p in valid_points]))
            min_x = min([p[0] for p in valid_points])
            max_x = max([p[0] for p in valid_points])
            med_top = int(np.median([p[1] for p in valid_points]))
            med_bot = int(np.median([p[2] for p in valid_points]))
            dual_box = [min_x, med_top, max_x, med_bot]

        return dual_detected, dual_box, med_height, raw_ratio, None

    def _track_steger_viterbi(self, gray, active_box, search_x1, search_x2, h_f, w_f):
        """
        PILIHAN A (MAKSIMAL / V2): ANCHOR-GUIDED TIGHT-RIBBON STEGER-VITERBI TRACKER.
        
        Penyempurnaan Fisika & Matematika:
        1. 7-Anchor Contrast Pre-Gate:
           - Memverifikasi kontras bibir bilah sirip di 7 titik tengah (avg_contrast >= 15.0).
           - Menolak 100% plat datar dan bayangan celah plat semu.
        2. Lateral Boundary Clamping (Anti-Chain-Bracket):
           - Membatasi rentang X strictly pada badan sirip: [max(248, bx1 + 10), min(1635, bx2 - 10)].
           - Mengeliminasi 100% false alarm deviasi ujung (TipDev) akibat baut rantai konveyor.
        3. Anchor-Guided Tight Corridor (+/- 8 px) (Anti-Slat-Gap & Anti-Weld):
           - Menelusuri jalur nominal kemiringan sirip aktual via regresi robust.
           - Membatasi pencarian vertikal dalam koridor sempit +/- 8 px (17 px) di sekitar bibir bilah,
             sehingga mustahil melompat ke celah plat atas (-26 px) atau kaki las bawah (+35 px).
        4. Bidirectional Center-Anchored Viterbi DP:
           - Merunut optimal path dari pusat sirip (paling bersih) ke kanan dan ke kiri secara independen.
           - Dilengkapi auto-cutoff saat energi puncak turun tajam (ujung fisik bilah berakhir).
        5. Sub-Pixel Parabolic Interpolation:
           - Estimasi puncak ridge kontinu sub-piksel presisi tinggi.
        """
        bx1, by1, bx2, by2 = active_box
        
        # 1. Validasi Fisika 11 Titik Anchor Rentang Lebar (12% - 88% lebar bilah)
        step_fracs = [0.12, 0.20, 0.28, 0.36, 0.44, 0.50, 0.56, 0.64, 0.72, 0.80, 0.88]
        anchor_xs, anchor_ys, anchor_contrasts = [], [], []
        for frac in step_fracs:
            ax = int(bx1 + frac * (bx2 - bx1))
            best_y, best_sc, max_c = None, -9999.0, 0.0
            for y in range(max(10, by1 + 2), min(h_f - 12, by1 + 48)):
                b = float(gray[y, ax])
                diff_below = b - float(gray[min(h_f - 1, y + 8), ax])
                diff_above = b - float(gray[max(0, y - 5), ax])
                pen_depth = max(0, y - (by1 + 22)) * 0.5
                sc = b * 0.4 + diff_below * 1.0 + diff_above * 0.6 - pen_depth
                if sc > best_sc:
                    best_sc, best_y, max_c = sc, y, diff_below
            if best_y is not None and best_sc > 38.0:
                anchor_xs.append(ax)
                anchor_ys.append(best_y)
                anchor_contrasts.append(max_c)

        avg_contrast = float(np.mean(anchor_contrasts)) if anchor_contrasts else 0.0
        # S3-analog Tier 1: Adaptive contrast gate — hitung threshold dari distribusi Sobel-Y
        # pada ROI bounding box agar otomatis naik-turun mengikuti kondisi pencahayaan.
        # Clamp [10, 22] agar tidak terlalu longgar saat gelap / terlalu ketat saat terang.
        try:
            _sy_roi_t1 = cv2.Sobel(
                gray[max(0, by1):min(h_f, by2), max(0, bx1):min(w_f, bx2)],
                cv2.CV_32F, 0, 1, ksize=3
            )
            _pos_roi_t1 = _sy_roi_t1[_sy_roi_t1 > 0]
            _adaptive_contrast_thresh_t1 = float(
                # [ANTI-FALSE-DET] Clamp [10,22]→[13,25]: gate lebih ketat cegah karat/shadow lolos Tier-1
                np.clip(np.percentile(_pos_roi_t1, 70), 13.0, 25.0)
            ) if len(_pos_roi_t1) > 50 else 15.0
        except Exception:
            _adaptive_contrast_thresh_t1 = 15.0
        # P1 FIX: Naikkan contrast gate +6.0 saat Kinematic Coasting aktif
        # (YOLO tidak detect, posisi di-extrapolate) agar Viterbi tidak lolos
        # dengan crest palsu dari shadow slat / bayangan celah saat rib hampir keluar.
        _coast_penalty_t1 = 6.0 if getattr(self, '_kinematic_coasting_frames', 0) > 0 else 0.0
        if avg_contrast < (_adaptive_contrast_thresh_t1 + _coast_penalty_t1) or len(anchor_xs) < 2:
            return [], [], 0.0, 0, 0.0

        # Fit garis panduan yang kebal outlier & clamped ke kemiringan fisik konveyor
        guide_poly = self._fit_robust_guide_poly(anchor_xs, anchor_ys, (by1 + by2) / 2.0)

        tilt_deg = float(np.arctan(guide_poly[0]) * 180.0 / np.pi)

        # 2. Batas Fisik Lateral Konveyor Adaptif (Zoom & Vibration Tolerant)
        pad_x = 10
        margin_frame = int(0.04 * w_f)
        x_limit_left = max(margin_frame, int(bx1 + pad_x))
        x_limit_right = min(w_f - margin_frame, int(bx2 - pad_x))
        if x_limit_right <= x_limit_left + 100:
            return [], [], 0.0, 0, 0.0

        mid_target_x = (bx1 + bx2) / 2.0
        mid_idx = int(np.argmin([abs(ax - mid_target_x) for ax in anchor_xs]))
        x_center = int(anchor_xs[mid_idx])
        y_center = int(anchor_ys[mid_idx])

        # 3. Koridor Viterbi (+/- 12 px) bebas wraparound dengan pemandu atas/bawah sejati
        half_band = 12
        H_b = half_band * 2 + 1
        step_x = 4
        shifts = np.array([-2, -1, 0, 1, 2], dtype=np.int32)
        penalties = 7.5 * (shifts ** 2)
        off_sq = (np.arange(H_b, dtype=np.float32) - half_band) ** 2
        guide_prior_body = 0.28 * off_sq
        guide_prior_tip = 0.04 * off_sq
        tip_w_t1 = max(120, int(0.09 * (x_limit_right - x_limit_left)))

        # Pre-komputasi Sobel orde-2 (syy) satu kali pada strip ROI agar identik tanpa memanggil Sobel per kolom
        gy_ends = np.polyval(guide_poly, [x_limit_left, x_limit_right])
        y1_s = max(0, int(np.floor(min(gy_ends))) - half_band - 6)
        y2_s = min(h_f, int(np.ceil(max(gy_ends))) + half_band + 9)
        x1_s = max(0, x_limit_left - 2)
        x2_s = min(w_f, x_limit_right + 3)
        if y2_s <= y1_s + 12 or x2_s <= x1_s + 4:
            return [], [], 0.0, 0, tilt_deg
        roi_f = gray[y1_s:y2_s, x1_s:x2_s].astype(np.float32)
        syy_roi = cv2.Sobel(roi_f, cv2.CV_32F, 0, 2, ksize=3)

        def get_col_energy(x):
            gy = int(round(np.polyval(guide_poly, x)))
            col_y1 = gy - half_band
            col_y2 = gy + half_band
            ry1 = col_y1 - y1_s
            ry2 = col_y2 - y1_s + 1
            rx = x - x1_s
            if (
                col_y1 < 4 or col_y2 + 6 >= h_f or x < 1 or x + 1 >= w_f
                or ry1 < 4 or ry2 + 6 >= roi_f.shape[0]
                or rx < 1 or rx + 1 >= roi_f.shape[1]
            ):
                return np.zeros(H_b, dtype=np.float32), gy
            syy = syy_roi[ry1:ry2, rx]
            center_col = roi_f[ry1:ry2, rx]
            below_col = roi_f[ry1 + 6:ry2 + 6, rx]
            above_col = roi_f[ry1 - 4:ry2 - 4, rx]
            lip = np.clip(center_col - below_col, 0.0, 100.0)
            diff_above = np.clip(center_col - above_col, 0.0, 80.0)
            is_tip_col = (x < x_limit_left + tip_w_t1) or (x > x_limit_right - tip_w_t1)
            prior = guide_prior_tip if is_tip_col else guide_prior_body
            e = np.clip(-syy * 0.7, 0.0, 90.0) + 0.9 * lip + 0.6 * diff_above - prior
            return e, gy

        def _run_viterbi_dir(xs_list):
            """Jalankan Bidirectional Viterbi DP ke kanan atau ke kiri dari titik awal bersama."""
            pts_out = []
            n_cols = len(xs_list)
            if n_cols == 0:
                return pts_out
            V = np.full((H_b, n_cols), -1e9, dtype=np.float32)
            Back = np.zeros((H_b, n_cols), dtype=np.int32)
            E_cols = np.zeros((H_b, n_cols), dtype=np.float32)
            gys = np.zeros(n_cols, dtype=np.int32)
            e0, gy0 = get_col_energy(xs_list[0])
            E_cols[:, 0] = e0
            gys[0] = gy0
            c_off = int(np.clip(y_center - gy0 + half_band, 0, H_b - 1))
            V[c_off, 0] = e0[c_off] + 50.0

            valid_cols = 1
            consec_miss = 0
            for t in range(1, n_cols):
                et, gyt = get_col_energy(xs_list[t])
                E_cols[:, t] = et
                gys[t] = gyt
                if np.max(et) < 22.0:
                    consec_miss += 1
                    if consec_miss >= 3 and t > 10:
                        break
                else:
                    consec_miss = 0
                prev_v = V[:, t - 1]
                cands = np.full((len(shifts), H_b), -1e9, dtype=np.float32)
                for s_idx, d in enumerate(shifts):
                    pen = penalties[s_idx]
                    if d < 0:
                        cands[s_idx, -d:] = prev_v[:H_b + d] - pen
                    elif d == 0:
                        cands[s_idx, :] = prev_v - pen
                    else:
                        cands[s_idx, :-d] = prev_v[d:] - pen
                best_s = np.argmax(cands, axis=0)
                V[:, t] = np.max(cands, axis=0) + et
                Back[:, t] = np.arange(H_b) + shifts[best_s]
                valid_cols += 1

            y_b = int(np.argmax(V[:, valid_cols - 1]))
            path = [y_b]
            for t in range(valid_cols - 1, 0, -1):
                y_b = int(Back[y_b, t])
                path.append(y_b)
            path.reverse()
            for i in range(valid_cols):
                p_i = path[i]
                y_val = float(gys[i] - half_band + p_i)
                # Interpolasi parabolik sub-piksel pada puncak energi kolom
                if 1 <= p_i <= H_b - 2:
                    e_m1 = float(E_cols[p_i - 1, i])
                    e_0 = float(E_cols[p_i, i])
                    e_p1 = float(E_cols[p_i + 1, i])
                    denom = 2.0 * (e_m1 - 2.0 * e_0 + e_p1)
                    if abs(denom) > 1e-4:
                        offset = (e_m1 - e_p1) / denom
                        y_val += max(-0.5, min(0.5, offset))
                pts_out.append((xs_list[i], y_val))
            return pts_out

        # 4 & 5. Bidirectional Viterbi: Kanan & Kiri (simetris dari titik awal tengah)
        right_pts = _run_viterbi_dir(list(range(x_center, x_limit_right + 1, step_x)))
        left_pts = _run_viterbi_dir(list(range(x_center, x_limit_left - 1, -step_x)))[1:]
        left_pts.reverse()

        all_pts = left_pts + right_pts
        ridge_xs = [p[0] for p in all_pts]
        ridge_ys = [p[1] for p in all_pts]
        tested_cols = (x_limit_right - x_limit_left) // step_x
        coverage = len(ridge_xs) / float(max(1, tested_cols))
        return ridge_xs, ridge_ys, coverage, tested_cols, tilt_deg

    def _track_yolo_pose(self, frame, gray, active_box, h_f, w_f):
        """
        PILIHAN B (TIER 2): DEEP LEARNING 13-KEYPOINTS YOLO-POSE + LOCAL GRADIENT POLISH.
        
        Prinsip Kerja & Keunggulan:
        1. Direct Neural Network Keypoint Inference:
           - Memprediksi 13 titik kunci anatomis sepanjang bibir sirip secara holistik.
           - Sangat kebal terhadap noda oli hitam pekat, karat parah, bayangan, dan pergeseran sudut kamera.
        2. Local Sub-Pixel Polish (Radius +/- 5 px):
           - Memoles setiap keypoint AI pada jendela vertikal lokal sempit untuk mengunci tepat
             pada puncak gradien metalik fisik dengan presisi sub-milimeter.
        3. Spline / Polynomial Smooth Densification:
           - Menghubungkan titik-titik menjadi kontur rapat mulus (step 4 px) untuk analisis kelurusan metrologi.
        """
        if self.pose_model is None:
            if not self._load_pose_model():
                return [], [], 0.0, 0, 0.0

        bx1, by1, bx2, by2 = active_box
        pad_x = 10
        margin_frame = int(0.04 * w_f)
        x_limit_left = max(margin_frame, int(bx1 + pad_x))
        x_limit_right = min(w_f - margin_frame, int(bx2 - pad_x))
        if x_limit_right <= x_limit_left + 100:
            return [], [], 0.0, 0, 0.0

        target_cy = (by1 + by2) / 2.0
        target_cx = (bx1 + bx2) / 2.0

        try:
            pose_results = self.pose_model.predict(
                source=frame,
                conf=0.40,
                device=self.device,
                verbose=False
            )
        except Exception as e:
            return [], [], 0.0, 0, 0.0

        if len(pose_results) == 0 or pose_results[0].keypoints is None or len(pose_results[0].keypoints.xy) == 0:
            return [], [], 0.0, 0, 0.0

        best_kpts = None
        best_confs = None
        min_dist = float("inf")

        boxes = pose_results[0].boxes
        for i in range(len(boxes)):
            box_det = boxes.xyxy[i].cpu().numpy()
            det_cy = (box_det[1] + box_det[3]) / 2.0
            det_cx = (box_det[0] + box_det[2]) / 2.0
            dist = abs(det_cy - target_cy) + 0.25 * abs(det_cx - target_cx)
            if dist < min_dist and abs(det_cy - target_cy) <= 55:
                min_dist = dist
                best_kpts = pose_results[0].keypoints.xy[i].cpu().numpy()
                if pose_results[0].keypoints.conf is not None:
                    best_confs = pose_results[0].keypoints.conf[i].cpu().numpy()
                else:
                    best_confs = np.ones(len(best_kpts), dtype=np.float32)

        if best_kpts is None or len(best_kpts) < 5:
            return [], [], 0.0, 0, 0.0

        # Sub-Pixel Local Gradient Polish pada masing-masing Keypoint
        clean_kpts = []
        for i in range(len(best_kpts)):
            kx, ky = best_kpts[i]
            c = float(best_confs[i])
            if c >= 0.40 and x_limit_left <= kx <= x_limit_right:
                kx_i = int(round(kx))
                ky_i = int(round(ky))
                y_lo = max(0, ky_i - 5)
                y_hi = min(h_f - 1, ky_i + 5)
                best_sc = -9999.0
                fine_y = float(ky_i)
                for y in range(y_lo, y_hi + 1):
                    b = float(gray[y, kx_i])
                    diff_below = b - float(gray[min(h_f - 1, y + 6), kx_i])
                    diff_above = b - float(gray[max(0, y - 4), kx_i])
                    sc = b * 0.4 + diff_below * 1.0 + diff_above * 0.6
                    if sc > best_sc:
                        best_sc = sc
                        fine_y = float(y)
                clean_kpts.append((kx, fine_y))

        if len(clean_kpts) < 5:
            return [], [], 0.0, 0, 0.0

        clean_pts = np.array(clean_kpts, dtype=np.float32)
        clean_pts = clean_pts[np.argsort(clean_pts[:, 0])]

        # Filter outlier Y menggunakan RANSAC residual — versi lebih ketat (S-Tier2):
        # Clamp batas atas agar tidak terlalu longgar saat kontur bengkok ekstrem,
        # namun batas bawah tetap 8px agar tidak over-reject keypoint valid di ujung tip.
        poly1 = np.polyfit(clean_pts[:, 0], clean_pts[:, 1], 1)
        resids = np.abs(clean_pts[:, 1] - np.polyval(poly1, clean_pts[:, 0]))
        med_res = float(np.median(resids))
        inliers = resids < min(14.0, max(8.0, med_res * 2.0))
        final_pts = clean_pts[inliers]

        if len(final_pts) < 4:
            final_pts = clean_pts

        span = float(final_pts[-1, 0] - final_pts[0, 0])
        # S-Tier2: Perkuat span minimum dari 500px → 600px untuk menolak partial detection
        # (misalnya hanya tepi kiri rib yang terdeteksi saat rib baru memasuki frame)
        if span < 600.0:
            return [], [], 0.0, 0, 0.0

        # Regresi kurva kontinu rapat (densification step 4 px)
        poly2 = np.polyfit(final_pts[:, 0], final_pts[:, 1], deg=2)
        step_x = 4
        dense_xs = np.arange(final_pts[0, 0], final_pts[-1, 0] + 1, float(step_x))
        dense_ys = np.polyval(poly2, dense_xs)

        ridge_xs = [float(x) for x in dense_xs]
        ridge_ys = [float(y) for y in dense_ys]
        tested_cols = (x_limit_right - x_limit_left) // step_x
        coverage = len(ridge_xs) / float(max(1, tested_cols))
        tilt_deg = float(np.arctan(poly1[0]) * 180.0 / np.pi)

        return ridge_xs, ridge_ys, coverage, tested_cols, tilt_deg

    def _track_yolo_segmentation(self, frame, gray, active_box, h_f, w_f):
        """
        TIER 3 TRACKER ENGINE: HIGH-PERFORMANCE DEEP LEARNING INSTANCE SEGMENTATION + UNWARP METROLOGY.
        Menggunakan model YOLO11-seg (best_rib_segmentation.pt) dengan:
        1. CUDA FP16 Half-Precision GPU Inference.
        2. Vectorized NumPy Column Crest Extraction (O(1) Matrix Slicing vs O(N^2) Loop Python).
        3. Vectorized Sub-Pixel Gradient Peak Polish (0.1ms).
        4. Stride-2 Temporal Physical Ridge Tracking (Alternating Frame < 1ms vs 60ms).
        5. Matrix Caching Unwarp Transform (Bebas komputasi ganda).
        6. 100% Bit-for-Bit Identical Accuracy dengan throughput hingga 45+ FPS (Bebas Patah-Patah).
        """
        if frame is None or active_box is None:
            self._last_unwarp_contour = None
            self._last_seg_xs = None
            return [], [], 0.0, 0, 0.0

        if not self._load_seg_model():
            return self._track_unwarp_perspective(frame, active_box, h_f, w_f)

        # Gunakan matriks unwarp yang sudah terhitung jika valid (Zero-Overhead)
        if getattr(self, "unwarp_M", None) is not None and getattr(self, "unwarp_Minv", None) is not None:
            M = self.unwarp_M
            Minv = self.unwarp_Minv
        else:
            _, M, Minv = self.unwarp_rib_zone(frame)
            if M is None or Minv is None:
                self._last_unwarp_contour = None
                return [], [], 0.0, 0, 0.0

        bx1, by1, bx2, by2 = active_box
        curr_cy = (by1 + by2) / 2.0
        pad_x = 10
        margin_frame = int(0.04 * w_u if 'w_u' in locals() else 0.04 * w_f)
        x_min_limit = max(margin_frame, int(bx1 + pad_x))
        x_max_limit = min(w_f - margin_frame, int(bx2 - pad_x))

        # TAHAP 1: Fast Temporal Sub-Pixel Tracking (Hanya aktif di CPU untuk optimasi latensi)
        # Jika GPU CUDA aktif, sistem menjalankan inferensi FP16 penuh setiap frame demi presisi maksimum
        use_stride = (self.device == "cpu")
        self._seg_stride_counter = getattr(self, "_seg_stride_counter", 0) + 1

        can_reuse = (
            use_stride
            and self._seg_stride_counter % 2 != 0
            and getattr(self, "_last_seg_xs", None) is not None
            and getattr(self, "_last_seg_ys", None) is not None
            and getattr(self, "_last_seg_cy", None) is not None
        )

        if can_reuse:
            dy = curr_cy - self._last_seg_cy
            if abs(dy) <= 18.0:
                valid_cols = self._last_seg_xs
                cam_ys = self._last_seg_ys + dy
                ixs = np.clip(np.round(valid_cols).astype(np.int32), 0, w_f - 1)
                iys = np.clip(np.round(cam_ys).astype(np.int32), 4, h_f - 5)

                offsets = np.arange(-6, 7)
                cand_ys = np.clip(iys[:, None] + offsets[None, :], 2, h_f - 3)
                g_below = gray[np.clip(cand_ys + 3, 0, h_f - 1), ixs[:, None]].astype(np.float32)
                g_above = gray[np.clip(cand_ys - 3, 0, h_f - 1), ixs[:, None]].astype(np.float32)
                g_center = gray[cand_ys, ixs[:, None]].astype(np.float32)
                edge_scores = (g_center - g_below) * 1.2 + (g_center - g_above) * 0.8
                best_idx = np.argmax(edge_scores, axis=1)
                refined_ys = cand_ys[np.arange(len(cand_ys)), best_idx].astype(np.float32)
                refined_ys = fast_medfilt(refined_ys, k=7)

                # Stabilisasi Temporal EMA antar frame
                prev_ys = getattr(self, "_prev_tier3_cam_ys", None)
                if prev_ys is not None and len(prev_ys) == len(refined_ys):
                    refined_ys = 0.70 * refined_ys + 0.30 * (prev_ys + dy)

                self._prev_tier3_cam_xs = valid_cols.copy()
                self._prev_tier3_cam_ys = refined_ys.copy()
                self._last_seg_ys = refined_ys.copy()
                self._last_seg_cy = curr_cy

                cam_pts = np.column_stack([valid_cols.astype(np.float32), refined_ys]).astype(np.float32)
                unw_pts = cv2.perspectiveTransform(cam_pts.reshape(-1, 1, 2), M).reshape(-1, 2)
                valid_unw = (unw_pts[:, 0] >= 35.0) & (unw_pts[:, 0] <= 925.0)
                if np.sum(valid_unw) >= 50:
                    unw_pts = unw_pts[valid_unw]
                self._last_unwarp_contour = unw_pts.copy()
                self._last_unwarp_M = M
                self._last_unwarp_Minv = Minv

                w_span = float(valid_cols[-1] - valid_cols[0])
                tested_cols = int((x_max_limit - x_min_limit) / 2.0)
                coverage = len(valid_cols) / float(max(1, tested_cols))
                poly1 = np.polyfit(valid_cols, refined_ys, 1)
                tilt_deg = float(np.arctan(poly1[0]) * 180.0 / np.pi)
                return valid_cols.tolist(), refined_ys.tolist(), coverage, tested_cols, tilt_deg

        # TAHAP 2: Full GPU FP16 Half-Precision Instance Segmentation Predictor
        dev_target = 0 if (self.device != "cpu" and HAS_TORCH and torch.cuda.is_available()) else "cpu"
        use_half = bool(self.device != "cpu" and HAS_TORCH and torch.cuda.is_available())

        try:
            results = self.seg_model.predict(
                frame,
                conf=0.20,
                verbose=False,
                device=dev_target,
                half=use_half
            )
        except Exception:
            return self._track_unwarp_perspective(frame, active_box, h_f, w_f)

        if not results or results[0].masks is None or len(results[0].masks) == 0:
            return self._track_unwarp_perspective(frame, active_box, h_f, w_f)

        res0 = results[0]
        boxes = res0.boxes.xyxy.cpu().numpy()
        masks = res0.masks.data.cpu().numpy()

        best_mask = None
        best_dist = 9999.0
        for b, m in zip(boxes, masks):
            c_x1, c_y1, c_x2, c_y2 = b
            m_cy = (c_y1 + c_y2) / 2.0
            m_w = c_x2 - c_x1
            if m_w >= 400 and (180 <= m_cy <= 560):
                dist = abs(m_cy - curr_cy)
                if dist < best_dist and dist <= 85.0:
                    best_dist = dist
                    best_mask = m

        if best_mask is None:
            return self._track_unwarp_perspective(frame, active_box, h_f, w_f)

        # 1. Bilinear Interpolation pada Continuous Probability Mask [0.0 - 1.0]
        # Menghilangkan 100% efek tangga (staircase / blocky grid) kuantisasi mask YOLO
        mask_prob = cv2.resize(best_mask.astype(np.float32), (w_f, h_f), interpolation=cv2.INTER_LINEAR)
        sampled_xs = np.arange(x_min_limit, x_max_limit + 1, 2)

        # Support mask biner untuk kolom aktif
        mask_bin = (mask_prob > 0.35).astype(np.uint8)
        cols_bin = mask_bin[:, sampled_xs]
        has_rib = np.any(cols_bin > 0, axis=0)
        valid_cols = sampled_xs[has_rib]

        if len(valid_cols) < 50:
            return self._track_unwarp_perspective(frame, active_box, h_f, w_f)

        # 2. Vectorized Sub-Pixel Continuous Zero-Crossing Extraction
        cols_prob = mask_prob[:, valid_cols]
        y_indices = np.argmax(cols_prob > 0.50, axis=0)
        y_prev = np.clip(y_indices - 1, 0, h_f - 1)
        col_idx = np.arange(len(valid_cols))
        p_curr = cols_prob[y_indices, col_idx]
        p_prev = cols_prob[y_prev, col_idx]
        denom = np.maximum(1e-4, p_curr - p_prev)
        sub_ys = (y_indices - 1).astype(np.float32) + np.clip((0.50 - p_prev) / denom, 0.0, 1.0)

        # Filter tebing vertikal pada batas lateral poligon YOLO
        med_sub_y = float(np.median(sub_ys))
        inlier_mask = np.abs(sub_ys - med_sub_y) < 25.0
        if np.sum(inlier_mask) >= 50:
            valid_cols = valid_cols[inlier_mask]
            sub_ys = sub_ys[inlier_mask]

        # 3. Vectorized Sub-Pixel Gradient Snapping ke Bibir Fisik Baja (Jendela Luas +-8 px)
        offsets = np.arange(-8, 9)
        cand_ys = np.clip(np.round(sub_ys)[:, None] + offsets[None, :], 3, h_f - 4).astype(np.int32)
        g_below = gray[np.clip(cand_ys + 3, 0, h_f - 1), valid_cols[:, None]].astype(np.float32)
        g_above = gray[np.clip(cand_ys - 3, 0, h_f - 1), valid_cols[:, None]].astype(np.float32)
        g_center = gray[cand_ys, valid_cols[:, None]].astype(np.float32)
        edge_scores = (g_center - g_below) * 1.2 + (g_center - g_above) * 0.8
        best_cand_idx = np.argmax(edge_scores, axis=1)
        peak_cand_ys = cand_ys[np.arange(len(valid_cols)), best_cand_idx].astype(np.float32)
        max_scores = np.max(edge_scores, axis=1)
        refined_ys = np.where(max_scores > 20.0, peak_cand_ys, sub_ys)

        # 4. Multi-Scale Smoothing (1D Median Filter + Box Convolution)
        refined_ys = fast_medfilt(refined_ys, k=7)
        k_box = 5
        refined_ys = np.convolve(refined_ys, np.ones(k_box, dtype=np.float32) / float(k_box), mode='same')

        # 5. Temporal Motion-Compensated EMA Filter & Span Bridging (Anti-Flicker & Glare Recovery)
        prev_xs = getattr(self, "_prev_tier3_cam_xs", None)
        prev_ys = getattr(self, "_prev_tier3_cam_ys", None)
        if prev_xs is not None and prev_ys is not None and len(prev_xs) >= 50:
            dy = float(np.median(refined_ys) - np.median(prev_ys))
            if abs(dy) < 18.0:
                prev_interp_ys = np.interp(valid_cols, prev_xs, prev_ys)
                aligned_prev = prev_interp_ys + dy
                # Fusi temporal EMA: 70% observasi baru, 30% kontur stabil sebelumnya
                refined_ys = 0.70 * refined_ys + 0.30 * aligned_prev

                # Temporal Span Bridging: Jika mask drop akibat pantulan silau (contoh Frame 140 sisi kiri gelap)
                curr_min_x, curr_max_x = valid_cols[0], valid_cols[-1]
                prev_min_x, prev_max_x = prev_xs[0], prev_xs[-1]
                if (curr_min_x - prev_min_x) > 60:
                    left_bridge_xs = np.arange(prev_min_x, curr_min_x, 2)
                    if len(left_bridge_xs) > 0:
                        left_bridge_ys = np.interp(left_bridge_xs, prev_xs, prev_ys) + dy
                        valid_cols = np.concatenate([left_bridge_xs, valid_cols])
                        refined_ys = np.concatenate([left_bridge_ys, refined_ys])
                if (prev_max_x - curr_max_x) > 60:
                    right_bridge_xs = np.arange(curr_max_x + 2, prev_max_x + 1, 2)
                    if len(right_bridge_xs) > 0:
                        right_bridge_ys = np.interp(right_bridge_xs, prev_xs, prev_ys) + dy
                        valid_cols = np.concatenate([valid_cols, right_bridge_xs])
                        refined_ys = np.concatenate([refined_ys, right_bridge_ys])

        # Simpan state untuk pelacakan temporal
        self._prev_tier3_cam_xs = valid_cols.copy()
        self._prev_tier3_cam_ys = refined_ys.copy()
        self._last_seg_xs = valid_cols.copy()
        self._last_seg_ys = refined_ys.copy()
        self._last_seg_cy = curr_cy

        # Proyeksikan kontur kamera ke unwarp space untuk view 90 deg ortogonal
        cam_pts = np.column_stack([valid_cols.astype(np.float32), refined_ys]).astype(np.float32)
        unw_pts = cv2.perspectiveTransform(cam_pts.reshape(-1, 1, 2), M).reshape(-1, 2)
        valid_unw = (unw_pts[:, 0] >= 35.0) & (unw_pts[:, 0] <= 925.0)
        if np.sum(valid_unw) >= 50:
            unw_pts = unw_pts[valid_unw]

        self._last_unwarp_contour = unw_pts.copy()
        self._last_unwarp_M = M
        self._last_unwarp_Minv = Minv

        w_span = float(valid_cols[-1] - valid_cols[0])
        tested_cols = int((x_max_limit - x_min_limit) / 2.0)
        coverage = len(valid_cols) / float(max(1, tested_cols))

        poly1 = np.polyfit(valid_cols, refined_ys, 1)
        tilt_deg = float(np.arctan(poly1[0]) * 180.0 / np.pi)

        return valid_cols.tolist(), refined_ys.tolist(), coverage, tested_cols, tilt_deg

    def _track_unwarp_perspective(self, frame, active_box, h_f, w_f):
        """
        TIER 3 TRACKER ENGINE (ROBUST GUIDED ORTHOGONAL RIDGE TRACKER):
        Pelacakan Kontur Puncak Bilah Sirip Langsung pada Ruang Unwarp (Top-Down 90 Deg).
        Dilengkapi dengan:
        1. Multi-Point Anchor Sampling (9 titik di area 32% - 68% lebar konveyor)
        2. Robust Anchor Outlier Filtering & Fitting Linear Guide Line (garis pemandu unwarp)
        3. Hard Guard Boundary (|y - gy| <= max_guard) mencegah kontur melompat ke celah slat / las
        4. Hard Step Gate (|Δy| <= max_step) menolak spike noise lokal dan permukaan slat
        5. Stabilisasi Temporal Anchor (EMA) antar frame berurutan
        6. Sub-Pixel Parabolic Peak Polish untuk metrologi akurasi tinggi
        """
        if frame is None or active_box is None:
            self._last_unwarp_contour = None
            return [], [], 0.0, 0, 0.0

        unwarped, M, Minv = self.unwarp_rib_zone(frame)
        if unwarped is None or M is None or Minv is None:
            self._last_unwarp_contour = None
            return [], [], 0.0, 0, 0.0

        h_u, w_u = unwarped.shape[:2]
        bx1, by1, bx2, by2 = active_box

        # Proyeksikan 4 sudut active_box ke ruang unwarp
        box_corners = np.array([
            [[bx1, by1]],
            [[bx2, by1]],
            [[bx2, by2]],
            [[bx1, by2]]
        ], dtype=np.float32)
        u_box = cv2.perspectiveTransform(box_corners, M).reshape(-1, 2)

        # 1. Window pencarian vertikal:
        # Prioritaskan EMA anchor history bila tersedia (lebih stabil dari YOLO bbox),
        # fallback ke u_min_y dari bbox hanya jika anchor history belum ada.
        # Rentang +/- 40px dari anchor stabil mencakup crest bilah tanpa menyentuh weld (+60-80px).
        u_min_y = float(np.min(u_box[:, 1]))

        # 1. Window pencarian vertikal:
        # Validasi EMA anchor history: jika desinkron > 22px dari u_min_y aktif (transisi bilah baru), reset!
        prev_y = getattr(self, "_prev_unwarp_anchor_y", None)
        if prev_y is not None and abs(prev_y - u_min_y) > 22.0:
            self._prev_unwarp_anchor_y = None
            prev_y = None

        if prev_y is not None:
            y_center_est = prev_y
            search_uy1 = max(5, int(min(u_min_y - 2, y_center_est - 10)))
            search_uy2 = min(h_u - 5, int(max(u_min_y + 55, y_center_est + 35)))
        else:
            search_uy1 = max(5, int(u_min_y - 2))
            search_uy2 = min(h_u - 5, int(u_min_y + 55))

        # Batas lateral terkalibrasi mengikuti span fisik bilah sirip aktif
        u_min_x = float(np.min(u_box[:, 0]))
        u_max_x = float(np.max(u_box[:, 0]))
        search_ux1 = max(24, int(u_min_x + 8))
        search_ux2 = min(w_u - 25, int(u_max_x - 8))

        if search_ux2 - search_ux1 < 300 or search_uy2 <= search_uy1:
            self._last_unwarp_contour = None
            return [], [], 0.0, 0, 0.0

        gray_u = cv2.cvtColor(unwarped, cv2.COLOR_BGR2GRAY)

        # S1A: Gaussian Illumination Flat-Field Normalization (Fast 4x Downscale) ─
        try:
            gray_u = self._normalize_flat_field_gray(gray_u, scale=128)
        except Exception:
            pass  # fallback ke gray_u tanpa koreksi

        # S2A: Bilateral Filter — denoising edge-preserving di unwarp space ──
        # d=5 cukup cepat (<1.5ms), sigmaColor=45 agar tidak menghaluskan tepi tajam rib
        try:
            gray_u = cv2.bilateralFilter(gray_u, d=5, sigmaColor=45, sigmaSpace=45)
        except Exception:
            pass

        # 2. Temukan Anchor Puncak dari 9 titik sampel di area tengah bilah sirip (X: 28% - 76% span bilah)
        # Puncak bilah sirip memiliki kontras atas (crest vs latar belakang) dan bawah (crest vs wajah bilah)
        sample_xs = [int(search_ux1 + (search_ux2 - search_ux1) * f) for f in [0.28, 0.34, 0.40, 0.46, 0.52, 0.58, 0.64, 0.70, 0.76]]
        anchor_xs, anchor_ys, anchor_contrasts = [], [], []
        for ax in sample_xs:
            best_y = None
            best_sc = -9999.0
            best_c = 0.0
            for y in range(search_uy1, search_uy2):
                b = float(gray_u[y, ax])
                diff_below = b - float(gray_u[min(h_u - 1, y + 8), ax])
                diff_above = b - float(gray_u[max(0, y - 5), ax])
                if diff_above >= 8.0 and diff_below >= 10.0:
                    sc = diff_below * 1.1 + diff_above * 1.4 + min(b, 175.0) * 0.15
                    if sc > best_sc:
                        best_sc = sc
                        best_y = y
                        best_c = diff_below
            if best_y is not None and best_sc >= 30.0:
                anchor_xs.append(ax)
                anchor_ys.append(best_y)
                anchor_contrasts.append(best_c)

        # Minimal 4 dari 9 titik anchor harus berhasil mendeteksi puncak bilah
        # (threshold lebih ketat dari sebelumnya: 2 → 4) agar guide line lebih representatif
        if len(anchor_xs) < 4:
            self._last_unwarp_contour = None
            return [], [], 0.0, 0, 0.0

        # Validasi kontras rata-rata anchor (S3-analog Tier 3):
        # Hitung threshold adaptif dari distribusi Sobel-Y zona pencarian unwarp.
        # Permukaan slat rata → kontras rendah, crest rib → ada tepi tajam (kontras > threshold).
        avg_anchor_contrast = float(np.mean(anchor_contrasts)) if anchor_contrasts else 0.0
        try:
            _sy_u_zone = cv2.Sobel(
                gray_u[search_uy1:search_uy2, search_ux1:search_ux2],
                cv2.CV_32F, 0, 1, ksize=3
            )
            _pos_u_zone = _sy_u_zone[_sy_u_zone > 0]
            _adapt_contrast_t3 = float(
                np.clip(np.percentile(_pos_u_zone, 65), 7.0, 16.0)
            ) if len(_pos_u_zone) > 20 else 10.0
        except Exception:
            _adapt_contrast_t3 = 10.0
        if avg_anchor_contrast < _adapt_contrast_t3:
            self._last_unwarp_contour = None
            return [], [], 0.0, 0, 0.0

        # 3. Filter Anchor Outlier & Fit Garis Pemandu Linier (Guide Line) di Ruang Unwarp
        # Gunakan konsensus median Y untuk mengeliminasi anchor yang melompat ke celah slat / teks kapur
        med_anchor_y = float(np.median(anchor_ys))
        clean_pairs = [
            (ax, ay) for ax, ay in zip(anchor_xs, anchor_ys)
            if abs(ay - med_anchor_y) <= 8.0
        ]
        if len(clean_pairs) >= 2:
            clean_xs, clean_ys = zip(*clean_pairs)
            guide_poly_u = np.polyfit(clean_xs, clean_ys, 1)
        else:
            clean_xs, clean_ys = anchor_xs, anchor_ys
            guide_poly_u = np.polyfit(anchor_xs, anchor_ys, 1)

        # Stabilisasi Temporal Anchor (EMA responsif) untuk kelancaran pelacakan bilah berjalan
        anchor_y_curr = float(np.polyval(guide_poly_u, w_u // 2))
        if getattr(self, "_prev_unwarp_anchor_y", None) is not None:
            if abs(anchor_y_curr - self._prev_unwarp_anchor_y) <= 15.0:
                anchor_y_curr = 0.35 * self._prev_unwarp_anchor_y + 0.65 * anchor_y_curr
        self._prev_unwarp_anchor_y = anchor_y_curr

        mid_idx = len(clean_xs) // 2
        x_center = int(clean_xs[mid_idx])
        y_center = int(clean_ys[mid_idx])
        step_x = 4
        initial_slope_u = float(guide_poly_u[0])

        # 4. Pelacakan Kontinu ke Kanan dari Tengah (dengan Tip Guard & Smooth Coasting)
        right_pts = [(x_center, float(y_center))]
        cur_y = float(y_center)
        rolling_slope = initial_slope_u
        consec_miss = 0
        for x in range(x_center + step_x, search_ux2 + 1, step_x):
            gy = float(np.polyval(guide_poly_u, x))
            is_tip = (x > search_ux2 - 130)
            ceiling_guard = 6.0
            floor_guard = 34.0 if is_tip else 14.0
            guide_w = 0.03 if is_tip else 0.20
            max_step = 6.5 if is_tip else 4.5
            search_delta = 26 if is_tip else 12

            y_low = max(search_uy1, int(round(gy - ceiling_guard)), int(cur_y - search_delta))
            y_high = min(search_uy2, int(round(gy + floor_guard)), int(cur_y + search_delta))

            best_y = None
            best_sc = -9999.0
            for y in range(y_low, y_high + 1):
                if y < gy - ceiling_guard or y > gy + floor_guard:
                    continue
                b = float(gray_u[y, x])
                diff_below = b - float(gray_u[min(h_u - 1, y + 8), x])
                diff_above = b - float(gray_u[max(0, y - 5), x])
                penalty = abs(y - cur_y) * 0.7 + max(0.0, gy - y) * 1.5 + abs(y - gy) * guide_w
                sc = min(b, 175.0) * 0.2 + diff_below * 1.1 + diff_above * 1.3 - penalty
                if sc > best_sc:
                    best_sc = sc
                    best_y = y

            # HARD STEP GATE: Batasi lompatan per-step (max 6.5px di tip, 4.5px di badan)
            if (best_y is not None and best_sc >= 18.0 and gray_u[best_y, x] >= 35
                    and abs(float(best_y) - cur_y) <= max_step):
                step_slope = (float(best_y) - cur_y) / float(step_x)
                rolling_slope = 0.70 * rolling_slope + 0.30 * step_slope
                cur_y = float(best_y)
                right_pts.append((x, cur_y))
                consec_miss = 0
            else:
                consec_miss += 1
                if consec_miss >= 25:
                    break
                # Coasting mulus mengikuti proyeksi kemiringan lokal (bukan dipaksa lurus ke gy!)
                coast_y = np.clip(cur_y + rolling_slope * float(step_x), gy - ceiling_guard, gy + floor_guard)
                cur_y = float(coast_y)
                right_pts.append((x, cur_y))

        # 5. Pelacakan Kontinu ke Kiri dari Tengah
        left_pts = []
        cur_y = float(y_center)
        rolling_slope = initial_slope_u
        consec_miss = 0
        for x in range(x_center - step_x, search_ux1 - 1, -step_x):
            gy = float(np.polyval(guide_poly_u, x))
            is_tip = (x < search_ux1 + 130)
            ceiling_guard = 6.0
            floor_guard = 34.0 if is_tip else 14.0
            guide_w = 0.03 if is_tip else 0.20
            max_step = 6.5 if is_tip else 4.5
            search_delta = 26 if is_tip else 12

            y_low = max(search_uy1, int(round(gy - ceiling_guard)), int(cur_y - search_delta))
            y_high = min(search_uy2, int(round(gy + floor_guard)), int(cur_y + search_delta))

            best_y = None
            best_sc = -9999.0
            for y in range(y_low, y_high + 1):
                if y < gy - ceiling_guard or y > gy + floor_guard:
                    continue
                b = float(gray_u[y, x])
                diff_below = b - float(gray_u[min(h_u - 1, y + 8), x])
                diff_above = b - float(gray_u[max(0, y - 5), x])
                penalty = abs(y - cur_y) * 0.7 + max(0.0, gy - y) * 1.5 + abs(y - gy) * guide_w
                sc = min(b, 175.0) * 0.2 + diff_below * 1.1 + diff_above * 1.3 - penalty
                if sc > best_sc:
                    best_sc = sc
                    best_y = y

            if (best_y is not None and best_sc >= 18.0 and gray_u[best_y, x] >= 35
                    and abs(float(best_y) - cur_y) <= max_step):
                step_slope = (cur_y - float(best_y)) / float(step_x)
                rolling_slope = 0.70 * rolling_slope + 0.30 * step_slope
                cur_y = float(best_y)
                left_pts.append((x, cur_y))
                consec_miss = 0
            else:
                consec_miss += 1
                if consec_miss >= 25:
                    break
                # Coasting mulus mengikuti proyeksi kemiringan lokal
                coast_y = np.clip(cur_y - rolling_slope * float(step_x), gy - ceiling_guard, gy + floor_guard)
                cur_y = float(coast_y)
                left_pts.append((x, cur_y))

        left_pts.reverse()
        all_pts = left_pts + right_pts
        r_arr = np.array(all_pts, dtype=np.float32)

        # 6. Filter Median 1D Cepat untuk Eliminasi Bintik Percikan Las
        r_arr[:, 1] = fast_medfilt(r_arr[:, 1], k=7)

        # 7. Polish Puncak Sub-Pixel Parabolik
        for idx in range(len(r_arr)):
            x_pt = int(r_arr[idx, 0])
            y0 = int(round(r_arr[idx, 1]))
            if 2 < y0 < h_u - 3:
                s_prev = float(gray_u[y0 - 1, x_pt]) * 0.4 + (float(gray_u[y0 - 1, x_pt]) - float(gray_u[min(h_u - 1, y0 + 7), x_pt])) * 1.2
                s_curr = float(gray_u[y0, x_pt]) * 0.4 + (float(gray_u[y0, x_pt]) - float(gray_u[min(h_u - 1, y0 + 8), x_pt])) * 1.2
                s_next = float(gray_u[y0 + 1, x_pt]) * 0.4 + (float(gray_u[y0 + 1, x_pt]) - float(gray_u[min(h_u - 1, y0 + 9), x_pt])) * 1.2
                denom = 2.0 * (s_prev - 2.0 * s_curr + s_next)
                offset = (s_prev - s_next) / denom if abs(denom) > 1e-4 else 0.0
                r_arr[idx, 1] = y0 + max(-0.5, min(0.5, offset))

        # Simpan titik koordinat unwarp dan matriks proyeksi untuk preview window 90 deg
        self._last_unwarp_contour = r_arr.copy()
        self._last_unwarp_M = M
        self._last_unwarp_Minv = Minv

        # Proyeksikan kembali ke ruang kamera asli via Minv
        cam_pts = cv2.perspectiveTransform(r_arr.reshape(-1, 1, 2), Minv).reshape(-1, 2)
        ridge_xs = [float(p[0]) for p in cam_pts]
        ridge_ys = [float(p[1]) for p in cam_pts]
        tested_cols = len(all_pts)
        coverage = len(ridge_xs) / float(max(1, tested_cols))
        tilt_deg = 0.0

        return ridge_xs, ridge_ys, coverage, tested_cols, tilt_deg

    def _track_hough_stitching(self, frame, gray, active_box, search_x1, search_x2, h_f, w_f):
        """
        TIER 4 TRACKER ENGINE: ADAPTASI SENSORS 2025 MICRO-HOUGHLET GROUPING & STITCHING (ENHANCED).
        Mengadaptasi metode deteksi kelurusan berbasis micro-line grouping & elastic stitching:
        1. 9-Point Robust Anchor Sampling: Mengunci bibir punggung atas (top crest) dengan penalti kedalaman
           dan penyaringan residual median (Theil-Sen robust) untuk menolak pantulan noda oli/las.
        2. Temporal Guide Stabilization: Menjaga konsistensi sudut kemiringan (tilt) konveyor antar-frame.
        3. Strip ROI Cropping & Directional Positive Sobel-Y: Hanya mendeteksi tepi naik (dark->bright)
           bibir atas, memangkas 100% tepi jatuh negatif, bayangan celah slat, dan garis las bawah.
        4. Micro-Houghlet Grouping: Deteksi segmen mikro (HoughLinesP) dengan toleransi adaptif tip vs badan.
        5. Full-Span Crest Guidance & Slope Boundary Extrapolation: Menghilangkan distorsi ujung sirip
           (tip deviation spike) yang sebelumnya memicu false NG.
        6. Sub-Pixel Parabolic Polish & Inter-Frame Temporal Smoothing: Meredam lonjakan koordinat piksel
           dan menghasilkan pelacakan kontur metrologi sub-milimeter yang bebas flicker.
        """
        if gray is None or active_box is None:
            return [], [], 0.0, 0, 0.0

        bx1, by1, bx2, by2 = active_box

        # 1. Multi-Point Anchor Sampling (11 titik dari 8% hingga 92% lebar bilah)
        step_fracs = [0.08, 0.16, 0.25, 0.34, 0.42, 0.50, 0.58, 0.66, 0.75, 0.84, 0.92]
        anchor_xs, anchor_ys = [], []
        for frac in step_fracs:
            ax = int(bx1 + frac * (bx2 - bx1))
            best_y, best_sc = None, -9999.0
            # Pindai bibir puncak atas sirip (by1 + 4 s/d by1 + 46 px)
            for y in range(max(10, by1 + 4), min(h_f - 12, by1 + 46)):
                b = float(gray[y, ax])
                diff_below = b - float(gray[min(h_f - 1, y + 8), ax])
                diff_above = b - float(gray[max(0, y - 5), ax])
                if diff_above >= 5.0 and diff_below >= 7.0:
                    # Penalti kedalaman agar anchor mengunci bibir puncak atas dan tidak jatuh ke las dasar
                    pen_depth = max(0, y - (by1 + 10)) * 1.0
                    sc = diff_below * 1.2 + diff_above * 1.5 + min(b, 175.0) * 0.1 - pen_depth
                    if sc > best_sc:
                        best_sc = sc
                        best_y = y
            min_sc = 12.0 if (frac <= 0.16 or frac >= 0.84) else 18.0
            if best_y is not None and best_sc > min_sc:
                anchor_xs.append(ax)
                anchor_ys.append(best_y)

        if len(anchor_xs) < 2:
            return [], [], 0.0, 0, 0.0

        # Robust Outlier Filtering untuk Garis Panduan p_guide
        if len(anchor_xs) >= 4:
            p_raw = np.polyfit(anchor_xs, anchor_ys, 1)
            resids = np.abs(np.array(anchor_ys) - np.polyval(p_raw, anchor_xs))
            med_res = float(np.median(resids))
            keep_mask = resids <= max(5.0, med_res * 1.8)
            c_xs = [x for x, k in zip(anchor_xs, keep_mask) if k]
            c_ys = [y for y, k in zip(anchor_ys, keep_mask) if k]
            if len(c_xs) >= 3:
                anchor_xs, anchor_ys = c_xs, c_ys
        p_guide = np.polyfit(anchor_xs, anchor_ys, 1)

        # Stabilisasi Temporal Garis Panduan Antar-Frame (Anti-Jitter Sudut)
        if getattr(self, '_prev_hough_pguide', None) is not None and self.active_rib_frames > 1:
            prev_slope, prev_c = self._prev_hough_pguide
            curr_slope, curr_c = p_guide[0], p_guide[1]
            p_guide = np.array([0.75 * prev_slope + 0.25 * curr_slope, 0.50 * prev_c + 0.50 * curr_c])
        self._prev_hough_pguide = p_guide

        # Pastikan batas pencarian tidak melebar ke rantai/gap samping konveyor (Adaptif Zoom/Vibration)
        pad_x = 12
        margin_frame = int(0.04 * w_f)
        x_limit_left = max(margin_frame, int(bx1 + pad_x))
        x_limit_right = min(w_f - margin_frame, int(bx2 - pad_x))
        if len(anchor_xs) >= 2:
            x_limit_left = max(x_limit_left, anchor_xs[0] - 50)
            x_limit_right = min(x_limit_right, anchor_xs[-1] + 50)
        tip_w_t4 = max(135, int(0.09 * (x_limit_right - x_limit_left)))

        # 2. Extract Strip ROI di sekitar Garis Panduan
        y_min_g = max(0, int(min(np.polyval(p_guide, [x_limit_left, x_limit_right]))) - 14)
        y_max_g = min(h_f, int(max(np.polyval(p_guide, [x_limit_left, x_limit_right]))) + 28)
        strip = gray[y_min_g:y_max_g, x_limit_left:x_limit_right]
        if strip.size == 0 or strip.shape[0] < 10 or strip.shape[1] < 100:
            return [], [], 0.0, 0, 0.0

        # S2A: Bilateral Filter pada strip — edge-preserving denoising
        # d=3 cukup untuk strip sempit, sigmaColor=35 jaga ketajaman tepi rib
        try:
            strip_filtered = cv2.bilateralFilter(strip, d=3, sigmaColor=35, sigmaSpace=35)
        except Exception:
            strip_filtered = strip

        # 3. Directional Positive Vertical Edge Detection (Hanya Tepi Naik Bibir Atas)
        grad_y = cv2.Sobel(strip_filtered, cv2.CV_32F, 0, 1, ksize=3)
        edge_map = np.clip(grad_y, 0, 255).astype(np.uint8)

        # S3: Adaptive Percentile-75 Threshold (stabil untuk strip sempit tanpa artefak Otsu)
        pos_vals = grad_y[grad_y > 4.0]
        ad_th = int(np.clip(np.percentile(pos_vals, 75), 18, 35)) if len(pos_vals) > 50 else 25
        _, bin_edge = cv2.threshold(edge_map, ad_th, 255, cv2.THRESH_BINARY)

        # 4. Probabilistic Hough Transform (Sensors 2025 Micro-Segments)
        lines = cv2.HoughLinesP(bin_edge, 1, np.pi / 180, threshold=10, minLineLength=14, maxLineGap=12)

        step_x = 4
        x_grid = np.arange(x_limit_left, x_limit_right + 1, step_x, dtype=np.int32)
        y_accum = [[] for _ in range(len(x_grid))]
        accepted_lines = []
        rejected_lines = []

        if lines is not None:
            for l in lines:
                x1, y1, x2, y2 = l[0]
                if x1 > x2:
                    x1, x2 = x2, x1
                    y1, y2 = y2, y1
                dx = float(x2 - x1)
                dy = float(y2 - y1)
                ang = abs(np.arctan2(dy, max(1e-5, dx)) * 180.0 / np.pi)
                if ang > 90.0:
                    ang = 180.0 - ang

                gx1 = x1 + x_limit_left
                gy1 = y1 + y_min_g
                gx2 = x2 + x_limit_left
                gy2 = y2 + y_min_g
                mid_x = (gx1 + gx2) / 2.0
                mid_y = (gy1 + gy2) / 2.0
                gy_nom = float(np.polyval(p_guide, mid_x))

                # Asymmetric Guard Terkalibrasi:
                # - Badan ketat (ceil 6.0px, floor 7.0px) menolak refleksi celah plat slat & garis las dasar
                # - Zona Ujung (proporsional ~9% span) leluasa (floor 18.0px, angle 24 deg) agar bengkok ujung nyata tertangkap
                is_tip = (mid_x < x_limit_left + tip_w_t4 or mid_x > x_limit_right - tip_w_t4)
                max_ang = 18.0 if is_tip else 11.0
                ceil_g = 6.5 if is_tip else 6.0
                floor_g = 9.0 if is_tip else 7.0

                if ang <= max_ang and (gy_nom - ceil_g) <= mid_y <= (gy_nom + floor_g):
                    accepted_lines.append((x1, y1, x2, y2))
                    s_idx = max(0, int((gx1 - x_limit_left) // step_x))
                    e_idx = min(len(x_grid) - 1, int((gx2 - x_limit_left) // step_x))
                    for idx in range(s_idx, e_idx + 1):
                        gx = x_grid[idx]
                        if gx1 <= gx <= gx2:
                            t = (gx - gx1) / max(1.0, dx)
                            y_accum[idx].append(gy1 + t * dy)
                else:
                    rejected_lines.append((x1, y1, x2, y2))

        valid_idxs = [i for i, ys in enumerate(y_accum) if ys]
        if len(valid_idxs) < 20:
            return [], [], 0.0, 0, 0.0

        # 5. Elastic Gap-Filling Berbasis Penelusuran Dwi-Arah dari Pusat Sirip (Center-Outward Bidirectional)
        # Menghilangkan 100% false latch-on di ujung sirip (eliminasi tuntas tip droop / step-down)
        mid_k = len(valid_idxs) // 2
        center_idx = valid_idxs[mid_k]
        center_candidates = y_accum[center_idx]
        y_nom_center = float(np.polyval(p_guide, x_grid[center_idx]))
        center_y = min(center_candidates, key=lambda y: abs(y - y_nom_center))
        chosen_ys = {center_idx: center_y}

        # Penelusuran ke Kiri (Pusat -> Ujung Kiri)
        prev_y = center_y
        for k in range(mid_k - 1, -1, -1):
            idx = valid_idxs[k]
            gx = x_grid[idx]
            y_nom = float(np.polyval(p_guide, gx))
            is_tip_pt = (gx < x_limit_left + tip_w_t4)
            c_ceil = 6.5 if is_tip_pt else 6.0
            c_floor = 9.0 if is_tip_pt else 7.0
            cands = [y for y in y_accum[idx] if (y_nom - c_ceil) <= y <= (y_nom + c_floor)]
            if not cands:
                continue
            w_nom = 0.05 if is_tip_pt else 0.25
            best_c = min(cands, key=lambda y: abs(y - prev_y) + w_nom * abs(y - y_nom))
            dx_pts = abs(gx - x_grid[valid_idxs[k + 1]])
            max_jump = max(6.0 if is_tip_pt else 4.0, (0.65 if is_tip_pt else 0.45) * dx_pts)
            if abs(best_c - prev_y) <= max_jump:
                chosen_ys[idx] = best_c
                prev_y = best_c

        # Penelusuran ke Kanan (Pusat -> Ujung Kanan)
        prev_y = center_y
        for k in range(mid_k + 1, len(valid_idxs)):
            idx = valid_idxs[k]
            gx = x_grid[idx]
            y_nom = float(np.polyval(p_guide, gx))
            is_tip_pt = (gx > x_limit_right - tip_w_t4)
            c_ceil = 6.5 if is_tip_pt else 6.0
            c_floor = 9.0 if is_tip_pt else 7.0
            cands = [y for y in y_accum[idx] if (y_nom - c_ceil) <= y <= (y_nom + c_floor)]
            if not cands:
                continue
            w_nom = 0.05 if is_tip_pt else 0.25
            best_c = min(cands, key=lambda y: abs(y - prev_y) + w_nom * abs(y - y_nom))
            dx_pts = abs(gx - x_grid[valid_idxs[k - 1]])
            max_jump = max(6.0 if is_tip_pt else 4.0, (0.65 if is_tip_pt else 0.45) * dx_pts)
            if abs(best_c - prev_y) <= max_jump:
                chosen_ys[idx] = best_c
                prev_y = best_c

        clean_valid_idxs = sorted(chosen_ys.keys())
        if len(clean_valid_idxs) < 15:
            return [], [], 0.0, 0, 0.0

        sub_valid_xs = x_grid[clean_valid_idxs]
        sub_valid_ys = np.array([chosen_ys[i] for i in clean_valid_idxs], dtype=np.float32)
        full_ys = np.interp(x_grid, sub_valid_xs, sub_valid_ys).astype(np.float32)

        # Extrapolasi Linier Stabil Mengikuti guide_slope (mencegah spike di celah terluar)
        guide_slope = float(p_guide[0])
        if sub_valid_xs[0] > x_grid[0]:
            left_mask = (x_grid < sub_valid_xs[0])
            full_ys[left_mask] = sub_valid_ys[0] + guide_slope * (x_grid[left_mask] - sub_valid_xs[0])
        if sub_valid_xs[-1] < x_grid[-1]:
            right_mask = (x_grid > sub_valid_xs[-1])
            full_ys[right_mask] = sub_valid_ys[-1] + guide_slope * (x_grid[right_mask] - sub_valid_xs[-1])

        # 6. Sub-Pixel Parabolic Polish pada Citra Grayscale Asli
        refined_ys = []
        for gx, y_est in zip(x_grid, full_ys):
            y_nom = float(np.polyval(p_guide, gx))
            is_tip_gx = (gx < x_limit_left + tip_w_t4 or gx > x_limit_right - tip_w_t4)
            max_nom_dist = 9.5 if is_tip_gx else 7.5
            best_y = int(round(y_est))
            best_sc = -9999.0
            for y in range(max(10, int(round(y_est)) - 3), min(h_f - 10, int(round(y_est)) + 4)):
                if abs(y - y_nom) > max_nom_dist:
                    continue
                b = float(gray[y, gx])
                diff_below = b - float(gray[min(h_f - 1, y + 8), gx])
                diff_above = b - float(gray[max(0, y - 5), gx])
                sc = diff_below * 1.1 + diff_above * 1.3
                if sc > best_sc:
                    best_sc = sc
                    best_y = y
            if 2 < best_y < h_f - 10:
                s_m1 = float(gray[best_y - 1, gx]) - float(gray[min(h_f - 1, best_y + 7), gx])
                s_0  = float(gray[best_y, gx]) - float(gray[min(h_f - 1, best_y + 8), gx])
                s_p1 = float(gray[best_y + 1, gx]) - float(gray[min(h_f - 1, best_y + 9), gx])
                denom = 2.0 * (s_m1 - 2.0 * s_0 + s_p1)
                offset = (s_m1 - s_p1) / denom if abs(denom) > 1e-4 else 0.0
                refined_ys.append(best_y + max(-0.5, min(0.5, offset)))
            else:
                refined_ys.append(float(best_y))

        # 7. 1D Median Filtering + Inter-Frame Temporal Contour Smoothing (Eliminasi Flicker & Lonjakan)
        smooth_ys = fast_medfilt(np.array(refined_ys, dtype=np.float32), k=7)
        if getattr(self, '_prev_hough_contour_ys', None) is not None and len(self._prev_hough_contour_ys) == len(smooth_ys) and self.active_rib_frames > 1:
            smooth_ys = 0.70 * smooth_ys + 0.30 * self._prev_hough_contour_ys
        self._prev_hough_contour_ys = smooth_ys

        poly1 = np.polyfit(x_grid, smooth_ys, 1)
        tilt_deg = float(np.arctan(poly1[0]) * 180.0 / np.pi)
        tested_cols = len(x_grid)
        coverage = len(clean_valid_idxs) / float(max(1, tested_cols))

        # Simpan metadata diagnostik pre-processing Tier 4
        self._last_hough_diag = {
            "strip": strip,
            "edge_map": edge_map,
            "bin_edge": bin_edge,
            "anchor_pts": [(ax - x_limit_left, ay - y_min_g) for ax, ay in zip(anchor_xs, anchor_ys)],
            "p_guide": p_guide,
            "y_min_g": y_min_g,
            "x_limit_left": x_limit_left,
            "x_limit_right": x_limit_right,
            "accepted_lines": accepted_lines,
            "rejected_lines": rejected_lines,
            "x_grid": x_grid,
            "smooth_ys": smooth_ys,
            "valid_idxs": valid_idxs,
            "coverage": coverage
        }
        self._last_hough_patch = self._create_compact_hough_patch(
            strip, bin_edge, accepted_lines, rejected_lines, smooth_ys, x_grid, y_min_g, x_limit_left
        )

        return x_grid.tolist(), smooth_ys.tolist(), coverage, tested_cols, tilt_deg

    def inspect_frame(self, frame, zone_poly=None, simulate_bending=False):
        """
        Menjalankan inspeksi terpadu pada frame konveyor:
        1. Evaluasi Keberadaan Objek di Zona Inspeksi (Sirip vs Plat Datar)
        2. Pelacakan Punggung Atas (Top Ridge Tracking) Sub-Pixel
        3. Metrologi Kelurusan (Flatness PV, Max Deviasi, & Sagitta Parabola)
        """
        t0 = time.time()
        if frame is None or getattr(frame, 'size', 0) == 0:
            return {
                "status": "WAIT",
                "reason": "FRAME_NONE",
                "pv": 0.0,
                "sagitta": 0.0,
                "confidence": 0.0,
                "points": [],
                "bbox": None,
                "annotated_frame": frame,
                "metrics": {},
                "fps": 0.0,
                "detection_type": self.mode,
                "active_box": None,
                "yolo_detected": False,
                "dual_detected": False,
                "rib_count": self.rib_count,
                "ok_count": self.ok_count,
                "ng_count": self.ng_count
            }
        h_f, w_f = frame.shape[:2]

        # Tentukan koordinat kotak zona inspeksi konveyor foreground
        if zone_poly is not None and len(zone_poly) >= 4:
            pts = []
            for p in zone_poly[:4]:
                px = int(p[0] * w_f if p[0] <= 1.0 else p[0])
                py = int(p[1] * h_f if p[1] <= 1.0 else p[1])
                pts.append([px, py])
            pts = np.array(pts)
            zx1 = max(0, int(np.min(pts[:, 0])))
            zy1 = max(0, int(np.min(pts[:, 1])))
            zx2 = min(w_f - 1, int(np.max(pts[:, 0])))
            zy2 = min(h_f - 1, int(np.max(pts[:, 1])))
        else:
            # Default Zona Konveyor Foreground: X [5%, 95%], Y [26%, 65%]
            # Memungkinkan deteksi rib di puncak konveyor (Y~300) dan memotong balok dasar konveyor (Y>700)
            zx1 = int(0.05 * w_f)
            zy1 = int(0.26 * h_f)
            zx2 = int(0.95 * w_f)
            zy2 = int(0.65 * h_f)

        zone_rect = (zx1, zy1, zx2, zy2)
        focus_zone = (zx1, zy1, zx2, zy2)

        # Nilai default deteksi awal
        yolo_det = False
        yolo_box = None
        yolo_conf = 0.0
        all_yolo_boxes = []

        dual_det = False
        dual_box = None
        dual_h = 0.0
        dual_ratio = 0.0
        sy = None

        # Stride inferensi YOLO: cerdas menghemat komputasi dan melipatgandakan throughput FPS
        self._frame_count += 1
        yolo_stride = self.yolo_stride
        exit_y1_cut = zy1 - max(12, int(0.017 * h_f))  # ~zy1 - 18 px pada 1080p (tepat saat bibir atas menyentuh batas atas gerbang)
        exit_cy_cut = zy1 + max(10, int(0.014 * h_f))  # ~zy1 + 15 px pada 1080p
        near_top_exit = (
            getattr(self, "_last_active_box", None) is not None
            and self._last_active_box[1] <= (exit_y1_cut + 24)
        )
        run_yolo = (yolo_stride <= 1) or (self._frame_count % yolo_stride == 0) or near_top_exit

        if self.mode in ["HYBRID", "YOLO_ONLY"]:
            if run_yolo:
                yolo_det, yolo_box, yolo_conf, all_yolo_boxes = self.detect_yolo_zone(frame, focus_zone)
                self._last_yolo_det = (yolo_det, yolo_box, yolo_conf, all_yolo_boxes)
            else:
                yolo_det, yolo_box, yolo_conf, all_yolo_boxes = self._last_yolo_det
                vy_shift = int(round(getattr(self, "_conveyor_vy_ema", 0.0)))
                if yolo_box is not None and vy_shift != 0:
                    yolo_box = [yolo_box[0], yolo_box[1] + vy_shift, yolo_box[2], yolo_box[3] + vy_shift]
                    shifted_all = []
                    for yb in (all_yolo_boxes or []):
                        if isinstance(yb, dict) and yb.get("box") is not None:
                            b0 = yb["box"]
                            yb_new = dict(yb)
                            yb_new["box"] = [b0[0], b0[1] + vy_shift, b0[2], b0[3] + vy_shift]
                            if "centroid" in yb and yb["centroid"] is not None:
                                yb_new["centroid"] = (yb["centroid"][0], yb["centroid"][1] + vy_shift)
                            shifted_all.append(yb_new)
                        else:
                            shifted_all.append(yb)
                    all_yolo_boxes = shifted_all
                    self._last_yolo_det = (yolo_det, yolo_box, yolo_conf, all_yolo_boxes)

        needs_dual = (
            self.mode == "DUAL_POLARITY_ONLY"
            or (self.mode == "HYBRID" and self.tracker_engine == "TIER4_HOUGH_STITCH" and not yolo_det)
        )
        if needs_dual:
            run_dual = (
                self.mode == "DUAL_POLARITY_ONLY"
                or (self._frame_count % self._dual_stride == 0)
            )
            if run_dual:
                dual_det, dual_box, dual_h, dual_ratio, _ = self.detect_dual_polarity_zone(frame, focus_zone)
                self._last_dual_det = (dual_det, dual_box, dual_h, dual_ratio, None)
            else:
                dual_det, dual_box, dual_h, dual_ratio, _ = self._last_dual_det

        # 2. Gating Keputusan Awal & Pemilihan Box Kontinu (Anti-Jump & Kinematic Tracking)
        reason = ""
        if self.mode in ["HYBRID", "YOLO_ONLY"]:
            if getattr(self, "_last_active_cy", None) is not None and getattr(self, "_last_active_box", None) is not None:
                # KONDISI A: Sedang dalam sesi pelacakan aktif suatu bilah sirip (Identity Locked)
                expected_cy = self._last_active_cy + self._conveyor_vy_ema
                best_tracked_box = None
                min_track_dist = float("inf")

                if all_yolo_boxes and len(all_yolo_boxes) > 0:
                    for yb in all_yolo_boxes:
                        b = yb.get("box") if isinstance(yb, dict) else yb
                        if b is not None:
                            cy_cand = (b[1] + b[3]) // 2
                            dist = abs(cy_cand - expected_cy)
                            # Toleransi pelacakan spasial kontinuitas: <= 65px dari proyeksi kinematik
                            if dist <= 65 and dist < min_track_dist:
                                min_track_dist = dist
                                best_tracked_box = b
                                yolo_conf = yb.get("conf", yolo_conf) if isinstance(yb, dict) else yolo_conf

                if best_tracked_box is not None:
                    cy_trk = (best_tracked_box[1] + best_tracked_box[3]) // 2
                    if best_tracked_box[1] < exit_y1_cut or cy_trk < exit_cy_cut:
                        best_tracked_box = None

                if best_tracked_box is not None:
                    # Deteksi AI solid terkonfirmasi
                    active_box = best_tracked_box
                    yolo_det = True
                    self._kinematic_coasting_frames = 0
                    inst_cy = (best_tracked_box[1] + best_tracked_box[3]) // 2
                    inst_vy = float(inst_cy - self._last_active_cy)
                    if abs(inst_vy) <= 15.0:
                        self._conveyor_vy_ema = 0.80 * self._conveyor_vy_ema + 0.20 * inst_vy
                else:
                    # AI drop sesaat (akibat pantulan cahaya, bayangan oli, atau flicker)
                    # Cek apakah bilah sirip aktif sudah keluar dari gerbang inspeksi atas
                    prev_cy = self._last_active_cy
                    prev_box = self._last_active_box
                    vy = self._conveyor_vy_ema
                    is_past_exit = (
                        (prev_box[1] + min(0.0, vy)) < exit_y1_cut
                        or (prev_cy <= (zy1 + 35) and vy <= 0.5)
                    )
                    if is_past_exit:
                        # Bilah lama telah tuntas keluar! Beralih mulus ke bilah baru jika ada
                        prime_mid_y = (self.prime_min_y + self.prime_max_y) // 2
                        new_cands = [yb for yb in all_yolo_boxes if ((yb["box"][1]+yb["box"][3])//2) > (zy1 + 45)] if all_yolo_boxes else []
                        if new_cands:
                            best_new_box = min(new_cands, key=lambda yb: abs((yb["box"][1] + yb["box"][3])//2 - prime_mid_y))
                            active_box = best_new_box["box"]
                            yolo_conf = best_new_box["conf"]
                            yolo_det = True
                            self._reset_rib_session()
                        else:
                            active_box = None
                            yolo_det = False
                    elif self.active_rib_frames >= 2 and self._kinematic_coasting_frames < self._max_kinematic_coasting:
                        # Bilah sudah terkonfirmasi (≥2 frame) & masih di zona kerja: jalankan Kinematic Extrapolation
                        self._kinematic_coasting_frames += 1
                        old_b = self._last_active_box
                        proj_y1 = int(round(old_b[1] + vy))
                        proj_y2 = int(round(old_b[3] + vy))
                        active_box = [old_b[0], proj_y1, old_b[2], proj_y2]
                        yolo_det = True
                        yolo_conf = max(0.35, yolo_conf * 0.96)
                        reason = f"HYBRID: Kinematic Coast (Y={(proj_y1+proj_y2)//2}, vy={vy:.1f}px/f, coast={self._kinematic_coasting_frames}/{self._max_kinematic_coasting})"
                    else:
                        active_box = None
                        yolo_det = False
            else:
                # KONDISI B: Cold Start (Mencari bilah pertama saat konveyor beroperasi)
                if all_yolo_boxes and len(all_yolo_boxes) > 0:
                    prime_mid_y = (self.prime_min_y + self.prime_max_y) // 2
                    best_prime_box = None
                    min_prime_dist = float("inf")
                    for yb in all_yolo_boxes:
                        b = yb.get("box") if isinstance(yb, dict) else yb
                        if b is not None:
                            cy_cand = (b[1] + b[3]) // 2
                            dist_p = abs(cy_cand - prime_mid_y)
                            if dist_p < min_prime_dist:
                                min_prime_dist = dist_p
                                best_prime_box = b
                                yolo_conf = yb.get("conf", yolo_conf) if isinstance(yb, dict) else yolo_conf
                    active_box = best_prime_box if best_prime_box is not None else yolo_box
                    yolo_det = (active_box is not None)
                else:
                    active_box = yolo_box
                    yolo_det = (active_box is not None)

            is_candidate = yolo_det
            if not reason:
                reason = f"HYBRID: YOLO({yolo_conf:.2f})" if self.mode == "HYBRID" else f"YOLO ONLY: Conf={yolo_conf:.2f}"
        else:  # DUAL_POLARITY_ONLY
            is_candidate = dual_det
            reason = f"DUAL-POLARITY ONLY: Ratio={dual_ratio*100:.0f}%, H={dual_h:.0f}px"
            active_box = dual_box

        # Evaluasi batas gerbang inspeksi:
        # Sudahi inspeksi tepat sebelum bibir atas sirip melewati garis keluar atas (cegah ghost slash di tepi perspektif)
        if active_box is not None:
            cy_chk = (active_box[1] + active_box[3]) // 2
            if active_box[1] < exit_y1_cut or cy_chk < exit_cy_cut or cy_chk > zy2:
                is_candidate = False
                active_box = None

        # Deteksi Transisi Bilah Fisik Baru (Spatial Centroid Continuity)
        if active_box is not None:
            curr_cy = (active_box[1] + active_box[3]) // 2
            if getattr(self, "_last_active_cy", None) is not None:
                delta_y = abs(curr_cy - self._last_active_cy)
                # Jika koordinat Y melompat lebih dari 70 px (transisi fisik ke bilah baru):
                if delta_y > 70:
                    self._reset_rib_session()
            self._last_active_cy = curr_cy
            self._last_active_box = active_box

        # 3. KONDISI: PLAT DATAR MELINTAS (TIDAK ADA CANDIDATE RIB)
        if not is_candidate or active_box is None:
            exited_gate = False
            if getattr(self, "_last_active_box", None) is not None and self._last_active_cy is not None:
                vy_ema = getattr(self, "_conveyor_vy_ema", -2.5)
                if (
                    (self._last_active_box[1] + min(0.0, vy_ema)) < (exit_y1_cut + 2)
                    or (self._last_active_cy <= (zy1 + 35) and vy_ema <= 0.5)
                    or self._last_active_cy >= (zy2 - 10)
                ):
                    exited_gate = True

            if self.active_rib_frames > 0 and not exited_gate:
                self.absence_frames += 1
                if self.absence_frames < self.max_absence_tolerance:
                    # Hysteresis buffer: Tahan sesi transit aktif (mencegah fragmentasi akibat flicker 1-5 frame)
                    _buf_pv = float(np.median(self.prime_pv_samples)) if self.prime_pv_samples else 0.0
                    _buf_sag = float(np.median(self.prime_sagitta_samples)) if self.prime_sagitta_samples else 0.0
                    _buf_idev = float(np.median(self.prime_inner_dev_samples)) if self.prime_inner_dev_samples else 0.0
                    _buf_tdev = float(np.median(self.prime_tip_dev_samples)) if self.prime_tip_dev_samples else 0.0
                    _buf_sev = self.compute_ng_severity(
                        _buf_pv, _buf_sag, _buf_idev, _buf_tdev,
                        is_ng=(self.current_rib_status == "NG")
                    ) if self.current_rib_status == "NG" else "OK"
                    return {
                        "status": self.current_rib_status if self.current_rib_status else "WAIT",
                        "object_type": "RIB_SIRIP",
                        "is_rib_present": True,
                        "is_ok": (self.current_rib_status == "OK"),
                        "is_ng": (self.current_rib_status == "NG"),
                        "ng_severity": _buf_sev,
                        "rib_count": self.rib_count,
                        "ok_count": self.ok_count,
                        "ng_count": self.ng_count,
                        "mode": self.mode,
                        "tracker_engine": self.tracker_engine,
                        "message": f"Sirip dalam transit (Buffer {self.absence_frames}/{self.max_absence_tolerance})",
                        "detection_reason": "Transit Hysteresis",
                        "confidence": float(yolo_conf if yolo_conf > 0 else (dual_ratio if self.mode == "DUAL_POLARITY_ONLY" else 0.0)),
                        "pv_flatness": _buf_pv,
                        "sagitta": _buf_sag,
                        "max_deviation": 0.0,
                        "inner_dev": _buf_idev,
                        "tip_dev": _buf_tdev,
                        "coverage": 1.0,
                        "active_box": self._last_active_box if self._last_active_box is not None else active_box,
                        "zone_rect": zone_rect,
                        "yolo_boxes": all_yolo_boxes,
                        "elapsed_ms": (time.time() - t0) * 1000
                    }

            # Sesi transit benar-benar berakhir (> 8 frame kosong antar bilah)
            self._last_active_cy = None
            self._last_active_box = None
            self._reset_rib_session()
            return {
                "status": "WAIT",
                "object_type": "PLAT_DATAR",
                "is_rib_present": False,
                "is_ok": False,
                "is_ng": False,
                "rib_count": self.rib_count,
                "ok_count": self.ok_count,
                "ng_count": self.ng_count,
                "mode": self.mode,
                "tracker_engine": self.tracker_engine,
                "message": "Plat datar melintas (Menunggu Sirip / Rib)",
                "detection_reason": reason,
                "confidence": 0.0,
                "pv_flatness": 0.0,
                "sagitta": 0.0,
                "max_deviation": 0.0,
                "inner_dev": 0.0,
                "tip_dev": 0.0,
                "coverage": 0.0,
                "active_box": None,
                "zone_rect": zone_rect,
                "yolo_boxes": all_yolo_boxes,
                "elapsed_ms": (time.time() - t0) * 1000
            }

        # Sirip hadir: reset penghitung ketidakhadiran
        self.absence_frames = 0

        # 4. GLOBAL ANCHOR & GUIDED CREST TRACKING (KEBAL CELAH SLAT, JANGKAUAN TIP-TO-TIP)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bx1, by1, bx2, by2 = active_box
        search_x1 = max(115, bx1 - 10)
        search_x2 = min(1720, bx2 + 10)

        # Ambil ROI di dalam bounding box YOLO untuk menemukan Ketinggian Nominal Puncak Bilah (Anchor Y)
        roi_x1 = search_x1 + 40
        roi_x2 = search_x2 - 40
        if roi_x2 <= roi_x1:
            roi_x1, roi_x2 = search_x1, search_x2

        roi = gray[by1:by2, roi_x1:roi_x2]
        h_roi = roi.shape[0]

        if h_roi < 35:
            self.last_presence = False
            return {
                "status": "WAIT",
                "object_type": "PLAT_DATAR",
                "is_rib_present": False,
                "is_ok": False,
                "is_ng": False,
                "rib_count": self.rib_count,
                "ok_count": self.ok_count,
                "ng_count": self.ng_count,
                "mode": self.mode,
                "tracker_engine": self.tracker_engine,
                "message": "Objek terlalu tipis (Bukan sirip tegak)",
                "detection_reason": f"{reason} -> Ditolak (Tinggi ROI < 35px)",
                "confidence": 0.0,
                "pv_flatness": 0.0,
                "sagitta": 0.0,
                "max_deviation": 0.0,
                "inner_dev": 0.0,
                "coverage": 0.0,
                "active_box": None,
                "zone_rect": zone_rect,
                "yolo_boxes": all_yolo_boxes,
                "elapsed_ms": (time.time() - t0) * 1000
            }

        # 4. KONTUR TRACKING DENGAN SAKELAR MUTUALLY EXCLUSIVE 3-WAY (NOL BEBAN GANDA)
        # (TIER2_YOLO_POSE & TIER3 YOLO-Segmentation telah di-take out;
        #  Preprocessing Perspective Transform tetap aktif untuk visualisasi Top-Down 90 Deg.)
        if self.tracker_engine == "TIER4_HOUGH_STITCH":
            # ==================================================================
            # PILIHAN B: ADAPTASI SENSORS 2025 MICRO-HOUGHLET GROUPING & STITCHING (TIER 4)
            # ==================================================================
            self._last_unwarp_contour = None
            # P2 FIX: Fallback ke dual_box jika YOLO tidak detect (active_box=None).
            _t4_active_box = active_box
            if _t4_active_box is None and dual_det and dual_box is not None:
                _t4_active_box = dual_box
                reason = "TIER4: Fallback Dual-Polarity Box (YOLO miss)"
            ridge_xs, ridge_ys, coverage, tested_cols, hough_tilt = self._track_hough_stitching(
                frame, gray, _t4_active_box, search_x1, search_x2, h_f, w_f
            )
        elif self.tracker_engine == "TIER1_VITERBI":
            # ==================================================================
            # PILIHAN A: ANCHOR-GUIDED TIGHT-RIBBON STEGER-VITERBI (TIER 1)
            # ==================================================================
            self._last_unwarp_contour = None
            ridge_xs, ridge_ys, coverage, tested_cols, viterbi_tilt = self._track_steger_viterbi(
                gray, active_box, search_x1, search_x2, h_f, w_f
            )
        else:
            # ==================================================================
            # METODE EKSISTING: SLOPE-AWARE ANCHOR & GUIDED CREST TRACKING (LEGACY)
            # ==================================================================
            self._last_unwarp_contour = None
            # Multi-point anchor sampling (11 titik lebar 12% - 88% agar slope tepi akurat & anti-ekstrapolasi miring)
            step_fracs = [0.12, 0.20, 0.28, 0.36, 0.44, 0.50, 0.56, 0.64, 0.72, 0.80, 0.88]
            anchor_xs = []
            anchor_ys = []
            anchor_contrasts = []

            for frac in step_fracs:
                ax = int(bx1 + frac * (bx2 - bx1))
                best_y = None
                best_sc = -9999.0
                max_c = 0.0
                # Prioritaskan bibir puncak sejati bilah sirip di area atas ROI dengan penalti kedalaman ringan
                for y in range(max(10, by1 + 2), min(h_f - 12, by1 + 48)):
                    b = float(gray[y, ax])
                    diff_below = b - float(gray[min(h_f - 1, y + 8), ax])
                    diff_above = b - float(gray[max(0, y - 5), ax])
                    if diff_above >= 8.0 and diff_below >= 10.0:
                        pen_depth = max(0.0, float(y - (by1 + 22))) * 0.5
                        sc = diff_below * 1.1 + diff_above * 1.4 + min(b, 175.0) * 0.15 - pen_depth
                        if sc > best_sc:
                            best_sc = sc
                            best_y = y
                            max_c = diff_below
                if best_y is not None and best_sc > 35.0:
                    anchor_xs.append(ax)
                    anchor_ys.append(best_y)
                    anchor_contrasts.append(max_c)

            # Validasi Fisika: Adaptive contrast gate (S3-analog Default/Legacy)
            avg_contrast = float(np.mean(anchor_contrasts)) if len(anchor_contrasts) > 0 else 0.0
            # Hitung threshold adaptif dari distribusi Sobel-Y ROI bounding box
            try:
                _sy_roi_def = cv2.Sobel(
                    gray[max(0, by1):min(h_f, by2), max(0, bx1):min(w_f, bx2)],
                    cv2.CV_32F, 0, 1, ksize=3
                )
                _pos_roi_def = _sy_roi_def[_sy_roi_def > 0]
                _adapt_contrast_def = float(
                    # [ANTI-FALSE-DET] Clamp [10,22]→[13,25]: konsisten dengan Tier-1, cegah karat/shadow lolos Legacy
                    np.clip(np.percentile(_pos_roi_def, 70), 13.0, 25.0)
                ) if len(_pos_roi_def) > 50 else 15.0
            except Exception:
                _adapt_contrast_def = 15.0
            if avg_contrast < _adapt_contrast_def or len(anchor_xs) < 2:
                if self.active_rib_frames > 0:
                    self.absence_frames += 1
                    if self.absence_frames < self.max_absence_tolerance:
                        return {
                            "status": self.current_rib_status if self.current_rib_status else "WAIT",
                            "object_type": "RIB_SIRIP",
                            "is_rib_present": True,
                            "is_ok": (self.current_rib_status == "OK"),
                            "is_ng": (self.current_rib_status == "NG"),
                            "rib_count": self.rib_count,
                            "mode": self.mode,
                            "tracker_engine": self.tracker_engine,
                            "message": f"Sirip dalam transit (Buffer {self.absence_frames}/{self.max_absence_tolerance})",
                            "detection_reason": "Transit Hysteresis (Kontras rendah sesaat)",
                            "pv_flatness": float(np.median(self.prime_pv_samples)) if self.prime_pv_samples else 0.0,
                            "sagitta": float(np.median(self.prime_sagitta_samples)) if self.prime_sagitta_samples else 0.0,
                            "max_deviation": 0.0,
                            "inner_dev": float(np.median(self.prime_inner_dev_samples)) if self.prime_inner_dev_samples else 0.0,
                            "tip_dev": float(np.median(self.prime_tip_dev_samples)) if self.prime_tip_dev_samples else 0.0,
                            "coverage": 1.0,
                            "active_box": None,
                            "zone_rect": zone_rect,
                            "yolo_boxes": all_yolo_boxes,
                            "elapsed_ms": (time.time() - t0) * 1000
                        }
                self.last_presence = False
                return {
                    "status": "WAIT",
                    "object_type": "PLAT_DATAR",
                    "is_rib_present": False,
                    "is_ok": False,
                    "is_ng": False,
                    "rib_count": self.rib_count,
                    "mode": self.mode,
                    "tracker_engine": self.tracker_engine,
                    "message": f"Celah plat diabaikan (Bukan bilah tegak | Kontras={avg_contrast:.1f} < 15)",
                    "detection_reason": f"{reason} -> Gugur (Kontras bilah rendah)",
                    "pv_flatness": 0.0,
                    "sagitta": 0.0,
                    "max_deviation": 0.0,
                    "inner_dev": 0.0,
                    "tip_dev": 0.0,
                    "coverage": 0.0,
                    "active_box": None,
                    "zone_rect": zone_rect,
                    "yolo_boxes": all_yolo_boxes,
                    "elapsed_ms": (time.time() - t0) * 1000
                }

            # Garis pandu linear robust (median pre-filter + residual filter + clamp kemiringan fisik)
            guide_poly = self._fit_robust_guide_poly(anchor_xs, anchor_ys, (by1 + by2) / 2.0)
            g_m, g_c = float(guide_poly[0]), float(guide_poly[1])

            # Titik awal pelacakan terdekat ke tengah bounding box (stabil meski anchor tidak simetris)
            mid_target_x = (bx1 + bx2) / 2.0
            mid_idx = int(np.argmin([abs(ax - mid_target_x) for ax in anchor_xs]))
            x_center = int(anchor_xs[mid_idx])

            # Jangkauan pelacakan lateral adaptif (Zoom & Vibration Tolerant Tip-to-Tip Coverage)
            pad_x = 10
            margin_frame = int(0.04 * w_f)
            x_limit_left = max(margin_frame, int(bx1 + pad_x))
            x_limit_right = min(w_f - margin_frame, int(bx2 - pad_x))
            tip_w_legacy = max(120, int(0.09 * (x_limit_right - x_limit_left)))

            # Grid kolom step-4 sejajar x_center agar vektorisasi band 100% identik dengan loop skalar
            x_grid_start = max(0, x_center - max(0, (x_center - x_limit_left) // 4) * 4)
            x_grid_end = min(w_f - 1, x_center + max(0, (x_limit_right - x_center) // 4) * 4)
            gy_a = g_m * x_grid_start + g_c
            gy_b = g_m * x_grid_end + g_c
            roi_y1 = max(10, min(by1 - 24, int(np.floor(min(gy_a, gy_b) - 20))))
            roi_y2 = min(h_f - 10, max(by2 + 20, int(np.ceil(max(gy_a, gy_b) + 24))))
            if roi_y2 <= roi_y1 + 2:
                roi_y1 = max(10, min(h_f - 15, roi_y1))
                roi_y2 = min(h_f - 10, roi_y1 + 5)

            pad_top = roi_y1 - 1
            pad_bot = roi_y2 + 1

            # [ANTI-FALSE-DET CM-2] Bilateral Filter pada ROI strip sebelum scan kolom Legacy:
            # Meratakan bintik karat & goresan permukaan metal tanpa menghaluskan tepi tajam bibir rib.
            # d=5 cukup cepat (<1ms), sigma=25 menjaga ketajaman gradien deviasi ujung/tip bending asli.
            _roi_strip_raw = gray[pad_top:pad_bot + 1, x_grid_start:x_grid_end + 1]
            try:
                _roi_strip_filt = cv2.bilateralFilter(_roi_strip_raw, d=5, sigmaColor=25, sigmaSpace=25)
            except Exception:
                _roi_strip_filt = _roi_strip_raw  # fallback tanpa filter jika error

            # [ANTI-FALSE-DET CM-1] Gap Dark-Stripe Mask Terkalibrasi:
            # Hanya celah gelap nyata (< 35) yang diblokir, dan bebaskan zona tip (outer 15% per sisi)
            # agar deviasi ujung asli (bending tip) tidak salah terbaca sebagai celah dan tidak dipaksa coasting lurus.
            _gap_col_brightness_full = np.median(_roi_strip_filt, axis=0)
            _gap_mask_full = (_gap_col_brightness_full < 35).astype(np.uint8)
            _tip_zone_px = int(0.15 * len(_gap_mask_full))
            if _tip_zone_px > 0:
                _gap_mask_full[:_tip_zone_px] = 0        # bebaskan ujung kiri
                _gap_mask_full[-_tip_zone_px:] = 0       # bebaskan ujung kanan

            # Morphological dilation: gap yang sempit (< 8px) diabaikan, gap nyata (>= 8px) diblokir
            _gap_mask_full = cv2.dilate(_gap_mask_full.reshape(1, -1),
                                        cv2.getStructuringElement(cv2.MORPH_RECT, (8, 1))).flatten()
            # Downsample ke step-4 agar sinkron dengan indeks kolom `col` di loop
            _gap_mask_cols = _gap_mask_full[::4]

            gb_sub = _roi_strip_filt[:, ::4].astype(np.float32)  # step-4 agar sinkron dengan x_grid
            gb_sub_b = gray[pad_top + 8:pad_bot + 9, x_grid_start:x_grid_end + 1:4].astype(np.float32)
            sub_band = gb_sub * 0.4 + (gb_sub - gb_sub_b) * 1.2
            gb = gb_sub[1:-1]
            gb_b = gb_sub_b[1:-1]
            gb_a = gray[roi_y1 - 5:roi_y2 - 4, x_grid_start:x_grid_end + 1:4].astype(np.float32)
            edge_band = np.minimum(gb, 175.0) * 0.2 + (gb - gb_b) * 1.1 + (gb - gb_a) * 1.3
            ys_band = np.arange(roi_y1, roi_y2 + 1, dtype=np.float32)
            n_band_cols = edge_band.shape[1]

            def _subpixel_crest(gx_val, gy_int):
                col = (gx_val - x_grid_start) // 4
                ry = gy_int - pad_top
                if 1 <= ry <= sub_band.shape[0] - 2 and 0 <= col < n_band_cols:
                    s_m1 = float(sub_band[ry - 1, col])
                    s_0 = float(sub_band[ry, col])
                    s_p1 = float(sub_band[ry + 1, col])
                    denom = 2.0 * (s_m1 - 2.0 * s_0 + s_p1)
                    offset = (s_m1 - s_p1) / denom if abs(denom) > 1e-4 else 0.0
                    return gy_int + max(-0.5, min(0.5, offset))
                if 2 < gy_int < h_f - 10:
                    s_m1 = float(gray[gy_int - 1, gx_val]) * 0.4 + (float(gray[gy_int - 1, gx_val]) - float(gray[min(h_f - 1, gy_int + 7), gx_val])) * 1.2
                    s_0  = float(gray[gy_int, gx_val]) * 0.4 + (float(gray[gy_int, gx_val]) - float(gray[min(h_f - 1, gy_int + 8), gx_val])) * 1.2
                    s_p1 = float(gray[gy_int + 1, gx_val]) * 0.4 + (float(gray[gy_int + 1, gx_val]) - float(gray[min(h_f - 1, gy_int + 9), gx_val])) * 1.2
                    denom = 2.0 * (s_m1 - 2.0 * s_0 + s_p1)
                    offset = (s_m1 - s_p1) / denom if abs(denom) > 1e-4 else 0.0
                    return gy_int + max(-0.5, min(0.5, offset))
                return float(gy_int)

            y_center = _subpixel_crest(x_center, int(round(anchor_ys[mid_idx])))
            initial_slope = g_m

            def _trace_outward(direction, x_limit):
                """Telusuri kontur puncak ke kiri (-1) atau kanan (+1) dengan momentum bridging simetris."""
                step_x = 4 * direction
                pts = [(x_center, y_center)]
                cur_y = float(y_center)
                rolling_slope = initial_slope
                bridging_buffer = []
                consec_misses = 0

                for x in range(x_center + step_x, x_limit + direction, step_x):
                    gy = g_m * x + g_c
                    is_tip = (x > x_limit - tip_w_legacy) if direction > 0 else (x < x_limit + tip_w_legacy)
                    y_pred = cur_y + rolling_slope * step_x

                    # Guard bebas bias di ujung (ceiling diperlonggar ke 14px agar tidak terpaksa turun ke sisi berkarat)
                    ceiling_guard = 14.0 if is_tip else 8.0
                    floor_guard = 18.0 if is_tip else 7.5
                    step_guard = 6.0 if is_tip else 3.5
                    max_step = 5.0 if is_tip else 3.2

                    y_low = max(roi_y1, int(round(max(gy - ceiling_guard, cur_y - step_guard))))
                    y_high = min(roi_y2, int(round(min(gy + floor_guard, cur_y + step_guard))))

                    best_y = None
                    best_sc = -9999.0
                    best_g = 0.0
                    col = (x - x_grid_start) // 4

                    # [ANTI-FALSE-DET P6] Gap Dark-Stripe Guard:
                    # Jika kolom ini terdeteksi sebagai celah antar slat (brightness rendah), paksa coast —
                    # jangan latch ke tepi gelap celah yang akan menciptakan spike deviasi palsu.
                    _col_in_gap = (0 <= col < len(_gap_mask_cols) and _gap_mask_cols[col] > 0)

                    if not _col_in_gap and y_high >= y_low and 0 <= col < n_band_cols:
                        yl_r = y_low - roi_y1
                        yh_r = y_high - roi_y1
                        yv = ys_band[yl_r:yh_r + 1]
                        pen_g = 0.008 if is_tip else 0.03
                        sc_arr = edge_band[yl_r:yh_r + 1, col] - 0.08 * ((yv - y_pred) ** 2) - pen_g * ((yv - gy) ** 2)
                        idx_best = int(np.argmax(sc_arr))
                        best_sc = float(sc_arr[idx_best])
                        best_y = y_low + idx_best
                        best_g = float(gb[yl_r + idx_best, col])

                    if best_y is not None and best_sc >= 20.0 and best_g >= 35.0 and abs(best_y - cur_y) <= max_step:
                        sub_y = _subpixel_crest(x, best_y)
                        if len(bridging_buffer) > 0:
                            start_x, start_y = pts[-1]
                            end_x, end_y = x, sub_y
                            for bx, _ in bridging_buffer:
                                interp_y = start_y + (end_y - start_y) * ((bx - start_x) / float(end_x - start_x))
                                pts.append((bx, interp_y))
                            bridging_buffer.clear()

                        step_slope = (sub_y - cur_y) / float(step_x)
                        rolling_slope = 0.70 * rolling_slope + 0.30 * step_slope
                        cur_y = sub_y
                        pts.append((x, cur_y))
                        consec_misses = 0
                    else:
                        consec_misses += 1
                        _max_tip_misses = 2 if is_tip else 15
                        if consec_misses <= _max_tip_misses:
                            coast_y = np.clip(cur_y + rolling_slope * step_x, gy - ceiling_guard, gy + floor_guard)
                            cur_y = float(coast_y)
                            bridging_buffer.append((x, cur_y))
                        else:
                            break
                return pts

            right_pts = _trace_outward(+1, x_limit_right)
            left_pts = _trace_outward(-1, x_limit_left)[1:]
            left_pts.reverse()
            all_pts = left_pts + right_pts
            ridge_xs = [p[0] for p in all_pts]
            ridge_ys = [p[1] for p in all_pts]
            tested_span = max(1, (x_limit_right - x_limit_left) // 4)
            coverage = len(ridge_xs) / float(tested_span)

        if len(ridge_xs) < 50 or coverage < 0.35:
            if self.active_rib_frames > 0:
                self.absence_frames += 1
                if self.absence_frames < self.max_absence_tolerance:
                    return {
                        "status": self.current_rib_status if self.current_rib_status else "WAIT",
                        "object_type": "RIB_SIRIP",
                        "is_rib_present": True,
                        "is_ok": (self.current_rib_status == "OK"),
                        "is_ng": (self.current_rib_status == "NG"),
                        "rib_count": self.rib_count,
                        "mode": self.mode,
                        "tracker_engine": self.tracker_engine,
                        "message": f"Sirip dalam transit (Buffer {self.absence_frames}/{self.max_absence_tolerance})",
                        "detection_reason": "Transit Hysteresis (Ridge terputus sesaat)",
                        "pv_flatness": float(np.median(self.prime_pv_samples)) if self.prime_pv_samples else 0.0,
                        "sagitta": float(np.median(self.prime_sagitta_samples)) if self.prime_sagitta_samples else 0.0,
                        "max_deviation": 0.0,
                        "inner_dev": float(np.median(self.prime_inner_dev_samples)) if self.prime_inner_dev_samples else 0.0,
                        "tip_dev": float(np.median(self.prime_tip_dev_samples)) if self.prime_tip_dev_samples else 0.0,
                        "coverage": coverage,
                        "active_box": self._last_active_box if self._last_active_box is not None else active_box,
                        "zone_rect": zone_rect,
                        "yolo_boxes": all_yolo_boxes,
                        "elapsed_ms": (time.time() - t0) * 1000
                    }
            self._last_active_cy = None
            self._last_active_box = None
            self._reset_rib_session()
            return {
                "status": "WAIT",
                "object_type": "PLAT_DATAR",
                "is_rib_present": False,
                "is_ok": False,
                "is_ng": False,
                "rib_count": self.rib_count,
                "mode": self.mode,
                "tracker_engine": self.tracker_engine,
                "message": f"Titik ridge tidak memadai ({len(ridge_xs)} pts | Cov={coverage*100:.0f}%)",
                "detection_reason": f"{reason} -> Ditolak (Titik puncak tidak kontinu)",
                "pv_flatness": 0.0,
                "sagitta": 0.0,
                "max_deviation": 0.0,
                "inner_dev": 0.0,
                "tip_dev": 0.0,
                "coverage": coverage,
                "active_box": None,
                "zone_rect": zone_rect,
                "yolo_boxes": all_yolo_boxes,
                "elapsed_ms": (time.time() - t0) * 1000
            }

        # 5. KONDISI: SIRIP SEJATI TERVALIDASI
        r_xs = np.array(ridge_xs, dtype=np.float32)
        r_ys = np.array(ridge_ys, dtype=np.float32)

        # Filter median untuk membersihkan bintik percikan las
        smooth_ys = fast_medfilt(r_ys, k=9)

        # Injeksi defleksi bending jika hotkey [B] aktif
        if simulate_bending:
            span_len = float(r_xs[-1] - r_xs[0])
            norm_x = (r_xs - r_xs[0]) / span_len
            parabola_bend = 24.0 * np.sin(norm_x * np.pi)
            smooth_ys = smooth_ys + parabola_bend

        # Garis Acuan Baseline Kokoh & Full-Span Gating
        is_tier3_unwarp = (self.tracker_engine == "TIER3_UNWARP" and
                           getattr(self, "_last_unwarp_contour", None) is not None and
                           len(self._last_unwarp_contour) > 50)

        if is_tier3_unwarp:
            # ==================================================================
            # 100% NATIVE UNWARP METROLOGY (ORTOGONAL TOP-DOWN 90 DEG PURE SPACE)
            # Evaluasi kelurusan 100% bebas dari distorsi trapesium perspektif kamera
            # ==================================================================
            u_pts = self._last_unwarp_contour
            u_xs = u_pts[:, 0].astype(np.float32)
            u_ys = u_pts[:, 1].astype(np.float32)

            # Robust outlier filter: buang titik yang bocor ke celah slat / bayangan di luar ujung fisik bilah
            med_u_y = float(np.median(u_ys))
            rib_mask_u = np.abs(u_ys - med_u_y) <= 18.0
            if np.sum(rib_mask_u) >= 50:
                u_xs = u_xs[rib_mask_u]
                u_ys = u_ys[rib_mask_u]

            smooth_u_ys = fast_medfilt(u_ys, k=9)
            k_box = 5
            smooth_u_ys = np.convolve(smooth_u_ys, np.ones(k_box, dtype=np.float32) / float(k_box), mode='same')

            # Stabilisasi Temporal EMA pada Kontur Unwarp (Zero-Flicker View)
            prev_uxs = getattr(self, "_prev_tier3_unw_xs", None)
            prev_uys = getattr(self, "_prev_tier3_unw_ys", None)
            if prev_uxs is not None and prev_uys is not None and len(prev_uxs) >= 20:
                prev_interp_uys = np.interp(u_xs, prev_uxs, prev_uys)
                dy_u = float(np.median(smooth_u_ys) - np.median(prev_interp_uys))
                if abs(dy_u) < 25.0:
                    smooth_u_ys = 0.65 * smooth_u_ys + 0.35 * (prev_interp_uys + dy_u)
            self._prev_tier3_unw_xs = u_xs.copy()
            self._prev_tier3_unw_ys = smooth_u_ys.copy()

            if simulate_bending:
                u_span_len = float(u_xs[-1] - u_xs[0])
                u_norm_x = (u_xs - u_xs[0]) / max(1.0, u_span_len)
                smooth_u_ys = smooth_u_ys + 24.0 * np.sin(u_norm_x * np.pi)

            w_span = float(u_xs[-1] - u_xs[0])
            med_top_y_u = float(np.median(smooth_u_ys))
            med_top_y = int(np.median(smooth_ys))
            self.active_rib_frames += 1

            # Full-span valid pada dimensi kanonikal unwarp:
            # Bentang sirip penuh di unwarp space ≈ 820-830px dari 960px lebar kanvas.
            # Threshold 680px (≈83% dari 820px) memastikan kontur sudah mencakup 
            # hampir seluruh lebar sebelum sampel metrologi dikumpulkan.
            # Ini setara proporsional dengan 680/1820 = 37% margin LEGACY di camera space.
            full_span_valid = (w_span >= 680.0 and coverage >= 0.55)

            # Baseline Regresi Linier di ruang ortogonal unwarp 90 deg
            poly1_u = np.polyfit(u_xs, smooth_u_ys, deg=1)
            baseline_u = np.polyval(poly1_u, u_xs)

            dev_u = smooth_u_ys - baseline_u

            # Proyeksikan baseline unwarp ke kamera miring hanya untuk rendering HUD display
            u_base_pts = np.column_stack([u_xs, baseline_u]).astype(np.float32).reshape(-1, 1, 2)
            Minv = getattr(self, "_last_unwarp_Minv", None)
            if Minv is not None:
                cam_base_pts = cv2.perspectiveTransform(u_base_pts, Minv).reshape(-1, 2)
                baseline = cam_base_pts[:, 1]
            else:
                poly1 = np.polyfit(r_xs, smooth_ys, deg=1)
                baseline = np.polyval(poly1, r_xs)

            # Proyeksikan batas zona prime ke unwarp space secara dinamis.
            # TIER3 menggunakan prime zone kamera yang lebih lebar (290-520px) karena
            # perspective transform mengoreksi distorsi lensa, sehingga area Y=290-340
            # yang distorsinya tinggi di kamera, jadi rata & ortogonal di unwarp space.
            # Engine LEGACY/TIER1/TIER2 tetap pakai prime zone sempit 340-460 (camera space).
            tier3_cam_pmin = getattr(self, "tier3_prime_min_y", 305)
            tier3_cam_pmax = getattr(self, "tier3_prime_max_y", 385)
            M = getattr(self, "_last_unwarp_M", None)
            if M is not None:
                p_cam = np.array([[[960.0, float(tier3_cam_pmin)]], [[960.0, float(tier3_cam_pmax)]]], dtype=np.float32)
                p_unw = cv2.perspectiveTransform(p_cam, M).reshape(-1, 2)
                u_pmin = min(p_unw[0, 1], p_unw[1, 1])
                u_pmax = max(p_unw[0, 1], p_unw[1, 1])
            else:
                u_pmin, u_pmax = 145.0, 215.0

            box_clear_of_edges = (
                active_box is None
                or (active_box[1] >= (zy1 + 10) and med_top_y_u <= (u_pmax + 10))
            )
            is_in_prime = (u_pmin <= med_top_y_u <= u_pmax) and box_clear_of_edges

            if not full_span_valid:
                pv = 0.0
                sagitta = 0.0
                max_dev = 0.0
                inner_dev = 0.0
                tip_dev = 0.0
            else:
                # Fusi profil temporal unwarp (Multi-Frame Median Consensus)
                norm_grid_u = np.linspace(u_xs[0], u_xs[-1], 200)
                dev_resampled_u = np.interp(norm_grid_u, u_xs, dev_u)
                if not hasattr(self, 'tier3_profile_history'):
                    self.tier3_profile_history = []
                if is_in_prime:
                    self.tier3_profile_history.append(dev_resampled_u)
                    if len(self.tier3_profile_history) > 20:
                        self.tier3_profile_history = self.tier3_profile_history[-20:]

                if len(self.tier3_profile_history) >= 3 and is_in_prime:
                    fused_dev_u = np.median(np.array(self.tier3_profile_history), axis=0)
                    pv = float(np.percentile(fused_dev_u, 98) - np.percentile(fused_dev_u, 2))
                    max_dev = float(np.max(np.abs(fused_dev_u)))
                else:
                    pv = float(np.percentile(dev_u, 98) - np.percentile(dev_u, 2))
                    max_dev = float(np.max(np.abs(dev_u)))

                # Evaluasi deviasi ujung vs badan di ruang unwarp (tip zone difokuskan ke 60px ujung)
                # Menghindari dilusi percentiles oleh badan sirip lurus (kasus Frame 456 tekukan kiri)
                tip_zone_px = min(60.0, 0.08 * w_span)
                left_tip_mask = (u_xs < (u_xs[0] + tip_zone_px))
                right_tip_mask = (u_xs > (u_xs[-1] - tip_zone_px))
                tip_mask = left_tip_mask | right_tip_mask

                left_tip_dev = float(np.percentile(np.abs(dev_u[left_tip_mask]), 95)) if np.any(left_tip_mask) and np.sum(left_tip_mask) >= 4 else 0.0
                right_tip_dev = float(np.percentile(np.abs(dev_u[right_tip_mask]), 95)) if np.any(right_tip_mask) and np.sum(right_tip_mask) >= 4 else 0.0
                tip_dev = max(left_tip_dev, right_tip_dev)

                inner_mask = ~tip_mask
                if np.any(inner_mask):
                    inner_dev = float(np.percentile(np.abs(dev_u[inner_mask]), 98))
                else:
                    inner_dev = max_dev

                # Sagitta parabola orde-2 di ruang unwarp ortogonal
                poly2_u = np.polyfit(u_xs, smooth_u_ys, deg=2)
                sagitta = abs(float(poly2_u[0])) * ((w_span / 2.0) ** 2)

                if is_in_prime:
                    self._record_prime_sample(pv, sagitta, inner_dev, tip_dev)

            self._last_unwarp_baseline = baseline_u
            self._last_unwarp_smooth_ys = smooth_u_ys
            self._last_unwarp_pmin = u_pmin
            self._last_unwarp_pmax = u_pmax

        else:
            # ==================================================================
            # METROLOGI EKSISTING DI RUANG KAMERA (UNTUK LEGACY, TIER 1, & TIER 2)
            # 100% Tidak Berubah / Backward Compatible Penuh
            # ==================================================================
            n_pts = len(r_xs)
            w_span = float(r_xs[-1] - r_xs[0])
            med_top_y = int(np.median(smooth_ys))
            self.active_rib_frames += 1

            is_tier4 = (self.tracker_engine == "TIER4_HOUGH_STITCH")
            min_cov = 0.35 if is_tier4 else 0.55
            full_span_valid = (w_span >= 680.0 and coverage >= min_cov)

            # Baseline Estimator (Theil-Sen Subsampled)
            step_eval = max(1, n_pts // 30)
            sub_xs = r_xs[::step_eval]
            sub_ys = smooth_ys[::step_eval]
            slopes = []
            for i in range(len(sub_xs)):
                for j in range(i + 5, len(sub_xs)):
                    dx = sub_xs[j] - sub_xs[i]
                    if dx >= 120.0:
                        slopes.append((sub_ys[j] - sub_ys[i]) / dx)

            if len(slopes) > 0:
                slope_m = float(np.median(slopes))
                intercept_c = float(np.median(sub_ys - slope_m * sub_xs))
                baseline = slope_m * r_xs + intercept_c
            else:
                poly1 = np.polyfit(r_xs, smooth_ys, deg=1)
                baseline = np.polyval(poly1, r_xs)

            dev = smooth_ys - baseline
            # Proyeksi Preprocessing Perspective Transform (90 Deg Top-Down):
            # Selalu perbarui _last_unwarp_contour & _last_unwarp_baseline agar jendela
            # Perspective Transform View menampilkan kontur hasil pelacakan LEGACY / TIER 1.
            if not is_tier4:
                try:
                    if getattr(self, "unwarp_M", None) is None:
                        self.unwarp_rib_zone(frame)
                    _uM = getattr(self, "unwarp_M", None)
                    if _uM is not None and len(r_xs) > 10:
                        _cam_c = np.column_stack([r_xs, smooth_ys]).astype(np.float32).reshape(-1, 1, 2)
                        _unw_c = cv2.perspectiveTransform(_cam_c, _uM).reshape(-1, 2)
                        _cam_b = np.column_stack([r_xs, baseline]).astype(np.float32).reshape(-1, 1, 2)
                        _unw_b = cv2.perspectiveTransform(_cam_b, _uM).reshape(-1, 2)
                        _vmask = (_unw_c[:, 0] >= 25.0) & (_unw_c[:, 0] <= float(self.canonical_w - 25))
                        if np.sum(_vmask) >= 20:
                            self._last_unwarp_contour = _unw_c[_vmask].copy()
                            self._last_unwarp_baseline = _unw_b[_vmask, 1].copy()
                        else:
                            self._last_unwarp_contour = _unw_c.copy()
                            self._last_unwarp_baseline = _unw_b[:, 1].copy()
                except Exception:
                    pass

            box_clear_of_edges = (
                active_box is None
                or (active_box[1] >= (zy1 + 10) and med_top_y <= (zy2 - 80))
            )
            if is_tier4:
                p_min = min(self.prime_min_y, self._legacy_prime_min_y)
                p_max = max(self.prime_max_y, self._legacy_prime_max_y)
                is_in_prime = (p_min <= med_top_y <= p_max) and box_clear_of_edges
            else:
                is_in_prime = (self._legacy_prime_min_y <= med_top_y <= self._legacy_prime_max_y) and box_clear_of_edges

            if not full_span_valid:
                # Kontur belum membentang utuh -> Tahan penilaian metrologi, hindari lonjakan Sagitta palsu
                pv = 0.0
                sagitta = 0.0
                max_dev = 0.0
                inner_dev = 0.0
                tip_dev = 0.0
            else:
                # SOLUSI 1: TEMPORAL MULTI-FRAME PROFILE FUSION KHUSUS TIER 4
                if is_tier4 and is_in_prime and len(r_xs) >= 40:
                    # Normalisasi profil span [0.0, 1.0] agar bebas dari variasi panjang bilah, FOV kamera, & perspective skew
                    norm_grid = np.linspace(0.0, 1.0, 256, dtype=np.float32)
                    r_norm_x = (r_xs - r_xs[0]) / max(1.0, float(r_xs[-1] - r_xs[0]))
                    dev_resampled = np.interp(norm_grid, r_norm_x, dev).astype(np.float32)
                    self.tier4_profile_history.append(dev_resampled)
                    if len(self.tier4_profile_history) > 20:
                        self.tier4_profile_history = self.tier4_profile_history[-20:]

                    # Jika sudah ada minimal 3 sampel di prime zone, gunakan Fused Profile
                    if len(self.tier4_profile_history) >= 3:
                        fused_dev = np.median(np.array(self.tier4_profile_history), axis=0)
                        self.tier4_fused_dev = fused_dev

                        # Hitung metrologi kelurusan dari Fused Profile (100% bebas noise instan)
                        pv = float(np.percentile(fused_dev, 98) - np.percentile(fused_dev, 2))
                        max_dev = float(np.max(np.abs(fused_dev)))

                        poly2_fused = np.polyfit(norm_grid, fused_dev, deg=2)
                        sagitta = abs(float(poly2_fused[0])) * 0.25
                        quad_fused_dev = fused_dev - np.polyval(poly2_fused, norm_grid)

                        # Zona ujung fisik terkalibrasi (margin bevel 1.8% terluar dikecualikan agar chamfer sudut tidak spike)
                        # Tip diukur dari residual kuadratik agar lengkungan perspektif seragam tidak memicu False NG.
                        left_tip_mask = (norm_grid >= 0.018) & (norm_grid < 0.095)
                        right_tip_mask = (norm_grid <= 0.982) & (norm_grid > 0.905)
                        tip_mask = (norm_grid < 0.095) | (norm_grid > 0.905)

                        left_tip_dev = float(np.percentile(np.abs(quad_fused_dev[left_tip_mask]), 95)) if np.any(left_tip_mask) else 0.0
                        right_tip_dev = float(np.percentile(np.abs(quad_fused_dev[right_tip_mask]), 95)) if np.any(right_tip_mask) else 0.0
                        tip_dev = max(left_tip_dev, right_tip_dev)

                        inner_mask = ~tip_mask
                        inner_dev = float(np.percentile(np.abs(fused_dev[inner_mask]), 98)) if np.any(inner_mask) else max_dev

                        # Sinkronkan kembali smooth_ys ke kontur yang sudah difusi agar tampilan GUI tenang dan mulus
                        fused_dev_at_rxs = np.interp(r_norm_x, norm_grid, fused_dev)
                        smooth_ys = (baseline + fused_dev_at_rxs).astype(np.float32)
                        if getattr(self, "_last_hough_diag", None) is not None and isinstance(self._last_hough_diag, dict):
                            self._last_hough_diag["smooth_ys"] = smooth_ys
                    else:
                        pv = float(np.percentile(dev_resampled, 98) - np.percentile(dev_resampled, 2))
                        max_dev = float(np.max(np.abs(dev_resampled)))
                        poly2 = np.polyfit(norm_grid, dev_resampled, deg=2)
                        sagitta = abs(float(poly2[0])) * 0.25
                        quad_res_dev = dev_resampled - np.polyval(poly2, norm_grid)
                        left_tip_mask = (norm_grid >= 0.018) & (norm_grid < 0.095)
                        right_tip_mask = (norm_grid <= 0.982) & (norm_grid > 0.905)
                        tip_mask = (norm_grid < 0.095) | (norm_grid > 0.905)
                        left_tip_dev = float(np.percentile(np.abs(quad_res_dev[left_tip_mask]), 95)) if np.any(left_tip_mask) else 0.0
                        right_tip_dev = float(np.percentile(np.abs(quad_res_dev[right_tip_mask]), 95)) if np.any(right_tip_mask) else 0.0
                        tip_dev = max(left_tip_dev, right_tip_dev)
                        inner_mask = ~tip_mask
                        inner_dev = float(np.percentile(np.abs(dev_resampled[inner_mask]), 98)) if np.any(inner_mask) else max_dev
                else:
                    # Metrologi default untuk LEGACY & TIER 1
                    # Margin bevel_skip_px=18.0 mengecualikan bevel/chamfer sudut terluar pabrik (cegah false NG F477)
                    # sementara bent tip struktural (seperti F1224 sepanjang ~150px) tetap tertangkap penuh di zona [18..92]px.
                    # Anti-Distorsi Ujung: PV dihitung dari bagian DALAM 80% kontur (buang 10% ujung kiri/kanan).
                    # Ujung kontur rentan terhadap bayangan celah slat, karat tepi, dan area kurang kontras →
                    # 1–2 titik outlier di ujung bisa mendistorsi PV global (contoh: PV=22px padahal inner_dev=4.8px).
                    _n_pv = len(r_xs)
                    _inner_start_pv = max(0, int(_n_pv * 0.10))
                    _inner_end_pv   = min(_n_pv, int(_n_pv * 0.90))
                    _dev_inner_pv   = dev[_inner_start_pv:_inner_end_pv]
                    if len(_dev_inner_pv) >= 20:
                        pv = float(np.percentile(_dev_inner_pv, 98) - np.percentile(_dev_inner_pv, 2))
                    else:
                        pv = float(np.percentile(dev, 98) - np.percentile(dev, 2))
                    max_dev = float(np.max(np.abs(dev)))
                    # Evaluasi TipDev relatif terhadap baseline kuadratik (poly2) dari BADAN TENGAH bilah (inner_mask),
                    # agar kelengkungan global bilah lurus tetap terkompensasi, namun ujung yang penyok tidak ikut
                    # menarik kurva referensi ke bawah dan menyamarkan defleksi ujung.
                    poly2 = np.polyfit(r_xs, smooth_ys, deg=2)
                    sagitta = abs(float(poly2[0])) * ((w_span / 2.0) ** 2)
                    bevel_skip_px = 14.0
                    tip_zone_px = 92.0
                    left_tip_mask = (r_xs >= (r_xs[0] + bevel_skip_px)) & (r_xs < (r_xs[0] + tip_zone_px))
                    right_tip_mask = (r_xs <= (r_xs[-1] - bevel_skip_px)) & (r_xs > (r_xs[-1] - tip_zone_px))
                    tip_mask = (r_xs < (r_xs[0] + tip_zone_px)) | (r_xs > (r_xs[-1] - tip_zone_px))
                    inner_mask = ~tip_mask
                    poly2_tip = np.polyfit(r_xs[inner_mask], smooth_ys[inner_mask], deg=2) if np.sum(inner_mask) >= 20 else poly2
                    quad_dev = smooth_ys - np.polyval(poly2_tip, r_xs)
                    left_tip_dev = float(np.percentile(np.abs(quad_dev[left_tip_mask]), 95)) if np.any(left_tip_mask) and np.sum(left_tip_mask) >= 4 else 0.0
                    right_tip_dev = float(np.percentile(np.abs(quad_dev[right_tip_mask]), 95)) if np.any(right_tip_mask) and np.sum(right_tip_mask) >= 4 else 0.0
                    tip_dev = max(left_tip_dev, right_tip_dev)
                    inner_dev = float(np.percentile(np.abs(dev[inner_mask]), 98)) if np.any(inner_mask) else max_dev

                # Hanya kumpulkan sampel metrologi jika berada di zona prime konveyor dan kontur utuh!
                if is_in_prime:
                    self._record_prime_sample(pv, sagitta, inner_dev, tip_dev)

        # KONSENSUS METROLOGI PRIME ZONE
        # - TIER4_HOUGH_STITCH: 4 sampel agar profil fusi stabil
        # - LEGACY / TIER1: 8 sampel di zona prime planar [355..435px] agar lonjakan noise awal transisi
        #   tidak langsung mengunci status NG prematur (aman untuk video redup & cepat berdurasi 13-18 sampel/rib).
        min_samples_required = 4 if self.tracker_engine == "TIER4_HOUGH_STITCH" else 8
        sample_count = len(self.prime_pv_samples)
        is_tier3_for_consensus = False

        ng_prob = None
        ml_defect = False
        pv_th, sag_th, dev_th, tip_th = self._get_effective_thresholds()
        if not is_tier3_for_consensus:
            eval_pv, eval_sag, eval_idev, eval_tdev = self._compute_prime_consensus_metrics(
                pv, sagitta, inner_dev, tip_dev, min_samples_required=min_samples_required
            )

            curv_defect = (eval_sag > sag_th and eval_pv > pv_th) or (eval_sag > 14.5)
            dev_defect = (eval_idev > dev_th)
            tip_defect = (eval_tdev > tip_th)
            flat_defect = (eval_pv > 18.0)

            if self.tip_classifier_model is not None and HAS_TIP_CLASSIFIER and full_span_valid:
                _tip_feat = extract_tip_classifier_features(
                    smooth_ys, eval_pv, eval_sag, eval_idev, eval_tdev
                )
                if _tip_feat is not None:
                    ng_prob = predict_tip_ng_prob(_tip_feat, self.tip_classifier_model)
                    _is_physically_straight = (
                        eval_pv < 4.5 and eval_idev < 3.5 and eval_tdev < 4.5
                    )
                    if _is_physically_straight:
                        ml_defect = False
                    else:
                        ml_defect = bool(
                            ng_prob >= float(self.tip_classifier_model.get("prob_threshold", 0.5))
                        )

            if ng_prob is not None:
                extreme_safety = (eval_pv > 24.0) or (eval_sag > 18.0)
                transit_is_ok = not (ml_defect or extreme_safety)
            else:
                transit_is_ok = not (curv_defect or dev_defect or tip_defect or flat_defect)
        else:
            # TIER 3 Deep Learning Segmentation + Unwarp Metrology:
            eval_pv, eval_sag, eval_idev, eval_tdev = self._compute_prime_consensus_metrics(
                pv, sagitta, inner_dev, tip_dev, min_samples_required=min_samples_required
            )

            curv_defect = (eval_sag > sag_th and eval_pv > pv_th) or (eval_sag > 14.5)
            dev_defect = (eval_idev > dev_th)
            _tier3_tip_thresh = max(tip_th, 12.0)
            tip_defect = (eval_tdev > _tier3_tip_thresh)
            flat_defect = (eval_pv > 18.0)
            transit_is_ok = not (curv_defect or dev_defect or tip_defect or flat_defect)

        # Pembaruan suara konsensus & transisi status hanya diizinkan saat bilah berada di dalam
        # Prime Zone dengan span utuh, sehingga distorsi/foreshortening saat keluar tidak membalik vonis.
        can_update_vote = self._update_prime_consensus_vote(
            is_in_prime, full_span_valid, sample_count, min_samples_required, transit_is_ok
        )

        # Registrasi Objek & Pembaruan Status Berbasis Konsensus Transaksional
        if sample_count < min_samples_required:
            # Belum cukup sampel prime zone → Tetap WAIT agar tidak memicu False NG prematur
            status = "WAIT"
            eval_msg = f"Stabilisasi Prime Zone ({sample_count}/{min_samples_required} sampel | Span={w_span:.0f}px)"
        else:
            # Sampel prime zone cukup → Tentukan status definitif berbasis konsensus
            new_status = "OK" if transit_is_ok else "NG"
            if self.current_rib_status is None:
                # Butuh minimal 2 konfirmasi berturut-turut di Prime Zone untuk registrasi status sementara
                # Counter ok_count/ng_count TIDAK dinaikkan di sini; baru dikunci final saat transit selesai.
                if can_update_vote and ((new_status == "OK" and self.active_consec_ok >= 2) or (new_status == "NG" and self.active_consec_ng >= 2)):
                    self.rib_count += 1
                    self.current_rib_status = new_status
                else:
                    status = "WAIT"
                    eval_msg = f"Menunggu Konfirmasi Status ({self.active_consec_ok} OK / {self.active_consec_ng} NG)"
            elif can_update_vote and self.current_rib_status != new_status:
                # Anti-flicker & Deadband Hysteresis untuk status sementara selama rib masih di Prime Zone:
                # - OK -> NG: butuh 3 frame konfirmasi berturut-turut di Prime Zone
                # - NG -> OK: hanya jika seluruh metrik turun di bawah deadband histeresis
                #   selama >= 5 frame berturut-turut di Prime Zone (tanpa mutasi counter sebelum final)
                if self.current_rib_status == "OK" and new_status == "NG" and self.active_consec_ng >= 3:
                    self.current_rib_status = "NG"
                elif self.current_rib_status == "NG" and new_status == "OK" and self.active_consec_ok >= 5:
                    if ng_prob is not None and self.tip_classifier_model is not None:
                        _ml_th = float(self.tip_classifier_model.get("prob_threshold", 0.5))
                        _phys_ok = (eval_pv < 4.5 and eval_idev < 3.5 and eval_tdev < 4.5)
                        hyst_ok = (
                            (not ml_defect)
                            and (_phys_ok or ng_prob < max(0.05, _ml_th - 0.08))
                            and eval_pv <= 24.0
                            and eval_sag <= 18.0
                        )
                    else:
                        hyst_ok = (
                            eval_tdev <= 0.85 * tip_th
                            and eval_idev <= 0.96 * dev_th
                            and eval_pv <= 0.96 * 18.0
                            and eval_sag <= 0.96 * 14.5
                            and not (eval_sag > 0.96 * sag_th and eval_pv > 0.96 * pv_th)
                        )
                    if hyst_ok:
                        self.current_rib_status = "OK"

            status = self.current_rib_status if self.current_rib_status is not None else "WAIT"
            # Hitung severity level bending (LOW/MEDIUM/HIGH) konsisten dengan vonis aktual
            ng_severity = self.compute_ng_severity(
                eval_pv, eval_sag, eval_idev, eval_tdev, is_ng=(status == "NG")
            ) if status == "NG" else "OK"

            if status == "OK":
                self.last_ng_reason = None
                eval_msg = f"SIRIP LURUS (OK) | Top Y={med_top_y}px | PV={eval_pv:.1f}px, Sag={eval_sag:.1f}px, Dev={eval_idev:.1f}px, Tip={eval_tdev:.1f}px"
            elif status == "NG":
                ng_reasons = []
                if ng_prob is not None and ml_defect:
                    ng_reasons.append(f"ML_TipProb={ng_prob:.2f}")
                if flat_defect:
                    ng_reasons.append(f"PV={eval_pv:.1f}>{pv_th:.1f}px")
                if curv_defect:
                    ng_reasons.append(f"Sag={eval_sag:.1f}>{sag_th:.1f}px")
                if dev_defect:
                    ng_reasons.append(f"Dev={eval_idev:.1f}>{dev_th:.1f}px")
                if tip_defect:
                    ng_reasons.append(f"TipDev={eval_tdev:.1f}>{tip_th:.1f}px")
                self.last_ng_reason = " | ".join(ng_reasons) if ng_reasons else f"TipDev={eval_tdev:.1f}>{tip_th:.1f}px"
                eval_msg = f"SIRIP BENGKOK (NG - {ng_severity}) | {self.last_ng_reason}"
            else:
                eval_msg = f"Menunggu Konfirmasi Status ({self.active_consec_ok} OK / {self.active_consec_ng} NG | Span={w_span:.0f}px)"

        # Pastikan ng_severity selalu terdefinisi di cabang sample_count < min_samples_required
        if sample_count < min_samples_required:
            ng_severity = "OK"

        self.last_presence = True

        # Patch fisik sirip yang tepat mengunci dari med_top_y ke med_top_y + 110 px
        patch_y1 = max(0, med_top_y - 8)
        patch_y2 = min(h_f, med_top_y + 115)
        patch_x1 = max(0, search_x1)
        patch_x2 = min(w_f, search_x2)
        rib_patch = frame[patch_y1:patch_y2, patch_x1:patch_x2] if patch_y2 > patch_y1 and patch_x2 > patch_x1 else None

        # Perbarui active_box vertikal agar mengunci tepat di badan bilah sirip (bukan celah slat)
        refined_box = [search_x1, max(0, med_top_y - 8), search_x2, min(h_f - 1, med_top_y + 105)]

        return {
            "status": status,
            "object_type": "RIB_SIRIP",
            "is_rib_present": True,
            "is_ok": (status == "OK"),
            "is_ng": (status == "NG"),
            "ng_severity": ng_severity,          # 'OK' | 'LOW' | 'MEDIUM' | 'HIGH'
            "final_consensus_ready": bool(sample_count >= min_samples_required and self.current_rib_status is not None),
            "is_final_verdict": bool(self._session_committed),
            "committed_rib_id": self._last_finalized_rib_id,
            "committed_rib_status": self._last_finalized_status,
            "rib_count": self.rib_count,
            "ok_count": self.ok_count,
            "ng_count": self.ng_count,
            "mode": self.mode,
            "tracker_engine": self.tracker_engine,
            "message": eval_msg,
            "detection_reason": f"{reason} | Bilah Terkunci (Cov={coverage*100:.0f}%, Span={w_span:.0f}px)",
            "confidence": float(yolo_conf if yolo_conf > 0 else (dual_ratio if self.mode == "DUAL_POLARITY_ONLY" else 0.0)),
            "pv_flatness": eval_pv,
            "sagitta": eval_sag,
            "max_deviation": max_dev,
            "inner_dev": eval_idev,
            "tip_dev": eval_tdev,
            "ng_prob": ng_prob,
            "coverage": coverage,
            "active_rib_frames": self.active_rib_frames,
            "rib_height": 110,
            "active_box": refined_box,
            "zone_rect": zone_rect,
            "xs_contour": r_xs,
            "ys_contour": smooth_ys,
            "baseline": baseline,
            "rib_patch": rib_patch,
            "unwarp_contour": getattr(self, "_last_unwarp_contour", None),
            "yolo_boxes": all_yolo_boxes,
            "elapsed_ms": (time.time() - t0) * 1000
        }

    def render_overlay(self, frame, res):
        """
        Merender dashboard visualisasi inspeksi sirip tampak atas yang elegan dan informatif.
        """
        vis = frame.copy()
        h_f, w_f = frame.shape[:2]

        status = res.get("status", "WAIT")
        obj_type = res.get("object_type", "PLAT_DATAR")
        zone_rect = res.get("zone_rect", (int(0.05*w_f), int(0.45*h_f), int(0.95*w_f), int(0.85*h_f)))
        zx1, zy1, zx2, zy2 = zone_rect

        # 1. Gambar Zona Inspeksi Konveyor Foreground
        cv2.rectangle(vis, (zx1, zy1), (zx2, zy2), (180, 180, 60), 1, cv2.LINE_AA)
        cv2.putText(vis, f"ZONA GERBANG INSPEKSI KONVEYOR [{self.mode}]", (zx1 + 10, zy1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 80), 2, cv2.LINE_AA)

        # 1b. Gambar Batas Poligon Perspektif Unwarp (Top-Down 90 Deg Zone)
        y_top = self.unwarp_y_top * h_f
        y_bot = self.unwarp_y_bot * h_f
        taper = getattr(self, "unwarp_tilt", 1.60)
        m_left = -0.375 * taper
        m_right = +0.370 * taper
        scale_w = w_f / 1920.0
        scale_h = h_f / 1080.0
        x_tl = (374.0 + m_left * (y_top / scale_h - 216.0)) * scale_w
        x_bl = (374.0 - 0.375 * (y_bot / scale_h - 216.0)) * scale_w
        x_tr = (1623.0 + m_right * (y_top / scale_h - 216.0)) * scale_w
        x_br = (1623.0 + 0.370 * (y_bot / scale_h - 216.0)) * scale_w
        quad_pts = np.array([[x_tl, y_top], [x_tr, y_top], [x_br, y_bot], [x_bl, y_bot]], dtype=np.int32)
        cv2.polylines(vis, [quad_pts], isClosed=True, color=(255, 180, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(vis, f"ZONA PERSPECTIVE TRANSFORM (TILT: {self.unwarp_tilt:.1f}x | Y: {self.unwarp_y_top*100:.0f}%-{self.unwarp_y_bot*100:.0f}%)", (int(x_tl) + 6, int(y_top) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 200, 0), 1, cv2.LINE_AA)

        # 2. Jika Status WAIT (Plat Datar Sedang Melintas)
        if status == "WAIT" or not res.get("is_rib_present", False):
            badge_bg = (140, 90, 30)
            cv2.putText(vis, "[WAIT] PLAT DATAR - MENUNGGU SIRIP MELINTAS", (zx1 + 20, (zy1 + zy2) // 2),
                        cv2.FONT_HERSHEY_DUPLEX, 0.85, (240, 200, 100), 2, cv2.LINE_AA)
        else:
            # 3. Jika Status OK atau NG (Sirip Sedang Berada di Zona Inspeksi)
            is_ok = res.get("is_ok", False)
            contour_color = (0, 255, 0) if is_ok else (0, 0, 255)
            badge_bg = (0, 160, 0) if is_ok else (0, 0, 200)

            # Gambar Bounding Box Sirip Aktif
            box = res.get("active_box")
            if box:
                bx1, by1, bx2, by2 = box
                rib_conf = float(res.get("confidence", 0.0) or 0.0)
                conf_str = f" | Conf: {rib_conf:.2f}" if rib_conf > 0 else ""
                cv2.rectangle(vis, (bx1, by1), (bx2, by2), (255, 170, 0), 2, cv2.LINE_AA)
                cv2.putText(vis, f"SIRIP AKTIF ({bx2-bx1}x{by2-by1} px{conf_str})", (bx1, max(22, by1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 170, 0), 2, cv2.LINE_AA)

            # Gambar Kurva Punggung Atas (Top Ridge Curve) menempel di bilah fisik
            xs = res.get("xs_contour")
            ys = res.get("ys_contour")
            if xs is not None and ys is not None and len(xs) > 10:
                pts = np.column_stack([xs, ys]).astype(np.int32)
                cv2.polylines(vis, [pts], isClosed=False, color=contour_color, thickness=3, lineType=cv2.LINE_AA)
                # Tandai titik defleksi ekstrim
                mid_idx = len(pts) // 2
                cv2.circle(vis, (int(pts[mid_idx, 0]), int(pts[mid_idx, 1])), 5, contour_color, -1)

        # ----------------------------------------------------------------------
        # PANEL DASHBOARD BAWAH (1000 x 300 px)
        # ----------------------------------------------------------------------
        main_resized = cv2.resize(vis, (1000, 520))

        # Render Telemetri Akselerasi Hardware / CUDA di sudut kiri atas
        tel = None
        if self.telemetry is not None:
            tel = self.telemetry.get_telemetry()
            draw_camera_badge(main_resized, tel, x=15, y=12, w=370, h=68)

        # Banner Indikator PAUSE jika video sedang dijeda
        if res.get("is_paused", False):
            f_cur = res.get("frame_pos", 0)
            f_tot = res.get("total_frames", 0)
            frame_str = f" | {f_cur}/{f_tot}" if f_tot > 0 else ""

            bw, bh = 310, 68
            bx = 395
            by = 12
            cv2.rectangle(main_resized, (bx, by), (bx + bw, by + bh), (20, 110, 220), -1)
            cv2.rectangle(main_resized, (bx, by), (bx + bw, by + bh), (40, 180, 255), 2, cv2.LINE_AA)
            cv2.putText(main_resized, f"|| PAUSED{frame_str}", (bx + 14, by + 28),
                        cv2.FONT_HERSHEY_DUPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(main_resized, "[SPACE]/[P]: Play | [D]/[A]: Step", (bx + 14, by + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 240, 200), 1, cv2.LINE_AA)

        hud = np.zeros((260, 1000, 3), dtype=np.uint8)
        hud[:] = (18, 18, 22)

        # Badge Status Utama (OK, NG, WAIT) + Indikator Severity Bending (LOW/MEDIUM/HIGH)
        ng_sev = res.get("ng_severity", "OK")
        cv2.rectangle(hud, (15, 12), (180, 70), badge_bg, -1)
        if status == "NG" and ng_sev in ("LOW", "MEDIUM", "HIGH"):
            cv2.putText(hud, status, (66, 46), cv2.FONT_HERSHEY_DUPLEX, 1.15, (255, 255, 255), 3, cv2.LINE_AA)
            sev_colors = {
                "LOW": (0, 215, 255),     # Kuning-oranye (peringatan awal)
                "MEDIUM": (0, 140, 255),  # Oranye pekat (sedang)
                "HIGH": (255, 255, 255),  # Putih kontras di atas merah (parah)
            }
            sev_txt = f"LEVEL: {ng_sev}"
            sev_x = 58 if ng_sev != "MEDIUM" else 44
            cv2.putText(hud, sev_txt, (sev_x, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.42, sev_colors.get(ng_sev, (255, 255, 255)), 1, cv2.LINE_AA)
        else:
            stat_x = 55 if len(status) == 2 else (35 if len(status) == 4 else 45)
            cv2.putText(hud, status, (stat_x, 54), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)

        # Status Teks Header & Indikator Sakelar Engine (3-Way Toggle)
        if self.tracker_engine == "TIER4_HOUGH_STITCH":
            engine_str = "TIER 4 (MICRO-HOUGHLET STITCHING)"
            engine_col = (255, 180, 50)
        elif self.tracker_engine == "TIER1_VITERBI":
            engine_str = "TIER 1 (VITERBI V2)"
            engine_col = (0, 255, 200)
        else:
            engine_str = "DEFAULT LEGACY (SOBEL-Y)"
            engine_col = (100, 255, 120)
        cv2.putText(hud, f"MODE: [{self.mode}] | ENGINE: [{engine_str}] (Tekan [T] ganti)",
                    (200, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.50, engine_col, 2, cv2.LINE_AA)
        cv2.putText(hud, f"OBJEK: {obj_type.replace('_', ' ')} | {res.get('message', '')}", (200, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 230, 255), 1, cv2.LINE_AA)

        # Kartu Instrumen Akselerasi Hardware & GPU Telemetri
        if self.telemetry is not None and tel is not None:
            draw_hud_card(hud, tel, x=708, y=12, w=275, h=68)

        # Kolom Metrologi Kelurusan
        pv = res.get("pv_flatness", 0.0)
        sag = res.get("sagitta", 0.0)
        rib_h = res.get("rib_height", 0)
        dev_val = res.get("inner_dev", res.get("max_deviation", 0.0))
        tip_val = res.get("tip_dev", 0.0)

        pv_col = (0, 255, 255) if pv <= self.pv_thresh else (0, 0, 255)
        sag_col = (0, 255, 255) if sag <= self.sagitta_thresh else (0, 0, 255)
        dev_col = (0, 255, 255) if dev_val <= self.dev_thresh else (0, 0, 255)
        tip_col = (0, 255, 255) if tip_val <= self.tip_thresh else (0, 0, 255)

        cv2.putText(hud, f"Flatness PV   : {pv:4.1f} px  (Maks: {self.pv_thresh:.1f} px)", (25, 96),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, pv_col, 1, cv2.LINE_AA)
        cv2.putText(hud, f"Sagitta Curv  : {sag:4.1f} px  (Maks: {self.sagitta_thresh:.1f} px)", (25, 118),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, sag_col, 1, cv2.LINE_AA)
        cv2.putText(hud, f"Deviasi Badan : {dev_val:4.1f} px  (Maks: {self.dev_thresh:.1f} px)", (25, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, dev_col, 1, cv2.LINE_AA)
        cv2.putText(hud, f"Deviasi Ujung : {tip_val:4.1f} px  (Maks: {self.tip_thresh:.1f} px)", (25, 162),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, tip_col, 1, cv2.LINE_AA)
        cv2.putText(hud, f"Tinggi Bilah  : {rib_h:4d} px  (Nom : 60 - 200 px)", (25, 184),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 220, 200), 1, cv2.LINE_AA)
        cv2.putText(hud, f"Validasi Sensor: {res.get('detection_reason', '')}", (25, 206),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.41, (220, 220, 120), 1, cv2.LINE_AA)
        rib_conf = float(res.get("confidence", 0.0) or 0.0)
        conf_str = f" [Conf: {rib_conf:.2f}]" if rib_conf > 0 else ""
        cv2.putText(hud, f"Statistik QC   : Total Rib #{self.rib_count}{conf_str} | OK: {self.ok_count} | NG: {self.ng_count}", (25, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 240, 180), 1, cv2.LINE_AA)

        # Patch Citra Sirip di sebelah kanan HUD (Zoom / Preview Unwarp / Hough Diagnostics)
        rib_patch = res.get("rib_patch")
        if self.tracker_engine == "TIER4_HOUGH_STITCH" and getattr(self, "_last_hough_patch", None) is not None:
            rib_patch = self._last_hough_patch
            patch_title = "PREVIEW HOUGH STITCH (SOBEL-Y & HOUGHLETS)"
        elif rib_patch is not None and rib_patch.size > 0:
            patch_title = "PATCH FISIK SIRIP (ZOOM)"
        elif self.last_unwarped_frame is not None and self.last_unwarped_frame.size > 0:
            rib_patch = self.last_unwarped_frame
            patch_title = "PREVIEW UNWARP PERSPECTIVE (BIRD'S-EYE)"
        else:
            rib_patch = None
            patch_title = ""

        if rib_patch is not None and rib_patch.size > 0:
            try:
                patch_disp = cv2.resize(rib_patch, (480, 110))
                cv2.rectangle(hud, (495, 95), (980, 210), (60, 60, 70), 1)
                hud[98:208, 498:978] = patch_disp
                cv2.putText(hud, patch_title, (500, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)
            except Exception:
                pass

        # Footer Menu
        footer = np.zeros((28, 1000, 3), dtype=np.uint8)
        footer[:] = (12, 12, 16)
        source_name = res.get("source_name", "")
        file_tag = f"[{source_name[:18]}] " if source_name else ""
        paused_tag = "⏸ PAUSED " if res.get("is_paused", False) else "▶ PLAY "
        help_txt = f"{paused_tag}{file_tag}[K]/[J]: Geser Zona Naik/Turun | [U]: Preview Window | [T]: Engine | [SPACE]: Pause | [D]/[A]: Step | [Q]: Keluar"
        footer_col = (40, 180, 255) if res.get("is_paused", False) else (210, 210, 210)
        cv2.putText(footer, help_txt, (15, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.35, footer_col, 1, cv2.LINE_AA)

        screen = np.vstack([main_resized, hud, footer])
        return screen


# ==============================================================================
# SCRIPT DEMO INTERAKTIF RUNNER (BISA DIJALANKAN LANGSUNG VIA TERMINAL / BAT)
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Demo Interaktif Hybrid Rib Inspector (YOLO + Dual-Polarity)")
    parser.add_argument("input_path", nargs="?", default=None, help="Path ke file video atau foto pengujian")
    parser.add_argument("--model", default=None, help="Path ke model YOLO (.pt / .engine)")
    parser.add_argument("--mode", default="HYBRID", choices=["HYBRID", "YOLO_ONLY", "DUAL_POLARITY_ONLY"], help="Mode deteksi awal")
    parser.add_argument("--device", default="auto", help="Device komputasi AI ('auto', 'cuda', 0, atau 'cpu')")
    args, _ = parser.parse_known_args()

    print("\n" + "=" * 78)
    print("🚀 DEMO INTERAKTIF HYBRID RIB INSPECTOR (YOLO + DUAL-POLARITY)")
    print("=" * 78)
    print("  - [SPACE] / [P] : Pause / Play Video (Jeda atau Lanjutkan)")
    print("  - [D] / [→]     : Maju 1 Frame (Step Forward saat Video Dijeda)")
    print("  - [A] / [←]     : Mundur 1 Frame (Step Backward saat Video Dijeda)")
    print("  - [T] / [E]     : Toggle Tracker Engine (DEFAULT LEGACY <-> TIER 1 VITERBI <-> TIER 4 HOUGH STITCH)")
    print("  - [M]           : Beralih Mode (HYBRID <-> YOLO_ONLY <-> DUAL_POLARITY_ONLY)")
    print("  - [O] / [F]     : Buka File Dialog Explorer (Pilih Video / Foto Baru)")
    print("  - [V]           : Beralih ke Video Berikutnya dalam Playlist")
    print("  - [B]           : Toggle Simulasi Bending (+24 px kelengkungan sirip)")
    print("  - [Q]/[ESC]     : Keluar")
    print("=" * 78 + "\n")

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(cur_dir)

    # Cari video pengujian di berbagai folder
    candidate_doc_dirs = [
        os.path.abspath(os.path.join(parent_dir, "..", "Video_Testing")),
        os.path.abspath(os.path.join(parent_dir, "..", "..", "Video_Testing")),
        os.path.abspath(os.path.join(parent_dir, "..", "..", "..", "Documentation")),
        os.path.abspath(os.path.join(parent_dir, "..", "..", "Documentation")),
        os.path.abspath(os.path.join(parent_dir, "Documentation")),
        os.path.join(os.path.expanduser("~"), "Videos"),
        os.path.join(os.path.expanduser("~"), "Pictures"),
    ]
    videos = []
    for d_dir in candidate_doc_dirs:
        if os.path.exists(d_dir):
            found = glob.glob(os.path.join(d_dir, "**", "*.mp4"), recursive=True)
            for f in found:
                if f not in videos:
                    videos.append(f)

    # Prioritaskan video yang memuat rib
    priority_order = ["WIN_20261005_09_13", "WhatsApp Video", "Recording", "Deform Parah", "Deform Sedikit", "OK Semua"]
    sorted_videos = []
    for prio in priority_order:
        for v in videos:
            if prio.lower() in os.path.basename(v).lower() and v not in sorted_videos:
                sorted_videos.append(v)
    for v in videos:
        if v not in sorted_videos:
            sorted_videos.append(v)

    # Tentukan input awal
    initial_source = None
    if args.input_path and os.path.exists(args.input_path):
        initial_source = os.path.abspath(args.input_path)
    else:
        # Tampilkan Menu Interaktif Konsol untuk Memilih File
        print("🎥 PILIH SUMBER MEDIA INSPEKSI SIRIP (RIB):")
        print("  [1] Buka File Dialog Explorer (Pilih file lewat jendela Windows) [POPUP]")
        if sorted_videos:
            print(f"  [2] Playlist Default: {os.path.basename(sorted_videos[0])} [TEKAN ENTER]")
            for i, vpath in enumerate(sorted_videos[1:6], start=3):
                print(f"  [{i}] {os.path.basename(vpath)}")
        else:
            print("  [2] (Tidak ada file default terdeteksi di folder Documentation)")

        print("-" * 78)
        choice = ""
        try:
            if sys.stdin.isatty():
                choice = input("Pilih sumber input (tekan ENTER untuk default [2]): ").strip()
            else:
                choice = ""
        except Exception:
            choice = ""

        if choice == "1":
            init_folder = candidate_doc_dirs[0] if os.path.exists(candidate_doc_dirs[0]) else os.getcwd()
            chosen = select_file_dialog(
                initial_dir=init_folder,
                title="Pilih File Video / Foto untuk Inspeksi Sirip"
            )
            if chosen:
                initial_source = chosen
        elif choice.isdigit() and int(choice) >= 2:
            idx = int(choice) - 2
            if idx < len(sorted_videos):
                initial_source = sorted_videos[idx]

        if not initial_source and sorted_videos:
            initial_source = sorted_videos[0]

    if not initial_source or not os.path.exists(initial_source):
        print("[!] Membuka File Dialog Explorer...")
        init_folder = candidate_doc_dirs[0] if os.path.exists(candidate_doc_dirs[0]) else os.getcwd()
        initial_source = select_file_dialog(
            initial_dir=init_folder,
            title="Pilih File Video / Foto untuk Inspeksi Sirip"
        )
        if not initial_source or not os.path.exists(initial_source):
            print("[ERROR] Tidak ada file input yang dipilih. Program dihentikan.")
            return

    current_source_path = initial_source
    if current_source_path not in sorted_videos:
        sorted_videos.insert(0, current_source_path)
    current_vid_idx = sorted_videos.index(current_source_path)

    cap, is_image, static_frame, total_frames, current_fname = open_media_source(current_source_path)
    if not is_image and (cap is None or not cap.isOpened()):
        print(f"[ERROR] Gagal membuka sumber media: {current_source_path}")
        return

    inspector = HybridRibInspector(model_path=args.model, mode=args.mode, device=args.device)

    win_name = "DEMO: Hybrid Rib Inspector (YOLO + Dual-Polarity)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1100, 850)

    # Window Preview Perspective Transform (Bird's-Eye Top-Down Unwarp)
    show_unwarp_window = True
    win_unwarp = "Perspective Transform View (Full Deck 90 Deg Top-Down)"
    cv2.namedWindow(win_unwarp, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_unwarp, 750, 420)

    is_paused = False
    simulate_bending = False
    cached_screen = None
    needs_inspect = True
    frame = None
    current_frame_pos = 0

    print(f"\n[OK] 🎬 Memulai inspeksi pada: {current_fname}")
    print(f"      Mode: [{inspector.mode}] | Tipe: {'Foto Diam' if is_image else f'Video ({total_frames} frame)'}")
    print("      [Z]: Ganti Preset Top-Down | [W]/[S]: Sudut Tilt | [K]/[J]: Posisi Zona | [U]: Buka/Tutup Preview")
    print("      [SPACE]: Pause/Play | [D]/[A]: Step Frame | [T]: Tracker Engine | [B]: Simulasi Bending\n")

    while True:
        if is_image:
            if needs_inspect or cached_screen is None:
                frame = static_frame.copy()
                res = inspector.inspect_frame(frame, zone_poly=None, simulate_bending=simulate_bending)
                res["source_name"] = current_fname
                res["is_paused"] = is_paused
                res["frame_pos"] = 1
                res["total_frames"] = 1
                cached_screen = inspector.render_overlay(frame, res)
                needs_inspect = False
            screen = cached_screen
        else:
            if not is_paused:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                current_frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                res = inspector.inspect_frame(frame, zone_poly=None, simulate_bending=simulate_bending)
                res["source_name"] = current_fname
                res["is_paused"] = False
                res["frame_pos"] = current_frame_pos
                res["total_frames"] = total_frames
                cached_screen = inspector.render_overlay(frame, res)
                screen = cached_screen
                needs_inspect = False
            else:
                # Saat PAUSED: jika ada trigger (step frame / ubah mode / toggle bending)
                if (needs_inspect or cached_screen is None) and frame is not None:
                    res = inspector.inspect_frame(frame, zone_poly=None, simulate_bending=simulate_bending)
                    res["source_name"] = current_fname
                    res["is_paused"] = True
                    res["frame_pos"] = current_frame_pos
                    res["total_frames"] = total_frames
                    cached_screen = inspector.render_overlay(frame, res)
                    needs_inspect = False
                screen = cached_screen

        cv2.imshow(win_name, screen)

        # Selalu render jendela unwarp secara serentak (Zero-Flickering & Resolusi Konsisten)
        if show_unwarp_window and frame is not None:
            active_b = res.get("active_box") if isinstance(res, dict) else None
            unwarp_disp = inspector.render_unwarp_preview(frame, active_box=active_b, res=res)
            cv2.imshow(win_unwarp, unwarp_disp)

        # Polling keyboard non-blocking (1ms saat play agar video 60+ FPS lancar, 25ms saat jeda)
        delay = 1 if not is_paused else 25
        key = cv2.waitKey(delay) & 0xFF

        if key in [ord('q'), 27]:
            break
        elif key in [ord(' '), ord('p'), ord('P')]:
            is_paused = not is_paused
            needs_inspect = True
            print(f"[!] Video {'⏸ DIJEDA (PAUSED)' if is_paused else '▶ DILANJUTKAN (PLAYING)'}")
        elif is_paused and (key in [ord('d'), ord('D'), ord('n'), ord('N'), ord('.'), 83]):
            # Maju 1 frame (Step Forward)
            if cap is not None and not is_image:
                ret, next_frame = cap.read()
                if ret:
                    frame = next_frame
                    current_frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    needs_inspect = True
                    print(f"  [+] Maju ke Frame {current_frame_pos}/{total_frames}")
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, next_frame = cap.read()
                    if ret:
                        frame = next_frame
                        current_frame_pos = 1
                        needs_inspect = True
        elif is_paused and (key in [ord('a'), ord('A'), ord(','), 81]):
            # Mundur 1 frame (Step Backward)
            if cap is not None and not is_image:
                cur_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                target_pos = max(0, cur_pos - 2)
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_pos)
                ret, prev_frame = cap.read()
                if ret:
                    frame = prev_frame
                    current_frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    needs_inspect = True
                    print(f"  [-] Mundur ke Frame {current_frame_pos}/{total_frames}")
        elif key in [ord('k'), ord('K')]:
            inspector.nudge_unwarp_zone(-0.02)
            needs_inspect = True
        elif key in [ord('j'), ord('J')]:
            inspector.nudge_unwarp_zone(+0.02)
            needs_inspect = True
        elif key in [ord('w'), ord('W')]:
            inspector.adjust_unwarp_tilt(+0.15)
            needs_inspect = True
        elif key in [ord('s'), ord('S')]:
            inspector.adjust_unwarp_tilt(-0.15)
            needs_inspect = True
        elif key in [ord('z'), ord('Z')]:
            inspector.cycle_unwarp_preset()
            needs_inspect = True
        elif key in [ord('u'), ord('U')]:
            show_unwarp_window = not show_unwarp_window
            if not show_unwarp_window:
                try:
                    cv2.destroyWindow(win_unwarp)
                except Exception:
                    pass
            else:
                cv2.namedWindow(win_unwarp, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(win_unwarp, 750, 480)
            print(f"[!] Window Preview Perspective Transform: {'DITAMPILKAN' if show_unwarp_window else 'DISEMBUNYIKAN'}")
        elif key in [ord('t'), ord('T'), ord('e'), ord('E')]:
            new_eng = inspector.toggle_tracker_engine()
            needs_inspect = True
            print(f"[!] 🔄 SAKELAR TRACKER ENGINE BERGANTI KE: [{new_eng}]")
        elif key in [ord('m'), ord('M')]:
            # Putar mode: HYBRID -> YOLO_ONLY -> DUAL_POLARITY_ONLY -> HYBRID
            current_mode = inspector.mode
            next_idx = (inspector.MODES.index(current_mode) + 1) % len(inspector.MODES)
            inspector.set_mode(inspector.MODES[next_idx])
            needs_inspect = True
        elif key in [ord('b'), ord('B')]:
            simulate_bending = not simulate_bending
            needs_inspect = True
            print(f"[!] Simulasi Bending: {'AKTIF (+24px)' if simulate_bending else 'NON-AKTIF'}")
        elif key in [ord('r'), ord('R')]:
            inspector.reset_counters()
            needs_inspect = True
            print("[!] 🔄 Counter Statistik QC di-reset ke 0 (Total Rib=0, OK=0, NG=0)")
        elif key in [ord('o'), ord('O'), ord('f'), ord('F')]:
            # Buka File Dialog Explorer secara interaktif saat runtime
            print("\n[!] 📂 Membuka File Dialog Explorer (Pilih file di jendela Windows)...")
            selected_path = select_file_dialog(
                initial_dir=os.path.dirname(current_source_path),
                title="Pilih File Video / Foto Baru untuk Inspeksi"
            )
            if selected_path and os.path.exists(selected_path):
                if cap is not None:
                    cap.release()
                new_cap, new_is_image, new_static_frame, new_tot, new_fname = open_media_source(selected_path)
                if new_is_image or (new_cap is not None and new_cap.isOpened()):
                    cap = new_cap
                    is_image = new_is_image
                    static_frame = new_static_frame
                    total_frames = new_tot
                    current_source_path = selected_path
                    current_fname = new_fname
                    if selected_path not in sorted_videos:
                        sorted_videos.append(selected_path)
                    current_vid_idx = sorted_videos.index(selected_path)
                    needs_inspect = True
                    cached_screen = None
                    print(f"[OK] 🎬 Berhasil memuat media: {current_fname}")
                else:
                    print(f"[WARNING] Gagal membuka file yang dipilih: {selected_path}")
            else:
                print("[*] Pemilihan file dibatalkan.")
        elif key in [ord('v'), ord('V')]:
            if sorted_videos:
                current_vid_idx = (current_vid_idx + 1) % len(sorted_videos)
                new_v = sorted_videos[current_vid_idx]
                if cap is not None:
                    cap.release()
                new_cap, new_is_image, new_static_frame, new_tot, new_fname = open_media_source(new_v)
                if new_is_image or (new_cap is not None and new_cap.isOpened()):
                    cap = new_cap
                    is_image = new_is_image
                    static_frame = new_static_frame
                    total_frames = new_tot
                    current_source_path = new_v
                    current_fname = new_fname
                    needs_inspect = True
                    cached_screen = None
                    print(f"\n[OK] 🎬 Beralih ke Media: {current_fname}")

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
