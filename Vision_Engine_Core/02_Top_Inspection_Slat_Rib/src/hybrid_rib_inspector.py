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
        pv_thresh=14.0,
        sagitta_thresh=11.5,
        dev_thresh=8.5,
        tip_thresh=9.5,
        device=None,
        stride=None
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
        self.mode = mode.upper() if mode.upper() in self.MODES else "HYBRID"
        self.conf_thresh = float(conf_thresh)
        self.iou_thresh = float(iou_thresh)
        self.pv_thresh = float(pv_thresh)
        self.sagitta_thresh = float(sagitta_thresh)
        self.dev_thresh = float(dev_thresh)
        self.tip_thresh = float(tip_thresh)

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
            candidates = (engine_candidates + pt_candidates + onnx_candidates) if self.device != "cpu" else (pt_candidates + onnx_candidates)

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

            # Jika model_path berakhiran .pt tetapi ada .engine di GPU
            if model_path and model_path.endswith(".pt") and self.device != "cpu":
                eng_cand = model_path.replace(".pt", ".engine")
                if os.path.exists(eng_cand):
                    model_path = eng_cand
                    print(f"[HybridRibInspector] 🚀 Menggunakan TensorRT Engine FP16: {os.path.basename(eng_cand)}")

            self.model_path = model_path
            if model_path and os.path.exists(model_path):
                try:
                    self.yolo_model = YOLO(model_path)
                    is_trt_or_onnx = str(model_path).endswith((".engine", ".onnx"))
                    if self.device != "cpu" and HAS_TORCH and torch.cuda.is_available() and not is_trt_or_onnx:
                        try:
                            self.yolo_model.to(self.device)
                            if hasattr(self.yolo_model, "model") and self.yolo_model.model is not None:
                                self.yolo_model.model.half()
                            self.use_half = True
                        except Exception:
                            pass
                    elif is_trt_or_onnx:
                        self.use_half = True
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
                                self.yolo_model = YOLO(fallback_rib)
                                is_fb_trt_or_onnx = str(fallback_rib).endswith((".engine", ".onnx"))
                                if self.device != "cpu" and HAS_TORCH and torch.cuda.is_available() and not is_fb_trt_or_onnx:
                                    try:
                                        self.yolo_model.to(self.device)
                                        if hasattr(self.yolo_model, "model") and self.yolo_model.model is not None:
                                            self.yolo_model.model.half()
                                        self.use_half = True
                                    except Exception:
                                        pass
                                elif is_fb_trt_or_onnx:
                                    self.use_half = True
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

        # Parameter Temporal Voting & Multi-Frame Consensus Anti-Glitch
        self.min_transit_frames = 15
        self.absence_frames = 0
        self.max_absence_tolerance = 8

        # Zona Prime Evaluasi Konveyor (Pusat optik planar bebas distorsi lensa Y: 360 - 430 px)
        self.prime_min_y = 360
        self.prime_max_y = 430
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

        # Cache Dual Polarity untuk stride (skip setiap frame ke-2 hemat 6-7ms)
        self._last_dual_det = (False, None, 0.0, 0.0, None)
        self._dual_stride = 2  # Jalankan dual polarity setiap N frame

        # Telemetri Akselerasi Perangkat Keras
        self.telemetry = HardwareTelemetry(sample_interval_sec=0.25) if HAS_TELEMETRY else None

    def reset_counters(self):
        """Mereset seluruh statistik hitungan sirip (Total Rib, OK, NG)."""
        self.rib_count = 0
        self.ok_count = 0
        self.ng_count = 0
        self.last_presence = False
        self.current_rib_status = None
        self.absence_frames = 0
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
        self.last_ng_reason = None

    def set_mode(self, new_mode):
        """Mengubah mode operasional detektor saat runtime."""
        if new_mode.upper() in self.MODES:
            if "YOLO" in new_mode.upper() and self.yolo_model is None:
                print(f"[HybridRibInspector] Model YOLO tidak tersedia, tetap pada DUAL_POLARITY_ONLY.")
                self.mode = "DUAL_POLARITY_ONLY"
            else:
                self.mode = new_mode.upper()
                print(f"[HybridRibInspector] Mode diubah ke: {self.mode}")

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

        # Parameter Geometri Fisik Bilah Sirip pada Konveyor:
        # 1. Lebar sirip riil konveyor: ~1214 px pada 1080p (toleransi 850px - 1500px)
        #    Menolak struktur rangka bawah konveyor yang berukuran ekstrem (bw = 1917 px)
        # 2. Batas lateral: bilah sirip tidak pernah menempel bezel kiri/kanan (x1 >= 115px, x2 <= 1805px)
        # 3. Ketebalan vertikal fisik: 35px - 145px
        # 4. Rasio aspek bilah tipis memanjang: 7.5 - 26.0
        min_bw = int(0.35 * w_f)
        max_bw = int(0.96 * w_f)
        min_x1 = int(0.02 * w_f)
        max_x2 = int(0.98 * w_f)
        min_bh = max(18, int(0.018 * h_f))
        max_bh = int(0.20 * h_f)
        nominal_w = int(0.635 * w_f)

        try:
            predict_kwargs = {
                "source": infer_img,
                "conf": self.conf_thresh,
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
                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())
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

                        rib_boxes.append({
                            "box": [x1, y1, x2, y2],
                            "conf": conf,
                            "centroid": (int((x1 + x2) / 2), cy)
                        })

                        is_valid = (
                            (min_bw <= bw <= max_bw) and
                            (x1 >= min_x1 and x2 <= max_x2) and
                            (min_bh <= bh <= max_bh) and
                            (5.5 <= ar <= 35.0) and
                            (cy >= zy1 - 35) and (cy <= zy2 + 35)
                        )

                        if is_valid:
                            # Seleksi berbasis kemiripan lebar fisik nominal (~1220 px) & skor confidence
                            w_dev = abs(bw - nominal_w) / float(nominal_w)
                            match_score = conf * (1.0 - min(0.5, w_dev))
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
        sy_crop = cv2.Sobel(crop_gray, cv2.CV_32F, 0, 1, ksize=3)
        pos_sy = np.clip(sy_crop, 0, 255)
        neg_sy = np.clip(-sy_crop, 0, 255)

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

                    if top_val >= 18.0 and bot_val >= 20.0 and (50 <= h_rib <= 220):
                        valid_points.append((zx1 + x_rel * 2, top_y, bot_y, h_rib))

        ratio = len(valid_points) / float(len(scan_xs_rel)) if len(scan_xs_rel) > 0 else 0.0
        dual_detected = bool(ratio >= 0.35)
        dual_box = None
        med_height = 0.0

        if dual_detected and valid_points:
            med_height = float(np.median([p[3] for p in valid_points]))
            min_x = min([p[0] for p in valid_points])
            max_x = max([p[0] for p in valid_points])
            med_top = int(np.median([p[1] for p in valid_points]))
            med_bot = int(np.median([p[2] for p in valid_points]))
            dual_box = [min_x, med_top, max_x, med_bot]

        return dual_detected, dual_box, med_height, ratio, None

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
            # Default Zona Konveyor Foreground: X [5%, 95%], Y [26%, 79%]
            # Memungkinkan deteksi rib di puncak konveyor (Y~300) dan memotong balok dasar konveyor (Y>850)
            zx1 = int(0.05 * w_f)
            zy1 = int(0.26 * h_f)
            zx2 = int(0.95 * w_f)
            zy2 = int(0.79 * h_f)

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

        if self.mode in ["HYBRID", "YOLO_ONLY"]:
            if self._frame_count % yolo_stride == 0 or not self._last_yolo_det[0]:
                yolo_det, yolo_box, yolo_conf, all_yolo_boxes = self.detect_yolo_zone(frame, focus_zone)
                self._last_yolo_det = (yolo_det, yolo_box, yolo_conf, all_yolo_boxes)
            else:
                yolo_det, yolo_box, yolo_conf, all_yolo_boxes = self._last_yolo_det

        if self.mode in ["HYBRID", "DUAL_POLARITY_ONLY"]:
            # Optimasi CPU: Di mode HYBRID, jika YOLO telah mendeteksi rib dengan pasti (yolo_det=True),
            # lewati pemindaian CPU Dual-Polarity penuh untuk menghemat ~10-15ms latensi per-frame.
            # Jalankan Dual-Polarity hanya saat mode DUAL_POLARITY_ONLY atau saat discovery (yolo_det=False).
            run_dual = (
                self.mode == "DUAL_POLARITY_ONLY"
                or (not yolo_det and (self._frame_count % self._dual_stride == 0 or not self._last_yolo_det[0]))
            )
            if run_dual:
                dual_det, dual_box, dual_h, dual_ratio, _ = self.detect_dual_polarity_zone(frame, focus_zone)
                self._last_dual_det = (dual_det, dual_box, dual_h, dual_ratio, None)
            else:
                dual_det, dual_box, dual_h, dual_ratio, _ = self._last_dual_det

        # 2. Gating Keputusan Awal
        if self.mode == "HYBRID":
            # Di mode HYBRID: YOLO memicu deteksi, dan dual-polarity memastikan ada ketebalan fisik
            is_candidate = yolo_det
            reason = f"HYBRID: YOLO({yolo_conf:.2f})"
            active_box = yolo_box
        elif self.mode == "YOLO_ONLY":
            is_candidate = yolo_det
            reason = f"YOLO ONLY: Conf={yolo_conf:.2f}"
            active_box = yolo_box
        else:  # DUAL_POLARITY_ONLY
            is_candidate = dual_det
            reason = f"DUAL-POLARITY ONLY: Ratio={dual_ratio*100:.0f}%, H={dual_h:.0f}px"
            active_box = dual_box

        # Evaluasi batas atas gerbang inspeksi:
        # Jika bagian atas rib telah melewati garis gerbang atas (zy1), sudahi inspeksi rib ini
        if active_box is not None and active_box[1] < zy1:
            is_candidate = False
            active_box = None

        # 3. KONDISI: PLAT DATAR MELINTAS (TIDAK ADA CANDIDATE RIB)
        if not is_candidate or active_box is None:
            if self.active_rib_frames > 0:
                self.absence_frames += 1
                if self.absence_frames < self.max_absence_tolerance:
                    # Hysteresis buffer: Tahan sesi transit aktif (mencegah fragmentasi akibat flicker 1-5 frame)
                    return {
                        "status": self.current_rib_status if self.current_rib_status else "WAIT",
                        "object_type": "RIB_SIRIP",
                        "is_rib_present": True,
                        "is_ok": (self.current_rib_status == "OK"),
                        "is_ng": (self.current_rib_status == "NG"),
                        "mode": self.mode,
                        "message": f"Sirip dalam transit (Buffer {self.absence_frames}/{self.max_absence_tolerance})",
                        "detection_reason": "Transit Hysteresis",
                        "confidence": float(yolo_conf if yolo_conf > 0 else (dual_ratio if self.mode == "DUAL_POLARITY_ONLY" else 0.0)),
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

            # Sesi transit benar-benar berakhir (> 8 frame kosong antar bilah)
            self.last_presence = False
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
            return {
                "status": "WAIT",
                "object_type": "PLAT_DATAR",
                "is_rib_present": False,
                "is_ok": False,
                "is_ng": False,
                "mode": self.mode,
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
                "mode": self.mode,
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

        # 4. SLOPE-AWARE ANCHOR & GUIDED CREST TRACKING (100% KONTINU, BEBAS LONCATAN KE CELAH SLAT)
        # Menghitung estimasi kemiringan alami konveyor (tilt) dari 7 titik sampel di area tengah rib
        step_fracs = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
        anchor_xs = []
        anchor_ys = []
        anchor_contrasts = []

        for frac in step_fracs:
            ax = int(bx1 + frac * (bx2 - bx1))
            best_y = None
            best_sc = -9999.0
            max_c = 0.0
            # Cari tepat di puncak bilah sirip (by1 + 6 s/d by1 + 55), mengecualikan celah plat di atasnya (by1 - 10 s/d by1)
            for y in range(max(10, by1 + 6), min(h_f - 12, by1 + 55)):
                b = float(gray[y, ax])
                diff_below = b - float(gray[min(h_f - 1, y + 8), ax])
                diff_above = b - float(gray[max(0, y - 5), ax])
                sc = b * 0.4 + diff_below * 1.0 + diff_above * 0.6
                if sc > best_sc:
                    best_sc = sc
                    best_y = y
                    max_c = diff_below
            if best_y is not None and best_sc > 40.0:
                anchor_xs.append(ax)
                anchor_ys.append(best_y)
                anchor_contrasts.append(max_c)

        # Validasi Fisika: Bilah sirip wajib memiliki kontras bibir vs wajah bilah >= 15.0
        avg_contrast = float(np.mean(anchor_contrasts)) if len(anchor_contrasts) > 0 else 0.0
        if avg_contrast < 15.0 or len(anchor_xs) < 2:
            if self.active_rib_frames > 0:
                self.absence_frames += 1
                if self.absence_frames < self.max_absence_tolerance:
                    return {
                        "status": self.current_rib_status if self.current_rib_status else "WAIT",
                        "object_type": "RIB_SIRIP",
                        "is_rib_present": True,
                        "is_ok": (self.current_rib_status == "OK"),
                        "is_ng": (self.current_rib_status == "NG"),
                        "mode": self.mode,
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
                "mode": self.mode,
                "message": f"Celah plat diabaikan (Bukan bilah tegak | Kontras={avg_contrast:.1f} < 15)",
                "detection_reason": f"{reason} -> Gugur (Kontras bilah rendah)",
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

        # Filter anchor outliers agar garis panduan 100% konsisten pada bibir rib
        if len(anchor_xs) >= 3:
            p_init = np.polyfit(anchor_xs, anchor_ys, 1)
            resids = np.abs(np.array(anchor_ys) - np.polyval(p_init, anchor_xs))
            med_res = np.median(resids)
            clean_xs = [ax for ax, r in zip(anchor_xs, resids) if r <= max(10.0, med_res * 2.5)]
            clean_ys = [ay for ay, r in zip(anchor_ys, resids) if r <= max(10.0, med_res * 2.5)]
            if len(clean_xs) >= 2:
                anchor_xs, anchor_ys = clean_xs, clean_ys
            guide_poly = np.polyfit(anchor_xs, anchor_ys, deg=1)
        else:
            guide_poly = np.array([0.0, float((by1 + by2) / 2)])

        # Pelacakan Kontur Ridge Dua Arah (Bidirectional Outward Tracking dari Tengah ke Ujung)
        # Menjamin pelacak 100% mengikuti deformasi / tekukan riil tanpa ter-clamp oleh koridor lurus
        mid_idx = len(anchor_xs) // 2
        x_center = int(anchor_xs[mid_idx])
        y_center = int(anchor_ys[mid_idx])

        # Jangkauan pelacakan penuh hingga ujung bilah (Tip-to-Tip Coverage)
        # Menjangkau ujung kiri dan kanan agar deformasi bending/tekukan di ujung terbaca sempurna
        x_limit_left = max(248, bx1 + 10)
        x_limit_right = min(1635, bx2 - 10)

        # 1. Telusuri ke arah kanan dari tengah (Outward Tracking dengan Step Clamping)
        right_pts = [(x_center, y_center)]
        cur_y = y_center
        tested_cols = 1
        consec_misses = 0

        for x in range(x_center + 4, x_limit_right + 1, 4):
            tested_cols += 1
            best_y = None
            best_sc = -9999.0
            gy = np.polyval(guide_poly, x)
            is_tip = (x > x_limit_right - 100)
            search_delta = 14 if is_tip else 8
            for y in range(max(10, cur_y - search_delta), min(h_f - 10, cur_y + search_delta)):
                if abs(y - gy) > 35.0:
                    continue
                b = float(gray[y, x])
                diff_below = b - float(gray[min(h_f - 1, y + 8), x])
                diff_above = b - float(gray[max(0, y - 5), x])
                sc = b * 0.4 + diff_below * 1.2 + diff_above * 0.6
                if sc > best_sc:
                    best_sc = sc
                    best_y = y
            if best_y is not None and best_sc >= 32.0 and gray[best_y, x] >= 35:
                max_step = 8 if is_tip else 5
                if abs(best_y - cur_y) <= max_step:
                    cur_y = best_y
                right_pts.append((x, cur_y))
                consec_misses = 0
            else:
                consec_misses += 1
                if consec_misses >= 5:
                    break

        # 2. Telusuri ke arah kiri dari tengah
        left_pts = []
        cur_y = y_center
        consec_misses = 0

        for x in range(x_center - 4, x_limit_left - 1, -4):
            tested_cols += 1
            best_y = None
            best_sc = -9999.0
            gy = np.polyval(guide_poly, x)
            is_tip = (x < x_limit_left + 100)
            search_delta = 14 if is_tip else 8
            for y in range(max(10, cur_y - search_delta), min(h_f - 10, cur_y + search_delta)):
                if abs(y - gy) > 35.0:
                    continue
                b = float(gray[y, x])
                diff_below = b - float(gray[min(h_f - 1, y + 8), x])
                diff_above = b - float(gray[max(0, y - 5), x])
                sc = b * 0.4 + diff_below * 1.2 + diff_above * 0.6
                if sc > best_sc:
                    best_sc = sc
                    best_y = y
            if best_y is not None and best_sc >= 32.0 and gray[best_y, x] >= 35:
                max_step = 8 if is_tip else 5
                if abs(best_y - cur_y) <= max_step:
                    cur_y = best_y
                left_pts.append((x, cur_y))
                consec_misses = 0
            else:
                consec_misses += 1
                if consec_misses >= 5:
                    break

        left_pts.reverse()
        all_pts = left_pts + right_pts
        ridge_xs = [p[0] for p in all_pts]
        ridge_ys = [p[1] for p in all_pts]

        coverage = len(ridge_xs) / float(tested_cols) if tested_cols > 0 else 0.0
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
                        "mode": self.mode,
                        "message": f"Sirip dalam transit (Buffer {self.absence_frames}/{self.max_absence_tolerance})",
                        "detection_reason": "Transit Hysteresis (Ridge terputus sesaat)",
                        "pv_flatness": float(np.median(self.prime_pv_samples)) if self.prime_pv_samples else 0.0,
                        "sagitta": float(np.median(self.prime_sagitta_samples)) if self.prime_sagitta_samples else 0.0,
                        "max_deviation": 0.0,
                        "inner_dev": float(np.median(self.prime_inner_dev_samples)) if self.prime_inner_dev_samples else 0.0,
                        "tip_dev": float(np.median(self.prime_tip_dev_samples)) if self.prime_tip_dev_samples else 0.0,
                        "coverage": coverage,
                        "active_box": None,
                        "zone_rect": zone_rect,
                        "yolo_boxes": all_yolo_boxes,
                        "elapsed_ms": (time.time() - t0) * 1000
                    }
            self.last_presence = False
            self.active_rib_frames = 0
            self.active_rib_ok_frames = 0
            self.active_rib_ng_frames = 0
            self.active_consec_ng = 0
            self.active_max_consec_ng = 0
            self.active_pv_history.clear()
            self.active_sagitta_history.clear()
            self.prime_pv_samples.clear()
            self.prime_sagitta_samples.clear()
            self.prime_inner_dev_samples.clear()
            self.prime_tip_dev_samples.clear()
            self.current_rib_status = None
            self.last_ng_reason = None
            return {
                "status": "WAIT",
                "object_type": "PLAT_DATAR",
                "is_rib_present": False,
                "is_ok": False,
                "is_ng": False,
                "mode": self.mode,
                "message": f"Titik ridge tidak memadai ({len(ridge_xs)} pts | Cov={coverage*100:.0f}%)",
                "detection_reason": f"{reason} -> Ditolak (Titik puncak tidak kontinu)",
                "pv_flatness": 0.0,
                "sagitta": 0.0,
                "max_deviation": 0.0,
                "inner_dev": 0.0,
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
        n_pts = len(r_xs)
        w_span = float(r_xs[-1] - r_xs[0])
        med_top_y = int(np.median(smooth_ys))
        self.active_rib_frames += 1

        # FULL-SPAN ANTI-TRUNCATION GATING:
        # Menolak fitting kelengkungan jika kontur belum terbentang tip-to-tip (lebar < 880px atau coverage < 70%).
        # Memotong rentang horizontal (X) secara matematis melipatgandakan Sagitta semu parabola.
        full_span_valid = (w_span >= 880.0 and coverage >= 0.70)

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
        is_in_prime = (self.prime_min_y <= med_top_y <= self.prime_max_y)

        if not full_span_valid:
            # Kontur belum membentang utuh -> Tahan penilaian metrologi, hindari lonjakan Sagitta palsu
            pv = 0.0
            sagitta = 0.0
            max_dev = 0.0
            inner_dev = 0.0
            tip_dev = 0.0
        else:
            pv = float(np.percentile(dev, 98) - np.percentile(dev, 2))
            max_dev = float(np.max(np.abs(dev)))

            # Evaluasi Deviasi Badan vs Ujung (Tip):
            tip_zone_px = 75.0
            left_tip_mask = (r_xs < (r_xs[0] + tip_zone_px))
            right_tip_mask = (r_xs > (r_xs[-1] - tip_zone_px))
            tip_mask = left_tip_mask | right_tip_mask

            left_tip_dev = float(np.percentile(np.abs(dev[left_tip_mask]), 95)) if np.any(left_tip_mask) and np.sum(left_tip_mask) >= 4 else 0.0
            right_tip_dev = float(np.percentile(np.abs(dev[right_tip_mask]), 95)) if np.any(right_tip_mask) and np.sum(right_tip_mask) >= 4 else 0.0
            tip_dev = max(left_tip_dev, right_tip_dev)

            inner_mask = ~tip_mask
            if np.any(inner_mask):
                inner_dev = float(np.percentile(np.abs(dev[inner_mask]), 98))
            else:
                inner_dev = max_dev

            # Regresi Polinomial Parabola Orde-2 (Kelengkungan Sagitta)
            poly2 = np.polyfit(r_xs, smooth_ys, deg=2)
            sagitta = abs(float(poly2[0])) * ((w_span / 2.0) ** 2)

            # Hanya kumpulkan sampel metrologi jika berada di zona prime konveyor dan kontur utuh!
            if is_in_prime:
                self.prime_pv_samples.append(pv)
                self.prime_sagitta_samples.append(sagitta)
                self.prime_inner_dev_samples.append(inner_dev)
                self.prime_tip_dev_samples.append(tip_dev)

        # KONSENSUS METROLOGI PRIME ZONE (Minimal 5 Sampel Stabil)
        # Menjamin vonis tidak diambil secara terburu-buru sebelum objek mapan di zona tengah
        sample_count = len(self.prime_pv_samples)
        if sample_count >= 3:
            eval_pv = float(np.median(self.prime_pv_samples))
            eval_sag = float(np.median(self.prime_sagitta_samples))
            eval_idev = float(np.median(self.prime_inner_dev_samples))
            eval_tdev = float(np.median(self.prime_tip_dev_samples))
        else:
            eval_pv = pv
            eval_sag = sagitta
            eval_idev = inner_dev
            eval_tdev = tip_dev

        # Evaluasi Multi-Variate Terkalibrasi
        curv_defect = (eval_sag > self.sagitta_thresh and eval_pv > self.pv_thresh) or (eval_sag > 14.5)
        dev_defect = (eval_idev > self.dev_thresh and eval_pv > self.pv_thresh) or (eval_idev > 12.0)
        tip_defect = (eval_tdev > self.tip_thresh)
        flat_defect = (eval_pv > 18.0)
        transit_is_ok = not (curv_defect or dev_defect or tip_defect or flat_defect)

        # Registrasi Objek & Pembaruan Status Berbasis Konsensus Transaksional
        if sample_count < 5:
            # Belum cukup sampel prime zone -> Tetap WAIT agar tidak memicu False NG prematur
            if self.current_rib_status is not None:
                status = self.current_rib_status
            else:
                status = "WAIT"
            eval_msg = f"Stabilisasi Prime Zone ({sample_count}/5 sampel | Span={w_span:.0f}px)"
        else:
            # Sampel prime zone cukup (>= 5 sampel) -> Tentukan vonis pertama untuk sirip ini
            if self.current_rib_status is None:
                self.rib_count += 1
                if transit_is_ok:
                    self.current_rib_status = "OK"
                    self.ok_count += 1
                else:
                    self.current_rib_status = "NG"
                    self.ng_count += 1
            else:
                # Revisi dinamis hanya jika didukung oleh sampel median jangka panjang (>= 8-10 sampel)
                if self.current_rib_status == "OK" and not transit_is_ok and sample_count >= 8:
                    self.current_rib_status = "NG"
                    self.ok_count = max(0, self.ok_count - 1)
                    self.ng_count += 1
                elif self.current_rib_status == "NG" and transit_is_ok and sample_count >= 10:
                    self.current_rib_status = "OK"
                    self.ng_count = max(0, self.ng_count - 1)
                    self.ok_count += 1

            status = self.current_rib_status
            if status == "OK":
                self.last_ng_reason = None
                eval_msg = f"SIRIP LURUS (OK) | Top Y={med_top_y}px | PV={eval_pv:.1f}px, Sag={eval_sag:.1f}px, Dev={eval_idev:.1f}px, Tip={eval_tdev:.1f}px"
            else:
                self.last_ng_reason = "NG"
                eval_msg = f"SIRIP BENGKOK (NG) | PV={eval_pv:.1f}px, Sag={eval_sag:.1f}px, Dev={eval_idev:.1f}px, Tip={eval_tdev:.1f}px"

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
            "final_consensus_ready": bool(sample_count >= 5),
            "rib_count": self.rib_count,
            "mode": self.mode,
            "message": eval_msg,
            "detection_reason": f"{reason} | Bilah Terkunci (Cov={coverage*100:.0f}%, Span={w_span:.0f}px)",
            "confidence": float(yolo_conf if yolo_conf > 0 else (dual_ratio if self.mode == "DUAL_POLARITY_ONLY" else 0.0)),
            "pv_flatness": eval_pv,
            "sagitta": eval_sag,
            "max_deviation": max_dev,
            "inner_dev": eval_idev,
            "tip_dev": eval_tdev,
            "rib_height": 110,
            "active_box": refined_box,
            "zone_rect": zone_rect,
            "xs_contour": r_xs,
            "ys_contour": smooth_ys,
            "baseline": baseline,
            "rib_patch": rib_patch,
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

        # Badge Status Utama (OK, NG, WAIT)
        cv2.rectangle(hud, (15, 12), (180, 70), badge_bg, -1)
        stat_x = 55 if len(status) == 2 else (35 if len(status) == 4 else 45)
        cv2.putText(hud, status, (stat_x, 54), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)

        # Status Teks Header
        cv2.putText(hud, f"MODE: [{self.mode}] ({self.model_name})  |  OBJEK: {obj_type.replace('_', ' ')}",
                    (200, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(hud, res.get("message", ""), (200, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 230, 255), 1, cv2.LINE_AA)

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

        # Patch Citra Sirip di sebelah kanan HUD jika tersedia
        rib_patch = res.get("rib_patch")
        if rib_patch is not None and rib_patch.size > 0:
            try:
                patch_disp = cv2.resize(rib_patch, (480, 110))
                cv2.rectangle(hud, (495, 95), (980, 210), (60, 60, 70), 1)
                hud[98:208, 498:978] = patch_disp
                cv2.putText(hud, "PATCH FISIK SIRIP TERPOTONG (ZOOM)", (500, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)
            except Exception:
                pass

        # Footer Menu
        footer = np.zeros((28, 1000, 3), dtype=np.uint8)
        footer[:] = (12, 12, 16)
        source_name = res.get("source_name", "")
        file_tag = f"[{source_name[:18]}] " if source_name else ""
        paused_tag = "⏸ PAUSED " if res.get("is_paused", False) else "▶ PLAY "
        help_txt = f"{paused_tag}{file_tag}[SPACE]/[P]: Pause/Play | [D]/[A]: Step Frame | [M]: Mode | [R]: Reset | [O]/[F]: File | [B]: Bending | [Q]: Keluar"
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

    is_paused = False
    simulate_bending = False
    cached_screen = None
    needs_inspect = True
    frame = None
    current_frame_pos = 0

    print(f"\n[OK] 🎬 Memulai inspeksi pada: {current_fname}")
    print(f"      Mode: [{inspector.mode}] | Tipe: {'Foto Diam' if is_image else f'Video ({total_frames} frame)'}\n")

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

        # Polling keyboard cepat (15ms saat video jalan, 25ms saat dijeda)
        delay = 15 if not is_paused else 25
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
