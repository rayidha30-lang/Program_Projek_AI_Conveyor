"""
==============================================================================
SISTEM TERPADU INSPEKSI TAMPAK ATAS KONVEYOR (CAM-03)
MASTER PIPELINE: HOUGH STITCHING SLAT + HYBRID RIB CONCURRENT INSPECTOR
==============================================================================
Deskripsi:
  File master ini menjalankan DUA modul inspeksi utama secara BEBARENGAN (CONCURRENT):
    1. HoughStitchingInspector (hough_stitching_inspector.py):
       - Berjalan 100% pada setiap frame di area "GERBANG INSPEKSI SLAT".
       - Mengukur kelurusan celah plat slat (Hough Line Grouping & Stitching).
       - Menghitung Flatness PV, Sagitta Parabola, Void Gap, dan Kontinuitas Celah.
       - Memproyeksikan garis kontur celah TERSTITCHING tepat pada celah fisik plat
         di dalam gerbang inspeksi menggunakan Inverse Perspective Transform (M_inv).
       - Dilengkapi Live Filter Radar PiP (Canny & Sobel-Y) di sudut kanan atas.
    2. HybridRibInspector (hybrid_rib_inspector.py):
       - Berjalan 100% pada setiap frame di zona inspeksi konveyor.
       - Mendeteksi bilah sirip tegak (Rib) berbasis YOLO + Dual-Polarity 3D.
       - Menelusuri puncak bilah fisik (Top Ridge Tracking) dan menghitung kelurusan:
         PV Flatness, Sagitta Parabola, Simpangan Badan, & Simpangan Ujung (Tip).

Prinsip Modularitas:
  - Modul Plat 100% mandiri di hough_stitching_inspector.py (ubah plat = ubah file tersebut).
  - Modul Sirip 100% mandiri di hybrid_rib_inspector.py (ubah sirip = ubah file tersebut).
  - File ini hanya memadukan dan menjalankan keduanya secara bebarengan tanpa saling menghilangkan garis atau ROI masing-masing!
==============================================================================
"""

import os
import sys
import time
import json
import argparse
import threading
import queue
import cv2
import numpy as np

# Tambahkan direktori saat ini ke sys.path agar impor modul selalu berhasil
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)


def _load_env_file():
    """Membaca file .env jika ada (tanpa ketergantungan library eksternal)"""
    candidate_paths = [
        os.path.join(CURRENT_DIR, ".env"),
        os.path.abspath(os.path.join(CURRENT_DIR, "..", ".env")),
        os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".env")),
        os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "..", ".env")),
        os.path.join(os.getcwd(), ".env"),
    ]
    for env_path in candidate_paths:
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v
                break
            except Exception:
                pass


_load_env_file()

# Bootstrap Environment CUDA 12.6 Jetson Orin Nano secara otomatis
try:
    import jetson_cuda_env  # noqa: F401
except ImportError:
    pass

# Aktifkan High-Resolution Multimedia Timer (1ms quantum) & Anti-Throttling Windows
def configure_windows_process_performance():
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes
        ctypes.windll.winmm.timeBeginPeriod(1)
    except Exception:
        pass
    try:
        import ctypes
        import os
        from ctypes import wintypes
        k = ctypes.windll.kernel32
        k.OpenProcess.restype = wintypes.HANDLE
        k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k.CloseHandle.argtypes = [wintypes.HANDLE]
        h_proc = k.OpenProcess(0x0200 | 0x0400, False, os.getpid())
        if h_proc:
            try:
                k.SetPriorityClass(h_proc, 0x00008000)
                class PROCESS_POWER_THROTTLING_STATE(ctypes.Structure):
                    _fields_ = [
                        ("Version", wintypes.DWORD),
                        ("ControlMask", wintypes.DWORD),
                        ("StateMask", wintypes.DWORD),
                    ]
                state = PROCESS_POWER_THROTTLING_STATE(1, 0x1, 0)
                k.SetProcessInformation.argtypes = [
                    wintypes.HANDLE,
                    wintypes.DWORD,
                    ctypes.c_void_p,
                    wintypes.DWORD
                ]
                k.SetProcessInformation(h_proc, 4, ctypes.byref(state), ctypes.sizeof(state))
            finally:
                k.CloseHandle(h_proc)
    except Exception:
        pass

configure_windows_process_performance()

# Impor langsung kedua engine inspector
from hybrid_rib_inspector import (
    HybridRibInspector,
    open_media_source,
    select_file_dialog
)
from hough_stitching_inspector import (
    HoughStitchingInspector,
    get_adaptive_slat_polygon
)

# Impor Modul Jembatan Database SQLite
DB_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "..", "..", "Database"))
if DB_DIR not in sys.path:
    sys.path.insert(0, DB_DIR)

try:
    from db_bridge import InspectionDBBridge
except ImportError:
    InspectionDBBridge = None


class AsyncFrameGrabber:
    """
    Background thread untuk cv2.VideoCapture non-blocking pre-decode.
    
    Arsitektur Produsen-Konsumen (Producer-Consumer):
    - Untuk FILE VIDEO (Offline/Testing):
      Menyimpan antrean frame (buffer_size=2) TANPA membuang frame (ZERO DROPPED FRAMES).
      Ketika buffer penuh, thread produsen menunggu main loop selesai memproses frame aktif.
      Ini menghilangkan latency decode 9.4ms per frame TANPA melompati frame (mencegah efek speed up / fast-forward).
    - Untuk KAMERA LIVE (RTSP/USB Stream):
      Membuang frame lama jika buffer penuh agar visualisasi selalu real-time dengan latensi terendah.
    """

    def __init__(self, source_path, buffer_size=2):
        self.source_path = source_path
        self.is_live = False
        src_str = str(source_path)
        if src_str.isdigit() or src_str.startswith(("rtsp://", "http://", "https://", "/dev/video")):
            self.is_live = True

        self.cap = cv2.VideoCapture(int(source_path) if src_str.isdigit() else source_path)
        self._buf = queue.Queue(maxsize=buffer_size)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._seek_target = None
        self._current_pos = 0

        # Ambil FPS native dan total frame dari video
        self._native_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not self._native_fps or self._native_fps <= 0 or self._native_fps > 120:
            self._native_fps = 30.0

        self._total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not self.is_live else 0

        self._thread = threading.Thread(target=self._grab_loop, daemon=True, name="FrameGrabber")
        self._thread.start()

    def _grab_loop(self):
        while not self._stop.is_set():
            # 1. Cek permintaan seek
            with self._lock:
                if self._seek_target is not None:
                    target = self._seek_target
                    self._seek_target = None
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                    while not self._buf.empty():
                        try:
                            self._buf.get_nowait()
                        except queue.Empty:
                            break

            # 2. Untuk file video: jika buffer penuh, tunggu main loop (JANGAN BACA & JANGAN BUANG FRAME!)
            # Ini kunci vital agar video TIDAK speed up atau melompati bilah slat/sirip!
            if not self.is_live and self._buf.full():
                time.sleep(0.003)
                continue

            pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            ret, frame = self.cap.read()
            if not ret or frame is None:
                if not self.is_live:
                    # Loop video dari frame 0
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    pos = 0
                    ret, frame = self.cap.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

            # 3. Masukkan ke antrean buffer
            if self.is_live:
                if self._buf.full():
                    try:
                        self._buf.get_nowait()
                    except queue.Empty:
                        pass
                try:
                    self._buf.put((pos, frame), timeout=0.02)
                except queue.Full:
                    pass
            else:
                while not self._stop.is_set() and self._seek_target is None:
                    try:
                        self._buf.put((pos, frame), timeout=0.05)
                        break
                    except queue.Full:
                        continue

    def read(self):
        """Ambil frame berikutnya dari antrean buffer (non-blocking)."""
        try:
            pos, frame = self._buf.get(timeout=0.25)
            self._current_pos = pos
            return True, frame
        except queue.Empty:
            return False, None

    def get_pos_frames(self):
        """Posisi nomor frame yang sedang diproses di layar."""
        return self._current_pos

    def set_pos_frames(self, pos):
        """Seek video ke nomor frame tertentu."""
        with self._lock:
            self._seek_target = max(0, pos)
            self._current_pos = max(0, pos)
            while not self._buf.empty():
                try:
                    self._buf.get_nowait()
                except queue.Empty:
                    break

    def release(self):
        self._stop.set()
        while not self._buf.empty():
            try:
                self._buf.get_nowait()
            except queue.Empty:
                break
        self._thread.join(timeout=1.0)
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

    def isOpened(self):
        return self.cap.isOpened()

    @property
    def total_frames(self):
        return self._total_frames

    @property
    def fps_native(self):
        return self._native_fps




class CombinedTopInspector:
    """
    Master Pipeline Inspeksi Terpadu CAM-03.
    Menjalankan inspeksi Plat Slat dan Bilah Sirip secara bebarengan (paralel pada setiap frame).
    """

    def __init__(
        self,
        config_path=None,
        rib_mode="HYBRID",
        rib_model_path=None,
        device=None,
        enable_db=False
    ):
        base_dir = os.path.dirname(CURRENT_DIR)
        if config_path is None:
            config_path = os.path.join(base_dir, "configs", "top_inspection_config.json")
        self.config_path = config_path
        self.device = device

        # Inisialisasi Database Bridge (Default: False / Dry Run)
        self.db_bridge = None
        if InspectionDBBridge is not None:
            self.db_bridge = InspectionDBBridge(enabled=enable_db, camera_id="CAM-01")

        # 1. Parameter Default Plat Slat
        calib_mll = 0.70
        calib_v = 116.0
        max_pv_slat = 16.5
        max_sag_slat = 8.6
        max_void_slat = 1450
        min_cov_slat = 0.25

        # 2. Parameter Default Sirip (Rib)
        rib_pv = 14.0
        rib_sag = 11.5
        rib_dev = 8.5
        rib_tip = 9.5
        rib_conf = 0.55

        # Polygon cache
        self.slat_polygon = None
        self.rib_zone = None

        # Muat konfigurasi dari JSON jika tersedia
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    cfg = json.load(f)

                if "calibration_profile" in cfg:
                    cp = cfg["calibration_profile"]
                    calib_mll = float(cp.get("calibrated_mll_ratio", calib_mll))
                    calib_v = float(cp.get("recommended_v_thresh", calib_v))

                th = cfg.get("thresholds", {})
                max_pv_slat = float(th.get("max_flatness_pv", max_pv_slat))
                max_sag_slat = float(th.get("max_sagitta", max_sag_slat))
                max_void_slat = int(th.get("max_center_void", max_void_slat))
                self.scale_factor_mm_per_px = float(th.get("scale_factor_mm_per_px", 0.515))
                self.scale_factor_y_mm_per_px = float(th.get("scale_factor_y_mm_per_px", 0.442))

                rib_pv = float(th.get("rib_pv_threshold", rib_pv))
                rib_sag = float(th.get("rib_sagitta_threshold", rib_sag))
                rib_dev = float(th.get("rib_dev_threshold", th.get("rib_max_dev_threshold", rib_dev)))
                rib_tip = float(th.get("rib_tip_threshold", rib_tip))
                rib_conf = float(th.get("rib_conf_threshold", rib_conf))

                if "rib_polygon" in cfg and len(cfg["rib_polygon"]) >= 4:
                    self.rib_zone = cfg["rib_polygon"]

                if "slat_polygon" in cfg and len(cfg["slat_polygon"]) >= 4:
                    self.slat_polygon = cfg["slat_polygon"]

                print(f"[*] Konfigurasi Terpadu dimuat dari: {os.path.basename(config_path)}")
            except Exception as e:
                print(f"[!] Gagal membaca file config: {e}")

        # Inisialisasi Engine Plat Slat (HoughStitchingInspector)
        self.slat_inspector = HoughStitchingInspector(
            v_thresh=calib_v,
            mll_ratio=calib_mll,
            delta_theta_thresh=1.5,
            min_line_len=45,
            bridge_gap_px=115.0,
            strict_mode=True
        )
        self.slat_inspector.max_flatness_pv = max_pv_slat
        self.slat_inspector.max_sagitta = max_sag_slat
        self.slat_inspector.max_center_void = max_void_slat
        self.slat_inspector.min_continuity_cov = min_cov_slat

        # Override nilai scale_factor dari .env / sistem jika didefinisikan oleh user
        env_scale_x = os.environ.get("SCALE_FACTOR_MM_PER_PX")
        if env_scale_x:
            try:
                self.scale_factor_mm_per_px = float(env_scale_x)
                print(f"[*] Skala Kalibrasi X dimuat dari .env: {self.scale_factor_mm_per_px} mm/px")
            except ValueError:
                pass
        env_scale_y = os.environ.get("SCALE_FACTOR_Y_MM_PER_PX")
        if env_scale_y:
            try:
                self.scale_factor_y_mm_per_px = float(env_scale_y)
                print(f"[*] Skala Kalibrasi Y dimuat dari .env: {self.scale_factor_y_mm_per_px} mm/px")
            except ValueError:
                pass

        if not hasattr(self, "scale_factor_mm_per_px"):
            self.scale_factor_mm_per_px = 0.515
        self.scale_factor_y_mm_per_px = getattr(self, "scale_factor_y_mm_per_px", 0.442)
        self.slat_inspector.scale_factor_mm_per_px = self.scale_factor_mm_per_px

        # Model bobot Sirip: periksa parameter arg, lalu .env MODEL_TOP_PATH
        if rib_model_path is None:
            rib_model_path = os.environ.get("MODEL_TOP_PATH")

        # Inisialisasi Engine Sirip (HybridRibInspector dengan akselerasi CUDA / CPU)
        self.rib_inspector = HybridRibInspector(
            mode=rib_mode,
            model_path=rib_model_path,
            conf_thresh=rib_conf,
            pv_thresh=rib_pv,
            sagitta_thresh=rib_sag,
            dev_thresh=rib_dev,
            tip_thresh=rib_tip,
            device=device
        )
        self.rib_inspector.scale_factor_mm_per_px = self.scale_factor_mm_per_px
        self.rib_inspector.scale_factor_y_mm_per_px = self.scale_factor_y_mm_per_px

        # State Early-Catch & Peak-Hold Latch (Proteksi Oklusi Plat Sebelum Sirip)
        self.slat_history_samples = []
        self.latched_slat_data = None
        self.latched_hold_frames = 0

        # Single-Capture Per-Object Latch & Best-Shot Peak Buffer
        self.logged_slat_id = None
        self.logged_rib_id = None
        self._best_top_candidate = None
        self._current_slat_cycle_id = None
        self._current_rib_cycle_id = None
        self._last_emitted_slat_id = None
        self._last_emitted_rib_id = None

        # Cache batas vertikal gerbang plat slat
        self.slat_y_min = None
        self.slat_y_max = None

        # Pre-allocated zero-copy canvas buffer (1000 x 820 px: 520 view + 300 HUD)
        self._screen_buf = np.zeros((820, 1000, 3), dtype=np.uint8)

    def reset_counters(self):
        """Mereset seluruh statistik QC (Slat dan Rib) ke nilai 0."""
        self.slat_inspector.reset_counters()
        self.rib_inspector.reset_counters()
        self.slat_history_samples.clear()
        self.latched_slat_data = None
        self.latched_hold_frames = 0
        self.logged_slat_id = None
        self.logged_rib_id = None
        self._best_top_candidate = None
        self._current_slat_cycle_id = None
        self._current_rib_cycle_id = None
        self._last_emitted_slat_id = None
        self._last_emitted_rib_id = None
        if self.db_bridge is not None and hasattr(self.db_bridge, "reset_entity_cache"):
            self.db_bridge.reset_entity_cache()

    def toggle_db(self):
        """Membalik status saklar database logging (ON <-> OFF)."""
        if self.db_bridge is not None:
            return self.db_bridge.toggle()
        return False

    def is_db_enabled(self):
        """Mengecek apakah database logging aktif."""
        if self.db_bridge is not None:
            return self.db_bridge.is_enabled()
        return False

    def render_annotated_fullres(self, frame_raw, inspect_res, slat_no=None, rib_no=None):
        """
        Merender anotasi metrologi langsung pada citra resolusi penuh frame asli (misal 1080p).
        Digunakan khusus untuk capture bukti NG sehingga gambar tajam saat dibuka di Web Dashboard & Teams.
        """
        if frame_raw is None:
            return None
        ann = frame_raw.copy()
        h_f, w_f = ann.shape[:2]

        slat_res = inspect_res.get("slat_res", {}) if inspect_res else {}
        rib_res = inspect_res.get("rib_res", {}) if inspect_res else {}
        scale = getattr(self, "scale_factor_mm_per_px", 0.515)

        # 1. Gambar Polygon Gerbang Slat jika ada
        if self.slat_polygon is not None and len(self.slat_polygon) >= 3:
            pts_px = []
            for p in self.slat_polygon:
                px = int(p[0] * w_f if p[0] <= 1.0 else p[0])
                py = int(p[1] * h_f if p[1] <= 1.0 else p[1])
                pts_px.append([px, py])
            pts_arr = np.array(pts_px, dtype=np.int32).reshape((-1, 1, 2))
            is_slat_ng = bool(slat_res.get("is_ng", False))
            poly_col = (0, 0, 255) if is_slat_ng else (0, 220, 0)
            cv2.polylines(ann, [pts_arr], True, poly_col, 3, cv2.LINE_AA)
            slat_conf = float(slat_res.get("confidence", slat_res.get("coverage", 0.0)) or 0.0)
            conf_slat_str = f" [Conf: {slat_conf:.2f}]" if slat_conf > 0 else ""
            cv2.putText(ann, f"GERBANG SLAT #{slat_no if slat_no is not None else ''}{conf_slat_str}",
                        (pts_px[0][0], max(26, pts_px[0][1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, poly_col, 2, cv2.LINE_AA)

        # 2. Proyeksikan Garis Celah Slat Terstitching ke Koordinat Asli Kamera
        xs = slat_res.get("xs_contour")
        ys = slat_res.get("ys_contour")
        if xs is not None and ys is not None and len(xs) > 10 and "polygon_px" in slat_res:
            try:
                src_pts = slat_res["polygon_px"].astype(np.float32)
                dst_pts = np.array([
                    [0, 0],
                    [self.slat_inspector.canonical_w - 1, 0],
                    [self.slat_inspector.canonical_w - 1, self.slat_inspector.canonical_h - 1],
                    [0, self.slat_inspector.canonical_h - 1]
                ], dtype=np.float32)
                M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)

                unwarp_pts = np.column_stack([xs, ys]).reshape(-1, 1, 2).astype(np.float32)
                cam_pts = cv2.perspectiveTransform(unwarp_pts, M_inv).reshape(-1, 2)
                cam_pts[:, 0] = np.clip(cam_pts[:, 0], 0, w_f - 1)
                cam_pts[:, 1] = np.clip(cam_pts[:, 1], 0, h_f - 1)
                cam_pts_int = cam_pts.astype(np.int32)

                seam_col = (0, 0, 255) if slat_res.get("is_ng", False) else (0, 255, 0)
                cv2.polylines(ann, [cam_pts_int], False, seam_col, 4, cv2.LINE_AA)
            except Exception:
                pass

        # 3. Gambar Bounding Box Sirip & Garis Puncak Sirip
        active_box = rib_res.get("active_box")
        if active_box:
            bx1, by1, bx2, by2 = active_box
            is_rib_ng = bool(rib_res.get("is_ng", False))
            b_col = (0, 0, 255) if is_rib_ng else (255, 170, 0)
            cv2.rectangle(ann, (bx1, by1), (bx2, by2), b_col, 3, cv2.LINE_AA)
            rib_conf = float(rib_res.get("confidence", 0.0) or 0.0)
            conf_rib_str = f" [Conf: {rib_conf:.2f}]" if rib_conf > 0 else ""
            cv2.putText(
                ann,
                f"SIRIP #{rib_no if rib_no is not None else ''} [{'NG' if is_rib_ng else 'OK'}]{conf_rib_str}",
                (bx1, max(25, by1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                b_col,
                2,
                cv2.LINE_AA
            )

        xs_r = rib_res.get("xs_contour")
        ys_r = rib_res.get("ys_contour")
        if xs_r is not None and ys_r is not None and len(xs_r) > 10:
            is_rib_ng = bool(rib_res.get("is_ng", False))
            r_col = (0, 0, 255) if is_rib_ng else (0, 255, 0)
            pts_r = np.column_stack([xs_r, ys_r]).astype(np.int32)
            cv2.polylines(ann, [pts_r], False, r_col, 4, cv2.LINE_AA)

        # 4. Top HUD Banner Anotasi Metrologi (80 px tinggi di puncak frame)
        banner_h = 80
        overlay = ann.copy()
        cv2.rectangle(overlay, (0, 0), (w_f, banner_h), (18, 18, 24), -1)
        cv2.addWeighted(overlay, 0.88, ann, 0.12, 0, ann)
        cv2.line(ann, (0, banner_h), (w_f, banner_h), (0, 0, 255), 2)

        s_pv = float(slat_res.get("pv_flatness", 0.0) or 0.0)
        s_sag = float(slat_res.get("sagitta", 0.0) or 0.0)
        s_void = int(slat_res.get("center_void", 0) or 0)
        r_dev = float(rib_res.get("max_dev", rib_res.get("dev", 0.0)) or 0.0)
        r_sag = float(rib_res.get("sagitta", 0.0) or 0.0)

        s_stat = slat_res.get("status", "WAIT")
        r_stat = rib_res.get("status", "WAIT")
        slat_conf = float(slat_res.get("confidence", slat_res.get("coverage", 0.0)) or 0.0)
        rib_conf = float(rib_res.get("confidence", 0.0) or 0.0)
        conf_slat_hdr = f" ({slat_conf:.2f})" if slat_conf > 0 else ""
        conf_rib_hdr = f" ({rib_conf:.2f})" if rib_conf > 0 else ""

        title = f"[AI METROLOGY EVIDENCE] TOP CAM-01 | SLAT #{slat_no if slat_no is not None else '-'}: [{s_stat}]{conf_slat_hdr} | RIB #{rib_no if rib_no is not None else '-'}: [{r_stat}]{conf_rib_hdr}"
        cv2.putText(ann, title, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 230, 255), 2, cv2.LINE_AA)

        metrics_txt = (
            f"PLAT: PV={s_pv*scale:.1f}mm ({s_pv:.1f}px) | Sag={s_sag*scale:.1f}mm ({s_sag:.1f}px) | Void={s_void}px  ||  "
            f"RIB: Dev={r_dev*scale:.1f}mm ({r_dev:.1f}px) | Sag={r_sag*scale:.1f}mm ({r_sag:.1f}px)"
        )
        cv2.putText(ann, metrics_txt, (20, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 240, 240), 1, cv2.LINE_AA)

        return ann

    def flush_best_shot_if_needed(self):
        """
        Memancarkan kandidat Best-Shot ke db_bridge jika siklus selesai dan terdapat NG terkonfirmasi.
        Menjamin frame yang tersimpan adalah frame dengan keparahan puncak (Peak Abnormality).
        """
        if self._best_top_candidate is None:
            return False
        cand = self._best_top_candidate
        self._best_top_candidate = None

        slat_no = cand["slat_no"]
        rib_no = cand["rib_no"]

        # Cek deduplikasi agar 1 siklus NG tidak pernah di-emit ganda
        if slat_no is not None and slat_no == self._last_emitted_slat_id and rib_no == self._last_emitted_rib_id:
            return False

        self._last_emitted_slat_id = slat_no
        self._last_emitted_rib_id = rib_no

        if self.db_bridge is not None and self.db_bridge.is_enabled():
            return self.db_bridge.emit_top_inspection(
                frame_raw=cand["frame_raw"],
                frame_ann=cand["frame_ann"],
                slat_result=cand["slat_res"],
                rib_result=cand["rib_res"],
                slat_number=slat_no,
                rib_number=rib_no
            )
        return False

    def record_if_ng(self, frame_raw, frame_ann, inspect_res, slat_no=None, rib_no=None):
        """
        Merekam frame ke SQLite & folder captures menggunakan mekanisme Best-Shot Peak Buffer.
        Alih-alih langsung menyimpan frame awal yang terpotong saat baru masuk, sistem mengumpulkan
        kandidat selama objek melintas dan menyimpan frame dengan deviasi keparahan tertinggi (Peak Abnormality)
        langsung pada resolusi penuh kamera asli (1080p).
        """
        if self.db_bridge is None or not self.db_bridge.is_enabled():
            return False

        slat_res = inspect_res.get("slat_res", {}) if inspect_res else {}
        rib_res = inspect_res.get("rib_res", {}) if inspect_res else {}

        is_slat_ng = bool(slat_res.get("is_ng", False))
        is_rib_ng = bool(rib_res.get("is_ng", False))

        # Filter eliminasi false alarm: Abaikan vonis NG jika kontur garis sirip belum membentang penuh (tip-to-tip < 850px)
        rib_xs = rib_res.get("xs_contour")
        rib_span = float(rib_xs[-1] - rib_xs[0]) if (rib_xs is not None and len(rib_xs) > 10) else 0.0
        if is_rib_ng and rib_span < 850.0:
            is_rib_ng = False

        curr_slat_no = slat_no if slat_no is not None else getattr(self.slat_inspector, "slat_count", 0)
        curr_rib_no = rib_no if rib_no is not None else getattr(self.rib_inspector, "rib_count", 0)

        # 1. Jika siklus berganti (slat baru atau rib baru melintas), emit kandidat siklus sebelumnya
        if self._current_slat_cycle_id is not None and curr_slat_no != self._current_slat_cycle_id:
            self.flush_best_shot_if_needed()
            self._current_slat_cycle_id = curr_slat_no
            self._current_rib_cycle_id = curr_rib_no
        elif self._current_slat_cycle_id is None:
            self._current_slat_cycle_id = curr_slat_no
            self._current_rib_cycle_id = curr_rib_no

        if not is_slat_ng and not is_rib_ng:
            return False

        # 2. Hitung severity score (tingkat keparahan cacat) pada frame saat ini
        slat_score = 0.0
        if is_slat_ng:
            s_pv = float(slat_res.get("pv_flatness", 0.0) or 0.0)
            s_sag = float(slat_res.get("sagitta", 0.0) or 0.0)
            s_void = float(slat_res.get("center_void", 0) or 0) / 40.0
            slat_score = max(s_pv, s_sag, s_void)

        rib_score = 0.0
        if is_rib_ng:
            r_dev = float(rib_res.get("max_dev", rib_res.get("dev", 0.0)) or 0.0)
            r_sag = float(rib_res.get("sagitta", 0.0) or 0.0)
            rib_score = max(r_dev, r_sag)

        total_severity = max(slat_score, rib_score)

        # 3. Update kandidat Best-Shot jika frame ini memiliki keparahan lebih tinggi
        need_update = False
        if self._best_top_candidate is None:
            need_update = True
        elif total_severity > self._best_top_candidate["severity"]:
            need_update = True

        if need_update:
            # Render anotasi 1080p full-res hanya saat frame terbaik terdeteksi (Zero FPS impact)
            full_ann = self.render_annotated_fullres(frame_raw, inspect_res, curr_slat_no, curr_rib_no)
            self._best_top_candidate = {
                "severity": total_severity,
                "frame_raw": frame_raw.copy() if frame_raw is not None else None,
                "frame_ann": full_ann if full_ann is not None else frame_ann,
                "slat_res": dict(slat_res),
                "rib_res": dict(rib_res),
                "slat_no": curr_slat_no,
                "rib_no": curr_rib_no
            }
            return True

        return False

    def close(self):
        """Pembersihan resource saat ditutup."""
        try:
            self.flush_best_shot_if_needed()
        except Exception:
            pass
        if hasattr(self, "db_bridge") and self.db_bridge is not None:
            self.db_bridge.close()

    def inspect(self, frame, simulate_bending=False):
        """
        Menjalankan KEDUA inspeksi secara bebarengan (bersamaan pada setiap frame):
        1. HoughStitchingInspector memproses gerbang celah plat slat.
        2. HybridRibInspector memproses zona gerbang sirip.
        Kedua hasil dikembalikan bersamaan sehingga tidak ada garis atau ROI yang hilang!
        """
        t0 = time.time()
        if frame is None or frame.size == 0:
            return {
                "status": "WAIT",
                "slat_res": None,
                "rib_res": None,
                "elapsed_ms": 0.0
            }

        h_f, w_f = frame.shape[:2]

        # 1. Tentukan ROI Plat Slat adaptif jika belum diinisialisasi
        if self.slat_polygon is None:
            self.slat_polygon, trig_box, _ = get_adaptive_slat_polygon(w_f, h_f, self.config_path)
            self.slat_inspector.trigger_box = trig_box

        if not hasattr(self, "slat_y_min") or self.slat_y_min is None:
            slat_pts = []
            for p in self.slat_polygon:
                px = p[0] * w_f if p[0] <= 1.0 else p[0]
                py = p[1] * h_f if p[1] <= 1.0 else p[1]
                slat_pts.append([px, py])
            slat_pts = np.array(slat_pts)
            self.slat_y_min = float(np.min(slat_pts[:, 1]))
            self.slat_y_max = float(np.max(slat_pts[:, 1]))

        # 2. JALANKAN INSPEKSI SIRIP (HYBRID RIB PADA GPU) TERLEBIH DAHULU
        rib_res = self.rib_inspector.inspect_frame(
            frame,
            zone_poly=self.rib_zone,
            simulate_bending=simulate_bending
        )
        is_rib_present = rib_res.get("is_rib_present", False)
        active_box = rib_res.get("active_box")

        # 3. MASKING DINAMIS BILAH SIRIP DENGAN NEUTRAL SLAT TEXTURE FILL
        # Melenyapkan kontur bilah sirip & las kaki bilah secara absolut dari citra plat slat
        # dengan mengisi seluruh bounding box sirip menggunakan warna median plat slat sekitar.
        frame_for_slat = frame
        is_slat_occluded = False

        # Kumpulkan semua bounding box sirip (active_box + seluruh deteksi YOLO)
        all_boxes_to_mask = []
        if active_box is not None:
            all_boxes_to_mask.append(active_box)
        for yb in rib_res.get("yolo_boxes", []):
            b = yb.get("box") if isinstance(yb, dict) else yb
            if b is not None and b not in all_boxes_to_mask:
                all_boxes_to_mask.append(b)

        if len(all_boxes_to_mask) > 0:
            frame_for_slat = frame.copy()
            for box in all_boxes_to_mask:
                bx1, by1, bx2, by2 = box
                # Cek apakah bilah sirip berada di dalam atau mendekati gerbang inspeksi plat slat
                if by2 >= (self.slat_y_min - 35) and by1 <= (self.slat_y_max + 35):
                    is_slat_occluded = True

                # Ambil sampel warna bersih plat slat dari bawah bilah sirip (atau dari atas jika di dasar frame)
                sample_y1 = min(h_f - 12, max(0, by2 + 10))
                sample_y2 = min(h_f, sample_y1 + 12)
                sample_strip = frame_for_slat[sample_y1:sample_y2, max(0, bx1):min(w_f, bx2)]
                if sample_strip.size == 0 or sample_strip.shape[0] < 2:
                    sample_y1 = max(0, by1 - 20)
                    sample_y2 = min(h_f, sample_y1 + 12)
                    sample_strip = frame_for_slat[sample_y1:sample_y2, max(0, bx1):min(w_f, bx2)]

                if sample_strip.size > 0:
                    med_col = np.median(sample_strip.reshape(-1, 3), axis=0).astype(np.uint8)
                else:
                    med_col = np.array([80, 80, 80], dtype=np.uint8)

                # Isi area bilah dan akar las dengan tekstur datar homogen (gradien Sobel Y = 0)
                pad_y1 = max(0, by1 - 10)
                pad_y2 = min(h_f, by2 + 18)
                pad_x1 = max(0, bx1 - 25)
                pad_x2 = min(w_f, bx2 + 25)
                cv2.rectangle(
                    frame_for_slat,
                    (pad_x1, pad_y1),
                    (pad_x2, pad_y2),
                    (int(med_col[0]), int(med_col[1]), int(med_col[2])),
                    -1
                )

        # 4. JALANKAN INSPEKSI PLAT SLAT DENGAN CITRA TERMASKING
        slat_res = self.slat_inspector.inspect_frame(frame_for_slat, polygon_pts=self.slat_polygon)

        # 5. ZONAL GATING & MULTI-FRAME CONSENSUS (ELIMINASI FALSE NG TRANSISI PLAT)
        # Menghilangkan lonjakan cacat semu saat plat pertama kali masuk ke gerbang kanonikal.
        y_seam = slat_res.get("y_seam", 0.0)
        seam_energy = slat_res.get("seam_energy", 0.0)
        is_trans = slat_res.get("is_transition", True)
        xs = slat_res.get("xs_contour")
        ys = slat_res.get("ys_contour")

        # Pembagian Zona Inspeksi Celah Slat:
        # - Probing Zone (36 <= y_seam < 58 px): Celah baru masuk gerbang -> status WAIT (tahan penilaian)
        # - Prime Measurement Zone (58 <= y_seam <= 86 px): Celah tepat di tengah stabil -> Konsensus Median
        # - Exiting / Occluded Zone (y_seam > 86 px atau tertutup sirip) -> Evaluasi Latch Peak-Hold
        in_probing_zone = (36.0 <= y_seam < 58.0)
        in_prime_zone = (58.0 <= y_seam <= 86.0)

        has_valid_seam = (
            (not is_trans) and
            (xs is not None) and
            (len(xs) > 30) and
            (seam_energy >= 22.0)
        )

        if in_probing_zone:
            # Fase transisi masuk: status wajib WAIT agar noise batas atas kanonikal tidak memicu False NG
            slat_res["status"] = "WAIT"
            slat_res["display_status"] = "WAIT"
            slat_res["is_ok"] = False
            slat_res["is_ng"] = False
            slat_res["eval_mode"] = f"WAIT (Masuk Gerbang | Y={y_seam:.0f}px)"

        elif in_prime_zone and has_valid_seam:
            pv = slat_res.get("pv_flatness", 0.0)
            sag = slat_res.get("sagitta", 0.0)
            cov = slat_res.get("coverage", 0.0)
            void = slat_res.get("center_void", 0)

            # Simpan sampel bersih selama celah berada di zona tengah optimal (58-86 px)
            self.slat_history_samples.append({
                "pv": pv,
                "sagitta": sag,
                "coverage": cov,
                "center_void": void,
                "y_seam": y_seam,
                "energy": seam_energy,
                "xs": xs,
                "ys": ys,
                "polygon_px": slat_res.get("polygon_px")
            })

            # Evaluasi berbasis konsensus multi-frame (minimal 3 sampel stabil di zona prime)
            if len(self.slat_history_samples) < 3:
                slat_res["status"] = "WAIT"
                slat_res["display_status"] = "WAIT"
                slat_res["is_ok"] = False
                slat_res["is_ng"] = False
                slat_res["eval_mode"] = f"WAIT (Stabilisasi Celah: {len(self.slat_history_samples)}/3 sampel)"
            else:
                all_sags = [s["sagitta"] for s in self.slat_history_samples]
                all_pvs = [s["pv"] for s in self.slat_history_samples]
                all_voids = [s["center_void"] for s in self.slat_history_samples]

                eval_sag = float(np.median(all_sags))
                eval_pv = float(np.median(all_pvs))
                eval_void = int(np.median(all_voids)) if all_voids else 0

                is_flat_ok = bool(eval_pv <= self.slat_inspector.max_flatness_pv)
                is_curv_ok = bool(eval_sag <= self.slat_inspector.max_sagitta)
                is_void_ok = bool(eval_void < self.slat_inspector.max_center_void)
                is_plate_ok = bool(is_flat_ok and is_curv_ok and is_void_ok)

                slat_res["status"] = "OK" if is_plate_ok else "NG"
                slat_res["display_status"] = slat_res["status"]
                slat_res["is_ok"] = is_plate_ok
                slat_res["is_ng"] = not is_plate_ok
                slat_res["pv_flatness"] = eval_pv
                slat_res["sagitta"] = eval_sag
                slat_res["center_void"] = eval_void

                if is_plate_ok:
                    slat_res["eval_mode"] = f"LURUS OK (PV={eval_pv:.1f}px, Sag={eval_sag:.1f}px)"
                elif not is_curv_ok:
                    slat_res["eval_mode"] = f"NG: BENGKOK/MELENGKUNG (Sag={eval_sag:.1f}px > {self.slat_inspector.max_sagitta:.1f}px)"
                elif not is_flat_ok:
                    slat_res["eval_mode"] = f"NG: KELURUSAN KASAR (PV={eval_pv:.1f}px > {self.slat_inspector.max_flatness_pv:.1f}px)"
                else:
                    slat_res["eval_mode"] = f"NG: CELAH MANGAP (Void={eval_void}px)"

        else:
            # Skenario: Celah terhalang/tertutup fisik sirip atau plat selesai lewat zona prime
            if len(self.slat_history_samples) >= 2:
                # Evaluasi seluruh sampel bersih yang berhasil terekam sebelum celah tertutup sirip
                all_sags = [s["sagitta"] for s in self.slat_history_samples]
                all_pvs = [s["pv"] for s in self.slat_history_samples]
                all_voids = [s["center_void"] for s in self.slat_history_samples]

                eval_sag = float(np.median(all_sags))
                eval_pv = float(np.median(all_pvs))
                eval_void = int(np.median(all_voids)) if all_voids else 0
                eval_cov = float(np.median([s["coverage"] for s in self.slat_history_samples]))

                is_flat_ok = bool(eval_pv <= self.slat_inspector.max_flatness_pv)
                is_curv_ok = bool(eval_sag <= self.slat_inspector.max_sagitta)
                is_void_ok = bool(eval_void < self.slat_inspector.max_center_void)
                is_plate_ok = bool(is_flat_ok and is_curv_ok and is_void_ok)

                # Pilih sampel representatif terbaik untuk proyeksi kontur visual
                best_sample = min(self.slat_history_samples, key=lambda s: abs(s["y_seam"] - 70.0))

                if is_plate_ok:
                    msg = f"LURUS OK (Early-Catch Latch: PV={eval_pv:.1f}px, Sag={eval_sag:.1f}px)"
                elif not is_curv_ok:
                    msg = f"NG: BENGKOK (Early-Catch Latch: Sag={eval_sag:.1f}px > {self.slat_inspector.max_sagitta:.1f}px)"
                elif not is_flat_ok:
                    msg = f"NG: DEFLEKSI KASAR (Early-Catch Latch: PV={eval_pv:.1f}px > {self.slat_inspector.max_flatness_pv:.1f}px)"
                else:
                    msg = f"NG: CELAH TERBUKA (Early-Catch Latch: Void={eval_void}px)"

                self.latched_slat_data = {
                    "status": "OK" if is_plate_ok else "NG",
                    "display_status": "OK" if is_plate_ok else "NG",
                    "is_ok": is_plate_ok,
                    "is_ng": not is_plate_ok,
                    "pv_flatness": eval_pv,
                    "sagitta": eval_sag,
                    "coverage": eval_cov,
                    "center_void": eval_void,
                    "eval_mode": msg,
                    "xs_contour": best_sample["xs"],
                    "ys_contour": best_sample["ys"],
                    "y_seam": best_sample["y_seam"],
                    "polygon_px": best_sample["polygon_px"]
                }
                self.latched_hold_frames = 28  # Tahan data selama 28 frame (~0.9 detik saat sirip melintas)
                self.slat_history_samples.clear()

            # Pertahankan durasi latch selama bilah sirip masih menutupi gerbang slat
            if is_slat_occluded and self.latched_slat_data is not None:
                self.latched_hold_frames = max(self.latched_hold_frames, 20)

            # Terapkan hasil latched jika masih dalam durasi hold (saat celah tertutup sirip)
            if self.latched_hold_frames > 0 and self.latched_slat_data is not None:
                self.latched_hold_frames -= 1
                slat_res["status"] = self.latched_slat_data["status"]
                slat_res["display_status"] = self.latched_slat_data["display_status"]
                slat_res["is_ok"] = self.latched_slat_data["is_ok"]
                slat_res["is_ng"] = self.latched_slat_data["is_ng"]
                slat_res["pv_flatness"] = self.latched_slat_data["pv_flatness"]
                slat_res["sagitta"] = self.latched_slat_data["sagitta"]
                slat_res["coverage"] = self.latched_slat_data["coverage"]
                slat_res["center_void"] = self.latched_slat_data["center_void"]
                slat_res["eval_mode"] = self.latched_slat_data["eval_mode"]
                slat_res["xs_contour"] = self.latched_slat_data["xs_contour"]
                slat_res["ys_contour"] = self.latched_slat_data["ys_contour"]

        slat_res["is_occluded"] = is_slat_occluded

        # 6. KONSENSUS STATUS SISTEM TERPADU
        slat_stat = slat_res.get("display_status", slat_res.get("status", "WAIT"))
        rib_stat = rib_res.get("status", "WAIT")
        is_conveyor_moving = slat_res.get("is_conveyor_moving", True)

        if rib_res.get("is_ng", False) or slat_res.get("is_ng", False):
            overall_status = "NG"
        elif rib_res.get("is_ok", False) or slat_res.get("is_ok", False):
            overall_status = "OK"
        elif not is_conveyor_moving:
            overall_status = "HOLD"
        else:
            overall_status = "WAIT"

        elapsed_ms = (time.time() - t0) * 1000.0
        return {
            "status": overall_status,
            "slat_res": slat_res,
            "rib_res": rib_res,
            "is_conveyor_moving": is_conveyor_moving,
            "elapsed_ms": elapsed_ms
        }

    def render_combined_dashboard(
        self,
        frame,
        inspect_result,
        fps=30.0,
        is_paused=False,
        frame_idx=0,
        total_frames=0,
        sync_realtime=True
    ):
        """
        Merender visualisasi terpadu pada resolusi asli frame kamera:
        1. Gerbang Plat Slat + Garis Celah Terstitching (Terproyeksi Persis di Celah Fisik Plat).
        2. Bounding Box Sirip + Garis Puncak Sirip (Ridge Tracking) menempel di Bilah Fisik.
        3. Live Radar PiP di sudut kanan atas.
        4. Panel Dashboard Bawah (1000 x 300 px) dengan Kartu Metrologi Plat & Sirip Lengkap.
        """
        h_f, w_f = frame.shape[:2]
        if self._screen_buf is None:
            self._screen_buf = np.zeros((820, 1000, 3), dtype=np.uint8)

        # Zero-copy canvas: langsung resize frame ke canvas atas (1000 x 520) tanpa copy 1080p
        main_resized = self._screen_buf[:520, :]
        cv2.resize(frame, (1000, 520), dst=main_resized, interpolation=cv2.INTER_LINEAR)

        # Faktor skala koordinat dari resolusi citra asli (w_f, h_f) ke tampilan (1000, 520)
        sx = 1000.0 / float(w_f)
        sy = 520.0 / float(h_f)

        slat_res = inspect_result.get("slat_res", {})
        rib_res = inspect_result.get("rib_res", {})
        overall_status = inspect_result.get("status", "WAIT")
        is_moving = inspect_result.get("is_conveyor_moving", True)

        # ======================================================================
        # 1. LAYER VISUALISASI PLAT SLAT (PERSIS HOUGH_STITCHING_INSPECTOR.PY)
        # ======================================================================
        if slat_res and "polygon_px" in slat_res:
            poly_pts_int = slat_res["polygon_px"].astype(np.float32)
            poly_s = (poly_pts_int * [sx, sy]).astype(np.int32)
            slat_display_status = slat_res.get("display_status", "WAIT")

            if slat_display_status == "OK":
                poly_color = (0, 230, 0)      # Hijau
            elif slat_display_status == "HOLD":
                poly_color = (0, 220, 255)    # Cyan
            elif slat_display_status == "WAIT":
                poly_color = (0, 165, 255)    # Oranye / Amber
            else:
                poly_color = (0, 30, 255)     # Merah (NG)

            # Gambar Polygon Gerbang Inspeksi Slat (DIBESARKAN)
            slat_conf = float(slat_res.get("confidence", slat_res.get("coverage", 0.0)) or 0.0)
            slat_tag = f"GERBANG INSPEKSI SLAT [Conf: {slat_conf:.2f}]" if slat_conf > 0 else "GERBANG INSPEKSI SLAT"
            cv2.polylines(main_resized, [poly_s], isClosed=True, color=poly_color, thickness=3, lineType=cv2.LINE_AA)
            cv2.putText(main_resized, slat_tag, (poly_s[0][0], max(26, poly_s[0][1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, poly_color, 2, cv2.LINE_AA)

            # Gambar Mini Trigger Zone pada rantai samping
            trigger_box = slat_res.get("trigger_box")
            if trigger_box:
                bx1, by1, bx2, by2 = trigger_box
                tx1, ty1, tx2, ty2 = int(bx1 * sx), int(by1 * sy), int(bx2 * sx), int(by2 * sy)
                trig_flash = self.slat_inspector.trigger_flash_frames > 0
                if trig_flash:
                    self.slat_inspector.trigger_flash_frames -= 1
                    trig_col = (0, 255, 0)
                    trig_thick = 3
                    trig_lbl = "BOLT [TRIGGERED!]"
                elif self.slat_inspector.use_bolt_trigger:
                    trig_col = (0, 220, 255)
                    trig_thick = 2
                    trig_lbl = f"BOLT SENSOR ({slat_res.get('bolt_energy', 0.0):.1f})"
                else:
                    trig_col = (120, 120, 120)
                    trig_thick = 1
                    trig_lbl = "TRIGGER OFF"
                cv2.rectangle(main_resized, (tx1, ty1), (tx2, ty2), trig_col, trig_thick)
                cv2.putText(main_resized, trig_lbl, (tx1, max(18, ty1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.58, trig_col, 2, cv2.LINE_AA)

            # PROYEKSI GARIS HOUGH STITCHING CELAH PLAT TEPAT KE CELAH FISIK (M_inv)
            # Menjamin garis celah duduk tepat di sambungan plat, bukan di langit-langit konveyor!
            xs = slat_res.get("xs_contour")
            ys = slat_res.get("ys_contour")
            if xs is not None and ys is not None and len(xs) > 10:
                try:
                    src_pts = slat_res["polygon_px"].astype(np.float32)
                    dst_pts = np.array([
                        [0, 0],
                        [self.slat_inspector.canonical_w - 1, 0],
                        [self.slat_inspector.canonical_w - 1, self.slat_inspector.canonical_h - 1],
                        [0, self.slat_inspector.canonical_h - 1]
                    ], dtype=np.float32)
                    M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)

                    unwarp_pts = np.column_stack([xs, ys]).reshape(-1, 1, 2).astype(np.float32)
                    cam_pts = cv2.perspectiveTransform(unwarp_pts, M_inv).reshape(-1, 2)
                    cam_pts[:, 0] = np.clip(cam_pts[:, 0], 0, w_f - 1)
                    cam_pts[:, 1] = np.clip(cam_pts[:, 1], 0, h_f - 1)
                    cam_pts_s = (cam_pts * [sx, sy]).astype(np.int32)

                    # Gambar garis kontur celah terproyeksi persis di celah fisik plat
                    seam_color = (0, 255, 0) if slat_display_status == "OK" else ((0, 0, 255) if slat_display_status == "NG" else (0, 220, 255))
                    cv2.polylines(main_resized, [cam_pts_s], isClosed=False, color=seam_color, thickness=2, lineType=cv2.LINE_AA)
                except Exception:
                    pass

        # ======================================================================
        # 2. LAYER VISUALISASI BILAH SIRIP (PERSIS HYBRID_RIB_INSPECTOR.PY)
        # ======================================================================
        if rib_res:
            is_rib_present = rib_res.get("is_rib_present", False)
            rib_stat = rib_res.get("status", "WAIT")
            rib_ok = rib_res.get("is_ok", False)

            # Gambar Bounding Box Sirip jika terdeteksi
            box = rib_res.get("active_box")
            if is_rib_present and box:
                bx1, by1, bx2, by2 = box
                rx1, ry1, rx2, ry2 = int(bx1 * sx), int(by1 * sy), int(bx2 * sx), int(by2 * sy)
                w_px = bx2 - bx1
                h_px = by2 - by1
                scale_x = getattr(self, "scale_factor_mm_per_px", 0.515)
                scale_y = getattr(self, "scale_factor_y_mm_per_px", 0.442)
                w_mm = int(round(w_px * scale_x))
                h_mm = int(round(h_px * scale_y))
                rib_conf = float(rib_res.get("confidence", 0.0) or 0.0)
                conf_lbl = f" | Conf: {rib_conf:.2f}" if rib_conf > 0 else ""
                rib_tag = f"SIRIP AKTIF ({w_mm}x{h_mm} mm{conf_lbl} | {w_px}x{h_px} px)"
                cv2.rectangle(main_resized, (rx1, ry1), (rx2, ry2), (255, 170, 0), 2, cv2.LINE_AA)
                cv2.putText(main_resized, rib_tag, (rx1, max(24, ry1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 170, 0), 2, cv2.LINE_AA)

            # Gambar Garis Puncak Sirip (Ridge Tracking) menempel di bibir bilah fisik
            xs_r = rib_res.get("xs_contour")
            ys_r = rib_res.get("ys_contour")
            if is_rib_present and xs_r is not None and ys_r is not None and len(xs_r) > 10:
                line_color = (0, 255, 0) if rib_ok else (0, 0, 255)
                pts_r = np.column_stack([xs_r, ys_r]).astype(np.float32)
                pts_r_s = (pts_r * [sx, sy]).astype(np.int32)
                cv2.polylines(main_resized, [pts_r_s], isClosed=False, color=line_color, thickness=2, lineType=cv2.LINE_AA)
                mid_idx = len(pts_r_s) // 2
                cv2.circle(main_resized, (int(pts_r_s[mid_idx, 0]), int(pts_r_s[mid_idx, 1])), 4, line_color, -1)

        # ======================================================================
        # 3. LIVE RADAR FILTER PiP DI SUDUT KANAN ATAS (CANNY/SOBEL-Y)
        # ======================================================================
        if slat_res and "edges" in slat_res and slat_res["edges"] is not None:
            try:
                edges_raw = slat_res["edges"]
                pip_w, pip_h = 270, 78
                pip_crop = cv2.resize(edges_raw, (pip_w, pip_h))
                pip_bgr = cv2.cvtColor(pip_crop, cv2.COLOR_GRAY2BGR)

                pip_y_seam = int(slat_res.get("y_seam", 70.0) * (pip_h / float(self.slat_inspector.canonical_h)))
                pip_y_seam = max(0, min(pip_h - 1, pip_y_seam))
                cv2.line(pip_bgr, (0, pip_y_seam), (pip_w - 1, pip_y_seam), (0, 255, 0), 2, cv2.LINE_AA)

                radar_col = (0, 230, 255) if slat_res.get("status") != "NG" else (0, 0, 255)
                cv2.rectangle(pip_bgr, (0, 0), (pip_w - 1, pip_h - 1), radar_col, 2)
                cv2.putText(pip_bgr, f"LIVE RADAR: {slat_res.get('edge_mode', 'Canny')[:16]}", (8, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 230, 255), 1, cv2.LINE_AA)
                cv2.putText(pip_bgr, f"SEAM Y: {slat_res.get('y_seam', 70):.0f}px | Energy: {slat_res.get('seam_energy', 0.0):.1f}",
                            (8, pip_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 120), 1, cv2.LINE_AA)

                main_resized[12:12 + pip_h, 1000 - pip_w - 15:1000 - 15] = pip_bgr
            except Exception:
                pass

        # Indikator Banner PAUSED jika video dijeda
        if is_paused:
            bw, bh = 430, 52
            bx = 15
            by = 12
            cv2.rectangle(main_resized, (bx, by), (bx + bw, by + bh), (20, 110, 220), -1)
            cv2.rectangle(main_resized, (bx, by), (bx + bw, by + bh), (40, 180, 255), 2, cv2.LINE_AA)
            cv2.putText(main_resized, f"|| PAUSED | Frame {frame_idx}/{total_frames}", (bx + 14, by + 24),
                        cv2.FONT_HERSHEY_DUPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(main_resized, "[SPACE]: Play | [N]: +1 Frame | [P]: -1 Frame", (bx + 14, by + 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 240, 200), 1, cv2.LINE_AA)

        # ======================================================================
        # 4. PANEL DASHBOARD BAWAH (1000 x 300 px) DENGAN KARTU METROLOGI GANDA
        # ======================================================================
        hud = self._screen_buf[520:, :]
        hud[:] = (18, 18, 22)

        # Badge Status Utama Pipeline Terpadu
        if overall_status == "OK":
            badge_bg = (0, 160, 0)
        elif overall_status == "NG":
            badge_bg = (0, 0, 200)
        elif overall_status == "HOLD":
            badge_bg = (30, 80, 160)
        else:
            badge_bg = (140, 90, 30)

        cv2.rectangle(hud, (15, 12), (180, 72), badge_bg, -1)
        stat_x = 55 if len(overall_status) == 2 else (35 if len(overall_status) == 4 else 45)
        cv2.putText(hud, overall_status, (stat_x, 56), cv2.FONT_HERSHEY_DUPLEX, 1.35, (255, 255, 255), 3, cv2.LINE_AA)

        # Header Status Terpadu (DIBESARKAN)
        cv2.putText(hud, "CAM-01 TOP INSPECTION PIPELINE (SLAT & RIB INTEGRATED)",
                    (195, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)

        # Badge Status Database Logging (ON / OFF)
        db_on = self.is_db_enabled()
        db_bg = (0, 140, 0) if db_on else (40, 40, 50)
        db_txt = "DB: ON [REC]" if db_on else "DB: OFF [DRY-RUN]"
        db_color = (255, 255, 255) if db_on else (180, 180, 180)
        cv2.rectangle(hud, (760, 10), (985, 42), db_bg, -1)
        cv2.rectangle(hud, (760, 10), (985, 42), (0, 255, 0) if db_on else (90, 90, 100), 1, cv2.LINE_AA)
        cv2.putText(hud, db_txt, (772, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.52, db_color, 2, cv2.LINE_AA)

        slat_tag = slat_res.get("display_status", "WAIT") if slat_res else "WAIT"
        rib_tag = "RIB: " + ("HADIR (" + rib_res.get("status", "WAIT") + ")" if rib_res.get("is_rib_present", False) else "PLAT DATAR")
        dev_tag = getattr(self.rib_inspector, "device_badge", "CUDA" if getattr(self.rib_inspector, "device", "cpu") != "cpu" else "CPU")
        cv2.putText(hud, f"STATUS: [SLAT: {slat_tag}] | [{rib_tag}]  |  FPS: {fps:4.1f}  |  Runtime: {inspect_result.get('elapsed_ms', 0):.1f} ms  |  [{dev_tag}]",
                    (195, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(hud, f"Plat: {slat_res.get('eval_mode', '-')} | Sirip: {rib_res.get('message', '-')}",
                    (195, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 230, 255), 1, cv2.LINE_AA)

        cv2.line(hud, (15, 88), (985, 88), (50, 50, 60), 1)

        # ----------------------------------------------------------------------
        # KARTU METROLOGI KIRI: PLAT SLAT (DIBESARKAN)
        # ----------------------------------------------------------------------
        cv2.rectangle(hud, (15, 96), (485, 184), (28, 28, 36), -1)
        cv2.rectangle(hud, (15, 96), (485, 184), (60, 60, 75), 1)
        cv2.putText(hud, "METROLOGI PLAT SLAT (HOUGH STITCHING)", (25, 118),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 200, 100), 2, cv2.LINE_AA)

        s_pv = slat_res.get("pv_flatness", 0.0) if slat_res else 0.0
        s_sag = slat_res.get("sagitta", 0.0) if slat_res else 0.0
        s_stat = slat_res.get("display_status", "WAIT") if slat_res else "WAIT"

        pv_c = (0, 255, 0) if s_pv <= self.slat_inspector.max_flatness_pv else (0, 0, 255)
        sag_c = (0, 255, 0) if s_sag <= self.slat_inspector.max_sagitta else (0, 0, 255)

        scale = getattr(self, "scale_factor_mm_per_px", 0.515)
        s_pv_mm = s_pv * scale
        max_pv_mm = self.slat_inspector.max_flatness_pv * scale
        s_sag_mm = s_sag * scale
        max_sag_mm = self.slat_inspector.max_sagitta * scale

        cv2.putText(hud, f"Flatness PV : {s_pv_mm:4.1f} mm ({s_pv:.1f} px) [Maks: {max_pv_mm:.1f} mm]",
                    (25, 139), cv2.FONT_HERSHEY_SIMPLEX, 0.46, pv_c, 1, cv2.LINE_AA)
        cv2.putText(hud, f"Sagitta Curv: {s_sag_mm:4.1f} mm ({s_sag:.1f} px) [Maks: {max_sag_mm:.1f} mm]",
                    (25, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.46, sag_c, 1, cv2.LINE_AA)
        slat_conf = float(slat_res.get("confidence", slat_res.get("coverage", 0.0)) or 0.0)
        conf_s_str = f" | Conf: {slat_conf:.2f}" if slat_conf > 0 else ""
        cv2.putText(hud, f"QC PLAT SLAT: #{self.slat_inspector.slat_count:02d}{conf_s_str} | OK: {self.slat_inspector.ok_count} | NG: {self.slat_inspector.ng_count} | [{s_stat}]",
                    (25, 178), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 240, 180), 2, cv2.LINE_AA)

        # ----------------------------------------------------------------------
        # KARTU METROLOGI KANAN: BILAH SIRIP (DIBESARKAN)
        # ----------------------------------------------------------------------
        cv2.rectangle(hud, (495, 96), (985, 184), (28, 28, 36), -1)
        cv2.rectangle(hud, (495, 96), (985, 184), (60, 60, 75), 1)
        cv2.putText(hud, f"METROLOGI BILAH SIRIP (HYBRID: {self.rib_inspector.mode})", (505, 118),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0, 255, 255), 2, cv2.LINE_AA)

        r_pv = rib_res.get("pv_flatness", 0.0) if rib_res else 0.0
        r_sag = rib_res.get("sagitta", 0.0) if rib_res else 0.0
        r_dev = rib_res.get("inner_dev", 0.0) if rib_res else 0.0
        r_tip = rib_res.get("tip_dev", 0.0) if rib_res else 0.0
        r_stat = rib_res.get("status", "WAIT") if rib_res else "WAIT"

        rpv_c = (0, 255, 0) if r_pv <= self.rib_inspector.pv_thresh else (0, 0, 255)
        rsag_c = (0, 255, 0) if r_sag <= self.rib_inspector.sagitta_thresh else (0, 0, 255)
        r_pv_mm = r_pv * scale
        r_sag_mm = r_sag * scale
        r_dev_mm = r_dev * scale
        dev_lim_mm = self.rib_inspector.dev_thresh * scale
        r_tip_mm = r_tip * scale
        tip_lim_mm = self.rib_inspector.tip_thresh * scale

        cv2.putText(hud, f"Flatness PV : {r_pv_mm:4.1f} mm | Sagitta: {r_sag_mm:.1f} mm",
                    (505, 139), cv2.FONT_HERSHEY_SIMPLEX, 0.46, rpv_c, 1, cv2.LINE_AA)
        rdev_c = (0, 255, 0) if r_dev <= self.rib_inspector.dev_thresh else (0, 0, 255)
        rtip_c = (0, 255, 0) if r_tip <= self.rib_inspector.tip_thresh else (0, 0, 255)
        cv2.putText(hud, f"Dev Badan: {r_dev_mm:4.1f}mm [Maks: {dev_lim_mm:.1f}mm] | Tip: {r_tip_mm:4.1f}mm",
                    (505, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.44, rdev_c, 1, cv2.LINE_AA)
        rib_conf = float(rib_res.get("confidence", 0.0) or 0.0)
        conf_r_str = f" | Conf: {rib_conf:.2f}" if rib_conf > 0 else ""
        cv2.putText(hud, f"QC SIRIP RIB: #{self.rib_inspector.rib_count:02d}{conf_r_str} | OK: {self.rib_inspector.ok_count} | NG: {self.rib_inspector.ng_count} | [{r_stat}]",
                    (505, 178), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 240, 180), 2, cv2.LINE_AA)

        # ----------------------------------------------------------------------
        # FOOTER PANDUAN KONTROL & PINTASAN KEYBOARD (DIBESARKAN)
        # ----------------------------------------------------------------------
        cv2.rectangle(hud, (15, 192), (985, 292), (22, 22, 28), -1)
        cv2.rectangle(hud, (15, 192), (985, 292), (45, 45, 55), 1)
        cv2.putText(hud, "PANDUAN KONTROL & PINTASAN KEYBOARD (HOTKEYS):", (25, 214),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 180, 220), 2, cv2.LINE_AA)
        cv2.putText(hud, "[SPACE]: Pause/Play   [N]: +1 Maju   [P]: -1 Mundur   [R]: Reset Counter", (25, 238),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(hud, "[W]/[O]/[F]: Media   [V]: Ganti Video   [T]: Turbo   [D]: DB Log   [Q]: Keluar", (25, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(hud, f"File: {inspect_result.get('filename', '-')} | Posisi: {frame_idx}/{total_frames}", (25, 282),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (140, 210, 140), 1, cv2.LINE_AA)

        # Selesai: kembalikan buffer terpadu yang sudah terisi penuh (zero-copy)
        return self._screen_buf


def main():
    parser = argparse.ArgumentParser(description="Master Pipeline Inspeksi Tampak Atas (Slat & Rib Conveyor)")
    parser.add_argument("--source", type=str, default=None, help="Path ke file video atau foto")
    parser.add_argument("--mode", type=str, default="HYBRID", choices=["HYBRID", "YOLO_ONLY", "DUAL_POLARITY_ONLY"],
                        help="Mode operasional detektor sirip")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device komputasi AI ('auto', 'cuda', '0', atau 'cpu')")
    parser.add_argument("--config", type=str, default=None, help="Path ke file config JSON")
    parser.add_argument("--enable-db", action="store_true", default=False,
                        help="Aktifkan pencatatan SQLite database dan penyimpanan captures secara langsung (Default: OFF / Dry-Run)")
    args = parser.parse_args()

    print("\n" + "=" * 78)
    print("🚀 MEMULAI PIPELINE TERPADU INSPEKSI TAMPAK ATAS KONVEYOR (CAM-03)")
    print("=" * 78)

    # Inisialisasi Master Orchestrator
    pipeline = CombinedTopInspector(
        config_path=args.config,
        rib_mode=args.mode,
        device=args.device,
        enable_db=args.enable_db
    )

    print("Modul yang Berjalan Bersamaan (Concurrent):")
    print("  1. HoughStitchingInspector -> hough_stitching_inspector.py (Gerbang Plat Slat - CPU)")
    print(f"  2. HybridRibInspector     -> hybrid_rib_inspector.py (Zona Deteksi Sirip - {pipeline.rib_inspector.device_info})")
    print("=" * 78)

    # Cek sumber dari argumen atau dari .env
    current_source = args.source
    if not current_source:
        env_top = os.environ.get("TOP_CAMERA_SOURCE")
        if env_top:
            if env_top.isdigit() or env_top.startswith(("rtsp://", "http://", "https://", "nvarguscamerasrc")):
                current_source = env_top
            elif os.path.exists(env_top):
                current_source = os.path.abspath(env_top)

    if not current_source:
        # Direktori pencarian video otomatis jika source tidak dispesifikasi
        search_dirs = [
            os.path.join(os.path.dirname(CURRENT_DIR), "Video_Testing"),
            os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "Video_Testing")),
            os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "..", "Video_Testing")),
            os.path.join(os.path.expanduser("~"), "Videos"),
            os.path.join(os.path.expanduser("~"), "Pictures"),
        ]
        sorted_videos = []
        for sdir in search_dirs:
            if os.path.exists(sdir):
                for f in sorted(os.listdir(sdir)):
                    if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                        p = os.path.abspath(os.path.join(sdir, f))
                        if p not in sorted_videos:
                            sorted_videos.append(p)

        if sorted_videos:
            current_source = sorted_videos[0]

    if not current_source or (not str(current_source).isdigit() and not str(current_source).startswith(("rtsp://", "http://")) and not os.path.exists(str(current_source))):
        print("\n[*] Tidak ada path video default, membuka File Dialog Explorer...")
        current_source = select_file_dialog(title="Pilih File Video untuk Inspeksi Terpadu")

    if not current_source:
        print("[ERROR] Tidak ada file video yang valid. Keluar.")
        return

    # Untuk sumber video: gunakan AsyncFrameGrabber agar frame decode non-blocking
    grabber = None
    cap = None

    # Deteksi apakah sumber adalah gambar statis atau video
    _cap_probe, is_image, static_frame, total_frames, fname = open_media_source(current_source)
    if _cap_probe is not None:
        _cap_probe.release()  # tutup probe segera; grabber akan membuka ulang

    if is_image:
        # Sumber gambar statis — tidak perlu grabber
        pass
    else:
        # Sumber video — gunakan AsyncFrameGrabber (background decode thread)
        grabber = AsyncFrameGrabber(current_source, buffer_size=2)
        if not grabber.isOpened():
            print(f"[ERROR] Gagal membuka sumber media: {current_source}")
            grabber.release()
            return
        total_frames = grabber.total_frames
        print(f"[✔] AsyncFrameGrabber aktif: frame decode non-blocking di background thread")


    win_name = "CAM-03 TOP INSPECTION PIPELINE (SLAT & RIB INTEGRATED)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1200, 920)

    is_paused = False
    step_forward = False
    step_backward = False
    cached_frame = None
    fps_val = 30.0
    sync_realtime = True
    current_vid_idx = sorted_videos.index(current_source) if current_source in sorted_videos else 0
    t_last_frame = time.time()
    last_video_pos = 0

    print(f"\n[OK] Memulai inspeksi pada media: {fname} (Total Frame: {total_frames})")
    print("Tekan [SPACE] untuk Pause/Play | [T] Toggle Realtime/Turbo | [Q] untuk Keluar\n")

    while True:
        t_loop_start = time.time()
        frame = None

        if is_image:
            # Sumber gambar statis
            frame = static_frame.copy()
            cur_pos = 1
        elif is_paused:
            # Mode Pause: frame sudah ter-cache, hanya step forward/backward yang membaca baru
            if step_forward:
                ret, frame = grabber.read()
                if ret and frame is not None:
                    cached_frame = frame.copy()
                step_forward = False
            elif step_backward:
                target_pos = max(0, grabber.get_pos_frames() - 2)
                grabber.set_pos_frames(target_pos)
                time.sleep(0.02)
                ret, frame = grabber.read()
                if ret and frame is not None:
                    cached_frame = frame.copy()
                step_backward = False
            else:
                if cached_frame is None and grabber is not None:
                    ret, fr = grabber.read()
                    if ret and fr is not None:
                        cached_frame = fr.copy()
                frame = cached_frame.copy() if cached_frame is not None else None
            cur_pos = grabber.get_pos_frames() if grabber else 1
        else:
            # Mode Running: ambil frame pre-decoded dari buffer (non-blocking)
            ret, frame = grabber.read()
            if not ret or frame is None:
                k = cv2.waitKey(5) & 0xFF
                if k in [ord('q'), ord('Q'), 27]:
                    break
                elif k in [ord('d'), ord('D')]:
                    pipeline.toggle_db()
                continue
            cur_pos = grabber.get_pos_frames()
            if cur_pos < last_video_pos - 15:
                # Video loop kembali ke frame awal: reset cache deduplikasi
                pipeline.logged_slat_id = None
                pipeline.logged_rib_id = None
                if pipeline.db_bridge is not None and hasattr(pipeline.db_bridge, "reset_entity_cache"):
                    pipeline.db_bridge.reset_entity_cache()
            last_video_pos = cur_pos
            cached_frame = frame.copy()

        if frame is None:
            # Cegah UI hang / 'Not Responding' dengan tetap memproses window message loop OpenCV
            k = cv2.waitKey(20) & 0xFF
            if k in [ord('q'), ord('Q'), 27]:
                print("\n[*] Menutup pipeline inspeksi...")
                break
            elif k == 32:  # [SPACE]
                is_paused = not is_paused
                print(f"[*] Status: {'PAUSED' if is_paused else 'RUNNING'}")
            elif k in [ord('d'), ord('D')]:
                st = pipeline.toggle_db()
                mode_str = "AKTIF (RECORDING)" if st else "MATI (DRY-RUN)"
                print(f"[*] SAKLAR DATABASE LOGGING: {mode_str}")
            continue

        # Jalankan KEDUA inspeksi secara bebarengan
        inspect_res = pipeline.inspect(frame)
        inspect_res["filename"] = fname

        # Hitung FPS real-time berdasarkan interval antar-frame aktual
        t_now = time.time()
        dt_frame = t_now - t_last_frame
        t_last_frame = t_now
        if dt_frame > 0:
            fps_val = 0.9 * fps_val + 0.1 * (1.0 / dt_frame)

        # Render antarmuka dashboard visual
        dashboard_screen = pipeline.render_combined_dashboard(
            frame,
            inspect_res,
            fps=fps_val,
            is_paused=is_paused,
            frame_idx=cur_pos,
            total_frames=total_frames,
            sync_realtime=sync_realtime
        )

        # Rekam ke database SQLite dan folder captures jika ada NG dan DB Logging aktif (Single-Capture per Objek)
        if inspect_res.get("status") == "NG":
            pipeline.record_if_ng(
                frame_raw=frame,
                frame_ann=dashboard_screen,
                inspect_res=inspect_res,
                slat_no=getattr(pipeline.slat_inspector, "slat_count", None),
                rib_no=getattr(pipeline.rib_inspector, "rib_count", None)
            )

        cv2.imshow(win_name, dashboard_screen)

        # Tangani interaksi keyboard & pacing kecepatan playback video secara presisi
        t_work = time.time() - t_loop_start
        if is_paused:
            delay = 30
        elif sync_realtime and grabber is not None and not grabber.is_live:
            # Pacing agar kecepatan video tepat 1.0x (sesuai FPS asli rekaman video)
            target_dt = 1.0 / grabber.fps_native
            remaining_ms = int(max(1, (target_dt - t_work) * 1000.0))
            delay = remaining_ms
        else:
            delay = 1

        key = cv2.waitKey(delay) & 0xFF

        if key in [ord('q'), ord('Q'), 27]:
            print("\n[*] Menutup pipeline inspeksi...")
            break
        elif key == 32:  # [SPACE]
            is_paused = not is_paused
            print(f"[*] Status: {'PAUSED' if is_paused else 'RUNNING'}")
        elif key in [ord('t'), ord('T')]:
            sync_realtime = not sync_realtime
            sync_mode = "1.0x Realtime Sync" if sync_realtime else "TURBO (Uncapped FPS)"
            print(f"[*] Mode Kecepatan Video: {sync_mode}")
        elif key in [ord('n'), ord('N')]:
            if is_paused:
                step_forward = True
        elif key in [ord('d'), ord('D')]:
            st = pipeline.toggle_db()
            mode_str = "AKTIF (RECORDING)" if st else "MATI (DRY-RUN)"
            print(f"[*] SAKLAR DATABASE LOGGING: {mode_str}")
        elif key in [ord('p'), ord('P'), ord('a'), ord('A')]:
            if is_paused:
                step_backward = True
        elif key in [ord('r'), ord('R')]:
            pipeline.reset_counters()
        elif key in [ord('w'), ord('W'), ord('o'), ord('O'), ord('f'), ord('F')]:
            # Buka dialog explorer
            print("\n[*] Membuka File Dialog Explorer...")
            new_file = select_file_dialog(
                initial_dir=os.path.dirname(current_source),
                title="Pilih File Video / Foto Baru untuk Inspeksi Terpadu"
            )
            cv2.waitKey(1)  # Refresh OpenCV window focus setelah Tkinter
            if new_file and os.path.exists(new_file):
                if grabber is not None:
                    grabber.release()
                    grabber = None
                new_cap, new_is_img, new_static, new_tot, new_fn = open_media_source(new_file)
                cached_frame = None
                if new_is_img:
                    is_image = True
                    static_frame = new_static
                    total_frames = 1
                    cached_frame = static_frame.copy()
                elif new_cap is not None and new_cap.isOpened():
                    new_cap.release()
                    grabber = AsyncFrameGrabber(new_file, buffer_size=2)
                    is_image = False
                    total_frames = grabber.total_frames
                    # Segera ambil frame pertama agar cached_frame langsung siap
                    t_w0 = time.time()
                    while time.time() - t_w0 < 1.0:
                        ret_f0, f0 = grabber.read()
                        if ret_f0 and f0 is not None:
                            cached_frame = f0.copy()
                            break
                        time.sleep(0.01)
                current_source = new_file
                fname = new_fn
                pipeline.reset_counters()
                if new_file not in sorted_videos:
                    sorted_videos.append(new_file)
                current_vid_idx = sorted_videos.index(new_file)
                print(f"[OK] Berhasil memuat media: {fname}")
        elif key in [ord('v'), ord('V')]:
            if sorted_videos:
                current_vid_idx = (current_vid_idx + 1) % len(sorted_videos)
                new_file = sorted_videos[current_vid_idx]
                if grabber is not None:
                    grabber.release()
                    grabber = None
                new_cap, new_is_img, new_static, new_tot, new_fn = open_media_source(new_file)
                cached_frame = None
                if new_is_img:
                    is_image = True
                    static_frame = new_static
                    total_frames = 1
                    cached_frame = static_frame.copy()
                elif new_cap is not None and new_cap.isOpened():
                    new_cap.release()
                    grabber = AsyncFrameGrabber(new_file, buffer_size=2)
                    is_image = False
                    total_frames = grabber.total_frames
                    # Segera ambil frame pertama agar cached_frame langsung siap
                    t_w0 = time.time()
                    while time.time() - t_w0 < 1.0:
                        ret_f0, f0 = grabber.read()
                        if ret_f0 and f0 is not None:
                            cached_frame = f0.copy()
                            break
                        time.sleep(0.01)
                current_source = new_file
                fname = new_fn
                pipeline.reset_counters()
                print(f"\n[OK] Beralih ke media: {fname}")

    pipeline.close()
    if grabber is not None:
        grabber.release()
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.winmm.timeEndPeriod(1)
        except Exception:
            pass


if __name__ == "__main__":
    main()
