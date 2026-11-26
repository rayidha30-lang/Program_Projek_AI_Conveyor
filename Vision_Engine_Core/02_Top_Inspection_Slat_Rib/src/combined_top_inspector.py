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
import subprocess
import threading
import queue
import cv2
import numpy as np
import math
from datetime import datetime

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
from conveyor_motion_calibrator import ConveyorMotionCalibrator
from frame_health_monitor import FrameHealthMonitor
from camera_pose_watchdog import CameraPoseWatchdog

# Impor Modul Jembatan Database SQLite dari root Program/ (fallback ke Program/Database)
PROGRAM_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
for _db_cand in (PROGRAM_DIR, os.path.join(PROGRAM_DIR, "Database")):
    if os.path.exists(os.path.join(_db_cand, "db_bridge.py")):
        if _db_cand not in sys.path:
            sys.path.insert(0, _db_cand)
        break

try:
    from db_bridge import InspectionDBBridge
except ImportError:
    InspectionDBBridge = None

from video_nav_utils import (
    SPEED_STEPS,
    IdleConveyorSkipper,
    compute_seek_target_from_key,
    format_seconds_mmss,
    format_speed_label,
    next_speed_step,
    parse_time_to_seconds,
    prompt_jump_time_dialog,
)


class AsyncFrameGrabber:
    """
    Background thread untuk cv2.VideoCapture non-blocking pre-decode.
    
    Arsitektur Produsen-Konsumen (Producer-Consumer):
    - Untuk FILE VIDEO (Offline/Testing):
      Menyimpan antrean frame (buffer_size=2) TANPA membuang frame (ZERO DROPPED FRAMES) pada speed 1x.
      Mendukung fast-forward speed multiplier, seek berbasis epoch (bebas stale frame), dan fast-grab skip.
    - Untuk KAMERA LIVE (RTSP/USB Stream):
      Membuang frame lama jika buffer penuh agar visualisasi selalu real-time dengan latensi terendah.
    """

    def __init__(self, source_path, buffer_size=2):
        self.source_path = source_path
        self.is_live = False
        src_str = str(source_path).strip()
        if src_str.isdigit() or src_str.startswith(("rtsp://", "http://", "https://", "/dev/video", "nvarguscamerasrc", "v4l2src")):
            self.is_live = True

        self.cap = cv2.VideoCapture(int(source_path) if src_str.isdigit() else source_path)
        self._buf = queue.Queue(maxsize=buffer_size)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._seek_target = None
        self._seek_epoch = 0
        self._speed_mult = 1
        self._current_pos = 0

        # Ambil FPS native dan total frame dari video
        self._native_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not self._native_fps or self._native_fps <= 0 or self._native_fps > 120:
            self._native_fps = 30.0

        self._total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not self.is_live else 0

        self._thread = threading.Thread(target=self._grab_loop, daemon=True, name="FrameGrabber")
        self._thread.start()

    def _drain_buffer_locked(self):
        while not self._buf.empty():
            try:
                self._buf.get_nowait()
            except queue.Empty:
                break

    def _clamp_pos(self, pos: int) -> int:
        if self._total_frames > 0:
            return max(0, min(self._total_frames - 1, int(pos)))
        return max(0, int(pos))

    def set_speed_multiplier(self, speed_mult: int) -> int:
        """Atur pengali kecepatan baca file video (1x..16x)."""
        val = max(1, int(speed_mult))
        with self._lock:
            self._speed_mult = val
        return val

    def _grab_loop(self):
        while not self._stop.is_set():
            did_explicit_seek = False
            # 1. Cek permintaan seek
            with self._lock:
                my_epoch = self._seek_epoch
                speed_mult = self._speed_mult
                if self._seek_target is not None:
                    target = self._seek_target
                    self._seek_target = None
                    self._drain_buffer_locked()
                    if not self.is_live:
                        cap_cur = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                        delta_fwd = target - cap_cur
                        if 0 < delta_fwd <= 75:
                            for _ in range(delta_fwd):
                                if not self.cap.grab():
                                    break
                        elif delta_fwd != 0:
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                    did_explicit_seek = True

            # 2. Untuk file video: jika buffer penuh, tunggu main loop (JANGAN BACA & JANGAN BUANG FRAME!)
            # Ini kunci vital agar video TIDAK speed up atau melompati bilah slat/sirip pada speed 1x.
            if not self.is_live and self._buf.full():
                time.sleep(0.003)
                continue

            # Fast-forward frame skipping saat speed > 1x pada pemutaran normal
            if not self.is_live and speed_mult > 1 and not did_explicit_seek:
                for _ in range(speed_mult - 1):
                    if not self.cap.grab():
                        break

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

            # Pastikan tidak menaruh frame basi sebelum seek ke antrean buffer
            with self._lock:
                if my_epoch != self._seek_epoch or self._seek_target is not None:
                    continue

            # 3. Masukkan ke antrean buffer
            item = (my_epoch, pos, frame)
            if self.is_live:
                if self._buf.full():
                    try:
                        self._buf.get_nowait()
                    except queue.Empty:
                        pass
                try:
                    self._buf.put(item, timeout=0.02)
                except queue.Full:
                    pass
            else:
                while not self._stop.is_set():
                    with self._lock:
                        if my_epoch != self._seek_epoch or self._seek_target is not None:
                            break
                    try:
                        self._buf.put(item, timeout=0.03)
                        break
                    except queue.Full:
                        continue

    def read(self, timeout: float = 0.25):
        """Ambil frame berikutnya dari antrean buffer sesuai epoch seek terbaru."""
        deadline = time.time() + max(0.02, float(timeout))
        while time.time() < deadline and not self._stop.is_set():
            rem = max(0.01, deadline - time.time())
            try:
                ep, pos, frame = self._buf.get(timeout=min(0.05, rem))
            except queue.Empty:
                continue
            with self._lock:
                cur_ep = self._seek_epoch
            if ep != cur_ep:
                continue
            self._current_pos = pos
            return True, frame
        return False, None

    def get_pos_frames(self):
        """Posisi nomor frame yang sedang diproses di layar."""
        return self._current_pos

    def set_pos_frames(self, pos):
        """Seek video ke nomor frame tertentu dengan epoch invalidation."""
        target = self._clamp_pos(pos)
        with self._lock:
            self._seek_epoch += 1
            self._seek_target = target
            self._current_pos = target
            self._drain_buffer_locked()
            return self._seek_epoch

    def read_after_seek(self, pos: int, timeout: float = 1.5):
        """Seek ke target pos dan tunggu hingga frame segar pasca-seek tersedia."""
        req_ep = self.set_pos_frames(pos)
        deadline = time.time() + max(0.1, float(timeout))
        while time.time() < deadline and not self._stop.is_set():
            rem = max(0.01, deadline - time.time())
            try:
                ep, got_pos, frame = self._buf.get(timeout=min(0.05, rem))
            except queue.Empty:
                continue
            if ep != req_ep:
                continue
            self._current_pos = got_pos
            return True, frame
        return False, None

    def release(self):
        self._stop.set()
        with self._lock:
            self._drain_buffer_locked()
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
        enable_db=False,
        force_default_on_start=False
    ):
        base_dir = os.path.dirname(CURRENT_DIR)
        if config_path is None:
            config_path = os.path.join(base_dir, "configs", "top_inspection_config.json")
        self.config_path = config_path
        self.default_config_path = os.path.join(os.path.dirname(config_path), "top_inspection_config.default.json")
        self.calibrated_config_path = os.path.join(os.path.dirname(config_path), "top_inspection_config.calibrated.json")
        self.active_profile_name = "DEFAULT_STABIL"
        self.profile_banner_text = ""
        self.profile_banner_timer = 0
        self.rect_matrix = None
        self._rect_calib_size = (1920, 1080)
        self._scaled_rect_cache = {}
        self.motion_calibrator = None
        self.device = device

        # Geometri Acuan Emas (Golden Reference - Trapesium Perspektif Posisi 3)
        self.ref_left_rail_px = 188.0
        self.ref_right_rail_px = 1759.0
        self.ref_center_x = 960.0
        self.ref_tilt_deg = 0.90
        self.ref_width_px = 1571.0
        self.ref_y_top_px = 216.0
        self.ref_y_bot_px = 864.0
        self.ref_top_left_rail_px = 278.0
        self.ref_top_right_rail_px = 1630.0
        self.ref_bot_left_rail_px = 104.0
        self.ref_bot_right_rail_px = 1808.0
        self.show_alignment_guides = False  # Sakelar Garis Panduan [G]

        # Inisialisasi Database Bridge (Default: False / Dry Run) & Konfigurasi TOP_CAMERA_ID Satu Pintu
        raw_env_cam = os.environ.get("TOP_CAMERA_ID", "CAM-01").strip().upper()
        self.top_camera_id = raw_env_cam if raw_env_cam in ("CAM-01", "CAM-02", "CAM-03") else "CAM-01"
        self.db_bridge = None
        if InspectionDBBridge is not None:
            self.db_bridge = InspectionDBBridge(enabled=enable_db, camera_id=self.top_camera_id)

        # 1. Parameter Default Plat Slat
        calib_mll = 0.70
        calib_v = 116.0
        max_pv_slat = 16.5
        max_sag_slat = 8.6
        max_void_slat = 1450
        min_cov_slat = 0.25

        # 2. Parameter Default Sirip (Rib)
        rib_pv = 11.4
        rib_sag = 11.5
        rib_dev = 8.0
        rib_tip = 7.0
        rib_conf = 0.55
        rib_sev_zones = None

        # Polygon cache
        self.slat_polygon = None
        self.rib_zone = None

        # Muat konfigurasi awal: prioritaskan profil DEFAULT_STABIL saat startup fresh
        cfg_to_load = None
        loaded_path = config_path
        if force_default_on_start and os.path.exists(self.default_config_path):
            try:
                with open(self.default_config_path, "r") as f:
                    cfg_to_load = json.load(f)
                loaded_path = self.default_config_path
                self.active_profile_name = "DEFAULT_STABIL"
            except Exception:
                pass

        if cfg_to_load is None and config_path and os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    cfg_to_load = json.load(f)
                loaded_path = config_path
                if "profile_name" in cfg_to_load:
                    self.active_profile_name = cfg_to_load["profile_name"]
            except Exception:
                pass

        if cfg_to_load is not None:
            try:
                cfg = cfg_to_load
                if "profile_name" in cfg:
                    self.active_profile_name = cfg["profile_name"]

                if "rectification_matrix" in cfg and cfg["rectification_matrix"] is not None:
                    self.rect_matrix = np.array(cfg["rectification_matrix"], dtype=np.float32)
                else:
                    self.rect_matrix = None
                self._rect_calib_size = self._infer_calib_frame_size(cfg)
                self._scaled_rect_cache.clear()

                if "calibration_profile" in cfg:
                    cp = cfg["calibration_profile"]
                    calib_mll = float(cp.get("calibrated_mll_ratio", calib_mll))
                    calib_v = float(cp.get("recommended_v_thresh", calib_v))
                    self._apply_golden_reference(cp)

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
                rib_sev_zones = th.get("rib_severity_zones", None)
                self.config_rib_prime_min = th.get("rib_prime_min_y", None)
                self.config_rib_prime_max = th.get("rib_prime_max_y", None)

                if "rib_polygon" in cfg and len(cfg["rib_polygon"]) >= 4:
                    self.rib_zone = cfg["rib_polygon"]

                if "slat_polygon" in cfg and len(cfg["slat_polygon"]) >= 4:
                    self.slat_polygon = cfg["slat_polygon"]

                print(f"[*] Konfigurasi Terpadu dimuat dari: {os.path.basename(loaded_path)} [PROFIL: {self.active_profile_name}]")
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
        if rib_sev_zones:
            self.rib_inspector._severity_zones = rib_sev_zones

        # Fase 2: Muat adaptive threshold dari Rib_Dataset jika db_bridge sudah tersedia,
        # difilter per profil kamera agar dataset posisi berbeda tidak tercampur
        if self.db_bridge is not None and hasattr(self.db_bridge, "db_path"):
            fb_prof = "DEFAULT_STABIL" if self.active_profile_name == "HASIL_KALIBRASI" else None
            self.rib_inspector.load_adaptive_thresholds(
                db_path=self.db_bridge.db_path,
                camera_profile=self.active_profile_name or None,
                fallback_profile=fb_prof,
            )

        # Konfigurasi Prime Zone Sirip (Adaptif & Fleksibel)
        cfg_pmin = getattr(self, "config_rib_prime_min", None)
        cfg_pmax = getattr(self, "config_rib_prime_max", None)
        if cfg_pmin is not None and cfg_pmax is not None:
            self.rib_inspector.set_prime_zone(cfg_pmin, cfg_pmax)
        else:
            self.rib_inspector.set_prime_zone(355, 435)

        # Sumber threshold aktif: default "DATASET" (Rib_Dataset adaptive), tekan [H] untuk beralih ke "WEB"
        self.threshold_source_mode = "DATASET"
        self._config_mtime = os.path.getmtime(config_path) if (config_path and os.path.exists(config_path)) else 0.0
        self._last_hot_reload_check = 0.0

        # Variabel State Kalibrasi Cepat Konveyor (Quick Re-Alignment)
        self.is_calibrating = False
        self.calib_frames_target = 60
        self.calib_buffer = []
        self.calib_banner_timer = 0
        self.calib_status_msg = ""
        self._last_fps = 30.0

        # State Early-Catch & Peak-Hold Latch (Proteksi Oklusi Plat Sebelum Sirip)
        self.slat_history_samples = []
        self.latched_slat_data = None
        self.latched_hold_frames = 0
        self._slat_frame_idx = 0
        self._last_rib_slat_occl_frame = -100

        # Single-Capture Per-Object Latch & Best-Shot Peak Buffer (Terpisah Slat vs Rib)
        self.logged_slat_id = None
        self.logged_rib_id = None
        self._best_slat_candidate = None
        self._best_rib_candidate = None
        self._current_slat_cycle_id = None
        self._current_rib_cycle_id = None
        self._last_emitted_slat_id = None
        self._last_emitted_rib_id = None

        # Thread synchronization locks untuk inspeksi & visualisasi aman (No Freeze)
        self._inspect_lock = threading.RLock()
        self._render_lock = threading.Lock()

        # Resolusi citra nominal & cache batas vertikal gerbang plat slat (non-None float)
        self._last_w_f = 1920
        self._last_h_f = 1080
        self.slat_y_min = 0.50 * 1080.0
        self.slat_y_max = 0.70 * 1080.0
        self._update_slat_y_bounds()

        # Pre-allocated zero-copy canvas buffer (1000 x 820 px: 520 view + 300 HUD)
        self._screen_buf = np.zeros((820, 1000, 3), dtype=np.uint8)

        # Fase 4: GPU Stage-0 Frame Health + cuFFT Auto-Direction & GPU/ArUco Camera Pose Watchdog (async agar bebas spike)
        self.health_monitor = FrameHealthMonitor(device=device)
        self.pose_watchdog = CameraPoseWatchdog(device=device, async_eval=True)
        if cfg_to_load is not None and isinstance(cfg_to_load, dict):
            cp_init = cfg_to_load.get("calibration_profile", {}) or {}
            self.pose_watchdog.set_aruco_reference(
                cp_init.get("aruco_landmarks"),
                calib_w=self._rect_calib_size[0],
                calib_h=self._rect_calib_size[1],
            )
            self.total_slats_in_loop = max(1, int(cfg_to_load.get("thresholds", {}).get("total_slats_in_loop", 16)))
        else:
            self.total_slats_in_loop = 16
        self._loop_anomaly_memory = {}
        self._last_tracked_rib_loop_id = None
        self._last_tracked_slat_loop_id = None

    def _apply_golden_reference(self, cp):
        """Memuat koordinat acuan emas (termasuk trapesium rel Posisi 3) dari calibration_profile."""
        gref = cp.get("golden_reference", {})
        self.ref_left_rail_px = float(gref.get("left_rail_px", cp.get("left_rail_px", self.ref_left_rail_px)))
        self.ref_right_rail_px = float(gref.get("right_rail_px", cp.get("right_rail_px", self.ref_right_rail_px)))
        self.ref_center_x = float(gref.get("target_cx", cp.get("conveyor_center_x", self.ref_center_x)))
        self.ref_tilt_deg = float(gref.get("target_tilt_deg", cp.get("bracket_tilt_deg", self.ref_tilt_deg)))
        self.ref_width_px = float(gref.get("conveyor_width_px", self.ref_right_rail_px - self.ref_left_rail_px))
        self.ref_y_top_px = float(gref.get("top_y_px", self.ref_y_top_px))
        self.ref_y_bot_px = float(gref.get("bot_y_px", self.ref_y_bot_px))
        self.ref_top_left_rail_px = float(gref.get("top_left_rail_px", self.ref_top_left_rail_px))
        self.ref_top_right_rail_px = float(gref.get("top_right_rail_px", self.ref_top_right_rail_px))
        self.ref_bot_left_rail_px = float(gref.get("bot_left_rail_px", self.ref_bot_left_rail_px))
        self.ref_bot_right_rail_px = float(gref.get("bot_right_rail_px", self.ref_bot_right_rail_px))
        if hasattr(self, "pose_watchdog") and self.pose_watchdog is not None:
            cw = int(cp.get("calib_frame_w", self._rect_calib_size[0]))
            ch = int(cp.get("calib_frame_h", self._rect_calib_size[1]))
            self.pose_watchdog.set_aruco_reference(cp.get("aruco_landmarks"), calib_w=cw, calib_h=ch)

    def _golden_rail_x(self, side, y_val, w_f=1920, h_f=1080):
        """Interpolasi linier X rel Acuan Emas pada tinggi y_val di koordinat frame."""
        sx_f = float(w_f) / 1920.0
        sy_f = float(h_f) / 1080.0
        y_top_s = self.ref_y_top_px * sy_f
        y_bot_s = self.ref_y_bot_px * sy_f
        denom = max(1e-5, y_bot_s - y_top_s)
        alpha = (float(y_val) - y_top_s) / denom
        if side == "left":
            x_top = self.ref_top_left_rail_px * sx_f
            x_bot = self.ref_bot_left_rail_px * sx_f
        else:
            x_top = self.ref_top_right_rail_px * sx_f
            x_bot = self.ref_bot_right_rail_px * sx_f
        return float(x_top + alpha * (x_bot - x_top))

    def _update_slat_y_bounds(self, w_f=None, h_f=None):
        """Menghitung batas vertikal min/max gerbang slat secara aman tanpa pernah None."""
        if w_f is not None and w_f > 0:
            self._last_w_f = int(w_f)
        if h_f is not None and h_f > 0:
            self._last_h_f = int(h_f)
        h = float(getattr(self, "_last_h_f", 1080))
        if self.slat_polygon and len(self.slat_polygon) >= 4:
            pts_y = [p[1] * h if p[1] <= 1.0 else p[1] for p in self.slat_polygon]
            self.slat_y_min = float(min(pts_y))
            self.slat_y_max = float(max(pts_y))
        else:
            self.slat_y_min = 0.50 * h
            self.slat_y_max = 0.70 * h

    @staticmethod
    def _infer_calib_frame_size(cfg):
        """Ambil atau simpulkan dimensi frame (w, h) asal tempat rectification_matrix dikalibrasi."""
        cp = (cfg or {}).get("calibration_profile", {}) if isinstance(cfg, dict) else {}
        cw = cp.get("calib_frame_w")
        ch = cp.get("calib_frame_h")
        try:
            if cw is not None and ch is not None and int(cw) > 0 and int(ch) > 0:
                return int(cw), int(ch)
        except (TypeError, ValueError):
            pass
        # Inferensi profil kalibrasi historis 720p (left_rail_pts Y <= 620 & center_x <= 780)
        lpts = cp.get("left_rail_pts")
        cx = cp.get("conveyor_center_x")
        try:
            if isinstance(lpts, list) and len(lpts) >= 2:
                max_y = max(float(pt[1]) for pt in lpts)
                if max_y <= 620.0 and (cx is None or float(cx) <= 780.0):
                    return 1280, 720
        except (TypeError, ValueError, IndexError):
            pass
        return 1920, 1080

    def _get_scaled_rect_matrix(self, w_f, h_f):
        """Kembalikan rectification_matrix terskala via S @ H @ S^-1 jika resolusi frame berbeda dari kalibrasi."""
        M = getattr(self, "rect_matrix", None)
        if M is None:
            return None
        cw, ch = getattr(self, "_rect_calib_size", (1920, 1080))
        wf_i, hf_i = int(w_f), int(h_f)
        if wf_i <= 0 or hf_i <= 0 or cw <= 0 or ch <= 0 or (wf_i == int(cw) and hf_i == int(ch)):
            return M
        cache_key = (wf_i, hf_i)
        cached = self._scaled_rect_cache.get(cache_key)
        if cached is not None:
            return cached
        sx = float(wf_i) / float(cw)
        sy = float(hf_i) / float(ch)
        S = np.diag([sx, sy, 1.0])
        S_inv = np.diag([1.0 / sx, 1.0 / sy, 1.0])
        if M.shape == (3, 3):
            M_scaled = (S @ M.astype(np.float64) @ S_inv).astype(np.float32)
        else:
            M3 = np.eye(3, dtype=np.float64)
            M3[:2, :] = M.astype(np.float64)
            M_scaled = (S @ M3 @ S_inv)[:2, :].astype(np.float32)
        self._scaled_rect_cache[cache_key] = M_scaled
        return M_scaled

    def reset_counters(self):
        """Mereset seluruh statistik QC (Slat dan Rib) ke nilai 0."""
        with self._inspect_lock:
            self.slat_inspector.reset_counters()
            self.rib_inspector.reset_counters()
            self.slat_history_samples.clear()
            self.latched_slat_data = None
            self.latched_hold_frames = 0
            self._slat_frame_idx = 0
            self._last_rib_slat_occl_frame = -100
            self.logged_slat_id = None
            self.logged_rib_id = None
            self._best_slat_candidate = None
            self._best_rib_candidate = None
            self._current_slat_cycle_id = None
            self._current_rib_cycle_id = None
            self._last_emitted_slat_id = None
            self._last_emitted_rib_id = None
            self._loop_anomaly_memory.clear()
            self._last_tracked_rib_loop_id = None
            self._last_tracked_slat_loop_id = None
            if hasattr(self, "health_monitor") and self.health_monitor is not None:
                self.health_monitor.reset()
            if hasattr(self, "pose_watchdog") and self.pose_watchdog is not None:
                self.pose_watchdog.reset_reference()
            if self.db_bridge is not None and hasattr(self.db_bridge, "reset_entity_cache"):
                self.db_bridge.reset_entity_cache()

    def reset_active_tracking(self):
        """Mereset state pelacakan aktif in-flight saja (saat seek video) tanpa menghapus counter QC."""
        with self._inspect_lock:
            if hasattr(self.rib_inspector, "reset_active_tracking"):
                self.rib_inspector.reset_active_tracking()
            if hasattr(self, "health_monitor") and self.health_monitor is not None and hasattr(self.health_monitor, "reset_phase_memory"):
                self.health_monitor.reset_phase_memory()
            self.slat_history_samples.clear()
            self.latched_slat_data = None
            self.latched_hold_frames = 0
            self._last_rib_slat_occl_frame = -100
            self._best_slat_candidate = None
            self._best_rib_candidate = None
            self._current_slat_cycle_id = None
            self._current_rib_cycle_id = None

    def toggle_db(self):
        """Membalik status saklar database logging (ON <-> OFF)."""
        if self.db_bridge is not None:
            return self.db_bridge.toggle()
        return False

    def apply_config_dict(self, cfg, apply_geometry=True):
        """Menerapkan konfigurasi dictionary ke Slat & Rib Inspector saat runtime."""
        if not cfg or not isinstance(cfg, dict):
            return False

        with self._inspect_lock:
            if apply_geometry:
                # Terapkan Matriks Rektifikasi Virtual Citra (Metode 4)
                if "rectification_matrix" in cfg and cfg["rectification_matrix"] is not None:
                    self.rect_matrix = np.array(cfg["rectification_matrix"], dtype=np.float32)
                else:
                    self.rect_matrix = None
                self._rect_calib_size = self._infer_calib_frame_size(cfg)
                self._scaled_rect_cache.clear()

                if "slat_polygon" in cfg and len(cfg["slat_polygon"]) >= 4:
                    self.slat_polygon = cfg["slat_polygon"]
                    self._update_slat_y_bounds()

                if "rib_polygon" in cfg and len(cfg["rib_polygon"]) >= 4:
                    self.rib_zone = cfg["rib_polygon"]

                if "calibration_profile" in cfg:
                    self._apply_golden_reference(cfg["calibration_profile"])
                    if hasattr(self.rib_inspector, "set_adaptive_geometry"):
                        cp = cfg["calibration_profile"] or {}
                        self.rib_inspector.set_adaptive_geometry(
                            min_span_px=cp.get("recommended_min_span_px")
                        )
                elif hasattr(self.rib_inspector, "set_adaptive_geometry"):
                    self.rib_inspector.set_adaptive_geometry(min_span_px=None)

            th = cfg.get("thresholds", {})
            if "max_flatness_pv" in th:
                self.slat_inspector.max_flatness_pv = float(th["max_flatness_pv"])
            if "max_sagitta" in th:
                self.slat_inspector.max_sagitta = float(th["max_sagitta"])
            if "max_center_void" in th:
                self.slat_inspector.max_center_void = int(th["max_center_void"])
            if "min_continuity_cov" in th:
                self.slat_inspector.min_continuity_cov = float(th["min_continuity_cov"])

            if "rib_pv_threshold" in th:
                self.rib_inspector.pv_thresh = float(th["rib_pv_threshold"])
                self.rib_inspector._base_pv_thresh = self.rib_inspector.pv_thresh
            if "rib_sagitta_threshold" in th:
                self.rib_inspector.sagitta_thresh = float(th["rib_sagitta_threshold"])
                self.rib_inspector._base_sagitta_thresh = self.rib_inspector.sagitta_thresh
            if "rib_dev_threshold" in th or "rib_max_dev_threshold" in th:
                dev_val = th.get("rib_dev_threshold", th.get("rib_max_dev_threshold"))
                self.rib_inspector.dev_thresh = float(dev_val)
                self.rib_inspector._base_dev_thresh = self.rib_inspector.dev_thresh
            if "rib_tip_threshold" in th:
                self.rib_inspector.tip_thresh = float(th["rib_tip_threshold"])
                self.rib_inspector._base_tip_thresh = self.rib_inspector.tip_thresh
            if "rib_conf_threshold" in th:
                self.rib_inspector.conf_thresh = float(th["rib_conf_threshold"])
            if "rib_severity_zones" in th:
                self.rib_inspector._severity_zones = th["rib_severity_zones"]

            if "rib_prime_min_y" in th or "rib_prime_max_y" in th:
                pmin = th.get("rib_prime_min_y", self.rib_inspector.prime_min_y)
                pmax = th.get("rib_prime_max_y", self.rib_inspector.prime_max_y)
                self.rib_inspector.set_prime_zone(pmin, pmax)

            return True

    def _sync_config_mtime(self):
        try:
            if self.config_path and os.path.exists(self.config_path):
                self._config_mtime = os.path.getmtime(self.config_path)
        except Exception:
            pass

    def _apply_dataset_adaptive_thresholds(self):
        """Memuat ulang adaptive threshold dari Rib_Dataset untuk profil aktif."""
        if self.db_bridge is not None and hasattr(self.db_bridge, "db_path"):
            fb_prof = "DEFAULT_STABIL" if self.active_profile_name == "HASIL_KALIBRASI" else None
            return self.rib_inspector.load_adaptive_thresholds(
                db_path=self.db_bridge.db_path,
                camera_profile=self.active_profile_name or None,
                fallback_profile=fb_prof,
            )
        return False

    def toggle_threshold_source(self):
        """
        Beralih antara sumber threshold:
        - 'DATASET': baseline config + adaptive threshold Rib_Dataset (default startup)
        - 'WEB'    : parameter murni dari Web Dashboard (top_inspection_config.json) + auto hot-reload
        """
        with self._inspect_lock:
            target_mode = "WEB" if self.threshold_source_mode == "DATASET" else "DATASET"
            cfg = None
            if self.config_path and os.path.exists(self.config_path):
                try:
                    with open(self.config_path, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                except Exception as err:
                    print(f"[!] Gagal membaca {os.path.basename(self.config_path)}: {err}")

            if cfg is not None:
                self.apply_config_dict(cfg, apply_geometry=False)
            self._sync_config_mtime()
            self.threshold_source_mode = target_mode

            if target_mode == "DATASET":
                self._apply_dataset_adaptive_thresholds()
                self.profile_banner_text = "MODE THRESHOLD: AUTO DATASET (Rib_Dataset)"
                print("\n" + "=" * 70)
                print("[CombinedTopInspector] 📊 BERALIH KE THRESHOLD HASIL DATASET (AUTO ADAPTIVE)")
            else:
                self.profile_banner_text = "MODE THRESHOLD: WEB DASHBOARD (Hot-Reload Aktif)"
                print("\n" + "=" * 70)
                print("[CombinedTopInspector] 🌐 BERALIH KE PARAMETER WEB DASHBOARD (HOT-RELOAD AKTIF)")

            self.profile_banner_timer = 90
            print(
                f"[*] Rib  : PV<={self.rib_inspector.pv_thresh:.2f}px | Sag<={self.rib_inspector.sagitta_thresh:.2f}px | "
                f"Dev<={self.rib_inspector.dev_thresh:.2f}px | Tip<={self.rib_inspector.tip_thresh:.2f}px | "
                f"Conf>={self.rib_inspector.conf_thresh:.2f}"
            )
            print(
                f"[*] Slat : PV<={self.slat_inspector.max_flatness_pv:.2f}px | Sag<={self.slat_inspector.max_sagitta:.2f}px | "
                f"Void<{self.slat_inspector.max_center_void}px"
            )
            print("=" * 70 + "\n")
            return self.threshold_source_mode

    def _check_web_config_hot_reload(self):
        """Hot-reload threshold dari top_inspection_config.json jika mode WEB aktif."""
        if getattr(self, "threshold_source_mode", "DATASET") != "WEB":
            return False
        if not self.config_path or not os.path.exists(self.config_path):
            return False
        now = time.time()
        if (now - getattr(self, "_last_hot_reload_check", 0.0)) < 0.5:
            return False
        self._last_hot_reload_check = now
        try:
            mtime = os.path.getmtime(self.config_path)
            if mtime > getattr(self, "_config_mtime", 0.0):
                with open(self.config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                self.apply_config_dict(cfg, apply_geometry=False)
                self._config_mtime = mtime
                self.profile_banner_text = "HOT-RELOAD: Parameter Web Dashboard Diperbarui!"
                self.profile_banner_timer = 90
                print(
                    f"[HOT-RELOAD WEB] Config diperbarui | Rib PV<={self.rib_inspector.pv_thresh:.2f}, "
                    f"Sag<={self.rib_inspector.sagitta_thresh:.2f}, Dev<={self.rib_inspector.dev_thresh:.2f}, "
                    f"Tip<={self.rib_inspector.tip_thresh:.2f} | Slat PV<={self.slat_inspector.max_flatness_pv:.2f}"
                )
                return True
        except Exception:
            pass
        return False

    def toggle_alignment_guides(self):
        """Membalik status tampilan garis referensi rel & sumbu konveyor (ON <-> OFF)."""
        self.show_alignment_guides = not getattr(self, "show_alignment_guides", False)
        return self.show_alignment_guides

    def restore_default_profile(self, preserve_web_thresholds=False):
        """
        Mengembalikan konfigurasi ke Profil Default Stabil (Baseline Terbukti).
        Jika preserve_web_thresholds=False (tombol [U]), reset total termasuk kembali ke mode DATASET.
        """
        if not os.path.exists(self.default_config_path):
            print(f"[!] File default config tidak ditemukan: {self.default_config_path}")
            return False

        with self._inspect_lock:
            try:
                with open(self.default_config_path, "r", encoding="utf-8") as f:
                    cfg_default = json.load(f)

                if preserve_web_thresholds and getattr(self, "threshold_source_mode", "DATASET") == "WEB":
                    if os.path.exists(self.config_path):
                        with open(self.config_path, "r", encoding="utf-8") as f_cur:
                            cur_cfg = json.load(f_cur)
                        if isinstance(cur_cfg.get("thresholds"), dict):
                            cfg_default["thresholds"] = cur_cfg["thresholds"]

                self.apply_config_dict(cfg_default, apply_geometry=True)

                with open(self.config_path, "w", encoding="utf-8") as f:
                    json.dump(cfg_default, f, indent=2)
                self._sync_config_mtime()

                self.active_profile_name = cfg_default.get("profile_name", "DEFAULT_STABIL")
                if not preserve_web_thresholds:
                    self.threshold_source_mode = "DATASET"
                    self._apply_dataset_adaptive_thresholds()
                self.profile_banner_text = "PROFIL PULIH: DEFAULT STABIL (Tanpa Rektifikasi)"
                self.profile_banner_timer = 90
                print("\n" + "=" * 70)
                print("[CombinedTopInspector] ✅ BERHASIL DIKEMBALIKAN KE PROFIL DEFAULT STABIL!")
                print(f"[*] Config Aktif: {os.path.basename(self.config_path)} (Dipulihkan dari .default.json)")
                print(f"[*] Rektifikasi : {'AKTIF' if self.rect_matrix is not None else 'NON-AKTIF (Asli)'}")
                print(f"[*] Mode Thresh : {self.threshold_source_mode}")
                print(f"[*] Parameter   : Rib PV={self.rib_inspector.pv_thresh:.1f}px, Tip={self.rib_inspector.tip_thresh:.1f}px, Prime=[{self.rib_inspector.prime_min_y}, {self.rib_inspector.prime_max_y}]")
                print("=" * 70 + "\n")
                return True
            except Exception as e:
                print(f"[!] Gagal restore default profile: {e}")
                return False

    def load_calibrated_profile(self):
        """
        Memuat konfigurasi dari Profil Hasil Kalibrasi Otomatis ([K]).
        """
        if not os.path.exists(self.calibrated_config_path):
            print(f"\n[!] File calibrated config belum ada: {self.calibrated_config_path}")
            print("    Silakan jalankan kalibrasi konveyor [K] terlebih dahulu.\n")
            return False

        with self._inspect_lock:
            try:
                with open(self.calibrated_config_path, "r", encoding="utf-8") as f:
                    cfg_calib = json.load(f)

                keep_web = (getattr(self, "threshold_source_mode", "DATASET") == "WEB")
                if keep_web and os.path.exists(self.config_path):
                    with open(self.config_path, "r", encoding="utf-8") as f_cur:
                        cur_cfg = json.load(f_cur)
                    if isinstance(cur_cfg.get("thresholds"), dict):
                        cfg_calib["thresholds"] = cur_cfg["thresholds"]

                self.apply_config_dict(cfg_calib, apply_geometry=True)

                with open(self.config_path, "w", encoding="utf-8") as f:
                    json.dump(cfg_calib, f, indent=2)
                self._sync_config_mtime()

                self.active_profile_name = cfg_calib.get("profile_name", "HASIL_KALIBRASI")
                if not keep_web:
                    self._apply_dataset_adaptive_thresholds()
                self.profile_banner_text = "PROFIL AKTIF: HASIL KALIBRASI (Rektifikasi Aktif)"
                self.profile_banner_timer = 90
                print("\n" + "=" * 70)
                print("[CombinedTopInspector] 🔄 BERALIH KE PROFIL HASIL KALIBRASI!")
                print(f"[*] Config Aktif: {os.path.basename(self.config_path)} (Dimuat dari .calibrated.json)")
                print(f"[*] Rektifikasi : {'AKTIF' if self.rect_matrix is not None else 'NON-AKTIF'}")
                print(f"[*] Mode Thresh : {self.threshold_source_mode}")
                print(f"[*] Parameter   : Rib PV={self.rib_inspector.pv_thresh:.1f}px, Tip={self.rib_inspector.tip_thresh:.1f}px, Prime=[{self.rib_inspector.prime_min_y}, {self.rib_inspector.prime_max_y}]")
                print("=" * 70 + "\n")
                return True
            except Exception as e:
                print(f"[!] Gagal memuat calibrated profile: {e}")
                return False

    def toggle_profile(self):
        """
        Beralih bergantian antara profil DEFAULT_STABIL <-> HASIL_KALIBRASI.
        """
        with self._inspect_lock:
            if self.active_profile_name == "DEFAULT_STABIL":
                if os.path.exists(self.calibrated_config_path):
                    return self.load_calibrated_profile()
                else:
                    print("\n[!] Profil Kalibrasi belum tersedia. Jalankan kalibrasi [K] terlebih dahulu.\n")
                    return False
            else:
                keep_web = (getattr(self, "threshold_source_mode", "DATASET") == "WEB")
                return self.restore_default_profile(preserve_web_thresholds=keep_web)

    def start_calibration(self, duration_frames=60):
        """Memulai sesi kalibrasi konveyor multi-frame (default 60 frame / ~2-3 detik)."""
        with self._inspect_lock:
            if not hasattr(self, "motion_calibrator") or self.motion_calibrator is None:
                self.motion_calibrator = ConveyorMotionCalibrator(
                    target_tilt_deg=getattr(self, "ref_tilt_deg", 0.90),
                    target_cx=getattr(self, "ref_center_x", 960.0),
                    target_left_rail=getattr(self, "ref_left_rail_px", 188.0),
                    target_right_rail=getattr(self, "ref_right_rail_px", 1759.0)
                )
            self.motion_calibrator.reset()
            self.is_calibrating = True
            self.calib_frames_target = max(30, int(duration_frames))
            self.calib_buffer.clear()
            self.calib_status_msg = f"Mengumpulkan frame konveyor (0/{self.calib_frames_target})..."
            self.calib_banner_timer = 0
            print(f"\n[*] [CombinedTopInspector] 🔄 Memulai Kalibrasi Konveyor ({self.calib_frames_target} frame)...")

    def _process_calibration_frame(self, frame):
        """Menampung frame citra untuk proses kalibrasi."""
        if not self.is_calibrating:
            return
        if frame is not None and frame.size > 0:
            self.calib_buffer.append(frame.copy())
            self.calib_status_msg = f"Mengumpulkan frame konveyor ({len(self.calib_buffer)}/{self.calib_frames_target})..."

        if len(self.calib_buffer) >= self.calib_frames_target:
            self._finish_calibration()

    def _finish_calibration(self):
        """
        Menganalisis frame kalibrasi menggunakan ConveyorMotionCalibrator (Metode 4: Hybrid Rectification).
        Menghasilkan Matriks Affine Rektifikasi (M_rect), menyelaraskan kamera baru ke Golden Baseline,
        dan menyimpan profil terkalibrasi ke top_inspection_config.calibrated.json.
        """
        if not self.calib_buffer:
            self.is_calibrating = False
            return

        with self._inspect_lock:
            try:
                if not hasattr(self, "motion_calibrator") or self.motion_calibrator is None:
                    self.motion_calibrator = ConveyorMotionCalibrator(
                        target_tilt_deg=getattr(self, "ref_tilt_deg", 0.90),
                        target_cx=getattr(self, "ref_center_x", 960.0),
                        target_left_rail=getattr(self, "ref_left_rail_px", 188.0),
                        target_right_rail=getattr(self, "ref_right_rail_px", 1759.0)
                    )

                calib_res = self.motion_calibrator.calibrate(self.calib_buffer)
                if not calib_res.get("success", False):
                    raise RuntimeError(calib_res.get("error", "Gagal menghitung matriks kalibrasi."))

                # Pasang matriks rektifikasi langsung ke memori runtime
                m_rect_list = calib_res["rectification_matrix"]
                self.rect_matrix = np.array(m_rect_list, dtype=np.float32)
                h_cal, w_cal = self.calib_buffer[0].shape[:2]
                self._rect_calib_size = (int(w_cal), int(h_cal))
                self._scaled_rect_cache.clear()

                # Muat struktur konfigurasi golden default untuk menjaga integritas seluruh batas QC
                cfg_data = {}
                if os.path.exists(self.default_config_path):
                    try:
                        with open(self.default_config_path, "r") as f:
                            cfg_data = json.load(f)
                    except Exception:
                        pass
                if not cfg_data and os.path.exists(self.config_path):
                    try:
                        with open(self.config_path, "r") as f:
                            cfg_data = json.load(f)
                    except Exception:
                        pass

                # Sisipkan matriks rektifikasi & metadata kalibrasi
                cfg_data["profile_name"] = "HASIL_KALIBRASI"
                cfg_data["description"] = "Profil Parameter Hasil Kalibrasi Otomatis Kamera Top (Virtual Rectification)"
                cfg_data["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cfg_data["rectification_matrix"] = m_rect_list

                if "thresholds" not in cfg_data:
                    cfg_data["thresholds"] = {}
                if getattr(self, "threshold_source_mode", "DATASET") == "WEB" and os.path.exists(self.config_path):
                    try:
                        with open(self.config_path, "r", encoding="utf-8") as f_cur:
                            cur_cfg = json.load(f_cur)
                        if isinstance(cur_cfg.get("thresholds"), dict):
                            cfg_data["thresholds"] = cur_cfg["thresholds"]
                    except Exception:
                        pass
                cfg_data["thresholds"]["bracket_tilt_deg"] = calib_res["measured_tilt_deg"]

                if "calibration_profile" not in cfg_data:
                    cfg_data["calibration_profile"] = {}
                cfg_data["calibration_profile"].update({
                    "calibrated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "calib_frame_w": int(w_cal),
                    "calib_frame_h": int(h_cal),
                    "golden_reference": {
                        "source_video": "Kamera posisi 3 dimmer 2 - Reverse.mp4",
                        "source_image": "WIN_20261005_09_01_08_Pro.jpg",
                        "target_tilt_deg": getattr(self, "ref_tilt_deg", 0.90),
                        "target_cx": getattr(self, "ref_center_x", 960.0),
                        "left_rail_px": int(getattr(self, "ref_left_rail_px", 188.0)),
                        "right_rail_px": int(getattr(self, "ref_right_rail_px", 1759.0)),
                        "conveyor_width_px": int(getattr(self, "ref_width_px", 1571.0)),
                        "top_y_px": int(getattr(self, "ref_y_top_px", 216.0)),
                        "bot_y_px": int(getattr(self, "ref_y_bot_px", 864.0)),
                        "top_left_rail_px": int(getattr(self, "ref_top_left_rail_px", 278.0)),
                        "top_right_rail_px": int(getattr(self, "ref_top_right_rail_px", 1630.0)),
                        "bot_left_rail_px": int(getattr(self, "ref_bot_left_rail_px", 104.0)),
                        "bot_right_rail_px": int(getattr(self, "ref_bot_right_rail_px", 1808.0)),
                    },
                    "measured_tilt_deg": calib_res["measured_tilt_deg"],
                    "target_tilt_deg": calib_res["target_tilt_deg"],
                    "angle_correction_deg": calib_res["angle_correction_deg"],
                    "conveyor_center_x": calib_res["conveyor_center_x"],
                    "lateral_offset_px": calib_res["lateral_offset_px"],
                    "left_rail_px": calib_res["left_rail_px"],
                    "right_rail_px": calib_res["right_rail_px"],
                    "conveyor_width_px": calib_res["conveyor_width_px"],
                    "motion_sample_count": calib_res["motion_sample_count"],
                })

                # Simpan ke config aktif dan file calibrated.json
                with open(self.config_path, "w") as f:
                    json.dump(cfg_data, f, indent=2)
                self._sync_config_mtime()

                if hasattr(self, "calibrated_config_path") and self.calibrated_config_path:
                    with open(self.calibrated_config_path, "w") as f:
                        json.dump(cfg_data, f, indent=2)

                self.active_profile_name = "HASIL_KALIBRASI"
                if hasattr(self, "pose_watchdog") and self.pose_watchdog is not None:
                    self.pose_watchdog.reset_reference()
                self.calib_status_msg = calib_res["status_message"]
                self.calib_banner_timer = 150
                self.profile_banner_text = f"KALIBRASI SUKSES: Koreksi {calib_res['angle_correction_deg']:+.2f} deg, Geser {calib_res['lateral_offset_px']:+.1f} px"
                self.profile_banner_timer = 150

                print("\n" + "=" * 70)
                print("[CombinedTopInspector] ✅ KALIBRASI OTOMATIS KONVEYOR SELESAI!")
                print(f"[*] Sudut Terukur    : {calib_res['measured_tilt_deg']:+.3f}° (Target Golden: {calib_res['target_tilt_deg']:+.3f}°)")
                print(f"[*] Koreksi Rotasi  : {calib_res['angle_correction_deg']:+.3f}° (Affine Warp M_rect)")
                print(f"[*] Pusat Konveyor  : X={calib_res['conveyor_center_x']:.1f} px (Offset: {calib_res['lateral_offset_px']:+.1f} px)")
                print(f"[*] Lebar Rel       : {calib_res['conveyor_width_px']} px (Kiri: {calib_res['left_rail_px']} px, Kanan: {calib_res['right_rail_px']} px)")
                print(f"[*] Sampel Gerak    : {calib_res['motion_sample_count']} vektor Optical Flow terverifikasi")
                print(f"[*] Profil Tersimpan: {os.path.basename(self.calibrated_config_path)}")
                print("=" * 70 + "\n")
            except Exception as err:
                self.calib_status_msg = f"Gagal kalibrasi: {err}"
                self.calib_banner_timer = 90
                print(f"\n[!] [CombinedTopInspector] Gagal kalibrasi konveyor: {err}\n")
            finally:
                self.calib_buffer.clear()
                self.is_calibrating = False

    def auto_align_rails_instant(self, frame):
        """
        Melakukan Auto-Snap deteksi tepi rel dan langsung menghitung serta menerapkan
        rektifikasi virtual (Perspective/Affine) seketika dari 1 frame saat ini.
        """
        if frame is None or frame.size == 0:
            return False, "Frame kosong."
        try:
            from calibrate_dual_rails import DualRailCalibrator
            calib = DualRailCalibrator(config_path=self.config_path)
            rib_boxes_hint = None
            try:
                if hasattr(self.rib_inspector, "detect_yolo_zone"):
                    h_fr, w_fr = frame.shape[:2]
                    _, _, _, rib_boxes_hint = self.rib_inspector.detect_yolo_zone(
                        frame, (int(0.02 * w_fr), int(0.18 * h_fr), int(0.98 * w_fr), int(0.68 * h_fr))
                    )
            except Exception:
                rib_boxes_hint = None
            calib.auto_snap_rails(frame, rib_boxes=rib_boxes_hint)
            M, m_type = calib.compute_rectification(frame.shape[1], frame.shape[0])

            with self._inspect_lock:
                self.rect_matrix = np.array(M, dtype=np.float32)
                self._rect_calib_size = (int(frame.shape[1]), int(frame.shape[0]))
                self._scaled_rect_cache.clear()
                saved_web_th = None
                if getattr(self, "threshold_source_mode", "DATASET") == "WEB" and os.path.exists(self.config_path):
                    try:
                        with open(self.config_path, "r", encoding="utf-8") as f_cur:
                            saved_web_th = json.load(f_cur).get("thresholds")
                    except Exception:
                        saved_web_th = None
                calib.save_calibration_to_config()
                if isinstance(saved_web_th, dict) and os.path.exists(self.config_path):
                    try:
                        with open(self.config_path, "r", encoding="utf-8") as f_new:
                            new_cfg = json.load(f_new)
                        tilt_new = new_cfg.get("thresholds", {}).get("bracket_tilt_deg")
                        new_cfg["thresholds"] = saved_web_th
                        if tilt_new is not None:
                            new_cfg["thresholds"]["bracket_tilt_deg"] = tilt_new
                        with open(self.config_path, "w", encoding="utf-8") as f_out:
                            json.dump(new_cfg, f_out, indent=2)
                    except Exception:
                        pass
                self._sync_config_mtime()
                self.active_profile_name = "HASIL_KALIBRASI"
                if hasattr(self, "pose_watchdog") and self.pose_watchdog is not None:
                    self.pose_watchdog.set_aruco_reference(
                        getattr(calib, "aruco_landmarks", None),
                        calib_w=int(frame.shape[1]),
                        calib_h=int(frame.shape[0]),
                    )
                    self.pose_watchdog.reset_reference()
                # Terapkan adaptasi zona & bentang minimum langsung di memori tanpa menarik batas ke lorong jauh
                rec_top_y = max(0.26, float(calib.metrology.get("recommended_rib_y_top_frac", 0.28)))
                rec_prime_min = max(295, int(calib.metrology.get("recommended_prime_min_y", 315)))
                rec_min_span = calib.metrology.get("recommended_min_span_px")
                if rec_top_y < 0.28 and isinstance(self.rib_zone, list) and len(self.rib_zone) >= 4:
                    new_zone = [list(pt) for pt in self.rib_zone]
                    new_zone[0][1] = min(float(new_zone[0][1]), rec_top_y)
                    new_zone[1][1] = min(float(new_zone[1][1]), rec_top_y)
                    self.rib_zone = new_zone
                    pmax_cur = getattr(self.rib_inspector, "prime_max_y", 505)
                    self.rib_inspector.set_prime_zone(
                        min(getattr(self.rib_inspector, "prime_min_y", 315), rec_prime_min),
                        pmax_cur,
                    )
                if hasattr(self.rib_inspector, "set_adaptive_geometry"):
                    self.rib_inspector.set_adaptive_geometry(min_span_px=rec_min_span)
                tilt_val = calib.metrology.get("measured_tilt_deg", 0.0)
                offset_val = calib.metrology.get("lateral_offset_px", 0.0)
                self.profile_banner_text = f"AUTO-SNAP REL: Tilt {tilt_val:+.2f} deg, Geser {offset_val:+.1f} px ({m_type})"
                self.profile_banner_timer = 150
                self.calib_status_msg = f"Auto-Snap Rel Sukses ({m_type})"
                self.calib_banner_timer = 150

            print("\n" + "=" * 70)
            print(f"[CombinedTopInspector] ⚡ AUTO-SNAP REL INSTAN SUKSES! ({m_type})")
            print(f"[*] Sudut Terukur    : {tilt_val:+.2f}° (Koreksi {-tilt_val:+.2f}°)")
            print(f"[*] Lebar Terukur    : {calib.metrology.get('cur_width_px', 0.0):.1f} px (Target: {self.ref_width_px:.0f} px)")
            print(f"[*] Offset Lateral   : {offset_val:+.1f} px")
            print(f"[*] Mode Transformasi: {m_type}")
            print(f"[*] Profil Tersimpan : {os.path.basename(self.calibrated_config_path)}")
            print("=" * 70 + "\n")
            return True, "Sukses"
        except Exception as e:
            err = f"Gagal auto-align rel: {e}"
            print(f"[!] {err}")
            return False, err

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
        rect_m_full = self._get_scaled_rect_matrix(frame_raw.shape[1], frame_raw.shape[0])
        if rect_m_full is not None:
            if rect_m_full.shape == (3, 3):
                ann = cv2.warpPerspective(frame_raw, rect_m_full, (frame_raw.shape[1], frame_raw.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            else:
                ann = cv2.warpAffine(frame_raw, rect_m_full, (frame_raw.shape[1], frame_raw.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        else:
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
        r_dev = self._extract_rib_deviation(rib_res)
        r_sag = float(rib_res.get("sagitta", 0.0) or 0.0)

        s_stat = slat_res.get("status", "WAIT")
        r_stat = rib_res.get("status", "WAIT")
        slat_conf = float(slat_res.get("confidence", slat_res.get("coverage", 0.0)) or 0.0)
        rib_conf = float(rib_res.get("confidence", 0.0) or 0.0)
        conf_slat_hdr = f" ({slat_conf:.2f})" if slat_conf > 0 else ""
        conf_rib_hdr = f" ({rib_conf:.2f})" if rib_conf > 0 else ""

        cam_lbl = getattr(self, "top_camera_id", "CAM-01")
        title = f"[AI METROLOGY EVIDENCE] TOP {cam_lbl} | SLAT #{slat_no if slat_no is not None else '-'}: [{s_stat}]{conf_slat_hdr} | RIB #{rib_no if rib_no is not None else '-'}: [{r_stat}]{conf_rib_hdr}"
        cv2.putText(ann, title, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 230, 255), 2, cv2.LINE_AA)

        rib_scale = float(rib_res.get("rib_mm_per_px", 750.0 / 1200.0) or (750.0 / 1200.0))
        metrics_txt = (
            f"PLAT: PV={s_pv*scale:.1f}mm ({s_pv:.1f}px) | Sag={s_sag*scale:.1f}mm ({s_sag:.1f}px) | Void={s_void}px  ||  "
            f"RIB: Dev={r_dev*rib_scale:.1f}mm ({r_dev:.1f}px) | Sag={r_sag*rib_scale:.1f}mm ({r_sag:.1f}px)"
        )
        cv2.putText(ann, metrics_txt, (20, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 240, 240), 1, cv2.LINE_AA)

        return ann

    @staticmethod
    def _extract_rib_deviation(rib_res):
        """Ambil nilai deviasi rib terbesar dari berbagai key hasil inspeksi."""
        if not rib_res:
            return 0.0
        for key in ("max_deviation", "max_dev", "dev"):
            val = rib_res.get(key)
            if val is not None and float(val) > 0.0:
                return float(val)
        inner_v = float(rib_res.get("inner_dev", 0.0) or 0.0)
        tip_v = float(rib_res.get("tip_dev", 0.0) or 0.0)
        return max(inner_v, tip_v)

    def purge_top_candidate_if_ok(self, curr_slat_ok=False, curr_rib_ok=False, slat_no=None, rib_no=None):
        """
        Mekanisme Transactional Latch:
        Membatalkan (purge) kandidat Best-Shot Slat/Rib secara independen jika vonis menjadi OK.
        """
        if self._best_rib_candidate is not None:
            cand_rib_no = self._best_rib_candidate.get("rib_no")
            committed_rib = (
                self.rib_inspector.get_committed_status(cand_rib_no)
                if hasattr(self.rib_inspector, "get_committed_status")
                else None
            )
            if committed_rib == "OK" or (curr_rib_ok and (rib_no is None or rib_no == cand_rib_no)):
                self._best_rib_candidate = None

        if self._best_slat_candidate is not None:
            cand_slat_no = self._best_slat_candidate.get("slat_no")
            if curr_slat_ok and (slat_no is None or slat_no == cand_slat_no):
                self._best_slat_candidate = None

    def _flush_slat_candidate_if_needed(self):
        """Memancarkan kandidat Best-Shot PLAT SLAT ke db_bridge saat siklus slat selesai."""
        if self._best_slat_candidate is None:
            return False
        cand = self._best_slat_candidate
        self._best_slat_candidate = None

        slat_no = cand.get("slat_no")
        current_slat_stat = getattr(self.slat_inspector, "latched_status", None)
        current_slat_count = getattr(self.slat_inspector, "slat_count", 0)
        if slat_no == current_slat_count and current_slat_stat == "OK":
            return False
        if slat_no is not None and slat_no == self._last_emitted_slat_id:
            return False

        self._last_emitted_slat_id = slat_no
        if self.db_bridge is not None and self.db_bridge.is_enabled():
            return self.db_bridge.emit_top_inspection(
                frame_raw=cand["frame_raw"],
                frame_ann=cand["frame_ann"],
                slat_result=cand["slat_res"],
                rib_result=None,
                slat_number=slat_no,
                rib_number=cand.get("rib_no"),
            )
        return False

    def _flush_rib_candidate_if_needed(self, force_final=False):
        """Memancarkan kandidat Best-Shot RIB ke db_bridge hanya setelah vonis rib final."""
        if self._best_rib_candidate is None:
            return False
        cand = self._best_rib_candidate
        rib_no = cand.get("rib_no")

        committed_stat = (
            self.rib_inspector.get_committed_status(rib_no)
            if hasattr(self.rib_inspector, "get_committed_status")
            else None
        )
        current_rib_stat = getattr(self.rib_inspector, "current_rib_status", None)
        current_rib_count = getattr(self.rib_inspector, "rib_count", 0)

        # Jika sesi rib belum final/komit dan bukan shutdown paksa, tahan di buffer!
        if hasattr(self.rib_inspector, "get_committed_status") and committed_stat is None and not force_final:
            if rib_no == current_rib_count:
                return False

        self._best_rib_candidate = None

        if committed_stat == "OK":
            return False
        if committed_stat is None and rib_no == current_rib_count and current_rib_stat == "OK":
            return False
        if committed_stat is None and force_final and current_rib_stat != "NG":
            return False

        if rib_no is not None and rib_no == self._last_emitted_rib_id:
            return False

        self._last_emitted_rib_id = rib_no
        if self.db_bridge is not None and self.db_bridge.is_enabled():
            return self.db_bridge.emit_top_inspection(
                frame_raw=cand["frame_raw"],
                frame_ann=cand["frame_ann"],
                slat_result=None,
                rib_result=cand["rib_res"],
                slat_number=cand.get("slat_no"),
                rib_number=rib_no,
            )
        return False

    def flush_best_shot_if_needed(self, force_final=False):
        """Memancarkan kandidat Best-Shot Slat dan Rib yang sudah selesai siklusnya."""
        res_slat = self._flush_slat_candidate_if_needed()
        res_rib = self._flush_rib_candidate_if_needed(force_final=force_final)
        return bool(res_slat or res_rib)

    def _sync_recording_lifecycle(self, slat_res, rib_res, slat_no=None, rib_no=None):
        """Sinkronisasi siklus Slat dan Rib secara terpisah pada setiap frame tanpa render HUD."""
        slat_res = slat_res or {}
        rib_res = rib_res or {}
        curr_slat_no = slat_no if slat_no is not None else getattr(self.slat_inspector, "slat_count", 0)
        curr_rib_no = rib_no if rib_no is not None else getattr(self.rib_inspector, "rib_count", 0)

        # 1a. Siklus Slat independen: pergantian slat hanya men-flush kandidat slat
        slat_changed = (self._current_slat_cycle_id is not None and curr_slat_no != self._current_slat_cycle_id)
        if self._best_slat_candidate is not None:
            cand_s_res = self._best_slat_candidate.get("slat_res", {})
            cand_void_only = (
                float(cand_s_res.get("pv_flatness", 0.0) or 0.0) <= self.slat_inspector.max_flatness_pv
                and float(cand_s_res.get("sagitta", 0.0) or 0.0) <= self.slat_inspector.max_sagitta
            )
            cand_y_frame = self._best_slat_candidate.get("y_seam_frame")
            pg_overlaps_cand = False
            if cand_y_frame is not None:
                for pb in (rib_res.get("pre_gate_rib_boxes", []) or []):
                    b = pb.get("box") if isinstance(pb, dict) else pb
                    if b is not None and (b[1] - 15) <= cand_y_frame <= (b[3] + 25):
                        pg_overlaps_cand = True
                        break
            if cand_void_only:
                active_box = rib_res.get("active_box")
                cand_frame = self._best_slat_candidate.get("captured_frame_idx", -999)
                curr_frame = getattr(self, "_slat_frame_idx", 0)
                y_slat_lo = float(getattr(self, "slat_y_min", 0.50 * getattr(self, "_last_h_f", 1080)))
                rising_out_of_gate = (
                    active_box is not None
                    and (curr_frame - cand_frame) <= 15
                    and float(active_box[3]) >= (y_slat_lo - max(18.0, 0.05 * float(getattr(self, "_last_h_f", 1080))))
                )
                if (
                    slat_res.get("is_rib_base_occluded", False)
                    or slat_res.get("is_seam_masked_occluded", False)
                    or pg_overlaps_cand
                    or rising_out_of_gate
                ):
                    self._best_slat_candidate = None

        if slat_changed:
            self._flush_slat_candidate_if_needed()
            self._current_slat_cycle_id = curr_slat_no
        elif self._current_slat_cycle_id is None:
            self._current_slat_cycle_id = curr_slat_no

        # 1b. Siklus Rib independen: flush hanya bila sesi rib sudah komit final atau rib berganti
        rib_changed = (self._current_rib_cycle_id is not None and curr_rib_no != self._current_rib_cycle_id and curr_rib_no > 0)
        cand_rib_no = self._best_rib_candidate.get("rib_no") if self._best_rib_candidate else None
        rib_committed = (
            cand_rib_no is not None
            and hasattr(self.rib_inspector, "get_committed_status")
            and self.rib_inspector.get_committed_status(cand_rib_no) is not None
        )
        if rib_committed or rib_changed:
            self._flush_rib_candidate_if_needed()
            self._current_rib_cycle_id = curr_rib_no
        elif self._current_rib_cycle_id is None:
            self._current_rib_cycle_id = curr_rib_no

        # 2. Transactional Purge segera bila status berbalik menjadi OK
        curr_rib_ok = bool(rib_res.get("is_ok", False)) or (rib_res.get("status") == "OK")
        curr_slat_ok = bool(slat_res.get("is_ok", False)) or (slat_res.get("status") == "OK")
        if curr_rib_ok or curr_slat_ok or self._best_rib_candidate is not None or self._best_slat_candidate is not None:
            self.purge_top_candidate_if_ok(
                curr_slat_ok=curr_slat_ok,
                curr_rib_ok=curr_rib_ok,
                slat_no=curr_slat_no,
                rib_no=curr_rib_no,
            )
        return curr_slat_no, curr_rib_no

    def record_if_ng(self, frame_raw, frame_ann, inspect_res, slat_no=None, rib_no=None):
        """
        Merekam frame ke buffer Best-Shot Slat/Rib secara terpisah.
        Kandidat Rib hanya di-emit ke DB/Teams setelah rib selesai melintas ROI dengan vonis final NG.
        """
        if self.db_bridge is None or not self.db_bridge.is_enabled():
            return False

        slat_res = inspect_res.get("slat_res", {}) if inspect_res else {}
        rib_res = inspect_res.get("rib_res", {}) if inspect_res else {}

        curr_slat_no, curr_rib_no = self._sync_recording_lifecycle(
            slat_res, rib_res, slat_no=slat_no, rib_no=rib_no
        )

        is_slat_ng = (
            bool(slat_res.get("is_ng", False))
            and (slat_res.get("status") == "NG")
            and not bool(slat_res.get("is_rib_base_occluded", False))
            and not bool(slat_res.get("is_seam_masked_occluded", False))
        )
        is_rib_ng = bool(rib_res.get("is_ng", False)) and (rib_res.get("status") == "NG")
        is_rib_coasting = bool(rib_res.get("is_coasting", False))

        # Filter eliminasi false alarm: Abaikan vonis NG saat kinematic coasting (0.35) atau span belum penuh
        rib_xs = rib_res.get("xs_contour")
        rib_span = float(rib_xs[-1] - rib_xs[0]) if (rib_xs is not None and len(rib_xs) > 10) else 0.0
        min_span_thresh = min(750.0, 0.45 * frame_raw.shape[1]) if (frame_raw is not None and frame_raw.ndim >= 2) else 750.0
        if is_rib_ng and (rib_span < min_span_thresh or is_rib_coasting):
            is_rib_ng = False

        if not is_slat_ng and not is_rib_ng:
            return False

        updated = False

        # 3a. Update kandidat Best-Shot SLAT jika lebih parah
        if is_slat_ng:
            s_pv = float(slat_res.get("pv_flatness", 0.0) or 0.0)
            s_sag = float(slat_res.get("sagitta", 0.0) or 0.0)
            s_void = float(slat_res.get("center_void", 0) or 0) / 40.0
            slat_score = max(s_pv, s_sag, s_void)
            if self._best_slat_candidate is None or slat_score > self._best_slat_candidate["severity"]:
                full_ann = self.render_annotated_fullres(frame_raw, inspect_res, curr_slat_no, curr_rib_no)
                self._best_slat_candidate = {
                    "severity": slat_score,
                    "frame_raw": frame_raw.copy() if frame_raw is not None else None,
                    "frame_ann": full_ann if full_ann is not None else frame_ann,
                    "slat_res": dict(slat_res),
                    "slat_no": curr_slat_no,
                    "rib_no": curr_rib_no,
                    "y_seam_frame": slat_res.get("y_seam_frame"),
                    "captured_frame_idx": getattr(self, "_slat_frame_idx", 0),
                }
                updated = True

        # 3b. Update kandidat Best-Shot RIB jika lebih parah
        if is_rib_ng:
            r_dev = self._extract_rib_deviation(rib_res)
            r_sag = float(rib_res.get("sagitta", 0.0) or 0.0)
            rib_score = max(r_dev, r_sag)
            if self._best_rib_candidate is None or rib_score > self._best_rib_candidate["severity"]:
                full_ann = self.render_annotated_fullres(frame_raw, inspect_res, curr_slat_no, curr_rib_no)
                self._best_rib_candidate = {
                    "severity": rib_score,
                    "frame_raw": frame_raw.copy() if frame_raw is not None else None,
                    "frame_ann": full_ann if full_ann is not None else frame_ann,
                    "rib_res": dict(rib_res),
                    "slat_no": curr_slat_no,
                    "rib_no": curr_rib_no,
                }
                updated = True

        return updated

    def close(self):
        """Pembersihan resource saat ditutup."""
        try:
            if hasattr(self.rib_inspector, "finalize_pending_session"):
                self.rib_inspector.finalize_pending_session()
            self.flush_best_shot_if_needed(force_final=True)
        except Exception:
            pass
        if hasattr(self, "pose_watchdog") and self.pose_watchdog is not None:
            try:
                self.pose_watchdog.close()
            except Exception:
                pass
        if hasattr(self, "db_bridge") and self.db_bridge is not None:
            self.db_bridge.close()

    def inspect(self, frame, simulate_bending=False):
        """
        Menjalankan KEDUA inspeksi secara bebarengan (bersamaan pada setiap frame):
        1. HoughStitchingInspector memproses gerbang celah plat slat.
        2. HybridRibInspector memproses zona gerbang sirip.
        Kedua hasil dikembalikan bersamaan dengan thread-safety penuh (bebas race condition).
        """
        with self._inspect_lock:
            return self._inspect_internal(frame, simulate_bending=simulate_bending)

    def _inspect_internal(self, frame, simulate_bending=False):
        t0 = time.time()
        if frame is None or frame.size == 0:
            return {
                "status": "WAIT",
                "slat_res": None,
                "rib_res": None,
                "elapsed_ms": 0.0
            }

        h_f, w_f = frame.shape[:2]
        self._slat_frame_idx = getattr(self, "_slat_frame_idx", 0) + 1

        # Cek hot-reload jika operator sedang memakai mode threshold WEB Dashboard
        self._check_web_config_hot_reload()

        # Proses frame kalibrasi jika mode kalibrasi sedang aktif
        if getattr(self, "is_calibrating", False):
            self._process_calibration_frame(frame)
        if getattr(self, "calib_banner_timer", 0) > 0:
            self.calib_banner_timer -= 1
        if getattr(self, "profile_banner_timer", 0) > 0:
            self.profile_banner_timer -= 1

        # 1. Tentukan ROI Plat Slat adaptif jika belum diinisialisasi
        if self.slat_polygon is None:
            self.slat_polygon, trig_box, _ = get_adaptive_slat_polygon(w_f, h_f, self.config_path)
            self.slat_inspector.trigger_box = trig_box
            self._update_slat_y_bounds(w_f, h_f)

        if not hasattr(self, "slat_y_min") or self.slat_y_min is None:
            self._update_slat_y_bounds(w_f, h_f)

        # Stage 0 (Fase 4 GPU): Evaluasi Kesehatan Citra, Oklusi Ingot, & Kecepatan cuFFT Phase-Correlation
        health_res = self.health_monitor.evaluate(frame)
        if hasattr(self.rib_inspector, "sync_external_conveyor_velocity"):
            self.rib_inspector.sync_external_conveyor_velocity(
                health_res.get("conveyor_vy", 0.0),
                health_res.get("conveyor_direction", "UNKNOWN"),
            )

        # Stage 1 (Fase 4 GPU/ArUco): Watchdog Pergeseran & Getaran Holder Kamera
        if not getattr(self, "is_calibrating", False):
            pose_res = self.pose_watchdog.update(frame, gpu_thumb=self.health_monitor.last_gpu_thumb)
        else:
            pose_res = self.pose_watchdog.last_status

        if health_res.get("is_invalid", False) or pose_res.get("is_invalid", False):
            inv_code = health_res["invalid_code"] if health_res.get("is_invalid") else pose_res.get("invalid_code")
            inv_reason = health_res["invalid_reason"] if health_res.get("is_invalid") else pose_res.get("invalid_reason")
            # Bersihkan buffer transien sebelum INVALID agar tidak bocor menjadi False NG
            self.slat_history_samples.clear()
            if self._best_slat_candidate is not None:
                self._best_slat_candidate = None
            slat_inv = {
                "status": "INVALID",
                "display_status": "INVALID",
                "is_ok": False,
                "is_ng": False,
                "is_occluded": False,
                "is_conveyor_moving": False,
                "pv_flatness": 0.0,
                "sagitta": 0.0,
                "center_void": 0,
                "coverage": 0.0,
                "confidence": 0.0,
                "eval_mode": inv_reason,
            }
            rib_inv = {
                "status": "INVALID",
                "is_rib_present": False,
                "is_ok": False,
                "is_ng": False,
                "pv_flatness": 0.0,
                "sagitta": 0.0,
                "inner_dev": 0.0,
                "tip_dev": 0.0,
                "max_deviation": 0.0,
                "confidence": 0.0,
                "message": inv_reason,
                "yolo_boxes": [],
                "pre_gate_rib_boxes": [],
            }
            if self.db_bridge is not None and self.db_bridge.is_enabled() and hasattr(self.db_bridge, "emit_invalid_inspection"):
                self.db_bridge.emit_invalid_inspection(
                    frame_raw=frame,
                    frame_ann=None,
                    invalid_code=str(inv_code or "INVALID"),
                    invalid_reason=str(inv_reason or "Kondisi kamera tidak layak inspeksi"),
                    slat_number=self.slat_inspector.slat_count,
                )
            return {
                "status": "INVALID",
                "invalid_code": inv_code,
                "invalid_reason": inv_reason,
                "frame_health": health_res,
                "pose_status": pose_res,
                "conveyor_direction": health_res.get("conveyor_direction", "UNKNOWN"),
                "slat_res": slat_inv,
                "rib_res": rib_inv,
                "is_conveyor_moving": False,
                "elapsed_ms": (time.time() - t0) * 1000.0,
            }

        # Virtual Image Rectification (Metode 4) + Kompensasi Micro-Drift Watchdog (digabung 1x warp)
        rect_m = self._get_scaled_rect_matrix(w_f, h_f)
        pose_corr_m = self.pose_watchdog.get_scaled_corrective_matrix(w_f, h_f)
        eff_warp_m = rect_m
        if pose_corr_m is not None:
            if rect_m is None:
                eff_warp_m = pose_corr_m
            else:
                r3 = rect_m.astype(np.float64) if rect_m.shape == (3, 3) else np.vstack([rect_m.astype(np.float64), [0.0, 0.0, 1.0]])
                p3 = pose_corr_m.astype(np.float64) if pose_corr_m.shape == (3, 3) else np.vstack([pose_corr_m.astype(np.float64), [0.0, 0.0, 1.0]])
                eff_warp_m = (r3 @ p3).astype(np.float32)

        if eff_warp_m is not None:
            if eff_warp_m.shape == (3, 3):
                frame_inspected = cv2.warpPerspective(
                    frame,
                    eff_warp_m,
                    (w_f, h_f),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE
                )
            else:
                frame_inspected = cv2.warpAffine(
                    frame,
                    eff_warp_m,
                    (w_f, h_f),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE
                )
        else:
            frame_inspected = frame

        # 2. JALANKAN INSPEKSI SIRIP (HYBRID RIB PADA GPU) TERLEBIH DAHULU
        rib_res = self.rib_inspector.inspect_frame(
            frame_inspected,
            zone_poly=self.rib_zone,
            simulate_bending=simulate_bending
        )
        is_rib_present = rib_res.get("is_rib_present", False)
        active_box = rib_res.get("active_box")

        # 3. MASKING DINAMIS BILAH SIRIP DENGAN NEUTRAL SLAT TEXTURE FILL
        # Melenyapkan kontur bilah sirip & las kaki bilah secara absolut dari citra plat slat
        # dengan mengisi seluruh bounding box sirip menggunakan warna median plat slat sekitar.
        frame_for_slat = frame_inspected
        is_slat_occluded = False

        # Kumpulkan bounding box sirip yang beririsan dengan gerbang slat (hindari frame.copy() 1080p bila jauh di luar zona)
        y_min_gate = float(getattr(self, "slat_y_min", None) or (0.50 * h_f))
        y_max_gate = float(getattr(self, "slat_y_max", None) or (0.70 * h_f))
        gate_top = y_min_gate - 35
        gate_bot = y_max_gate + 35

        all_boxes_to_mask = []
        if active_box is not None and active_box[3] >= gate_top and active_box[1] <= gate_bot:
            all_boxes_to_mask.append(active_box)
        for yb in rib_res.get("yolo_boxes", []):
            b = yb.get("box") if isinstance(yb, dict) else yb
            if b is not None and b[3] >= gate_top and b[1] <= gate_bot and b not in all_boxes_to_mask:
                all_boxes_to_mask.append(b)
        for pb in rib_res.get("pre_gate_rib_boxes", []):
            b = pb.get("box") if isinstance(pb, dict) else pb
            if b is not None and b[3] >= gate_top and b[1] <= gate_bot and b not in all_boxes_to_mask:
                all_boxes_to_mask.append(b)

        is_foreign_ingot_occl = bool(health_res.get("is_foreign_occluded", False))
        foreign_box = health_res.get("foreign_box")
        if is_foreign_ingot_occl and foreign_box is not None:
            if foreign_box[3] >= gate_top and foreign_box[1] <= gate_bot:
                all_boxes_to_mask.append(foreign_box)
            # Tahan vonis NG rib sesaat bila gumpalan ingot terang menutupi area kerja
            if rib_res.get("is_ng", False):
                rib_res["status"] = "WAIT"
                rib_res["is_ng"] = False
                rib_res["is_ok"] = False
                rib_res["message"] = "WAIT (Oklusi Benda Asing / Ingot)"

        masked_y_intervals = []
        if len(all_boxes_to_mask) > 0:
            is_slat_occluded = True
            frame_for_slat = frame_inspected.copy()
            pad_bot_extra = max(22, int(round(0.032 * h_f)))
            feather_px = 6
            for box in all_boxes_to_mask:
                bx1, by1, bx2, by2 = box

                # Ambil sampel warna bersih plat slat dari bawah bilah sirip (di luar bayangan akar las)
                sample_y1 = min(h_f - 12, max(0, by2 + pad_bot_extra + 4))
                sample_y2 = min(h_f, sample_y1 + 12)
                sample_strip = frame_for_slat[sample_y1:sample_y2, max(0, bx1):min(w_f, bx2)]
                if sample_strip.size == 0 or sample_strip.shape[0] < 2:
                    sample_y1 = max(0, by1 - 24)
                    sample_y2 = min(h_f, sample_y1 + 12)
                    sample_strip = frame_for_slat[sample_y1:sample_y2, max(0, bx1):min(w_f, bx2)]

                if sample_strip.size > 0:
                    sub_strip = sample_strip[::2, ::8]
                    if sub_strip.size == 0:
                        sub_strip = sample_strip
                    med_col = np.median(sub_strip.reshape(-1, 3), axis=0).astype(np.uint8)
                else:
                    med_col = np.array([80, 80, 80], dtype=np.uint8)

                # Isi area bilah dan akar las dengan tekstur datar homogen + feathering batas atas/bawah
                pad_y1 = max(0, int(by1) - 6)
                pad_y2 = min(h_f, int(by2) + pad_bot_extra)
                pad_x1 = max(0, int(bx1) - 25)
                pad_x2 = min(w_f, int(bx2) + 25)
                masked_y_intervals.append((pad_y1, pad_y2))

                cv2.rectangle(
                    frame_for_slat,
                    (pad_x1, pad_y1),
                    (pad_x2, pad_y2),
                    (int(med_col[0]), int(med_col[1]), int(med_col[2])),
                    -1
                )
                # Feathering 6px di batas horizontal atas & bawah agar tidak timbul tepi Sobel-Y buatan
                if pad_x2 > pad_x1:
                    col_f = med_col.astype(np.float32)[None, None, :]
                    top_f1 = max(0, pad_y1 - feather_px // 2)
                    top_f2 = min(h_f, pad_y1 + feather_px // 2)
                    if top_f2 > top_f1 + 1:
                        alphas = np.linspace(0.0, 1.0, top_f2 - top_f1, dtype=np.float32)[:, None, None]
                        orig_top = frame_inspected[top_f1:top_f2, pad_x1:pad_x2].astype(np.float32)
                        frame_for_slat[top_f1:top_f2, pad_x1:pad_x2] = np.clip(
                            orig_top * (1.0 - alphas) + col_f * alphas, 0, 255
                        ).astype(np.uint8)
                    bot_f1 = max(0, pad_y2 - feather_px // 2)
                    bot_f2 = min(h_f, pad_y2 + feather_px // 2)
                    if bot_f2 > bot_f1 + 1:
                        alphas = np.linspace(1.0, 0.0, bot_f2 - bot_f1, dtype=np.float32)[:, None, None]
                        orig_bot = frame_inspected[bot_f1:bot_f2, pad_x1:pad_x2].astype(np.float32)
                        frame_for_slat[bot_f1:bot_f2, pad_x1:pad_x2] = np.clip(
                            orig_bot * (1.0 - alphas) + col_f * alphas, 0, 255
                        ).astype(np.uint8)

        # 4. JALANKAN INSPEKSI PLAT SLAT DENGAN CITRA TERMASKING
        slat_res = self.slat_inspector.inspect_frame(frame_for_slat, polygon_pts=self.slat_polygon)

        # 5. ZONAL GATING & MULTI-FRAME CONSENSUS (ELIMINASI FALSE NG TRANSISI PLAT)
        # Menghilangkan lonjakan cacat semu saat plat pertama kali masuk ke gerbang kanonikal.
        y_seam = slat_res.get("y_seam", 0.0)
        seam_energy = slat_res.get("seam_energy", 0.0)
        is_trans = slat_res.get("is_transition", True)
        is_rib_base_occ = bool(slat_res.get("is_rib_base_occluded", False))
        xs = slat_res.get("xs_contour")
        ys = slat_res.get("ys_contour")

        # Hitung posisi Y seam di koordinat frame kamera untuk memeriksa oklusi oleh box Rib
        poly_px_slat = slat_res.get("polygon_px")
        if poly_px_slat is not None and len(poly_px_slat) >= 4:
            top_mid_y = 0.5 * (float(poly_px_slat[0][1]) + float(poly_px_slat[1][1]))
            bot_mid_y = 0.5 * (float(poly_px_slat[2][1]) + float(poly_px_slat[3][1]))
        else:
            top_mid_y = y_min_gate
            bot_mid_y = y_max_gate
        seam_frac = float(y_seam) / float(max(1, self.slat_inspector.canonical_h - 1))
        y_seam_frame = top_mid_y + seam_frac * (bot_mid_y - top_mid_y)
        slat_res["y_seam_frame"] = float(y_seam_frame)

        seam_tol_px = max(5.0, 0.005 * h_f)
        seam_masked_occluded = any(
            (my1 - seam_tol_px) <= y_seam_frame <= (my2 + seam_tol_px)
            for (my1, my2) in masked_y_intervals
        )
        slat_res["is_seam_masked_occluded"] = bool(seam_masked_occluded)

        # Pembagian Zona Inspeksi Celah Slat:
        # - Probing Zone (36 <= y_seam < 58 px): Celah baru masuk gerbang -> status WAIT (tahan penilaian)
        # - Prime Measurement Zone (58 <= y_seam <= 86 px): Celah tepat di tengah stabil -> Konsensus Median
        # - Exiting / Occluded Zone (y_seam > 86 px atau tertutup sirip) -> Evaluasi Latch Peak-Hold
        in_probing_zone = (36.0 <= y_seam < 58.0)
        in_prime_zone = (58.0 <= y_seam <= 86.0)

        has_valid_seam = (
            (not is_trans) and
            (not is_rib_base_occ) and
            (not seam_masked_occluded) and
            (xs is not None) and
            (len(xs) > 30) and
            (seam_energy >= 22.0)
        )

        if in_probing_zone and not is_rib_base_occ and not seam_masked_occluded:
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

            # Simpan sampel bersih selama celah berada di zona tengah optimal (58-86 px) dengan bobot kualitas
            _slat_w = max(0.20, float(cov) * min(2.0, float(seam_energy) / 30.0))
            self.slat_history_samples.append({
                "pv": pv,
                "sagitta": sag,
                "coverage": cov,
                "center_void": void,
                "y_seam": y_seam,
                "energy": seam_energy,
                "weight": _slat_w,
                "xs": xs,
                "ys": ys,
                "polygon_px": slat_res.get("polygon_px")
            })

            # Evaluasi berbasis konsensus multi-frame berbobot kualitas (minimal 3 sampel stabil di zona prime)
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
                all_ws = [s.get("weight", 1.0) for s in self.slat_history_samples]

                eval_sag = self.rib_inspector._weighted_median(all_sags, all_ws)
                eval_pv = self.rib_inspector._weighted_median(all_pvs, all_ws)
                eval_void = int(round(self.rib_inspector._weighted_median(all_voids, all_ws))) if all_voids else 0

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
            if is_rib_base_occ or seam_masked_occluded or is_slat_occluded:
                self._last_rib_slat_occl_frame = self._slat_frame_idx

            if (
                (is_rib_base_occ or seam_masked_occluded)
                and self.latched_slat_data is not None
                and self.latched_slat_data.get("is_ng", False)
                and float(self.latched_slat_data.get("pv_flatness", 0.0) or 0.0) <= self.slat_inspector.max_flatness_pv
                and float(self.latched_slat_data.get("sagitta", 0.0) or 0.0) <= self.slat_inspector.max_sagitta
            ):
                # Bersihkan latch void-only yang terkontaminasi kaki rib agar tidak ditahan 28 frame
                self.latched_slat_data = None
                self.latched_hold_frames = 0

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
                is_void_only_ng = bool(is_flat_ok and is_curv_ok and (not is_void_ok))

                if (is_rib_base_occ or seam_masked_occluded) and is_void_only_ng:
                    # Jangan kunci void-only NG ke latch saat oklusi rib terjadi (mencegah bocornya False Slat NG)
                    self.slat_history_samples.clear()
                    self.latched_slat_data = None
                    self.latched_hold_frames = 0
                else:
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
            elif is_rib_base_occ or seam_masked_occluded:
                slat_res["status"] = "WAIT"
                slat_res["display_status"] = "WAIT"
                slat_res["is_ok"] = False
                slat_res["is_ng"] = False
                slat_res["eval_mode"] = "WAIT (Oklusi Bagian Bawah Rib / Sirip)"

        if is_foreign_ingot_occl:
            self.slat_history_samples.clear()
            if self.latched_slat_data is not None and self.latched_slat_data.get("is_ng", False):
                self.latched_slat_data = None
                self.latched_hold_frames = 0
            if slat_res.get("is_ng", False):
                slat_res["status"] = "WAIT"
                slat_res["display_status"] = "WAIT"
                slat_res["is_ng"] = False
                slat_res["is_ok"] = False
                slat_res["eval_mode"] = "WAIT (Oklusi Benda Asing / Ingot)"

        slat_res["is_occluded"] = bool(is_slat_occluded or seam_masked_occluded or is_rib_base_occ or is_foreign_ingot_occl)

        # Loop-Aware Anomaly Memory (total_slats_in_loop = 16): lacak anomali berulang antar-putaran tanpa menunda NG
        loop_n = max(1, int(getattr(self, "total_slats_in_loop", 16)))
        curr_rib_c = int(getattr(self.rib_inspector, "rib_count", 0))
        if curr_rib_c > 0 and curr_rib_c != self._last_tracked_rib_loop_id and rib_res.get("status") in ("OK", "NG"):
            self._last_tracked_rib_loop_id = curr_rib_c
            r_slot = ("RIB", curr_rib_c % loop_n)
            if rib_res.get("is_ng", False) or rib_res.get("review_anomaly", False):
                self._loop_anomaly_memory[r_slot] = int(self._loop_anomaly_memory.get(r_slot, 0)) + 1
            else:
                self._loop_anomaly_memory[r_slot] = max(0, int(self._loop_anomaly_memory.get(r_slot, 0)) - 1)
        if curr_rib_c > 0:
            rib_res["repeat_pass_anomaly"] = int(self._loop_anomaly_memory.get(("RIB", curr_rib_c % loop_n), 0)) >= 2

        curr_slat_c = int(getattr(self.slat_inspector, "slat_count", 0))
        if curr_slat_c > 0 and curr_slat_c != self._last_tracked_slat_loop_id and slat_res.get("status") in ("OK", "NG"):
            self._last_tracked_slat_loop_id = curr_slat_c
            s_slot = ("SLAT", curr_slat_c % loop_n)
            if slat_res.get("is_ng", False):
                self._loop_anomaly_memory[s_slot] = int(self._loop_anomaly_memory.get(s_slot, 0)) + 1
            else:
                self._loop_anomaly_memory[s_slot] = max(0, int(self._loop_anomaly_memory.get(s_slot, 0)) - 1)
        if curr_slat_c > 0:
            slat_res["repeat_pass_anomaly"] = int(self._loop_anomaly_memory.get(("SLAT", curr_slat_c % loop_n), 0)) >= 2

        # Sinkronkan lifecycle purge/flush kandidat NG secara independen di setiap frame
        if self.db_bridge is not None and self.db_bridge.is_enabled():
            self._sync_recording_lifecycle(slat_res, rib_res)

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
            "invalid_code": None,
            "invalid_reason": "",
            "frame_health": health_res,
            "pose_status": pose_res,
            "conveyor_direction": health_res.get("conveyor_direction", "UNKNOWN"),
            "slat_res": slat_res,
            "rib_res": rib_res,
            "is_conveyor_moving": is_conveyor_moving,
            "elapsed_ms": elapsed_ms
        }

    def _render_alignment_guides(self, canvas, w_f, h_f):
        """
        Merender garis panduan referensi (Golden Reticle Alignment Guide)
        pada canvas preview (1000 x 520 px).
        Menampilkan rel kiri, rel kanan, sumbu simetri tengah, dan horizon celah plat
        berdasarkan acuan emas (Kamera posisi 3 dimmer 2 / WIN_20261005_09_01_08_Pro).
        """
        if canvas is None or canvas.size == 0:
            return
        ch, cw = canvas.shape[:2]
        sx = float(cw) / float(w_f if w_f > 0 else 1920)
        sy = float(ch) / float(h_f if h_f > 0 else 1080)

        lx_top = int(round(self._golden_rail_x("left", 0.0, w_f, h_f) * sx))
        lx_bot = int(round(self._golden_rail_x("left", float(h_f), w_f, h_f) * sx))
        rx_top = int(round(self._golden_rail_x("right", 0.0, w_f, h_f) * sx))
        rx_bot = int(round(self._golden_rail_x("right", float(h_f), w_f, h_f) * sx))
        lx = int(round(getattr(self, "ref_left_rail_px", 188.0) * sx))
        rx = int(round(getattr(self, "ref_right_rail_px", 1759.0) * sx))
        cx = int(round(getattr(self, "ref_center_x", 960.0) * sx))
        tilt = float(getattr(self, "ref_tilt_deg", 0.90))

        guide_col = (0, 235, 255)    # Amber / Emas Kanonikal
        center_col = (255, 205, 0)   # Cyan terang untuk sumbu tengah

        # 1. Garis Trapesium Rel Kiri & Kanan (Mengikuti Perspektif Asli Posisi 3)
        cv2.line(canvas, (lx_top, 0), (lx_bot, ch), guide_col, 2, cv2.LINE_AA)
        cv2.line(canvas, (rx_top, 0), (rx_bot, ch), guide_col, 2, cv2.LINE_AA)
        for y_tick in range(40, ch - 20, 60):
            alpha_t = float(y_tick) / float(max(1, ch))
            lx_t = int(round(lx_top + alpha_t * (lx_bot - lx_top)))
            rx_t = int(round(rx_top + alpha_t * (rx_bot - rx_top)))
            cv2.line(canvas, (lx_t - 7, y_tick), (lx_t + 7, y_tick), guide_col, 2, cv2.LINE_AA)
            cv2.line(canvas, (rx_t - 7, y_tick), (rx_t + 7, y_tick), guide_col, 2, cv2.LINE_AA)

        # 3. Sumbu Simetri Tengah (Centerline) - Garis Putus-putus
        for y_dash in range(0, ch, 20):
            cv2.line(canvas, (cx, y_dash), (cx, min(ch, y_dash + 10)), center_col, 1, cv2.LINE_AA)

        # Crosshair di Prime Inspection Zone (Y ~ 200 px pada kanvas 520)
        y_prime_center = int(round(395.0 * sy))
        cv2.drawMarker(canvas, (cx, y_prime_center), center_col, cv2.MARKER_CROSS, 28, 2, cv2.LINE_AA)

        # 4. Garis Horizon Celah Plat Acuan (Golden Seam Tilt)
        rad = math.radians(tilt)
        tan_t = math.tan(rad)
        for x_dash in range(lx, rx, 25):
            x2_d = min(rx, x_dash + 14)
            y1_d = int(round(y_prime_center + (x_dash - cx) * tan_t))
            y2_d = int(round(y_prime_center + (x2_d - cx) * tan_t))
            cv2.line(canvas, (x_dash, y1_d), (x2_d, y2_d), (50, 220, 100), 1, cv2.LINE_AA)

        # 5. Label Pills agar teks 100% kontras dan terbaca
        # Badge Label Rel Kiri
        lbl_l = f"REF KIRI ({int(round(self.ref_left_rail_px))}px)"
        cv2.rectangle(canvas, (lx - 10, ch - 30), (lx + 130, ch - 8), (18, 22, 28), -1)
        cv2.rectangle(canvas, (lx - 10, ch - 30), (lx + 130, ch - 8), guide_col, 1)
        cv2.putText(canvas, lbl_l, (lx - 4, ch - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, guide_col, 1, cv2.LINE_AA)

        # Badge Label Rel Kanan
        lbl_r = f"REF KANAN ({int(round(self.ref_right_rail_px))}px)"
        cv2.rectangle(canvas, (rx - 138, ch - 30), (rx + 10, ch - 8), (18, 22, 28), -1)
        cv2.rectangle(canvas, (rx - 138, ch - 30), (rx + 10, ch - 8), guide_col, 1)
        cv2.putText(canvas, lbl_r, (rx - 132, ch - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, guide_col, 1, cv2.LINE_AA)

        # Badge Sumbu Tengah
        lbl_c = f"SUMBU TENGAH ({int(round(self.ref_center_x))}px)"
        cv2.rectangle(canvas, (cx - 75, ch - 30), (cx + 75, ch - 8), (18, 22, 28), -1)
        cv2.rectangle(canvas, (cx - 75, ch - 30), (cx + 75, ch - 8), center_col, 1)
        cv2.putText(canvas, lbl_c, (cx - 70, ch - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, center_col, 1, cv2.LINE_AA)

        # Header Badge Reticle Aktif di Bagian Tengah Atas
        cv2.rectangle(canvas, (cx - 120, 12), (cx + 120, 36), (15, 20, 26), -1)
        cv2.rectangle(canvas, (cx - 120, 12), (cx + 120, 36), guide_col, 1)
        cv2.putText(canvas, "📐 RETICLE ACUAN EMAS [G]", (cx - 110, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, guide_col, 1, cv2.LINE_AA)

    def render_combined_dashboard(
        self,
        frame,
        inspect_result,
        fps=30.0,
        is_paused=False,
        frame_idx=0,
        total_frames=0,
        sync_realtime=True,
        video_time_str="",
        speed_label=""
    ):
        """
        Merender visualisasi terpadu pada resolusi asli frame kamera dengan proteksi thread lock.
        """
        with self._render_lock:
            return self._render_combined_dashboard_internal(
                frame, inspect_result, fps, is_paused, frame_idx, total_frames,
                sync_realtime, video_time_str=video_time_str, speed_label=speed_label
            )

    def _render_combined_dashboard_internal(
        self,
        frame,
        inspect_result,
        fps=30.0,
        is_paused=False,
        frame_idx=0,
        total_frames=0,
        sync_realtime=True,
        video_time_str="",
        speed_label=""
    ):
        """
        Merender visualisasi terpadu pada resolusi asli frame kamera:
        1. Gerbang Plat Slat + Garis Celah Terstitching (Terproyeksi Persis di Celah Fisik Plat).
        2. Bounding Box Sirip + Garis Puncak Sirip (Ridge Tracking) menempel di Bilah Fisik.
        3. Live Radar PiP di sudut kanan atas.
        4. Panel Dashboard Bawah (1000 x 300 px) dengan Kartu Metrologi Plat & Sirip Lengkap.
        """
        h_f, w_f = frame.shape[:2]

        # Cek hot-reload jika operator sedang memakai mode threshold WEB Dashboard
        self._check_web_config_hot_reload()
        if self._screen_buf is None:
            self._screen_buf = np.zeros((820, 1000, 3), dtype=np.uint8)

        sx = 1000.0 / float(w_f)
        sy = 520.0 / float(h_f)
        main_resized = self._screen_buf[:520, :]
        if getattr(self, "rect_matrix", None) is not None:
            if self.rect_matrix.shape == (3, 3):
                m_disp = self.rect_matrix.copy()
                m_disp[0, :] *= sx
                m_disp[1, :] *= sy
                cv2.warpPerspective(
                    frame,
                    m_disp,
                    (1000, 520),
                    dst=main_resized,
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE
                )
            else:
                m_disp = self.rect_matrix.copy()
                m_disp[0, :] *= sx
                m_disp[1, :] *= sy
                cv2.warpAffine(
                    frame,
                    m_disp,
                    (1000, 520),
                    dst=main_resized,
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE
                )
        else:
            cv2.resize(frame, (1000, 520), dst=main_resized, interpolation=cv2.INTER_LINEAR)

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

            # Indikator pasif tipis khusus Plat A yang sudah [LOCKED] (~0.02 ms, tanpa turun FPS)
            for lb in (rib_res.get("locked_boxes") or [])[:1]:
                if lb and len(lb) >= 4:
                    lx1, ly1, lx2, ly2 = int(lb[0] * sx), int(lb[1] * sy), int(lb[2] * sx), int(lb[3] * sy)
                    cv2.rectangle(main_resized, (lx1, ly1), (lx2, ly2), (160, 175, 190), 1, cv2.LINE_AA)
                    cv2.putText(main_resized, "LOCKED", (lx1, max(18, ly1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 175, 190), 1, cv2.LINE_AA)

            # Gambar Bounding Box Sirip jika terdeteksi
            box = rib_res.get("active_box")
            if is_rib_present and box:
                bx1, by1, bx2, by2 = box
                rx1, ry1, rx2, ry2 = int(bx1 * sx), int(by1 * sy), int(bx2 * sx), int(by2 * sy)
                w_px = bx2 - bx1
                h_px = by2 - by1
                scale_x = float(rib_res.get("rib_raw_mm_per_px", 0.0) or getattr(self, "scale_factor_mm_per_px", 0.515))
                scale_y = float(rib_res.get("rib_raw_mm_per_px", 0.0) or getattr(self, "scale_factor_y_mm_per_px", 0.442))
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

        # ======================================================================
        # 3B. LAYER PANDUAN REFERENSI REL & SUMBU EMAS (GOLDEN ALIGNMENT RETICLE)
        # ======================================================================
        if getattr(self, "show_alignment_guides", False) or getattr(self, "is_calibrating", False):
            self._render_alignment_guides(main_resized, w_f, h_f)

        # Indikator Banner PAUSED jika video dijeda
        if is_paused:
            bw, bh = 510, 52
            bx = 15
            by = 12
            cv2.rectangle(main_resized, (bx, by), (bx + bw, by + bh), (20, 110, 220), -1)
            cv2.rectangle(main_resized, (bx, by), (bx + bw, by + bh), (40, 180, 255), 2, cv2.LINE_AA)
            cv2.putText(main_resized, f"|| PAUSED | Frame {frame_idx}/{total_frames}", (bx + 14, by + 24),
                        cv2.FONT_HERSHEY_DUPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(main_resized, "[SPACE]: Play | [N/P]: +-1f | [,/.]: +-10f | [;/']: +-60f", (bx + 14, by + 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 240, 200), 1, cv2.LINE_AA)

        # Indikator Banner Kalibrasi Konveyor Otomatis
        if getattr(self, "is_calibrating", False):
            cal_y = 70 if is_paused else 15
            cv2.rectangle(main_resized, (15, cal_y), (600, cal_y + 42), (0, 130, 255), -1)
            cv2.rectangle(main_resized, (15, cal_y), (600, cal_y + 42), (255, 255, 255), 2)
            cv2.putText(main_resized, f"KALIBRASI: {self.calib_status_msg}", (25, cal_y + 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 2, cv2.LINE_AA)
        elif getattr(self, "calib_banner_timer", 0) > 0:
            cal_y = 70 if is_paused else 15
            cv2.rectangle(main_resized, (15, cal_y), (720, cal_y + 42), (0, 160, 40), -1)
            cv2.rectangle(main_resized, (15, cal_y), (720, cal_y + 42), (255, 255, 255), 2)
            cv2.putText(main_resized, self.calib_status_msg, (25, cal_y + 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 2, cv2.LINE_AA)
        elif getattr(self, "profile_banner_timer", 0) > 0:
            cal_y = 70 if is_paused else 15
            p_bg = (0, 150, 40) if "DEFAULT" in getattr(self, "active_profile_name", "") else (0, 120, 240)
            cv2.rectangle(main_resized, (15, cal_y), (720, cal_y + 42), p_bg, -1)
            cv2.rectangle(main_resized, (15, cal_y), (720, cal_y + 42), (255, 255, 255), 2)
            cv2.putText(main_resized, self.profile_banner_text, (25, cal_y + 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 2, cv2.LINE_AA)

        # Indikator Banner INVALID (PRD §25.6) jika lingkungan/kamera tidak layak inspeksi
        if overall_status == "INVALID":
            inv_txt = str(inspect_result.get("invalid_reason") or "INVALID: Kondisi Kamera / Lingkungan Tidak Layak Inspeksi")
            cv2.rectangle(main_resized, (15, 14), (985, 64), (135, 0, 145), -1)
            cv2.rectangle(main_resized, (15, 14), (985, 64), (255, 120, 255), 2, cv2.LINE_AA)
            cv2.putText(main_resized, inv_txt[:88], (28, 46), cv2.FONT_HERSHEY_DUPLEX, 0.56, (255, 255, 255), 2, cv2.LINE_AA)

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
        elif overall_status == "INVALID":
            badge_bg = (145, 0, 155)
        elif overall_status == "HOLD":
            badge_bg = (30, 80, 160)
        else:
            badge_bg = (140, 90, 30)

        cv2.rectangle(hud, (15, 12), (180, 72), badge_bg, -1)
        if overall_status == "INVALID":
            cv2.putText(hud, "INVALID", (22, 53), cv2.FONT_HERSHEY_DUPLEX, 1.05, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            stat_x = 55 if len(overall_status) == 2 else (35 if len(overall_status) == 4 else 45)
            cv2.putText(hud, overall_status, (stat_x, 56), cv2.FONT_HERSHEY_DUPLEX, 1.35, (255, 255, 255), 3, cv2.LINE_AA)

        # Header Status Terpadu (DIBESARKAN) & Badge Profil
        prof_tag = "[DEFAULT STABIL]" if "DEFAULT" in getattr(self, "active_profile_name", "") else "[HASIL KALIBRASI]"
        pose_st = inspect_result.get("pose_status", {}) or {}
        if pose_st.get("has_correction"):
            prof_tag += " [AUTO-POSE]"
        prof_col = (100, 255, 140) if "DEFAULT" in getattr(self, "active_profile_name", "") else (255, 200, 50)
        cam_hdr = getattr(self, "top_camera_id", "CAM-01")
        cv2.putText(hud, f"{cam_hdr} TOP INSPECTION PIPELINE {prof_tag}",
                    (195, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.60, prof_col, 2, cv2.LINE_AA)

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
        if getattr(self.rib_inspector, "tracker_engine", "LEGACY") == "TIER4_HOUGH_STITCH":
            eng_badge = "HOUGH"
        elif getattr(self.rib_inspector, "tracker_engine", "LEGACY") == "TIER1_VITERBI":
            eng_badge = "VITERBI"
        elif getattr(self.rib_inspector, "tracker_engine", "LEGACY") == "LEGACY_1DCNN":
            eng_badge = "1DCNN+MAHAL"
        else:
            eng_badge = "SOBEL"
        src_badge = "WEB" if getattr(self, "threshold_source_mode", "DATASET") == "WEB" else "DATA"
        cv2.putText(hud, f"METROLOGI BILAH SIRIP (HYBRID | {eng_badge} | [H]:{src_badge})", (505, 118),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.47, (0, 255, 255), 2, cv2.LINE_AA)

        r_pv = rib_res.get("pv_flatness", 0.0) if rib_res else 0.0
        r_sag = rib_res.get("sagitta", 0.0) if rib_res else 0.0
        r_dev = rib_res.get("inner_dev", 0.0) if rib_res else 0.0
        r_tip = rib_res.get("tip_dev", 0.0) if rib_res else 0.0
        r_stat = rib_res.get("status", "WAIT") if rib_res else "WAIT"
        r_sev = rib_res.get("ng_severity", "OK") if rib_res else "OK"
        r_stat_disp = f"{r_stat}:{r_sev}" if (r_stat == "NG" and r_sev in ("LOW", "MEDIUM", "HIGH")) else r_stat

        rpv_c = (0, 255, 0) if r_pv <= self.rib_inspector.pv_thresh else (0, 0, 255)
        rsag_c = (0, 255, 0) if r_sag <= self.rib_inspector.sagitta_thresh else (0, 0, 255)
        rib_scale = float(rib_res.get("rib_mm_per_px", 750.0 / 1200.0) or (750.0 / 1200.0)) if rib_res else (750.0 / 1200.0)
        r_pv_mm = r_pv * rib_scale
        r_sag_mm = r_sag * rib_scale
        r_dev_mm = r_dev * rib_scale
        dev_lim_mm = self.rib_inspector.dev_thresh * rib_scale
        r_tip_mm = r_tip * rib_scale
        tip_lim_mm = self.rib_inspector.tip_thresh * rib_scale

        cv2.putText(hud, f"Flatness PV : {r_pv_mm:4.1f} mm | Sagitta: {r_sag_mm:.1f} mm",
                    (505, 139), cv2.FONT_HERSHEY_SIMPLEX, 0.46, rpv_c, 1, cv2.LINE_AA)
        rdev_c = (0, 255, 0) if r_dev <= self.rib_inspector.dev_thresh else (0, 0, 255)
        rtip_c = (0, 255, 0) if r_tip <= self.rib_inspector.tip_thresh else (0, 0, 255)
        cv2.putText(hud, f"Dev Badan: {r_dev_mm:4.1f}mm [Maks: {dev_lim_mm:.1f}mm] | Tip: {r_tip_mm:4.1f}mm",
                    (505, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.44, rdev_c, 1, cv2.LINE_AA)
        rib_conf = float(rib_res.get("confidence", 0.0) or 0.0)
        conf_r_str = f" | Conf: {rib_conf:.2f}" if rib_conf > 0 else ""
        cv2.putText(hud, f"QC SIRIP RIB: #{self.rib_inspector.rib_count:02d}{conf_r_str} | OK: {self.rib_inspector.ok_count} | NG: {self.rib_inspector.ng_count} | [{r_stat_disp}]",
                    (505, 178), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (180, 240, 180), 2, cv2.LINE_AA)

        # ----------------------------------------------------------------------
        # FOOTER PANDUAN KONTROL & PINTASAN KEYBOARD (DIBESARKAN)
        # ----------------------------------------------------------------------
        cv2.rectangle(hud, (15, 192), (985, 292), (22, 22, 28), -1)
        cv2.rectangle(hud, (15, 192), (985, 292), (45, 45, 55), 1)
        cv2.putText(hud, "PANDUAN KONTROL & PINTASAN KEYBOARD (HOTKEYS):", (25, 213),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 180, 220), 2, cv2.LINE_AA)
        cv2.putText(hud, "[SPACE]: Pause | [,/.]: -/+10f | [;/']: -/+60f | [[/]]: -/+10s | [-/=]: -/+60s | [Z]: Loncat", (25, 235),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(hud, "[X]: Speed | [I]: Skip-Diam | [T]: Sync | [E]: Engine | [K]: Align | [G]: Acuan | [P]: Profil | [R]: Reset | [D]: DB | [Q]: Keluar", (25, 257),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1, cv2.LINE_AA)
        pos_parts = [f"File: {inspect_result.get('filename', '-')}", f"Frame: {frame_idx}/{total_frames}"]
        if video_time_str:
            pos_parts.append(f"Waktu: {video_time_str}")
        if speed_label:
            pos_parts.append(speed_label)
        cv2.putText(hud, " | ".join(pos_parts), (25, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 210, 140), 1, cv2.LINE_AA)

        # Selesai: kembalikan buffer terpadu yang sudah terisi penuh (thread-safe copy)
        return self._screen_buf.copy()


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
    parser.add_argument("--start-time", type=str, default=None,
                        help="Mulai pemutaran video pada waktu tertentu (contoh: 05:30 atau 330)")
    parser.add_argument("--speed", type=int, default=1, choices=list(SPEED_STEPS),
                        help="Pengali kecepatan pemutaran file video awal (1, 2, 4, 8, atau 16)")
    parser.add_argument("--skip-idle", action="store_true", default=False,
                        help="Aktifkan Auto-Skip segmen conveyor diam sejak awal")
    args = parser.parse_args()

    print("\n" + "=" * 78)
    print(f"🚀 MEMULAI PIPELINE TERPADU INSPEKSI TAMPAK ATAS KONVEYOR")
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

    # Inisialisasi daftar video playlist
    sorted_videos = []

    # Cek sumber dari argumen atau dari .env
    current_source = args.source
    if not current_source:
        env_top = os.environ.get("TOP_CAMERA_SOURCE")
        if env_top:
            if env_top.isdigit() or env_top.startswith(("rtsp://", "http://", "https://", "nvarguscamerasrc", "v4l2src")):
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

    # Jika sumber berupa file video, pastikan folder tersebut menjadi basis playlist
    if isinstance(current_source, str) and os.path.isfile(current_source):
        src_dir = os.path.dirname(os.path.abspath(current_source))
        if os.path.exists(src_dir):
            for f in sorted(os.listdir(src_dir)):
                if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    p = os.path.abspath(os.path.join(src_dir, f))
                    if p not in sorted_videos:
                        sorted_videos.append(p)
        if current_source not in sorted_videos:
            sorted_videos.append(os.path.abspath(current_source))

    # Untuk sumber video: gunakan AsyncFrameGrabber agar frame decode non-blocking
    grabber = None
    cap = None

    # Deteksi apakah sumber adalah gambar statis atau video
    _cap_probe, is_image, static_frame, total_frames, fname = open_media_source(current_source)
    if _cap_probe is not None:
        _cap_probe.release()  # tutup probe segera; grabber akan membuka ulang

    speed_mult = max(1, int(getattr(args, "speed", 1) or 1))
    idle_skipper = IdleConveyorSkipper(enabled=bool(getattr(args, "skip_idle", False)))
    pending_seek = {"frame": None, "from_trackbar": False}

    if is_image:
        pass
    else:
        grabber = AsyncFrameGrabber(current_source, buffer_size=2)
        if not grabber.isOpened():
            print(f"[ERROR] Gagal membuka sumber media: {current_source}")
            grabber.release()
            return
        total_frames = grabber.total_frames
        grabber.set_speed_multiplier(speed_mult)
        if args.start_time and not grabber.is_live and total_frames > 0:
            s_sec = parse_time_to_seconds(args.start_time)
            if s_sec is not None:
                init_f = max(0, min(total_frames - 1, int(round(s_sec * grabber.fps_native))))
                pending_seek["frame"] = init_f
                print(f"[*] [START-TIME] Melompat awal ke {format_seconds_mmss(s_sec)} (Frame {init_f}/{total_frames})")
        print("[✔] AsyncFrameGrabber aktif: frame decode non-blocking di background thread")

    win_name = (
        f"{pipeline.top_camera_id} TOP INSPECTION | [SPACE]: Pause | [,/.]: +-10f | [;/']: +-60f | "
        f"[[/]]: +-10s | [-/=]: +-60s | [X]: Speed | [I]: Skip-Diam | [Z]: Loncat | [Q]: Keluar"
    )
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1200, 920)

    trackbar_name = "Seek (s)"
    trackbar_state = {"attached": False, "ignore": False, "last_set_sec": 0}

    def _is_seekable_video():
        return (not is_image) and (grabber is not None) and (not grabber.is_live) and (total_frames > 0)

    def _on_seek_trackbar(sec_val):
        if trackbar_state["ignore"] or int(sec_val) == int(trackbar_state["last_set_sec"]) or not _is_seekable_video():
            return
        tgt = max(0, min(total_frames - 1, int(round(float(sec_val) * grabber.fps_native))))
        pending_seek["frame"] = tgt
        pending_seek["from_trackbar"] = True

    def _sync_trackbar_bounds():
        if not _is_seekable_video():
            return
        max_sec = max(1, int(total_frames / max(1e-5, grabber.fps_native)))
        if not trackbar_state["attached"]:
            cv2.createTrackbar(trackbar_name, win_name, 0, max_sec, _on_seek_trackbar)
            trackbar_state["attached"] = True
        else:
            try:
                cv2.setTrackbarMax(trackbar_name, win_name, max_sec)
            except Exception:
                pass

    def _update_trackbar_pos(frame_pos: int):
        if trackbar_state["attached"] and _is_seekable_video():
            cur_sec = int(max(0, round(float(frame_pos) / max(1e-5, grabber.fps_native))))
            if cur_sec == trackbar_state["last_set_sec"]:
                return
            trackbar_state["last_set_sec"] = cur_sec
            trackbar_state["ignore"] = True
            try:
                cv2.setTrackbarPos(trackbar_name, win_name, cur_sec)
            except Exception:
                pass
            trackbar_state["ignore"] = False

    _sync_trackbar_bounds()

    is_paused = False
    step_forward = False
    cached_frame = None
    fps_val = 30.0
    sync_realtime = True
    current_vid_idx = sorted_videos.index(current_source) if current_source in sorted_videos else 0
    t_last_frame = time.time()
    last_video_pos = 0
    cur_pos = 0

    def _load_media_file(new_file: str):
        nonlocal grabber, is_image, static_frame, total_frames, cached_frame
        nonlocal current_source, fname, last_video_pos, cur_pos
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
            grabber.set_speed_multiplier(speed_mult)
            is_image = False
            total_frames = grabber.total_frames
            ret_f0, f0 = grabber.read(timeout=1.0)
            if ret_f0 and f0 is not None:
                cached_frame = f0.copy()
        current_source = new_file
        fname = new_fn
        last_video_pos = 0
        cur_pos = 0
        pending_seek["frame"] = None
        idle_skipper.reset_scan_state()
        pipeline.reset_counters()
        _sync_trackbar_bounds()
        _update_trackbar_pos(0)

    def _handle_key(key_code: int, calib_frame=None) -> str:
        nonlocal is_paused, step_forward, sync_realtime, speed_mult, current_vid_idx
        if key_code in (ord('q'), ord('Q'), 27):
            print("\n[*] Menutup pipeline inspeksi...")
            return "BREAK"
        if key_code == 32:  # [SPACE]
            is_paused = not is_paused
            idle_skipper.reset_scan_state()
            print(f"[*] Status: {'PAUSED' if is_paused else 'RUNNING'}")
            return "HANDLED"
        if key_code in (ord('d'), ord('D')):
            st = pipeline.toggle_db()
            print(f"[*] SAKLAR DATABASE LOGGING: {'AKTIF (RECORDING)' if st else 'MATI (DRY-RUN)'}")
            return "HANDLED"
        if key_code in (ord('t'), ord('T')):
            sync_realtime = not sync_realtime
            print(f"[*] Mode Kecepatan Video: {'1.0x Realtime Sync' if sync_realtime else 'TURBO (Uncapped FPS)'}")
            return "HANDLED"
        if key_code in (ord('e'), ord('E')):
            new_eng = pipeline.rib_inspector.toggle_tracker_engine()
            print(f"[*] 🔄 SAKELAR TRACKER ENGINE SIRIP: [{new_eng}]")
            return "HANDLED"
        if key_code in (ord('x'), ord('X'), ord('>')):
            speed_mult = next_speed_step(speed_mult)
            if grabber is not None:
                grabber.set_speed_multiplier(speed_mult)
            print(f"[*] [SPEED] Kecepatan pemutaran diubah ke: {speed_mult}x")
            return "HANDLED"
        if key_code in (ord('i'), ord('I')):
            sk_on = idle_skipper.toggle()
            print(f"[*] [AUTO-SKIP DIAM] Mode skip conveyor diam: {'AKTIF' if sk_on else 'NON-AKTIF'}")
            return "HANDLED"
        if key_code in (ord('n'), ord('N')):
            if is_paused:
                step_forward = True
            return "HANDLED"
        if key_code in (ord('p'), ord('P')):
            if is_paused and _is_seekable_video():
                pending_seek["frame"] = max(0, cur_pos - 1)
            elif not is_paused:
                pipeline.toggle_profile()
            return "HANDLED"
        if key_code in (ord('h'), ord('H')):
            pipeline.toggle_threshold_source()
            return "HANDLED"
        if key_code in (ord('k'), ord('K'), ord('a'), ord('A')):
            if is_paused and _is_seekable_video():
                pending_seek["frame"] = max(0, cur_pos - 1)
            elif calib_frame is not None:
                pipeline.auto_align_rails_instant(calib_frame)
            return "HANDLED"
        if key_code in (ord('u'), ord('U'), ord('0')):
            pipeline.restore_default_profile()
            return "HANDLED"
        if key_code in (ord('r'), ord('R')):
            pipeline.reset_counters()
            return "HANDLED"
        if key_code in (ord('g'), ord('G')):
            st = pipeline.toggle_alignment_guides()
            print(f"[*] 📐 GARIS REFERENSI REL & SUMBU KONVEYOR: {'AKTIF (RETICLE ACUAN EMAS)' if st else 'NON-AKTIF'}")
            return "HANDLED"
        if key_code in (ord('l'), ord('L')):
            print("\n[*] 🚀 Membuka Wizard Kalibrasi Interaktif Rel Ganda...")
            calib_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibrate_dual_rails.py")
            subprocess.Popen([sys.executable, calib_script, "--source", str(current_source)])
            return "HANDLED"
        if _is_seekable_video():
            tgt_f, lbl = compute_seek_target_from_key(key_code, cur_pos, total_frames, grabber.fps_native)
            if tgt_f is not None:
                pending_seek["frame"] = tgt_f
                t_sec = float(tgt_f) / max(1e-5, grabber.fps_native)
                print(f"[*] [SEEK {lbl}] -> Frame {tgt_f}/{total_frames} ({format_seconds_mmss(t_sec)})")
                return "HANDLED"
            if key_code in (ord('z'), ord('Z')):
                v_fps = max(1e-5, grabber.fps_native)
                tot_s = float(total_frames) / v_fps
                cur_s = float(cur_pos) / v_fps
                entered = prompt_jump_time_dialog(format_seconds_mmss(cur_s), format_seconds_mmss(tot_s))
                cv2.waitKey(1)
                tgt_sec = parse_time_to_seconds(entered) if entered else None
                if tgt_sec is not None:
                    tgt_j = max(0, min(total_frames - 1, int(round(tgt_sec * v_fps))))
                    pending_seek["frame"] = tgt_j
                    print(f"[*] [JUMP-TIME] Melompat ke {format_seconds_mmss(tgt_sec)} (Frame {tgt_j}/{total_frames})")
                return "HANDLED"
        if key_code in (ord('w'), ord('W'), ord('o'), ord('O'), ord('f'), ord('F')):
            print("\n[*] Membuka File Dialog Explorer...")
            new_file = select_file_dialog(
                initial_dir=os.path.dirname(str(current_source)) if os.path.exists(str(current_source)) else os.getcwd(),
                title="Pilih File Video / Foto Baru untuk Inspeksi Terpadu"
            )
            cv2.waitKey(1)  # Refresh OpenCV window focus setelah Tkinter
            if new_file and os.path.exists(new_file):
                _load_media_file(new_file)
                if new_file not in sorted_videos:
                    sorted_videos.append(new_file)
                current_vid_idx = sorted_videos.index(new_file)
                print(f"[OK] Berhasil memuat media: {fname}")
            return "HANDLED"
        if key_code in (ord('v'), ord('V')) and sorted_videos:
            current_vid_idx = (current_vid_idx + 1) % len(sorted_videos)
            _load_media_file(sorted_videos[current_vid_idx])
            print(f"\n[OK] Beralih ke media: {fname}")
            return "HANDLED"
        return "NONE"

    print(f"\n[OK] Memulai inspeksi pada media: {fname} (Total Frame: {total_frames})")
    print(
        "Kontrol: [SPACE]: Pause | [,/.]: -/+10 frame | [;/']: -/+60 frame | "
        "[[/]]: -/+10s | [-/=]: -/+60s | [X]: Speed | [I]: Skip-Diam | [Z]: Loncat | [Q]: Keluar\n"
    )

    while True:
        t_loop_start = time.time()
        frame = None
        force_render_seek = False

        if is_image:
            frame = static_frame.copy()
            cur_pos = 1
        elif pending_seek["frame"] is not None and _is_seekable_video():
            req_seek = pending_seek["frame"]
            pending_seek["frame"] = None
            pending_seek["from_trackbar"] = False
            pipeline.reset_active_tracking()
            idle_skipper.reset_scan_state()
            ret, frame = grabber.read_after_seek(req_seek)
            if ret and frame is not None:
                cur_pos = grabber.get_pos_frames()
                last_video_pos = cur_pos
                cached_frame = frame.copy()
                force_render_seek = True
        elif is_paused:
            if step_forward:
                ret, frame = grabber.read()
                if ret and frame is not None:
                    cached_frame = frame.copy()
                step_forward = False
            else:
                if cached_frame is None and grabber is not None:
                    ret, fr = grabber.read()
                    if ret and fr is not None:
                        cached_frame = fr.copy()
                frame = cached_frame.copy() if cached_frame is not None else None
            cur_pos = grabber.get_pos_frames() if grabber else 1
        else:
            ret, frame = grabber.read()
            if not ret or frame is None:
                if _handle_key(cv2.waitKey(5) & 0xFF) == "BREAK":
                    break
                continue
            cur_pos = grabber.get_pos_frames()
            if cur_pos < last_video_pos - 15:
                pipeline.logged_slat_id = None
                pipeline.logged_rib_id = None
                pipeline.reset_active_tracking()
                idle_skipper.reset_scan_state()
                if pipeline.db_bridge is not None and hasattr(pipeline.db_bridge, "reset_entity_cache"):
                    pipeline.db_bridge.reset_entity_cache()
            last_video_pos = cur_pos
            cached_frame = frame.copy()

        if frame is None:
            if _handle_key(cv2.waitKey(20) & 0xFF) == "BREAK":
                break
            continue

        # Smart Auto-Skip Conveyor Diam ([I] / --skip-idle)
        if idle_skipper.enabled and _is_seekable_video() and not is_paused and not force_render_seek:
            skip_action, skip_info = idle_skipper.evaluate(cur_pos, frame, grabber.fps_native, speed_mult)
            if skip_action == "SKIP_FORWARD":
                next_pos = cur_pos + skip_info["skip_frames"]
                if total_frames > 0 and next_pos >= total_frames:
                    pipeline.reset_counters()
                    idle_skipper.reset_scan_state()
                    pending_seek["frame"] = 0
                else:
                    grabber.set_pos_frames(next_pos)
                    cur_pos = next_pos
                    last_video_pos = cur_pos
                    stub_res = dict(skip_info["stub_result"])
                    stub_res["filename"] = fname
                    v_fps = max(1e-5, grabber.fps_native)
                    t_str = f"{format_seconds_mmss(cur_pos / v_fps)}/{format_seconds_mmss(total_frames / v_fps)}"
                    s_lbl = format_speed_label(speed_mult, idle_skipper.enabled, idle_skipper.is_scanning_idle, sync_realtime)
                    dash_skip = pipeline.render_combined_dashboard(
                        frame,
                        stub_res,
                        fps=fps_val,
                        is_paused=False,
                        frame_idx=cur_pos,
                        total_frames=total_frames,
                        sync_realtime=sync_realtime,
                        video_time_str=t_str,
                        speed_label=s_lbl,
                    )
                    _update_trackbar_pos(cur_pos)
                    cv2.imshow(win_name, dash_skip)
                    if _handle_key(cv2.waitKey(1) & 0xFF, calib_frame=frame) == "BREAK":
                        break
                continue
            elif skip_action == "REWIND_RESUME":
                rewind_f = skip_info["rewind_frame"]
                pending_seek["frame"] = rewind_f
                print(
                    f"\n[*] [AUTO-SKIP] Gerakan conveyor terdeteksi! Kembali ke 1x di "
                    f"{format_seconds_mmss(rewind_f / max(1e-5, grabber.fps_native))} (Frame {rewind_f}).\n"
                )
                continue

        # Reset state in-flight bila speed >= 4x agar voting rib tidak mencampur frame berjarak jauh
        if speed_mult >= 4 and not is_image and not is_paused and not force_render_seek:
            pipeline.reset_active_tracking()

        # Jalankan KEDUA inspeksi secara bebarengan
        inspect_res = pipeline.inspect(frame)
        inspect_res["filename"] = fname

        # Hitung FPS real-time berdasarkan interval antar-frame aktual
        t_now = time.time()
        dt_frame = t_now - t_last_frame
        t_last_frame = t_now
        if dt_frame > 0:
            fps_val = 0.9 * fps_val + 0.1 * (1.0 / dt_frame)

        video_time_str = ""
        speed_label = ""
        if _is_seekable_video():
            v_fps = max(1e-5, grabber.fps_native)
            video_time_str = f"{format_seconds_mmss(cur_pos / v_fps)}/{format_seconds_mmss(total_frames / v_fps)}"
            speed_label = format_speed_label(speed_mult, idle_skipper.enabled, idle_skipper.is_scanning_idle, sync_realtime)

        # Render antarmuka dashboard visual
        dashboard_screen = pipeline.render_combined_dashboard(
            frame,
            inspect_res,
            fps=fps_val,
            is_paused=is_paused,
            frame_idx=cur_pos,
            total_frames=total_frames,
            sync_realtime=sync_realtime,
            video_time_str=video_time_str,
            speed_label=speed_label,
        )

        # Rekam ke database SQLite dan folder captures jika ada NG dan DB Logging aktif (Single-Capture per Objek)
        if inspect_res.get("status") == "NG" and not force_render_seek:
            pipeline.record_if_ng(
                frame_raw=frame,
                frame_ann=dashboard_screen,
                inspect_res=inspect_res,
                slat_no=getattr(pipeline.slat_inspector, "slat_count", None),
                rib_no=getattr(pipeline.rib_inspector, "rib_count", None)
            )

        _update_trackbar_pos(cur_pos)
        cv2.imshow(win_name, dashboard_screen)

        # Tangani interaksi keyboard & pacing kecepatan playback video secara presisi
        t_work = time.time() - t_loop_start
        if is_paused:
            delay = 30
        elif sync_realtime and speed_mult == 1 and grabber is not None and not grabber.is_live:
            target_dt = 1.0 / grabber.fps_native
            delay = int(max(1, (target_dt - t_work) * 1000.0))
        else:
            delay = 1

        if _handle_key(cv2.waitKey(delay) & 0xFF, calib_frame=frame) == "BREAK":
            break

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
