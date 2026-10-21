"""
video_hough_line_inspector.py
=============================
Aplikasi Pemutar & Inspeksi Video / Live Webcam Konveyor Slat Otomatis Menggunakan Hough Line Transform.

Fitur Unggulan:
1. Dukungan Live Webcam & Pemutaran Video:
   - Live streaming langsung dari Kamera Webcam USB / Industrial secara real-time.
   - Deteksi otomatis indeks kamera yang tersedia (Cam 0, Cam 1, dst).
   - Freeze Frame [SPASI] pada webcam untuk kalibrasi slider dan inspeksi detail.
   - Pemutaran video otomatis [SPASI] dengan smooth loop.
2. Deteksi Slat-by-Slat Otomatis:
   - Menghitung setiap plat yang melintasi Gerbang Inspeksi (Slat #01, #02, dst).
   - Menghitung jumlah plat OK dan NG secara real-time.
3. Virtual Inspection Gate Dinamis:
   - Slider 'Gate Y Pos (%)' dan 'Gate Height (%)' untuk memposisikan gerbang secara instan.
4. Slider Kalibrasi Hough Real-Time:
   - Toleransi Sudut, Min Line Length, Max Line Gap, Hough Threshold, Filter Non-Garis.
5. Hotkey Lengkap:
   - [SPASI]       : Live Stream / Freeze (Webcam) atau Play / Pause (Video)
   - [TAB] / [C]   : Ganti Kamera Webcam Berikutnya (Cam 0 -> Cam 1) / Beralih ke Webcam
   - [D] / [→]     : Ambil 1 Frame Baru (saat Freeze) / Maju 1 Frame (saat Pause Video)
   - [A] / [←]     : Mundur 1 Frame (saat Pause Video)
   - [E]           : Ekspor Rekap Laporan Inspeksi ke CSV (Audit QC)
   - [P]           : Simpan Konfigurasi Preset Slider ke JSON
   - [V]           : Beralih ke Video / Sumber Berikutnya
   - [R]           : Reset Counter Plat (OK/NG)
   - [S]           : Simpan Snapshot Frame Saat Ini
   - [Q] / [ESC]   : Keluar
"""

import os
import sys
import time
import json
import csv
import argparse
from collections import deque
import cv2
import numpy as np

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Konfigurasi Direktori Output & Preset Config
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
OUTPUT_DIR   = os.path.join(CURRENT_DIR, "output_hough_video")
SNAPSHOT_DIR = os.path.join(OUTPUT_DIR, "snapshots")
DEFECTS_DIR  = os.path.join(OUTPUT_DIR, "defects")
CONFIG_PRESET_FILE = os.path.join(CURRENT_DIR, "hough_slider_config.json")

for d in [OUTPUT_DIR, SNAPSHOT_DIR, DEFECTS_DIR]:
    os.makedirs(d, exist_ok=True)

# Daftar Video Default (Prioritaskan direktori Documentation/Video Testing)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_VIDEO_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "Documentation", "Video Testing"))
if not os.path.exists(TEST_VIDEO_DIR):
    TEST_VIDEO_DIR = os.path.abspath(os.path.join(BASE_DIR, "Video_Testing"))
DOWNLOADS_DIR  = os.path.join(os.path.expanduser("~"), "Downloads")

DEFAULT_VIDEOS = []
if os.path.exists(TEST_VIDEO_DIR):
    for f in sorted(os.listdir(TEST_VIDEO_DIR)):
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".wmv")):
            DEFAULT_VIDEOS.append(os.path.join(TEST_VIDEO_DIR, f))

# Fallback cadangan dari folder Downloads
for fname in ["deform_parah.mp4", "deform_sedikit.mp4", "ok_semua.mp4",
              "Conveyor_belt_moves_forward_1080p_20260921092943.mp4",
              "Industrial_conveyor_belt_moving_1080p_20260921094028.mp4"]:
    fpath = os.path.join(DOWNLOADS_DIR, fname)
    if os.path.exists(fpath) and fpath not in DEFAULT_VIDEOS:
        DEFAULT_VIDEOS.append(fpath)

try:
    import tkinter as tk
    from tkinter import filedialog
    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False


def detect_available_cameras(max_tested=4):
    """Mendeteksi daftar indeks kamera webcam yang aktif dan dapat dibuka."""
    available = []
    for i in range(max_tested):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
            cap.release()
    return available


def select_video_dialog():
    if not HAS_TKINTER:
        return None
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    fpath = filedialog.askopenfilename(
        title="Pilih Video Konveyor untuk Diinspeksi Hough Line",
        filetypes=[
            ("Video Files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
            ("All Files", "*.*")
        ]
    )
    root.destroy()
    return fpath if fpath else None


def load_slider_preset(source_identifier, orig_w, orig_h):
    """Memuat konfigurasi slider tersimpan dari file JSON."""
    if not os.path.exists(CONFIG_PRESET_FILE):
        return None
    try:
        with open(CONFIG_PRESET_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            sname = os.path.basename(str(source_identifier))
            if sname in data:
                return data[sname]

            # Pencocokan fleksibel (case-insensitive & variasi spasi / underscore)
            norm_target = sname.lower().replace(" ", "_")
            for k, v in data.items():
                if k.lower().replace(" ", "_") == norm_target:
                    return v

            if str(source_identifier).lower().startswith("webcam") or "cam" in str(source_identifier).lower():
                if "WEBCAM" in data:
                    return data["WEBCAM"]
                if "webcam" in data:
                    return data["webcam"]

            aspect_key = "portrait" if orig_h > orig_w else "landscape"
            if aspect_key in data:
                return data[aspect_key]
    except Exception:
        pass
    return None


def save_slider_preset(source_identifier, params):
    """Menyimpan konfigurasi slider saat ini ke file JSON."""
    data = {}
    if os.path.exists(CONFIG_PRESET_FILE):
        try:
            with open(CONFIG_PRESET_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    sname = os.path.basename(str(source_identifier))
    data[sname] = params
    if str(source_identifier).lower().startswith("webcam") or "cam" in str(source_identifier).lower():
        data["WEBCAM"] = params
    try:
        with open(CONFIG_PRESET_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"  💾 [PRESET] Konfigurasi slider untuk {sname} berhasil disimpan ke {os.path.basename(CONFIG_PRESET_FILE)}.")
    except Exception as e:
        print(f"  [WARN] Gagal menyimpan preset slider: {e}")


class VideoHoughInspector:
    def __init__(self, source=0):
        self.source = source
        self.cam_index = 0
        self.is_webcam = False

        # Identifikasi tipe sumber: Webcam vs File Video
        if isinstance(source, int):
            self.is_webcam = True
            self.cam_index = source
            self.source_name = f"Webcam_{self.cam_index}"
            self.video_path = f"Webcam_{self.cam_index}"
        elif isinstance(source, str):
            if source.isdigit():
                self.is_webcam = True
                self.cam_index = int(source)
                self.source_name = f"Webcam_{self.cam_index}"
                self.video_path = f"Webcam_{self.cam_index}"
            elif source.lower().startswith("cam") and len(source) > 3 and source[3:].isdigit():
                self.is_webcam = True
                self.cam_index = int(source[3:])
                self.source_name = f"Webcam_{self.cam_index}"
                self.video_path = f"Webcam_{self.cam_index}"
            elif source.lower() == "webcam":
                self.is_webcam = True
                self.cam_index = 0
                self.source_name = "Webcam_0"
                self.video_path = "Webcam_0"
            else:
                self.is_webcam = False
                self.source_name = os.path.basename(source)
                self.video_path = source

        if self.is_webcam:
            backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(self.cam_index, backend)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.cam_index)
            if not self.cap.isOpened():
                raise RuntimeError(f"Gagal membuka kamera webcam (Indeks: {self.cam_index})")
            
            # Coba set resolusi kamera ke 1280x720 untuk ketajaman optimal
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            ret, frame = self.cap.read()
            if not ret or frame is None:
                raise RuntimeError(f"Gagal membaca frame dari webcam (Indeks: {self.cam_index})")
            
            self.orig_h, self.orig_w = frame.shape[:2]
            self.current_frame = frame
            self.fps = 30.0
            self.total_frames = 0
            self.is_playing = True  # Default langsung streaming live
            self.fps_tracker = deque(maxlen=20)
            self.last_frame_time = time.time()
            self.live_fps = 30.0
            self.start_time = time.time()
        else:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                raise FileNotFoundError(f"Gagal membuka video: {self.video_path}")

            self.orig_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.orig_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 24.0
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.is_playing = False
            self.fps_tracker = None
            self.live_fps = self.fps
            self.start_time = time.time()

        # Penskalaan Tampilan Adaptif agar muat di monitor
        if self.orig_h > self.orig_w:
            # Video Vertikal / Portrait (misal 1080x1920)
            target_h = 750
            target_w = int(self.orig_w * (target_h / self.orig_h))
            # Pastikan lebar minimal 540px agar nama slider Windows tidak terpotong "..."
            self.disp_w = max(540, target_w)
            self.disp_h = target_h
            self.display_scale = self.disp_w / self.orig_w
        else:
            # Video Horizontal / Landscape (misal 1920x1080 atau 1280x720)
            max_disp_w = 1100
            max_disp_h = 680
            scale_w = max_disp_w / self.orig_w
            scale_h = max_disp_h / self.orig_h
            self.display_scale = min(1.0, scale_w, scale_h)
            self.disp_w = int(self.orig_w * self.display_scale)
            self.disp_h = int(self.orig_h * self.display_scale)

        # State Player & Pemutaran Otomatis
        self.current_frame_idx = 0
        self.internal_slider_update = False

        # State Deteksi Slat-by-Slat
        self.slat_counter = 0
        self.ok_count = 0
        self.ng_count = 0
        self.last_trigger_frame = -999
        self.inspect_interval = 16  # Minimal jarak antar plat (frame)
        self.slat_latched = False   # State machine latch: mencegah pemicuan ganda/tiga kali pada plat yang sama
        self.last_defect_capture_time = 0
        self.inspection_records = []

        # State Anti-Flicker & Temporal Debouncing
        self.ng_consecutive = 0
        self.hold_ng_timer = 0
        self.is_ng_stable = False
        self.stable_def_angle = 0.0
        self.stable_def_count = 0
        self.cached_lines = None      # Buffer persistensi garis anti-flicker
        self.line_fade_frames = 0

        # Inisialisasi Default Gate Terkalibrasi
        if self.orig_h > self.orig_w:
            self.default_gate_y = 61
            self.default_gate_h = 10
        else:
            self.default_gate_y = 56
            self.default_gate_h = 8

        # Windows
        if self.is_webcam:
            self.win_main = f"INSPEKSI LIVE WEBCAM HOUGH LINE (CAM #{self.cam_index})"
        else:
            self.win_main = "INSPEKSI VIDEO HOUGH LINE (CONVEYOR)"
        self.win_canny = "TEPI TERISOLASI (GATE CANNY)"

        cv2.namedWindow(self.win_main, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.win_canny, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win_main, self.disp_w, self.disp_h)
        cv2.resizeWindow(self.win_canny, 420, 200)

        self.setup_trackbars()

    def setup_trackbars(self):
        source_key = "WEBCAM" if self.is_webcam else self.source_name
        preset = load_slider_preset(source_key, self.orig_w, self.orig_h) or {}
        ang_th   = preset.get("angle_tolerance", 30)
        min_len  = preset.get("min_line_len", 85)
        max_gap  = preset.get("max_line_gap", 35)
        hough_th = preset.get("hough_threshold", 35)
        filter_k = preset.get("filter_non_garis", 0)
        gate_y   = preset.get("gate_y_pct", self.default_gate_y)
        gate_h   = preset.get("gate_h_pct", self.default_gate_h)

        # 1. Slider Frame Scrubber (Hanya untuk file video yang memiliki total frame terhingga)
        if not self.is_webcam and self.total_frames > 1:
            cv2.createTrackbar("Frame", self.win_main, 0, max(1, self.total_frames - 1), self.on_frame_slider)
        
        # 2. Parameter Deteksi Hough (Terkalibrasi Bebas Flickering)
        cv2.createTrackbar("Toleransi Sudut (x0.1 deg)", self.win_main, ang_th, 150, lambda x: None)
        cv2.createTrackbar("Min Line Length", self.win_main, min_len, 250, lambda x: None)
        cv2.createTrackbar("Max Line Gap", self.win_main, max_gap, 100, lambda x: None)
        cv2.createTrackbar("Hough Threshold", self.win_main, hough_th, 150, lambda x: None)
        
        # 3. Parameter Pembersih Non-Garis & Posisi Gerbang
        cv2.createTrackbar("Filter Non-Garis", self.win_main, filter_k, 60, lambda x: None)
        cv2.createTrackbar("Gate Y Pos (%)", self.win_main, gate_y, 90, lambda x: None)
        cv2.createTrackbar("Gate Height (%)", self.win_main, gate_h, 30, lambda x: None)

    def _get_tb(self, name, fallback=0):
        """Aman membaca nilai trackbar; kembalikan fallback jika window belum tersedia."""
        try:
            val = cv2.getTrackbarPos(name, self.win_main)
            return val if val >= 0 else fallback
        except Exception:
            return fallback

    def save_current_preset(self):
        """Menyimpan pengaturan slider saat ini ke JSON preset."""
        params = {
            "angle_tolerance": cv2.getTrackbarPos("Toleransi Sudut (x0.1 deg)", self.win_main),
            "min_line_len": cv2.getTrackbarPos("Min Line Length", self.win_main),
            "max_line_gap": cv2.getTrackbarPos("Max Line Gap", self.win_main),
            "hough_threshold": cv2.getTrackbarPos("Hough Threshold", self.win_main),
            "filter_non_garis": cv2.getTrackbarPos("Filter Non-Garis", self.win_main),
            "gate_y_pct": cv2.getTrackbarPos("Gate Y Pos (%)", self.win_main),
            "gate_h_pct": cv2.getTrackbarPos("Gate Height (%)", self.win_main)
        }
        if self.is_webcam:
            save_slider_preset("WEBCAM", params)
            save_slider_preset(self.source_name, params)
        else:
            save_slider_preset(self.video_path, params)

    def on_frame_slider(self, val):
        if self.is_webcam:
            return
        # Abaikan jika slider diperbarui secara internal oleh pemutar otomatis
        if self.internal_slider_update:
            return
        # Jika pengguna menggeser slider dengan mouse:
        if abs(val - self.current_frame_idx) > 1:
            self.is_playing = False
            self.seek_to(val)

    def seek_to(self, idx):
        """Seek acak ke frame tertentu (hanya pada file video)."""
        if self.is_webcam:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.current_frame = frame
            return ret, self.current_frame

        idx = max(0, min(self.total_frames - 1, idx))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if ret:
            self.current_frame_idx = idx
            self.current_frame = frame
            self.last_trigger_frame = -999
            self.slat_latched = False
            self.ng_consecutive = 0
            self.hold_ng_timer = 0
            self.is_ng_stable = False
            self.cached_lines = None
            self.line_fade_frames = 0
        return ret, frame

    def read_next_frame(self):
        """Membaca frame berikutnya (Streaming webcam real-time atau frame video)."""
        if self.is_webcam:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.current_frame_idx += 1
                self.current_frame = frame
                now = time.time()
                dt = now - self.last_frame_time
                self.last_frame_time = now
                if dt > 0:
                    self.fps_tracker.append(1.0 / dt)
                    self.live_fps = sum(self.fps_tracker) / len(self.fps_tracker)
            return ret, frame

        ret, frame = self.cap.read()
        if not ret:
            # Video selesai -> loop kembali secara otomatis ke frame 0
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame_idx = 0
            self.last_trigger_frame = -999
            self.slat_latched = False
            self.ng_consecutive = 0
            self.hold_ng_timer = 0
            self.is_ng_stable = False
            self.cached_lines = None
            self.line_fade_frames = 0
            ret, frame = self.cap.read()
        else:
            self.current_frame_idx += 1

        if ret:
            self.current_frame = frame
        return ret, frame

    def get_gate_polygon(self, gate_y_pct, gate_h_pct):
        y_center = int(self.orig_h * (gate_y_pct / 100.0))
        half_h   = int(self.orig_h * ((gate_h_pct / 2.0) / 100.0))
        y1 = max(0, y_center - half_h)
        y2 = min(self.orig_h - 1, y_center + half_h)

        if self.orig_h > self.orig_w:
            # Video Vertikal (1080x1920)
            x1 = int(self.orig_w * 0.03)
            x2 = int(self.orig_w * 0.97)
        else:
            # Video Horizontal (1920x1080 / 1280x720) - Pangkas chain wings sisi samping
            x1 = int(self.orig_w * 0.1875)
            x2 = int(self.orig_w * 0.8125)

        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32), (x1, y1, x2, y2)

    def check_new_slat_arrival(self, cleaned_edges, gx1, gy1, gx2, gy2):
        """
        Mengecek apakah ada plat baru yang mencapai garis tengah kanonikal gerbang.
        Menggunakan Deteksi Virtual Center Beam dengan Debounce Latch.
        """
        gate_crop = cleaned_edges[gy1:gy2, gx1:gx2]
        if gate_crop.size == 0:
            return False, 0.0

        rsum = np.sum(gate_crop > 0, axis=1)
        best_r = int(np.argmax(rsum))
        best_v = int(rsum[best_r])
        gate_h_px = gy2 - gy1

        # Garis tengah kanonikal gerbang (Virtual Center Beam)
        r_mid = gate_h_px // 2
        # Jendela pemicu sempit di sekitar garis tengah (+/- 15% dari tinggi gerbang, minimal 10px)
        half_window = max(10, int(0.15 * gate_h_px))
        in_center_window = (r_mid - half_window <= best_r <= r_mid + half_window)

        # Antisipasi video looping / frame scrubbed backward
        if self.current_frame_idx < self.last_trigger_frame:
            frames_since_last = 999
            self.slat_latched = False
        else:
            frames_since_last = self.current_frame_idx - self.last_trigger_frame

        # Jarak minimum adaptif berdasarkan tinggi gerbang (minimal 16 frame)
        effective_interval = max(self.inspect_interval, int(gate_h_px / 16))

        # Buka kunci (unlatch) jika garis sudah bergerak melewati jendela tengah ke arah bawah,
        # atau jika interval frame antar-plat sudah terpenuhi
        if self.slat_latched and (best_r > r_mid + half_window or frames_since_last >= effective_interval):
            self.slat_latched = False

        if not self.slat_latched and in_center_window and best_v >= 120 and frames_since_last >= effective_interval:
            self.last_trigger_frame = self.current_frame_idx
            self.slat_latched = True
            return True, best_v

        return False, 0.0

    def save_csv_log(self):
        """Menyimpan seluruh riwayat inspeksi slat ke file CSV (Audit Trail QC)."""
        if not self.inspection_records:
            return
        
        if self.is_webcam:
            vname = f"webcam_cam{self.cam_index}_{int(time.time())}"
        else:
            vname = os.path.splitext(os.path.basename(self.video_path))[0]

        csv_path = os.path.join(OUTPUT_DIR, f"laporan_hough_{vname}.csv")
        csv_latest = os.path.join(OUTPUT_DIR, "laporan_inspeksi_hough.csv")
        
        for target_path in [csv_path, csv_latest]:
            try:
                keys = self.inspection_records[0].keys()
                with open(target_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    writer.writerows(self.inspection_records)
            except Exception as e:
                print(f"  [ERROR] Gagal menulis CSV ke {target_path}: {e}")

        print(f"\n📊 [AUDIT TRAIL QC] Berhasil mengekspor {len(self.inspection_records)} plat ke:\n   📁 {csv_path}")

    def process_frame(self, frame):
        """Memproses deteksi garis Hough Line pada zona Virtual Gate."""
        source_key = "WEBCAM" if self.is_webcam else self.source_name
        preset = load_slider_preset(source_key, self.orig_w, self.orig_h) or {}
        angle_th_val = self._get_tb("Toleransi Sudut (x0.1 deg)", preset.get("angle_tolerance", 35))
        min_len      = self._get_tb("Min Line Length",             preset.get("min_line_len", 85))
        max_gap      = self._get_tb("Max Line Gap",                preset.get("max_line_gap", 35))
        hough_th     = self._get_tb("Hough Threshold",             preset.get("hough_threshold", 35))
        filter_k     = self._get_tb("Filter Non-Garis",            preset.get("filter_non_garis", 0))
        gate_y_pct   = self._get_tb("Gate Y Pos (%)",              preset.get("gate_y_pct", self.default_gate_y))
        gate_h_pct   = self._get_tb("Gate Height (%)",             preset.get("gate_h_pct", self.default_gate_h))

        angle_th = max(5, angle_th_val) / 10.0
        min_len  = max(20, min_len)
        max_gap  = max(5, max_gap)
        hough_th = max(10, hough_th)
        gate_h_pct = max(4, gate_h_pct)

        # 1. Masking Virtual Gate
        poly_pts, (gx1, gy1, gx2, gy2) = self.get_gate_polygon(gate_y_pct, gate_h_pct)
        mask = np.zeros((self.orig_h, self.orig_w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly_pts], 255)

        # 2. Canny Edge pada area Gate dengan Anti-Karat Pipeline
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

        # Filter Dominansi Gradien Sobel-Y
        sy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        sx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        abs_sy = np.abs(sy)
        abs_sx = np.abs(sx)
        h_mask = ((abs_sy >= 1.15 * abs_sx) * 255).astype(np.uint8)

        # Canny threshold dikombinasikan dengan masker gerbang dan orientasi horizontal
        edges = cv2.Canny(blurred, 55, 140, apertureSize=3)
        edges_directional = cv2.bitwise_and(edges, h_mask)
        masked_edges = cv2.bitwise_and(edges_directional, mask)

        # 3. Morphological Horizontal Bridge
        bridge_w = 17 if self.orig_h > self.orig_w else 23
        k_bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (bridge_w, 1))
        bridged_edges = cv2.morphologyEx(masked_edges, cv2.MORPH_CLOSE, k_bridge)

        # 4. Filter Pembersih Non-Garis
        if filter_k > 1:
            eff_k = min(filter_k, 5)
            k_elem = cv2.getStructuringElement(cv2.MORPH_RECT, (eff_k, 1))
            cleaned_edges = cv2.morphologyEx(bridged_edges, cv2.MORPH_OPEN, k_elem)
        else:
            cleaned_edges = bridged_edges

        # 5. Hough Line Transform
        lines = cv2.HoughLinesP(cleaned_edges, rho=1, theta=np.pi/180,
                                threshold=hough_th, minLineLength=min_len, maxLineGap=max_gap)

        # Anti-Flicker Line Persistence
        if lines is not None and len(lines) > 0:
            self.cached_lines = lines
            self.line_fade_frames = 0
        elif self.cached_lines is not None and self.line_fade_frames < 2:
            lines = self.cached_lines
            self.line_fade_frames += 1
        else:
            self.cached_lines = None
            self.line_fade_frames = 0

        output = frame.copy()
        normal_count = 0
        deform_count = 0
        total_def_len = 0.0
        max_def_angle = 0.0

        # Dual-Threshold Hysteresis Band
        hysteresis_margin = max(0.3, angle_th * 0.15)
        th_high = angle_th
        th_low  = max(0.5, angle_th - hysteresis_margin)
        active_th = th_low if self.is_ng_stable else th_high

        if lines is not None and len(lines) > 0:
            lines_arr = np.asarray(lines).reshape(-1, 4)
            for x1, y1, x2, y2 in lines_arr:
                dx = x2 - x1
                dy = y2 - y1
                length = np.sqrt(dx * dx + dy * dy)
                angle = abs(np.degrees(np.arctan2(dy, dx)))
                if angle > 90:
                    angle = 180 - angle

                if angle > active_th:
                    deform_count += 1
                    total_def_len += length
                    max_def_angle = max(max_def_angle, angle)
                    line_thick = 5 if self.orig_h > self.orig_w else 3
                    font_sc = 0.85 if self.orig_h > self.orig_w else 0.45
                    cv2.line(output, (x1, y1), (x2, y2), (0, 0, 255), line_thick, cv2.LINE_AA)
                    cv2.putText(output, f"DEF {angle:.1f}d", (x1, max(y1 - 10, 35)),
                                cv2.FONT_HERSHEY_SIMPLEX, font_sc, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    normal_count += 1
                    line_thick = 3 if self.orig_h > self.orig_w else 2
                    cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), line_thick, cv2.LINE_AA)

        is_solid_defect = (
            (deform_count >= 2) or
            (deform_count == 1 and max_def_angle >= th_high and total_def_len >= min_len)
        )
        raw_ng = (deform_count >= 1 and max_def_angle >= active_th and is_solid_defect)

        # 6. Deteksi Plat Baru & Log Slat-by-Slat Presisi
        is_new = False
        if self.is_playing:
            is_new, seam_val = self.check_new_slat_arrival(cleaned_edges, gx1, gy1, gx2, gy2)
            if is_new:
                self.slat_counter += 1
                slat_is_ng = raw_ng
                
                if slat_is_ng:
                    self.ng_count += 1
                    print(f"  ⚠️ [NG DETECTED] Slat #{self.slat_counter:02d} pada frame {self.current_frame_idx:04d} (Sudut Maks: {max_def_angle:.1f}°, {deform_count} garis)")
                    self.is_ng_stable = True
                    self.hold_ng_timer = max(14, int(self.inspect_interval * 1.1))
                    self.stable_def_angle = max_def_angle
                    self.stable_def_count = deform_count
                    self.ng_consecutive = 2
                else:
                    self.ok_count += 1
                    self.is_ng_stable = False
                    self.hold_ng_timer = 0
                    self.ng_consecutive = 0
                    self.stable_def_angle = 0.0
                    self.stable_def_count = 0

                record = {
                    "Slat_ID": f"SLAT #{self.slat_counter:02d}",
                    "Frame": self.current_frame_idx,
                    "Timestamp_Sec": round(time.time() - self.start_time, 2) if self.is_webcam else round(self.current_frame_idx / max(1.0, self.fps), 2),
                    "Max_Angle_Deg": round(max_def_angle, 2),
                    "Deform_Lines_Count": deform_count,
                    "Status": "NG" if slat_is_ng else "OK"
                }
                self.inspection_records.append(record)

        # Anti-Flicker Temporal Hysteresis Filter
        if not is_new:
            if self.is_playing:
                if raw_ng:
                    self.ng_consecutive += 1
                    if self.ng_consecutive >= 2:
                        self.is_ng_stable = True
                        self.hold_ng_timer = max(12, self.hold_ng_timer)
                        self.stable_def_angle = max(self.stable_def_angle, max_def_angle)
                        self.stable_def_count = max(self.stable_def_count, deform_count)
                else:
                    self.ng_consecutive = 0
                    if self.hold_ng_timer > 0:
                        self.hold_ng_timer -= 1
                    else:
                        self.is_ng_stable = False
                        self.stable_def_angle = 0.0
                        self.stable_def_count = 0
            else:
                self.is_ng_stable = raw_ng
                self.hold_ng_timer = 0
                self.stable_def_angle = max_def_angle if raw_ng else 0.0
                self.stable_def_count = deform_count if raw_ng else 0

        is_deformed = self.is_ng_stable
        display_def_cnt = self.stable_def_count if is_deformed else deform_count
        display_def_ang = self.stable_def_angle if is_deformed else max_def_angle

        # Visualisasi Virtual Gate
        gate_col = (0, 0, 255) if is_deformed else (0, 230, 255)
        gate_thick = 4 if self.orig_h > self.orig_w else 2
        cv2.polylines(output, [poly_pts], isClosed=True, color=gate_col, thickness=gate_thick, lineType=cv2.LINE_AA)
        gate_font_sc = 0.9 if self.orig_h > self.orig_w else 0.55
        cv2.putText(output, "VIRTUAL INSPECTION GATE", (gx1 + 10, gy1 - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, gate_font_sc, gate_col, 2, cv2.LINE_AA)

        return output, normal_count, display_def_cnt, angle_th, is_deformed, display_def_ang, cleaned_edges, gx1, gy1, gx2, gy2

    def draw_hud(self, disp, norm_cnt, def_cnt, angle_th, is_deformed, max_def_angle):
        """Menggambar HUD langsung pada resolusi layar (display canvas)."""
        h, w = disp.shape[:2]
        hud_h = 138
        
        # 1. Kotak HUD Utama
        overlay = disp.copy()
        cv2.rectangle(overlay, (8, 8), (w - 8, hud_h), (12, 12, 16), -1)
        
        status_border = (0, 0, 255) if is_deformed else (0, 220, 0)
        border_thick = 3 if is_deformed else 2
        cv2.rectangle(overlay, (8, 8), (w - 8, hud_h), status_border, border_thick)
        cv2.addWeighted(overlay, 0.88, disp, 0.12, 0, disp)

        # 2. STATUS BADGE UTAMA
        if is_deformed:
            badge_text = f"[!] STATUS: NG - DEFORMASI ({def_cnt} Garis)"
            badge_color = (0, 0, 255)
            sub_status = f"Sudut Maks: {max_def_angle:.1f} deg  |  Toleransi: {angle_th:.1f} deg"
            sub_color = (80, 130, 255)
        else:
            badge_text = "[OK] STATUS: PLAT NORMAL"
            badge_color = (0, 255, 100)
            sub_status = f"Semua garis rata & sejajar (< {angle_th:.1f} deg)"
            sub_color = (180, 230, 180)

        # Baris 1: Status Utama
        cv2.putText(disp, badge_text, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.68, badge_color, 2, cv2.LINE_AA)
        # Baris 2: Detail Sudut
        cv2.putText(disp, sub_status, (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.44, sub_color, 1, cv2.LINE_AA)

        # Garis Pembatas
        cv2.line(disp, (20, 68), (w - 20, 68), (60, 60, 70), 1)

        # Baris 3: Rekapitulasi Slat
        pct_ng = (self.ng_count / self.slat_counter * 100) if self.slat_counter > 0 else 0.0
        cv2.putText(disp, f"TOTAL: {self.slat_counter}", (20, 93), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(disp, f"OK: {self.ok_count}", (135, 93), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 100), 2, cv2.LINE_AA)
        ng_col = (0, 0, 255) if self.ng_count > 0 else (180, 180, 180)
        cv2.putText(disp, f"NG: {self.ng_count} ({pct_ng:.1f}%)", (225, 93), cv2.FONT_HERSHEY_SIMPLEX, 0.56, ng_col, 2, cv2.LINE_AA)

        # Baris 4: Info Playback / Live Stream
        if self.is_webcam:
            play_str = "LIVE STREAMING" if self.is_playing else "FREEZE / PAUSED"
            play_col = (0, 255, 0) if self.is_playing else (0, 200, 255)
            info_line = f"[{play_str}] CAM #{self.cam_index} | FPS: {self.live_fps:.1f} | Frame #{self.current_frame_idx} | [SPASI] Freeze | [TAB] Cam | [E] CSV"
        else:
            time_sec = self.current_frame_idx / max(1.0, self.fps)
            play_str = "PLAY (OTOMATIS)" if self.is_playing else "PAUSED [Tekan SPASI]"
            play_col = (0, 255, 0) if self.is_playing else (0, 200, 255)
            info_line = f"[{play_str}] Frame {self.current_frame_idx}/{self.total_frames} ({time_sec:.1f}s) | [SPASI] Play | [E] CSV"

        cv2.putText(disp, info_line, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.42, play_col, 1, cv2.LINE_AA)

        return disp

    def run(self):
        if self.is_webcam:
            print(f"\n🎥 Membuka Live Webcam: CAM #{self.cam_index}")
            print(f"   Dimensi: {self.orig_w}x{self.orig_h} | Status: LIVE STREAMING")
            print("\n=== KONTROL KEYBOARD LIVE WEBCAM ===")
            print("  [SPASI]       : Live Stream / Freeze Frame (Tahan frame untuk kalibrasi slider)")
            print("  [D / Panah ->]: Ambil 1 Frame Baru (Step Frame saat Freeze Frame)")
            print("  [TAB] / [C]   : Ganti Kamera Webcam Berikutnya (Cam 0 -> Cam 1 -> dst)")
            print("  [V]           : Beralih ke Video File Testing")
            print("  [E]           : Ekspor Hasil Inspeksi Slat ke File CSV (Audit QC)")
            print("  [P]           : Simpan Preset Konfigurasi Slider ke JSON")
            print("  [R]           : Reset Counter Plat (OK/NG)")
            print("  [S]           : Simpan Snapshot Frame Saat Ini")
            print("  [Q / ESC]     : Keluar Program")
            print("====================================\n")
        else:
            print(f"\n🎥 Membuka video: {os.path.basename(self.video_path)}")
            print(f"   Dimensi: {self.orig_w}x{self.orig_h} | FPS: {self.fps:.2f} | Total: {self.total_frames} frame")
            print("\n=== KONTROL KEYBOARD VIDEO ===")
            print("  [SPASI]       : Play / Pause Otomatis (Video akan berjalan terus)")
            print("  [D / Panah ->]: Maju 1 Frame (Step Forward saat pause)")
            print("  [A / Panah <-]: Mundur 1 Frame (Step Backward saat pause)")
            print("  [TAB] / [C]   : Beralih ke Kamera Webcam Live")
            print("  [V]           : Beralih ke Video Berikutnya")
            print("  [E]           : Ekspor Hasil Inspeksi Slat ke File CSV (Audit QC)")
            print("  [P]           : Simpan Preset Konfigurasi Slider ke JSON")
            print("  [R]           : Reset Counter Plat (OK/NG)")
            print("  [S]           : Simpan Snapshot Frame Saat Ini")
            print("  [Q / ESC]     : Keluar Program")
            print("==============================\n")

        if not self.is_webcam:
            self.seek_to(0)
        frame_period = 1.0 / max(1.0, self.fps)

        while True:
            t0 = time.time()

            # Mode Play / Live Stream: Membaca frame secara kontinu
            if self.is_playing:
                ret, frame = self.read_next_frame()
                if not ret or frame is None:
                    if self.is_webcam:
                        print("  [WARN] Gagal membaca frame dari webcam. Mencoba kembali...")
                        time.sleep(0.05)
                        continue
                    else:
                        break
                if not self.is_webcam:
                    # Update slider posisi frame setiap 5 frame (atau akhir) agar Win32 trackbar tidak flickering
                    if self.current_frame_idx % 5 == 0 or self.current_frame_idx >= self.total_frames - 1:
                        self.internal_slider_update = True
                        cv2.setTrackbarPos("Frame", self.win_main, self.current_frame_idx)
                        self.internal_slider_update = False

            # Tampilkan frame beranotasi
            if self.current_frame is not None:
                annotated, norm_cnt, def_cnt, ang_th, is_def, max_ang, cleaned_edges, gx1, gy1, gx2, gy2 = self.process_frame(self.current_frame)
                disp = cv2.resize(annotated, (self.disp_w, self.disp_h))

                # 7. Render PiP Mini-Canny Gate langsung pada canvas display
                gate_crop = cleaned_edges[gy1:gy2, gx1:gx2]
                if gate_crop.size > 0:
                    pip_w = 160 if self.orig_h > self.orig_w else 210
                    pip_h = 65 if self.orig_h > self.orig_w else 75
                    pip_crop = cv2.resize(gate_crop, (pip_w, pip_h))
                    pip_bgr = cv2.cvtColor(pip_crop, cv2.COLOR_GRAY2BGR)
                    cv2.rectangle(pip_bgr, (0, 0), (pip_w - 1, pip_h - 1), (0, 230, 255), 2)
                    cv2.putText(pip_bgr, "CANNY RADAR", (6, 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 230, 255), 1, cv2.LINE_AA)
                    
                    if self.orig_h > self.orig_w:
                        # Video Vertikal: PiP di pojok kanan bawah
                        py2 = self.disp_h - 12
                        py1 = py2 - pip_h
                        px2 = self.disp_w - 12
                        px1 = px2 - pip_w
                    else:
                        # Video Horizontal: PiP di pojok kanan atas
                        py1 = 12
                        py2 = py1 + pip_h
                        px2 = self.disp_w - 12
                        px1 = px2 - pip_w
                        
                    if px1 >= 0 and py2 < self.disp_h and py1 >= 0:
                        disp[py1:py2, px1:px2] = pip_bgr

                    # Tetap sinkronkan jendela pendamping jika dibuka pengguna
                    if cv2.getWindowProperty(self.win_canny, cv2.WND_PROP_VISIBLE) >= 1:
                        disp_crop = cv2.resize(gate_crop, (420, 180))
                        cv2.imshow(self.win_canny, disp_crop)

                # 8. Render HUD Super Jelas & Tajam Langsung pada Canvas Display
                disp = self.draw_hud(disp, norm_cnt, def_cnt, ang_th, is_def, max_ang)
                cv2.imshow(self.win_main, disp)

            # Hitung delay agar FPS presisi
            if self.is_webcam:
                delay_ms = 1 if self.is_playing else 30
            else:
                if self.is_playing:
                    elapsed = time.time() - t0
                    delay_ms = max(1, int((frame_period - elapsed) * 1000))
                else:
                    delay_ms = 30

            raw_key = cv2.waitKeyEx(delay_ms)
            ascii_key = raw_key & 0xFF if raw_key != -1 else -1

            if ascii_key in (27, ord('q'), ord('Q')):
                break
            elif ascii_key == 32:  # SPASI: Play/Pause atau Live/Freeze
                self.is_playing = not self.is_playing
                if self.is_webcam:
                    status_str = "▶ LIVE STREAMING AKTIF" if self.is_playing else "⏸ FREEZE FRAME (PAUSED / TAHAN)"
                    print(f"  ⏯️ [SPASI] Kamera Webcam {status_str}")
                else:
                    status_str = "▶ BERJALAN OTOMATIS" if self.is_playing else "⏸ DIHENTIKAN (PAUSED)"
                    print(f"  ⏯️ [SPASI] Video {status_str}")
            elif ascii_key in (ord('d'), ord('D')) or raw_key == 2555904:  # Step Forward
                if self.is_webcam:
                    # Ambil frame baru saat freeze
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        self.current_frame_idx += 1
                        self.current_frame = frame
                        print(f"  ⏩ [D] Freeze Frame: Mengambil frame baru #{self.current_frame_idx}")
                else:
                    self.is_playing = False
                    self.seek_to(self.current_frame_idx + 1)
                    self.internal_slider_update = True
                    cv2.setTrackbarPos("Frame", self.win_main, self.current_frame_idx)
                    self.internal_slider_update = False
            elif ascii_key in (ord('a'), ord('A')) or raw_key == 2424832:  # Step Backward
                if not self.is_webcam:
                    self.is_playing = False
                    self.seek_to(self.current_frame_idx - 1)
                    self.internal_slider_update = True
                    cv2.setTrackbarPos("Frame", self.win_main, self.current_frame_idx)
                    self.internal_slider_update = False
            elif ascii_key in (ord('e'), ord('E')):  # Ekspor CSV
                self.save_csv_log()
            elif ascii_key in (ord('p'), ord('P')):  # Simpan Preset Slider
                self.save_current_preset()
            elif ascii_key in (ord('r'), ord('R')):  # Reset Counter Plat
                self.slat_counter = 0
                self.ok_count = 0
                self.ng_count = 0
                self.last_trigger_frame = -999
                self.slat_latched = False
                self.inspection_records = []
                self.ng_consecutive = 0
                self.hold_ng_timer = 0
                self.is_ng_stable = False
                self.stable_def_angle = 0.0
                self.stable_def_count = 0
                self.cached_lines = None
                self.line_fade_frames = 0
                print("  🔄 [RESET] Counter slat dan rekaman di-reset ke 0.")
            elif ascii_key in (ord('s'), ord('S')):  # Snapshot
                if self.is_webcam:
                    snap_name = f"snap_webcam_cam{self.cam_index}_{int(time.time())}.jpg"
                else:
                    snap_name = f"snap_frame_{self.current_frame_idx:04d}_{int(time.time())}.jpg"
                p = os.path.join(SNAPSHOT_DIR, snap_name)
                cv2.imwrite(p, self.current_frame)
                print(f"  💾 [SNAPSHOT] Frame tersimpan: {p}")
            elif ascii_key in (9, ord('c'), ord('C')):  # TAB / C : Ganti Kamera Webcam
                self.save_current_preset()
                self.save_csv_log()
                cv2.destroyAllWindows()
                self.cap.release()
                return 'SWITCH_CAMERA'
            elif ascii_key in (ord('v'), ord('V')):  # V : Ganti Video / Mode
                self.save_current_preset()
                self.save_csv_log()
                cv2.destroyAllWindows()
                self.cap.release()
                return 'SWITCH_VIDEO'

            # Cek jika window ditutup
            if cv2.getWindowProperty(self.win_main, cv2.WND_PROP_VISIBLE) < 1:
                break

        # Simpan otomatis konfigurasi preset dan riwayat CSV sebelum keluar
        self.save_current_preset()
        self.save_csv_log()
        cv2.destroyAllWindows()
        self.cap.release()
        return 'EXIT'


def main():
    parser = argparse.ArgumentParser(description="Inspeksi Video & Live Webcam Konveyor Menggunakan Hough Line.")
    parser.add_argument("--video", "-v", type=str, default=None, help="Path file video konveyor")
    parser.add_argument("--cam", "--camera", "-c", type=int, default=None, help="Indeks kamera webcam (misal: 0, 1)")
    parser.add_argument("--webcam", "-w", action="store_true", help="Gunakan live webcam secara langsung (default: Cam 0)")
    parser.add_argument("--list-cams", action="store_true", help="Deteksi dan tampilkan daftar kamera webcam yang tersedia")
    args = parser.parse_args()

    available_cams = detect_available_cameras(max_tested=4)

    if args.list_cams:
        print("\n🔍 === DAFTAR KAMERA WEBCAM TERSEDIA ===")
        if available_cams:
            for c in available_cams:
                print(f"  📷 Kamera #{c} (Tersedia & Siap Digunakan)")
        else:
            print("  [!] Tidak ada kamera webcam yang terdeteksi.")
        print("=======================================\n")
        sys.exit(0)

    # 1. Jika pengguna menentukan --cam / --camera / --webcam
    if args.cam is not None or args.webcam:
        cam_idx = args.cam if args.cam is not None else (available_cams[0] if available_cams else 0)
        cams_list = available_cams if available_cams else [cam_idx]
        current_cam_pos = cams_list.index(cam_idx) if cam_idx in cams_list else 0
        
        while True:
            target_cam = cams_list[current_cam_pos]
            try:
                app = VideoHoughInspector(target_cam)
                res = app.run()
                if res == 'SWITCH_CAMERA':
                    current_cam_pos = (current_cam_pos + 1) % len(cams_list)
                    print(f"\n🔄 Beralih ke Kamera Webcam: CAM #{cams_list[current_cam_pos]}")
                    continue
                elif res == 'SWITCH_VIDEO':
                    if DEFAULT_VIDEOS:
                        print("\n🔄 Beralih dari Webcam ke Video Testing...")
                        break
                    else:
                        print("  [INFO] Tidak ada video default untuk beralih.")
                        continue
                else:
                    print("\nProgram selesai. Sampai jumpa!")
                    sys.exit(0)
            except Exception as e:
                print(f"\n[ERROR] Terjadi kesalahan saat membuka webcam #{target_cam}: {e}")
                break

    # 2. Kumpulkan file video yang tersedia
    available_videos = []
    if args.video and os.path.exists(args.video):
        available_videos.append(args.video)

    for v in DEFAULT_VIDEOS:
        if os.path.exists(v) and v not in available_videos:
            available_videos.append(v)

    # 3. Jika dijalankan tanpa argumen khusus, tampilkan menu interaktif
    if not args.video and not args.cam and not args.webcam:
        print("\n" + "="*56)
        print("   SISTEM INSPEKSI HOUGH LINE KONVEYOR SLAT")
        print("="*56)
        print("  Pilih Mode Input:")
        if available_cams:
            print(f"  [1] 📷 Live Webcam (Kamera terdeteksi: #{available_cams})")
        else:
            print("  [1] 📷 Live Webcam (Coba buka Cam #0)")
        print(f"  [2] 🎬 Video Testing ({len(available_videos)} file tersedia)")
        print("  [3] 📂 Pilih File Video Manual (File Dialog)")
        print("="*56)
        
        try:
            choice = input("  Masukkan pilihan [1/2/3] (Default [1] Webcam / [2] Video): ").strip()
        except (EOFError, KeyboardInterrupt):
            choice = ""

        if choice == "1" or (choice == "" and available_cams and not available_videos):
            target_cam = available_cams[0] if available_cams else 0
            cams_list = available_cams if available_cams else [0]
            current_cam_pos = 0
            while True:
                target_cam = cams_list[current_cam_pos]
                try:
                    app = VideoHoughInspector(target_cam)
                    res = app.run()
                    if res == 'SWITCH_CAMERA':
                        current_cam_pos = (current_cam_pos + 1) % len(cams_list)
                        print(f"\n🔄 Beralih ke Kamera Webcam: CAM #{cams_list[current_cam_pos]}")
                        continue
                    elif res == 'SWITCH_VIDEO':
                        if available_videos:
                            break
                        else:
                            continue
                    else:
                        print("\nProgram selesai. Sampai jumpa!")
                        sys.exit(0)
                except Exception as e:
                    print(f"\n[ERROR] Gagal menjalankan webcam #{target_cam}: {e}")
                    break

        elif choice == "3":
            chosen = select_video_dialog()
            if chosen and os.path.exists(chosen):
                available_videos.insert(0, chosen)

    # 4. Fallback jika video kosong
    if not available_videos:
        chosen = select_video_dialog()
        if chosen and os.path.exists(chosen):
            available_videos.append(chosen)

    if not available_videos and not available_cams:
        print("[ERROR] Tidak ada file video atau webcam yang ditemukan!")
        sys.exit(1)
    elif not available_videos and available_cams:
        print("[INFO] Tidak ada file video, menjalankan mode Live Webcam Cam 0...")
        app = VideoHoughInspector(available_cams[0])
        app.run()
        sys.exit(0)

    # 5. Loop pemutaran video
    vid_idx = 0
    while True:
        target_video = available_videos[vid_idx]
        app = VideoHoughInspector(target_video)
        res = app.run()
        if res == 'SWITCH_VIDEO':
            vid_idx = (vid_idx + 1) % len(available_videos)
            print(f"\n🔄 Beralih ke video: {os.path.basename(available_videos[vid_idx])}")
            continue
        elif res == 'SWITCH_CAMERA':
            if available_cams:
                print(f"\n🔄 Beralih dari Video ke Live Webcam CAM #{available_cams[0]}...")
                webcam_app = VideoHoughInspector(available_cams[0])
                webcam_res = webcam_app.run()
                if webcam_res == 'SWITCH_VIDEO':
                    continue
                else:
                    break
            else:
                print("  [WARN] Tidak ada webcam yang terdeteksi.")
                continue
        else:
            break

    print("\nProgram selesai. Sampai jumpa!")


if __name__ == '__main__':
    main()
