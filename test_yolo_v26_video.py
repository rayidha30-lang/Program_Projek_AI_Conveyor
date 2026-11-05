"""
================================================================================
PROGRAM BENCHMARK & TESTER MODEL YOLO v26 (BOLT & PIN INSPECTION)
File: test_yolo_v26_video.py
Proyek: AI Slat & Rib Conveyor - YOLO-Pose rib inspection
================================================================================
Fitur Utama:
  1. ANALISIS BENDING RIB:
     - Deteksi rib dan 13 keypoint dengan YOLO-Pose.
     - Rectifikasi perspektif lalu ukur deviasi profil P95/MAX.
     - Temporal voting pada zona inspeksi satu rib.
     - Slider interaktif (Trackbar OpenCV) untuk mengatur Confidence & IoU / NMS
       secara langsung saat video diputar untuk menguji sensitivitas model.
     - Akumulasi statistik deteksi (total deteksi per kelas & rata-rata confidence).
  2. BENCHMARK FPS & LATENSI PRESISI:
     - Pengukuran latensi per fase (Preprocess, Inference, Postprocess / NMS) dalam ms.
     - Rolling FPS rata-rata (window 30 frame) dan instant FPS.
     - Dukungan akselerasi CUDA / GPU NVIDIA otomatis (fallback ke CPU jika tidak ada).
  3. KONTROL INTERAKTIF VIDEO:
     - [SPASI]       : Play / Pause
     - [D] / [Kanan] : Maju 1 frame (saat pause)
     - [A] / [Kiri]  : Mundur 1 frame (saat pause)
     - [S]           : Simpan snapshot frame & hasil deteksi ke folder output
     - [O] / [V]     : Buka file video baru via File Dialog
     - [T]           : Sembunyikan / Tampilkan HUD Overlay
     - [R]           : Reset akumulasi statistik
     - [Q] / [ESC]   : Keluar dan cetak Laporan Rangkuman Benchmark (FPS & Akurasi)
================================================================================
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path
from collections import deque
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Setup encoding UTF-8 untuk konsol Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ==============================================================================
# KONFIGURASI DEFAULT
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent

# Lokasi model v26 default
DEFAULT_MODEL_CANDIDATES = [
    BASE_DIR / "Vision_Engine_Core" / "02_Top_Inspection_Slat_Rib" / "models" / "best_rib_pose.pt"
]

DEFAULT_VIDEO_PATH = Path(r"C:\Users\Victus\Pictures\Video & Foto Test Slat Rib\New\Reverse\Kamera posisi 3 dimmer 2 - Reverse.mp4")
DEFAULT_VIDEO_DIR = DEFAULT_VIDEO_PATH.parent
OUTPUT_DIR = BASE_DIR / "output_test_v26"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PERSPECTIVE_CALIBRATION_FILE = BASE_DIR / "perspective_calibration.json"

# Skema warna kelas (BGR)
CLASS_COLORS = {
    "bolt_ok": (0, 235, 120),    # Hijau Emerald Cerah
    "pin_ok": (255, 180, 0),     # Biru Cyan / Deep Sky
    "bolt_ng": (40, 40, 255),    # Merah Cerah
    "pin_ng": (0, 140, 255),     # Oranye Cerah
}
DEFAULT_COLOR = (200, 200, 200)
CONTOUR_STATUS_COLORS = {
    "OK": (0, 235, 120),
    "NG": (40, 40, 255),
    "WAIT": (0, 180, 255),
}
DEFAULT_ROBUST_NG_THRESHOLD = 0.020

WINDOW_NAME = "Perspective Calibration - Source View"
RECTIFIED_WINDOW_NAME = "Perspective Transform Preview"


def find_default_model():
    """Mencari file model default yang tersedia di direktori proyek."""
    for cand in DEFAULT_MODEL_CANDIDATES:
        if cand.exists():
            return str(cand)
    return str(DEFAULT_MODEL_CANDIDATES[0])


def run_pose_self_check():
    """Memverifikasi scoring profil keypoint pose sintetis."""
    analyzer = RibProfileAnalyzer(ng_threshold=0.02, min_span_px=30, min_points=7)
    x = np.linspace(0, 200, 13, dtype=np.float32)
    straight = np.column_stack((x, np.full_like(x, 50)))
    assert analyzer.analyze_points(straight)["status"] == "OK"
    bent = np.column_stack((x, 50 + 15 * ((x - 100) / 100) ** 2))
    assert analyzer.analyze_points(bent)["status"] == "NG"
    assert analyzer.analyze_points(straight[:4])["status"] == "WAIT"

    reference = np.random.default_rng(42).integers(0, 255, (180, 280, 3), dtype=np.uint8)
    anchors = [(50, 45), (225, 50), (220, 135), (55, 140)]
    stabilizer = PerspectiveStabilizer(template_half_size=12, min_match_score=0.80)
    stabilizer.start_calibration()
    for point in anchors:
        assert stabilizer.add_anchor(reference, *point)

    translation = np.float32([[1, 0, 9], [0, 1, -7]])
    shifted = cv2.warpAffine(reference, translation, (280, 180), borderMode=cv2.BORDER_REFLECT)
    aligned, valid, score = stabilizer.stabilize(shifted)
    assert valid and score >= 0.80
    assert np.mean(np.abs(aligned[20:160, 20:260].astype(np.float32) - reference[20:160, 20:260])) < 3.0

    gated = apply_calibration_gate({"status": "NG", "score": 0.03}, calibration_enabled=True, calibration_valid=False)
    assert gated["status"] == "WAIT" and gated["score"] is None

    vote = RibVoteTracker(min_frames=3, max_shift_px=50)
    assert vote.update((100, 100), 4.0, threshold=3.0) == "WAIT"
    assert vote.update((104, 101), 4.0, threshold=3.0) == "WAIT"
    assert vote.update((108, 102), 0.5, threshold=3.0) == "NG"

    print("[OK] Self-check: pose profile OK/NG/WAIT dan perspective stabilization")


def pick_video_file_dialog(initial_dir=None):
    """Membuka dialog file explorer Windows untuk memilih video."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    start_dir = str(initial_dir) if initial_dir and os.path.exists(initial_dir) else str(BASE_DIR)
    file_path = filedialog.askopenfilename(
        title="Pilih File Video untuk Pengujian YOLO v26",
        initialdir=start_dir,
        filetypes=[
            ("Video Files", "*.mp4;*.avi;*.mov;*.mkv;*.wmv"),
            ("All Files", "*.*")
        ]
    )
    root.destroy()
    return file_path if file_path else None


class RibProfileAnalyzer:
    """Mengukur deviasi keypoint YOLO-Pose terhadap garis robust di bidang kanonikal."""

    def __init__(self, ng_threshold=DEFAULT_ROBUST_NG_THRESHOLD, min_span_px=80, min_points=7):
        self.ng_threshold = ng_threshold
        self.min_span_px = min_span_px
        self.min_points = min_points

    @staticmethod
    def wait(points=None, span_px=0.0):
        return {
            "status": "WAIT", "score": None, "p95": None, "peak": None,
            "centreline": points, "baseline": None, "span_px": span_px,
        }

    def analyze_points(self, points):
        if points is None:
            return self.wait()
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2:
            return self.wait()
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) < self.min_points:
            return self.wait(points)
        points = points[np.argsort(points[:, 0])]
        points = points[np.concatenate(([True], np.diff(points[:, 0]) > 1.0))]
        span_px = float(points[-1, 0] - points[0, 0]) if len(points) else 0.0
        if len(points) < self.min_points or span_px < self.min_span_px:
            return self.wait(points, span_px)

        # Single-point pose spikes are prediction noise; retain bends spanning
        # multiple adjacent keypoints while filtering isolated jumps.
        padded_y = np.pad(points[:, 1], (1, 1), mode="edge")
        filtered_y = np.median(
            np.stack((padded_y[:-2], padded_y[1:-1], padded_y[2:])), axis=0
        )
        profile_points = np.column_stack((points[:, 0], filtered_y)).astype(np.float32)
        fit_points = profile_points
        vx, vy, fit_x, fit_y = cv2.fitLine(fit_points, cv2.DIST_WELSCH, 0, 0.01, 0.01).flatten()
        if abs(float(vx)) < 1e-6:
            return self.wait(points, span_px)

        baseline_y = fit_y + (profile_points[:, 0] - fit_x) * vy / vx
        baseline = np.column_stack((profile_points[:, 0], baseline_y)).astype(np.float32)
        residual = np.abs(
            (profile_points[:, 0] - fit_x) * vy - (profile_points[:, 1] - fit_y) * vx
        )
        p95 = float(np.percentile(residual, 95) / span_px)
        peak = float(np.max(residual) / span_px)
        score = max(p95, peak)
        return {
            "status": "NG" if score > self.ng_threshold else "OK",
            "score": score,
            "p95": p95,
            "peak": peak,
            "centreline": profile_points,
            "baseline": baseline,
            "span_px": span_px,
        }


class PerspectiveStabilizer:
    """Menstabilkan frame terhadap empat titik acuan tetap pada rangka conveyor."""

    def __init__(self, template_half_size=24, min_match_score=0.70):
        self.template_half_size = template_half_size
        self.min_match_score = min_match_score
        self.reference_points = []
        self.templates = []
        self.collecting = False

    @property
    def ready(self):
        return len(self.reference_points) == 4 and len(self.templates) == 4

    def start_calibration(self):
        self.reference_points.clear()
        self.templates.clear()
        self.collecting = True

    def reset(self):
        self.reference_points.clear()
        self.templates.clear()
        self.collecting = False

    def add_anchor(self, frame, x, y):
        """Menyimpan patch unik di titik rangka yang dipilih pengguna."""
        if not self.collecting or self.ready:
            return False

        h, w = frame.shape[:2]
        r = self.template_half_size
        if x - r < 0 or y - r < 0 or x + r >= w or y + r >= h:
            return False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        template = gray[y - r:y + r + 1, x - r:x + r + 1].copy()
        if template.std() < 4.0:
            return False

        self.reference_points.append((float(x), float(y)))
        self.templates.append(template)
        if self.ready:
            self.collecting = False
        return True

    def stabilize(self, frame):
        """Mengembalikan frame pada pose referensi dan status kecocokan anchor."""
        if not self.ready:
            return frame, False, None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_points = []
        match_scores = []
        for template in self.templates:
            response = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            _, score, _, location = cv2.minMaxLoc(response)
            r = self.template_half_size
            current_points.append((location[0] + r, location[1] + r))
            match_scores.append(float(score))

        min_score = min(match_scores)
        if min_score < self.min_match_score:
            return frame, False, min_score

        transform = cv2.getPerspectiveTransform(
            np.float32(current_points), np.float32(self.reference_points)
        )
        h, w = frame.shape[:2]
        stabilized = cv2.warpPerspective(frame, transform, (w, h), flags=cv2.INTER_LINEAR)
        return stabilized, True, min_score


class PerspectiveRectifier:
    """Memetakan bidang inspeksi conveyor menjadi bidang persegi panjang."""

    def __init__(self):
        self.points = []
        self.collecting = False
        self.matrix = None
        self.inverse_matrix = None
        self.output_size = None

    @property
    def ready(self):
        return self.matrix is not None

    def start_calibration(self):
        self.points.clear()
        self.collecting = True
        self.matrix = None
        self.inverse_matrix = None
        self.output_size = None

    def reset(self):
        self.points.clear()
        self.collecting = False
        self.matrix = None
        self.inverse_matrix = None
        self.output_size = None

    def add_point(self, frame, x, y):
        """Ambil sudut berurutan: kiri-atas, kanan-atas, kanan-bawah, kiri-bawah."""
        if not self.collecting or self.ready:
            return False
        h, w = frame.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return False
        if any(np.linalg.norm(np.asarray((x, y), dtype=np.float32) - point) < 10 for point in self.points):
            return False

        self.points.append(np.float32((x, y)))
        if len(self.points) < 4:
            return True

        src = np.asarray(self.points, dtype=np.float32)
        if not cv2.isContourConvex(src.reshape(-1, 1, 2)):
            self.reset()
            return False

        width = int(round((np.linalg.norm(src[1] - src[0]) + np.linalg.norm(src[2] - src[3])) / 2))
        height = int(round((np.linalg.norm(src[3] - src[0]) + np.linalg.norm(src[2] - src[1])) / 2))
        if width < 80 or height < 40:
            self.reset()
            return False

        dst = np.float32(((0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)))
        matrix = cv2.getPerspectiveTransform(src, dst)
        if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-10:
            self.reset()
            return False

        self.matrix = matrix
        self.inverse_matrix = np.linalg.inv(matrix)
        self.output_size = (width, height)
        self.collecting = False
        return True

    def map_to_source(self, points, offset=(0, 0)):
        if not self.ready or points is None or len(points) == 0:
            return points
        points = np.asarray(points, dtype=np.float32) + np.asarray(offset, dtype=np.float32)
        points = points.reshape(-1, 1, 2)
        return cv2.perspectiveTransform(points, self.inverse_matrix).reshape(-1, 2)

    def map_to_rectified(self, points):
        if not self.ready or points is None or len(points) == 0:
            return points
        points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(points, self.matrix).reshape(-1, 2)


def apply_calibration_gate(analysis, calibration_enabled, calibration_valid):
    """Menahan keputusan pose saat anchor stabilisasi gagal ditemukan."""
    if calibration_enabled and not calibration_valid:
        gated = dict(analysis)
        gated["status"] = "WAIT"
        gated["score"] = None
        gated["robust_status"] = "WAIT"
        gated["robust_score"] = None
        return gated
    return analysis


class RibVoteTracker:
    """Menggabungkan beberapa skor berurutan dari rib yang sama dalam satu zona."""

    def __init__(self, min_frames=3, max_frames=5, max_shift_px=100):
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.max_shift_px = max_shift_px
        self.last_center = None
        self.last_bbox = None
        self.scores = deque(maxlen=max_frames)

    def reset(self):
        self.last_center = None
        self.last_bbox = None
        self.scores.clear()

    def update(self, center, score, threshold, bbox=None):
        center = np.asarray(center, dtype=np.float32)
        if self.last_center is not None and np.linalg.norm(center - self.last_center) > self.max_shift_px:
            self.scores.clear()
        if bbox is not None and self.last_bbox is not None:
            x1, y1, x2, y2 = bbox
            px1, py1, px2, py2 = self.last_bbox
            overlap_w = max(0.0, min(x2, px2) - max(x1, px1))
            overlap_h = max(0.0, min(y2, py2) - max(y1, py1))
            overlap = overlap_w * overlap_h
            area = min((x2 - x1) * (y2 - y1), (px2 - px1) * (py2 - py1))
            if area <= 0 or overlap / area < 0.20:
                self.scores.clear()
        self.last_center = center
        self.last_bbox = bbox
        self.scores.append(float(score))
        if len(self.scores) < self.min_frames:
            return "WAIT"
        ng_votes = sum(value > threshold for value in self.scores)
        return "NG" if ng_votes >= (len(self.scores) + 1) // 2 else "OK"


class VideoYoloTester:
    def __init__(self, model_path: str = None, video_path: str = None, conf_thres: float = 0.25, iou_thres: float = 0.45, device: str = None, preview_only: bool = True):
        self.model_path = model_path
        self.preview_only = preview_only
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.show_hud = True
        self.paused = False
        self.current_raw_frame = None
        self.current_inference_frame = None
        self.stabilizer = PerspectiveStabilizer()
        self.rectifier = PerspectiveRectifier()
        self.current_rectified_preview = None
        self.profile_roi = None
        self.selecting_profile_roi = False
        self.profile_roi_first_point = None
        self.calibration_valid = False
        self.calibration_score = None
        self.inspection_zone = None
        self.selecting_zone = False
        self.zone_start = None
        self.robust_vote_tracker = RibVoteTracker()
        self.robust_status_counts = {"OK": 0, "NG": 0, "WAIT": 0}
        self.robust_scores = []

        # Tentukan device komputasi (CUDA GPU jika tersedia, jika tidak CPU)
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if self.preview_only:
            self.model = None
            self.class_names = {0: "rib"}
            print("[INFO] Mode preview perspektif aktif; YOLO dan analisis bending tidak dijalankan.")
        else:
            print(f"\n{'='*70}")
            print(f" Memuat Model YOLO-Pose: {Path(self.model_path).name}")
            print(f" Path: {self.model_path}")
            print(f" Device Hardware: {self.device.upper()}", end="")
            if "cuda" in self.device and torch.cuda.is_available():
                print(f" ({torch.cuda.get_device_name(0)})")
            else:
                print(" (Processor CPU)")
            print(f"{'='*70}\n")

            self.model = YOLO(self.model_path)
            if getattr(self.model, "task", None) != "pose":
                raise ValueError(f"Model harus YOLO-Pose 13 keypoint, tetapi task model adalah {getattr(self.model, 'task', 'unknown')}")
            self.class_names = self.model.names if hasattr(self.model, 'names') else {0: 'rib'}
            print(f"[INFO] Kelas Deteksi Terdaftar: {self.class_names}")
            self._warmup_model()

        # Konfigurasi Video
        self.video_path = video_path
        self.cap = None
        self.total_video_frames = 0
        self.fps_video_source = 30.0

        # Metrik Latensi & FPS
        self.rolling_fps = deque(maxlen=30)
        self.history_preprocess = deque(maxlen=30)
        self.history_inference = deque(maxlen=30)
        self.history_postprocess = deque(maxlen=30)
        self.history_total_latency = deque(maxlen=30)

        # Statistik Keseluruhan (Untuk Final Report)
        self.total_processed_frames = 0
        self.all_latencies_ms = []
        self.all_inference_ms = []
        self.detection_counts = {name: 0 for name in self.class_names.values()}
        self.detection_confs = {name: [] for name in self.class_names.values()}

        # Slider Trackbar State
        self.slider_conf = int(self.conf_thres * 100)
        self.slider_iou = int(self.iou_thres * 100)
        self.robust_threshold = DEFAULT_ROBUST_NG_THRESHOLD
        self.slider_robust = int(self.robust_threshold * 1000)
        self.profile_analyzer = RibProfileAnalyzer(ng_threshold=self.robust_threshold)

    def _warmup_model(self):
        """Melakukan warm-up inferensi sebanyak 3 iterasi agar GPU clock stabil."""
        print("[INFO] Melakukan warm-up model untuk benchmark stabil...")
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(3):
            _ = self.model.predict(source=dummy_img, device=self.device, verbose=False, conf=self.conf_thres)
        if torch.cuda.is_available() and "cuda" in self.device:
            torch.cuda.synchronize()
        print("[INFO] Warm-up selesai. Siap memulai pengujian.\n")

    def open_video(self, video_path: str) -> bool:
        """Membuka video stream atau webcam."""
        if self.cap is not None:
            self.cap.release()

        # Cek jika string angka (misal "0" untuk webcam)
        if isinstance(video_path, str) and video_path.isdigit():
            src = int(video_path)
            self.cap = cv2.VideoCapture(src)
        else:
            self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            print(f"[ERROR] Gagal membuka sumber video: {video_path}")
            return False

        self.video_path = video_path
        self.total_video_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps_video_source = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"[OK] Video Berhasil Dibuka:")
        print(f"     Sumber        : {self.video_path}")
        print(f"     Resolusi Asli : {width} x {height}")
        print(f"     Total Frame   : {self.total_video_frames}")
        print(f"     FPS Asli      : {self.fps_video_source:.1f}\n")
        return True

    def _on_conf_change(self, val):
        self.slider_conf = max(1, val)
        self.conf_thres = self.slider_conf / 100.0

    def _on_iou_change(self, val):
        self.slider_iou = max(1, val)
        self.iou_thres = self.slider_iou / 100.0

    def _on_robust_change(self, val):
        self.slider_robust = max(1, val)
        self.robust_threshold = self.slider_robust / 1000.0
        self.profile_analyzer.ng_threshold = self.robust_threshold

    def setup_window(self):
        """Membuat window OpenCV dengan slider Trackbar interaktif."""
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.createTrackbar("Conf (%)", WINDOW_NAME, self.slider_conf, 100, self._on_conf_change)
        cv2.createTrackbar("IoU/NMS (%)", WINDOW_NAME, self.slider_iou, 100, self._on_iou_change)
        cv2.createTrackbar("Robust x0.1%", WINDOW_NAME, self.slider_robust, 100, self._on_robust_change)
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse_event)

    def _on_mouse_event(self, event, x, y, flags, param):
        """Menerima empat klik anchor saat mode kalibrasi aktif."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if self.selecting_profile_roi:
            if self.current_inference_frame is None or not self.rectifier.ready:
                return
            source_point = self._map_window_click_to_image(
                self.current_inference_frame, x, y
            )
            rectified_point = self.rectifier.map_to_rectified(
                np.asarray([source_point], dtype=np.float32)
            )[0]
            self._add_profile_roi_point(rectified_point)
            return

        if self.selecting_zone:
            if self.zone_start is None:
                self.zone_start = (x, y)
                print("[ZONA] Titik pertama tersimpan. Klik titik berlawanan untuk menyelesaikan zona.")
            else:
                x0, y0 = self.zone_start
                self.inspection_zone = (min(x0, x), min(y0, y), max(x0, x), max(y0, y))
                self.selecting_zone = False
                self.zone_start = None
                self.robust_vote_tracker.reset()
                print("[ZONA] Zona inspeksi tersimpan. Buat setinggi satu rib agar kandidat tidak bercampur.")
            return

        if self.rectifier.collecting and self.current_inference_frame is not None:
            added = self.rectifier.add_point(self.current_inference_frame, x, y)
            if not added:
                print("[RECTIFY] Sudut tidak valid atau urutannya membentuk bidang silang; kalibrasi dibatalkan.")
            elif self.rectifier.ready:
                print(f"[RECTIFY] Bidang inspeksi diratakan ke {self.rectifier.output_size[0]} x {self.rectifier.output_size[1]} px.")
            else:
                print(f"[RECTIFY] Sudut {len(self.rectifier.points)}/4 tersimpan. Urutan: kiri-atas, kanan-atas, kanan-bawah, kiri-bawah.")
            return

        if not self.stabilizer.collecting or self.current_raw_frame is None:
            return

        added = self.stabilizer.add_anchor(self.current_raw_frame, x, y)
        if not added:
            print("[KALIBRASI] Titik tidak valid. Pilih detail rangka yang unik dan tidak dekat tepi frame.")
            return

        count = len(self.stabilizer.reference_points)
        if self.stabilizer.ready:
            print("[KALIBRASI] Empat anchor tersimpan. Stabilisasi perspektif aktif.")
        else:
            print(f"[KALIBRASI] Anchor {count}/4 tersimpan. Urutan: kiri-atas, kanan-atas, kanan-bawah, kiri-bawah.")

    @staticmethod
    def _map_window_click_to_image(image, x, y):
        image_h, image_w = image.shape[:2]
        # On Windows HighGUI already converts resized-window coordinates to image coordinates.
        px = int(np.clip(round(x), 0, image_w - 1))
        py = int(np.clip(round(y), 0, image_h - 1))
        return px, py

    def _add_profile_roi_point(self, point):
        if self.current_rectified_preview is None:
            return
        image_h, image_w = self.current_rectified_preview.shape[:2]
        px = int(np.clip(round(float(point[0])), 0, image_w - 1))
        py = int(np.clip(round(float(point[1])), 0, image_h - 1))
        if self.profile_roi_first_point is None:
            self.profile_roi_first_point = (px, py)
            print(f"[ROI] Titik pertama ({px}, {py}). Klik sudut berlawanan di salah satu jendela.")
            return

        x0, y0 = self.profile_roi_first_point
        x1, y1 = px, py
        roi = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        if roi[2] - roi[0] < 80 or roi[3] - roi[1] < 5:
            print("[ROI] Area terlalu kecil. Pilih rentang lebar sepanjang bibir rib dan tinggi beberapa piksel.")
            self.profile_roi_first_point = None
            return
        self.profile_roi = roi
        self.selecting_profile_roi = False
        self.profile_roi_first_point = None
        print(f"[ROI] Area profil tersimpan: {roi}. Tepi atas/bawah dan garis tengah ditampilkan.")

    def _on_preview_mouse_event(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or not self.selecting_profile_roi:
            return
        if self.current_rectified_preview is None:
            return
        point = self._map_window_click_to_image(
            self.current_rectified_preview, x, y
        )
        self._add_profile_roi_point(point)

    def draw_calibration_overlay(self, frame):
        """Menampilkan status dan lokasi titik acuan pada frame yang sedang dilihat."""
        if self.stabilizer.collecting:
            status = f"KALIBRASI: klik {len(self.stabilizer.reference_points) + 1}/4 (TL, TR, BR, BL)"
            color = (0, 220, 255)
        elif self.stabilizer.ready and self.calibration_valid:
            status = f"CAL STABIL | match {self.calibration_score:.2f}"
            color = (0, 235, 120)
        elif self.stabilizer.ready:
            score_text = "NA" if self.calibration_score is None else f"{self.calibration_score:.2f}"
            status = f"CAL LOST | match {score_text}"
            color = (40, 40, 255)
        else:
            status = "CAL OFF | tekan C untuk 4 anchor rangka"
            color = (160, 160, 160)

        cv2.putText(frame, status, (18, frame.shape[0] - 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        for index, point in enumerate(self.stabilizer.reference_points, start=1):
            x, y = map(int, point)
            cv2.circle(frame, (x, y), 7, color, 2, cv2.LINE_AA)
            cv2.putText(frame, str(index), (x + 9, y - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    def draw_rectification_overlay(self, frame):
        if self.rectifier.collecting:
            status = f"RECTIFY: klik {len(self.rectifier.points) + 1}/4 (TL, TR, BR, BL)"
            color = (0, 220, 255)
        elif self.rectifier.ready:
            status = f"RECTIFIED | {self.rectifier.output_size[0]}x{self.rectifier.output_size[1]} px"
            color = (0, 235, 120)
        else:
            status = "RECTIFY OFF | tekan P, klik 4 sudut bidang conveyor"
            color = (160, 160, 160)

        cv2.putText(frame, status, (18, frame.shape[0] - 78), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2, cv2.LINE_AA)
        for index, point in enumerate(self.rectifier.points, start=1):
            x, y = map(int, point)
            cv2.circle(frame, (x, y), 6, color, 2, cv2.LINE_AA)
            cv2.putText(frame, str(index), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    def map_rectified_analysis_to_source(self, analysis):
        mapped = dict(analysis)
        for key in ("centreline", "baseline"):
            points = analysis.get(key)
            if points is not None and len(points):
                mapped[key] = self.rectifier.map_to_source(points)
        return mapped

    def draw_inspection_zone(self, frame):
        """Menggambar zona satu-rib tempat temporal voting pose dijalankan."""
        if self.inspection_zone is None:
            status = "ZONE OFF | tekan Z lalu klik dua sudut zona"
            color = (160, 160, 160)
        else:
            x1, y1, x2, y2 = self.inspection_zone
            color = (255, 180, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            status = "ZONE | satu rib untuk temporal vote"

        cv2.putText(frame, status, (18, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2, cv2.LINE_AA)

    def point_in_inspection_zone(self, point):
        if self.inspection_zone is None:
            return False
        x1, y1, x2, y2 = self.inspection_zone
        x, y = point
        return x1 <= x <= x2 and y1 <= y <= y2

    def draw_corner_box(self, img, pt1, pt2, color, thickness=2, corner_len=15):
        """Menggambar bounding box modern dengan aksen sudut (corner accent)."""
        x1, y1 = pt1
        x2, y2 = pt2

        # Bounding box utama tipis semi transparan
        cv2.rectangle(img, (x1, y1), (x2, y2), color, max(1, thickness - 1), lineType=cv2.LINE_AA)

        # Sudut tebal
        c_len = min(corner_len, (x2 - x1) // 3, (y2 - y1) // 3)
        th = thickness + 1

        # Kiri Atas
        cv2.line(img, (x1, y1), (x1 + c_len, y1), color, th, lineType=cv2.LINE_AA)
        cv2.line(img, (x1, y1), (x1, y1 + c_len), color, th, lineType=cv2.LINE_AA)
        # Kanan Atas
        cv2.line(img, (x2, y1), (x2 - c_len, y1), color, th, lineType=cv2.LINE_AA)
        cv2.line(img, (x2, y1), (x2, y1 + c_len), color, th, lineType=cv2.LINE_AA)
        # Kiri Bawah
        cv2.line(img, (x1, y2), (x1 + c_len, y2), color, th, lineType=cv2.LINE_AA)
        cv2.line(img, (x1, y2), (x1, y2 - c_len), color, th, lineType=cv2.LINE_AA)
        # Kanan Bawah
        cv2.line(img, (x2, y2), (x2 - c_len, y2), color, th, lineType=cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2, y2 - c_len), color, th, lineType=cv2.LINE_AA)

        # Titik pusat
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(img, (cx, cy), 3, color, -1, lineType=cv2.LINE_AA)

    def draw_pose_analysis(self, frame, analysis, status):
        """Menggambar keypoint dan garis dasar profil pose yang dipakai menilai bending."""
        color = CONTOUR_STATUS_COLORS.get(status, CONTOUR_STATUS_COLORS["WAIT"])
        centreline, baseline = analysis.get("centreline"), analysis.get("baseline")
        if centreline is not None:
            for point in centreline:
                if np.isfinite(point).all():
                    cv2.circle(frame, tuple(np.round(point).astype(int)), 3, (0, 255, 255), -1, cv2.LINE_AA)
        if centreline is not None and len(centreline) > 1:
            cv2.polylines(frame, [np.round(centreline).astype(np.int32)], False, color, 2, cv2.LINE_AA)
        if baseline is not None and len(baseline) > 1:
            cv2.polylines(frame, [np.round(baseline).astype(np.int32)], False, (255, 255, 255), 1, cv2.LINE_AA)

    def draw_hud(self, frame, current_fps, pre_ms, inf_ms, post_ms, current_counts):
        """Menggambar panel HUD (Heads-Up Display) modern di atas frame video."""
        if not self.show_hud:
            return

        h, w = frame.shape[:2]
        hud_w = 420
        hud_h = 240
        x0, y0 = 16, 16

        # Buat overlay semi-transparan hitam elegan
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + hud_w, y0 + hud_h), (20, 20, 24), -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

        # Border panel HUD
        cv2.rectangle(frame, (x0, y0), (x0 + hud_w, y0 + hud_h), (80, 85, 95), 1, lineType=cv2.LINE_AA)

        # Header Title
        title = "YOLO-POSE RIB BENDING INSPECTION"
        cv2.putText(frame, title, (x0 + 14, y0 + 26), cv2.FONT_HERSHEY_DUPLEX, 0.58, (0, 235, 255), 1, cv2.LINE_AA)
        cv2.line(frame, (x0 + 14, y0 + 34), (x0 + hud_w - 14, y0 + 34), (60, 65, 75), 1)

        # Meter FPS Utama
        fps_color = (0, 255, 120) if current_fps >= 25.0 else ((0, 220, 255) if current_fps >= 15.0 else (50, 50, 255))
        fps_text = f"FPS: {current_fps:5.1f}"
        cv2.putText(frame, fps_text, (x0 + 14, y0 + 64), cv2.FONT_HERSHEY_DUPLEX, 0.85, fps_color, 2, cv2.LINE_AA)

        total_lat = pre_ms + inf_ms + post_ms
        lat_text = f"Lat: {total_lat:4.1f} ms"
        cv2.putText(frame, lat_text, (x0 + 220, y0 + 64), cv2.FONT_HERSHEY_DUPLEX, 0.65, (220, 220, 220), 1, cv2.LINE_AA)

        # Rincian Latensi Profiling
        y_lat = y0 + 92
        cv2.putText(frame, f"Preprocess : {pre_ms:4.1f} ms", (x0 + 16, y_lat), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Inference  : {inf_ms:4.1f} ms", (x0 + 150, y_lat), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 240, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Post/NMS   : {post_ms:4.1f} ms", (x0 + 290, y_lat), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

        # Garis Pembatas
        cv2.line(frame, (x0 + 14, y0 + 104), (x0 + hud_w - 14, y0 + 104), (60, 65, 75), 1)

        # Statistik Deteksi Frame Saat Ini
        cv2.putText(frame, "DETEKSI RIB (FRAME INI):", (x0 + 14, y0 + 124), cv2.FONT_HERSHEY_DUPLEX, 0.45, (160, 200, 255), 1, cv2.LINE_AA)

        y_det = y0 + 148
        for cls_name, count in current_counts.items():
            color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)
            cum_count = self.detection_counts.get(cls_name, 0)
            text = f"• {cls_name:<10}: {count:2d} (Total: {cum_count})"
            cv2.putText(frame, text, (x0 + 18, y_det), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)
            y_det += 22

        # Status threshold & kontrol
        y_bottom = y0 + hud_h - 12
        info_sub = (
            f"C:{int(self.conf_thres*100)} I:{int(self.iou_thres*100)} "
            f"POSE-P95/MAX:{self.robust_threshold*100:.1f}% {self.device.upper()}"
        )
        cv2.putText(frame, info_sub, (x0 + 14, y_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 140, 150), 1, cv2.LINE_AA)

        # Petunjuk Keyboard di pojok kanan bawah
        help_bg = frame.copy()
        cv2.rectangle(help_bg, (w - 380, h - 38), (w - 16, h - 14), (20, 20, 24), -1)
        cv2.addWeighted(help_bg, 0.75, frame, 0.25, 0, frame)
        cv2.putText(frame, "[Z] Zona satu rib | [P] perspective | [U] reset | [R] statistik | [Q] keluar",
                    (w - 370, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

    @staticmethod
    def trace_profile_viterbi(gray, roi):
        """Lacak dua tepi rib dengan anchor lokal dan koridor Viterbi sempit."""
        x0, y0, x1, y1 = roi
        if x1 - x0 < 80 or y1 - y0 < 20:
            return None
        region = gray[y0:y1 + 1, x0:x1 + 1]
        if region.size == 0:
            return None
        xs = np.arange(0, region.shape[1], 4, dtype=np.int32)
        if len(xs) < 12:
            return None

        vertical_gradient = cv2.Sobel(region, cv2.CV_32F, 0, 1, ksize=3)
        padded_region = cv2.copyMakeBorder(region, 3, 3, 0, 0, cv2.BORDER_REPLICATE).astype(np.float32)
        above = padded_region[:region.shape[0]]
        below = padded_region[6:6 + region.shape[0]]
        contrast = np.abs(above - below)
        energy = np.clip(np.abs(vertical_gradient) * 0.5 + contrast * 0.5, 0, 100)
        sampled_energy = energy[:, xs]
        height, count = sampled_energy.shape
        split_row = height // 2
        if split_row < 5 or height - split_row < 5:
            return None

        anchor_columns = np.unique(np.linspace(0, count - 1, 7).round().astype(np.int32))

        def smooth_anchors(rows):
            rows = np.asarray(rows, dtype=np.float32)
            padded = np.pad(rows, (1, 1), mode="edge")
            return np.median(np.stack((padded[:-2], padded[1:-1], padded[2:])), axis=0)

        top_anchors = smooth_anchors([
            int(np.argmax(sampled_energy[:split_row, column]))
            for column in anchor_columns
        ])
        bottom_anchors = smooth_anchors([
            split_row + int(np.argmax(sampled_energy[split_row:, column]))
            for column in anchor_columns
        ])
        top_guide = np.interp(np.arange(count), anchor_columns, top_anchors)
        bottom_guide = np.interp(np.arange(count), anchor_columns, bottom_anchors)
        if np.median(bottom_anchors - top_anchors) < 8:
            return None

        corridor = min(14, max(4, (height - 1) // 2))
        offsets = np.arange(-corridor, corridor + 1, dtype=np.int32)
        shifts = (-2, -1, 0, 1, 2)
        penalties = (16.0, 4.0, 0.0, 4.0, 16.0)
        center = count // 2

        def trace(guide_rows):
            guided_energy = np.empty((len(offsets), count), dtype=np.float32)
            for column, guide_row in enumerate(guide_rows):
                rows = np.clip(int(round(guide_row)) + offsets, 0, height - 1)
                guided_energy[:, column] = sampled_energy[rows, column]

            def trace_direction(indices):
                scores = np.full((len(offsets), len(indices)), -1e9, dtype=np.float32)
                back = np.zeros((len(offsets), len(indices)), dtype=np.int32)
                scores[corridor, 0] = guided_energy[corridor, center]
                for step in range(1, len(indices)):
                    previous = scores[:, step - 1]
                    candidates = np.full((len(shifts), len(offsets)), -1e9, dtype=np.float32)
                    for shift_index, (shift, penalty) in enumerate(zip(shifts, penalties)):
                        if shift < 0:
                            candidates[shift_index, -shift:] = previous[:len(offsets) + shift] - penalty
                        elif shift > 0:
                            candidates[shift_index, :-shift] = previous[shift:] - penalty
                        else:
                            candidates[shift_index] = previous - penalty
                    best_shift = np.argmax(candidates, axis=0)
                    candidates_best = np.max(candidates, axis=0)
                    column = indices[step]
                    scores[:, step] = candidates_best + guided_energy[:, column]
                    back[:, step] = np.arange(len(offsets)) + np.asarray(shifts)[best_shift]
                state = int(np.argmax(scores[:, -1]))
                path = [state]
                for step in range(len(indices) - 1, 0, -1):
                    state = int(back[state, step])
                    path.append(state)
                path.reverse()
                return list(zip(indices, path))

            right = trace_direction(list(range(center, count)))
            left = trace_direction(list(range(center, -1, -1)))
            return list(reversed(left[1:])) + right

        top_path = trace(top_guide)
        bottom_path = trace(bottom_guide)
        path_x = xs[np.asarray([point[0] for point in top_path], dtype=np.int32)].astype(np.float32)

        def path_points(path, guide_rows):
            path_y = np.asarray(
                [guide_rows[point[0]] + offsets[point[1]] for point in path], dtype=np.float32
            )
            padded = np.pad(path_y, (1, 1), mode="edge")
            path_y = np.median(np.stack((padded[:-2], padded[1:-1], padded[2:])), axis=0)
            return np.column_stack((path_x + x0, path_y + y0)).astype(np.int32)

        top_edge = path_points(top_path, top_guide)
        bottom_edge = path_points(bottom_path, bottom_guide)
        centerline = np.column_stack((
            top_edge[:, 0],
            np.rint((top_edge[:, 1] + bottom_edge[:, 1]) / 2.0),
        )).astype(np.int32)
        return {"top": top_edge, "bottom": bottom_edge, "center": centerline}

    def run_perspective_preview(self):
        """Tampilkan frame sumber dan hasil homography tanpa menjalankan deteksi."""
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.namedWindow(RECTIFIED_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse_event)
        cv2.setMouseCallback(RECTIFIED_WINDOW_NAME, self._on_preview_mouse_event)
        frame = None
        frame_idx = 0
        calibration_checked = False
        print("\n[PREVIEW] Mode perspective transform saja; tidak ada inferensi YOLO.")
        print("          P pilih sudut; K simpan; R pilih ROI satu rib; U reset; SPASI jeda; S screenshot; Q keluar.\n")

        try:
            while True:
                if not self.paused:
                    ret, frame = self.cap.read()
                    if not ret:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = self.cap.read()
                    if not ret:
                        break
                    frame_idx += 1

                if frame is None:
                    break
                if not calibration_checked:
                    self.load_perspective_calibration(frame)
                    calibration_checked = True
                self.current_raw_frame = frame
                self.current_inference_frame = frame
                source_view = frame.copy()
                self.draw_rectification_overlay(source_view)
                cv2.putText(
                    source_view, "P: 4 sudut | K: simpan | R: ROI klik di salah satu jendela | U: reset | Q: keluar",
                    (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 235, 255), 2, cv2.LINE_AA,
                )
                cv2.imshow(WINDOW_NAME, source_view)

                if self.rectifier.ready:
                    width, height = self.rectifier.output_size
                    preview = cv2.warpPerspective(
                        frame, self.rectifier.matrix, (width, height), flags=cv2.INTER_LINEAR
                    )
                    self.current_rectified_preview = preview
                    if self.profile_roi is not None:
                        gray_preview = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
                        profile_edges = self.trace_profile_viterbi(gray_preview, self.profile_roi)
                        x0, y0, x1, y1 = self.profile_roi
                        cv2.rectangle(preview, (x0, y0), (x1, y1), (0, 220, 255), 1, cv2.LINE_AA)
                        if profile_edges is not None and len(profile_edges["center"]) > 1:
                            for edge_name, color in (("top", (255, 255, 0)), ("bottom", (0, 165, 255)), ("center", (0, 255, 0))):
                                points = profile_edges[edge_name].reshape(-1, 1, 2)
                                cv2.polylines(preview, [points], False, color, 2, cv2.LINE_AA)
                            cv2.putText(preview, "DUAL EDGE: top / bottom | CENTERLINE", (16, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                        else:
                            cv2.putText(preview, "ROI terlalu kecil untuk dilacak", (16, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2, cv2.LINE_AA)
                    elif self.selecting_profile_roi:
                        cv2.putText(preview, "ROI: cakup kedua tepi satu rib, hindari rib tetangga", (16, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2, cv2.LINE_AA)
                        if self.profile_roi_first_point is not None:
                            cv2.circle(preview, self.profile_roi_first_point, 6, (0, 220, 255), 2, cv2.LINE_AA)
                    cv2.putText(
                        preview, f"PERSPECTIVE PREVIEW | {width} x {height} px",
                        (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 120), 2, cv2.LINE_AA,
                    )
                else:
                    preview = np.zeros((360, 960, 3), dtype=np.uint8)
                    self.current_rectified_preview = preview
                    cv2.putText(
                        preview, "Belum dikalibrasi. Tekan P lalu klik empat sudut bidang conveyor.",
                        (24, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (210, 210, 210), 2, cv2.LINE_AA,
                    )
                cv2.imshow(RECTIFIED_WINDOW_NAME, preview)

                key = cv2.waitKey(1 if not self.paused else 30) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    break
                if key == 32:
                    self.paused = not self.paused
                elif key in (ord('p'), ord('P')):
                    self.rectifier.start_calibration()
                    self.profile_roi = None
                    self.selecting_profile_roi = False
                    self.profile_roi_first_point = None
                    print("[RECTIFY] Klik empat sudut pada jendela sumber: kiri-atas, kanan-atas, kanan-bawah, kiri-bawah.")
                elif key in (ord('r'), ord('R')):
                    if not self.rectifier.ready:
                        print("[ROI] Kalibrasi perspektif dulu dengan P.")
                    else:
                        self.profile_roi = None
                        self.profile_roi_first_point = None
                        self.selecting_profile_roi = True
                        print("[ROI] Pilih area sepanjang satu rib yang mencakup tepi atas dan bawah beserta sedikit margin; hindari rib tetangga/rel. Klik dua sudut di salah satu jendela.")
                elif key in (ord('u'), ord('U')):
                    self.rectifier.reset()
                    self.profile_roi = None
                    self.selecting_profile_roi = False
                    self.profile_roi_first_point = None
                    print("[RECTIFY] Kalibrasi direset.")
                elif key in (ord('k'), ord('K')):
                    self.save_perspective_calibration()
                elif key in (ord('s'), ord('S')):
                    stamp = int(time.time())
                    source_path = OUTPUT_DIR / f"perspective_source_{frame_idx:05d}_{stamp}.jpg"
                    cv2.imwrite(str(source_path), source_view)
                    if self.rectifier.ready:
                        preview_path = OUTPUT_DIR / f"perspective_preview_{frame_idx:05d}_{stamp}.jpg"
                        cv2.imwrite(str(preview_path), preview)
                        print(f"[SNAPSHOT] Sumber: {source_path}\n           Hasil transform: {preview_path}")
                    else:
                        print(f"[SNAPSHOT] Frame sumber: {source_path}")
                elif key in (ord('o'), ord('O')):
                    chosen = pick_video_file_dialog(Path(self.video_path).parent if self.video_path else DEFAULT_VIDEO_DIR)
                    if chosen and self.open_video(chosen):
                        self.paused = False
                        self.rectifier.reset()
                        self.profile_roi = None
                        self.selecting_profile_roi = False
                        self.profile_roi_first_point = None
                        calibration_checked = False
                        frame = None
        finally:
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()

    def save_perspective_calibration(self):
        if not self.rectifier.ready or self.current_inference_frame is None:
            print("[RECTIFY] Pilih empat sudut sampai preview muncul sebelum menyimpan dengan K.")
            return False
        height, width = self.current_inference_frame.shape[:2]
        payload = {
            "version": 1,
            "source_size": [width, height],
            "points": np.asarray(self.rectifier.points, dtype=float).tolist(),
            "output_size": list(self.rectifier.output_size),
        }
        try:
            PERSPECTIVE_CALIBRATION_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            print(f"[RECTIFY] Gagal menyimpan kalibrasi: {exc}")
            return False
        print(f"[RECTIFY] Kalibrasi disimpan: {PERSPECTIVE_CALIBRATION_FILE}")
        return True

    def load_perspective_calibration(self, frame):
        if not PERSPECTIVE_CALIBRATION_FILE.exists():
            return False
        try:
            payload = json.loads(PERSPECTIVE_CALIBRATION_FILE.read_text(encoding="utf-8"))
            expected_size = [frame.shape[1], frame.shape[0]]
            if payload.get("version") != 1 or payload.get("source_size") != expected_size:
                print(
                    f"[RECTIFY] File kalibrasi ada, tetapi resolusinya tidak cocok "
                    f"({payload.get('source_size')} != {expected_size}); tekan P untuk kalibrasi ulang."
                )
                return False
            points = payload.get("points", [])
            if len(points) != 4:
                raise ValueError("File harus berisi tepat empat titik.")
            self.rectifier.start_calibration()
            for x, y in points:
                if not self.rectifier.add_point(frame, int(round(x)), int(round(y))):
                    raise ValueError("Empat titik tersimpan tidak membentuk bidang valid.")
            if not self.rectifier.ready or list(self.rectifier.output_size) != payload.get("output_size"):
                raise ValueError("Ukuran hasil transform tidak sesuai dengan metadata.")
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            self.rectifier.reset()
            print(f"[RECTIFY] File kalibrasi tidak dapat digunakan: {exc}; tekan P untuk pilih ulang.")
            return False
        print(f"[RECTIFY] Kalibrasi otomatis dimuat dari {PERSPECTIVE_CALIBRATION_FILE}")
        return True

    def run(self):
        """Loop eksekusi utama pengujian video."""
        if self.cap is None or not self.cap.isOpened():
            print("[WARN] Sumber video belum ditentukan. Membuka dialog pilihan file...")
            chosen_video = pick_video_file_dialog(DEFAULT_VIDEO_DIR)
            if not chosen_video or not self.open_video(chosen_video):
                print("[INFO] Pengujian dibatalkan karena tidak ada video yang dipilih.")
                return

        if self.preview_only:
            self.run_perspective_preview()
            return

        self.setup_window()
        frame_idx = 0

        print("\n[START] Memulai pemutaran video dan kalkulasi benchmark...")
        print("         Tekan [SPASI] untuk jeda, [S] untuk screenshot, [Q] untuk selesai.\n")

        try:
            while True:
                if not self.paused:
                    ret, frame = self.cap.read()
                    if not ret:
                        print("[INFO] Video telah selesai diputar (mencapai akhir frame). Mengulang ke awal...")
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = self.cap.read()
                        if not ret:
                            break
                    frame_idx += 1
                else:
                    # Saat pause, tetap render frame terakhir
                    pass

                self.current_raw_frame = frame
                inference_frame, self.calibration_valid, self.calibration_score = self.stabilizer.stabilize(frame)
                self.current_inference_frame = inference_frame

                # Ukur Latensi Inferensi
                t_start = time.perf_counter()

                # Eksekusi YOLO predict
                results = self.model.predict(
                    source=inference_frame,
                    conf=self.conf_thres,
                    iou=self.iou_thres,
                    device=self.device,
                    verbose=False
                )

                if torch.cuda.is_available() and "cuda" in self.device:
                    torch.cuda.synchronize()

                t_end = time.perf_counter()
                total_cycle_ms = (t_end - t_start) * 1000.0

                # Ambil metrik latensi bawaan dari ultralytics (preprocess, inference, postprocess)
                speed_dict = results[0].speed if results and hasattr(results[0], 'speed') else {}
                pre_ms = speed_dict.get('preprocess', 1.0)
                inf_ms = speed_dict.get('inference', total_cycle_ms * 0.8)
                post_ms = speed_dict.get('postprocess', 1.0)

                # Update history
                self.history_preprocess.append(pre_ms)
                self.history_inference.append(inf_ms)
                self.history_postprocess.append(post_ms)
                self.history_total_latency.append(total_cycle_ms)

                # Hitung Rolling FPS
                inst_fps = 1000.0 / total_cycle_ms if total_cycle_ms > 0 else 0.0
                self.rolling_fps.append(inst_fps)
                current_avg_fps = float(np.mean(self.rolling_fps)) if self.rolling_fps else inst_fps

                # Simpan untuk statistik keseluruhan jika tidak dalam status pause
                if not self.paused:
                    self.total_processed_frames += 1
                    self.all_latencies_ms.append(total_cycle_ms)
                    self.all_inference_ms.append(inf_ms)

                # Pose keypoint menjadi satu-satunya sumber geometri untuk bending.
                annotated_frame = inference_frame.copy()
                result = results[0] if results else None
                boxes = result.boxes if result is not None else None
                pose = result.keypoints if result is not None else None
                current_frame_counts = {name: 0 for name in self.class_names.values()}
                robust_tracker_updated = False
                pose_xy = pose.xy.cpu().numpy() if pose is not None and pose.xy is not None else np.empty((0, 0, 2))
                pose_conf = pose.conf.cpu().numpy() if pose is not None and pose.conf is not None else None

                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    clss = boxes.cls.cpu().numpy().astype(int)
                    centers = [((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0) for box in xyxy]
                    zone_detection_count = sum(self.point_in_inspection_zone(center) for center in centers)

                    for index, (box, conf, cls_id) in enumerate(zip(xyxy, confs, clss)):
                        cls_name = self.class_names.get(cls_id, f"cls_{cls_id}")
                        current_frame_counts[cls_name] = current_frame_counts.get(cls_name, 0) + 1
                        if not self.paused:
                            self.detection_counts[cls_name] = self.detection_counts.get(cls_name, 0) + 1
                            self.detection_confs[cls_name].append(float(conf))

                        x1, y1, x2, y2 = map(int, box)
                        center = centers[index]
                        in_zone = self.point_in_inspection_zone(center)
                        raw_points = pose_xy[index] if index < len(pose_xy) else None
                        if raw_points is not None and pose_conf is not None and index < len(pose_conf):
                            raw_points = raw_points[pose_conf[index] >= 0.40]
                        analysis = RibProfileAnalyzer.wait()
                        rectified_points = None
                        if raw_points is not None and self.rectifier.ready and len(raw_points):
                            rectified_points = self.rectifier.map_to_rectified(raw_points)
                            analysis = self.profile_analyzer.analyze_points(rectified_points)
                        if self.stabilizer.ready and not self.calibration_valid:
                            analysis = RibProfileAnalyzer.wait(rectified_points)

                        status = analysis["status"]
                        if in_zone and zone_detection_count > 1:
                            self.robust_vote_tracker.reset()
                            status = "WAIT"
                        elif in_zone and status in ("OK", "NG") and not self.paused:
                            status = self.robust_vote_tracker.update(
                                center, analysis["score"], self.robust_threshold,
                                bbox=(x1, y1, x2, y2),
                            )
                            robust_tracker_updated = True
                        if in_zone and not self.paused:
                            self.robust_status_counts[status] = self.robust_status_counts.get(status, 0) + 1
                            if analysis["score"] is not None:
                                self.robust_scores.append(float(analysis["score"]))

                        display_analysis = analysis
                        if rectified_points is not None:
                            display_analysis = self.map_rectified_analysis_to_source(analysis)
                        self.draw_pose_analysis(annotated_frame, display_analysis, status)

                        color = CONTOUR_STATUS_COLORS.get(status, CLASS_COLORS.get(cls_name, DEFAULT_COLOR))
                        self.draw_corner_box(annotated_frame, (x1, y1), (x2, y2), color, thickness=2, corner_len=18)
                        if analysis["score"] is None:
                            detail = "P95/MAX:WAIT"
                        else:
                            detail = f"P95 {analysis['p95']*100:.2f}% MAX {analysis['peak']*100:.2f}%"
                        zone_tag = " ZONE" if in_zone else ""
                        label = f"{cls_name} {conf*100:.1f}% | RB-{status} {detail}{zone_tag}"
                        (lbl_w, lbl_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
                        badge_y1 = max(0, y1 - lbl_h - 10)
                        badge_overlay = annotated_frame.copy()
                        cv2.rectangle(badge_overlay, (x1, badge_y1), (x1 + lbl_w + 12, y1), (20, 20, 24), -1)
                        cv2.addWeighted(badge_overlay, 0.85, annotated_frame, 0.15, 0, annotated_frame)
                        cv2.rectangle(annotated_frame, (x1, badge_y1), (x1 + lbl_w + 12, y1), color, 1, cv2.LINE_AA)
                        cv2.putText(annotated_frame, label, (x1 + 6, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)

                if not self.paused and not robust_tracker_updated:
                    self.robust_vote_tracker.reset()

                # Gambar Panel HUD
                self.draw_hud(annotated_frame, current_avg_fps, pre_ms, inf_ms, post_ms, current_frame_counts)
                self.draw_calibration_overlay(annotated_frame)
                self.draw_rectification_overlay(annotated_frame)
                self.draw_inspection_zone(annotated_frame)

                # Tampilkan ke Layar
                cv2.imshow(WINDOW_NAME, annotated_frame)

                # Handle Keyboard Input
                key = cv2.waitKey(1 if not self.paused else 30) & 0xFF

                if key in [ord('q'), ord('Q'), 27]:  # Q atau ESC: Keluar
                    break
                elif key == 32:  # SPASI: Play / Pause
                    self.paused = not self.paused
                    print(f"[{'PAUSE' if self.paused else 'RESUME'}] Status video: {'Jeda' if self.paused else 'Berjalan'}")
                elif key in [ord('d'), ord('D'), 83]:  # Tombol D atau panah kanan: Maju 1 frame
                    if self.paused:
                        ret, frame = self.cap.read()
                        if ret:
                            frame_idx += 1
                elif key in [ord('a'), ord('A'), 81]:  # Tombol A atau panah kiri: Mundur 1 frame
                    if self.paused and self.cap.get(cv2.CAP_PROP_POS_FRAMES) > 2:
                        pos = max(0, int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 2)
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                        ret, frame = self.cap.read()
                        if ret:
                            frame_idx = pos + 1
                elif key in [ord('s'), ord('S')]:  # S: Simpan Snapshot
                    save_path = OUTPUT_DIR / f"snapshot_frame_{frame_idx:05d}_{int(time.time())}.jpg"
                    cv2.imwrite(str(save_path), annotated_frame)
                    print(f"[SNAPSHOT] Berhasil disimpan ke: {save_path}")
                elif key in [ord('o'), ord('O')]:  # O: Buka Video Baru
                    chosen = pick_video_file_dialog(Path(self.video_path).parent if self.video_path else DEFAULT_VIDEO_DIR)
                    if chosen:
                        self.open_video(chosen)
                elif key in [ord('t'), ord('T')]:  # T: Toggle HUD
                    self.show_hud = not self.show_hud
                elif key in [ord('c'), ord('C')]:
                    self.stabilizer.start_calibration()
                    print("[KALIBRASI] Klik empat detail rangka yang diam: kiri-atas, kanan-atas, kanan-bawah, kiri-bawah.")
                elif key in [ord('x'), ord('X')]:
                    self.stabilizer.reset()
                    self.calibration_valid = False
                    self.calibration_score = None
                    print("[KALIBRASI] Kalibrasi direset.")
                elif key in [ord('p'), ord('P')]:
                    self.rectifier.start_calibration()
                    self.robust_vote_tracker.reset()
                    print("[RECTIFY] Pilih empat sudut bidang conveyor (kiri-atas, kanan-atas, kanan-bawah, kiri-bawah).")
                elif key in [ord('u'), ord('U')]:
                    self.rectifier.reset()
                    self.robust_vote_tracker.reset()
                    print("[RECTIFY] Kalibrasi perspektif direset; status bending WAIT sampai dikalibrasi ulang.")
                elif key in [ord('z'), ord('Z')]:
                    self.selecting_zone = True
                    self.zone_start = None
                    self.inspection_zone = None
                    self.robust_vote_tracker.reset()
                    print("[ZONA] Klik dua sudut zona tetap. Buat zona setinggi satu rib dan lebar area inspeksi.")
                elif key in [ord('r'), ord('R')]:  # R: Reset counter
                    self.total_processed_frames = 0
                    self.all_latencies_ms.clear()
                    self.all_inference_ms.clear()
                    self.detection_counts = {name: 0 for name in self.class_names.values()}
                    self.detection_confs = {name: [] for name in self.class_names.values()}
                    self.robust_status_counts = {"OK": 0, "NG": 0, "WAIT": 0}
                    self.robust_scores.clear()
                    self.robust_vote_tracker.reset()
                    print("[RESET] Statistik pengujian telah di-reset ke nol.")

        finally:
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
            self.print_final_benchmark_report()

    def print_final_benchmark_report(self):
        """Mencetak ringkasan laporan hasil pengujian dan benchmark FPS ke terminal."""
        print("\n" + "=" * 76)
        print("        LAPORAN PENGUJIAN YOLO-POSE RIB & FPS")
        print("=" * 76)
        print(f" Model Target        : {Path(self.model_path).name}")
        print(f" Path Model          : {self.model_path}")
        print(f" File Video          : {Path(self.video_path).name if self.video_path else 'Webcam/N/A'}")
        print(f" Perangkat Komputasi : {self.device.upper()}")
        if "cuda" in self.device and torch.cuda.is_available():
            print(f" Nama GPU            : {torch.cuda.get_device_name(0)}")
        print(f" Ambang Conf / IoU   : Conf >= {self.conf_thres*100:.1f}%, IoU >= {self.iou_thres*100:.1f}%")
        print(
            f" Ambang pose P95/MAX  : {self.robust_threshold*100:.2f}% dari rentang keypoint"
        )
        print(f" Total Frame Diuji   : {self.total_processed_frames} frame")
        print("-" * 76)

        if self.all_latencies_ms:
            latencies = np.array(self.all_latencies_ms)
            inf_latencies = np.array(self.all_inference_ms)
            fps_array = 1000.0 / latencies

            avg_fps = np.mean(fps_array)
            med_fps = np.median(fps_array)
            min_fps = np.min(fps_array)
            max_fps = np.max(fps_array)

            avg_lat = np.mean(latencies)
            p95_lat = np.percentile(latencies, 95)
            avg_inf = np.mean(inf_latencies)

            print(" [A] PERFORMA KECEPATAN (FPS & LATENSI):")
            print(f"   • Rata-rata FPS (Throughput) : {avg_fps:6.2f} FPS")
            print(f"   • Median FPS                 : {med_fps:6.2f} FPS")
            print(f"   • Rentang FPS (Min - Max)    : {min_fps:6.2f} - {max_fps:6.2f} FPS")
            print(f"   • Total Latensi Rata-rata    : {avg_lat:6.2f} ms / frame")
            print(f"   • Pure Inference Rata-rata   : {avg_inf:6.2f} ms / frame")
            print(f"   • Latensi 95th Percentile    : {p95_lat:6.2f} ms")
        else:
            print(" [A] PERFORMA KECEPATAN: Tidak ada frame yang diproses.")

        print("-" * 76)
        print(" [B] STATISTIK DETEKSI KELAS:")
        for cls_id, cls_name in self.class_names.items():
            total_det = self.detection_counts.get(cls_name, 0)
            confs = self.detection_confs.get(cls_name, [])
            avg_conf = (np.mean(confs) * 100) if confs else 0.0
            max_conf = (np.max(confs) * 100) if confs else 0.0
            min_conf = (np.min(confs) * 100) if confs else 0.0
            det_per_frame = (total_det / self.total_processed_frames) if self.total_processed_frames > 0 else 0.0

            print(f"   • Kelas '{cls_name}' (ID: {cls_id}):")
            print(f"       - Total Objek Terdeteksi  : {total_det} kali (rata-rata {det_per_frame:.2f} per frame)")
            print(f"       - Rata-rata Confidence    : {avg_conf:5.1f}%")
            if confs:
                print(f"       - Rentang Confidence      : {min_conf:5.1f}% s/d {max_conf:5.1f}%")

        print("-" * 76)
        print(" [C] JUDGMENT BENDING DARI KEYPOINT YOLO-POSE:")
        print(f"   • RB-OK / RB-NG / RB-WAIT    : {self.robust_status_counts['OK']} / {self.robust_status_counts['NG']} / {self.robust_status_counts['WAIT']}")
        if self.robust_scores:
            print(f"   • Skor median / maksimum     : {np.median(self.robust_scores)*100:.2f}% / {np.max(self.robust_scores)*100:.2f}%")
        print("   • Skor adalah deviasi relatif terhadap panjang profil keypoint; bukan mm.")

        print("=" * 76)
        print(f" Snapshot frame deteksi disimpan di: {OUTPUT_DIR}")
        print("=" * 76 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Preview perspective transform conveyor, dengan deteksi YOLO sebagai opsi")
    parser.add_argument("--video", "-v", type=str, default=None,
                        help="Path ke file video pengujian (.mp4/.avi) atau indeks kamera (misal: 0)")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="Path ke checkpoint YOLO-Pose rib (.pt). Default: best_rib_pose.pt.")
    parser.add_argument("--conf", "-c", type=float, default=0.25,
                        help="Nilai batas ambang confidence (0.01 - 1.0, default: 0.25)")
    parser.add_argument("--iou", "-i", type=float, default=0.45,
                        help="Nilai batas ambang NMS / IoU (0.01 - 1.0, default: 0.45)")
    parser.add_argument("--device", "-d", type=str, default=None,
                        help="Device komputasi: 'cuda:0', 'cpu', dll.")
    parser.add_argument("--detect", action="store_true",
                        help="Aktifkan YOLO-Pose dan analisis bending; default hanya preview perspective transform.")
    parser.add_argument("--self-check", action="store_true",
                        help="Jalankan pemeriksaan sintetik scoring keypoint YOLO-Pose.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.self_check:
        run_pose_self_check()
        sys.exit(0)

    preview_only = not args.detect
    model_to_use = None
    if not preview_only:
        model_to_use = args.model or find_default_model()
        if not os.path.exists(model_to_use):
            print(f"[ERROR] File model tidak ditemukan di: {model_to_use}")
            print("[INFO] Silakan tentukan file model menggunakan argumen --model <path_model.pt>")
            sys.exit(1)

    # Inisialisasi tester
    tester = VideoYoloTester(
        model_path=model_to_use,
        video_path=args.video,
        conf_thres=args.conf,
        iou_thres=args.iou,
        device=args.device,
        preview_only=preview_only,
    )

    # Jika video tidak diberikan lewat CLI, periksa apakah ada default video yang tersedia
    if not args.video:
        if DEFAULT_VIDEO_PATH.exists():
            print(f"[INFO] Menggunakan video default: {DEFAULT_VIDEO_PATH.name}")
            tester.open_video(str(DEFAULT_VIDEO_PATH))
        else:
            # Cari video .mp4 apa pun di DEFAULT_VIDEO_DIR
            mp4_list = list(DEFAULT_VIDEO_DIR.glob("*.mp4")) if DEFAULT_VIDEO_DIR.exists() else []
            if mp4_list:
                print(f"[INFO] Menggunakan video sampel: {mp4_list[0].name}")
                tester.open_video(str(mp4_list[0]))

    # Jalankan proses pengujian
    tester.run()
