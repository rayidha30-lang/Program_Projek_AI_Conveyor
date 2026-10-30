"""
================================================================================
PROGRAM BENCHMARK & TESTER MODEL YOLO v26 (BOLT & PIN INSPECTION)
File: test_yolo_v26_video.py
Proyek: AI Slat & Rib Conveyor - Side Inspection (Bolt & Pin)
================================================================================
Fitur Utama:
  1. ANALISIS AKURASI REAL-TIME:
     - Deteksi objek diskrit: 'bolt_ok' dan 'pin_ok'.
     - Confidence score (%) dan bounding box presisi tinggi dengan corner accents.
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
    BASE_DIR / "Vision_Engine_Core" / "02_Top_Inspection_Slat_Rib" / "models" / "best_rib_segmentation.pt"
]

DEFAULT_VIDEO_DIR = Path(r"C:\Users\Victus\Pictures\Video & Foto Test Slat Rib\New\Reverse\Kamera posisi 3 dimmer 2 - Reverse.mp4")
OUTPUT_DIR = BASE_DIR / "output_test_v26"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Skema warna kelas (BGR)
CLASS_COLORS = {
    "bolt_ok": (0, 235, 120),    # Hijau Emerald Cerah
    "pin_ok": (255, 180, 0),     # Biru Cyan / Deep Sky
    "bolt_ng": (40, 40, 255),    # Merah Cerah
    "pin_ng": (0, 140, 255),     # Oranye Cerah
}
DEFAULT_COLOR = (200, 200, 200)

WINDOW_NAME = "YOLO v26 Tester - Bolt & Pin (Accuracy & FPS Benchmark)"


def find_default_model():
    """Mencari file model default yang tersedia di direktori proyek."""
    for cand in DEFAULT_MODEL_CANDIDATES:
        if cand.exists():
            return str(cand)
    return str(DEFAULT_MODEL_CANDIDATES[0])


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


class VideoYoloTester:
    def __init__(self, model_path: str, video_path: str = None, conf_thres: float = 0.25, iou_thres: float = 0.45, device: str = None):
        self.model_path = model_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.show_hud = True
        self.paused = False

        # Tentukan device komputasi (CUDA GPU jika tersedia, jika tidak CPU)
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"\n{'='*70}")
        print(f" Memuat Model YOLO: {Path(self.model_path).name}")
        print(f" Path: {self.model_path}")
        print(f" Device Hardware: {self.device.upper()}", end="")
        if "cuda" in self.device and torch.cuda.is_available():
            print(f" ({torch.cuda.get_device_name(0)})")
        else:
            print(" (Processor CPU)")
        print(f"{'='*70}\n")

        # Load Model YOLO via Ultralytics
        self.model = YOLO(self.model_path)
        self.class_names = self.model.names if hasattr(self.model, 'names') else {0: 'bolt_ok', 1: 'pin_ok'}
        print(f"[INFO] Kelas Deteksi Terdaftar: {self.class_names}")

        # Warmup GPU untuk stabilitas pengukuran latency
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

    def setup_window(self):
        """Membuat window OpenCV dengan slider Trackbar interaktif."""
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.createTrackbar("Conf (%)", WINDOW_NAME, self.slider_conf, 100, self._on_conf_change)
        cv2.createTrackbar("IoU/NMS (%)", WINDOW_NAME, self.slider_iou, 100, self._on_iou_change)

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
        title = "YOLO v26 INSPECTION BENCHMARK"
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
        cv2.putText(frame, "DETEKSI OBJEK (FRAME INI):", (x0 + 14, y0 + 124), cv2.FONT_HERSHEY_DUPLEX, 0.45, (160, 200, 255), 1, cv2.LINE_AA)

        y_det = y0 + 148
        for cls_name, count in current_counts.items():
            color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)
            cum_count = self.detection_counts.get(cls_name, 0)
            text = f"• {cls_name:<10}: {count:2d} (Total: {cum_count})"
            cv2.putText(frame, text, (x0 + 18, y_det), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)
            y_det += 22

        # Status threshold & kontrol
        y_bottom = y0 + hud_h - 12
        info_sub = f"Conf: {int(self.conf_thres*100)}% | IoU: {int(self.iou_thres*100)}% | Device: {self.device.upper()}"
        cv2.putText(frame, info_sub, (x0 + 14, y_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 140, 150), 1, cv2.LINE_AA)

        # Petunjuk Keyboard di pojok kanan bawah
        help_bg = frame.copy()
        cv2.rectangle(help_bg, (w - 380, h - 38), (w - 16, h - 14), (20, 20, 24), -1)
        cv2.addWeighted(help_bg, 0.75, frame, 0.25, 0, frame)
        cv2.putText(frame, "[SPASI] Pause | [S] Snapshot | [O] Video | [Q] Selesai",
                    (w - 370, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

    def run(self):
        """Loop eksekusi utama pengujian video."""
        if self.cap is None or not self.cap.isOpened():
            print("[WARN] Sumber video belum ditentukan. Membuka dialog pilihan file...")
            chosen_video = pick_video_file_dialog(DEFAULT_VIDEO_DIR)
            if not chosen_video or not self.open_video(chosen_video):
                print("[INFO] Pengujian dibatalkan karena tidak ada video yang dipilih.")
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

                # Ukur Latensi Inferensi
                t_start = time.perf_counter()

                # Eksekusi YOLO predict
                results = self.model.predict(
                    source=frame,
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

                # Gambar Bounding Box Hasil Deteksi
                annotated_frame = frame.copy()
                boxes = results[0].boxes if len(results) > 0 else None
                current_frame_counts = {name: 0 for name in self.class_names.values()}

                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    clss = boxes.cls.cpu().numpy().astype(int)

                    for box, conf, cls_id in zip(xyxy, confs, clss):
                        cls_name = self.class_names.get(cls_id, f"cls_{cls_id}")
                        current_frame_counts[cls_name] = current_frame_counts.get(cls_name, 0) + 1

                        if not self.paused:
                            self.detection_counts[cls_name] = self.detection_counts.get(cls_name, 0) + 1
                            self.detection_confs[cls_name].append(conf)

                        x1, y1, x2, y2 = map(int, box)
                        color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)

                        # Bounding Box dengan Corner Accent
                        self.draw_corner_box(annotated_frame, (x1, y1), (x2, y2), color, thickness=2, corner_len=18)

                        # Label Badge
                        label = f"{cls_name} {conf * 100:.1f}%"
                        (lbl_w, lbl_h), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
                        badge_y1 = max(0, y1 - lbl_h - 10)
                        badge_y2 = y1

                        # Badge Background
                        badge_overlay = annotated_frame.copy()
                        cv2.rectangle(badge_overlay, (x1, badge_y1), (x1 + lbl_w + 12, badge_y2), (20, 20, 24), -1)
                        cv2.addWeighted(badge_overlay, 0.85, annotated_frame, 0.15, 0, annotated_frame)
                        cv2.rectangle(annotated_frame, (x1, badge_y1), (x1 + lbl_w + 12, badge_y2), color, 1, lineType=cv2.LINE_AA)

                        # Label Text
                        cv2.putText(annotated_frame, label, (x1 + 6, badge_y2 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, lineType=cv2.LINE_AA)

                # Gambar Panel HUD
                self.draw_hud(annotated_frame, current_avg_fps, pre_ms, inf_ms, post_ms, current_frame_counts)

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
                elif key in [ord('o'), ord('O'), ord('v'), ord('V')]:  # O / V: Buka Video Baru
                    chosen = pick_video_file_dialog(Path(self.video_path).parent if self.video_path else DEFAULT_VIDEO_DIR)
                    if chosen:
                        self.open_video(chosen)
                elif key in [ord('t'), ord('T')]:  # T: Toggle HUD
                    self.show_hud = not self.show_hud
                elif key in [ord('r'), ord('R')]:  # R: Reset counter
                    self.total_processed_frames = 0
                    self.all_latencies_ms.clear()
                    self.all_inference_ms.clear()
                    self.detection_counts = {name: 0 for name in self.class_names.values()}
                    self.detection_confs = {name: [] for name in self.class_names.values()}
                    print("[RESET] Statistik pengujian telah di-reset ke nol.")

        finally:
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
            self.print_final_benchmark_report()

    def print_final_benchmark_report(self):
        """Mencetak ringkasan laporan hasil pengujian dan benchmark FPS ke terminal."""
        print("\n" + "=" * 76)
        print("        LAPORAN PENGUJIAN AKURASI & FPS MODEL YOLO v26 (BOLT & PIN)        ")
        print("=" * 76)
        print(f" Model Target        : {Path(self.model_path).name}")
        print(f" Path Model          : {self.model_path}")
        print(f" File Video          : {Path(self.video_path).name if self.video_path else 'Webcam/N/A'}")
        print(f" Perangkat Komputasi : {self.device.upper()}")
        if "cuda" in self.device and torch.cuda.is_available():
            print(f" Nama GPU            : {torch.cuda.get_device_name(0)}")
        print(f" Ambang Conf / IoU   : Conf >= {self.conf_thres*100:.1f}%, IoU >= {self.iou_thres*100:.1f}%")
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
        print(" [B] STATISTIK DETEKSI & AKURASI KELAS:")
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

        print("=" * 76)
        print(f" Snapshot frame deteksi disimpan di: {OUTPUT_DIR}")
        print("=" * 76 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Program Pengujian Akurasi & FPS YOLO v26 (Bolt & Pin Video Inspector)")
    parser.add_argument("--video", "-v", type=str, default=None,
                        help="Path ke file video pengujian (.mp4/.avi) atau indeks kamera (misal: 0)")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="Path ke file model YOLO checkpoint (.pt). Jika kosong, mencari otomatis model v26.")
    parser.add_argument("--conf", "-c", type=float, default=0.25,
                        help="Nilai batas ambang confidence (0.01 - 1.0, default: 0.25)")
    parser.add_argument("--iou", "-i", type=float, default=0.45,
                        help="Nilai batas ambang NMS / IoU (0.01 - 1.0, default: 0.45)")
    parser.add_argument("--device", "-d", type=str, default=None,
                        help="Device komputasi: 'cuda:0', 'cpu', dll.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Pilih model
    model_to_use = args.model
    if not model_to_use:
        model_to_use = find_default_model()

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
        device=args.device
    )

    # Jika video tidak diberikan lewat CLI, periksa apakah ada default video yang tersedia
    if not args.video:
        default_vid_path = DEFAULT_VIDEO_DIR / "Result Yolo(Bolt&Pin) + PatchCore (Crack) - 290926.mp4"
        if default_vid_path.exists():
            print(f"[INFO] Menggunakan video default dari folder uji: {default_vid_path.name}")
            tester.open_video(str(default_vid_path))
        else:
            # Cari video .mp4 apa pun di DEFAULT_VIDEO_DIR
            mp4_list = list(DEFAULT_VIDEO_DIR.glob("*.mp4")) if DEFAULT_VIDEO_DIR.exists() else []
            if mp4_list:
                print(f"[INFO] Menggunakan video sampel: {mp4_list[0].name}")
                tester.open_video(str(mp4_list[0]))

    # Jalankan proses pengujian
    tester.run()
