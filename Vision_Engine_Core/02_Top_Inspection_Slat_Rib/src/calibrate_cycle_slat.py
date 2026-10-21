"""
================================================================================
MODUL KALIBRASI 1 SIKLUS PENUH KONVEYOR (SLAT RIB AUTO-CALIBRATION WIZARD)
================================================================================
Dirancang khusus untuk lingkungan overhaul konveyor dengan bracket kamera
bongkar-pasang (removable).

Fitur Utama:
1. Deteksi Otomatis Penanda Merah (Red Marker - Plat #01) via Filter Warna HSV.
2. Penyelarasan Posisi & Sudut Bracket (Auto-Deskew & Auto-Centering Gerbang ROI).
3. Profiling Pencahayaan Workshop Otomatis (Auto-Lighting & Filter Canny/Sobel Tuning).
4. Auto-Benchmark Panjang Celah Acuan (MLL Locking dari Median Plat Normal).
5. Pemetaan Total Plat Fisik dalam 1 Siklus Rantai Konveyor (Slat Fingerprinting).
6. Penyimpanan Otomatis Hasil Kalibrasi ke File Konfigurasi JSON.

Tombol Kontrol:
  - [SPACE]     : Jeda / Lanjut Pemindaian (Pause / Play)
  - [M]         : Trigger Manual Penanda Merah (Simulasi / Jika tanpa cat merah)
  - [C]         : Kunci & Selesaikan Kalibrasi Seketika (Finish Calibration)
  - [S]         : Simpan Profil Kalibrasi ke File JSON Config
  - [R]         : Reset / Ulangi Kalibrasi dari Awal
  - [O] / [F]   : Pilih File Video Baru
  - [Q] / [ESC] : Keluar
================================================================================
"""

import os
import sys
import json
import math
import time
import argparse
from datetime import datetime
from collections import deque

import cv2
import numpy as np

# Tambahkan src ke sys.path untuk impor dependensi lokal
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from hough_stitching_inspector import (
    HoughStitchingInspector,
    get_adaptive_slat_polygon,
    order_quad_points,
    select_video_file
)


# ==============================================================================
# DETEKTOR PENANDA MERAH (RED REFERENCE MARKER DETECTOR)
# ==============================================================================
class RedMarkerDetector:
    """
    Mendeteksi penanda visual berwarna merah cerah (cat/stiker penanda plat #01)
    pada area plat konveyor menggunakan segmentasi warna HSV ganda.
    """
    def __init__(self, min_area_px=800, max_area_px=60000):
        self.min_area = min_area_px
        self.max_area = max_area_px
        # Ambang batas warna merah dalam ruang HSV (Merah melingkar di H: 0-10 & 170-180)
        self.lower_red1 = np.array([0, 90, 70], dtype=np.uint8)
        self.upper_red1 = np.array([12, 255, 255], dtype=np.uint8)
        self.lower_red2 = np.array([168, 90, 70], dtype=np.uint8)
        self.upper_red2 = np.array([180, 255, 255], dtype=np.uint8)
        self.last_detection_frame = -999
        self.debounce_frames = 25  # Jeda frame minimum agar tidak double trigger

    def detect(self, frame_bgr, search_roi_norm=None, current_frame_idx=0):
        """
        Mendeteksi kehadiran penanda merah pada frame.
        :param frame_bgr: Citra frame asli.
        :param search_roi_norm: [ymin, ymax] normalisasi area pencarian vertikal (default: 0.3 s.d. 0.8)
        :return: (is_detected, confidence_score, bbox_px, debug_mask)
        """
        h_f, w_f = frame_bgr.shape[:2]
        
        # Batasi area pencarian vertikal untuk efisiensi & hindari latar belakang workshop
        if search_roi_norm is not None:
            y1 = max(0, int(search_roi_norm[0] * h_f))
            y2 = min(h_f - 1, int(search_roi_norm[1] * h_f))
        else:
            y1 = int(0.35 * h_f)
            y2 = int(0.80 * h_f)

        crop = frame_bgr[y1:y2, :]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # Masking warna merah dua sisi (Hue 0-12 dan 168-180)
        m1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        m2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        red_mask = cv2.bitwise_or(m1, m2)

        # Operasi morfologi untuk membersihkan bintik karat non-marker
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        red_clean = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        red_clean = cv2.morphologyEx(red_clean, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(red_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_cnt = None
        max_area = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_area <= area <= self.max_area:
                if area > max_area:
                    max_area = area
                    best_cnt = cnt

        is_detected = False
        bbox_global = None

        if best_cnt is not None:
            bx, by, bw, bh = cv2.boundingRect(best_cnt)
            bbox_global = (bx, y1 + by, bx + bw, y1 + by + bh)
            # Cek debounce antar-marker
            if current_frame_idx - self.last_detection_frame >= self.debounce_frames:
                is_detected = True
                self.last_detection_frame = current_frame_idx

        return is_detected, max_area, bbox_global, red_clean


# ==============================================================================
# ENGINE AUTO-KALIBRASI 1 SIKLUS KONVEYOR
# ==============================================================================
class SlatCycleCalibrator:
    """
    Mengelola siklus kalibrasi konveyor:
    - Menunggu penanda merah Plat #01.
    - Merekam seluruh plat selama 1 putaran tertutup.
    - Menghitung koreksi tilt bracket, horizon center beam, dan target MLL.
    """
    STATE_WAIT_MARKER = "WAIT_FOR_MARKER"
    STATE_CALIBRATING = "CALIBRATING_CYCLE"
    STATE_COMPLETE    = "CYCLE_COMPLETE"

    def __init__(self, config_path=None, fps=30.0, is_video_mode=False):
        self.config_path = config_path
        self.fps = fps if (fps is not None and fps > 1.0) else 30.0
        self.is_video_mode = bool(is_video_mode)
        self.state = self.STATE_WAIT_MARKER

        self.marker_detector = RedMarkerDetector()
        self.inspector = HoughStitchingInspector(strict_mode=True)

        self.current_frame_idx = 0
        self.cycle_start_frame = 0
        self.cycle_end_frame = 0
        self.cycle_start_time = 0.0
        self.cycle_duration_sec = 0.0

        # Penampung data metrik selama 1 siklus
        self.slat_records = []
        self.v_mean_history = []
        self.measured_slat_count = 0

        # Hasil Analisis Kalibrasi
        self.calib_results = {}
        self.is_profile_saved = False

    def reset(self):
        """Mereset seluruh state kalibrasi dari awal."""
        self.state = self.STATE_WAIT_MARKER
        self.cycle_start_frame = 0
        self.cycle_end_frame = 0
        self.cycle_start_time = 0.0
        self.cycle_duration_sec = 0.0
        self.slat_records.clear()
        self.v_mean_history.clear()
        self.measured_slat_count = 0
        self.calib_results.clear()
        self.is_profile_saved = False
        self.inspector.reset_counters()

    def trigger_cycle_start(self, frame_idx):
        """Pemicuan awal siklus kalibrasi (Plat #01 terdeteksi atau awal file video)."""
        self.state = self.STATE_CALIBRATING
        self.cycle_start_frame = frame_idx
        self.cycle_start_time = time.time()
        self.slat_records.clear()
        self.v_mean_history.clear()
        self.measured_slat_count = 0
        self.inspector.reset_counters()
        if self.is_video_mode:
            print(f"\n[CALIBRATION START] Mode File Video: Merekam profil seluruh plat dari frame #{frame_idx} hingga akhir video...")
        else:
            print(f"\n[CALIBRATION START] Penanda Merah (Plat #01) Terdeteksi di Frame #{frame_idx}!")
            print("[*] Konveyor sedang merekam 1 siklus putaran penuh...")

    def trigger_cycle_finish(self, frame_idx):
        """Pemicuan akhir siklus kalibrasi saat penanda merah kembali terdeteksi atau video habis."""
        self.state = self.STATE_COMPLETE
        self.cycle_end_frame = frame_idx
        cycle_frames = max(1, self.cycle_end_frame - self.cycle_start_frame)
        self.cycle_duration_sec = cycle_frames / self.fps
        self.compute_calibration_profile()
        print(f"\n[CALIBRATION FINISH] Siklus Selesai di Frame #{frame_idx}! Durasi: {self.cycle_duration_sec:.1f} detik ({cycle_frames} frame).")
        print(f"[OK] Total Plat Terhitung: {self.measured_slat_count} plat.")
        print(f"[OK] Kemiringan Bracket Rata-Rata: {self.calib_results.get('bracket_tilt_deg', 0.0):+.2f} deg.")
        print(f"[OK] Target Ambang Batas MLL Terkunci: {self.calib_results.get('calibrated_mll_px', 700):.0f} px.")

    def process_frame(self, frame_bgr, polygon_pts):
        """
        Memproses 1 frame konveyor dalam alur kalibrasi.
        """
        self.current_frame_idx += 1
        h_f, w_f = frame_bgr.shape[:2]

        # 1. Jalankan inspeksi unwarped dan horizon tracking
        insp_res = self.inspector.inspect_frame(frame_bgr, polygon_pts)
        self.v_mean_history.append(insp_res.get("v_mean", 110.0))

        # 2. Deteksi Penanda Merah (Red Marker)
        marker_detected, marker_area, marker_bbox, red_mask = self.marker_detector.detect(
            frame_bgr, search_roi_norm=[0.35, 0.85], current_frame_idx=self.current_frame_idx
        )

        # 3. Logika State Machine Kalibrasi
        if self.state == self.STATE_WAIT_MARKER:
            frames_in_cycle = 0  # Siklus belum dimulai
            if self.is_video_mode:
                self.trigger_cycle_start(self.current_frame_idx)
                frames_in_cycle = 1
            elif marker_detected:
                self.trigger_cycle_start(self.current_frame_idx)
                frames_in_cycle = 1

        elif self.state == self.STATE_CALIBRATING:
            frames_in_cycle = self.current_frame_idx - self.cycle_start_frame
            # Rekam data setiap plat yang baru memasuki gerbang inspeksi
            if insp_res["is_new_slat"]:
                self.measured_slat_count += 1
                record = {
                    "slat_id": self.measured_slat_count,
                    "frame": self.current_frame_idx,
                    "y_seam": insp_res["y_seam"],
                    "l_seam": insp_res["l_seam"],
                    "peak_l": insp_res["peak_l"],
                    "angle": insp_res.get("seam_res", {}).get("theta_ref", 0.0),
                    "energy": insp_res["seam_energy"],
                    "v_mean": insp_res.get("v_mean", 110.0),
                    "pv": insp_res.get("pv_flatness", 6.0),
                    "sagitta": insp_res.get("sagitta", 2.5)
                }
                self.slat_records.append(record)

            # Jika penanda merah muncul kembali setelah putaran (minimal jeda 40 frame antar-marker)
            # PERHATIAN: HANYA aktif jika BUKAN mode file video!
            # Pada mode file video, kalibrasi DILARANG berhenti di tengah jalan karena bintik merah/karat.
            # Mode file video HANYA selesai saat seluruh frame video habis (EOF) atau jika ditekan [C].
            if not self.is_video_mode:
                if marker_detected and frames_in_cycle >= 45 and self.measured_slat_count >= 3:
                    self.trigger_cycle_finish(self.current_frame_idx)
        else:
            frames_in_cycle = max(0, self.cycle_end_frame - self.cycle_start_frame)

        return {
            "state": self.state,
            "marker_detected": marker_detected,
            "marker_bbox": marker_bbox,
            "marker_area": marker_area,
            "insp_res": insp_res,
            "frames_in_cycle": frames_in_cycle,
            "measured_slat_count": self.measured_slat_count,
            "slat_records": self.slat_records,
            "calib_results": self.calib_results
        }

    def compute_calibration_profile(self):
        """
        Menghitung profil matematis kalibrasi optimal berdasarkan seluruh plat
        yang melintas selama 1 siklus konveyor.
        """
        if not self.slat_records:
            return

        angles = [r["angle"] for r in self.slat_records if abs(r["angle"]) <= 12.0]
        y_seams = [r["y_seam"] for r in self.slat_records if 35 <= r["y_seam"] <= 105]
        peak_ls = [r["peak_l"] for r in self.slat_records if r["peak_l"] >= 400.0]

        mean_tilt = float(np.mean(angles)) if angles else 0.0
        median_y_seam = float(np.median(y_seams)) if y_seams else 70.0
        median_length = float(np.median(peak_ls)) if peak_ls else 950.0
        ambient_v_mean = float(np.mean(self.v_mean_history)) if self.v_mean_history else 115.0

        # Ambang batas MLL otomatis: 72% dari panjang median plat normal
        calib_mll_ratio = max(0.60, min(0.85, 0.72 * (median_length / 1000.0)))
        calib_mll_px = 1000.0 * calib_mll_ratio

        # Rekomendasi ambang batas kecerahan (v_thresh)
        rec_v_thresh = float(np.clip(ambient_v_mean, 85.0, 145.0))

        # Rekomendasi toleransi geometris kelurusan & kelengkungan (Metrologi PV & Sagitta)
        pvs = [r["pv"] for r in self.slat_records if "pv" in r and r["pv"] < 30.0]
        sags = [r["sagitta"] for r in self.slat_records if "sagitta" in r and r["sagitta"] < 20.0]
        rec_pv = float(np.percentile(pvs, 95) * 1.4) if pvs else 14.0
        rec_sag = float(np.percentile(sags, 95) * 1.5) if sags else 7.0
        rec_pv = float(np.clip(rec_pv, 10.0, 20.0))
        rec_sag = float(np.clip(rec_sag, 4.5, 12.0))

        self.calib_results = {
            "calibrated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_slats_in_loop": self.measured_slat_count,
            "bracket_tilt_deg": round(mean_tilt, 2),
            "median_y_seam_px": round(median_y_seam, 1),
            "median_seam_length_px": round(median_length, 1),
            "calibrated_mll_px": round(calib_mll_px, 1),
            "calibrated_mll_ratio": round(calib_mll_ratio, 3),
            "ambient_v_mean": round(ambient_v_mean, 1),
            "recommended_v_thresh": round(rec_v_thresh, 1),
            "recommended_max_pv": round(rec_pv, 1),
            "recommended_max_sagitta": round(rec_sag, 1),
            "cycle_duration_sec": round(self.cycle_duration_sec, 1)
        }

    def save_calibration_to_config(self, target_config_path=None):
        """
        Menyimpan hasil profil kalibrasi ke dalam file konfigurasi JSON sistem.
        """
        if target_config_path is None:
            target_config_path = self.config_path

        if not target_config_path:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            target_config_path = os.path.join(base_dir, "configs", "top_inspection_config.json")

        os.makedirs(os.path.dirname(os.path.abspath(target_config_path)), exist_ok=True)

        existing_cfg = {}
        if os.path.exists(target_config_path):
            try:
                with open(target_config_path, "r") as f:
                    existing_cfg = json.load(f)
            except Exception:
                pass

        # Update metadata kalibrasi
        existing_cfg["calibration_profile"] = self.calib_results
        existing_cfg["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Update parameter ambang batas yang aktif
        if "thresholds" not in existing_cfg:
            existing_cfg["thresholds"] = {}

        existing_cfg["thresholds"]["calibrated_mll_ratio"] = self.calib_results.get("calibrated_mll_ratio", 0.70)
        existing_cfg["thresholds"]["calibrated_v_thresh"] = self.calib_results.get("recommended_v_thresh", 116.0)
        existing_cfg["thresholds"]["total_slats_in_loop"] = self.calib_results.get("total_slats_in_loop", self.measured_slat_count)
        existing_cfg["thresholds"]["bracket_tilt_deg"] = self.calib_results.get("bracket_tilt_deg", 0.0)
        existing_cfg["thresholds"]["max_flatness_pv"] = self.calib_results.get("recommended_max_pv", existing_cfg["thresholds"].get("max_flatness_pv", 14.0))
        existing_cfg["thresholds"]["max_sagitta"] = self.calib_results.get("recommended_max_sagitta", existing_cfg["thresholds"].get("max_sagitta", 7.0))
        existing_cfg["thresholds"]["max_center_void"] = existing_cfg["thresholds"].get("max_center_void", 3500)

        with open(target_config_path, "w") as f:
            json.dump(existing_cfg, f, indent=2)

        self.is_profile_saved = True
        print(f"\n[SUCCESS] Profil kalibrasi 1 siklus berhasil disimpan ke: {target_config_path}")
        return True


# ==============================================================================
# RENDER TAMPILAN DASHBOARD KALIBRASI INTERAKTIF (HIGH CONTRAST HUD)
# ==============================================================================
def render_calibration_hud(frame_bgr, calib_data, polygon_pts):
    """
    Merender antarmuka visual kalibrasi lengkap dengan panduan status,
    indikator radar penanda merah, dan ringkasan metrik 1 siklus.
    """
    h_f, w_f = frame_bgr.shape[:2]
    canvas_cam = frame_bgr.copy()

    state = calib_data["state"]
    marker_detected = calib_data["marker_detected"]
    marker_bbox = calib_data["marker_bbox"]
    insp_res = calib_data["insp_res"]
    calib_res = calib_data["calib_results"]
    measured_count = calib_data["measured_slat_count"]

    # 1. Gambar Polygon Gerbang Inspeksi
    poly_pts_int = polygon_pts.astype(np.int32)
    if state == SlatCycleCalibrator.STATE_COMPLETE:
        poly_col = (0, 230, 0)      # Hijau: Selesai
    elif state == SlatCycleCalibrator.STATE_CALIBRATING:
        poly_col = (0, 220, 255)    # Cyan: Sedang Merekam
    else:
        poly_col = (0, 165, 255)    # Oren: Menunggu Penanda

    cv2.polylines(canvas_cam, [poly_pts_int], isClosed=True, color=poly_col, thickness=3, lineType=cv2.LINE_AA)
    cv2.putText(canvas_cam, f"GERBANG INSPEKSI SLAT ({w_f}x{h_f})", 
                (poly_pts_int[0][0], max(25, poly_pts_int[0][1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, poly_col, 2, cv2.LINE_AA)

    # 2. Gambar Highlight Penanda Merah jika terdeteksi
    if marker_bbox is not None:
        bx1, by1, bx2, by2 = marker_bbox
        cv2.rectangle(canvas_cam, (bx1, by1), (bx2, by2), (0, 0, 255), 3)
        cv2.putText(canvas_cam, "PENANDA MERAH [PLAT #01]", (bx1, max(20, by1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 255), 2, cv2.LINE_AA)

    # Resize tampilan kamera ke standar 1000 x 500 px
    cam_resized = cv2.resize(canvas_cam, (1000, 500))

    is_video_file = calib_data.get("is_video_file", False)
    is_paused = calib_data.get("is_paused", False)

    # --- TOP STATUS BANNER (Tinggi 42 px) ---
    banner = np.zeros((42, 1000, 3), dtype=np.uint8)
    if state == SlatCycleCalibrator.STATE_WAIT_MARKER:
        banner[:] = (20, 35, 48)
        b_text = "FASE 1: MENUNGGU PENANDA MERAH [PLAT #01] -> Putar konveyor sampai stiker/tanda merah melintas..."
        b_col = (0, 180, 255)
    elif state == SlatCycleCalibrator.STATE_CALIBRATING:
        banner[:] = (45, 35, 15)
        frames_c = calib_data["frames_in_cycle"]
        if is_video_file:
            b_text = f"FASE 2: KALIBRASI FILE VIDEO -> Plat Terhitung: #{measured_count:02d} ({frames_c} frame) | Merekam hingga akhir video..."
        else:
            b_text = f"FASE 2: SEDANG KALIBRASI SIKLUS -> Plat Terhitung: #{measured_count:02d} ({frames_c} frame) | Tunggu tanda merah kembali..."
        b_col = (0, 230, 255)
    else:
        banner[:] = (20, 48, 25)
        if is_video_file:
            b_text = f"FASE 3: KALIBRASI SELESAI (AKHIR VIDEO TERCAPAI) -> Total: {calib_res.get('total_slats_in_loop', measured_count)} Plat | Tekan [S] Simpan Profil"
        else:
            b_text = f"FASE 3: SIKLUS SELESAI & TERKUNCI! -> Total: {calib_res.get('total_slats_in_loop', measured_count)} Plat | Tekan [S] untuk Simpan Profil"
        b_col = (0, 255, 120)

    if is_paused:
        b_text += " [JEDA / PAUSED]"

    cv2.putText(banner, b_text, (18, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.50, b_col, 2, cv2.LINE_AA)

    # --- BOTTOM DASHBOARD HUD (Tinggi 240 px) ---
    hud = np.zeros((240, 1000, 3), dtype=np.uint8)
    hud[:] = (18, 18, 24)

    # KARTU 1: STATUS SIKLUS & DETEKTOR PENANDA (X: 12 - 290)
    cv2.rectangle(hud, (12, 10), (290, 230), (28, 28, 36), -1)
    cv2.rectangle(hud, (12, 10), (290, 230), b_col, 2)
    cv2.putText(hud, "STATUS SIKLUS KALIBRASI", (25, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 180, 180), 1, cv2.LINE_AA)

    status_tag = "MENUNGGU" if state == SlatCycleCalibrator.STATE_WAIT_MARKER else ("MEREKAM" if state == SlatCycleCalibrator.STATE_CALIBRATING else "SELESAI")
    if is_video_file and state == SlatCycleCalibrator.STATE_CALIBRATING:
        status_tag = "MEREKAM VIDEO"
    if is_paused:
        status_tag += " (JEDA)"
    cv2.putText(hud, status_tag, (25, 78), cv2.FONT_HERSHEY_DUPLEX, 1.15 if not is_video_file else 0.95, b_col, 2, cv2.LINE_AA)

    # Indikator Radar Marker
    m_color = (0, 0, 255) if marker_detected else (80, 80, 80)
    cv2.circle(hud, (35, 115), 9, m_color, -1)
    cv2.circle(hud, (35, 115), 11, (160, 160, 160), 1)
    if is_video_file:
        m_lbl = "Mode: Full Video Recording"
    else:
        m_lbl = "MARKER TERDETEKSI!" if marker_detected else "Mencari Marker Merah..."
    cv2.putText(hud, m_lbl, (55, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (230, 230, 230), 1, cv2.LINE_AA)

    cv2.putText(hud, f"Plat Terdata : #{measured_count:02d} plat", (25, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    
    if state == SlatCycleCalibrator.STATE_WAIT_MARKER:
        dur_str = "Menunggu Plat #01"
    else:
        dur_str = f"{calib_data.get('frames_in_cycle', 0)} frame"
    if is_paused and state != SlatCycleCalibrator.STATE_COMPLETE:
        dur_str += " [DIJEDA]"
    elif state == SlatCycleCalibrator.STATE_COMPLETE:
        dur_str += " [SELESAI]"

    cv2.putText(hud, f"Durasi Scan  : {dur_str}", (25, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(hud, f"Tersimpan    : {'YA [SUKSES]' if calib_data.get('is_saved', False) else 'BELUM (Tekan S)'}",
                (25, 208), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 255, 120) if calib_data.get('is_saved', False) else (140, 140, 140), 1, cv2.LINE_AA)

    # KARTU 2: PARAMETER HASIL ANALISIS SIKLUS (X: 304 - 700)
    cv2.rectangle(hud, (304, 10), (700, 230), (28, 28, 36), -1)
    cv2.rectangle(hud, (304, 10), (700, 230), (55, 55, 68), 1)
    cv2.putText(hud, "HASIL KALIBRASI GEOMETRI & LINGKUNGAN", (318, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 220, 255), 1, cv2.LINE_AA)

    tilt_val = calib_res.get("bracket_tilt_deg", insp_res.get("seam_res", {}).get("theta_ref", 0.0))
    mll_val = calib_res.get("calibrated_mll_px", 700.0)
    mll_pct = calib_res.get("calibrated_mll_ratio", 0.70) * 100.0
    med_len = calib_res.get("median_seam_length_px", insp_res.get("l_seam", 0.0))
    v_mean_val = calib_res.get("ambient_v_mean", insp_res.get("v_mean", 110.0))

    cv2.putText(hud, f"1. Kemiringan Bracket (Tilt) : {tilt_val:+.2f} deg", (318, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(hud, f"2. Median Panjang Celah      : {med_len:.1f} px", (318, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(hud, f"3. Target Ambang Batas MLL   : {mll_val:.0f} px ({mll_pct:.0f}%)", (318, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 120), 1, cv2.LINE_AA)
    cv2.putText(hud, f"4. Profil Kecerahan Ruangan  : V={v_mean_val:.1f} (Mode Adaptif)", (318, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(hud, f"5. Total Plat 1 Loop Konveyor: {calib_res.get('total_slats_in_loop', measured_count)} plat fisik",
                (318, 188), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 255), 2, cv2.LINE_AA)

    # KARTU 3: LOG REAL-TIME PLAT & PETUNJUK OPERASI (X: 714 - 988)
    cv2.rectangle(hud, (714, 10), (988, 230), (28, 28, 36), -1)
    cv2.rectangle(hud, (714, 10), (988, 230), (55, 55, 68), 1)
    cv2.putText(hud, "PANDUAN OPERASI OPERATOR", (728, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1, cv2.LINE_AA)

    if is_video_file:
        cv2.putText(hud, "[AUTO]: Selesai saat Video Habis", (728, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(hud, "[C] : Selesaikan Sekarang", (728, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
    else:
        cv2.putText(hud, "[M] : Trigger Marker Manual", (728, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(hud, "[C] : Selesaikan Siklus Skrg", (728, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(hud, "[S] : Simpan Profil ke JSON", (728, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 120), 1, cv2.LINE_AA)
    cv2.putText(hud, "[R] : Putar Ulang dari Awal", (728, 144), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 180, 0), 1, cv2.LINE_AA)
    cv2.putText(hud, "[SPACE]: Jeda | [D]: Step", (728, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(hud, "[Q] : Keluar", (728, 196), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1, cv2.LINE_AA)

    # FOOTER BAR (Tinggi 24 px)
    footer = np.zeros((24, 1000, 3), dtype=np.uint8)
    footer[:] = (14, 14, 18)
    ft_text = "[AI SLAT RIB CONVEYOR - AUTO-CALIBRATION WIZARD M-02 SENSORS 2025]"
    cv2.putText(footer, ft_text, (20, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)

    combined = np.vstack([banner, cam_resized, hud, footer])
    return combined


# ==============================================================================
# MAIN RUNNER PROGRAM KALIBRASI
# ==============================================================================
def run_cycle_calibration(video_source=None, config_path=None):
    """
    Menjalankan proses kalibrasi 1 siklus konveyor interaktif.
    :param video_source: Path file video atau index kamera USB (0, 1, 2)
    :param config_path: Path file konfigurasi JSON tujuan
    """
    if config_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, "configs", "top_inspection_config.json")

    # Daftar pencarian video demo jika tidak dispesifikasikan
    search_dirs = [
        os.path.abspath(os.path.join(base_dir, "..", "Video_Testing")),
        os.path.abspath(os.path.join(base_dir, "..", "..", "Video_Testing")),
        os.path.abspath(os.path.join(base_dir, "..", "Documentation", "Video Testing")),
        os.path.abspath(os.path.join(base_dir, "..", "..", "Documentation", "Video Testing")),
        os.path.join(os.path.expanduser("~"), "Videos"),
        os.path.join(os.path.expanduser("~"), "Pictures"),
    ]
    playlist = []
    if video_source and os.path.exists(str(video_source)):
        playlist.append(os.path.abspath(video_source))
    for sdir in search_dirs:
        if os.path.exists(sdir):
            for fname in sorted(os.listdir(sdir)):
                if fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    playlist.append(os.path.abspath(os.path.join(sdir, fname)))

    if video_source is None:
        if playlist:
            video_source = playlist[0]
        else:
            video_source = select_video_file(title="Pilih Video / Aliran Konveyor untuk Kalibrasi")

    if video_source is None:
        print("[ERROR] Tidak ada input video / kamera yang valid.")
        return

    # Inisialisasi VideoCapture (bisa file path atau camera index)
    is_video_file = False
    try:
        cam_idx = int(video_source)
        cap = cv2.VideoCapture(cam_idx)
        src_label = f"Live Camera USB #{cam_idx}"
        is_video_file = False
    except ValueError:
        cap = cv2.VideoCapture(str(video_source))
        src_label = f"File: {os.path.basename(str(video_source))}"
        is_video_file = True

    if not cap.isOpened():
        print(f"[ERROR] Gagal membuka sumber video: {video_source}")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1.0 or np.isnan(fps):
        fps = 30.0

    # Dapatkan gerbang polygon awal adaptif terhadap orientasi kamera
    slat_polygon_norm, trig_box, roi_desc = get_adaptive_slat_polygon(frame_w, frame_h, config_path)
    slat_polygon_px = np.array([[p[0] * frame_w, p[1] * frame_h] for p in slat_polygon_norm], dtype=np.float32)

    calibrator = SlatCycleCalibrator(config_path=config_path, fps=fps, is_video_mode=is_video_file)

    # Untuk input file video, kalibrasi otomatis dimulai dari awal hingga akhir video
    if is_video_file:
        calibrator.trigger_cycle_start(0)

    win_name = "AUTO-CALIBRATION WIZARD: Slat Rib Conveyor (1-Cycle Baseline)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1100, 850)

    print("\n" + "=" * 78)
    print("🚀 MODUL AUTO-KALIBRASI 1 SIKLUS KONVEYOR DIAKTIFKAN")
    print("=" * 78)
    print(f"Sumber Input     : {src_label} ({frame_w}x{frame_h} @ {fps:.1f} FPS)")
    print(f"Config Target    : {config_path}")
    print("Mode Operasi     : " + ("File Video (Otomatis Selesai di Akhir Video)" if is_video_file else "Kamera Langsung (Deteksi Marker Merah Plat #01)"))
    print("Petunjuk Singkat :")
    if is_video_file:
        print("  1. Sistem merekam seluruh profil plat dari awal sampai akhir file video.")
        print("  2. Saat video selesai (frame terakhir), kalibrasi otomatis diselesaikan.")
        print("  3. Tekan [S] untuk menyimpan hasil kalibrasi ke file JSON.")
        print("  4. Tekan [SPACE] / [R] untuk memutar ulang dari awal jika diperlukan.")
    else:
        print("  1. Konveyor berputar sampai Penanda Merah (Plat #01) terdeteksi.")
        print("  2. Sistem merekam seluruh profil plat selama 1 siklus penuh.")
        print("  3. Saat penanda merah kembali melintas, kalibrasi selesai otomatis.")
        print("  4. Tekan [S] untuk menyimpan hasil ke file JSON.")
    print("=" * 78 + "\n")

    is_paused = False
    last_calib_data = None
    last_frame = None

    while True:
        if not is_paused:
            ret, frame = cap.read()
            if not ret:
                if is_video_file:
                    # Video mencapai akhir file (EOF): otomatis selesaikan siklus kalibrasi!
                    if calibrator.state != SlatCycleCalibrator.STATE_COMPLETE:
                        calibrator.trigger_cycle_finish(calibrator.current_frame_idx)
                        print(f"\n[AUTO-FINISH] Akhir video tercapai (Total: {calibrator.current_frame_idx} frame)!")
                        print(f"[OK] Kalibrasi otomatis diselesaikan dengan {calibrator.measured_slat_count} plat terdata.")
                        print(f"[OK] Silakan tekan [S] untuk menyimpan hasil kalibrasi ke JSON.")
                    is_paused = True
                    if last_calib_data:
                        calib_data = dict(last_calib_data)
                        calib_data["state"] = calibrator.state
                        calib_data["frames_in_cycle"] = max(0, calibrator.cycle_end_frame - calibrator.cycle_start_frame)
                        calib_data["measured_slat_count"] = calibrator.measured_slat_count
                        calib_data["calib_results"] = calibrator.calib_results
                        calib_data["is_video_file"] = is_video_file
                        calib_data["is_paused"] = is_paused
                        last_calib_data = calib_data
                    continue
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

            calib_data = calibrator.process_frame(frame, slat_polygon_norm)
            calib_data["is_video_file"] = is_video_file
            last_calib_data = calib_data
            last_frame = frame
        else:
            time.sleep(0.03)
            # Saat jeda, tahan frame dan data analisis terakhir tanpa memajukan counter
            if last_calib_data is not None and last_frame is not None:
                calib_data = dict(last_calib_data)
                frame = last_frame
            else:
                ret, frame = cap.read()
                if ret:
                    calib_data = calibrator.process_frame(frame, slat_polygon_norm)
                    calib_data["is_video_file"] = is_video_file
                    last_calib_data = calib_data
                    last_frame = frame

        calib_data["is_saved"] = calibrator.is_profile_saved
        calib_data["is_paused"] = is_paused

        # Render antarmuka dashboard
        display_img = render_calibration_hud(frame, calib_data, slat_polygon_px)
        cv2.imshow(win_name, display_img)

        wait_ms = 40 if is_paused else 25
        key = cv2.waitKey(wait_ms) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord(' '):
            if is_video_file and is_paused and calibrator.state == SlatCycleCalibrator.STATE_COMPLETE:
                # Jika sudah di akhir video dan tekan SPACE, ulang putar dari awal
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                calibrator.reset()
                calibrator.trigger_cycle_start(0)
                is_paused = False
                print("\n[*] Memutar ulang video kalibrasi dari frame awal...")
            else:
                is_paused = not is_paused
        elif key in [ord('d'), ord('D')]:
            # Step 1 frame maju saat sedang jeda / pause
            if is_paused:
                ret, frame = cap.read()
                if ret:
                    calib_data = calibrator.process_frame(frame, slat_polygon_norm)
                    calib_data["is_video_file"] = is_video_file
                    last_calib_data = calib_data
                    last_frame = frame
        elif key in [ord('m'), ord('M')]:
            # Trigger Penanda Merah Manual (Simulasi)
            if calibrator.state == SlatCycleCalibrator.STATE_WAIT_MARKER:
                calibrator.trigger_cycle_start(calibrator.current_frame_idx)
            elif calibrator.state == SlatCycleCalibrator.STATE_CALIBRATING:
                calibrator.trigger_cycle_finish(calibrator.current_frame_idx)
        elif key in [ord('c'), ord('C')]:
            # Selesaikan kalibrasi seketika tanpa menunggu akhir
            if calibrator.state == SlatCycleCalibrator.STATE_CALIBRATING:
                calibrator.trigger_cycle_finish(calibrator.current_frame_idx)
        elif key in [ord('s'), ord('S')]:
            # Simpan hasil kalibrasi ke file JSON config
            if calibrator.calib_results:
                calibrator.save_calibration_to_config(config_path)
            else:
                print("[!] Belum ada data siklus yang selesai. Selesaikan siklus kalibrasi terlebih dahulu.")
        elif key in [ord('r'), ord('R')]:
            calibrator.reset()
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if is_video_file:
                calibrator.trigger_cycle_start(0)
            is_paused = False
            print("\n[*] Kalibrasi direset. Memulai ulang dari awal video...")
        elif key in [ord('o'), ord('f')]:
            new_file = select_video_file(title="Pilih File Video Baru untuk Kalibrasi")
            if new_file and os.path.exists(new_file):
                cap.release()
                video_source = new_file
                cap = cv2.VideoCapture(new_file)
                frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                is_video_file = True
                slat_polygon_norm, trig_box, roi_desc = get_adaptive_slat_polygon(frame_w, frame_h, config_path)
                slat_polygon_px = np.array([[p[0] * frame_w, p[1] * frame_h] for p in slat_polygon_norm], dtype=np.float32)
                calibrator.fps = fps
                calibrator.is_video_mode = True
                calibrator.reset()
                calibrator.trigger_cycle_start(0)
                is_paused = False

    cap.release()
    cv2.destroyAllWindows()


# ==============================================================================
# ENTRY POINT CLI
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool Auto-Kalibrasi 1 Siklus Konveyor Slat Rib")
    parser.add_argument("--video", "-v", type=str, default=None, help="Path video atau index kamera (misal: 0)")
    parser.add_argument("--config", "-c", type=str, default=None, help="Path file konfigurasi JSON target")
    args = parser.parse_args()

    run_cycle_calibration(video_source=args.video, config_path=args.config)
