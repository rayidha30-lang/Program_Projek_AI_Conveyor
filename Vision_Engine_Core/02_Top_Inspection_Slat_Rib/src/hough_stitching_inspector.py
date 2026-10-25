"""
==============================================================================
MODUL INSPEKSI KELURUSAN PLAT BERBASIS HOUGH LINE GROUPING & STITCHING
(M-02 ENHANCED - IMPLEMENTASI ADAPTASI PAPER SENSORS 2025)
==============================================================================
Referensi Ilmiah:
  Chen, Y.-J., Yeh, Y.-H., & Yang, J.-F. (2025). 
  "A System for the Real-Time Detection of the U-Shaped Steel Bar Straightness on a Production Line." 
  Sensors 2025, 25(13), 3972. https://doi.org/10.3390/s25133972

Prinsip Algoritma:
  1. Canonical ROI Unwarp: Menormalkan perspektif gerbang plat ke dimensi kanonikal (1000 x 140 px).
  2. Adaptive Edge Detection (1-Stage vs 2-Stage):
     - Menghitung kecerahan rata-rata ROI (V).
     - Jika V >= V_thresh (116) -> 1-Stage Canny (kondisi terang).
     - Jika V < V_thresh (116) -> 2-Stage Sobel-Y + Canny (kondisi redup / bayangan).
  3. Hough Line Detection & Horizontal Angle Filter:
     - Deteksi segmen garis dengan Probabilistic Hough Transform (cv2.HoughLinesP).
     - Filter noise: Eliminasi garis non-horizontal (|theta| > 15 deg) & segmen < min_line_len.
  4. Line Grouping (Tepi Atas vs Tepi Bawah):
     - Mengelompokkan garis berdasarkan koordinat Y tengah (Top Edge vs Bottom Edge).
  5. Line Stitching (Penyambungan Garis Berdasarkan Kesamaan Sudut):
     - Mengambil garis terpanjang sebagai sudut acuan (theta_ref).
     - Menggabungkan semua segmen garis yang memiliki selisih sudut |Delta_theta| <= delta_theta_thresh.
     - Menghitung panjang gabungan yang tersambung (L_stitched) & koreksi sudut tilt L_m = L_s * cos(theta_ref).
  6. Evaluasi Kelurusan (Minimum Line Length - MLL):
     - MLL = lambda * W_ROI (misal: 75% dari bentang 1000 px = 750 px).
     - Jika L_m >= MLL -> LURUS (OK).
     - Jika L_m < MLL  -> BENGKOK (NG) (garis patah/sudut terdisrupsi oleh kurva kelengkungan).
==============================================================================
"""

import os
import sys
import json
import time
import math
from collections import deque
import cv2
import numpy as np


def fast_medfilt(arr, k=17):
    """Filter median 1D ter-vektorisasi cepat menggunakan sliding_window_view NumPy."""
    pad = k // 2
    padded = np.pad(arr, pad, mode='edge')
    windows = np.lib.stride_tricks.sliding_window_view(padded, k)
    return np.median(windows, axis=1)

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def order_quad_points(pts):
    """Mengurutkan 4 titik kuadran: [Top-Left, Top-Right, Bottom-Right, Bottom-Left]."""
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


class HoughStitchingInspector:
    """
    Inspector Kelurusan Plat Slat Konveyor berbasis Adaptive Edge + Hough Line Stitching.
    """
    def __init__(
        self,
        v_thresh=116.0,
        mll_ratio=0.70,
        delta_theta_thresh=1.5,
        min_line_len=45,
        bridge_gap_px=110.0,
        angle_filter_deg=15.0,
        strict_mode=True
    ):
        """
        :param v_thresh: Ambang batas kecerahan rata-rata untuk memilih 1-Stage vs 2-Stage Edge Detection (default: 116).
        :param mll_ratio: Rasio Minimum Line Length terhadap lebar ROI kanonikal (default: 0.70 -> 700 px dari 1000 px).
        :param delta_theta_thresh: Toleransi selisih sudut untuk penyambungan garis dalam derajat (default: 1.5 deg).
        :param min_line_len: Panjang piksel minimum segmen garis Hough (default: 45 px).
        :param bridge_gap_px: Toleransi jeda celah penyambungan garis, disesuaikan dengan jeda las penguat tengah (default: 110 px).
        :param angle_filter_deg: Batas kemiringan sudut horizontal maksimum yang diizinkan (default: 15.0 deg).
        :param strict_mode: Jika True, kedua tepi (atas & bawah) harus >= MLL. Jika False, salah satu cukup.
        """
        self.v_thresh = float(v_thresh)
        self.mll_ratio = float(mll_ratio)
        self.delta_theta_thresh = float(delta_theta_thresh)
        self.min_line_len = int(min_line_len)
        self.bridge_gap_px = float(bridge_gap_px)
        self.angle_filter_deg = float(angle_filter_deg)
        self.strict_mode = bool(strict_mode)

        self.canonical_w = 1000
        self.canonical_h = 140

        # Anti-Flicker Line Persistence (Diadopsi dari video_hough_line_inspector)
        self.cached_lines = None
        self.line_fade_frames = 0

        # Slat-by-Slat Debounce & Counting State (Diadopsi dari video_hough_line_inspector)
        self.slat_count = 0
        self.ok_count = 0
        self.ng_count = 0
        self.slat_latched = False
        self.last_trigger_frame = -999
        self.current_frame_idx = 0
        self.current_slat_max_l = 0.0
        self.latched_status = "WAIT"
        self.latched_color = (255, 180, 0)

        # Pelacakan Akumulasi Transit Plat (Peak-Hold Slat Transit)
        self.slat_active = False
        self.current_slat_peak_l = 0.0
        self.current_slat_best_res = None
        self.transit_frames = 0
        self.last_finish_frame = -999
        self._was_active_prev = False
        self.transit_samples = []

        # Ambang Batas Metrologi Geometris Profil (Metrologi Kelurusan & Kelengkungan)
        self.max_flatness_pv = 16.5       # px: Batas maksimum defleksi Peak-to-Valley dari baseline lurus
        self.max_sagitta = 8.6            # px: Batas maksimum kelengkungan parabola (Sagitta)
        self.min_continuity_cov = 0.25    # Rasio kontinuitas celah minimum (25% dari bentang metal 500 px)
        self.max_center_void = 1450       # px: Batas luas celah terbuka (void gap) akibat plat meleyot
        self.tracked_y_seam = 70.0        # px: Pelacakan kontinu horizon celah plat
        self.scale_factor_mm_per_px = 0.515  # Faktor skala fisik konveyor (Ground Truth: Plat Slat 820mm / 1592px = 0.515 mm/px)

        # Mini Trigger Zone Baut Rantai Samping (Anchor Pemicu Celah Slat)
        self.use_bolt_trigger = False
        self.trigger_sensitivity = 18.0
        self.trigger_box = [0.01, 0.52, 0.08, 0.67]  # [x1, y1, x2, y2] proporsional lebar-tinggi frame
        self.trigger_history = deque(maxlen=15)
        self.last_bolt_frame = -999
        self.min_bolt_interval = 18
        self.trigger_flash_frames = 0
        self.last_triggered_results = None
        self.current_bolt_energy = 0.0

        # Deteksi Gerakan Konveyor (Conveyor Motion & Standby Protection)
        self.prev_unwarped_gray = None
        self.motion_energy = 10.0          # Nilai awal diasumsikan konveyor bergerak
        self.is_conveyor_moving = True
        self.conveyor_stopped_frames = 0
        self.motion_thresh = 2.2           # Ambang batas energi piksel gerak (noise kamera ~0.5 - 1.5, gerak konveyor ~5 - 20)
        self.gate_armed = True             # Latch arming: Hanya siap memicu jika plat sebelumnya sudah keluar & konveyor bergerak

    def reset_counters(self):
        """Mereset seluruh counter inspeksi plat (Slat #, OK, NG, dan status latch)."""
        self.slat_count = 0
        self.ok_count = 0
        self.ng_count = 0
        self.slat_latched = False
        self.slat_active = False
        self.current_slat_peak_l = 0.0
        self.current_slat_best_res = None
        self.transit_frames = 0
        self.transit_samples = []
        self.last_trigger_frame = -999
        self.last_finish_frame = -999
        self._was_active_prev = False
        self.current_slat_max_l = 0.0
        self.latched_status = "WAIT"
        self.tracked_y_seam = 70.0
        self.cached_lines = None
        self.line_fade_frames = 0
        self.prev_unwarped_gray = None
        self.motion_energy = 10.0
        self.is_conveyor_moving = True
        self.conveyor_stopped_frames = 0
        self.gate_armed = True

    def adaptive_edge_detection(self, unwarped_bgr):
        """
        Pipeline Deteksi Tepi Super Bersih & Anti-Karat (Diadopsi dari video_hough_line_inspector):
        1. Bilateral Filter d=9, sigma=75 (meratakan karat & oli tanpa memudarkan celah fisik).
        2. Filter Dominansi Gradien Sobel-Y (|Sy| >= 1.15*|Sx|) untuk memotong goresan vertikal.
        3. Canny selektif tegas (55-140) untuk memotong noise permukaan metal.
        4. Morphological Horizontal Bridge (25x1) untuk menyambung patahan mikro garis celah.
        5. Morphological Open (3x1) untuk membersihkan bintik karat non-garis yang tersisa.
        """
        gray = cv2.cvtColor(unwarped_bgr, cv2.COLOR_BGR2GRAY)
        v_mean = float(np.mean(gray))

        # 1. Bilateral filter anti-karat kuat (d=9, sigma=75)
        blurred = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

        # 2. Filter Dominansi Gradien Sobel-Y (memblokir serat karat vertikal & bayangan)
        sy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        sx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        abs_sy = np.abs(sy)
        abs_sx = np.abs(sx)
        h_mask = ((abs_sy >= 1.15 * abs_sx) * 255).astype(np.uint8)

        # 3. Canny Edge dengan Ambang Batas Tegas & Selektif (Toleran Pantulan Oli & Kapur)
        if v_mean >= self.v_thresh:
            edges_raw = cv2.Canny(blurred, 45, 120, apertureSize=3)
            mode_tag = "Canny Tegas (Bright: 45-120)"
            max_line_gap = 35
        else:
            edges_raw = cv2.Canny(blurred, 35, 100, apertureSize=3)
            mode_tag = "Canny Adaptif (Low-Light: 35-100)"
            max_line_gap = 40

        edges_directional = cv2.bitwise_and(edges_raw, h_mask)

        # 4. Morphological Horizontal Bridging (25x1)
        k_bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        bridged_edges = cv2.morphologyEx(edges_directional, cv2.MORPH_CLOSE, k_bridge)

        # 5. Morphological Opening (Membersihkan bintik non-garis yang tersisa)
        k_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        cleaned_edges = cv2.morphologyEx(bridged_edges, cv2.MORPH_OPEN, k_clean)

        return cleaned_edges, mode_tag, v_mean, max_line_gap

    def detect_and_filter_lines(self, edges, max_line_gap):
        """
        Deteksi Garis HoughLineP Bersih & Filter Sudut Horizontal.
        """
        lines_raw = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=35,
            minLineLength=self.min_line_len,
            maxLineGap=max_line_gap
        )

        valid_lines = []
        if lines_raw is not None:
            for l in lines_raw:
                x1, y1, x2, y2 = l[0]
                dx = x2 - x1
                dy = y2 - y1
                length = math.hypot(dx, dy)
                if length < self.min_line_len:
                    continue

                # Hitung sudut kemiringan dalam derajat [-90 s.d. +90]
                angle_deg = math.degrees(math.atan2(dy, dx))
                if angle_deg > 90.0:
                    angle_deg -= 180.0
                elif angle_deg < -90.0:
                    angle_deg += 180.0

                # Filter garis yang sejajar bidang horizontal konveyor (|angle| <= angle_filter_deg)
                if abs(angle_deg) <= self.angle_filter_deg:
                    # Pastikan urutan x1 <= x2 untuk mempermudah perhitungan span
                    if x1 > x2:
                        x1, x2 = x2, x1
                        y1, y2 = y2, y1
                    valid_lines.append({
                        "pts": (x1, y1, x2, y2),
                        "length": length,
                        "angle": angle_deg,
                        "y_mid": (y1 + y2) / 2.0,
                        "x_min": min(x1, x2),
                        "x_max": max(x1, x2)
                    })

        # Anti-Flicker Line Persistence (Diadopsi dari video_hough_line_inspector)
        # Menjaga persistensi garis selama 2 frame jika terjadi drop sesaat akibat kompresi video
        if len(valid_lines) > 0:
            self.cached_lines = valid_lines
            self.line_fade_frames = 0
        elif self.cached_lines is not None and self.line_fade_frames < 2:
            valid_lines = self.cached_lines
            self.line_fade_frames += 1
        else:
            self.cached_lines = None
            self.line_fade_frames = 0

        return valid_lines

    def locate_seam_horizon(self, edges):
        """
        Melokalisasi garis horizon celah utama plat (True Seam Horizon)
        berdasarkan proyeksi konsentrasi energi gradien horizontal pada bentang metal tengah (x in [150, 850])
        dengan prior bobot tengah Gaussian (mencegah loncat ke tepi rusuk plat tetangga).
        """
        # Proyeksi energi hanya pada area plat metal (x: 150 s.d. 850), menolak rantai & bracket samping
        row_sum = np.sum(edges[:, 150:850] > 0, axis=1).astype(np.float32)
        kernel = cv2.getGaussianKernel(11, 2.5)
        smoothed = cv2.filter2D(row_sum, -1, kernel).flatten()

        y_min, y_max = 25, 115
        if y_max <= y_min:
            return self.canonical_h // 2, 0.0

        active_region = smoothed[y_min:y_max].copy()
        ys = np.arange(y_min, y_max)
        center_weight = np.exp(-0.5 * ((ys - 70.0) / 28.0) ** 2)

        if self.slat_active and abs(self.tracked_y_seam - 70.0) < 40:
            temporal_weight = np.exp(-0.5 * ((ys - self.tracked_y_seam) / 18.0) ** 2)
            weighted_region = active_region * (0.6 * center_weight + 0.4 * temporal_weight)
        else:
            weighted_region = active_region * center_weight

        peak_idx = int(np.argmax(weighted_region))
        y_seam = y_min + peak_idx
        peak_energy = float(active_region[peak_idx])

        if self.slat_active:
            self.tracked_y_seam = 0.7 * y_seam + 0.3 * self.tracked_y_seam
        else:
            self.tracked_y_seam = float(y_seam)

        return y_seam, peak_energy

    def group_lines_by_seam(self, lines, y_seam, band_half=28):
        """
        Mengelompokkan garis yang berada di sekitar horizon celah utama plat.
        Menolak garis bintik karat, pantulan oli, atau sambungan las sirip di luar pita celah.
        """
        seam_lines = []
        outlier_lines = []

        for line in lines:
            if abs(line["y_mid"] - y_seam) <= band_half:
                seam_lines.append(line)
            else:
                outlier_lines.append(line)

        return seam_lines, outlier_lines

    def stitch_line_group(self, line_group):
        """
        Stage 5 Paper: Line Stitching & Angular Clustering.
        1. Menemukan segmen garis terpanjang sebagai sudut referensi (theta_ref).
        2. Mengelompokkan semua garis dengan |Delta_theta| <= delta_theta_thresh.
        3. Menghitung total bentang horizontal tersambung (L_stitched) & koreksi tilt.
        """
        if not line_group:
            return {
                "l_stitched": 0.0,
                "l_corrected": 0.0,
                "theta_ref": 0.0,
                "stitched_lines": [],
                "rejected_lines": [],
                "span": (0, 0),
                "is_empty": True
            }

        # 1. Cari sudut acuan (Reference Angle) berdasarkan rata-rata berbobot panjang garis horizontal
        horiz_candidates = [l for l in line_group if abs(l["angle"]) <= 8.0]
        if horiz_candidates:
            weights = [l["length"] for l in horiz_candidates]
            angles = [l["angle"] for l in horiz_candidates]
            theta_ref = float(np.average(angles, weights=weights))
        else:
            line_group_sorted = sorted(line_group, key=lambda item: item["length"], reverse=True)
            theta_ref = line_group_sorted[0]["angle"]

        # 2. Angular proximity clustering (|Delta_theta| <= threshold)
        stitched_lines = []
        rejected_lines = []

        intervals = []
        for line in line_group:
            delta_theta = abs(line["angle"] - theta_ref)
            if delta_theta <= self.delta_theta_thresh:
                stitched_lines.append(line)
                intervals.append((line["x_min"], line["x_max"]))
            else:
                rejected_lines.append(line)

        if not intervals:
            return {
                "l_stitched": 0.0,
                "l_corrected": 0.0,
                "theta_ref": theta_ref,
                "stitched_lines": [],
                "rejected_lines": line_group,
                "span": (0, 0),
                "is_empty": True
            }

        # 3. Gabungkan interval X yang bertumpukan / terhubung
        intervals.sort(key=lambda x: x[0])
        merged_intervals = []
        cur_start, cur_end = intervals[0]

        # Toleransi celah sambungan horizontal antar garis (disesuaikan dengan jeda las tengah)
        bridge_gap = self.bridge_gap_px

        for next_start, next_end in intervals[1:]:
            if next_start <= cur_end + bridge_gap:
                cur_end = max(cur_end, next_end)
            else:
                merged_intervals.append((cur_start, cur_end))
                cur_start, cur_end = next_start, next_end
        merged_intervals.append((cur_start, cur_end))

        # Cari bentang sambungan kontinyu terpanjang
        longest_span = max(merged_intervals, key=lambda span: span[1] - span[0])
        span_length = float(longest_span[1] - longest_span[0])

        # Koreksi sudut kemiringan plat L_m = L_s * cos(theta_ref)
        l_corrected = span_length * math.cos(math.radians(theta_ref))

        return {
            "l_stitched": span_length,
            "l_corrected": l_corrected,
            "theta_ref": theta_ref,
            "stitched_lines": stitched_lines,
            "rejected_lines": rejected_lines,
            "span": longest_span,
            "is_empty": False
        }

    def inspect_unwarped(self, unwarped_bgr):
        """
        Menjalankan seluruh pipeline inspeksi pada citra kanonikal unwarped (1000 x 140).
        Mengimplementasikan Metrologi Profil Geometris:
        1. Adaptive Edge Detection (Bilateral Filter + Sobel-Y Directional + Canny)
        2. Lokalisasi Horizon Celah Utama (True Seam Horizon Gating Y in [45, 95])
        3. Ekstraksi Kontur Fisik Celah pada Bentang Plat Metal (x in [150, 850])
        4. Metrologi Flatness (Peak-to-Valley Deviation terhadap baseline lurus)
        5. Metrologi Kelengkungan Parabola (Euler-Bernoulli Sagitta)
        6. Metrologi Kontinuitas Celah (Coverage Ratio)
        7. Deteksi Garis Hough Line Grouping & Stitching (Kompatibilitas Mundur Penuh)
        """
        t0 = time.time()
        h, w = unwarped_bgr.shape[:2]

        # 1. Adaptive Edge Detection anti-karat & oli
        edges, mode_tag, v_mean, max_line_gap = self.adaptive_edge_detection(unwarped_bgr)

        # 2. Deteksi Garis Hough & Filter Sudut (untuk visualisasi & stitching kompatibel)
        lines = self.detect_and_filter_lines(edges, max_line_gap)

        # 3. Lokalisasi Horizon Celah Utama (True Seam Horizon pada bentang metal tengah)
        y_seam, seam_energy = self.locate_seam_horizon(edges)

        # 4. Kumpulkan garis hanya pada zona celah utama & lakukan line stitching
        seam_lines, outlier_lines = self.group_lines_by_seam(lines, y_seam, band_half=24)
        seam_res = self.stitch_line_group(seam_lines)

        # 5. METROLOGI PROFIL GEOMETRIS KONTUR CELAH PLAT (x in [220, 720])
        # 5. METROLOGI PROFIL GEOMETRIS KONTUR CELAH PLAT (x in [240, 680])
        # Rentang bentang pelat baja aktif (mengecualikan ceruk engsel baut rantai samping x < 240 dan x > 680)
        # sehingga celah baut rantai tidak memicu lonjakan deviasi Flatness PV palsu.
        xs = np.arange(240, 680)
        w_span = len(xs)

        band = 18 if (self.slat_active and abs(self.tracked_y_seam - 70.0) < 30) else 26
        y_lo = max(0, y_seam - band)
        y_hi = min(h, y_seam + band)
        sub_edges = edges[y_lo:y_hi, 240:680]

        # Citra Grayscale untuk Dark Valley Intensity Weighting (parit gelap celah vs pantulan terang)
        gray = cv2.cvtColor(unwarped_bgr, cv2.COLOR_BGR2GRAY)
        sub_gray = gray[y_lo:y_hi, 240:680]

        # Ekstraksi kontur fisik celah kontinu (Tahap 1: Ekstraksi tepi kasar dengan Dark Valley Weighting - Vectorized)
        kernel_vert = np.ones((3, 1), dtype=np.uint8)
        min_gray = cv2.erode(sub_gray, kernel_vert)
        penalty = 5.0 * (min_gray.astype(np.float32) / 255.0)

        nom = float(y_seam - y_lo)
        y_coords = np.arange(sub_edges.shape[0], dtype=np.float32)[:, None]
        cost1 = np.abs(y_coords - nom) + penalty
        cost1[sub_edges == 0] = 1e6

        min_cost1 = np.min(cost1, axis=0)
        best_pts1 = np.argmin(cost1, axis=0).astype(np.float32)

        # Sub-pixel Parabolic Refinement Tahap 1 (O(1) Vectorized - Presisi < 0.1 mm)
        if cost1.shape[0] >= 3:
            idx_int1 = np.clip(best_pts1.astype(np.int32), 1, cost1.shape[0] - 2)
            cols1 = np.arange(cost1.shape[1])
            c_prev1 = cost1[idx_int1 - 1, cols1]
            c_curr1 = cost1[idx_int1, cols1]
            c_next1 = cost1[idx_int1 + 1, cols1]
            denom1 = 2.0 * (c_prev1 - 2.0 * c_curr1 + c_next1)
            sub_shift1 = np.divide(c_prev1 - c_next1, denom1, out=np.zeros_like(denom1, dtype=np.float32), where=(np.abs(denom1) > 1e-4))
            sub_shift1 = np.clip(sub_shift1, -0.5, 0.5)
            best_pts1 = best_pts1 + sub_shift1

        valid_mask1 = (min_cost1 < 1e5) & (np.abs(best_pts1 - nom) <= 12.0)

        raw_ys = np.where(valid_mask1, y_lo + best_pts1, np.nan)
        valid_mask = ~np.isnan(raw_ys)

        # Tahap 2: Estimasi Garis Seam Adaptif (Batch Parallel RANSAC 1D) untuk mengoreksi tilt
        # dan menolak jebakan bekas las (weld bead) atau bintik spatter metal
        if np.sum(valid_mask) > 50:
            vx = xs[valid_mask]
            vy = raw_ys[valid_mask]
            n_samples = len(vx)

            np.random.seed(42)
            idx1 = np.random.randint(0, n_samples, size=100)
            idx2 = np.random.randint(0, n_samples, size=100)
            pair_mask = np.abs(vx[idx2] - vx[idx1]) >= 100

            best_line = None
            if np.any(pair_mask):
                v_x1, v_x2 = vx[idx1[pair_mask]], vx[idx2[pair_mask]]
                v_y1, v_y2 = vy[idx1[pair_mask]], vy[idx2[pair_mask]]
                slopes = (v_y2 - v_y1) / (v_x2 - v_x1)
                slope_mask = np.abs(slopes) <= 0.08

                if np.any(slope_mask):
                    s_filt = slopes[slope_mask]
                    i_filt = v_y1[slope_mask] - s_filt * v_x1[slope_mask]

                    # Evaluasi semua kandidat secara serentak (Matrix Broadcasting)
                    preds = s_filt[:, None] * vx[None, :] + i_filt[:, None]
                    resids = np.abs(vy[None, :] - preds)
                    inliers_counts = np.sum(resids <= 2.5, axis=1)

                    best_idx = int(np.argmax(inliers_counts))
                    best_inliers_count = inliers_counts[best_idx]

                    if (best_inliers_count / float(n_samples)) >= 0.35:
                        best_line = (float(s_filt[best_idx]), float(i_filt[best_idx]))

            # Jika garis RANSAC ditemukan dengan inlier memadai (>= 35% titik)
            if best_line is not None:
                m_fit, c_fit = best_line
                # Tahap 3: Re-ekstraksi tepi kolom menggunakan baseline adaptif m_fit*x + c_fit + Dark Valley Weighting (Vectorized)
                nom_x = (m_fit * xs + c_fit) - y_lo
                cost3 = np.abs(y_coords - nom_x[None, :]) + penalty
                cost3[sub_edges == 0] = 1e6

                min_cost3 = np.min(cost3, axis=0)
                best_pts3 = np.argmin(cost3, axis=0).astype(np.float32)

                # Sub-pixel Parabolic Refinement Tahap 3 (O(1) Vectorized - Presisi < 0.1 mm)
                if cost3.shape[0] >= 3:
                    idx_int3 = np.clip(best_pts3.astype(np.int32), 1, cost3.shape[0] - 2)
                    cols3 = np.arange(cost3.shape[1])
                    c_prev3 = cost3[idx_int3 - 1, cols3]
                    c_curr3 = cost3[idx_int3, cols3]
                    c_next3 = cost3[idx_int3 + 1, cols3]
                    denom3 = 2.0 * (c_prev3 - 2.0 * c_curr3 + c_next3)
                    sub_shift3 = np.divide(c_prev3 - c_next3, denom3, out=np.zeros_like(denom3, dtype=np.float32), where=(np.abs(denom3) > 1e-4))
                    sub_shift3 = np.clip(sub_shift3, -0.5, 0.5)
                    best_pts3 = best_pts3 + sub_shift3

                valid_mask3 = (min_cost3 < 1e5) & (np.abs(best_pts3 - nom_x) <= 6.5)

                raw_ys = np.where(valid_mask3, y_lo + best_pts3, np.nan)
                valid_mask = ~np.isnan(raw_ys)

        # Deteksi Gap Void pada celah tengah
        center_void = int(np.sum(gray[max(0, y_seam - 22):min(h, y_seam + 22), 400:600] < 45))

        if np.sum(valid_mask) > 50:
            valid_indices = np.where(valid_mask)[0]
            interp_temp = np.interp(np.arange(w_span), valid_indices, raw_ys[valid_mask])
            smooth_baseline = fast_medfilt(interp_temp, 31)

            # Buang titik yang melompat > 4.0 px dari baseline lokal (penolakan noise multi-tepi)
            diff = np.abs(raw_ys - smooth_baseline)
            clean_mask = valid_mask & (diff <= 4.0)

            if np.sum(clean_mask) > 40:
                clean_indices = np.where(clean_mask)[0]
                ys_interp = np.interp(np.arange(w_span), clean_indices, raw_ys[clean_mask])
                y_smooth = fast_medfilt(ys_interp, 17)

                poly1 = np.polyfit(xs, y_smooth, deg=1)
                baseline = np.polyval(poly1, xs)
                dev = y_smooth - baseline
                pv = float(np.percentile(dev, 95) - np.percentile(dev, 5))
                max_dev = float(np.max(np.abs(dev)))

                poly2 = np.polyfit(xs, y_smooth, deg=2)
                sagitta = abs(float(poly2[0])) * ((w_span / 2.0) ** 2)
                cov = float(np.mean(clean_mask))
            else:
                y_smooth = np.full(w_span, float(y_seam))
                baseline = np.full(w_span, float(y_seam))
                pv = 0.0
                max_dev = 0.0
                sagitta = 0.0
                cov = 0.0
        else:
            y_smooth = np.full(w_span, float(y_seam))
            baseline = np.full(w_span, float(y_seam))
            pv = 0.0
            max_dev = 0.0
            sagitta = 0.0
            cov = 0.0

        # Evaluasi Kriteria Kualitas Geometris Plat QC:
        is_flat_ok = bool(pv <= self.max_flatness_pv)
        is_curv_ok = bool(sagitta <= self.max_sagitta)
        # Jika plat terbukti lurus (is_flat_ok dan is_curv_ok), penurunan coverage akibat pantulan oli/kapur/cahaya
        # tidak boleh memicu status cacat NG. Cacat celah terputus hanya sah jika ada distorsi fisik.
        if is_flat_ok and is_curv_ok:
            is_cov_ok = bool(cov >= 0.10)
        else:
            is_cov_ok = bool(cov >= self.min_continuity_cov)

        # Void gap memicu NG jika luas celah terbuka melebihi toleransi (indikasi plat meleyot/mangap)
        is_void_ok = bool(center_void < getattr(self, "max_center_void", 1300))

        is_seam_ok = bool(is_flat_ok and is_curv_ok and is_cov_ok and is_void_ok)

        # Cek kondisi transisi: Jika energi celah rendah ATAU sampel kontur tidak memadai (< 40 titik)
        is_transition = bool(seam_energy < 18.0 or np.sum(valid_mask) <= 40)

        # Panjang efektif celah tersambung (proporsional bentang terukur)
        l_seam = float(cov * float(w_span))
        mll_px = float(w_span * self.mll_ratio)

        theta_ref = seam_res.get("theta_ref", 0.0)
        deform_lines = []
        for l in seam_lines + seam_res.get("rejected_lines", []):
            d_ang = abs(l["angle"] - theta_ref)
            if (d_ang >= 2.5 or abs(l["angle"]) >= 3.5) and l["length"] >= 35.0:
                deform_lines.append(l)

        if is_transition:
            status = "WAIT"
            is_ok = False
            is_ng = False
            mode_eval = "WAIT (Fase Transisi Celah Plat)"
        else:
            is_ok = is_seam_ok
            is_ng = not is_ok
            status = "OK" if is_ok else "NG"
            if is_seam_ok:
                mode_eval = f"LURUS OK (PV={pv:.1f}px, Sag={sagitta:.1f}px, Cov={cov*100:.0f}%)"
            elif not is_void_ok:
                mode_eval = f"NG: CELAH MANGAP/MELEYOT (Void={center_void}px > {self.max_center_void}px)"
            elif not is_curv_ok:
                mode_eval = f"NG: BENGKOK/MELENGKUNG (Sag={sagitta:.1f}px > {self.max_sagitta:.1f}px)"
            elif not is_flat_ok:
                mode_eval = f"NG: KELURUSAN KASAR (PV={pv:.1f}px > {self.max_flatness_pv:.1f}px)"
            else:
                mode_eval = f"NG: CELAH TERPUTUS/RETAK (Cov={cov*100:.0f}% < {self.min_continuity_cov*100:.0f}%)"

        elapsed_ms = (time.time() - t0) * 1000.0

        return {
            "status": status,
            "is_ok": is_ok,
            "is_ng": is_ng,
            "is_transition": is_transition,
            "l_seam": l_seam,
            "y_seam": y_seam,
            "seam_energy": seam_energy,
            "mll_px": mll_px,
            "mll_ratio": self.mll_ratio,
            "is_seam_ok": is_seam_ok,
            # Metrik Metrologi Geometris Utama:
            "confidence": float(cov),
            "pv_flatness": pv,
            "sagitta": sagitta,
            "coverage": cov,
            "center_void": center_void,
            "max_dev": max_dev,
            "is_flat_ok": is_flat_ok,
            "is_curv_ok": is_curv_ok,
            "is_cov_ok": is_cov_ok,
            "is_void_ok": is_void_ok,
            "xs_contour": xs,
            "ys_contour": y_smooth,
            "baseline": baseline,
            # Kompatibilitas mundur dengan script pipeline & visualisasi:
            "seam_res": seam_res,
            "outlier_lines": outlier_lines,
            "deform_lines": deform_lines,
            "has_severe_deform": bool(not is_curv_ok or not is_flat_ok),
            "has_deform_angle": bool(not is_curv_ok),
            "v_mean": v_mean,
            "edge_mode": mode_tag,
            "eval_mode": mode_eval,
            "edges": edges,
            "num_lines": len(lines),
            "num_seam_lines": len(seam_lines),
            "elapsed_ms": elapsed_ms,
            "l_top": l_seam,
            "l_bot": l_seam,
            "is_top_ok": is_seam_ok,
            "is_bot_ok": is_seam_ok,
            "top_res": seam_res,
            "bot_res": seam_res
        }

    def check_bolt_trigger(self, frame, trigger_box_norm=None):
        """
        Mendeteksi kehadiran baut rantai samping (Chain Bolt / Pin)
        yang bergerak melewati Mini Trigger Zone di sisi samping konveyor.
        """
        if trigger_box_norm is None:
            trigger_box_norm = self.trigger_box

        h_f, w_f = frame.shape[:2]
        x1 = max(0, int(trigger_box_norm[0] * w_f))
        y1 = max(0, int(trigger_box_norm[1] * h_f))
        x2 = min(w_f - 1, int(trigger_box_norm[2] * w_f))
        y2 = min(h_f - 1, int(trigger_box_norm[3] * h_f))

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
            return False, 0.0, (x1, y1, x2, y2)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # Hitung energi gradien vertikal dari kepala baut yang lewat
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        energy = float(np.mean(np.abs(sobel_y)))
        self.current_bolt_energy = energy

        self.trigger_history.append(energy)

        frames_since = self.current_frame_idx - self.last_bolt_frame
        is_triggered = False

        # Kondisi pemicuan: energi melampaui ambang batas dan melewati jeda antar plat
        if len(self.trigger_history) >= 3 and frames_since >= self.min_bolt_interval:
            hist = list(self.trigger_history)
            curr = hist[-1]
            prev = hist[-2]
            # Deteksi lonjakan energi kepala baut saat mencapai puncak
            if curr >= self.trigger_sensitivity and (curr >= prev or curr >= self.trigger_sensitivity * 1.25):
                is_triggered = True
                self.last_bolt_frame = self.current_frame_idx
                self.trigger_flash_frames = 8

        return is_triggered, energy, (x1, y1, x2, y2)

    def inspect_frame(self, frame, polygon_pts):
        """
        Inspeksi frame kamera utuh dengan polygon ROI 4-titik dan True Seam Horizon Gating (Solusi 1).
        Menggunakan Virtual Center Beam Debounce Latch untuk stabilitas pembacaan 100%.
        """
        h_f, w_f = frame.shape[:2]
        pts_px = []
        for p in polygon_pts:
            x = p[0] * w_f if p[0] <= 1.0 else p[0]
            y = p[1] * h_f if p[1] <= 1.0 else p[1]
            pts_px.append([x, y])

        src_pts = order_quad_points(pts_px)
        dst_pts = np.array([
            [0, 0],
            [self.canonical_w - 1, 0],
            [self.canonical_w - 1, self.canonical_h - 1],
            [0, self.canonical_h - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        unwarped = cv2.warpPerspective(frame, M, (self.canonical_w, self.canonical_h))

        # 0. Deteksi Gerakan Konveyor (Conveyor Motion Detector)
        # Menghitung perbedaan intensitas piksel frame-ke-frame pada pelat baja kanonikal
        unwarped_gray = cv2.cvtColor(unwarped, cv2.COLOR_BGR2GRAY)
        if self.prev_unwarped_gray is not None:
            roi_curr = unwarped_gray[15:125, 200:800]
            roi_prev = self.prev_unwarped_gray[15:125, 200:800]
            diff_val = float(np.mean(cv2.absdiff(roi_curr, roi_prev)))
            self.motion_energy = 0.35 * diff_val + 0.65 * self.motion_energy
        else:
            self.motion_energy = 10.0

        self.prev_unwarped_gray = unwarped_gray.copy()

        # Konveyor dinyatakan berhenti jika energi gerak rendah selama beberapa frame berturut-turut
        if self.motion_energy < self.motion_thresh:
            self.conveyor_stopped_frames += 1
        else:
            self.conveyor_stopped_frames = 0

        self.is_conveyor_moving = bool(self.conveyor_stopped_frames < 4)

        # 1. Jalankan inspeksi unwarped pada horizon celah utama
        res = self.inspect_unwarped(unwarped)
        y_seam = res["y_seam"]
        seam_energy = res["seam_energy"]

        # 2. Periksa Mini Bolt Trigger jika diaktifkan pengguna
        is_bolt_triggered, bolt_energy, trigger_bbox = self.check_bolt_trigger(frame)

        # 3. Virtual Center Beam Debounce Latch & Transit Tracking Metrologi
        self.current_frame_idx += 1
        frames_since_finish = self.current_frame_idx - self.last_finish_frame
        frames_since_start = self.current_frame_idx - self.last_trigger_frame

        # Re-arm Gerbang: Hanya siap memicu plat berikutnya jika konveyor bergerak
        # dan plat sebelumnya telah berpindah (y_seam keluar zona atau terjadi transisi jeda)
        if not self.gate_armed:
            if self.is_conveyor_moving and (y_seam < 55 or res["is_transition"] or y_seam >= 95):
                self.gate_armed = True

        # Kondisi pemicuan awal: Konveyor HARUS bergerak, gerbang ter-arm, celah plat di zona masuk tengah
        in_entry_zone = (not res["is_transition"]) and (60 <= y_seam <= 84)
        trigger_now = (
            self.is_conveyor_moving and
            self.gate_armed and
            not self.slat_active and
            in_entry_zone and
            frames_since_finish >= 6 and
            frames_since_start >= 12
        )
        if self.use_bolt_trigger and is_bolt_triggered and not self.slat_active and self.is_conveyor_moving and self.gate_armed:
            trigger_now = True

        is_new_slat = False
        if trigger_now:
            self.slat_active = True
            self.gate_armed = False  # Kunci gerbang agar tidak memicu berulang pada plat yang sama
            self.last_trigger_frame = self.current_frame_idx
            self.slat_count += 1
            self.transit_samples = []
            self.transit_frames = 0
            self.latched_status = "WAIT"
            is_new_slat = True

        # Selama plat melintasi zona gerbang, akumulasikan sampel profil metrologi
        if self.slat_active:
            if self.is_conveyor_moving:
                self.transit_frames += 1
            dist_center = abs(y_seam - 70.0)
            if not res["is_transition"]:
                self.transit_samples.append({
                    "res": dict(res),
                    "dist": dist_center,
                    "y_seam": y_seam,
                    "is_ok": res["is_ok"],
                    "is_ng": res["is_ng"],
                    "pv": res["pv_flatness"],
                    "sagitta": res["sagitta"],
                    "cov": res["coverage"],
                    "void": res["center_void"]
                })

            # Kondisi keluar gerbang: Hanya boleh keluar jika konveyor bergerak!
            # Jika konveyor berhenti, tahan plat di dalam gerbang tanpa exit paksa
            if self.is_conveyor_moving:
                exited_zone = (y_seam < 56) or (self.transit_frames >= 5 and y_seam >= 88) or (self.transit_frames >= 14)
            else:
                exited_zone = False

            if exited_zone:
                self.slat_active = False
                self.last_finish_frame = self.current_frame_idx

                # Evaluasi transit berdasarkan sampel zona tengah prima (58 <= y_seam <= 88)
                prime_samples = [s for s in self.transit_samples if 58 <= s["y_seam"] <= 88]
                eval_samples = prime_samples

                if len(eval_samples) >= 2:
                    s_sags = [s["sagitta"] for s in eval_samples]
                    s_pvs = [s["pv"] for s in eval_samples]
                    prime_voids = [s["void"] for s in eval_samples if 58 <= s["y_seam"] <= 88]
                    s_voids = prime_voids if prime_voids else [s["void"] for s in eval_samples]
                    s_covs = [s["cov"] for s in eval_samples]

                    med_sag = float(np.median(s_sags))
                    med_pv = float(np.median(s_pvs))
                    max_void = max(s_voids) if s_voids else 0
                    med_cov = float(np.median(s_covs)) if s_covs else 1.0
                    max_cov = float(np.max(s_covs)) if s_covs else 1.0

                    is_physically_flat = (med_sag <= self.max_sagitta) and (med_pv <= self.max_flatness_pv)

                    # 1. Cacat kelengkungan / bengkok fisik nyata
                    bend_reject = (med_sag > self.max_sagitta) or (med_pv > self.max_flatness_pv)

                    # 2. Cacat celah terbuka / meleyot (void area melampaui batas toleransi)
                    void_reject = (max_void >= self.max_center_void)

                    # 3. Cacat kontinuitas hanya berlaku jika sepanjang transit plat sama sekali tidak pernah terdeteksi utuh
                    #    (max_cov < 20%) DAN tidak memenuhi syarat kelurusan flat
                    cov_reject = (max_cov < 0.20) and not is_physically_flat

                    is_slat_ng = bend_reject or void_reject or cov_reject

                    if is_slat_ng:
                        self.ng_count += 1
                        self.latched_status = "NG"
                    else:
                        self.ok_count += 1
                        self.latched_status = "OK"

                    best = min(eval_samples, key=lambda s: s["dist"])
                    self.current_slat_best_res = best["res"]
                else:
                    self.latched_status = "WAIT"

        # Single Source of Truth: Penentuan status tampilan (display_status) bebas ghosting & flicker
        if not self.is_conveyor_moving:
            # Konveyor berhenti: kunci status ke HOLD (Standby) agar tidak memicu alarm palsu
            display_status = "HOLD"
        elif self.slat_active:
            if len(self.transit_samples) > 0:
                # Jika terdeteksi cacat kelengkungan nyata atau void parah pada sampel terkini, tampilkan NG
                recent_ngs = [
                    s for s in self.transit_samples[-3:]
                    if s["is_ng"] and (58 <= s.get("y_seam", 0) <= 88) and (
                        s["sagitta"] > self.max_sagitta or
                        s["pv"] > self.max_flatness_pv or
                        s["void"] >= self.max_center_void
                    )
                ]
                if len(recent_ngs) >= 1:
                    display_status = "NG"
                elif min([s["dist"] for s in self.transit_samples]) <= 20:
                    display_status = "OK"
                else:
                    display_status = "WAIT"
            else:
                display_status = "WAIT"
        else:
            # Jeda antar-plat: tahan status hasil plat sebelumnya selama 25 frame (~0.8 detik), lalu kembali ke WAIT
            frames_since_fin = self.current_frame_idx - self.last_finish_frame
            if frames_since_fin <= 25 and self.latched_status in ["OK", "NG"]:
                display_status = self.latched_status
            elif res.get("is_ng", False) and not res.get("is_transition", True) and (58 <= res.get("y_seam", 0) <= 88) and (
                res.get("sagitta", 0) > self.max_sagitta or
                res.get("pv_flatness", 0) > self.max_flatness_pv or
                res.get("center_void", 0) > self.max_center_void
            ):
                display_status = "NG"
            else:
                display_status = "WAIT"

        self.slat_latched = self.slat_active

        # Tambahkan metadata frame tersinkronisasi 100%
        res["display_status"] = display_status
        res["status"] = display_status
        res["is_ok"] = (display_status == "OK")
        res["is_ng"] = (display_status == "NG")
        res["is_conveyor_moving"] = self.is_conveyor_moving
        res["motion_energy"] = self.motion_energy
        res["unwarped"] = unwarped
        res["polygon_px"] = src_pts
        res["trigger_box"] = trigger_bbox
        res["bolt_energy"] = bolt_energy
        res["is_triggered"] = is_bolt_triggered or is_new_slat
        res["is_new_slat"] = is_new_slat
        res["slat_count"] = self.slat_count
        res["ok_count"] = self.ok_count
        res["ng_count"] = self.ng_count
        res["latched_status"] = self.latched_status
        res["peak_l"] = self.current_slat_peak_l

        return res

    def render_diagnostic_overlay(self, unwarped_bgr, results):
        """
        Merender visualisasi anotasi lengkap hasil deteksi, stitching, dan metrik
        berdasarkan True Seam Horizon Locking (Solusi 1) dengan tampilan HUD besar,
        kontras tinggi, dan mudah dibaca dari jarak jauh.
        """
        annotated = unwarped_bgr.copy()
        seam_res = results.get("seam_res", {})
        y_seam = int(results.get("y_seam", self.canonical_h // 2))
        mll_px = float(results.get("mll_px", self.canonical_w * self.mll_ratio))
        l_seam = float(results.get("l_seam", 0.0))
        status = results.get("display_status", results.get("status", "WAIT"))
        is_ok = bool(status == "OK")

        pv = float(results.get("pv_flatness", 0.0))
        sagitta = float(results.get("sagitta", 0.0))
        cov = float(results.get("coverage", 0.0))
        max_dev = float(results.get("max_dev", 0.0))
        xs = results.get("xs_contour", None)
        ys = results.get("ys_contour", None)
        baseline = results.get("baseline", None)

        if is_ok:
            c_seam = (0, 255, 0)       # Hijau: Keputusan OK (Passed)
        elif status == "HOLD":
            c_seam = (0, 220, 255)     # Cyan: Konveyor Berhenti (Hold / Standby)
        elif status == "WAIT":
            c_seam = (0, 165, 255)     # Oren: Deteksi Awal / Sedang Transit (Wait)
        else:
            c_seam = (0, 0, 255)       # Merah: Keputusan NG (Reject)

        # 1. Gambar Pita Horizon Celah Utama (True Seam Horizon Band)
        band_half = 20
        y_band_top = max(0, y_seam - band_half)
        y_band_bot = min(self.canonical_h - 1, y_seam + band_half)

        # Overlay transparan halus untuk zona pita horizon
        overlay_band = annotated.copy()
        cv2.rectangle(overlay_band, (0, y_band_top), (self.canonical_w - 1, y_band_bot), (30, 50, 40), -1)
        cv2.addWeighted(overlay_band, 0.20, annotated, 0.80, 0, annotated)

        # Garis batas zona horizon celah (garis pandu)
        cv2.line(annotated, (0, y_band_top), (self.canonical_w - 1, y_band_top), (60, 95, 70), 1, cv2.LINE_4)
        cv2.line(annotated, (0, y_band_bot), (self.canonical_w - 1, y_band_bot), (60, 95, 70), 1, cv2.LINE_4)
        cv2.putText(annotated, f"SEAM GATE ZONE (Y={y_seam}px)", (15, max(15, y_band_top - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 190, 150), 1, cv2.LINE_AA)

        # 2. Gambar Garis Baseline Netral (Lurus Ideal)
        if xs is not None and baseline is not None and len(xs) > 1:
            pt_start = (int(xs[0]), int(baseline[0]))
            pt_end = (int(xs[-1]), int(baseline[-1]))
            cv2.line(annotated, pt_start, pt_end, (220, 240, 255), 1, cv2.LINE_AA)
            cv2.putText(annotated, "BASELINE NETRAL", (int(xs[0]), max(12, int(baseline[0]) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 220, 240), 1, cv2.LINE_AA)

        # 3. Gambar Kontur Fisik Celah Terukur (Metrologi Profil Kontinu)
        if xs is not None and ys is not None and len(xs) > 1:
            pts_contour = np.column_stack((xs, ys)).astype(np.int32)
            cv2.polylines(annotated, [pts_contour], isClosed=False, color=c_seam, thickness=3, lineType=cv2.LINE_AA)

            # Titik penanda di ujung kiri dan kanan bentang metal
            cv2.circle(annotated, (int(xs[0]), int(ys[0])), 5, c_seam, -1)
            cv2.circle(annotated, (int(xs[-1]), int(ys[-1])), 5, c_seam, -1)

            # Gambar titik puncak defleksi (Sagitta) di tengah bentang
            mid_idx = len(xs) // 2
            cv2.circle(annotated, (int(xs[mid_idx]), int(ys[mid_idx])), 6, (0, 255, 255), -1)

            # Pill Badge Metrologi di Atas Kontur
            scale = getattr(self, "scale_factor_mm_per_px", 0.515)
            lbl_metro = f"METROLOGI: Conf={cov:.2f} | PV={pv*scale:.1f}mm ({pv:.1f}px) | Sag={sagitta*scale:.1f}mm ({sagitta:.1f}px)"
            pill_w = 510
            pill_h = 24
            pill_x1 = max(10, min(self.canonical_w - pill_w - 10, int(xs[mid_idx]) - (pill_w // 2)))
            pill_y1 = max(6, min(self.canonical_h - pill_h - 6, y_seam - 32))
            pill_x2 = pill_x1 + pill_w
            pill_y2 = pill_y1 + pill_h

            cv2.rectangle(annotated, (pill_x1, pill_y1), (pill_x2, pill_y2), (18, 18, 22), -1)
            cv2.rectangle(annotated, (pill_x1, pill_y1), (pill_x2, pill_y2), c_seam, 2)
            cv2.putText(annotated, lbl_metro, (pill_x1 + 10, pill_y1 + 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.41, (255, 255, 255), 1, cv2.LINE_AA)

        # 4. Gambar Batas Bentang Metal Konveyor (x=180 dan x=820)
        cv2.line(annotated, (180, 0), (180, self.canonical_h - 1), (80, 80, 100), 1, cv2.LINE_4)
        cv2.line(annotated, (820, 0), (820, self.canonical_h - 1), (80, 80, 100), 1, cv2.LINE_4)
        cv2.putText(annotated, "CHAIN L", (110, self.canonical_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (120, 120, 140), 1)
        cv2.putText(annotated, "CHAIN R", (830, self.canonical_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (120, 120, 140), 1)

        # 5. HUD Panel Informasi di bawah Unwarped (Tinggi 175px - Industrial Metrology Dashboard)
        hud_h = 175
        hud = np.zeros((hud_h, self.canonical_w, 3), dtype=np.uint8)
        hud[:] = (18, 18, 24)

        if status == "OK":
            status_color = (0, 230, 0)
            status_tag = "PASSED (LURUS)"
            tag_bg = (24, 48, 30)
        elif status == "HOLD":
            status_color = (0, 220, 255)  # Cyan
            status_tag = "STANDBY (STOP)"
            tag_bg = (15, 35, 48)
        elif status == "WAIT":
            status_color = (0, 165, 255)  # Oren / Amber
            status_tag = "TRANSISI PLAT"
            tag_bg = (15, 35, 48)
        else:
            status_color = (0, 30, 255)
            status_tag = "REJECT (BENGKOK)"
            tag_bg = (48, 20, 20)

        # --- KARTU 1: STATUS INSPEKSI BESAR (X: 12 - 195) ---
        cv2.rectangle(hud, (12, 10), (195, hud_h - 10), (28, 28, 36), -1)
        cv2.rectangle(hud, (12, 10), (195, hud_h - 10), status_color, 2)
        cv2.putText(hud, "STATUS QC METROLOGI", (25, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)
        
        stat_x = 55 if len(status) == 2 else (40 if len(status) == 3 else 32)
        cv2.putText(hud, status, (stat_x, 88), cv2.FONT_HERSHEY_DUPLEX, 1.8, status_color, 3, cv2.LINE_AA)
        
        cv2.rectangle(hud, (22, 110), (185, 142), tag_bg, -1)
        cv2.rectangle(hud, (22, 110), (185, 142), status_color, 1)
        cv2.putText(hud, status_tag, (26, 131), cv2.FONT_HERSHEY_SIMPLEX, 0.40, status_color, 1, cv2.LINE_AA)

        # --- KARTU 2: METROLOGI KELURUSAN & KELENGKUNGAN (X: 208 - 690) ---
        cv2.rectangle(hud, (208, 10), (690, hud_h - 10), (28, 28, 36), -1)
        cv2.rectangle(hud, (208, 10), (690, hud_h - 10), (55, 55, 68), 1)

        cv2.putText(hud, "METROLOGI KELURUSAN & KELENGKUNGAN (PARABOLA DEFLECTION)", (222, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)

        # Bar 1: Flatness (Peak-to-Valley Deviation)
        scale = getattr(self, "scale_factor_mm_per_px", 0.515)
        pv_mm = pv * scale
        max_pv_mm = self.max_flatness_pv * scale
        pv_col = (0, 220, 0) if pv <= self.max_flatness_pv else (0, 40, 255)
        cv2.putText(hud, f"Flatness PV : {pv_mm:4.1f} mm ({pv:.1f} px) [Maks: {max_pv_mm:.1f} mm ({self.max_flatness_pv:.1f} px)]", (222, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.41, (240, 240, 240), 1, cv2.LINE_AA)
        
        # Gauge bar Flatness
        gx1, gy1, gx2, gy2 = 475, 42, 675, 56
        gw = gx2 - gx1
        cv2.rectangle(hud, (gx1, gy1), (gx2, gy2), (40, 40, 50), -1)
        fill_pv = int(np.clip(pv / max(1.0, self.max_flatness_pv * 1.5), 0.0, 1.0) * gw)
        cv2.rectangle(hud, (gx1, gy1), (gx1 + fill_pv, gy2), pv_col, -1)
        # Limit marker
        m_pv_x = gx1 + int((self.max_flatness_pv / (self.max_flatness_pv * 1.5)) * gw)
        cv2.line(hud, (m_pv_x, gy1 - 2), (m_pv_x, gy2 + 2), (0, 255, 255), 2)
        cv2.rectangle(hud, (gx1, gy1), (gx2, gy2), (80, 80, 95), 1)

        # Bar 2: Kelengkungan Parabola (Sagitta)
        sag_col = (0, 220, 0) if sagitta <= self.max_sagitta else (0, 40, 255)
        sag_mm = sagitta * scale
        max_sag_mm = self.max_sagitta * scale
        cv2.putText(hud, f"Sagitta Curv: {sag_mm:4.1f} mm ({sagitta:.1f} px) [Maks: {max_sag_mm:.1f} mm ({self.max_sagitta:.1f} px)]", (222, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.41, (240, 240, 240), 1, cv2.LINE_AA)
        
        # Gauge bar Sagitta
        sy1, sy2 = 68, 82
        cv2.rectangle(hud, (gx1, sy1), (gx2, sy2), (40, 40, 50), -1)
        fill_sag = int(np.clip(sagitta / max(1.0, self.max_sagitta * 1.5), 0.0, 1.0) * gw)
        cv2.rectangle(hud, (gx1, sy1), (gx1 + fill_sag, sy2), sag_col, -1)
        # Limit marker
        m_sag_x = gx1 + int((self.max_sagitta / (self.max_sagitta * 1.5)) * gw)
        cv2.line(hud, (m_sag_x, sy1 - 2), (m_sag_x, sy2 + 2), (0, 255, 255), 2)
        cv2.rectangle(hud, (gx1, sy1), (gx2, sy2), (80, 80, 95), 1)

        # Baris 3: Kontinuitas & Deviasi Maksimum
        cov_col = (0, 220, 0) if cov >= self.min_continuity_cov else (0, 165, 255)
        max_dev_mm = max_dev * scale
        l_seam_mm = l_seam * scale
        cv2.putText(hud, f"Kontinuitas Celah: {cov*100:.1f}% [Conf: {cov:.2f}] (Min {self.min_continuity_cov*100:.0f}%) | Max Dev: {max_dev_mm:.1f} mm ({max_dev:.1f} px) | Celah: {l_seam_mm:.0f} mm ({l_seam:.0f} px)",
                    (222, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (190, 215, 235), 1, cv2.LINE_AA)

        # Baris 4: Deskripsi Diagnosa
        if status == "HOLD" or not results.get("is_conveyor_moving", True):
            status_detail = "STANDBY (KONVEYOR BERHENTI - PENGUKURAN DI-HOLD)"
            diag_color = (0, 220, 255)
        else:
            status_detail = results.get("eval_mode", ("OK (LURUS & UTUH)" if is_ok else ("WAIT (TRANSISI PLAT)" if status == "WAIT" else "NG (BENGKOK)")))
            diag_color = (200, 245, 200) if is_ok else ((255, 210, 150) if status == "WAIT" else (100, 130, 255))
        cv2.putText(hud, f"Diagnosa QC: {status_detail}", (222, 134),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, diag_color, 1, cv2.LINE_AA)

        cv2.putText(hud, f"Filter: {results.get('edge_mode', '-')} | Horizon Seam Y: {y_seam} px | Motion: {results.get('motion_energy', 0.0):.1f}",
                    (222, 156), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 170), 1, cv2.LINE_AA)

        # --- KARTU 3: STATISTIK QC PRODUKSI & TELEMETRI (X: 703 - 988) ---
        cv2.rectangle(hud, (703, 10), (988, hud_h - 10), (28, 28, 36), -1)
        cv2.rectangle(hud, (703, 10), (988, hud_h - 10), (55, 55, 68), 1)

        slat_count = results.get("slat_count", self.slat_count)
        ok_count = results.get("ok_count", self.ok_count)
        ng_count = results.get("ng_count", self.ng_count)
        defect_rate = (ng_count / max(1, slat_count)) * 100.0 if slat_count > 0 else 0.0

        cv2.putText(hud, "STATISTIK QC PRODUKSI", (718, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(hud, f"TOTAL SLAT : #{slat_count:02d}", (718, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(hud, f"PASSED (OK): {ok_count:02d}", (718, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 230, 0), 2, cv2.LINE_AA)
        cv2.putText(hud, f"REJECT (NG): {ng_count:02d}", (718, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 50, 255), 2, cv2.LINE_AA)
        cv2.putText(hud, f"DEFECT RATE: {defect_rate:.1f}% | {results.get('elapsed_ms', 0.0):.1f} ms",
                    (718, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (200, 200, 200), 1, cv2.LINE_AA)

        # ----------------------------------------------------------------------
        # SEKAT PEMISAH (DIVIDER BAR) ANTARA KAMERA UTAMA DAN HASIL DETEKSI
        # ----------------------------------------------------------------------
        # Berikan border bingkai tegas pada viewport annotated agar terpisah jelas dari kamera
        cv2.rectangle(annotated, (0, 0), (self.canonical_w - 1, self.canonical_h - 1), (0, 200, 255), 2)

        div_h = 36
        divider = np.zeros((div_h, self.canonical_w, 3), dtype=np.uint8)
        divider[:] = (16, 20, 26)

        # Garis aksen pemisah atas yang tegas, tebal, dan berpendar (Cyan Kontras Tinggi)
        cv2.line(divider, (0, 0), (self.canonical_w - 1, 0), (0, 220, 255), 3, cv2.LINE_AA)
        cv2.line(divider, (0, 2), (self.canonical_w - 1, 2), (0, 140, 180), 1, cv2.LINE_AA)
        cv2.line(divider, (0, div_h - 1), (self.canonical_w - 1, div_h - 1), (0, 180, 220), 2, cv2.LINE_AA)

        # Badge Label Sekat di sisi kiri dengan background kontras dan border menyala
        badge_w = 485
        cv2.rectangle(divider, (12, 6), (12 + badge_w, div_h - 6), (25, 35, 48), -1)
        cv2.rectangle(divider, (12, 6), (12 + badge_w, div_h - 6), (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(divider, "HASIL DETEKSI METROLOGI (CANONICAL UNWARP SLAT)", (22, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 230, 255), 1, cv2.LINE_AA)

        # Label Keterangan di sisi kanan
        cv2.putText(divider, f"[PERSPEKTIF GERBANG NORMALISASI | SEAM Y={y_seam}px]", (self.canonical_w - 425, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 220, 255), 1, cv2.LINE_AA)

        # Garis pemisah antara annotated dan hud dashboard
        cv2.line(hud, (0, 0), (self.canonical_w - 1, 0), (0, 200, 255), 2, cv2.LINE_AA)

        combined = np.vstack([divider, annotated, hud])
        return combined


# ==============================================================================
# DEMO RUNNER INTERAKTIF (OPENCV GUI)
# ==============================================================================

def get_adaptive_slat_polygon(frame_w, frame_h, config_path=None):
    """
    Menentukan batas gerbang inspeksi polygon secara adaptif berdasarkan rasio aspek:
    - Vertikal / Portret (H > W, misal 1080x1920 seperti Deform Parah.mp4):
      Gunakan batas gerbang presisi [0.03 s.d. 0.97, Y: 0.56 s.d. 0.66]
    - Horizontal / Lanskap (W > H, misal 1920x1080 seperti Opsi Posisi 1 - Slat Rib.mp4):
      Gunakan polygon dari config JSON jika ada, atau polygon standar bersudut.
    """
    if frame_h > frame_w:
        poly = [
            [0.03, 0.385],
            [0.97, 0.385],
            [0.97, 0.490],
            [0.03, 0.490]
        ]
        trigger_box = [0.01, 0.385, 0.08, 0.490]
        desc = f"Portret Vertikal ({frame_w}x{frame_h} - Auto Gate)"
        return poly, trigger_box, desc

    # Landscape default
    poly = [
        [0.079, 0.523],
        [0.924, 0.505],
        [0.961, 0.658],
        [0.048, 0.668]
    ]
    trigger_box = [0.01, 0.52, 0.08, 0.67]
    desc = f"Lanskap Horizontal ({frame_w}x{frame_h})"

    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
                if "slat_polygon" in cfg and len(cfg["slat_polygon"]) == 4:
                    poly = cfg["slat_polygon"]
                    desc = f"Lanskap Horizontal ({frame_w}x{frame_h} - Config JSON)"
                if "trigger_box" in cfg and len(cfg["trigger_box"]) == 4:
                    trigger_box = cfg["trigger_box"]
        except Exception:
            pass

    return poly, trigger_box, desc


def select_video_file(initial_dir=None, title="Pilih Video Inspeksi"):
    """Membuka file dialog Tkinter untuk memilih file video."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        file_path = filedialog.askopenfilename(
            title=title,
            initialdir=initial_dir if initial_dir and os.path.exists(initial_dir) else os.getcwd(),
            filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov"), ("All Files", "*.*")]
        )
        root.destroy()
        return file_path if file_path else None
    except Exception as e:
        print(f"[!] Gagal membuka file dialog: {e}")
        return None


def get_camera_backend():
    """Mengembalikan backend kamera yang sesuai dengan OS (DirectShow untuk Windows, CAP_ANY / V4L2 untuk Linux Jetson)."""
    return cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY


def get_available_cameras(max_cams=3):
    """Mendeteksi indeks webcam yang terhubung ke PC/laptop/Jetson."""
    available = []
    backend = get_camera_backend()
    for idx in range(max_cams):
        try:
            c = cv2.VideoCapture(idx, backend)
            if c.isOpened():
                ret, _ = c.read()
                if ret:
                    available.append(idx)
                c.release()
        except Exception:
            pass
    return available


def choose_input_source(playlist, available_cams):
    """Menampilkan menu interaktif konsol untuk memilih antara Video Testing atau Live Webcam."""
    print("\n" + "=" * 76)
    print("🎥 PILIH SUMBER INPUT INSPEKSI PLAT KONVEYOR:")
    print("=" * 76)
    print(f"  [1] Video Testing Playlist ({len(playlist)} file video terdeteksi) [DEFAULT]")
    print(f"  [2] Buka File Video Baru (Browse File Dialog)")
    for idx, c_num in enumerate(available_cams):
        opt_num = idx + 3
        c_name = "Webcam Laptop / Kamera Bawaan" if c_num == 0 else f"Kamera USB / Eksternal #{c_num}"
        print(f"  [{opt_num}] Live {c_name} (Kamera #{c_num})")
    print("-" * 76)
    try:
        if sys.stdin.isatty():
            choice = input(f"Pilih sumber input [1-{len(available_cams)+2}] (tekan ENTER untuk default [1]): ").strip()
        else:
            choice = "1"
    except Exception:
        choice = "1"

    if not choice or choice == "1":
        return "playlist", (playlist[0] if playlist else None), None
    elif choice == "2":
        selected = select_video_file(title="Pilih File Video untuk Demo Hough Stitching")
        return "file", selected, None
    else:
        try:
            c_choice = int(choice)
            cam_idx_pos = c_choice - 3
            if 0 <= cam_idx_pos < len(available_cams):
                return "webcam", None, available_cams[cam_idx_pos]
        except Exception:
            pass
        return "playlist", (playlist[0] if playlist else None), None


def run_hough_stitching_demo(video_path=None, config_path=None, cam_index=None):
    """
    Menjalankan demo interaktif metode Hough Line Stitching pada aliran video konveyor atau live webcam.
    """
    # Bangun playlist video yang dapat dirotasi dengan tombol [V]
    base_proj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    search_dirs = [
        os.path.join(base_proj_dir, "Video_Testing"),
        os.path.abspath(os.path.join(base_proj_dir, "..", "Video_Testing")),
        os.path.join(base_proj_dir, "Documentation", "Video Testing"),
        os.path.abspath(os.path.join(base_proj_dir, "..", "Documentation", "Video Testing")),
        os.path.join(os.path.expanduser("~"), "Videos"),
        os.path.join(os.path.expanduser("~"), "Pictures"),
        os.path.join(os.path.expanduser("~"), "Downloads")
    ]
    playlist = []
    if video_path and os.path.exists(video_path):
        playlist.append(os.path.abspath(video_path))
    for sdir in search_dirs:
        if os.path.exists(sdir):
            for fname in sorted(os.listdir(sdir)):
                if fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    fpath = os.path.abspath(os.path.join(sdir, fname))
                    if fpath not in playlist:
                        playlist.append(fpath)

    # Deteksi kamera webcam yang tersedia
    available_cams = get_available_cameras(max_cams=3)

    is_webcam = False
    current_cam_idx = None

    if cam_index is not None:
        is_webcam = True
        current_cam_idx = int(cam_index)
        video_path = None
    elif video_path is not None:
        is_webcam = False
        current_cam_idx = None
    else:
        # Tampilkan prompt pemilihan sumber jika dijalankan tanpa parameter
        src_type, v_ret, c_ret = choose_input_source(playlist, available_cams)
        if src_type == "webcam":
            is_webcam = True
            current_cam_idx = c_ret
            video_path = None
        else:
            is_webcam = False
            video_path = v_ret

    # Inisialisasi Capture
    if is_webcam:
        backend = get_camera_backend()
        cap = cv2.VideoCapture(current_cam_idx, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not cap.isOpened():
            print(f"[!] Gagal membuka Webcam #{current_cam_idx}. Beralih ke video...")
            is_webcam = False
            video_path = playlist[0] if playlist else None
            cap = cv2.VideoCapture(video_path)
    else:
        if not video_path and playlist:
            video_path = playlist[0]
        cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("[ERROR] Tidak ada sumber input (video maupun webcam) yang valid. Keluar.")
        return

    # Muat konfigurasi ROI jika ada
    if config_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, "configs", "top_inspection_config.json")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else 1
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

    # Tentukan polygon ROI secara adaptif terhadap orientasi frame
    slat_polygon, trig_box, roi_desc = get_adaptive_slat_polygon(frame_w, frame_h, config_path)
    if is_webcam:
        roi_desc = f"Live Webcam #{current_cam_idx} ({frame_w}x{frame_h})"
    print(f"[*] Sumber Input Aktif: {roi_desc}")

    calib_mll = 0.70
    calib_v = 116.0
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
                if "calibration_profile" in cfg:
                    cp = cfg["calibration_profile"]
                    calib_mll = float(cp.get("calibrated_mll_ratio", 0.70))
                    calib_v = float(cp.get("recommended_v_thresh", 116.0))
                    print(f"[*] Profil Kalibrasi Dimuat: MLL={calib_mll*100:.0f}%, V_Thresh={calib_v:.1f}, Tilt={cp.get('bracket_tilt_deg', 0):+.2f} deg, Slats={cp.get('total_slats_in_loop', '-')}")
        except Exception:
            pass

    inspector = HoughStitchingInspector(
        v_thresh=calib_v,
        mll_ratio=calib_mll,
        delta_theta_thresh=1.5,
        min_line_len=45,
        bridge_gap_px=115.0,
        strict_mode=True
    )
    inspector.trigger_box = trig_box

    # Terapkan ambang batas metrologi jika tersimpan di file config JSON
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
                th = cfg.get("thresholds", {})
                if "max_flatness_pv" in th:
                    inspector.max_flatness_pv = float(th["max_flatness_pv"])
                if "max_sagitta" in th:
                    inspector.max_sagitta = float(th["max_sagitta"])
                if "max_center_void" in th:
                    inspector.max_center_void = int(th["max_center_void"])
                if "min_continuity_cov" in th:
                    inspector.min_continuity_cov = float(th["min_continuity_cov"])
        except Exception:
            pass

    win_name = "DEMO: Geometric Contour Metrology Inspector (Top Inspection Slat Rib)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 890)

    win_edge = "TEPI TERISOLASI (CANNY & SOBEL-Y RADAR)"
    show_edge_window = True
    cv2.namedWindow(win_edge, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_edge, 720, 220)

    # Trackbar Callbacks
    def on_flatness_pv(val):
        inspector.max_flatness_pv = max(4.0, float(val))

    def on_sagitta(val):
        inspector.max_sagitta = max(1.5, float(val))

    def on_continuity(val):
        inspector.min_continuity_cov = max(0.15, val / 100.0)

    def on_void_max(val):
        inspector.max_center_void = max(500, int(val * 100))

    def on_v_thresh(val):
        inspector.v_thresh = float(val)

    def on_bolt_trig(val):
        inspector.use_bolt_trigger = bool(val == 1)

    def on_trig_sens(val):
        inspector.trigger_sensitivity = max(1.0, float(val))

    cv2.createTrackbar("Flatness PV Max (px)", win_name, int(inspector.max_flatness_pv), 30, on_flatness_pv)
    cv2.createTrackbar("Sagitta Max (px)", win_name, int(inspector.max_sagitta), 15, on_sagitta)
    cv2.createTrackbar("Kontinuitas Min (%)", win_name, int(inspector.min_continuity_cov * 100), 95, on_continuity)
    cv2.createTrackbar("Void Max (x100 px)", win_name, int(inspector.max_center_void // 100), 40, on_void_max)
    cv2.createTrackbar("V Thresh (Bright)", win_name, int(inspector.v_thresh), 200, on_v_thresh)
    cv2.createTrackbar("Bolt Trig (1/0)", win_name, 1 if inspector.use_bolt_trigger else 0, 1, on_bolt_trig)
    cv2.createTrackbar("Trig Sens", win_name, int(inspector.trigger_sensitivity), 50, on_trig_sens)
    cv2.createTrackbar("Timeline", win_name, 0, max(1, total_frames - 1), lambda v: None)

    is_paused = False
    simulate_bending = False
    speed_mult = 1.0

    print("\n" + "=" * 78)
    print("🚀 DEMO INTERAKTIF INSPEKTOR METROLOGI GEOMETRIS & KONTUR KELURUSAN PLAT")
    print("=" * 78)
    print("  - [SPACE]              : Pause / Play (Video) atau Freeze Frame Hold (Webcam)")
    print("  - [W]                  : Toggle / Beralih antara Live Webcam <-> Video File")
    print("  - [V]                  : Switch ke Video Berikutnya (Playlist)")
    print("  - [O] / [F]            : Buka File Video Baru (File Dialog)")
    print("  - [E]                  : Toggle Window Streaming Filter Tepi (Radar Canny/Sobel-Y)")
    print("  - [T]                  : Toggle Sensor Baut Samping (Mini Bolt Trigger)")
    print("  - [B]                  : Toggle Simulasi Plat Bengkok (+Defleksi Parabola 24px)")
    print("  - [R]                  : Reset Counter QC (Slat #0, OK: 0, NG: 0)")
    print("  - [D] / [A]            : Step Frame Maju / Mundur (Mode Video)")
    print("  - [1] - [4]            : Atur Kecepatan Pemutaran (1x, 2x, 3x, 4x)")
    print("  - [Q] / [ESC]          : Keluar")
    print("=" * 78 + "\n")

    frozen_frame = None

    while True:
        if is_webcam:
            if not is_paused:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.02)
                    continue
                frozen_frame = frame.copy()
            else:
                frame = frozen_frame.copy() if frozen_frame is not None else cap.read()[1]
                if frame is None:
                    continue
        else:
            if not is_paused:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                cur_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                cv2.setTrackbarPos("Timeline", win_name, cur_pos)
            else:
                cur_pos = cv2.getTrackbarPos("Timeline", win_name)
                cap.set(cv2.CAP_PROP_POS_FRAMES, cur_pos)
                ret, frame = cap.read()
                if not ret:
                    continue

        h_f, w_f = frame.shape[:2]

        # Jalankan inspeksi
        results = inspector.inspect_frame(frame, slat_polygon)
        unwarped_raw = results["unwarped"].copy()

        # Jika mode simulasi bending aktif, suntikkan kelengkungan kurva parabola pada unwarped
        if simulate_bending:
            h_u, w_u = unwarped_raw.shape[:2]
            bend_map_x, bend_map_y = np.meshgrid(np.arange(w_u), np.arange(h_u))
            # Defleksi parabola di tengah bentang X (+26 px)
            x_norm = (bend_map_x - w_u / 2.0) / (w_u / 2.0)
            deflection = 26.0 * (1.0 - x_norm**2)
            bend_map_y = np.clip(bend_map_y + deflection, 0, h_u - 1).astype(np.float32)
            bend_map_x = bend_map_x.astype(np.float32)
            unwarped_sim = cv2.remap(unwarped_raw, bend_map_x, bend_map_y, cv2.INTER_LINEAR)
            sim_res = inspector.inspect_unwarped(unwarped_sim)
            # Salin metadata frame agar overlay tidak error
            sim_res["polygon_px"] = results["polygon_px"]
            sim_res["trigger_box"] = results["trigger_box"]
            sim_res["bolt_energy"] = results["bolt_energy"]
            sim_res["is_triggered"] = results["is_triggered"]
            sim_res["display_status"] = "NG"
            sim_res["status"] = "NG"
            sim_res["is_ok"] = False
            sim_res["is_ng"] = True
            sim_res["latched_status"] = "NG"
            sim_res["eval_mode"] = f"NG: SIMULASI BENGKOK (Defleksi +26px, Sag={sim_res['sagitta']:.1f}px)"
            sim_res["slat_count"] = results.get("slat_count", inspector.slat_count)
            sim_res["ok_count"] = results.get("ok_count", inspector.ok_count)
            if inspector.slat_active and inspector.ng_count == 0:
                inspector.ng_count = 1
            sim_res["ng_count"] = inspector.ng_count
            results = sim_res
            unwarped_for_render = unwarped_sim
        else:
            unwarped_for_render = unwarped_raw

        # Render visualisasi unwarped + diagnostic HUD (Solusi 1)
        diag_img = inspector.render_diagnostic_overlay(unwarped_for_render, results)

        # Gambar Edge Map di samping kanan
        edge_bgr = cv2.cvtColor(results["edges"], cv2.COLOR_GRAY2BGR)
        cv2.putText(edge_bgr, f"Edge Map: {results['edge_mode']}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        edge_annotated = np.vstack([edge_bgr, np.zeros((100, results["edges"].shape[1], 3), dtype=np.uint8)])

        # Frame Utama Kamera dengan Polygon ROI
        main_display = frame.copy()
        if is_webcam:
            cv2.putText(main_display, f"LIVE WEBCAM #{current_cam_idx} [{frame_w}x{frame_h}]",
                        (main_display.shape[1] - 420, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
        poly_pts_int = results["polygon_px"].astype(np.int32)

        # Gunakan status tersinkronisasi (Single Source of Truth) bebas flicker & ghosting
        display_status = results.get("display_status", "WAIT")

        if display_status == "OK":
            poly_color = (0, 230, 0)
            badge_bg = (0, 160, 0)
        elif display_status == "HOLD":
            poly_color = (0, 220, 255)  # Cyan
            badge_bg = (0, 110, 160)
        elif display_status == "WAIT":
            poly_color = (0, 165, 255)  # Oren / Amber
            badge_bg = (0, 100, 180)    # Amber gelap
        else:
            poly_color = (0, 30, 255)
            badge_bg = (0, 0, 200)

        # Gambar Polygon Gerbang Inspeksi pada koordinat asli frame
        cv2.polylines(main_display, [poly_pts_int], isClosed=True, color=poly_color, thickness=3, lineType=cv2.LINE_AA)
        cv2.putText(main_display, "GERBANG INSPEKSI SLAT", (poly_pts_int[0][0], max(25, poly_pts_int[0][1] - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, poly_color, 2, cv2.LINE_AA)

        # Gambar Mini Trigger Zone pada lintasan baut rantai samping
        bx1, by1, bx2, by2 = results["trigger_box"]
        trig_flash = inspector.trigger_flash_frames > 0
        if trig_flash:
            inspector.trigger_flash_frames -= 1
            trig_color = (0, 255, 0)
            trig_thick = 3
            trig_label = "BOLT [TRIGGERED!]"
        elif inspector.use_bolt_trigger:
            trig_color = (0, 220, 255)
            trig_thick = 2
            trig_label = f"BOLT SENSOR ({results['bolt_energy']:.1f})"
        else:
            trig_color = (120, 120, 120)
            trig_thick = 1
            trig_label = "TRIGGER OFF"

        cv2.rectangle(main_display, (bx1, by1), (bx2, by2), trig_color, trig_thick)
        cv2.putText(main_display, trig_label, (bx1, max(18, by1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, trig_color, 1, cv2.LINE_AA)

        # Garis penghubung titik jangkar baut ke gerbang celah slat
        if inspector.use_bolt_trigger:
            connect_color = (0, 255, 255) if trig_flash else (80, 140, 160)
            cv2.line(main_display, (bx2, (by1 + by2) // 2), (poly_pts_int[0][0], poly_pts_int[0][1]),
                     connect_color, 1, cv2.LINE_4)

        # Skalakan frame utama ke dimensi kanonik 1000x500 agar overlay UI tajam 1:1 tanpa distorsi
        main_resized = cv2.resize(main_display, (1000, 500))

        # ----------------------------------------------------------------------
        # GAMBAR OVERLAY UI LANGSUNG PADA main_resized (1000 x 500 - PIXEL PERFECT)
        # ----------------------------------------------------------------------
        # 1. Kartu Status & Slat Counter di sudut kiri atas (Ukuran 460 x 68 px)
        cv2.rectangle(main_resized, (15, 12), (465, 78), (20, 20, 26), -1)
        cv2.rectangle(main_resized, (15, 12), (465, 78), poly_color, 2)

        # Pill Status di dalam kartu
        cv2.rectangle(main_resized, (20, 18), (135, 72), badge_bg, -1)
        stat_cam_x = 42 if len(display_status) == 2 else (30 if len(display_status) == 3 else 24)
        cv2.putText(main_resized, display_status, (stat_cam_x, 57),
                    cv2.FONT_HERSHEY_DUPLEX, 1.25, (255, 255, 255), 3, cv2.LINE_AA)

        # Metrik teks di samping pill
        cv2.putText(main_resized, f"SLAT #{inspector.slat_count:02d}  |  OK: {inspector.ok_count}  |  NG: {inspector.ng_count}",
                    (148, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)

        if display_status == "OK":
            status_sub = "PASSED (LURUS & UTUH)"
            sub_color = (180, 240, 200)
        elif display_status == "HOLD":
            status_sub = "⏸️ KONVEYOR BERHENTI (STANDBY)"
            sub_color = (180, 230, 255)
        elif display_status == "WAIT":
            status_sub = "TRANSISI PLAT"
            sub_color = (255, 220, 150)
        else:
            status_sub = "DEFECT REJECT (CACAT)"
            sub_color = (100, 140, 255)
        cv2.putText(main_resized, f"{status_sub} | {fps:.0f} FPS",
                    (148, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.44, sub_color, 1, cv2.LINE_AA)

        # 2. Defect Warning Banner jika status NG saat konveyor bergerak (bukan saat HOLD/berhenti)
        if display_status == "NG" and inspector.is_conveyor_moving:
            eval_note = results.get("eval_mode", "CACAT TERDETEKSI")
            cv2.rectangle(main_resized, (15, 86), (650, 126), (0, 0, 180), -1)
            cv2.rectangle(main_resized, (15, 86), (650, 126), (0, 0, 255), 2)
            cv2.putText(main_resized, f"[PERINGATAN CACAT NG] {eval_note}",
                        (25, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 255, 255), 2, cv2.LINE_AA)
        elif display_status == "HOLD" or not inspector.is_conveyor_moving:
            cv2.rectangle(main_resized, (15, 86), (580, 126), (30, 45, 60), -1)
            cv2.rectangle(main_resized, (15, 86), (580, 126), (0, 220, 255), 2)
            cv2.putText(main_resized, "[STANDBY] KONVEYOR BERHENTI - PENGUKURAN DI-HOLD",
                        (25, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 240, 255), 2, cv2.LINE_AA)

        if simulate_bending:
            cv2.rectangle(main_resized, (15, 134), (430, 168), (0, 80, 180), -1)
            cv2.rectangle(main_resized, (15, 134), (430, 168), (0, 140, 255), 2)
            cv2.putText(main_resized, "[SIMULASI DEFLEKSI/BENDING AKTIF]",
                        (25, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

        # 3. Mini Filter Radar PiP di pojok kanan atas (270 x 78 px)
        edges_raw = results["edges"]
        pip_w = 270
        pip_h = 78
        pip_crop = cv2.resize(edges_raw, (pip_w, pip_h))
        pip_bgr = cv2.cvtColor(pip_crop, cv2.COLOR_GRAY2BGR)

        # Garis acuan horizon celah di dalam radar
        pip_y_seam = int(results["y_seam"] * (pip_h / float(inspector.canonical_h)))
        cv2.line(pip_bgr, (0, pip_y_seam), (pip_w - 1, pip_y_seam), (0, 255, 0), 2, cv2.LINE_AA)

        # Border & Label Radar
        radar_col = (0, 230, 255) if results["status"] != "NG" else (0, 0, 255)
        cv2.rectangle(pip_bgr, (0, 0), (pip_w - 1, pip_h - 1), radar_col, 2)
        cv2.putText(pip_bgr, f"LIVE RADAR: {results['edge_mode'][:18]}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 230, 255), 1, cv2.LINE_AA)
        cv2.putText(pip_bgr, f"SEAM Y: {results['y_seam']}px | Energy: {results['seam_energy']:.1f}", (8, pip_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 120), 1, cv2.LINE_AA)

        # Tempelkan ke sudut kanan atas main_resized
        main_resized[12:12 + pip_h, 1000 - pip_w - 15:1000 - 15] = pip_bgr

        # 4. Window Pendamping Filter Tepi (Companion Window) jika aktif
        if show_edge_window:
            edge_disp = cv2.cvtColor(edges_raw, cv2.COLOR_GRAY2BGR)
            y_s = int(results["y_seam"])
            # Gambar garis horizon celah aktif
            cv2.line(edge_disp, (0, y_s), (edge_disp.shape[1] - 1, y_s), (0, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(edge_disp, (30, y_s), 4, (0, 255, 0), -1)
            cv2.circle(edge_disp, (edge_disp.shape[1] - 30, y_s), 4, (0, 255, 0), -1)

            # Banner informasi filter di bagian atas
            cv2.rectangle(edge_disp, (0, 0), (edge_disp.shape[1] - 1, 26), (15, 15, 20), -1)
            cv2.putText(edge_disp, f"STREAM FILTER TEPI (RADAR): {results['edge_mode']} | Horizon Seam Y={y_s}px | Energy={results['seam_energy']:.1f}",
                        (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 230, 255), 1, cv2.LINE_AA)

            disp_edge_resized = cv2.resize(edge_disp, (720, 220))
            cv2.imshow(win_edge, disp_edge_resized)

        # 5. Footer Bar Bantuan / Tombol Operasi (Tinggi 26 px)
        footer = np.zeros((26, 1000, 3), dtype=np.uint8)
        footer[:] = (14, 14, 18)
        cam_tag = f"CAM #{current_cam_idx}" if is_webcam else "OFF"
        help_bar = f"[SPACE]: {'Freeze' if is_webcam else 'Pause'} | [+]/[-]: PV ({inspector.max_flatness_pv:.1f}) | [[/]]: Sag ({inspector.max_sagitta:.1f}) | [S]: Save | [W]: Cam | [V]: Video | [Q]: Keluar"
        cv2.putText(footer, help_bar, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (190, 190, 190), 1, cv2.LINE_AA)

        # Garis aksen pemisah di bagian paling bawah frame kamera
        cv2.line(main_resized, (0, main_resized.shape[0] - 1), (main_resized.shape[1] - 1, main_resized.shape[0] - 1), (0, 220, 255), 2, cv2.LINE_AA)

        # Susun tata letak layar vertikal: Kamera (500) + Unwarped & HUD (315) + Footer (26) = 841 px
        screen = np.vstack([main_resized, diag_img, footer])
        cv2.imshow(win_name, screen)

        base_delay = max(1, int(1000 / fps))
        calc_delay = max(1, int(base_delay / speed_mult))
        wait_time = calc_delay if not is_paused else 30

        key = cv2.waitKey(wait_time) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            is_paused = not is_paused
        elif key in [ord('e'), ord('E')]:
            show_edge_window = not show_edge_window
            if not show_edge_window:
                try:
                    cv2.destroyWindow(win_edge)
                except Exception:
                    pass
            else:
                cv2.namedWindow(win_edge, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(win_edge, 720, 220)
            print(f"\n[*] Window Streaming Filter Tepi: {'AKTIF' if show_edge_window else 'NON-AKTIF'}")
        elif key in [ord('t'), ord('T')]:
            inspector.use_bolt_trigger = not inspector.use_bolt_trigger
            cv2.setTrackbarPos("Bolt Trig (1/0)", win_name, 1 if inspector.use_bolt_trigger else 0)
            print(f"\n[*] Mode Bolt Trigger: {'AKTIF' if inspector.use_bolt_trigger else 'NON-AKTIF'}")
        elif key in [ord('b'), ord('B')]:
            simulate_bending = not simulate_bending
            print(f"\n[!] Mode Simulasi Plat Bengkok: {'AKTIF' if simulate_bending else 'NON-AKTIF'}")
        elif key in [ord('r'), ord('R')]:
            inspector.reset_counters()
            print(f"\n[OK] Counter Inspeksi Direset: Slat #0, OK: 0, NG: 0")
        elif key in [ord('w'), ord('W')]:
            # Beralih ke mode Webcam atau berganti ke kamera berikutnya
            if not is_webcam:
                cams = available_cams or get_available_cameras(3)
                target_cam = cams[0] if cams else 0
                new_cap = cv2.VideoCapture(target_cam, get_camera_backend())
                new_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                new_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                if new_cap.isOpened():
                    cap.release()
                    cap = new_cap
                    is_webcam = True
                    current_cam_idx = target_cam
                    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
                    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
                    slat_polygon, trig_box, roi_desc = get_adaptive_slat_polygon(frame_w, frame_h, config_path)
                    inspector.trigger_box = trig_box
                    inspector.reset_counters()
                    cv2.setTrackbarMax("Timeline", win_name, 1)
                    cv2.setTrackbarPos("Timeline", win_name, 0)
                    print(f"\n[OK] 🔴 Beralih ke LIVE WEBCAM #{current_cam_idx} ({frame_w}x{frame_h})")
                else:
                    print(f"\n[!] Gagal membuka Kamera #{target_cam}")
            else:
                # Jika sudah di webcam, cek apakah ada kamera berikutnya, atau kembali ke video
                cams = available_cams or get_available_cameras(3)
                switched = False
                if cams and len(cams) > 1:
                    cur_c_pos = cams.index(current_cam_idx) if current_cam_idx in cams else 0
                    if cur_c_pos + 1 < len(cams):
                        next_cam = cams[cur_c_pos + 1]
                        new_cap = cv2.VideoCapture(next_cam, get_camera_backend())
                        new_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                        new_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                        if new_cap.isOpened():
                            cap.release()
                            cap = new_cap
                            current_cam_idx = next_cam
                            frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
                            frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
                            slat_polygon, trig_box, roi_desc = get_adaptive_slat_polygon(frame_w, frame_h, config_path)
                            inspector.trigger_box = trig_box
                            inspector.reset_counters()
                            print(f"\n[OK] 🔴 Beralih ke LIVE WEBCAM #{current_cam_idx} ({frame_w}x{frame_h})")
                            switched = True
                if not switched:
                    # Kembali ke Video File
                    cap.release()
                    is_webcam = False
                    current_cam_idx = None
                    video_path = video_path or (playlist[0] if playlist else None)
                    cap = cv2.VideoCapture(video_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    slat_polygon, trig_box, roi_desc = get_adaptive_slat_polygon(frame_w, frame_h, config_path)
                    inspector.trigger_box = trig_box
                    inspector.reset_counters()
                    cv2.setTrackbarMax("Timeline", win_name, max(1, total_frames - 1))
                    cv2.setTrackbarPos("Timeline", win_name, 0)
                    print(f"\n[OK] 🎬 Kembali ke Video File: {os.path.basename(video_path)} ({roi_desc})")
        elif key in [ord('v'), ord('V')]:
            # Beralih ke video berikutnya dalam playlist (atau keluar dari mode webcam)
            if is_webcam:
                cap.release()
                is_webcam = False
                current_cam_idx = None
                video_path = video_path or (playlist[0] if playlist else None)
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                slat_polygon, trig_box, roi_desc = get_adaptive_slat_polygon(frame_w, frame_h, config_path)
                inspector.trigger_box = trig_box
                inspector.reset_counters()
                cv2.setTrackbarMax("Timeline", win_name, max(1, total_frames - 1))
                cv2.setTrackbarPos("Timeline", win_name, 0)
                print(f"\n[OK] 🎬 Beralih dari Webcam ke Video File: {os.path.basename(video_path)} ({roi_desc})")
            elif playlist:
                cur_idx = playlist.index(os.path.abspath(video_path)) if os.path.abspath(video_path) in playlist else 0
                next_idx = (cur_idx + 1) % len(playlist)
                new_video = playlist[next_idx]
                cap.release()
                video_path = new_video
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                slat_polygon, trig_box, roi_desc = get_adaptive_slat_polygon(frame_w, frame_h, config_path)
                inspector.trigger_box = trig_box
                inspector.reset_counters()
                cv2.setTrackbarMax("Timeline", win_name, max(1, total_frames - 1))
                cv2.setTrackbarPos("Timeline", win_name, 0)
                print(f"\n[OK] Berpindah ke video [{next_idx+1}/{len(playlist)}]: {os.path.basename(video_path)} ({roi_desc})")
        elif key in [ord('o'), ord('O'), ord('f'), ord('F')]:
            init_d = os.path.dirname(video_path) if (video_path and os.path.exists(video_path)) else os.getcwd()
            new_video = select_video_file(initial_dir=init_d, title="Pilih File Video Baru")
            if new_video and os.path.exists(new_video):
                cap.release()
                is_webcam = False
                current_cam_idx = None
                video_path = new_video
                if os.path.abspath(video_path) not in playlist:
                    playlist.append(os.path.abspath(video_path))
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                slat_polygon, trig_box, roi_desc = get_adaptive_slat_polygon(frame_w, frame_h, config_path)
                inspector.trigger_box = trig_box
                inspector.reset_counters()
                cv2.setTrackbarMax("Timeline", win_name, max(1, total_frames - 1))
                cv2.setTrackbarPos("Timeline", win_name, 0)
                print(f"\n[OK] Berhasil memuat video: {os.path.basename(video_path)} ({roi_desc})")
        elif key == ord('1'):
            speed_mult = 1.0
        elif key == ord('2'):
            speed_mult = 2.0
        elif key == ord('3'):
            speed_mult = 3.0
        elif key == ord('4'):
            speed_mult = 4.0
        elif key in [ord('+'), ord('=')]:
            inspector.max_flatness_pv = min(40.0, inspector.max_flatness_pv + 1.0)
            cv2.setTrackbarPos("Flatness PV", win_name, int(inspector.max_flatness_pv))
            print(f"[*] Toleransi Flatness PV dinaikkan: {inspector.max_flatness_pv:.1f} px")
        elif key in [ord('-'), ord('_')]:
            inspector.max_flatness_pv = max(4.0, inspector.max_flatness_pv - 1.0)
            cv2.setTrackbarPos("Flatness PV", win_name, int(inspector.max_flatness_pv))
            print(f"[*] Toleransi Flatness PV diturunkan: {inspector.max_flatness_pv:.1f} px")
        elif key == ord(']'):
            inspector.max_sagitta = min(25.0, inspector.max_sagitta + 0.5)
            cv2.setTrackbarPos("Sagitta", win_name, int(inspector.max_sagitta * 2))
            print(f"[*] Toleransi Sagitta dinaikkan: {inspector.max_sagitta:.1f} px")
        elif key == ord('['):
            inspector.max_sagitta = max(2.0, inspector.max_sagitta - 0.5)
            cv2.setTrackbarPos("Sagitta", win_name, int(inspector.max_sagitta * 2))
            print(f"[*] Toleransi Sagitta diturunkan: {inspector.max_sagitta:.1f} px")
        elif key in [ord('s'), ord('S')]:
            # Simpan nilai ambang batas terkini langsung ke file konfigurasi JSON
            if config_path and os.path.exists(config_path):
                try:
                    with open(config_path, "r") as f:
                        cfg_data = json.load(f)
                    cfg_data.setdefault("thresholds", {})
                    cfg_data["thresholds"]["max_flatness_pv"] = round(inspector.max_flatness_pv, 1)
                    cfg_data["thresholds"]["max_sagitta"] = round(inspector.max_sagitta, 1)
                    cfg_data["thresholds"]["max_center_void"] = inspector.max_center_void
                    with open(config_path, "w") as f:
                        json.dump(cfg_data, f, indent=2)
                    print(f"\n[OK] 💾 Konfigurasi berhasil disimpan ke {os.path.basename(config_path)}:")
                    print(f"     -> max_flatness_pv : {inspector.max_flatness_pv:.1f} px")
                    print(f"     -> max_sagitta     : {inspector.max_sagitta:.1f} px")
                    print(f"     -> max_center_void : {inspector.max_center_void} px")
                except Exception as e:
                    print(f"\n[!] Gagal menyimpan konfigurasi: {e}")
        elif key == ord('d'):
            if not is_webcam:
                cur = cap.get(cv2.CAP_PROP_POS_FRAMES)
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(total_frames - 1, cur + 1))
                cv2.setTrackbarPos("Timeline", win_name, int(cur + 1))
        elif key == ord('a'):
            if not is_webcam:
                cur = cap.get(cv2.CAP_PROP_POS_FRAMES)
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, cur - 2))
                cv2.setTrackbarPos("Timeline", win_name, int(max(0, cur - 2)))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hough Line Stitching Straightness Inspector (Paper Sensors 2025)")
    parser.add_argument("video", nargs="?", default=None, help="Path file video konveyor (opsional)")
    parser.add_argument("--cam", "--webcam", dest="cam_index", type=int, default=None,
                        help="Indeks Webcam Live (misal: 0 untuk kamera laptop, 1 untuk USB eksternal)")
    parser.add_argument("--config", default=None, help="Path ke top_inspection_config.json")
    args = parser.parse_args()

    run_hough_stitching_demo(video_path=args.video, config_path=args.config, cam_index=args.cam_index)
