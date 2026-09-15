"""
Benchmark Preprocessing & Seleksi Filter Adaptif Multi-Environment
untuk Deteksi Kelurusan Slat & Rib Conveyor (Pure Computer Vision).

Fokus Peningkatan & Solusi Masalah:
  1. Solusi Plat Gelap / Berminyak (Oli/Karat):
     - Menghilangkan noise garis bergelombang seperti gunung ("riak gunung") akibat oli pekat.
     - Menggunakan Horizontal Morphological BlackHat + Directional Sobel-Y + Savitzky-Golay / RANSAC.
  2. Solusi Tampak Samping / Miring (CAM-02 / CAM-03):
     - Dual-Mode ROI: Mendukung Kotak Persegi (Tampak Depan) dan Poligon 4-Titik Perspektif (Tampak Samping).
     - Dilengkapi Perspective Warp Rectification (Homography Unwarp) sehingga benda kerja yang miring
       secara otomatis diluruskan menjadi persegi panjang horizontal sebelum pemindaian.
       Mengeliminasi false NG akibat sudut kemiringan kamera!
  3. Visualisasi Deteksi Garis Riil (Cyan) & Garis Acuan (Orange):
     - Membandingkan hasil profil garis nyata, deviasi P-V Flatness, Kehalusan (Smoothness), dan #Spikes.
  4. Ekspor Preset Otomatis:
     - Menghasilkan file 'filter_config_presets.py' & 'filter_config.json' siap pakai untuk program utama.
"""

import cv2
import numpy as np
import time
import sys
import os
import json

# Flush & Encode stdout secara aman untuk konsol Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass

# Dialog pemilih file Tkinter
try:
    import tkinter as tk
    from tkinter import filedialog
    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False

# Scipy untuk Savitzky-Golay filter
try:
    from scipy.signal import savgol_filter
    HAS_SAVGOL = True
except ImportError:
    HAS_SAVGOL = False


# ============================================================
#   UTILITAS STRING & PATH
# ============================================================
def clean_path_string(p):
    if not p:
        return ""
    p = p.strip()
    if p.startswith('&'):
        p = p[1:].strip()
    p = p.strip('\'"')
    if p.startswith('file:///'):
        p = p[8:]
    return os.path.normpath(p)


def choose_file_dialog(initial_dir=None):
    if not HAS_TKINTER:
        return None
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        f_path = filedialog.askopenfilename(
            title="Pilih Foto Slat / Rib Conveyor untuk Benchmark",
            initialdir=initial_dir if (initial_dir and os.path.exists(initial_dir)) else None,
            filetypes=[
                ("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.webp;*.tiff"),
                ("Semua File", "*.*")
            ]
        )
        root.destroy()
        return clean_path_string(f_path) if f_path else None
    except Exception:
        return None


# ============================================================
#   ROI DRAWER (Dual-Mode: Kotak Drag vs Poligon 4-Titik Unwarp)
# ============================================================
class ROIDrawerDualMode:
    """
    Antarmuka visual seleksi ROI dengan 2 Mode:
      Mode 1: Drag Kotak Persegi (Default untuk Tampak Depan CAM-01)
      Mode 2: Poligon 4-Titik dengan Perspective Unwarp (Untuk Tampak Samping CAM-02/CAM-03)
              Titik 1: Kiri-Atas, Titik 2: Kanan-Atas, Titik 3: Kanan-Bawah, Titik 4: Kiri-Bawah.
    """
    def __init__(self, window_name, image):
        self.window_name = window_name
        self.original = image.copy()
        self.h_orig, self.w_orig = image.shape[:2]
        
        # Scaling agar pas dengan resolusi layar monitor
        max_dim = 1000
        if max(self.h_orig, self.w_orig) > max_dim:
            self.scale = max_dim / max(self.h_orig, self.w_orig)
            self.display_base = cv2.resize(image, None, fx=self.scale, fy=self.scale)
        else:
            self.scale = 1.0
            self.display_base = image.copy()

        self.mode = "BOX"       # "BOX" atau "QUAD"
        self.drawing_box = False
        self.box_start = None
        self.box_end = None
        self.selected_box = None
        
        self.quad_points = []   # Titik poligon 4-titik (dalam koordinat display)
        self.confirmed_roi_data = None

    def mouse_callback(self, event, x, y, flags, param):
        if self.mode == "BOX":
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing_box = True
                self.box_start = (x, y)
                self.box_end = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and self.drawing_box:
                self.box_end = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                self.drawing_box = False
                self.box_end = (x, y)
                x1 = int(min(self.box_start[0], x) / self.scale)
                y1 = int(min(self.box_start[1], y) / self.scale)
                x2 = int(max(self.box_start[0], x) / self.scale)
                y2 = int(max(self.box_start[1], y) / self.scale)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(self.w_orig, x2), min(self.h_orig, y2)
                if (x2 - x1) > 25 and (y2 - y1) > 10:
                    self.selected_box = (x1, y1, x2, y2)
                else:
                    self.selected_box = None
        elif self.mode == "QUAD":
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(self.quad_points) < 4:
                    self.quad_points.append((x, y))
            elif event == cv2.EVENT_RBUTTONDOWN:
                if len(self.quad_points) > 0:
                    self.quad_points.pop()

    def render(self):
        canvas = self.display_base.copy()
        dh, dw = canvas.shape[:2]

        # Top Control Banner
        cv2.rectangle(canvas, (0, 0), (dw, 55), (20, 20, 20), -1)
        mode_str = "MODE: [1] KOTAK PERSEGI (Tampak Depan)" if self.mode == "BOX" else "MODE: [2] 4-TITIK PERSPEKTIF UNWARP (Tampak Samping)"
        cv2.putText(canvas, mode_str, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        help_str = "[M]: Ganti Mode | [ENTER/SPACE]: Konfirmasi | [C/R]: Reset | [Q/ESC]: Keluar"
        cv2.putText(canvas, help_str, (12, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # Render Mode BOX
        if self.mode == "BOX":
            if self.drawing_box and self.box_start and self.box_end:
                cv2.rectangle(canvas, self.box_start, self.box_end, (0, 255, 255), 2)
            elif self.selected_box:
                x1 = int(self.selected_box[0] * self.scale)
                y1 = int(self.selected_box[1] * self.scale)
                x2 = int(self.selected_box[2] * self.scale)
                y2 = int(self.selected_box[3] * self.scale)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(canvas, f"ROI: {self.selected_box[2]-self.selected_box[0]}x{self.selected_box[3]-self.selected_box[1]} px",
                            (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Render Mode QUAD
        else:
            labels = ["1. Kiri-Atas (TL)", "2. Kanan-Atas (TR)", "3. Kanan-Bawah (BR)", "4. Kiri-Bawah (BL)"]
            colors = [(0, 255, 255), (0, 200, 255), (0, 255, 0), (255, 100, 0)]
            n = len(self.quad_points)
            for i in range(n):
                pt = self.quad_points[i]
                cv2.circle(canvas, pt, 6, colors[i], -1)
                cv2.circle(canvas, pt, 8, (255, 255, 255), 1)
                cv2.putText(canvas, labels[i], (pt[0] + 10, pt[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors[i], 2)
                if i > 0:
                    cv2.line(canvas, self.quad_points[i - 1], pt, (0, 255, 0), 2)
            if n == 4:
                cv2.line(canvas, self.quad_points[3], self.quad_points[0], (0, 255, 0), 2)
                # Shading area poligon
                pts_arr = np.array(self.quad_points, dtype=np.int32)
                overlay = canvas.copy()
                cv2.fillPoly(overlay, [pts_arr], (0, 255, 120))
                cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)
                cv2.putText(canvas, "4 Titik Terpilih! Tekan [ENTER] untuk Unwarp & Benchmark",
                            (12, dh - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2)
            else:
                prompt = f"Klik Titik #{n+1}: {labels[n]} | Klik Kanan = Undo"
                cv2.putText(canvas, prompt, (12, dh - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 200, 255), 2)

        return canvas

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.display_base.shape[1], self.display_base.shape[0])
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        while True:
            display_img = self.render()
            cv2.imshow(self.window_name, display_img)
            key = cv2.waitKey(25) & 0xFF

            if key in [ord('m'), ord('M')]:
                self.mode = "QUAD" if self.mode == "BOX" else "BOX"
                self.selected_box = None
                self.quad_points = []
            elif key in [ord('c'), ord('C'), ord('r'), ord('R')]:
                self.selected_box = None
                self.quad_points = []
            elif key in [13, 32]:  # ENTER / SPACE
                if self.mode == "BOX" and self.selected_box is not None:
                    x1, y1, x2, y2 = self.selected_box
                    crop = self.original[y1:y2, x1:x2].copy()
                    return {
                        'type': 'BOX',
                        'roi_box': self.selected_box,
                        'crop': crop,
                        'is_unwarped': False,
                        'homography': None,
                        'inv_homography': None,
                        'pts_orig': None
                    }
                elif self.mode == "QUAD" and len(self.quad_points) == 4:
                    pts_orig = np.array([[pt[0] / self.scale, pt[1] / self.scale] for pt in self.quad_points], dtype=np.float32)
                    tl, tr, br, bl = pts_orig
                    w_top = np.hypot(tr[0] - tl[0], tr[1] - tl[1])
                    w_bot = np.hypot(br[0] - bl[0], br[1] - bl[1])
                    w_unwarp = int(max(w_top, w_bot))
                    h_left = np.hypot(bl[0] - tl[0], bl[1] - tl[1])
                    h_right = np.hypot(br[0] - tr[0], br[1] - tr[1])
                    h_unwarp = int(max(h_left, h_right))
                    h_unwarp = max(35, min(180, h_unwarp))
                    w_unwarp = max(100, w_unwarp)

                    dst_pts = np.array([
                        [0, 0],
                        [w_unwarp, 0],
                        [w_unwarp, h_unwarp],
                        [0, h_unwarp]
                    ], dtype=np.float32)

                    M = cv2.getPerspectiveTransform(pts_orig, dst_pts)
                    M_inv = cv2.getPerspectiveTransform(dst_pts, pts_orig)
                    unwarped = cv2.warpPerspective(self.original, M, (w_unwarp, h_unwarp))

                    return {
                        'type': 'QUAD_UNWARP',
                        'roi_box': None,
                        'crop': unwarped,
                        'is_unwarped': True,
                        'homography': M,
                        'inv_homography': M_inv,
                        'pts_orig': pts_orig,
                        'unwarp_size': (w_unwarp, h_unwarp)
                    }
            elif key in [ord('q'), ord('Q'), 27]:
                return None


# ============================================================
#   8 PREPROCESSING FILTER PIPELINES
# ============================================================
def run_all_filter_pipelines(crop_bgr):
    """
    Menjalankan 8 variasi pipeline filter preprocessing untuk menguji berbagai
    karakteristik fisik plat logam konveyor:
      1. [STD] Baseline Sobel V4.5
      2. [OILY] Dark Metal Morph BlackHat (Solusi Plat Gelap & Karat)
      3. [SMOOTH] Savitzky-Golay Robust Poly (Solusi Riak Gunung)
      4. [RANSAC] Parabolic RANSAC Filter (Solusi Outlier Ekstrem)
      5. [SCHARR] High-Sensitivity Scharr Gradient (Tepi Redup)
      6. [CANNY] Directional Canny Edge Detector
      7. [LOG] Laplacian of Gaussian (LoG)
      8. [ADAPT] Adaptive Local Contrast Normalizer
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    pipelines = []

    # ------------------------------------------------------------
    # 1. [STD] Baseline Sobel V4.5 (Standar Citra Terang)
    # ------------------------------------------------------------
    t0 = time.perf_counter()
    clahe_std = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    bilat_std = cv2.bilateralFilter(clahe_std, d=7, sigmaColor=50, sigmaSpace=50)
    sx = cv2.Sobel(bilat_std, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(bilat_std, cv2.CV_64F, 0, 1, ksize=3)
    mag1 = np.uint8(np.clip(cv2.magnitude(sx, sy), 0, 255))
    _, th1 = cv2.threshold(mag1, 35, 255, cv2.THRESH_BINARY)
    cl1 = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5)))
    t1 = (time.perf_counter() - t0) * 1000
    pipelines.append({
        'id': 'STD_SOBEL',
        'name': '1. Baseline Sobel V4.5',
        'desc': 'CLAHE 2.5 + Bilateral + Isotropic Sobel 35 (Standar Kamera Tampak Depan)',
        'mask': cl1,
        'time_ms': t1,
        'params': {'filter_type': 'sobel', 'clahe_clip': 2.5, 'bilateral_d': 7, 'thresh': 35, 'morph': (13, 5)}
    })

    # ------------------------------------------------------------
    # 2. [OILY] Dark Metal Morph BlackHat (Solusi Plat Gelap/Oli)
    # ------------------------------------------------------------
    t0 = time.perf_counter()
    clahe_oily = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)).apply(gray)
    bilat_oily = cv2.bilateralFilter(clahe_oily, d=9, sigmaColor=75, sigmaSpace=75)
    # BlackHat menonjolkan batas celah bayangan gelap di bawah genangan oli
    k_blackhat = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 7))
    blackhat = cv2.morphologyEx(bilat_oily, cv2.MORPH_BLACKHAT, k_blackhat)
    sobel_y = np.uint8(np.clip(np.abs(cv2.Sobel(blackhat, cv2.CV_64F, 0, 1, ksize=3)), 0, 255))
    _, th2 = cv2.threshold(sobel_y, 25, 255, cv2.THRESH_BINARY)
    cl2 = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5)))
    t2 = (time.perf_counter() - t0) * 1000
    pipelines.append({
        'id': 'OILY_BLACKHAT',
        'name': '2. Dark Metal Morph BlackHat',
        'desc': 'CLAHE 4.0 + Horizontal BlackHat 35x7 + Sobel-Y (Spesialis Plat Gelap & Oli)',
        'mask': cl2,
        'time_ms': t2,
        'params': {'filter_type': 'morph_blackhat', 'clahe_clip': 4.0, 'blackhat_kernel': (35, 7), 'thresh': 25, 'morph': (15, 5)}
    })

    # ------------------------------------------------------------
    # 3. [SMOOTH] Savitzky-Golay Robust Poly (Solusi Riak Gunung)
    # ------------------------------------------------------------
    t0 = time.perf_counter()
    # Horizontal Morph Open menghapus partikel debu/butiran oli vertikal
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    opened = cv2.morphologyEx(bilat_std, cv2.MORPH_OPEN, k_open)
    sobel_y3 = np.uint8(np.clip(np.abs(cv2.Sobel(opened, cv2.CV_64F, 0, 1, ksize=3)), 0, 255))
    _, th3 = cv2.threshold(sobel_y3, 28, 255, cv2.THRESH_BINARY)
    cl3 = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3)))
    t3 = (time.perf_counter() - t0) * 1000
    pipelines.append({
        'id': 'SMOOTH_SAVGOL',
        'name': '3. Savitzky-Golay Robust Poly',
        'desc': 'Horizontal Morph Open + Sobel-Y + Savitzky-Golay (Anti-Gunung / Mulus Sempurna)',
        'mask': cl3,
        'time_ms': t3,
        'post_smooth': 'savgol',
        'params': {'filter_type': 'savgol_poly', 'morph_open': (25, 3), 'thresh': 28, 'poly_order': 2, 'window_length': 21}
    })

    # ------------------------------------------------------------
    # 4. [RANSAC] Parabolic RANSAC Filter (Solusi Outlier Ekstrem)
    # ------------------------------------------------------------
    t0 = time.perf_counter()
    sobel_y4 = np.uint8(np.clip(np.abs(cv2.Sobel(bilat_std, cv2.CV_64F, 0, 1, ksize=3)), 0, 255))
    _, th4 = cv2.threshold(sobel_y4, 25, 255, cv2.THRESH_BINARY)
    cl4 = cv2.morphologyEx(th4, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)))
    t4 = (time.perf_counter() - t0) * 1000
    pipelines.append({
        'id': 'RANSAC_POLY',
        'name': '4. Parabolic RANSAC Filter',
        'desc': 'Sobel-Y + RANSAC Parabolic Curve Fitting (Kebal Partikel Besi & Pantulan)',
        'mask': cl4,
        'time_ms': t4,
        'post_smooth': 'ransac',
        'params': {'filter_type': 'ransac_poly', 'thresh': 25, 'ransac_residual': 3.5, 'degree': 2}
    })

    # ------------------------------------------------------------
    # 5. [SCHARR] High-Sensitivity Scharr Gradient (Tepi Redup)
    # ------------------------------------------------------------
    t0 = time.perf_counter()
    clahe_med = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    bilat_med = cv2.bilateralFilter(clahe_med, d=7, sigmaColor=60, sigmaSpace=60)
    scharr_y = np.uint8(np.clip(np.abs(cv2.Scharr(bilat_med, cv2.CV_64F, 0, 1)) / 2.0, 0, 255))
    _, th5 = cv2.threshold(scharr_y, 65, 255, cv2.THRESH_BINARY)
    cl5 = cv2.morphologyEx(th5, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 5)))
    t5 = (time.perf_counter() - t0) * 1000
    pipelines.append({
        'id': 'SCHARR_GRAD',
        'name': '5. High-Sensitivity Scharr',
        'desc': 'Scharr Gradient Y /2.0 Thresh 65 (Sensitif Terhadap Tepi Tipis / Kontras Rendah)',
        'mask': cl5,
        'time_ms': t5,
        'params': {'filter_type': 'scharr', 'thresh': 65, 'clahe_clip': 3.0, 'morph': (11, 5)}
    })

    # ------------------------------------------------------------
    # 6. [CANNY] Directional Canny Edge Detector
    # ------------------------------------------------------------
    t0 = time.perf_counter()
    v = np.median(bilat_std)
    sigma = 0.33
    lower_canny = int(max(0, (1.0 - sigma) * v))
    upper_canny = int(min(255, (1.0 + sigma) * v))
    canny_edges = cv2.Canny(bilat_std, lower_canny, upper_canny)
    cl6 = cv2.morphologyEx(canny_edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)))
    t6 = (time.perf_counter() - t0) * 1000
    pipelines.append({
        'id': 'CANNY_EDGE',
        'name': '6. Directional Canny Edge',
        'desc': f'Auto-Otsu Canny ({lower_canny}, {upper_canny}) + Morph Close Horizontal 9x3',
        'mask': cl6,
        'time_ms': t6,
        'params': {'filter_type': 'canny', 'lower': lower_canny, 'upper': upper_canny, 'morph': (9, 3)}
    })

    # ------------------------------------------------------------
    # 7. [LOG] Laplacian of Gaussian (LoG)
    # ------------------------------------------------------------
    t0 = time.perf_counter()
    gauss = cv2.GaussianBlur(gray, (5, 5), 0)
    lap = cv2.Laplacian(gauss, cv2.CV_64F, ksize=3)
    mag_lap = np.uint8(np.clip(np.abs(lap), 0, 255))
    _, th7 = cv2.threshold(mag_lap, 25, 255, cv2.THRESH_BINARY)
    cl7 = cv2.morphologyEx(th7, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5)))
    t7 = (time.perf_counter() - t0) * 1000
    pipelines.append({
        'id': 'LOG_LAPLACIAN',
        'name': '7. Laplacian of Gaussian (LoG)',
        'desc': 'Gaussian 5x5 + Laplacian Zero-Crossing Thresh 25 (Pendeteksi Tepi Orde 2)',
        'mask': cl7,
        'time_ms': t7,
        'params': {'filter_type': 'log', 'gaussian_k': 5, 'thresh': 25, 'morph': (7, 5)}
    })

    # ------------------------------------------------------------
    # 8. [ADAPT] Adaptive Local Contrast Normalizer
    # ------------------------------------------------------------
    t0 = time.perf_counter()
    k_local = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k_local)
    blackhat_loc = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k_local)
    enhanced_local = cv2.add(cv2.subtract(gray, blackhat_loc), tophat)
    sy8 = np.uint8(np.clip(np.abs(cv2.Sobel(enhanced_local, cv2.CV_64F, 0, 1, ksize=3)), 0, 255))
    _, th8 = cv2.threshold(sy8, 30, 255, cv2.THRESH_BINARY)
    cl8 = cv2.morphologyEx(th8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 5)))
    t8 = (time.perf_counter() - t0) * 1000
    pipelines.append({
        'id': 'ADAPT_CONTRAST',
        'name': '8. Adaptive Contrast Normalizer',
        'desc': 'Local TopHat/BlackHat Contrast Fusion + Sobel-Y (Penyeimbang Area Belang)',
        'mask': cl8,
        'time_ms': t8,
        'params': {'filter_type': 'adaptive_contrast', 'local_kernel': (21, 5), 'thresh': 30, 'morph': (11, 5)}
    })

    return pipelines


# ============================================================
#   PROFILE EXTRACTION & METROLOGY EVALUATOR
# ============================================================
def extract_and_evaluate_profile(pipeline_entry, crop_shape):
    """
    Mengekstrak kurva profil tepi fisik y(x) dari mask biner dan mengevaluasi
    kualitas metrologi:
      - Continuity (% bentang terlacak)
      - Smoothness (kehalusan kontur fisik)
      - Spike / Mountain Count (jumlah lonjakan bergerigi tajam)
      - P-V Flatness (selisih defleksi puncak-lembah)
      - Composite Metrology Score (0 - 100)
    """
    b_mask = pipeline_entry['mask']
    post_smooth = pipeline_entry.get('post_smooth', None)
    h, w = crop_shape[:2]

    step_x = 2
    x_coords = []
    raw_y_coords = []

    # 1. Pindai kolom per kolom untuk mencari tepi teratas
    for x in range(0, w, step_x):
        idx = np.where(b_mask[:, x] > 0)[0]
        if len(idx) > 0:
            x_coords.append(x)
            raw_y_coords.append(float(idx[0]))

    n_det = len(x_coords)
    total_steps = len(range(0, w, step_x))

    if n_det < 15:
        return {
            'success': False,
            'continuity': 0.0,
            'smoothness': 0.0,
            'spikes': 999,
            'pv_flatness': 999.0,
            'max_dev': 999.0,
            'score': 0.0,
            'x_coords': np.array([]),
            'y_profile': np.array([]),
            'baseline': np.array([]),
            'deviations': np.array([])
        }

    x_arr = np.array(x_coords, dtype=np.float32)
    y_arr = np.array(raw_y_coords, dtype=np.float32)

    # 2. Pembersihan Outlier Ringan dengan Median Moving Window
    # Interpolasi jika ada celah kecil
    full_x = np.arange(0, w, step_x, dtype=np.float32)
    interp_y = np.interp(full_x, x_arr, y_arr)

    # 3. Post-Processing Khusus Pipeline
    if post_smooth == 'savgol' and HAS_SAVGOL and len(interp_y) > 21:
        # Savitzky-Golay regularizer menghapus riak gunung
        smooth_y = savgol_filter(interp_y, window_length=21, polyorder=2)
    elif post_smooth == 'ransac' and len(x_arr) > 20:
        # RANSAC Parabolic Curve Fitting
        try:
            from sklearn.linear_model import RANSACRegressor
            from sklearn.preprocessing import PolynomialFeatures
            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(x_arr.reshape(-1, 1))
            ransac = RANSACRegressor(residual_threshold=3.5, random_state=42)
            ransac.fit(X_poly, y_arr)
            smooth_y = ransac.predict(poly.transform(full_x.reshape(-1, 1)))
        except Exception:
            smooth_y = interp_y
    else:
        # Standar Industrial Median + Gaussian Filter
        k_med = 9
        pad = k_med // 2
        padded = np.pad(interp_y, (pad, pad), mode='edge')
        med = np.array([np.median(padded[i:i + k_med]) for i in range(len(interp_y))])
        kernel_s = np.array([1, 2, 4, 2, 1], dtype=np.float32)
        kernel_s /= kernel_s.sum()
        smooth_y = np.convolve(med, kernel_s, mode='same')
        smooth_y[:2] = med[:2]
        smooth_y[-2:] = med[-2:]

    # 4. Garis Acuan Ideal (Linear Chord dari Ujung ke Ujung)
    # Menggunakan regresi linear ujung bentang
    trim = max(3, int(0.05 * len(full_x)))
    x_valid = full_x[trim:-trim]
    y_valid = smooth_y[trim:-trim]

    fit_poly = np.polyfit(x_valid, y_valid, 1)
    baseline_y = np.polyval(fit_poly, full_x)
    deviations = smooth_y - baseline_y

    dev_valid = deviations[trim:-trim]
    pv_flatness = float(np.max(dev_valid) - np.min(dev_valid))
    max_dev = float(np.max(np.abs(dev_valid)))

    # 5. Perhitungan Metrik Kualitas (Smoothness & Spikes)
    # Turunan kedua d^2y/dx^2: semakin kecil variansinya, semakin bebas dari riak "gunung"
    d2 = np.diff(smooth_y, n=2)
    var_d2 = float(np.var(d2))
    smoothness = max(0.0, min(100.0, 100.0 / (1.0 + var_d2 * 0.25)))

    # Hitung jumlah riak gunung (spikes): osilasi lokal curam > 4.5 px
    spikes = int(np.sum(np.abs(d2) > 4.5))

    # Continuity: persentase bentang yang berhasil dilacak secara riil
    continuity = (n_det / total_steps) * 100.0

    # Composite Quality Score (0 - 100)
    # Filter juara harus: Kontinuitas tinggi, Halus (bebas riak gunung), dan Minim loncatan liar
    penalty = min(35.0, spikes * 3.5)
    score = (0.35 * continuity) + (0.50 * smoothness) + (0.15 * max(0, 100 - var_d2 * 2)) - penalty
    score = max(0.0, min(100.0, score))

    return {
        'success': True,
        'continuity': continuity,
        'smoothness': smoothness,
        'spikes': spikes,
        'pv_flatness': pv_flatness,
        'max_dev': max_dev,
        'score': score,
        'x_coords': full_x,
        'y_profile': smooth_y,
        'baseline': baseline_y,
        'deviations': deviations,
        'trim': trim
    }


# ============================================================
#   VISUAL COMPOSITOR & HUD OVERLAY
# ============================================================
def render_tile_overlay(crop_bgr, b_mask, eval_data, title, badge, is_selected=False):
    """
    Merender panel visual tunggal yang menampilkan:
      - Citra asli benda kerja
      - Garis deteksi CYAN (profil nyata)
      - Garis acuan ORANGE (chord ideal)
      - Titik defleksi maksimum (MAGENTA)
      - Status OK/NG dan skor kuantitatif
    """
    h_c, w_c = crop_bgr.shape[:2]
    canvas = crop_bgr.copy()

    # Overlay garis profil jika evaluasi sukses
    if eval_data['success']:
        x_pts = eval_data['x_coords']
        y_pts = eval_data['y_profile']
        base_y = eval_data['baseline']
        trim = eval_data['trim']

        # Garis acuan ideal (ORANGE)
        pt_base_start = (int(x_pts[trim]), int(base_y[trim]))
        pt_base_end = (int(x_pts[-trim]), int(base_y[-trim]))
        cv2.line(canvas, pt_base_start, pt_base_end, (0, 140, 255), 2, cv2.LINE_AA)

        # Garis deteksi tepi riil (CYAN)
        pts_curve = np.column_stack((x_pts[trim:-trim].astype(np.int32), y_pts[trim:-trim].astype(np.int32)))
        cv2.polylines(canvas, [pts_curve], isClosed=False, color=(255, 255, 0), thickness=2, lineType=cv2.LINE_AA)

        # Titik defleksi maksimum (MAGENTA)
        dev_valid = eval_data['deviations'][trim:-trim]
        idx_max = int(np.argmax(np.abs(dev_valid))) + trim
        max_x = int(x_pts[idx_max])
        max_y = int(y_pts[idx_max])
        max_val = eval_data['deviations'][idx_max]

        cv2.circle(canvas, (max_x, max_y), 5, (255, 0, 255), -1)
        cv2.line(canvas, (max_x, int(base_y[idx_max])), (max_x, max_y), (255, 0, 255), 2, cv2.LINE_AA)
        sign = "+" if max_val >= 0 else ""
        cv2.putText(canvas, f"{sign}{max_val:.1f}px", (max_x - 20, max(18, max_y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 0, 255), 1, cv2.LINE_AA)

    # Resize ke dimensi tile seragam
    tw, th = 380, 155
    tile = cv2.resize(canvas, (tw, th))

    # Banner Header
    cv2.rectangle(tile, (0, 0), (tw, 26), (15, 15, 15), -1)
    badge_col = (0, 255, 255) if "1" in badge else ((0, 220, 100) if "2" in badge else (200, 200, 200))
    cv2.putText(tile, f"{badge} {title[:28]}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44, badge_col, 1, cv2.LINE_AA)

    # Banner Footer dengan Skor Metrologi
    cv2.rectangle(tile, (0, th - 26), (tw, th), (20, 20, 20), -1)
    if eval_data['success']:
        sc_col = (0, 255, 0) if eval_data['score'] >= 80 else ((0, 200, 255) if eval_data['score'] >= 60 else (0, 0, 255))
        stats_str = f"Skor: {eval_data['score']:.1f} | P-V: {eval_data['pv_flatness']:.1f}px | Halus: {eval_data['smoothness']:.0f}"
        cv2.putText(tile, stats_str, (8, th - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, sc_col, 1, cv2.LINE_AA)
    else:
        cv2.putText(tile, "Gagal Melacak Garis Tepi", (8, th - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 255), 1)

    # Border highlight jika terpilih
    if is_selected:
        cv2.rectangle(tile, (0, 0), (tw - 1, th - 1), (0, 255, 0), 2)

    return tile


def render_tile_mask(b_mask, title, badge):
    """Merender tile biner hitam-putih untuk memperlihatkan kebersihan filter."""
    tw, th = 380, 155
    bgr_mask = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
    tile = cv2.resize(bgr_mask, (tw, th))
    cv2.rectangle(tile, (0, 0), (tw, 24), (15, 15, 15), -1)
    cv2.putText(tile, f"{badge} Mask Biner ({title[:20]})", (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
    return tile


def get_sibling_images(current_path):
    """Mendapatkan seluruh file gambar yang ada di folder yang sama untuk navigasi cepat."""
    if not current_path or not os.path.exists(current_path):
        return []
    folder = os.path.dirname(os.path.abspath(current_path))
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
    try:
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]
        files.sort()
        return files
    except Exception:
        return [current_path]


def build_top4_dashboard(crop_bgr, benchmark_results, is_unwarped=False, selected_idx=0):
    """
    Membangun tampilan perbandingan visual Grid Top 4 Filter Terbaik:
      Baris 1: [Foto Asli / Unwarp]  |  [Juara 1: Overlay Garis]  |  [Juara 1: Mask Biner]
      Baris 2: [Juara 2: Overlay Garis]  |  [Juara 3: Overlay Garis]  |  [Juara 4: Overlay Garis]
    """
    tw, th = 380, 155
    top4 = benchmark_results[:4]
    badges = ["[1] JUARA 1", "[2] PERINGKAT 2", "[3] PERINGKAT 3", "[4] PERINGKAT 4"]

    # Tile Asli
    orig_resized = cv2.resize(crop_bgr, (tw, th))
    cv2.rectangle(orig_resized, (0, 0), (tw, 26), (15, 15, 15), -1)
    label_orig = "[FOTO] Unwarped Flat" if is_unwarped else "[FOTO] Crop Asli"
    cv2.putText(orig_resized, label_orig, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 255), 1)
    cv2.rectangle(orig_resized, (0, th - 26), (tw, th), (20, 20, 20), -1)
    cv2.putText(orig_resized, "Benda Kerja Tanpa Filter", (8, th - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)

    # Top 1 Overlay & Mask
    top1_res = top4[0]
    tile_top1_overlay = render_tile_overlay(crop_bgr, top1_res['mask'], top1_res['eval'], top1_res['name'], badges[0], is_selected=(selected_idx == 0))
    tile_top1_mask = render_tile_mask(top1_res['mask'], top1_res['name'], "[MASK]")

    # Top 2, 3, 4 Overlays
    tile_top2_overlay = render_tile_overlay(crop_bgr, top4[1]['mask'], top4[1]['eval'], top4[1]['name'], badges[1], is_selected=(selected_idx == 1))
    tile_top3_overlay = render_tile_overlay(crop_bgr, top4[2]['mask'], top4[2]['eval'], top4[2]['name'], badges[2], is_selected=(selected_idx == 2))
    tile_top4_overlay = render_tile_overlay(crop_bgr, top4[3]['mask'], top4[3]['eval'], top4[3]['name'], badges[3], is_selected=(selected_idx == 3))

    row1 = np.hstack([orig_resized, tile_top1_overlay, tile_top1_mask])
    row2 = np.hstack([tile_top2_overlay, tile_top3_overlay, tile_top4_overlay])
    grid = np.vstack([row1, row2])

    # Footer Controller HUD
    fh, fw = 42, grid.shape[1]
    footer = np.zeros((fh, fw, 3), dtype=np.uint8)
    ctrl_text = "[1-8]: Pilih Filter | [ENTER / S]: Tetapkan Filter Ini | [R]: Ulangi ROI | [N/P]: Foto Lain | [O]: Buka Folder | [Q]: Keluar"
    cv2.putText(footer, ctrl_text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 255, 255), 1, cv2.LINE_AA)

    return np.vstack([grid, footer])


def render_single_detail_view(crop_bgr, entry, is_unwarped=False):
    """
    Tampilan resolusi tinggi detail untuk filter tertentu:
      - Panel atas: Gambar resolusi penuh dengan garis Cyan, Orange, dan titik Max Deviasi
      - Panel tengah: Mask biner filter
      - Panel bawah: Grafik profil defleksi (px) sepanjang bentang konveyor
    """
    eval_data = entry['eval']
    h_c, w_c = crop_bgr.shape[:2]
    
    # Target lebar window detail
    dw = 960
    scale = dw / max(w_c, 1)
    dh = max(180, int(h_c * scale))
    scaled_crop = cv2.resize(crop_bgr, (dw, dh))
    
    if eval_data['success']:
        x_pts = (eval_data['x_coords'] * scale).astype(np.int32)
        y_pts = (eval_data['y_profile'] * scale).astype(np.int32)
        base_y = (eval_data['baseline'] * scale).astype(np.int32)
        trim = eval_data['trim']

        # Baseline
        cv2.line(scaled_crop, (x_pts[trim], base_y[trim]), (x_pts[-trim], base_y[-trim]), (0, 140, 255), 2, cv2.LINE_AA)
        
        # Detected curve
        curve = np.column_stack((x_pts[trim:-trim], y_pts[trim:-trim]))
        cv2.polylines(scaled_crop, [curve], False, (255, 255, 0), 2, cv2.LINE_AA)

        # Max deflection
        dev_valid = eval_data['deviations'][trim:-trim]
        idx_max = int(np.argmax(np.abs(dev_valid))) + trim
        mx = x_pts[idx_max]
        my = y_pts[idx_max]
        cv2.circle(scaled_crop, (mx, my), 6, (255, 0, 255), -1)
        cv2.line(scaled_crop, (mx, base_y[idx_max]), (mx, my), (255, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(scaled_crop, f"Max: {eval_data['deviations'][idx_max]:+.1f} px", (mx - 30, max(25, my - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2, cv2.LINE_AA)

    # Header Panel Atas
    header = np.zeros((48, dw, 3), dtype=np.uint8)
    cv2.putText(header, f"DETAIL INSPEKSI FILTER: {entry['name']}", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2)
    cv2.putText(header, entry['desc'], (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

    # Mask Biner Strip
    mask_h = 75
    mask_bgr = cv2.cvtColor(entry['mask'], cv2.COLOR_GRAY2BGR)
    scaled_mask = cv2.resize(mask_bgr, (dw, mask_h))
    cv2.rectangle(scaled_mask, (0, 0), (dw, 20), (20, 20, 20), -1)
    cv2.putText(scaled_mask, "Binary Edge Mask (Hasil Preprocessing)", (8, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1)

    # Grafik Defleksi (Plotting Pure OpenCV)
    graph_h = 130
    graph = np.zeros((graph_h, dw, 3), dtype=np.uint8)
    graph[:] = (25, 25, 25)
    zero_y = graph_h // 2
    cv2.line(graph, (0, zero_y), (dw, zero_y), (80, 80, 80), 1)
    cv2.putText(graph, "0 px (Garis Acuan)", (8, zero_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)

    if eval_data['success']:
        dev = eval_data['deviations']
        max_abs = max(10.0, float(np.max(np.abs(dev))))
        pts_graph = []
        for i in range(len(dev)):
            gx = int(eval_data['x_coords'][i] * scale)
            gy = int(zero_y - (dev[i] / max_abs) * (graph_h * 0.40))
            gy = max(10, min(graph_h - 10, gy))
            pts_graph.append((gx, gy))
        if len(pts_graph) > 1:
            cv2.polylines(graph, [np.array(pts_graph, dtype=np.int32)], False, (0, 255, 255), 2, cv2.LINE_AA)

        stat_info = f"Flatness (P-V): {eval_data['pv_flatness']:.1f} px | Kehalusan: {eval_data['smoothness']:.1f} | Riak: {eval_data['spikes']} | Skor: {eval_data['score']:.1f}"
        cv2.putText(graph, stat_info, (12, graph_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 0), 1)

    # Bottom Help Banner
    bot_help = np.zeros((36, dw, 3), dtype=np.uint8)
    ctrl_str = "[ENTER / S]: Tetapkan Filter Ini | [B]: Kembali ke Top 4 | [R]: Ulangi ROI | [Q]: Keluar"
    cv2.putText(bot_help, ctrl_str, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    return np.vstack([header, scaled_crop, scaled_mask, graph, bot_help])


# ============================================================
#   EKSPOR PRESET FILE OTOMATIS
# ============================================================
def export_presets_to_code(top_entry, all_ranked, env_name="AUTO_BENCHMARK"):
    """
    Menyimpan parameter filter juara ke file:
      1. Program AI/filter_config_presets.py (Python code siap import)
      2. Program AI/filter_config.json (JSON format)
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    py_path = os.path.join(base_dir, "filter_config_presets.py")
    json_path = os.path.join(base_dir, "filter_config.json")

    export_dict = {
        "ENV_NAME": env_name,
        "BEST_FILTER_NAME": top_entry['name'],
        "SCORE": round(top_entry['score'], 2),
        "SMOOTHNESS": round(top_entry['eval']['smoothness'], 2),
        "PV_FLATNESS": round(top_entry['eval']['pv_flatness'], 2),
        "SPIKES_COUNT": top_entry['eval']['spikes'],
        "EXEC_TIME_MS": round(top_entry['time_ms'], 2),
        "PARAMETERS": top_entry['params']
    }

    # Tulis JSON
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(export_dict, f, indent=2)
    except Exception as e:
        print(f"[WARN] Gagal menulis filter_config.json: {e}")

    # Tulis Python Code
    py_code = f'''# =========================================================================
# CONFIG PRESETS FILTER OTOMATIS (Hasil Benchmark Preprocessing)
# Generated at: {time.strftime("%Y-%m-%d %H:%M:%S")}
# =========================================================================

# Juara 1: {top_entry['name']}
# Skor Kualitas: {top_entry['score']:.1f}/100 | Kehalusan: {top_entry['eval']['smoothness']:.1f} | Riak: {top_entry['eval']['spikes']}
FILTER_CONFIG_OPTIMAL = {repr(top_entry['params'])}

# Preset Berdasarkan Environment:
PRESET_FRONT_NORMAL = {{
    "filter_type": "sobel",
    "clahe_clip": 2.5,
    "bilateral_d": 7,
    "thresh": 35,
    "morph": (13, 5),
    "smooth_method": "median_convolve"
}}

PRESET_FRONT_DARK_OILY = {{
    "filter_type": "morph_blackhat",
    "clahe_clip": 4.0,
    "blackhat_kernel": (35, 7),
    "thresh": 25,
    "morph": (15, 5),
    "smooth_method": "savgol_poly",
    "poly_order": 2,
    "window_length": 21
}}

PRESET_SIDE_PERSPECTIVE = {{
    "use_perspective_unwarp": True,
    "filter_type": "sobel_directional_y",
    "clahe_clip": 3.0,
    "bilateral_d": 7,
    "thresh": 28,
    "morph": (15, 3),
    "smooth_method": "median_convolve"
}}


if __name__ == "__main__":
    import sys
    import os
    import subprocess

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print("=" * 76)
    print("  [INFO] FILTER CONFIG PRESETS - Slat & Rib Conveyor AI")
    print("=" * 76)
    print("  File ini adalah KAMUS PRESET KONFIGURASI parameter filter.")
    print("  Digunakan sebagai modul konfigurasi untuk:")
    print("    - AI_Slat_Rib_Conveyor_V4_7_Robust_Profile_W_db_OK.py")
    print("    - api.py")
    print("-" * 76)
    print("  [PRESET AKTIF]")
    print(f"  * Juara 1 Optimal     : {{FILTER_CONFIG_OPTIMAL}}")
    print(f"  * Front View Normal   : {{PRESET_FRONT_NORMAL}}")
    print(f"  * Front View Gelap/Oli: {{PRESET_FRONT_DARK_OILY}}")
    print(f"  * Side View Miring    : {{PRESET_SIDE_PERSPECTIVE}}")
    print("=" * 76)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bench_script = os.path.join(script_dir, "Benchmark_Filter_Slat_Rib.py")

    print("\\n[PETUNJUK] Untuk menjalankan PENGUJIAN VISUAL & BENCHMARK INTERAKTIF:")
    print(f"   Jalankan: python \\"{{bench_script}}\\"\\n")

    try:
        ans = input(">> Buka modul Benchmark_Filter_Slat_Rib sekarang? [Y/n]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        ans = "n"

    if ans in ["", "y", "ya", "yes"]:
        print("\\n>> Membuka Benchmark Filter Slat & Rib...\\n")
        subprocess.run([sys.executable, bench_script])
'''

    try:
        with open(py_path, 'w', encoding='utf-8') as f:
            f.write(py_code)
        print(f"\n[INFO] Preset Konfigurasi Filter Berhasil Disimpan!")
        print(f"       -> {py_path}")
        print(f"       -> {json_path}")
    except Exception as e:
        print(f"[WARN] Gagal menulis filter_config_presets.py: {e}")


# ============================================================
#   MAIN BENCHMARK SUITE EXECUTION
# ============================================================
def main():
    print("=" * 82)
    print("  [BENCHMARK] FILTER PREPROCESSING - Slat & Rib Conveyor AI")
    print("  Alur: 1. Pilih Foto dari Folder -> 2. Gambar ROI -> 3. Pilih Filter Paling Sesuai")
    print("=" * 82)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    initial_folder = os.path.join(script_dir, "test_samples")
    if not os.path.exists(initial_folder):
        initial_folder = script_dir

    current_img_path = None
    if len(sys.argv) > 1:
        current_img_path = clean_path_string(sys.argv[1])

    # 1. LANGSUNG BUKA DIALOG PEMILIH FILE DARI FOLDER
    if not current_img_path or not os.path.exists(current_img_path):
        print("\n[INFO] Membuka jendela File Explorer untuk memilih foto dari folder...")
        current_img_path = choose_file_dialog(initial_dir=initial_folder)

    if not current_img_path or not os.path.exists(current_img_path):
        print("-" * 55)
        try:
            inp = input(">> Masukkan / Drag & Drop path foto gambar di sini (atau tekan ENTER untuk keluar): ").strip()
            current_img_path = clean_path_string(inp)
        except (EOFError, KeyboardInterrupt):
            current_img_path = None
        print("-" * 55)

    if not current_img_path or not os.path.exists(current_img_path):
        print("[INFO] Tidak ada foto yang dipilih. Program benchmark ditutup.")
        return

    # Ambil seluruh daftar foto di folder yang sama untuk navigasi cepat
    sibling_images = get_sibling_images(current_img_path)
    img_idx = sibling_images.index(current_img_path) if current_img_path in sibling_images else 0

    # Loop Pengujian Gambar
    while True:
        original_img = cv2.imread(current_img_path)
        if original_img is None:
            print(f"[ERROR] Gagal memuat gambar: {current_img_path}")
            break

        fname = os.path.basename(current_img_path)
        h_img, w_img = original_img.shape[:2]
        print(f"\n[INFO] Foto Terbuka: {fname} ({w_img}x{h_img} px) [{img_idx + 1}/{len(sibling_images)}]")

        # 2. GAMBAR ROI PADA FOTO
        print("-" * 72)
        print("  Silakan Gambar Area ROI pada Benda Kerja:")
        print("    * Mode KOTAK   : Drag kotak melintang pada plat/sirip (Default)")
        print("    * Mode 4-TITIK : Tekan [M] lalu klik 4 sudut plat miring (Unwarp)")
        print("    * Tekan [ENTER] atau [SPACE] untuk selesai.")
        print("-" * 72)

        drawer = ROIDrawerDualMode(f"1. Gambar ROI - [{fname}] ({img_idx+1}/{len(sibling_images)})", original_img)
        roi_data = drawer.run()
        cv2.destroyAllWindows()

        if roi_data is None:
            print("[INFO] Seleksi ROI dibatalkan.")
            break

        crop = roi_data['crop']
        is_unwarped = roi_data['is_unwarped']
        ch, cw = crop.shape[:2]
        print(f"[INFO] ROI Berhasil Diekstrak: {cw}x{ch} px | Unwarped: {is_unwarped}")

        # 3. JALANKAN 8 PIPELINE PREPROCESSING & EVALUASI
        print(f"\n{'='*78}")
        print(f"  MENGEVALUASI 8 FILTER PREPROCESSING...")
        print(f"{'='*78}\n")

        pipelines = run_all_filter_pipelines(crop)
        benchmark_results = []

        for p in pipelines:
            print(f"  [*] Menguji: {p['name']:<32} ...", end=" ")
            t_eval_0 = time.perf_counter()
            eval_metrics = extract_and_evaluate_profile(p, crop.shape)
            t_eval_ms = (time.perf_counter() - t_eval_0) * 1000

            res_entry = {
                'id': p['id'],
                'name': p['name'],
                'desc': p['desc'],
                'mask': p['mask'],
                'time_ms': p['time_ms'] + t_eval_ms,
                'params': p['params'],
                'eval': eval_metrics,
                'score': eval_metrics['score']
            }
            benchmark_results.append(res_entry)
            print(f"Skor: {eval_metrics['score']:5.1f} | Halus: {eval_metrics['smoothness']:5.1f} | Riak: {eval_metrics['spikes']:2d} | [{p['time_ms']:.1f}ms]")

        # Urutkan berdasarkan Skor Tertinggi
        benchmark_results.sort(key=lambda x: x['score'], reverse=True)

        # Cetak Leaderboard di Terminal
        print(f"\n\n{'='*92}")
        print(f"  HASIL EVALUASI FILTER - Peringkat Kualitas Metrologi:")
        print(f"{'='*92}")
        print(f"  {'Rank':<5}{'Pipeline Preprocessing':<36}{'Kontinuitas':<13}{'Kehalusan':<11}{'Riak/Spike':<12}{'Flatness':<10}{'Skor':<6}")
        print(f"  {'-'*92}")

        for rank, b in enumerate(benchmark_results, start=1):
            star = " 🌟 #1" if rank == 1 else (" **" if rank <= 3 else "")
            ev = b['eval']
            print(f"  {rank:<5}{b['name']:<36}{ev['continuity']:<6.1f}%      {ev['smoothness']:<6.1f}     {ev['spikes']:<12}{ev['pv_flatness']:<6.1f}px   {b['score']:<5.1f}{star}")

        # 4. JENDELA VISUAL & PEMILIHAN FILTER OLEH USER
        win_dash = f"2. Hasil Evaluasi & Pemilihan Filter - [{fname}]"
        cv2.namedWindow(win_dash, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_dash, 1180, 530)

        current_view = "DASHBOARD"
        selected_filter_idx = 0  # Default juara 1
        toast_message = ""
        toast_timer = 0
        user_action = None

        while True:
            if current_view == "DASHBOARD":
                disp = build_top4_dashboard(crop, benchmark_results, is_unwarped=is_unwarped, selected_idx=selected_filter_idx)
            else:
                disp = render_single_detail_view(crop, benchmark_results[selected_filter_idx], is_unwarped=is_unwarped)

            # Overlay Banner Sukses jika user baru saja menetapkan filter
            if toast_message and (time.time() - toast_timer < 3.0):
                tw_b = disp.shape[1]
                overlay = disp.copy()
                cv2.rectangle(overlay, (0, 0), (tw_b, 36), (0, 150, 0), -1)
                cv2.addWeighted(overlay, 0.85, disp, 0.15, 0, disp)
                cv2.putText(disp, toast_message, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(win_dash, disp)
            key = cv2.waitKey(30) & 0xFF

            if key in [ord('q'), ord('Q'), 27]:
                user_action = 'QUIT'
                break
            elif key in [13, ord('s'), ord('S')]:  # ENTER atau S: TERAPKAN & SIMPAN FILTER PILIHAN INI
                chosen = benchmark_results[selected_filter_idx]
                export_presets_to_code(chosen, benchmark_results, env_name=fname)
                toast_message = f"[TERPILIH] Filter '{chosen['name']}' Berhasil Disimpan & Diterapkan!"
                toast_timer = time.time()
                print(f"\n>>> [TERPILIH] Anda menetapkan filter: {chosen['name']}")
                print(f"    Parameter tersimpan ke filter_config_presets.py & filter_config.json!\n")
            elif ord('1') <= key <= ord('8'):
                chosen_rank = key - ord('0')
                if chosen_rank <= len(benchmark_results):
                    selected_filter_idx = chosen_rank - 1
                    current_view = "DETAIL"
            elif key in [ord('b'), ord('B')]:
                current_view = "DASHBOARD"
            elif key in [ord('r'), ord('R')]:
                user_action = 'REDRAW'
                break
            elif key in [ord('n'), ord('N')]:
                if len(sibling_images) > 1:
                    img_idx = (img_idx + 1) % len(sibling_images)
                    current_img_path = sibling_images[img_idx]
                    user_action = 'NAV'
                    break
            elif key in [ord('p'), ord('P')]:
                if len(sibling_images) > 1:
                    img_idx = (img_idx - 1) % len(sibling_images)
                    current_img_path = sibling_images[img_idx]
                    user_action = 'NAV'
                    break
            elif key in [ord('o'), ord('O')]:
                cv2.destroyAllWindows()
                new_f = choose_file_dialog(initial_dir=os.path.dirname(current_img_path))
                if new_f and os.path.exists(new_f):
                    current_img_path = new_f
                    sibling_images = get_sibling_images(current_img_path)
                    img_idx = sibling_images.index(current_img_path) if current_img_path in sibling_images else 0
                    user_action = 'NAV'
                    break

        cv2.destroyAllWindows()
        if user_action == 'QUIT':
            break
        elif user_action == 'REDRAW':
            continue
        elif user_action == 'NAV':
            continue

    print("\n[INFO] Benchmark Slat & Rib Conveyor Selesai.")


if __name__ == "__main__":
    main()
