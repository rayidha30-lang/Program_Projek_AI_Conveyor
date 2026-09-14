"""
Benchmark Preprocessing - Seleksi Pipeline & Parameter Filter Paling Efektif
untuk Deteksi Kelurusan Slat & Rib Conveyor.

Script ini menguji setiap variasi preprocessing (CLAHE, Bilateral, Sobel, Scharr, Canny, LoG, Adaptive)
secara TERPISAH dan BERTAHAP, lalu membandingkan hasilnya secara kuantitatif:
  1. Kontinuitas Garis Tepi (%)  -> Bebas dari garis putus-putus
  2. Kehalusan Garis (Smoothness)-> Minim riak/gerigi akibat karat (Second Derivative Variance)
  3. Ketahanan Outlier (#Outlier) -> Bebas dari loncatan liar debu/baut
  4. Waktu Proses (ms)           -> Kecepatan eksekusi untuk real-time conveyor

Output: Leaderboard peringkat, Rekomendasi Top 3, dan Dictionary Parameter
yang siap disalin langsung ke program utama (Tes_Garis_Acuan_PureCV_V4_w_Rib&Slat.py).
"""

import cv2
import numpy as np
import time
import sys
import os

# Pastikan output konsol ter-flush secara real-time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# Dialog pemilih file Tkinter
try:
    import tkinter as tk
    from tkinter import filedialog
    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False


# ============================================================
#   CONFIG DEFAULT
# ============================================================
CONFIG = {
    "CLAHE_CLIP_LIMIT": 2.5,
    "CLAHE_TILE_SIZE": (8, 8),
    "BILATERAL_D": 7,
    "BILATERAL_SIGMA_COLOR": 50,
    "BILATERAL_SIGMA_SPACE": 50,
    "SOBEL_KSIZE": 3,
    "SOBEL_THRESH": 35,
    "MORPH_KERNEL_SIZE": (7, 5),
    "CANNY_LOW": 50,
    "CANNY_HIGH": 150,
    "SCHARR_THRESH": 70,
    "LOG_KSIZE": 5,
    "LOG_THRESH": 25,
    "ADAPTIVE_BLOCK": 15,
    "ADAPTIVE_C": 3,
    "CORNER_TRIM_PCT": 0.06,
    "MEDIAN_K": 9,
    "SMOOTH_W": 7,
}


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


def choose_file_dialog():
    if not HAS_TKINTER:
        return None
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        f_path = filedialog.askopenfilename(
            title="Pilih Foto Slat / Rib Conveyor untuk Benchmark",
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
#   ROI DRAWER (Interaktif Drag Kotak / Poligon)
# ============================================================
class ROIDrawer:
    """
    Antarmuka visual untuk menandai area uji (ROI) pada gambar conveyor.
    Mendukung seleksi drag kotak persegi maupun klik sudut poligon.
    """
    def __init__(self, window_name, image):
        self.window_name = window_name
        self.original = image.copy()
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.roi = None
        self.scale = 1.0

        h, w = image.shape[:2]
        max_display = 960
        if max(h, w) > max_display:
            self.scale = max_display / max(h, w)
            self.display_img = cv2.resize(image, None, fx=self.scale, fy=self.scale)
        else:
            self.display_img = image.copy()
        self.display_clean = self.display_img.copy()

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.end_point = (x, y)
            self.display_img = self.display_clean.copy()
            cv2.rectangle(self.display_img, self.start_point, self.end_point, (0, 255, 255), 2)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
            x1 = int(min(self.start_point[0], x) / self.scale)
            y1 = int(min(self.start_point[1], y) / self.scale)
            x2 = int(max(self.start_point[0], x) / self.scale)
            y2 = int(max(self.start_point[1], y) / self.scale)
            h, w = self.original.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if (x2 - x1) > 20 and (y2 - y1) > 10:
                self.roi = (x1, y1, x2, y2)
                self.display_img = self.display_clean.copy()
                cv2.rectangle(self.display_img, self.start_point, self.end_point, (0, 255, 0), 2)
                print(f"  [ROI] Terpilih: ({x1},{y1}) -> ({x2},{y2}) = {x2-x1}x{y2-y1} px")
            else:
                print("  [ROI] Terlalu kecil! Silakan gambar kotak yang lebih lebar melintang pada plat.")
                self.roi = None

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.display_img.shape[1], self.display_img.shape[0])
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        overlay = self.display_img.copy()
        cv2.rectangle(overlay, (0, 0), (self.display_img.shape[1], 45), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.75, self.display_img, 0.25, 0, self.display_img)
        cv2.putText(self.display_img, "Drag Kotak ROI melintang di Slat/Sirip | Tekan [ENTER] = Selesai | [Q] = Batal",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1, cv2.LINE_AA)
        self.display_clean = self.display_img.copy()

        while True:
            cv2.imshow(self.window_name, self.display_img)
            key = cv2.waitKey(30) & 0xFF
            if key in [13, 32] and self.roi is not None:
                return self.roi
            elif key in [ord('q'), ord('Q'), 27]:
                return None


# ============================================================
#   PREPROCESSING VARIANTS (Pengujian Variasi Filter)
# ============================================================
def make_variants(crop, config=CONFIG):
    """
    Membuat 14 varian pipeline preprocessing untuk di-benchmark secara independen.
    Return: list of tuple (nama_varian, binary_mask, preproc_time_ms, parameter_dict)
    """
    variants = []
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # ------------------------------------------------------------
    # 1. Baseline: Grayscale + Gaussian Blur + Sobel Isotropik (Tanpa CLAHE)
    # ------------------------------------------------------------
    t0 = time.perf_counter()
    b1 = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(b1, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(b1, cv2.CV_64F, 0, 1, ksize=3)
    mag1 = np.uint8(np.clip(cv2.magnitude(gx, gy), 0, 255))
    _, th1 = cv2.threshold(mag1, config["SOBEL_THRESH"], 255, cv2.THRESH_BINARY)
    cl1 = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, config["MORPH_KERNEL_SIZE"]))
    t1 = (time.perf_counter() - t0) * 1000
    variants.append((
        "Classic Sobel (No CLAHE)",
        cl1, t1,
        {"pipeline": "Gaussian + Sobel", "thresh": config["SOBEL_THRESH"], "morph": config["MORPH_KERNEL_SIZE"]}
    ))

    # ------------------------------------------------------------
    # 2. Pipeline Utama: CLAHE + Bilateral + Sobel Isotropik (V4 Default)
    # ------------------------------------------------------------
    t0 = time.perf_counter()
    clahe_std = cv2.createCLAHE(clipLimit=config["CLAHE_CLIP_LIMIT"], tileGridSize=config["CLAHE_TILE_SIZE"]).apply(gray)
    bilat_std = cv2.bilateralFilter(clahe_std, d=config["BILATERAL_D"], sigmaColor=config["BILATERAL_SIGMA_COLOR"], sigmaSpace=config["BILATERAL_SIGMA_SPACE"])
    gx = cv2.Sobel(bilat_std, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(bilat_std, cv2.CV_64F, 0, 1, ksize=3)
    mag2 = np.uint8(np.clip(cv2.magnitude(gx, gy), 0, 255))
    _, th2 = cv2.threshold(mag2, config["SOBEL_THRESH"], 255, cv2.THRESH_BINARY)
    cl2 = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, config["MORPH_KERNEL_SIZE"]))
    t2 = (time.perf_counter() - t0) * 1000
    variants.append((
        "CLAHE(2.5)+Bilateral(d=7)+Sobel-35",
        cl2, t2,
        {"pipeline": "CLAHE+Bilateral+Sobel", "clip_limit": 2.5, "bilateral_d": 7, "thresh": 35, "morph": config["MORPH_KERNEL_SIZE"]}
    ))

    # ------------------------------------------------------------
    # 3. Variasi Parameter: Sobel Threshold Rendah (Sensitif thd Plat Gelap)
    # ------------------------------------------------------------
    t0 = time.perf_counter()
    _, th3 = cv2.threshold(mag2, 22, 255, cv2.THRESH_BINARY)
    cl3 = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, config["MORPH_KERNEL_SIZE"]))
    t3 = (time.perf_counter() - t0) * 1000 + t2
    variants.append((
        "CLAHE+Bilateral+Sobel-22 (Low Thresh)",
        cl3, t3,
        {"pipeline": "CLAHE+Bilateral+Sobel", "clip_limit": 2.5, "bilateral_d": 7, "thresh": 22, "morph": config["MORPH_KERNEL_SIZE"]}
    ))

    # ------------------------------------------------------------
    # 4. Variasi Parameter: Sobel Threshold Tinggi (Anti-Noise Karat)
    # ------------------------------------------------------------
    t0 = time.perf_counter()
    _, th4 = cv2.threshold(mag2, 48, 255, cv2.THRESH_BINARY)
    cl4 = cv2.morphologyEx(th4, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, config["MORPH_KERNEL_SIZE"]))
    t4 = (time.perf_counter() - t0) * 1000 + t2
    variants.append((
        "CLAHE+Bilateral+Sobel-48 (High Thresh)",
        cl4, t4,
        {"pipeline": "CLAHE+Bilateral+Sobel", "clip_limit": 2.5, "bilateral_d": 7, "thresh": 48, "morph": config["MORPH_KERNEL_SIZE"]}
    ))

    # ------------------------------------------------------------
    # 5. Variasi CLAHE Kuat (clip=4.0) untuk Karat Pekat
    # ------------------------------------------------------------
    t0 = time.perf_counter()
    clahe_hi = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)).apply(gray)
    bilat_hi = cv2.bilateralFilter(clahe_hi, d=9, sigmaColor=75, sigmaSpace=75)
    gx = cv2.Sobel(bilat_hi, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(bilat_hi, cv2.CV_64F, 0, 1, ksize=3)
    mag5 = np.uint8(np.clip(cv2.magnitude(gx, gy), 0, 255))
    _, th5 = cv2.threshold(mag5, 42, 255, cv2.THRESH_BINARY)
    cl5 = cv2.morphologyEx(th5, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5)))
    t5 = (time.perf_counter() - t0) * 1000
    variants.append((
        "CLAHE(4.0)+Bilateral(d=9)+Sobel-42",
        cl5, t5,
        {"pipeline": "CLAHE+Bilateral+Sobel", "clip_limit": 4.0, "bilateral_d": 9, "thresh": 42, "morph": (9, 5)}
    ))

    # ------------------------------------------------------------
    # 6. Variasi Morph Close Panjang (13x5) untuk Celah Terputus
    # ------------------------------------------------------------
    t0 = time.perf_counter()
    cl6 = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5)))
    t6 = (time.perf_counter() - t0) * 1000 + t2
    variants.append((
        "CLAHE+Bilateral+Sobel (Morph 13x5)",
        cl6, t6,
        {"pipeline": "CLAHE+Bilateral+Sobel", "clip_limit": 2.5, "thresh": 35, "morph": (13, 5)}
    ))

    # ------------------------------------------------------------
    # 7. Scharr Gradient Operator (Sensitivitas Tinggi)
    # ------------------------------------------------------------
    t0 = time.perf_counter()
    sx = cv2.Scharr(bilat_std, cv2.CV_64F, 1, 0)
    sy = cv2.Scharr(bilat_std, cv2.CV_64F, 0, 1)
    mag_scharr = np.uint8(np.clip(cv2.magnitude(sx, sy) / 2.0, 0, 255))
    _, th7 = cv2.threshold(mag_scharr, config["SCHARR_THRESH"], 255, cv2.THRESH_BINARY)
    cl7 = cv2.morphologyEx(th7, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5)))
    t7 = (time.perf_counter() - t0) * 1000
    variants.append((
        "Scharr Gradient (Thresh=70)",
        cl7, t7,
        {"pipeline": "CLAHE+Bilateral+Scharr", "operator": "Scharr", "thresh": 70, "morph": (7, 5)}
    ))

    # ------------------------------------------------------------
    # 8. Scharr Gradient Threshold Ketat (th=90)
    # ------------------------------------------------------------
    t0 = time.perf_counter()
    _, th8 = cv2.threshold(mag_scharr, 90, 255, cv2.THRESH_BINARY)
    cl8 = cv2.morphologyEx(th8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5)))
    t8 = (time.perf_counter() - t0) * 1000 + t7
    variants.append((
        "Scharr Gradient (Thresh=90)",
        cl8, t8,
        {"pipeline": "CLAHE+Bilateral+Scharr", "operator": "Scharr", "thresh": 90, "morph": (7, 5)}
    ))

    # ------------------------------------------------------------
    # 9. Canny Edge Detector Standar (50, 150)
    # ------------------------------------------------------------
    t0 = time.perf_counter()
    canny_std = cv2.Canny(bilat_std, config["CANNY_LOW"], config["CANNY_HIGH"])
    cl9 = cv2.morphologyEx(canny_std, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    t9 = (time.perf_counter() - t0) * 1000
    variants.append((
        "Canny Edge (50, 150)",
        cl9, t9,
        {"pipeline": "CLAHE+Bilateral+Canny", "canny_low": 50, "canny_high": 150, "morph": (5, 5)}
    ))

    # ------------------------------------------------------------
    # 10. Canny Edge Detector Sensitif (25, 80)
    # ------------------------------------------------------------
    t0 = time.perf_counter()
    canny_sens = cv2.Canny(bilat_std, 25, 80)
    cl10 = cv2.morphologyEx(canny_sens, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    t10 = (time.perf_counter() - t0) * 1000
    variants.append((
        "Canny Edge (25, 80) [Sensitif]",
        cl10, t10,
        {"pipeline": "CLAHE+Bilateral+Canny", "canny_low": 25, "canny_high": 80, "morph": (5, 5)}
    ))

    # ------------------------------------------------------------
    # 11. Laplacian of Gaussian (LoG)
    # ------------------------------------------------------------
    t0 = time.perf_counter()
    lap = cv2.Laplacian(bilat_std, cv2.CV_64F, ksize=3)
    mag_lap = np.uint8(np.clip(np.absolute(lap), 0, 255))
    _, th11 = cv2.threshold(mag_lap, config["LOG_THRESH"], 255, cv2.THRESH_BINARY)
    cl11 = cv2.morphologyEx(th11, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    t11 = (time.perf_counter() - t0) * 1000
    variants.append((
        "Laplacian of Gaussian (LoG)",
        cl11, t11,
        {"pipeline": "Laplacian of Gaussian", "thresh": config["LOG_THRESH"], "morph": (5, 5)}
    ))

    # ------------------------------------------------------------
    # 12. Adaptive Gaussian Thresholding
    # ------------------------------------------------------------
    t0 = time.perf_counter()
    adap = cv2.adaptiveThreshold(bilat_std, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, config["ADAPTIVE_BLOCK"], config["ADAPTIVE_C"])
    cl12 = cv2.morphologyEx(adap, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)))
    t12 = (time.perf_counter() - t0) * 1000
    variants.append((
        "Adaptive Gaussian (Block=15, C=3)",
        cl12, t12,
        {"pipeline": "Adaptive Gaussian", "block": 15, "C": 3, "morph": (5, 3)}
    ))

    return variants


# ============================================================
#   METROLOGY QUALITY EVALUATOR (Perhitungan Skor Kuantitatif)
# ============================================================
def evaluate_variant(binary_mask, is_horizontal=True):
    """
    Evaluasi kuantitatif kualitas hasil filter untuk metrologi kelurusan:
    - Continuity (%): persentase bentang yang berhasil dilacak
    - Smoothness: kehalusan garis (variansi second difference d^2y)
    - Outliers: jumlah loncatan tajam liar (> 8 px)
    - SNR: rasio kepresisian piksel tepi terhadap noise liar
    """
    h, w = binary_mask.shape[:2]
    raw_coords = []

    if is_horizontal:
        steps = list(range(0, w, 2))
        for col in steps:
            idx = np.where(binary_mask[:, col] > 0)[0]
            if len(idx) > 0:
                raw_coords.append((col, idx[0]))
    else:
        steps = list(range(0, h, 2))
        for row in steps:
            idx = np.where(binary_mask[row, :] > 0)[0]
            if len(idx) > 0:
                raw_coords.append((idx[0], row))

    n_det = len(raw_coords)
    total_steps = len(steps) if steps else 1

    if n_det < 10:
        return {
            'continuity': 0.0,
            'smoothness': 0.0,
            'outliers': 999,
            'snr_db': 0.0,
            'score': 0.0,
            'coords': []
        }

    # 1. Continuity (%): Kelengkapan garis
    continuity = (n_det / total_steps) * 100.0

    # 2. Smoothness: Kehalusan kurva (100 / (1 + 0.25 * var(d^2y)))
    vals = np.array([pt[1] if is_horizontal else pt[0] for pt in raw_coords], dtype=np.float32)
    if len(vals) >= 5:
        d2 = np.diff(vals, n=2)
        var_d2 = float(np.var(d2))
        smoothness = max(0.0, min(100.0, 100.0 / (1.0 + var_d2 * 0.25)))
    else:
        smoothness = 40.0
        var_d2 = 99.0

    # 3. Outlier Count: Loncatan tajam > 8 px antar titik bertetangga
    d1 = np.abs(np.diff(vals))
    outliers = int(np.sum(d1 > 8.0))

    # 4. Kebersihan Garis (SNR)
    total_white = np.count_nonzero(binary_mask)
    noise_white = max(0, total_white - n_det * 3)
    snr = 10.0 * np.log10((n_det + 1e-4) / (noise_white + 1e-4) + 1.0)
    snr_norm = max(0.0, min(100.0, snr * 12.0))

    # Skor Gabungan Terbobot (Composite Quality Score 0 - 100)
    penalty = min(20.0, outliers * 3.0)
    score = (0.42 * continuity) + (0.38 * smoothness) + (0.10 * snr_norm) - penalty
    score = max(0.0, min(100.0, score))

    return {
        'continuity': continuity,
        'smoothness': smoothness,
        'outliers': outliers,
        'snr_db': snr,
        'score': score,
        'coords': raw_coords
    }


# ============================================================
#   MAIN BENCHMARK EXECUTION & REPORTING
# ============================================================
def main():
    print("=" * 76)
    print("  BENCHMARK PREPROCESSING - Seleksi Pipeline & Parameter Paling Efektif")
    print("  Proyek Inspeksi Slat & Rib Conveyor (Pure Computer Vision)")
    print("=" * 76)

    # 1. Pilih Gambar
    img_path = None
    if len(sys.argv) > 1:
        img_path = clean_path_string(sys.argv[1])

    if not img_path or not os.path.exists(img_path):
        print("\n[INFO] Membuka dialog pemilih file gambar...")
        img_path = choose_file_dialog()

    if not img_path or not os.path.exists(img_path):
        print("-" * 55)
        try:
            inp = input("👉 Masukkan / Drag & Drop path foto gambar di sini: ")
            img_path = clean_path_string(inp)
        except (EOFError, KeyboardInterrupt):
            img_path = None
        print("-" * 55)

    if not img_path or not os.path.exists(img_path):
        print("[ERROR] Gambar tidak ditemukan. Benchmark dibatalkan.")
        return

    original = cv2.imread(img_path)
    if original is None:
        print(f"[ERROR] Gagal membuka gambar: {img_path}")
        return

    h, w = original.shape[:2]
    file_name = os.path.basename(img_path)
    print(f"\n[INFO] Foto Terbuka: {file_name} ({w}x{h} px)")

    # 2. Gambar Kotak ROI Interaktif
    print("[INFO] Silakan tandai area kotak ROI pada benda kerja (Slat atau Sirip)...")
    drawer = ROIDrawer(f"Gambar ROI untuk Benchmark - [{file_name}]", original)
    roi = drawer.run()
    cv2.destroyAllWindows()

    if roi is None:
        print("[INFO] Seleksi ROI dibatalkan oleh pengguna.")
        return

    x1, y1, x2, y2 = roi
    rw, rh = x2 - x1, y2 - y1
    print(f"[INFO] ROI terpilih: ({x1},{y1}) -> ({x2},{y2}) = {rw}x{rh} px")

    crop = original[y1:y2, x1:x2]
    is_horizontal = rw >= rh

    # 3. Jalankan Benchmark Varian Preprocessing
    print(f"\n{'='*76}")
    print(f"  MENJALANKAN BENCHMARK: 12 varian filter preprocessing...")
    print(f"{'='*76}\n")

    variants = make_variants(crop, config=CONFIG)
    benchmark_results = []

    for idx, (v_name, b_mask, preproc_time_ms, params) in enumerate(variants, start=1):
        print(f"  [{idx:02d}/12] Menguji: {v_name[:38]:<38}", end=" ")
        t_eval_start = time.perf_counter()
        eval_metrics = evaluate_variant(b_mask, is_horizontal=is_horizontal)
        eval_time_ms = (time.perf_counter() - t_eval_start) * 1000

        total_time_ms = preproc_time_ms + eval_time_ms
        res_entry = {
            'name': v_name,
            'mask': b_mask,
            'params': params,
            'continuity': eval_metrics['continuity'],
            'smoothness': eval_metrics['smoothness'],
            'outliers': eval_metrics['outliers'],
            'score': eval_metrics['score'],
            'preproc_ms': preproc_time_ms,
            'total_ms': total_time_ms,
        }
        benchmark_results.append(res_entry)

        print(f"-> Skor: {eval_metrics['score']:5.1f} | Kontinuitas: {eval_metrics['continuity']:5.1f}% | [{preproc_time_ms:.1f}ms]")

    # 4. Urutkan Leaderboard berdasarkan Skor Kualitas Tertinggi
    benchmark_results.sort(key=lambda x: x['score'], reverse=True)

    # 5. Tampilan Leaderboard di Command Prompt (Sesuai Gaya benchmark_preprocess.py)
    print(f"\n\n{'='*82}")
    print(f"  HASIL BENCHMARK - Ranked by Metrology Edge Quality")
    print(f"{'='*82}")
    print(f"  {'Rank':<5}{'Preprocessing Pipeline':<40}{'Kontinuitas':<13}{'Kehalusan':<11}{'#Outlier':<10}{'Waktu':<10}{'Skor':<6}")
    print(f"  {'-'*88}")

    for i, b in enumerate(benchmark_results, 1):
        star = " 🌟 #1" if i == 1 else (" **" if i <= 3 else "")
        print(f"  {i:<5}{b['name']:<40}{b['continuity']:<6.1f}%      {b['smoothness']:<6.1f}     {b['outliers']:<10}{b['preproc_ms']:<5.1f} ms   {b['score']:<5.1f}{star}")

    # 6. Rekomendasi TOP 3
    print(f"\n  {'='*56}")
    print(f"  REKOMENDASI TOP 3 FILTER UNTUK INSPEKSI CONVEYOR:")
    print(f"  {'='*56}")
    for i, b in enumerate(benchmark_results[:3], 1):
        p_str = ", ".join([f"{k}={v}" for k, v in b['params'].items()])
        print(f"  {i}. {b['name']}")
        print(f"     Parameter: {p_str}")
        print(f"     Kinerja  : Skor {b['score']:.1f} | Kontinuitas {b['continuity']:.1f}% | Kehalusan {b['smoothness']:.1f} | Outlier: {b['outliers']}")
        print(f"     Latency  : {b['preproc_ms']:.2f} ms per frame (Sangat cepat untuk conveyor)")
        print()

    # 7. Pilihan Terbaik & Kode Siap Salin
    top = benchmark_results[0]
    print(f"  >>> PILIHAN TERBAIK UNTUK KODINGAN ANDA: {top['name']}")
    print(f"      Skor={top['score']:.1f}/100, Kontinuitas={top['continuity']:.1f}%, Waktu={top['preproc_ms']:.2f}ms")
    print("\n" + "#"*78)
    print("📋 SALIN PARAMETER DI BAWAH INI KE FILE Tes_Garis_Acuan_PureCV_V4_w_Rib&Slat.py:")
    print("#"*78)
    print("# Parameter Optimal Hasil Benchmark:")
    for k, v in top['params'].items():
        print(f"BEST_{k.upper()} = {repr(v)}")
    print("#" * 78)

    # 8. Jendela Perbandingan Visual Top-4
    top4 = benchmark_results[:4]
    tw, th = 380, 160

    def prep_tile(img, title, score, badge):
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()
        r = cv2.resize(bgr, (tw, th))
        cv2.rectangle(r, (0, 0), (tw, 28), (15, 15, 15), -1)
        cv2.putText(r, f"{badge} {title[:28]}", (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 255), 1)
        cv2.rectangle(r, (0, th - 24), (tw, th), (25, 25, 25), -1)
        col = (0, 255, 0) if score >= 80 else (0, 200, 255)
        cv2.putText(r, f"Skor: {score:.1f}/100", (10, th - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 2)
        return r

    tile_orig = prep_tile(crop, "Foto Asli (Crop ROI)", 100.0, "📷")
    badges = ["🥇 #1", "🥈 #2", "🥉 #3", "🎖️ #4"]
    tiles = [prep_tile(b['mask'], b['name'], b['score'], badges[i]) for i, b in enumerate(top4)]

    blank = np.zeros((th, tw, 3), dtype=np.uint8)
    cv2.putText(blank, "Benchmark Selesai!", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    cv2.putText(blank, "Tekan sembarang tombol", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    row1 = np.hstack([tile_orig, tiles[0], tiles[1]])
    row2 = np.hstack([tiles[2], tiles[3], blank])
    grid = np.vstack([row1, row2])

    win_comp = "HASIL VISUAL BENCHMARK PREPROCESSING (TOP 4 TERBAIK)"
    cv2.namedWindow(win_comp, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_comp, 1200, 560)
    cv2.imshow(win_comp, grid)
    print("\n[INFO] Menampilkan preview visual Top 4 Filter Terbaik di layar...")
    print("       Tekan sembarang tombol pada jendela gambar untuk menutup.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("[INFO] Benchmark selesai dengan sukses!\n")


if __name__ == "__main__":
    main()
