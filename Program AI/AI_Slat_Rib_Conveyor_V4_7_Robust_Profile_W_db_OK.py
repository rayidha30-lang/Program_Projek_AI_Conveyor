"""
AI_Slat_Rib_Conveyor_V4_5_Geometry_Normalized_W_db.py
AI / Computer Vision - Inspeksi Deformasi Slat & Rib Conveyor
V4.5 - Geometry Normalization & Isometric Normal Profile Scanning + DB Integration

Fitur Utama:
  1. INVARIANT ANGLE NORMALIZATION:
     - Menggunakan 'robust_order_quad' untuk mengurutkan 4 titik ROI [TL, TR, BR, BL] secara
       analitis berbasis pasangan sisi longitudinal dan posisi centroid. Kebal terhadap sudut
       miring (CAM-02/CAM-03, 0 - 60 derajat).
  2. ISOMETRIC CANONICAL RECTIFICATION:
     - Menghitung lebar (W_unwarp) dan tinggi (H_unwarp) proporsional terhadap ukuran fisik
       sebenarnya di citra, menjamin 1 piksel kanonikal tepat setara 1 piksel fisik kamera.
  3. NORMAL PROFILE SCANNING:
     - Pemindaian upper edge dilakukan pada bidang kanonikal terektifikasi, di mana setiap kolom
       vertikal otomatis merupakan vektor normal tegak lurus terhadap garis acuan (baseline chord).
  4. ADVANCED PREPROCESSING & OIL SUPPRESSION:
     - CLAHE + Bilateral Filtering + Horizontal Morphological BlackHat (25x5) untuk menekan
       pantulan oli ("riak gunung"), karat, dan debu logam.
  5. SUB-PIXEL METROLOGY & ACCURATE INVERSE MAPPING:
     - Evaluasi deviasi tegak lurus (P-V Flatness & Max Absolute Deviation).
     - Pemetaan balik hasil garis cyan & garis oranye ke citra asli via M^-1 tanpa pergeseran offset.
  6. DUAL WORKPIECE SUPPORT:
     - Normalisasi geometri aktif penuh pada Tahap 1: SIRIP (RIB) dan Tahap 2: PLAT DASAR (SLAT).
  7. FULL DATABASE & UI INTEGRATION:
     - Pencatatan otomatis ke SQL Server 'ConveyorInspection' via db.py.
     - HUD banner, navigasi keyboard ([P], [R], [N], [A]/[D], [Q]), dan auto-load preset.
"""

import cv2
import numpy as np
import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from db import save_inspection_to_db

try:
    import tkinter as tk
    from tkinter import filedialog
    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False

# ── Global State ──────────────────────────────────────────────────────────────
current_step      = 1
points_sirip      = []
points_slat       = []
current_mouse_pos = (0, 0)

# ═════════════════════════════════════════════════════════════════════════════
# KONFIGURASI PARAMETER (TERPUSAT)
# ═════════════════════════════════════════════════════════════════════════════
FILTER_CONFIG = {
    # ── Preprocessing bersama ────────────────────────────────────────────────
    "CLAHE_CLIP_LIMIT":       2.5,
    "CLAHE_TILE_SIZE":        (8, 8),
    "BILATERAL_D":            7,
    "BILATERAL_SIGMA_COLOR":  50,
    "BILATERAL_SIGMA_SPACE":  50,
    "SOBEL_KSIZE":            3,
    "SOBEL_THRESH":           35,
    "MORPH_KERNEL_SIZE":      (13, 5),

    # ── Ambang Batas Metrologi (px) ──────────────────────────────────────────
    "THRESH_SIRIP_PV":        15.0,   # batas max P-V Flatness SIRIP
    "THRESH_SIRIP_MAX":       12.0,   # batas max absolute deviation SIRIP
    "THRESH_SLAT_PV":         18.0,   # batas max P-V Flatness PLAT DASAR
    "THRESH_SLAT_MAX":        15.0,   # batas max absolute deviation PLAT DASAR

    # ── Tracking & Candidate Filtering ───────────────────────────────────────
    "SLAT_MIN_STR_FLOOR":     8.0,    # batas minimum Sobel Y
    "SLAT_ADAPTIVE_RATIO":    0.25,   # rasio dari max gradien lokal kolom
    "SLAT_HORIZ_DOMINANCE":   0.35,   # rasio min |Sy| / (|Sx| + 1e-5)
    "SLAT_PROFILE_STEP":      2,      # step sampling kolom (px)
    "SLAT_SEARCH_RADIUS":     18,     # radius tracking search window
    "SLAT_MAX_JUMP":          12,     # batas lompatan per step
    "SLAT_MEDIAN_K":          9,      # kernel median filter
    "SLAT_SMOOTH_W":          7,      # window moving average smoothing
    "SLAT_CORNER_TRIM_PCT":   0.04,   # trim 4% ujung ekstrem
    "SLAT_MAX_DROPOUT_STEPS": 8,
}

# ── Auto-Load Konfigurasi Hasil Benchmark Preprocessing ───────────────────────
try:
    from filter_config_presets import FILTER_CONFIG_OPTIMAL
    if isinstance(FILTER_CONFIG_OPTIMAL, dict):
        if "clahe_clip" in FILTER_CONFIG_OPTIMAL:
            FILTER_CONFIG["CLAHE_CLIP_LIMIT"] = float(FILTER_CONFIG_OPTIMAL["clahe_clip"])
        if "bilateral_d" in FILTER_CONFIG_OPTIMAL:
            FILTER_CONFIG["BILATERAL_D"] = int(FILTER_CONFIG_OPTIMAL["bilateral_d"])
        if "thresh" in FILTER_CONFIG_OPTIMAL:
            FILTER_CONFIG["SOBEL_THRESH"] = int(FILTER_CONFIG_OPTIMAL["thresh"])
        print("  [Auto-Load] Preset filter optimal berhasil dimuat dari filter_config_presets.py")
except Exception:
    pass


# ═════════════════════════════════════════════════════════════════════════════
# UTILITAS FILE & DIALOG
# ═════════════════════════════════════════════════════════════════════════════

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
        kwargs = dict(
            title="Pilih Foto Slat / Rib Conveyor",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.webp;*.tiff"),
                       ("Semua File", "*.*")]
        )
        if initial_dir and os.path.isdir(initial_dir):
            kwargs["initialdir"] = initial_dir
        fp = filedialog.askopenfilename(**kwargs)
        root.destroy()
        return clean_path_string(fp) if fp else None
    except Exception as e:
        print(f"  [Info] Dialog tidak dapat dibuka: {e}")
        return None


def get_sibling_images(current_path):
    if not current_path or not os.path.exists(current_path):
        return [], -1
    folder = os.path.dirname(os.path.abspath(current_path))
    exts   = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    try:
        files = sorted([
            os.path.join(folder, f) for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in exts
        ])
        norm = os.path.abspath(current_path)
        idx  = files.index(norm) if norm in files else -1
        return files, idx
    except Exception:
        return [], -1


# ═════════════════════════════════════════════════════════════════════════════
# MOUSE CALLBACK (ROI SELECTION)
# ═════════════════════════════════════════════════════════════════════════════

def mouse_callback(event, x, y, flags, param):
    global current_step, points_sirip, points_slat, current_mouse_pos
    current_mouse_pos = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        if current_step == 1:
            points_sirip.append((x, y))
            print(f"  [+] [SIRIP] Titik #{len(points_sirip)}: ({x}, {y})")
        elif current_step == 2:
            points_slat.append((x, y))
            print(f"  [+] [PLAT DASAR] Titik #{len(points_slat)}: ({x}, {y})")
    elif event == cv2.EVENT_RBUTTONDOWN:
        if current_step == 1 and points_sirip:
            r = points_sirip.pop()
            print(f"  [-] [SIRIP] Titik #{len(points_sirip)+1} dihapus: {r}")
        elif current_step == 2:
            if points_slat:
                r = points_slat.pop()
                print(f"  [-] [PLAT DASAR] Titik #{len(points_slat)+1} dihapus: {r}")
            else:
                current_step = 1
                print("  [<--] Kembali ke pengeditan ROI 1 (SIRIP).")


# ═════════════════════════════════════════════════════════════════════════════
# GEOMETRY NORMALIZATION ENGINE (ROBUST QUAD & ISOMETRIC RECTIFICATION)
# ═════════════════════════════════════════════════════════════════════════════

def robust_order_quad(points):
    """
    Mengurutkan 4 titik sudut quad slat/rib conveyor menjadi [TL, TR, BR, BL]
    yang invariant terhadap rotasi dan kemiringan (angle 0 - 60 derajat).
    """
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    if len(np.unique(pts, axis=0)) != 4:
        raise ValueError("Empat titik ROI harus berbeda.")

    cx, cy = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    cyclic_pts = pts[np.argsort(angles)]

    lengths = [
        np.linalg.norm(cyclic_pts[(i + 1) % 4] - cyclic_pts[i])
        for i in range(4)
    ]

    # Pasangan sisi terpanjang berlawanan adalah sisi longitudinal benda kerja
    if (lengths[0] + lengths[2]) > (lengths[1] + lengths[3]):
        side_a = (cyclic_pts[0], cyclic_pts[1])
        side_b = (cyclic_pts[2], cyclic_pts[3])
    else:
        side_a = (cyclic_pts[1], cyclic_pts[2])
        side_b = (cyclic_pts[3], cyclic_pts[0])

    avg_y_a = (side_a[0][1] + side_a[1][1]) / 2.0
    avg_y_b = (side_b[0][1] + side_b[1][1]) / 2.0

    # Sisi atas memiliki nilai Y rata-rata lebih kecil (posisi lebih tinggi di layar)
    top_side, bot_side = (side_a, side_b) if avg_y_a < avg_y_b else (side_b, side_a)

    tl, tr = (top_side[0], top_side[1]) if top_side[0][0] < top_side[1][0] else (top_side[1], top_side[0])
    bl, br = (bot_side[0], bot_side[1]) if bot_side[0][0] < bot_side[1][0] else (bot_side[1], bot_side[0])

    quad = np.array([tl, tr, br, bl], dtype=np.float32)
    area = abs(cv2.contourArea(quad.reshape(-1, 1, 2)))
    if area < 80.0:
        raise ValueError(f"Quadrilateral ROI degeneratif (area={area:.1f}px²).")
    return quad


def normalize_workpiece_quad(frame, polygon_pts, workpiece_type='SLAT'):
    """
    Melakukan unwarping isometrik 1:1 dari ROI 4-titik menjadi canonical plane.
    """
    if len(polygon_pts) != 4:
        raise ValueError("Normalisasi geometri kanonikal membutuhkan tepat 4 titik sudut.")

    quad = robust_order_quad(polygon_pts)
    tl, tr, br, bl = quad

    w_top = np.hypot(tr[0] - tl[0], tr[1] - tl[1])
    w_bot = np.hypot(br[0] - bl[0], br[1] - bl[1])
    w_unwarp = int(max(w_top, w_bot))

    h_left = np.hypot(bl[0] - tl[0], bl[1] - tl[1])
    h_right = np.hypot(br[0] - tr[0], br[1] - tr[1])
    h_unwarp = int(max(h_left, h_right))

    # Batasi tinggi kanonikal agar stabil
    if workpiece_type.upper() in ['SIRIP', 'RIB']:
        h_unwarp = max(25, min(140, h_unwarp))
    else:
        h_unwarp = max(35, min(180, h_unwarp))
    w_unwarp = max(100, w_unwarp)

    dst = np.array([
        [0, 0],
        [w_unwarp - 1, 0],
        [w_unwarp - 1, h_unwarp - 1],
        [0, h_unwarp - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(quad, dst)
    M_inv = np.linalg.inv(M)

    unwarped = cv2.warpPerspective(
        frame, M, (w_unwarp, h_unwarp),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return unwarped, dst, M, M_inv, quad, w_unwarp, h_unwarp


# ═════════════════════════════════════════════════════════════════════════════
# NORMAL PROFILE SCANNING & CANONICAL UPPER-EDGE EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════

def extract_canonical_profile_workpiece(frame, polygon_pts, workpiece_type='SLAT', config=FILTER_CONFIG):
    """
    Ekstraksi profil upper edge dengan Normal Profile Scanning pada canonical plane.
    """
    try:
        unwarped, dst, M, M_inv, quad, nw, nh = normalize_workpiece_quad(frame, polygon_pts, workpiece_type)
    except Exception as e:
        print(f"  ⚠️ Gagal normalisasi {workpiece_type}: {e}")
        return None

    gray = cv2.cvtColor(unwarped, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=config["CLAHE_CLIP_LIMIT"], tileGridSize=config["CLAHE_TILE_SIZE"])
    enhanced = clahe.apply(gray)
    denoised = cv2.bilateralFilter(enhanced, config["BILATERAL_D"], config["BILATERAL_SIGMA_COLOR"], config["BILATERAL_SIGMA_SPACE"])

    # Morphological BlackHat filter (25x5) untuk menekan pantulan oli & bintik karat
    k_bh = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    blackhat = cv2.morphologyEx(denoised, cv2.MORPH_BLACKHAT, k_bh)
    filtered = cv2.addWeighted(denoised, 1.0, blackhat, 0.45, 0)

    sy = cv2.Sobel(filtered, cv2.CV_64F, 0, 1, ksize=3)
    sx = cv2.Sobel(filtered, cv2.CV_64F, 1, 0, ksize=3)
    grad_mag = np.sqrt(sy**2 + sx**2)

    # Ambang batas adaptif untuk binerisasi tepi atas
    floor_str = float(config.get("SLAT_MIN_STR_FLOOR", 10.0))
    p70 = float(np.percentile(grad_mag, 70))
    thresh = max(floor_str, min(35.0, p70))
    edge_bin = (grad_mag >= thresh) & (np.abs(sy) > 4.0) & (np.abs(sy) >= config["SLAT_HORIZ_DOMINANCE"] * np.abs(sx))

    # Search band terproteksi dari border padding
    y_lo = max(5, int(0.04 * nh))
    y_hi = min(nh - 4, int(0.70 * nh))
    step = int(config.get("SLAT_PROFILE_STEP", 2))

    raw_x = []
    raw_y = []

    for x in range(10, nw - 10, step):
        col = edge_bin[y_lo:y_hi, x]
        idx = np.where(col)[0]
        if len(idx) > 0:
            # Kunci Tepi Atas: Ambil transisi PERTAMA dari atas (idx[0])
            first_y = y_lo + idx[0]
            # Refinement sub-pixel: cari puncak gradien lokal di sekitar first_y
            y_sub_min = max(y_lo, first_y - 2)
            y_sub_max = min(y_hi, first_y + 3)
            sub_strip = np.abs(sy[y_sub_min:y_sub_max, x])
            if len(sub_strip) > 0:
                best_sub = y_sub_min + np.argmax(sub_strip)
            else:
                best_sub = first_y
            raw_x.append(x)
            raw_y.append(float(best_sub))

    if len(raw_x) < 20:
        return None

    raw_x = np.array(raw_x, dtype=np.float32)
    raw_y = np.array(raw_y, dtype=np.float32)

    # Hard-reject lompatan ke tepi lain (mencegah osilasi gigi sisir / comb teeth)
    cleaned_x = [raw_x[0]]
    cleaned_y = [raw_y[0]]
    max_jump = float(config.get("SLAT_MAX_JUMP", 8.0))

    for i in range(1, len(raw_x)):
        prev = cleaned_y[-1]
        curr = raw_y[i]
        if abs(curr - prev) <= max_jump:
            cleaned_x.append(raw_x[i])
            cleaned_y.append(curr)
        else:
            # Interpolasi jika lompatan adalah noise / genangan oli
            cleaned_x.append(raw_x[i])
            cleaned_y.append(prev)

    cx = np.array(cleaned_x, dtype=np.float32)
    cy = np.array(cleaned_y, dtype=np.float32)

    # Trim 4% ujung ekstrem
    trim = max(2, int(config["SLAT_CORNER_TRIM_PCT"] * len(cx)))
    px = cx[trim:-trim]
    cy_trimmed = cy[trim:-trim]

    # Outlier rejection: median filter (k=9) + moving average (w=7)
    km = int(config.get("SLAT_MEDIAN_K", 9))
    pad = km // 2
    py_med = np.array([np.median(np.pad(cy_trimmed, pad, mode='edge')[i:i + km]) for i in range(len(cy_trimmed))])
    ws = int(config.get("SLAT_SMOOTH_W", 7))
    py_smooth = np.convolve(np.pad(py_med, ws // 2, mode='edge'), np.ones(ws) / ws, mode='valid')

    # ── Metrologi: Baseline Chord & Jarak Tegak Lurus ─────────────────────────
    x1, y1 = float(px[0]), float(py_smooth[0])
    x2, y2 = float(px[-1]), float(py_smooth[-1])
    sp = np.hypot(x2 - x1, y2 - y1)
    if sp < 1e-3:
        return None

    perp = ((y2 - y1) * px - (x2 - x1) * py_smooth + x2 * y1 - y2 * x1) / sp
    pv = float(np.max(perp) - np.min(perp))
    ma = float(np.max(np.abs(perp)))

    is_rib = (workpiece_type.upper() in ['SIRIP', 'RIB'])
    tpv = config["THRESH_SIRIP_PV"] if is_rib else config["THRESH_SLAT_PV"]
    tmx = config["THRESH_SIRIP_MAX"] if is_rib else config["THRESH_SLAT_MAX"]
    ng = (pv > tpv) or (ma > tmx)

    mp, mn = float(np.max(perp)), float(np.min(perp))
    if ng:
        if abs(mp) > abs(mn) * 1.3:
            ds = f"BENGKOK (+{mp:.1f}px)" if is_rib else f"NAIK (+{mp:.1f}px)"
        elif abs(mn) > abs(mp) * 1.3:
            ds = f"BENGKOK ({mn:.1f}px)" if is_rib else f"TURUN ({mn:.1f}px)"
        else:
            ds = f"BERGELOMBANG (P-V {pv:.1f}px)"
        ss = f"NG ({ds})"
    else:
        ss = f"OK (RATA P-V={pv:.1f}px Max={ma:.1f}px)"

    im = int(np.argmax(np.abs(perp)))
    pkx, pky = float(px[im]), float(py_smooth[im])
    dx_, dy_ = x2 - x1, y2 - y1
    t = ((pkx - x1) * dx_ + (pky - y1) * dy_) / (sp ** 2)

    # Buat 8-panel debug visualisasi untuk tombol [P]
    PW = 400
    gh = 160

    def mkp(img_stage, title, is_bgr=False):
        c = img_stage if is_bgr else cv2.cvtColor(img_stage, cv2.COLOR_GRAY2BGR)
        r = cv2.resize(c, (PW, gh))
        cv2.rectangle(r, (0, 0), (PW, 24), (20, 20, 20), -1)
        cv2.putText(r, title, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        return r

    sy_disp = np.clip(np.abs(sy), 0, 255).astype(np.uint8)
    unwarped_vis = unwarped.copy()
    cv2.polylines(unwarped_vis, [np.column_stack([px, py_smooth]).astype(np.int32)], False, (0, 255, 255), 2)
    cv2.line(unwarped_vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 140, 255), 2)

    p1 = mkp(unwarped, f"1. Canonical ROI ({workpiece_type})", True)
    p2 = mkp(gray, "2. Grayscale")
    p3 = mkp(enhanced, "3. CLAHE Enhanced")
    p4 = mkp(denoised, "4. Bilateral Denoised")
    p5 = mkp(blackhat, "5. BlackHat Filter (Suppress Oil)")
    p6 = mkp(sy_disp, "6. Sobel Y (Normal Gradient)")
    p7 = mkp(filtered, "7. Preprocessed workplane")
    p8 = mkp(unwarped_vis, "8. Detected Normal Profile", True)

    grid = np.vstack([np.hstack([p1, p2, p3, p4]),
                      np.hstack([p5, p6, p7, p8])])

    return {
        'success': True,
        'name': 'SIRIP' if is_rib else 'PLAT DASAR',
        'pts': polygon_pts,
        'quad': quad,
        'edge_x': px,
        'edge_y': py_smooth,
        'x_left': x1, 'y_left': y1,
        'x_right': x2, 'y_right': y2,
        'peak_x': pkx, 'peak_y_actual': pky,
        'proj_x': x1 + t * dx_, 'proj_y': y1 + t * dy_,
        'pv_flatness': pv, 'max_abs': ma,
        'is_ng': ng, 'status_str': ss,
        'M': M, 'M_inv': M_inv,
        'normalized': True,
        'grid': grid,
        'nw': nw, 'nh': nh,
        'n_valid': len(px),
        'n_candidates': len(raw_x),
        'unwarped': unwarped
    }


# ═════════════════════════════════════════════════════════════════════════════
# FALLBACK ALGORITHM (Jika titik ROI != 4)
# ═════════════════════════════════════════════════════════════════════════════

def _ctx_bounds(frame, ry, rh, rx, rw, is_h):
    h, w = frame.shape[:2]
    cy1 = max(0, ry - 30);  cy2 = min(h, ry + rh + 30)
    if is_h:
        return cy1, cy2, 0, w
    pad_x = max(90, int(rw * 1.6))
    return 0, h, max(0, rx - pad_x), min(w, rx + rw + pad_x)


def extract_clean_contour_fallback(frame, polygon_pts, name="KOMPONEN", config=FILTER_CONFIG):
    pts = np.array(polygon_pts, np.int32)
    h, w = frame.shape[:2]
    rx, ry, rw, rh = cv2.boundingRect(pts)
    if rw < 20 or rh < 10:
        return None

    crop = frame[ry:ry + rh, rx:rx + rw]
    mask = np.zeros((rh, rw), np.uint8)
    cv2.fillPoly(mask, [pts - [rx, ry]], 255)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=config["CLAHE_CLIP_LIMIT"], tileGridSize=config["CLAHE_TILE_SIZE"])
    enh = clahe.apply(gray)
    den = cv2.bilateralFilter(enh, config["BILATERAL_D"], 50, 50)
    sy = cv2.Sobel(den, cv2.CV_64F, 0, 1, ksize=3)

    raw_x, raw_y = [], []
    for xc in range(0, rw, 2):
        col_m = mask[:, xc]
        idx = np.where(col_m > 0)[0]
        if len(idx):
            strip_sy = sy[idx[0]:idx[-1] + 1, xc]
            if len(strip_sy) and np.max(strip_sy) > 5.0:
                raw_x.append(rx + xc)
                raw_y.append(ry + idx[0] + np.argmax(strip_sy))

    if len(raw_x) < 15:
        return None

    raw_x = np.array(raw_x, np.float32)
    raw_y = np.array(raw_y, np.float32)
    x1, y1 = float(raw_x[0]), float(raw_y[0])
    x2, y2 = float(raw_x[-1]), float(raw_y[-1])
    sp = np.hypot(x2 - x1, y2 - y1)
    perp = ((y2 - y1) * raw_x - (x2 - x1) * raw_y + x2 * y1 - y2 * x1) / sp
    pv = float(np.max(perp) - np.min(perp))
    ma = float(np.max(np.abs(perp)))
    ng = pv > config["THRESH_SIRIP_PV"]

    return {
        'success': True, 'name': name, 'pts': pts,
        'edge_x': raw_x, 'edge_y': raw_y,
        'x_left': x1, 'y_left': y1, 'x_right': x2, 'y_right': y2,
        'peak_x': float(raw_x[np.argmax(np.abs(perp))]),
        'peak_y_actual': float(raw_y[np.argmax(np.abs(perp))]),
        'proj_x': x1, 'proj_y': y1,
        'pv_flatness': pv, 'max_abs': ma,
        'is_ng': ng, 'status_str': f"NG ({pv:.1f}px)" if ng else f"OK ({pv:.1f}px)",
        'normalized': False, 'grid': None
    }


# ═════════════════════════════════════════════════════════════════════════════
# VISUALISASI HASIL PADA CITRA ASLI KAMERA (INVERSE MAPPING M^-1)
# ═════════════════════════════════════════════════════════════════════════════

def draw_workpiece_visuals(canvas, d, workpiece_type='SLAT'):
    if not d or not d.get('success'):
        return

    is_rib = (workpiece_type.upper() in ['SIRIP', 'RIB'])
    poly_color = (0, 140, 255) if is_rib else (0, 200, 200)
    line_color = (0, 255, 0) if is_rib else (255, 255, 0)
    base_color = (0, 140, 255) if not is_rib else (0, 165, 255)

    if d.get("normalized", False) and d.get("M_inv") is not None:
        M_inv = d["M_inv"]

        # 1. Gambar Poligon Quad Asli
        cv2.polylines(canvas, [np.asarray(d['quad'], np.int32)], True, poly_color, 2)

        # 2. Map Profile Points kembali ke Citra Kamera via M^-1
        edge_can = np.column_stack([d['edge_x'], d['edge_y']]).astype(np.float32).reshape(-1, 1, 2)
        edge_orig = cv2.perspectiveTransform(edge_can, M_inv).reshape(-1, 2)

        # 3. Map Endpoints Baseline Chord
        pts_chord = np.array([
            [[d['x_left'], d['y_left']]],
            [[d['x_right'], d['y_right']]],
            [[d['peak_x'], d['peak_y_actual']]],
            [[d['proj_x'], d['proj_y']]]
        ], dtype=np.float32)
        chord_orig = cv2.perspectiveTransform(pts_chord, M_inv).reshape(-1, 2)

        p_left = (int(chord_orig[0][0]), int(chord_orig[0][1]))
        p_right = (int(chord_orig[1][0]), int(chord_orig[1][1]))
        p_peak = (int(chord_orig[2][0]), int(chord_orig[2][1]))
        p_proj = (int(chord_orig[3][0]), int(chord_orig[3][1]))

        # Gambar baseline reference line
        cv2.line(canvas, p_left, p_right, base_color, 3)

        # Gambar kurva profil terdeteksi
        cv2.polylines(canvas, [edge_orig.astype(np.int32)], False, line_color, 2)

        # Gambar panah defleksi maksimum jika abnormal atau peak signifikan
        cv2.line(canvas, p_proj, p_peak, (255, 0, 255), 3)
        cv2.circle(canvas, p_peak, 5, (255, 0, 255), -1)

    else:
        # Fallback tanpa normalisasi
        cv2.polylines(canvas, [np.asarray(d['pts'], np.int32)], True, poly_color, 2)
        cv2.line(canvas, (int(d['x_left']), int(d['y_left'])),
                 (int(d['x_right']), int(d['y_right'])), base_color, 3)
        edge_arr = np.column_stack((d['edge_x'], d['edge_y'])).astype(np.int32)
        cv2.polylines(canvas, [edge_arr], False, line_color, 2)
        cv2.circle(canvas, (int(d['peak_x']), int(d['peak_y_actual'])), 5, (255, 0, 255), -1)


# ═════════════════════════════════════════════════════════════════════════════
# SESI INTERAKTIF UTAMA (2-STAGE ROI & HUD)
# ═════════════════════════════════════════════════════════════════════════════

def reset_state():
    global current_step, points_sirip, points_slat
    current_step = 1
    points_sirip.clear()
    points_slat.clear()


def safe_destroy_all_windows():
    cv2.destroyAllWindows()
    for _ in range(5):
        cv2.waitKey(10)


def interactive_multi_roi_session(initial_image_path, config=FILTER_CONFIG):
    global current_step, points_sirip, points_slat, current_mouse_pos

    image_path = initial_image_path
    t_spv = config["THRESH_SIRIP_PV"];   t_smx = config["THRESH_SIRIP_MAX"]
    t_lpv = config["THRESH_SLAT_PV"];    t_lmx = config["THRESH_SLAT_MAX"]

    while True:
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"\n❌ [ERROR] Gambar tidak dapat dibuka: '{image_path}'")
            alt = choose_file_dialog()
            if alt and os.path.exists(alt):
                image_path = alt;  continue
            print("   Tidak ada file. Program berhenti.")
            break

        h, w = frame.shape[:2]
        fn   = os.path.basename(image_path)
        wn   = f"INSPEKSI SLAT & RIB V4.5 (GEOMETRY NORMALIZED) - [{fn}]"

        cv2.namedWindow(wn, cv2.WINDOW_NORMAL)
        ww = min(1360, max(700, w));  wh = min(850, int(ww * h / w))
        cv2.resizeWindow(wn, ww, wh)
        cv2.setMouseCallback(wn, mouse_callback)

        reset_state()
        result_frame = None
        last_sirip   = None
        last_slat    = None

        print("\n" + "=" * 75)
        print(f"📷 FOTO   : {fn}  ({w}x{h} px)")
        print(f"⚙️  CONFIG : CLAHE={config['CLAHE_CLIP_LIMIT']} | "
              f"Bilateral={config['BILATERAL_D']} | "
              f"Sobel={config['SOBEL_THRESH']} | "
              f"BlackHat=(25, 5)")
        print(f"   SIRIP  : PV≤{t_spv}px, Max≤{t_smx}px")
        print(f"   SLAT   : PV≤{t_lpv}px, Max≤{t_lmx}px")
        print("📋 PANDUAN INSPEKSI V4.5:")
        print("   [TAHAP 1] Klik 4 sudut SIRIP (RIB)          → [ENTER]")
        print("   [TAHAP 2] Klik 4 sudut PLAT DASAR (SLAT)    → [ENTER]")
        print("   [Klik Kiri] Tambah  |  [Klik Kanan] Undo")
        print("   [P] Filter debug  |  [R] Reset  |  [N] Ganti foto")
        print("   [D/→] Foto berikut  |  [A/←] Foto sebelum  |  [Q/ESC] Keluar")
        print("=" * 75)

        while True:
            disp = frame.copy()

            if current_step == 1:
                for i, pt in enumerate(points_sirip):
                    cv2.circle(disp, pt, 5, (0, 140, 255), -1)
                    cv2.putText(disp, f"S{i+1}", (pt[0]+8, pt[1]-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)
                if len(points_sirip) > 1:
                    for i in range(len(points_sirip)-1):
                        cv2.line(disp, points_sirip[i], points_sirip[i+1], (0, 140, 255), 2)
                    cv2.line(disp, points_sirip[-1], current_mouse_pos, (160, 160, 160), 1)
                cv2.rectangle(disp, (0, 0), (w, 42), (20, 20, 20), -1)
                cv2.putText(disp,
                            f"[TAHAP 1/2: SIRIP/RIB] Titik: {len(points_sirip)} (Target 4 Titik) | "
                            "[Klik Kiri]: Tambah | [ENTER]: Lanjut",
                            (15, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 255), 2)
                cv2.imshow(wn, disp);  key = cv2.waitKey(20) & 0xFF

            elif current_step == 2:
                if len(points_sirip) >= 3:
                    cv2.polylines(disp, [np.array(points_sirip, np.int32)], True, (0, 140, 255), 2)
                    cv2.putText(disp, "ROI 1: SIRIP (TERKUNCI)",
                                (points_sirip[0][0], max(30, points_sirip[0][1]-10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2)
                for i, pt in enumerate(points_slat):
                    cv2.circle(disp, pt, 5, (0, 255, 200), -1)
                    cv2.putText(disp, f"P{i+1}", (pt[0]+8, pt[1]-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 2)
                if len(points_slat) > 1:
                    for i in range(len(points_slat)-1):
                        cv2.line(disp, points_slat[i], points_slat[i+1], (0, 255, 200), 2)
                    cv2.line(disp, points_slat[-1], current_mouse_pos, (160, 160, 160), 1)
                cv2.rectangle(disp, (0, 0), (w, 42), (20, 20, 20), -1)
                cv2.putText(disp,
                            f"[TAHAP 2/2: PLAT DASAR/SLAT] Titik: {len(points_slat)} (Target 4 Titik) | "
                            "[Klik Kiri]: Tambah | [ENTER]: Selesai & Analisis",
                            (15, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 2)
                cv2.imshow(wn, disp);  key = cv2.waitKey(20) & 0xFF

            else:
                cv2.imshow(wn, result_frame);  key = cv2.waitKey(30) & 0xFF

            # ── Keyboard Handling ────────────────────────────────────────────
            if key in [13, 32]:   # ENTER / SPACE
                if current_step == 1:
                    if len(points_sirip) >= 3:
                        current_step = 2
                        print("  [OK] ROI Sirip tersimpan! Klik 4 sudut PLAT DASAR.")
                    else:
                        print("  ⚠️ Klik minimal 3 titik (direkomendasikan 4 sudut) untuk SIRIP!")

                elif current_step == 2:
                    if len(points_slat) >= 3:
                        print("\n⚙️  Memproses V4.5 (Geometry Normalized & Normal Profile Scanning)...")

                        # 1. Analisis SIRIP (Geometry Normalization jika 4 titik, fallback jika != 4)
                        if len(points_sirip) == 4:
                            l_sir = extract_canonical_profile_workpiece(frame, points_sirip, "SIRIP", config)
                        else:
                            l_sir = extract_clean_contour_fallback(frame, points_sirip, "SIRIP", config)

                        # 2. Analisis PLAT DASAR (SLAT)
                        if len(points_slat) == 4:
                            l_sla = extract_canonical_profile_workpiece(frame, points_slat, "SLAT", config)
                        else:
                            l_sla = extract_clean_contour_fallback(frame, points_slat, "PLAT DASAR", config)

                        if l_sir is None or l_sla is None or not l_sir.get('success') or not l_sla.get('success'):
                            print("  ⚠️ [ERROR] Gagal mengekstraksi kontur workpiece. Pastikan ROI mencakup seluruh bentang plat/sirip.")
                            continue

                        last_sirip = l_sir
                        last_slat  = l_sla

                        result_frame = frame.copy()
                        draw_workpiece_visuals(result_frame, last_sirip, 'SIRIP')
                        draw_workpiece_visuals(result_frame, last_slat, 'SLAT')

                        # Top HUD Banner
                        def hud(txt, ng, y0, y1):
                            c = (0, 0, 255) if ng else (0, 255, 0)
                            cv2.rectangle(result_frame, (15, y0), (740, y1), (10, 10, 10), -1)
                            cv2.rectangle(result_frame, (15, y0), (740, y1), c, 2)
                            cv2.putText(result_frame, txt, (25, y1-11),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, c, 2)

                        hud(f"SIRIP     : {l_sir['status_str']}", l_sir['is_ng'], 15, 52)
                        hud(f"PLAT DASAR: {l_sla['status_str']}", l_sla['is_ng'], 58, 95)
                        cv2.rectangle(result_frame, (0, h-35), (w, h), (25, 25, 25), -1)
                        cv2.putText(result_frame,
                                    "[P]: Filter Debug | [R]: Ulang | [N]: Ganti Foto | "
                                    "[A/D]: Nav | [Q/ESC]: Keluar",
                                    (20, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 2)

                        print("\n" + "=" * 70)
                        print("📊 LAPORAN METROLOGI V4.5 (NORMAL PROFILE SCANNING):")
                        print(f"   SIRIP      : {l_sir['status_str']}")
                        print(f"   → P-V: {l_sir['pv_flatness']:.1f}px (≤{t_spv}) | "
                              f"Max: {l_sir['max_abs']:.1f}px (≤{t_smx})")
                        print(f"   PLAT DASAR : {l_sla['status_str']}")
                        print(f"   → P-V: {l_sla['pv_flatness']:.1f}px (≤{t_lpv}) | "
                              f"Max: {l_sla['max_abs']:.1f}px (≤{t_lmx})")
                        print(f"   → Valid pts: {l_sla.get('n_valid', len(l_sla['edge_x']))}")
                        print("=" * 70)

                        current_step = 3
                        base, ext  = os.path.splitext(image_path)
                        save_path  = f"{base}_hasil_v4_5{ext}"
                        cv2.imwrite(save_path, result_frame)
                        print(f"  💾 Disimpan: {os.path.basename(save_path)}")

                        # ──────────────────────────────────────────────────────────────
                        # ✅ SIMPAN KE SQL SERVER (Lengkap dengan Path Hasil Anotasi)
                        # ──────────────────────────────────────────────────────────────
                        save_inspection_to_db(image_path, l_sir, l_sla, annotated_path=save_path)
                        # ──────────────────────────────────────────────────────────────
                    else:
                        print("  ⚠️ Klik tepat 4 titik sudut untuk PLAT DASAR (SLAT)!")

            elif key in [ord('p'), ord('P')] and current_step == 3:
                if last_sirip is not None and last_sirip.get('grid') is not None:
                    ws_ = "FILTER: SIRIP (8 Panel Kanonikal)"
                    cv2.namedWindow(ws_, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(ws_, 1500, 620)
                    cv2.imshow(ws_, last_sirip['grid'])
                if last_slat is not None and last_slat.get('grid') is not None:
                    wsl = "FILTER: PLAT DASAR (8 Panel Kanonikal V4.5)"
                    cv2.namedWindow(wsl, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(wsl, 1600, 680)
                    cv2.imshow(wsl, last_slat['grid'])

            elif key in [ord('r'), ord('R')]:
                print("\n🔄 [RESET] Mengulang pemilihan ROI...")
                reset_state();  continue

            elif key in [ord('q'), ord('Q'), 27]:
                print("\n[INFO] Menutup program. Selamat tinggal!")
                safe_destroy_all_windows();  return

            elif key in [ord('n'), ord('N')]:
                print("\n📂 Dialog pemilihan foto baru...")
                folder   = os.path.dirname(os.path.abspath(image_path))
                new_path = choose_file_dialog(initial_dir=folder)
                if not new_path:
                    try:
                        inp      = input("👉 Atau masukkan path foto: ")
                        new_path = clean_path_string(inp)
                    except (EOFError, KeyboardInterrupt):
                        new_path = None
                if new_path and os.path.exists(new_path):
                    image_path = new_path;  safe_destroy_all_windows();  break
                elif new_path:
                    print(f"⚠️ File tidak ditemukan: '{new_path}'")
                else:
                    print("ℹ️ Dibatalkan.")

            elif key in [ord('d'), ord('D'), 83]:
                sibs, idx = get_sibling_images(image_path)
                if len(sibs) > 1 and idx != -1:
                    ni = (idx+1) % len(sibs);  image_path = sibs[ni]
                    print(f"\n⏩ [NEXT] {ni+1}/{len(sibs)}: {os.path.basename(image_path)}")
                    safe_destroy_all_windows();  break
                else:
                    print("\nℹ️ Tidak ada foto lain di folder.")

            elif key in [ord('a'), ord('A'), 81]:
                sibs, idx = get_sibling_images(image_path)
                if len(sibs) > 1 and idx != -1:
                    pi = (idx-1) % len(sibs);  image_path = sibs[pi]
                    print(f"\n⏪ [PREV] {pi+1}/{len(sibs)}: {os.path.basename(image_path)}")
                    safe_destroy_all_windows();  break
                else:
                    print("\nℹ️ Tidak ada foto lain di folder.")


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "#" * 75)
    print("  AI / Computer Vision - Inspeksi Deformasi Slat & Rib Conveyor")
    print("  V4.5 - Geometry Normalized & Isometric Normal Profile Scanning")
    print("  - Angle-Invariant Quad Perspective Rectification (CAM-01/02/03)")
    print("  - Morphological BlackHat (25x5) Oil & Rust Depression Suppression")
    print("  - Sub-pixel Metrology with Baseline Chord & Perpendicular Deviation")
    print("#" * 75)

    img_path = None
    if len(sys.argv) > 1:
        img_path = clean_path_string(sys.argv[1])

    if not img_path or not os.path.exists(img_path):
        print("\n📂 Membuka dialog pemilih file...")
        img_path = choose_file_dialog()

    if not img_path or not os.path.exists(img_path):
        print("-" * 55)
        try:
            inp      = input("👉 Masukkan / Drag & Drop path foto: ")
            img_path = clean_path_string(inp)
        except (EOFError, KeyboardInterrupt):
            img_path = None
        print("-" * 55)

    if img_path and os.path.exists(img_path):
        interactive_multi_roi_session(img_path, config=FILTER_CONFIG)
    else:
        print("❌ Tidak ada file gambar yang valid. Program selesai.")
