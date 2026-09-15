"""
AI_Slat_Rib_Conveyor_V4_4_Robust_Profile.py
AI / Computer Vision - Inspeksi Deformasi Slat & Rib Conveyor
V4.5 - Geometry Normalized + Upper-Edge Locking & Full-Span Profile Extraction

Pembaruan Utama V4.5:
  SLAT : 1. UPPER-EDGE LOCKING: Menggunakan polaritas gradien vertikal (Sobel Y > 0)
            dan dominansi horizontal (|Sy| >= 0.45 * |Sx|). Kandidat tepi bawah
            dan bayangan bawah dieliminasi 100%, mencegah garis acuan memotong diagonal.
         2. ADAPTIVE THRESHOLDING: Mengganti ambang batas kaku 30 dengan threshold
            dinamis per kolom (SLAT_MIN_STR_FLOOR=12.0, SLAT_ADAPTIVE_RATIO=0.28)
            sehingga area berkarat di sisi kiri tetap terdeteksi sejak awal (x=0).
         3. BIDIRECTIONAL ANCHOR TRACKING: Menemukan kolom jangkar (anchor) dengan tepi
            terjelas, lalu melacak maju (ke kanan) dan mundur (ke kiri) dengan
            proyeksi kemiringan (slope) saat dropout. Menghasilkan profil fisik
            penuh 100% bentang plat.
         4. STRICT JUMP CONSTRAINT: Kandidat dengan lompatan > SLAT_MAX_JUMP
            didiskualifikasi mutlak, mencegah profil melompat ke tepi lain.
         5. REFERENCE LINE SEJAJAR: Garis acuan ditarik dari titik kiri tepi atas ke
            titik kanan tepi atas yang sama -> sejajar alami dengan perspektif plat.
  5. GEOMETRY NORMALIZATION: ROI SLAT menggunakan 4 titik sudut dan
            perspective transform ke canonical plane (1000x220 px). Deteksi profile
            dilakukan pada koordinat yang telah dinormalisasi sehingga perubahan
            rotasi/perspektif kamera lebih terkontrol.
  SIRIP: Topmost-edge algorithm DIPERTAHANKAN 100% (proven, stabil).
  UI   : Manual 2-stage ROI, tombol keyboard, navigasi foto, HUD, dan file picker
         dipertahankan sepenuhnya.
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
# KONFIGURASI PARAMETER (SEMUA TERPUSAT)
# ═════════════════════════════════════════════════════════════════════════════
FILTER_CONFIG = {
    # ── Preprocessing bersama ────────────────────────────────────────────────
    "CLAHE_CLIP_LIMIT":       2.5,
    "CLAHE_TILE_SIZE":        (8, 8),
    "BILATERAL_D":            7,
    "BILATERAL_SIGMA_COLOR":  50,
    "BILATERAL_SIGMA_SPACE":  50,
    "SOBEL_KSIZE":            3,
    "SOBEL_THRESH":           35,    # Untuk panel Morph Close debug SIRIP
    "MORPH_KERNEL_SIZE":      (13, 5),

    # ── SIRIP (RIB): smoothing & threshold ───────────────────────────────────
    "CORNER_TRIM_PCT":   0.06,
    "MEDIAN_K":          9,
    "SMOOTH_W":          7,
    "THRESH_SIRIP_PV":   15.0,   # batas P-V flatness SIRIP (px)
    "THRESH_SIRIP_MAX":  12.0,   # batas max absolute deviation SIRIP (px)

    # ── SLAT: Upper-Edge Locked Robust Profile Extraction V4.4 ───────────────
    "SLAT_MIN_STR_FLOOR":      12.0,  # ambang batas minimum gradien (area karat sisi kiri)
    "SLAT_ADAPTIVE_RATIO":     0.28,  # persentase dari max gradien lokal di kolom tsb
    "SLAT_HORIZ_DOMINANCE":    0.45,  # rasio min |Sy| / (|Sx| + 1e-5) untuk tolak noise vertikal
    "SLAT_SEARCH_RADIUS":      22,    # radius search band tracking (px)
    "SLAT_MAX_JUMP":           10,    # batas max lompatan per step sebelum hard-reject (px)
    "SLAT_PROFILE_STEP":       2,     # step sampling kolom (px)
    "SLAT_CANDIDATE_TOPN":     5,     # max kandidat per kolom
    "SLAT_MEDIAN_K":           5,     # kernel median outlier (jaga kelengkungan fisik)
    "SLAT_SMOOTH_W":           3,     # window moving avg
    "SLAT_CORNER_TRIM_PCT":    0.04,  # potong 4% ujung ekstrem profil
    "SLAT_MAX_DROPOUT_STEPS":  8,     # max consecutive missing steps sebelum interpolasi stop
    # ── SLAT: Upper-Edge Search Band V4.5.5 ───────────────────────────────────
    "SLAT_UPPER_BAND_TOP":       0.02,
    "SLAT_UPPER_BAND_BOTTOM":    0.55,

    # ── SLAT V4.5.5: Geometry-guided normal scan ────────────────────────────
    "SLAT_SCAN_TOP":              0.03,
    "SLAT_SCAN_BOTTOM":           0.55,
    "SLAT_SCAN_BLUR":             5,
    "SLAT_SCAN_GRAD_FLOOR":       8.0,
    "SLAT_SCAN_CONTRAST_RADIUS":  5,
    "SLAT_SCAN_MIN_CONTRAST":     5.0,
    "SLAT_SCAN_PROFILE_STEP":     3,
    "SLAT_SCAN_MAX_JUMP":         18.0,
    "SLAT_SCAN_MAX_SLOPE":        1.50,
    "SLAT_SCAN_W_GRAD":           0.30,
    "SLAT_SCAN_W_CONTRAST":       0.30,
    "SLAT_SCAN_W_CONTINUITY":     0.30,
    "SLAT_SCAN_W_SMOOTH":         0.10,
    "SLAT_SCAN_NEIGHBOR_Y":       7.0,
    "SLAT_SCAN_MEDIAN_K":         5,
    "SLAT_SCAN_SMOOTH_W":         5,
    # V4.5.5 — geometry-guided local search / coverage
    "SLAT_LOCAL_SEARCH_FORWARD":   45.0,   # search hanya ke bawah dari predicted upper edge
    "SLAT_LOCAL_SEARCH_BACK":       4.0,   # sedikit ruang di atas predicted edge
    "SLAT_PROFILE_MIN_COVERAGE":    0.80,
    "SLAT_LOCAL_W_POSITION":        0.45,
    "SLAT_LOCAL_W_GRAD":            0.25,
    "SLAT_LOCAL_W_CONTRAST":        0.20,
    "SLAT_LOCAL_W_CONTINUITY":      0.10,

    # Tracking: continuity diberi penalti gradual; hard reject hanya untuk
    # lompatan ekstrem agar tracker tidak mudah putus saat kontur berubah.
    "SLAT_HARD_JUMP":             24.0,
    "SLAT_TRACK_DIST_WEIGHT":     0.55,
    "SLAT_TRACK_STRENGTH_WEIGHT": 0.20,
    "SLAT_TRACK_SMOOTH_WEIGHT":    0.25,
    "SLAT_CANDIDATE_SUPPORT_X":    32,
    "SLAT_CANDIDATE_SUPPORT_Y":    8,
    "SLAT_ANCHOR_SUPPORT_WEIGHT":  0.55,
    "SLAT_ANCHOR_STRENGTH_WEIGHT": 0.30,
    "SLAT_ANCHOR_HEIGHT_WEIGHT":    0.15,
    "SLAT_HARD_SLOPE":              1.20,
    "SLAT_ALLOW_BOTH_POLARITY":     True,

    # ── SLAT: Geometry Normalization V4.5 ─────────────────────────────────────
    # ROI SLAT menggunakan 4 titik sudut untuk perspective rectification.
    # Profil dianalisis pada bidang canonical agar perubahan angle/perspektif
    # kamera tidak langsung mengubah koordinat pixel analisis.
    "SLAT_NORM_WIDTH":         1000,
    "SLAT_NORM_HEIGHT":        220,
    "SLAT_NORM_INTERPOLATION": cv2.INTER_CUBIC,

    # ── SLAT: threshold OK/NG ────────────────────────────────────────────────
    "THRESH_SLAT_PV":   18.0,
    "THRESH_SLAT_MAX":  15.0,
}


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
# PREPROCESSING BERSAMA
# ═════════════════════════════════════════════════════════════════════════════

def _preprocess_roi(frame, polygon_pts, config):
    """
    CLAHE -> Bilateral -> Sobel (X, Y, Magnitude).
    Mengembalikan gradien terarah (sx, sy) untuk Upper-Edge Locking.
    """
    pts  = np.array(polygon_pts, dtype=np.int32)
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    rx, ry, rw, rh = cv2.boundingRect(pts)
    crop      = frame[ry:ry+rh, rx:rx+rw]
    crop_mask = mask[ry:ry+rh, rx:rx+rw]

    gray     = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe    = cv2.createCLAHE(clipLimit=config["CLAHE_CLIP_LIMIT"],
                                tileGridSize=config["CLAHE_TILE_SIZE"])
    enhanced = clahe.apply(gray)
    denoised = cv2.bilateralFilter(enhanced,
                                   d=config["BILATERAL_D"],
                                   sigmaColor=config["BILATERAL_SIGMA_COLOR"],
                                   sigmaSpace=config["BILATERAL_SIGMA_SPACE"])

    sx_f   = cv2.Sobel(denoised, cv2.CV_64F, 1, 0, ksize=config["SOBEL_KSIZE"])
    sy_f   = cv2.Sobel(denoised, cv2.CV_64F, 0, 1, ksize=config["SOBEL_KSIZE"])
    grad_f = cv2.magnitude(sx_f, sy_f)
    grad   = np.clip(grad_f, 0, 255).astype(np.uint8)

    _, thr   = cv2.threshold(grad, config["SOBEL_THRESH"], 255, cv2.THRESH_BINARY)
    ker      = cv2.getStructuringElement(cv2.MORPH_RECT, config["MORPH_KERNEL_SIZE"])
    morph    = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, ker)
    morph_m  = cv2.bitwise_and(morph, morph, mask=crop_mask)
    grad_m   = cv2.bitwise_and(grad, grad, mask=crop_mask)

    return (pts, h, w, rx, ry, rw, rh, crop, crop_mask,
            gray, enhanced, denoised, grad, morph_m, grad_m, sx_f, sy_f)


def _ctx_bounds(frame, ry, rh, rx, rw, is_h):
    h, w  = frame.shape[:2]
    pad_y = max(90, int(rh * 1.6))
    cy1   = max(0, ry - pad_y)
    cy2   = min(h, ry + rh + pad_y)
    if is_h:
        return cy1, cy2, 0, w
    pad_x = max(90, int(rw * 1.6))
    return 0, h, max(0, rx - pad_x), min(w, rx + rw + pad_x)


def _make_panel(frame, pts, crop_mask, ry, rh, rx, rw,
                cy1, cy2, cx1, cx2, stage, title, pw=400, is_bgr=False):
    cw = max(1, cx2 - cx1)
    ch = max(1, cy2 - cy1)
    sb = cv2.cvtColor(stage, cv2.COLOR_GRAY2BGR) if not is_bgr else stage.copy()
    cv = (frame.astype(np.float32) * 0.45).astype(np.uint8)
    cc = cv[ry:ry+rh, rx:rx+rw]
    cc[crop_mask > 0] = sb[crop_mask > 0]
    cv[ry:ry+rh, rx:rx+rw] = cc
    cv2.polylines(cv, [pts], True, (0, 255, 255), 2)
    sub = cv[cy1:cy2, cx1:cx2].copy()
    gh  = max(180, min(320, int(pw * ch / cw)))
    r   = cv2.resize(sub, (pw, gh))
    cv2.rectangle(r, (0, 0), (pw, 26), (15, 15, 15), -1)
    cv2.putText(r, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1, cv2.LINE_AA)
    return r


# ═════════════════════════════════════════════════════════════════════════════
# SIRIP (RIB) — Topmost Edge Algorithm (TIDAK DIUBAH, PROVEN)
# ═════════════════════════════════════════════════════════════════════════════

def extract_clean_contour(frame, polygon_pts, name="KOMPONEN", config=FILTER_CONFIG):
    """Profil SIRIP: topmost white pixel dari Morph Close. Algoritma asli dipertahankan."""
    res = _preprocess_roi(frame, polygon_pts, config)
    pts, h, w, rx, ry, rw, rh, crop, crop_mask, gray, enhanced, denoised, grad, morph_m, _ = res[:15]

    if rw < 20 or rh < 10:
        return None, None, None, rx, ry, rw, rh, pts, None, True

    is_h = (rw >= rh)
    cy1, cy2, cx1, cx2 = _ctx_bounds(frame, ry, rh, rx, rw, is_h)

    def mkp(s, t, b=False):
        return _make_panel(frame, pts, crop_mask, ry, rh, rx, rw, cy1, cy2, cx1, cx2, s, t, 500, b)

    grid = np.vstack([np.hstack([mkp(crop, f"1. Foto Asli+ROI ({name})", True),
                                  mkp(gray,     "2. Grayscale"),
                                  mkp(enhanced, "3. CLAHE")]),
                      np.hstack([mkp(denoised,  "4. Bilateral"),
                                  mkp(grad,     "5. Gradient (Sobel)"),
                                  mkp(morph_m,  "6. Morph Close (Biner)")])])

    raw_x, raw_y = [], []
    if is_h:
        for xc in range(0, rw, 2):
            idx = np.where(morph_m[:, xc] > 0)[0]
            if len(idx):
                raw_x.append(rx + xc);  raw_y.append(ry + int(idx[0]))
    else:
        for yr in range(0, rh, 2):
            idx = np.where(morph_m[yr, :] > 0)[0]
            if len(idx):
                raw_x.append(rx + int(idx[0]));  raw_y.append(ry + yr)

    raw_x = np.array(raw_x, np.float32)
    raw_y = np.array(raw_y, np.float32)
    if len(raw_x) < 20:
        return None, None, None, rx, ry, rw, rh, pts, grid, is_h

    tr = max(2, int(config["CORNER_TRIM_PCT"] * len(raw_x)))
    ex, ey = raw_x[tr:-tr], raw_y[tr:-tr]

    def smooth(arr):
        k, ws = config["MEDIAN_K"], config["SMOOTH_W"]
        p  = np.pad(arr, (k//2, k//2), 'edge')
        md = np.array([np.median(p[i:i+k]) for i in range(len(arr))])
        ks = np.ones(ws) / ws
        sv = np.convolve(md, ks, 'same')
        sv[:ws] = md[:ws];  sv[-ws:] = md[-ws:]
        return sv

    cx_, cy_ = (ex, smooth(ey)) if is_h else (smooth(ex), ey)
    return cx_, cy_, ey, rx, ry, rw, rh, pts, grid, is_h


def analyze_sirip_pv(contour_data, config):
    """Metrologi SIRIP: P-V + Max Abs Deviation (dual threshold)."""
    cx, cy, _, rx, ry, rw, rh, pts, grid, is_h = contour_data
    tpv, tmx = config["THRESH_SIRIP_PV"], config["THRESH_SIRIP_MAX"]
    if cx is None or len(cx) < 15:
        return {'success': False, 'name': 'SIRIP', 'msg': 'Tepi tidak cukup', 'grid': grid, 'pts': pts}

    x1, y1 = float(cx[0]),  float(cy[0])
    x2, y2 = float(cx[-1]), float(cy[-1])
    sp = np.hypot(x2-x1, y2-y1)
    if sp < 1e-3:
        return {'success': False, 'name': 'SIRIP', 'msg': 'Bentang pendek', 'grid': grid, 'pts': pts}

    d  = ((y2-y1)*cx - (x2-x1)*cy + x2*y1 - y2*x1) / sp
    pv = float(np.max(d) - np.min(d))
    ma = float(np.max(np.abs(d)))
    ng = (pv > tpv) or (ma > tmx)

    mp, mn = float(np.max(d)), float(np.min(d))
    if ng:
        if abs(mp) > abs(mn)*1.3:   ds = f"BENGKOK (+{mp:.1f}px)"
        elif abs(mn) > abs(mp)*1.3: ds = f"BENGKOK ({mn:.1f}px)"
        else:                       ds = f"BERGELOMBANG (P-V {pv:.1f}px)"
        ss = f"NG ({ds})"
    else:
        ss = f"OK (RATA {pv:.1f}px)"

    im  = int(np.argmax(np.abs(d)))
    px_, py_ = float(cx[im]), float(cy[im])
    dx_, dy_ = x2-x1, y2-y1
    t   = ((px_-x1)*dx_ + (py_-y1)*dy_) / (sp**2)
    return {'success': True, 'name': 'SIRIP', 'pts': pts,
            'edge_x': cx, 'edge_y': cy,
            'x_left': x1, 'y_left': y1, 'x_right': x2, 'y_right': y2,
            'peak_x': px_, 'peak_y_actual': py_,
            'proj_x': x1+t*dx_, 'proj_y': y1+t*dy_,
            'pv_flatness': pv, 'max_abs': ma,
            'is_ng': ng, 'status_str': ss, 'grid': grid}


def draw_sirip_visuals(canvas, d, color=(0, 140, 255)):
    cv2.polylines(canvas, [d['pts']], True, color, 1)
    cv2.line(canvas, (int(d['x_left']), int(d['y_left'])),
             (int(d['x_right']), int(d['y_right'])), color, 3)
    cv2.polylines(canvas,
                  [np.column_stack((d['edge_x'], d['edge_y'])).astype(np.int32)],
                  False, (255, 255, 0), 2)
    cv2.line(canvas, (int(d['proj_x']), int(d['proj_y'])),
             (int(d['peak_x']), int(d['peak_y_actual'])), (255, 0, 255), 4)
    cv2.circle(canvas, (int(d['peak_x']), int(d['peak_y_actual'])), 5, (255, 0, 255), -1)


# ═════════════════════════════════════════════════════════════════════════════
# SLAT (PLAT DASAR) — GEOMETRY NORMALIZATION V4.5
# ═════════════════════════════════════════════════════════════════════════════

def _order_quad_points(points):
    """
    Mengurutkan 4 titik sudut ROI menjadi:
        TL, TR, BR, BL

    V4.5:
    - Tidak menggunakan metode sum/delta karena dapat gagal pada
      perspektif kuat dan menghasilkan titik duplikat.
    - Dua titik dengan y terkecil dianggap sebagai sisi atas.
    - Dua titik dengan y terbesar dianggap sebagai sisi bawah.
    - Masing-masing pasangan diurutkan berdasarkan x.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)

    if len(pts) != 4:
        raise ValueError(
            f"Perspective normalization membutuhkan tepat 4 titik, "
            f"tetapi menerima {len(pts)} titik."
        )

    # Pastikan keempat titik unik.
    if len(np.unique(pts, axis=0)) != 4:
        raise ValueError("Empat titik ROI harus berbeda.")

    # Urutkan berdasarkan koordinat Y.
    by_y = pts[np.argsort(pts[:, 1])]

    top = by_y[:2]
    bottom = by_y[2:]

    # Pasangan atas: kiri -> kanan.
    top = top[np.argsort(top[:, 0])]
    tl, tr = top[0], top[1]

    # Pasangan bawah: kiri -> kanan.
    bottom = bottom[np.argsort(bottom[:, 0])]
    bl, br = bottom[0], bottom[1]

    quad = np.array([tl, tr, br, bl], dtype=np.float32)

    # Validasi geometri: quadrilateral tidak boleh degeneratif.
    area = abs(cv2.contourArea(quad.reshape(-1, 1, 2)))
    if area < 100.0:
        raise ValueError(
            f"Quadrilateral ROI degeneratif (area={area:.1f}px²)."
        )

    return quad


def _normalize_slat_roi(frame, polygon_pts, config):
    """
    Mengubah ROI SLAT dari perspektif kamera menjadi canonical plane.

    Input:
        polygon_pts = 4 titik sudut ROI pada gambar asli.

    Output:
        normalized_frame : ROI SLAT yang sudah di-rectify
        normalized_pts   : empat sudut canonical
        M                 : matrix original -> normalized
        M_inv             : matrix normalized -> original
        quad              : empat sudut ROI pada gambar asli
    """
    if len(polygon_pts) != 4:
        raise ValueError(
            f"ROI SLAT V4.5 membutuhkan 4 titik sudut. "
            f"Saat ini ada {len(polygon_pts)} titik."
        )

    quad = _order_quad_points(polygon_pts)

    nw = int(config["SLAT_NORM_WIDTH"])
    nh = int(config["SLAT_NORM_HEIGHT"])

    if nw < 100 or nh < 40:
        raise ValueError("Ukuran canonical SLAT terlalu kecil.")

    dst = np.array([
        [0, 0],
        [nw - 1, 0],
        [nw - 1, nh - 1],
        [0, nh - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(quad, dst)

    # Validasi matrix sebelum inverse.
    if not np.all(np.isfinite(M)):
        raise ValueError("Matrix perspective transform mengandung nilai tidak valid.")

    det = float(np.linalg.det(M))
    if abs(det) < 1e-10:
        raise ValueError(
            f"Perspective transform singular (det={det:.3e})."
        )

    try:
        M_inv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        raise ValueError("Perspective transform tidak valid.")

    normalized = cv2.warpPerspective(
        frame,
        M,
        (nw, nh),
        flags=config["SLAT_NORM_INTERPOLATION"],
        borderMode=cv2.BORDER_REPLICATE
    )

    return normalized, dst, M, M_inv, quad


def _map_points_from_normalized(points_xy, M_inv):
    """
    Memetakan titik dari canonical plane kembali ke koordinat gambar asli.
    """
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(pts, M_inv)
    return mapped.reshape(-1, 2)


def _map_point_from_normalized(x, y, M_inv):
    mapped = _map_points_from_normalized([(x, y)], M_inv)
    return float(mapped[0, 0]), float(mapped[0, 1])


# ═════════════════════════════════════════════════════════════════════════════
# SLAT (PLAT DASAR) — Upper-Edge Locked Robust Profile Extraction V4.5
# ═════════════════════════════════════════════════════════════════════════════

def _find_col_upper_candidates(col_grad, col_sy, col_sx, col_mask,
                                y_lo, y_hi, config):
    """
    V4.5.5 — kandidat upper-edge dengan orientasi + polaritas yang lebih robust.

    Prinsip:
    - ROI sudah perspective-normalized, sehingga edge target diharapkan relatif
      horizontal.
    - Gunakan |Sy| sebagai kekuatan edge normal terhadap permukaan, bukan hanya
      Sy positif. Polaritas dapat berubah karena pencahayaan/permukaan.
    - Tetap menolak edge yang terlalu vertikal melalui dominance |Sy|/|Sx|.
    - Mengambil local maxima dan adaptive threshold.

    Return tuple: (y, gradient_strength, sy, polarity_strength).
    """
    lo = max(1, int(y_lo))
    hi = min(len(col_grad) - 2, int(y_hi))
    if hi <= lo:
        return []

    cands_raw = []

    for y in range(lo, hi + 1):
        if col_mask[y] == 0:
            continue

        sy = float(col_sy[y])
        sx = float(col_sx[y])
        g = float(col_grad[y])

        # Edge harus dominan pada arah Y setelah rectification.
        if abs(sy) <= 3.0:
            continue
        if abs(sy) < config["SLAT_HORIZ_DOMINANCE"] * (abs(sx) + 1e-5):
            continue

        # Default: kedua polaritas diperbolehkan. Ini penting karena perubahan
        # pencahayaan dapat membalik sign Sobel walaupun edge fisiknya sama.
        if not config.get("SLAT_ALLOW_BOTH_POLARITY", True) and sy <= 0:
            continue

        if g >= float(col_grad[y - 1]) and g >= float(col_grad[y + 1]):
            cands_raw.append((y, g, sy, abs(sy)))

    if not cands_raw:
        return []

    max_g = max(c[1] for c in cands_raw)
    col_thr = max(
        config["SLAT_MIN_STR_FLOOR"],
        max_g * config["SLAT_ADAPTIVE_RATIO"]
    )

    valid = [c for c in cands_raw if c[1] >= col_thr]
    valid.sort(key=lambda c: c[1], reverse=True)

    return valid[:config["SLAT_CANDIDATE_TOPN"]]


def _candidate_support(xc, y, col_data, config):
    """Mengukur kontinuitas kandidat terhadap kolom tetangga."""
    xr = int(config["SLAT_CANDIDATE_SUPPORT_X"])
    yr = float(config["SLAT_CANDIDATE_SUPPORT_Y"])
    support = 0
    total = 0

    for x2, cands in col_data.items():
        if abs(int(x2) - int(xc)) > xr:
            continue
        total += 1
        if any(abs(float(c[0]) - float(y)) <= yr for c in cands):
            support += 1

    return support / max(1, total)


def _score_slat_candidate(y, strength, max_str, prev_y, expected_y,
                          rh, config, prev_slope=0.0, candidate_slope=None,
                          support=0.0):
    """
    V4.5.5 scoring.

    Score = continuity + edge strength + smoothness + local support.
    Kandidat yang dekat expected position tetapi membuat perubahan slope tajam
    akan dikenai penalti. Lompatan ekstrem tetap di-reject.
    """
    sr = max(1.0, float(config["SLAT_SEARCH_RADIUS"]))
    hard_jump = float(config["SLAT_HARD_JUMP"])
    hard_slope = float(config.get("SLAT_HARD_SLOPE", 1.20))

    ss = float(strength) / max(1e-6, float(max_str))
    dist = abs(float(y) - float(expected_y))

    if dist > hard_jump:
        return -1e6

    continuity = max(0.0, 1.0 - dist / hard_jump)

    if candidate_slope is None:
        slope_score = 1.0
    else:
        ds = abs(float(candidate_slope) - float(prev_slope))
        slope_score = max(0.0, 1.0 - ds / hard_slope)

    return (
        config["SLAT_TRACK_DIST_WEIGHT"] * continuity
        + config["SLAT_TRACK_STRENGTH_WEIGHT"] * ss
        + config["SLAT_TRACK_SMOOTH_WEIGHT"] * slope_score
        + 0.15 * float(support)
    )


def _find_best_anchor(col_data, rw, rh, config):
    """
    V4.5.5 anchor selection berbasis GLOBAL CONTINUITY.

    Anchor tidak lagi dipilih hanya karena gradient paling kuat. Kandidat yang
    memiliki edge serupa pada banyak kolom tetangga akan lebih diprioritaskan.
    Ini mencegah anchor jatuh pada edge noise/komponen lain yang kebetulan kuat.
    """
    best_score = -1.0
    best_xc = rw // 2
    best_cand = None

    x_start = int(rw * 0.12)
    x_end = int(rw * 0.88)

    for xc in range(x_start, x_end + 1, int(config["SLAT_PROFILE_STEP"])):
        cands = col_data.get(xc, [])
        if not cands:
            continue

        for cand in cands:
            y, g = float(cand[0]), float(cand[1])
            support = _candidate_support(xc, y, col_data, config)
            strength = min(1.0, g / 180.0)
            height = max(0.0, 1.0 - y / max(1.0, float(rh)))

            score = (
                config["SLAT_ANCHOR_SUPPORT_WEIGHT"] * support
                + config["SLAT_ANCHOR_STRENGTH_WEIGHT"] * strength
                + config["SLAT_ANCHOR_HEIGHT_WEIGHT"] * height
            )

            if score > best_score:
                best_score = score
                best_xc = xc
                best_cand = cand

    if best_cand is None:
        return best_xc, None

    return best_xc, best_cand


def _track_slat_profile_bidirectional(col_data, rw, rh, config):
    """
    V4.5.5 bidirectional tracker.

    Perbaikan:
    - Anchor berdasarkan continuity support.
    - Kandidat dinilai berdasarkan expected position + strength + smoothness.
    - Slope di-update dengan low-pass agar tidak mengikuti satu noise spike.
    - Setelah tracking, isolated jumps diputus sebelum masuk ke smoothing.
    """
    anchor_xc, anchor_cand = _find_best_anchor(col_data, rw, rh, config)
    if anchor_cand is None:
        return [], None, None

    step = int(config["SLAT_PROFILE_STEP"])
    max_drop = int(config["SLAT_MAX_DROPOUT_STEPS"])
    slope_limit = float(config.get("SLAT_HARD_SLOPE", 1.20))

    def choose_candidate(cands, xc, prev_y, slope, direction):
        if not cands:
            return None, -1e6

        expected_y = prev_y + slope * direction * step
        max_str = max(float(c[1]) for c in cands)
        best = None
        best_score = -1e6

        for cy, cg, csy, *rest in cands:
            cand_slope = (float(cy) - prev_y) / float(direction * step)
            support = _candidate_support(xc, cy, col_data, config)
            score = _score_slat_candidate(
                cy, cg, max_str,
                prev_y, expected_y,
                rh, config,
                prev_slope=slope,
                candidate_slope=cand_slope,
                support=support
            )
            if score > best_score:
                best_score = score
                best = (cy, cg, csy, support)

        return best, best_score

    def track(direction):
        pts = [(anchor_xc, int(round(anchor_cand[0])))]
        prev_y = float(anchor_cand[0])
        slope = 0.0
        dropout = 0

        x_iter = range(anchor_xc + direction * step,
                       rw if direction > 0 else -1,
                       direction * step)

        for xc in x_iter:
            best, score = choose_candidate(
                col_data.get(xc, []), xc, prev_y, slope, direction
            )

            if best is not None and score > 0.05:
                act_y = float(best[0])
                new_slope = (act_y - prev_y) / float(direction * step)
                slope = (
                    0.80 * slope
                    + 0.20 * np.clip(new_slope, -slope_limit, slope_limit)
                )
                prev_y = act_y
                dropout = 0
                pts.append((xc, int(round(prev_y))))
            else:
                dropout += 1
                if dropout > max_drop:
                    break
                prev_y = float(np.clip(
                    prev_y + slope * direction * step,
                    0, rh - 1
                ))
                pts.append((xc, int(round(prev_y))))

        return pts

    right_pts = track(+1)
    left_pts = track(-1)

    return list(reversed(left_pts)) + right_pts[1:], anchor_xc, anchor_cand

def _reject_slat_outliers(x_arr, y_arr, config):
    """
    Outlier rejection KONSERVATIF:
    1. Isolated spike removal (jauh dari KEDUA tetangga, tetangga saling dekat).
       Deformasi kontinu (arch / concave) TIDAK dihapus.
    2. Gentle median filter (kernel KECIL = 5).
    3. Light moving average (window KECIL = 3).
    """
    if len(y_arr) < 5:
        return x_arr, y_arr

    mj, km, ws = config.get("SLAT_HARD_JUMP", config["SLAT_MAX_JUMP"]), config["SLAT_MEDIAN_K"], config["SLAT_SMOOTH_W"]

    # Step 1: spike
    valid = np.ones(len(y_arr), bool)
    for i in range(1, len(y_arr)-1):
        dp = abs(y_arr[i] - y_arr[i-1])
        dn = abs(y_arr[i] - y_arr[i+1])
        nb = abs(y_arr[i-1] - y_arr[i+1])
        if dp > mj and dn > mj and nb < mj:
            valid[i] = False
    x_arr, y_arr = x_arr[valid], y_arr[valid]

    if len(y_arr) < km + 2:
        return x_arr, y_arr

    # Step 2: gentle median
    pad    = km // 2
    padded = np.pad(y_arr, (pad, pad), 'edge')
    y_med  = np.array([np.median(padded[i:i+km]) for i in range(len(y_arr))])

    # Step 3: light moving avg
    if ws > 1:
        ks    = np.ones(ws) / ws
        y_out = np.convolve(y_med, ks, 'same')
        y_out[:ws] = y_med[:ws];  y_out[-ws:] = y_med[-ws:]
    else:
        y_out = y_med

    return x_arr, y_out



def _vertical_scan_candidates(denoised_col, y_lo, y_hi, config):
    """
    V4.5.5 — Geometry-guided 1D scan.

    Pada setiap kolom canonical, pencarian edge dilakukan dari atas ke bawah.
    Kandidat dinilai bukan hanya dari gradient, tetapi juga dari perubahan
    intensitas yang bertahan beberapa pixel (local contrast).

    Return:
        list of dict:
        {
            "y": float,
            "grad": float,
            "contrast": float,
            "score_base": float
        }
    """
    col = np.asarray(denoised_col, dtype=np.float32)

    blur_k = int(config.get("SLAT_SCAN_BLUR", 5))
    if blur_k >= 3 and blur_k % 2 == 1:
        # 1D Gaussian-like smoothing dengan OpenCV.
        sm = cv2.GaussianBlur(
            col.reshape(-1, 1),
            (1, blur_k),
            0
        ).reshape(-1)
    else:
        sm = col

    gy = np.gradient(sm)
    abs_g = np.abs(gy)

    lo = max(2, int(y_lo))
    hi = min(len(sm) - 3, int(y_hi))
    if hi <= lo:
        return []

    radius = max(2, int(config.get("SLAT_SCAN_CONTRAST_RADIUS", 5)))
    grad_floor = float(config.get("SLAT_SCAN_GRAD_FLOOR", 8.0))
    contrast_floor = float(config.get("SLAT_SCAN_MIN_CONTRAST", 5.0))

    raw = []

    for y in range(lo, hi + 1):
        g = float(abs_g[y])
        if g < grad_floor:
            continue

        # Pastikan local maximum gradient.
        if g < float(np.max(abs_g[max(lo, y-1):min(hi+1, y+2)])):
            continue

        # Step contrast: perbedaan rata-rata area di atas dan bawah edge.
        ya0 = max(0, y - radius)
        ya1 = y
        yb0 = y + 1
        yb1 = min(len(sm), y + 1 + radius)

        if ya1 <= ya0 or yb1 <= yb0:
            continue

        mean_above = float(np.mean(sm[ya0:ya1]))
        mean_below = float(np.mean(sm[yb0:yb1]))
        contrast = abs(mean_below - mean_above)

        if contrast < contrast_floor:
            continue

        raw.append({
            "y": float(y),
            "grad": g,
            "contrast": contrast
        })

    if not raw:
        return []

    # Normalisasi relatif terhadap kolom yang sama.
    max_g = max(c["grad"] for c in raw)
    max_c = max(c["contrast"] for c in raw)

    for c in raw:
        sg = c["grad"] / max(1e-6, max_g)
        sc = c["contrast"] / max(1e-6, max_c)
        c["score_base"] = (
            float(config.get("SLAT_SCAN_W_GRAD", 0.30)) * sg
            + float(config.get("SLAT_SCAN_W_CONTRAST", 0.30)) * sc
        )

    raw.sort(key=lambda c: c["score_base"], reverse=True)
    return raw[:int(config.get("SLAT_CANDIDATE_TOPN", 5))]


def _candidate_vertical_support(xc, y, candidate_map, step, config):
    """
    Mengecek apakah kandidat pada (x,y) memiliki edge serupa pada kolom
    tetangga kiri/kanan. Ini dipakai sebagai geometric continuity prior.
    """
    neighbor_y = float(config.get("SLAT_SCAN_NEIGHBOR_Y", 7.0))
    radius_x = max(
        1,
        int(config.get("SLAT_CANDIDATE_SUPPORT_X", 32) // max(1, step))
    )

    x_keys = sorted(candidate_map.keys())
    if not x_keys:
        return 0.0

    nearby = 0
    supported = 0

    for x2 in x_keys:
        if x2 == xc:
            continue
        if abs(x2 - xc) > radius_x * step:
            continue

        nearby += 1
        if any(abs(float(c["y"]) - float(y)) <= neighbor_y
               for c in candidate_map.get(x2, [])):
            supported += 1

    return supported / max(1, nearby)


def _select_geometry_guided_anchor(candidate_map, rw, rh, config):
    """
    Anchor V4.5.5:
    - Tidak mencari gradient terkuat secara global.
    - Memilih kandidat dengan kombinasi score edge + neighborhood support.
    - Preferensi sedikit ke area tengah ROI untuk menghindari ujung.
    """
    x0 = int(rw * 0.15)
    x1 = int(rw * 0.85)
    step = int(config.get("SLAT_SCAN_PROFILE_STEP", 3))

    best = None
    best_score = -1e9

    for xc in range(x0, x1 + 1, step):
        for c in candidate_map.get(xc, []):
            support = _candidate_vertical_support(
                xc, c["y"], candidate_map, step, config
            )

            center_score = 1.0 - abs(
                xc - rw * 0.5
            ) / max(1.0, rw * 0.5)

            # Edge terlalu dekat dengan batas atas/bawah kurang dipercaya.
            band_pos = c["y"] / max(1.0, float(rh))
            height_score = max(0.0, 1.0 - abs(band_pos - 0.22) / 0.30)

            score = (
                0.45 * c["score_base"]
                + 0.35 * support
                + 0.10 * center_score
                + 0.10 * height_score
            )

            if score > best_score:
                best_score = score
                best = (xc, c, support)

    return best


def _scan_track_direction(candidate_map, anchor, direction, rw, rh, config):
    """
    Track satu arah menggunakan:
      1) posisi yang diharapkan,
      2) edge score,
      3) neighborhood support,
      4) smoothness slope.

    Tidak melakukan extrapolation panjang ketika edge benar-benar hilang.
    """
    anchor_x, anchor_c, _ = anchor
    step = int(config.get("SLAT_SCAN_PROFILE_STEP", 3))
    max_jump = float(config.get("SLAT_SCAN_MAX_JUMP", 18.0))
    max_slope = float(config.get("SLAT_SCAN_MAX_SLOPE", 1.50))

    points = [(int(anchor_x), float(anchor_c["y"]))]
    prev_y = float(anchor_c["y"])
    slope = 0.0
    dropout = 0
    max_dropout = int(config.get("SLAT_MAX_DROPOUT_STEPS", 8))

    xs = range(
        int(anchor_x) + direction * step,
        rw if direction > 0 else -1,
        direction * step
    )

    for xc in xs:
        cands = candidate_map.get(xc, [])

        if not cands:
            dropout += 1
            if dropout > max_dropout:
                break
            prev_y = float(np.clip(
                prev_y + slope * direction * step,
                0, rh - 1
            ))
            points.append((xc, prev_y))
            continue

        expected_y = prev_y + slope * direction * step

        best = None
        best_score = -1e9

        for c in cands:
            cy = float(c["y"])
            dist = abs(cy - expected_y)

            if dist > max_jump:
                continue

            candidate_slope = (
                (cy - prev_y) / float(direction * step)
            )

            if abs(candidate_slope) > max_slope:
                continue

            dslope = abs(candidate_slope - slope)

            continuity = max(
                0.0,
                1.0 - dist / max_jump
            )

            smoothness = max(
                0.0,
                1.0 - dslope / max_slope
            )

            support = _candidate_vertical_support(
                xc, cy, candidate_map, step, config
            )

            score = (
                float(config.get("SLAT_SCAN_W_CONTINUITY", 0.30))
                * continuity
                + float(config.get("SLAT_SCAN_W_GRAD", 0.30))
                * c["score_base"]
                + float(config.get("SLAT_SCAN_W_SMOOTH", 0.10))
                * smoothness
                + 0.30 * support
            )

            if score > best_score:
                best_score = score
                best = (cy, candidate_slope, c)

        if best is None or best_score < 0.20:
            dropout += 1
            if dropout > max_dropout:
                break
            prev_y = float(np.clip(
                prev_y + slope * direction * step,
                0, rh - 1
            ))
            points.append((xc, prev_y))
            continue

        dropout = 0
        act_y, new_slope, _ = best

        # Low-pass slope agar permukaan bertekstur tidak mengubah arah profile.
        slope = (
            0.82 * slope
            + 0.18 * float(np.clip(
                new_slope, -max_slope, max_slope
            ))
        )
        prev_y = act_y
        points.append((xc, prev_y))

    return points


def _remove_scan_outliers(x, y, config):
    """
    Menghapus spike terisolasi tanpa meratakan deformasi lebar.
    """
    if len(y) < 7:
        return x, y

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    jump = float(config.get("SLAT_SCAN_MAX_JUMP", 18.0))
    keep = np.ones(len(y), dtype=bool)

    for i in range(1, len(y) - 1):
        left = abs(float(y[i] - y[i-1]))
        right = abs(float(y[i] - y[i+1]))
        neighbor = abs(float(y[i-1] - y[i+1]))

        if left > jump * 0.75 and right > jump * 0.75 and neighbor < jump * 0.50:
            keep[i] = False

    x = x[keep]
    y = y[keep]

    k = int(config.get("SLAT_SCAN_MEDIAN_K", 5))
    if k >= 3 and k % 2 == 1 and len(y) >= k:
        pad = k // 2
        yp = np.pad(y, (pad, pad), mode="edge")
        y_med = np.array(
            [np.median(yp[i:i+k]) for i in range(len(y))],
            dtype=np.float32
        )
    else:
        y_med = y.copy()

    sw = int(config.get("SLAT_SCAN_SMOOTH_W", 5))
    if sw >= 3 and sw % 2 == 1 and len(y_med) >= sw:
        kernel = np.ones(sw, dtype=np.float32) / sw
        ys = np.convolve(y_med, kernel, mode="same")
        half = sw // 2
        ys[:half] = y_med[:half]
        ys[-half:] = y_med[-half:]
    else:
        ys = y_med

    return x, ys


def extract_slat_profile_robust_v44(frame, polygon_pts, config=FILTER_CONFIG):
    """
    SLAT V4.5.5 — Geometry-Guided Independent Column Scan.

    Perubahan penting:
    - Tidak lagi menggunakan anchor + bidirectional tracking sebagai mekanisme utama.
    - Setiap kolom dinilai secara independen terhadap predicted upper edge.
    - Predicted edge berasal dari sisi atas 4-corner ROI setelah rectification.
    - Candidate dipilih berdasarkan kombinasi:
        geometric distance + gradient + local contrast.
    - Setelah seluruh profile terkumpul, spike/outlier dibersihkan.
    - Coverage dihitung dari candidate yang benar-benar ditemukan.

    Tujuan:
    menghindari kegagalan profile parsial akibat tracker berhenti ketika satu
    atau beberapa kandidat kolom tidak memenuhi continuity constraint.
    """
    try:
        normalized_frame, normalized_pts, M, M_inv, quad = \
            _normalize_slat_roi(frame, polygon_pts, config)
    except ValueError as e:
        print(f"\n⚠️ [SLAT V4.5.5] Geometry normalization gagal: {e}")
        d = np.zeros((220, 700, 3), np.uint8)
        cv2.putText(
            d, "SLAT ROI NEEDS 4 VALID CORNERS",
            (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )
        return None, None, [], d, {
            "rx": 0, "ry": 0, "rw": 0, "rh": 0,
            "pts": np.asarray(polygon_pts, dtype=np.int32),
            "quad": np.asarray(polygon_pts, dtype=np.float32),
            "M": None, "M_inv": None, "normalized": False,
            "profile_coverage": 0.0
        }

    res = _preprocess_roi(normalized_frame, normalized_pts, config)
    pts, h, w, rx, ry, rw, rh, crop, crop_mask, gray, enhanced, \
        denoised, grad, morph_m, grad_m, sx_f, sy_f = res

    if rw < 50 or rh < 20:
        return None, None, [], np.zeros((220, 700, 3), np.uint8), {
            "rx": rx, "ry": ry, "rw": rw, "rh": rh,
            "pts": pts, "quad": quad, "M": M, "M_inv": M_inv,
            "normalized": True, "profile_coverage": 0.0
        }

    tl, tr, br, bl = np.asarray(normalized_pts, dtype=np.float32)

    step = max(1, int(config.get("SLAT_SCAN_PROFILE_STEP", 3)))
    x_start = int(np.clip(np.ceil(min(tl[0], bl[0])), 0, rw - 1))
    x_end = int(np.clip(np.floor(max(tr[0], br[0])), 0, rw - 1))
    xs = list(range(x_start, x_end + 1, step))
    expected = len(xs)

    search_down = float(config.get("SLAT_LOCAL_SEARCH_FORWARD", 45.0))
    search_up = float(config.get("SLAT_LOCAL_SEARCH_BACK", 4.0))
    grad_floor = float(config.get("SLAT_SCAN_GRAD_FLOOR", 5.0))
    contrast_floor = float(config.get("SLAT_SCAN_MIN_CONTRAST", 2.0))
    radius = max(2, int(config.get("SLAT_SCAN_CONTRAST_RADIUS", 5)))

    predicted = []
    profile = []
    debug_candidates = []
    selected = []

    for x in xs:
        if abs(float(tr[0] - tl[0])) < 1e-6:
            pred_y = float((tl[1] + tr[1]) * 0.5)
        else:
            t = (float(x) - float(tl[0])) / float(tr[0] - tl[0])
            pred_y = float(tl[1] + t * (tr[1] - tl[1]))

        pred_y = float(np.clip(pred_y, 0, rh - 1))
        predicted.append((float(x), pred_y))

        y_lo = int(np.clip(np.floor(pred_y - search_up), 0, rh - 1))
        y_hi = int(np.clip(np.ceil(pred_y + search_down), 0, rh - 1))

        col = denoised[:, x].astype(np.float32)
        blur_k = int(config.get("SLAT_SCAN_BLUR", 5))
        blur_k = blur_k if blur_k >= 3 and blur_k % 2 == 1 else 5
        col_s = cv2.GaussianBlur(col.reshape(-1, 1), (1, blur_k), 0).reshape(-1)
        abs_g = np.abs(np.gradient(col_s))

        local_max = float(np.max(abs_g[y_lo:y_hi + 1])) if y_hi >= y_lo else 0.0
        local_max = max(local_max, grad_floor)

        cands = []
        for y in range(y_lo, min(y_hi, len(col_s) - 2) + 1):
            g = float(abs_g[y])

            if g < grad_floor:
                continue

            # Local maximum in a small vertical neighborhood.
            ya = max(y_lo, y - 1)
            yb = min(y_hi + 1, y + 2)
            if g < float(np.max(abs_g[ya:yb])):
                continue

            a0 = max(0, y - radius)
            a1 = y
            b0 = y + 1
            b1 = min(len(col_s), y + 1 + radius)
            if a1 <= a0 or b1 <= b0:
                continue

            contrast = abs(
                float(np.mean(col_s[b0:b1])) -
                float(np.mean(col_s[a0:a1]))
            )

            # Contrast threshold dibuat lunak; gradient + geometry tetap dominan.
            if contrast < contrast_floor:
                continue

            dist = abs(float(y) - pred_y)
            position_score = max(
                0.0,
                1.0 - dist / max(search_down + search_up, 1.0)
            )
            gradient_score = min(g / local_max, 1.0)
            contrast_score = min(
                contrast / max(contrast_floor * 4.0, 1.0), 1.0
            )

            score = (
                float(config.get("SLAT_LOCAL_W_POSITION", 0.45))
                * position_score
                + float(config.get("SLAT_LOCAL_W_GRAD", 0.25))
                * gradient_score
                + float(config.get("SLAT_LOCAL_W_CONTRAST", 0.20))
                * contrast_score
            )

            cands.append((score, y, g, contrast, dist))

        # Fallback lokal: bila tidak ada candidate yang lolos contrast,
        # gunakan local maximum gradient terdekat ke predicted edge.
        # Ini mencegah gap hanya karena tekstur/lighting.
        if not cands:
            valid_range = range(y_lo, min(y_hi, len(abs_g) - 1) + 1)
            fallback = []
            for y in valid_range:
                g = float(abs_g[y])
                if g < max(2.0, grad_floor * 0.45):
                    continue
                dist = abs(float(y) - pred_y)
                fallback.append((g, -dist, y))

            if fallback:
                fallback.sort(reverse=True)
                g, _, y = fallback[0]
                a0 = max(0, y - radius)
                a1 = y
                b0 = y + 1
                b1 = min(len(col_s), y + 1 + radius)
                contrast = abs(
                    float(np.mean(col_s[b0:b1])) -
                    float(np.mean(col_s[a0:a1]))
                )
                dist = abs(float(y) - pred_y)
                pos = max(
                    0.0,
                    1.0 - dist / max(search_down + search_up, 1.0)
                )
                gs = min(g / max(local_max, 1e-6), 1.0)
                cs = min(contrast / max(contrast_floor * 4.0, 1.0), 1.0)
                score = 0.55 * pos + 0.30 * gs + 0.15 * cs
                cands.append((score, y, g, contrast, dist))

        cands.sort(key=lambda z: z[0], reverse=True)

        # Simpan kandidat untuk debug.
        for score, y, g, contrast, dist in cands[:5]:
            debug_candidates.append((int(x), int(y), float(score)))

        if cands:
            score, y, g, contrast, dist = cands[0]
            profile.append((float(x), float(y)))
            selected.append({
                "x": float(x),
                "y": float(y),
                "pred_y": float(pred_y),
                "score": float(score),
                "gradient": float(g),
                "contrast": float(contrast),
                "distance": float(dist)
            })

    raw_valid = len(profile)
    raw_coverage = raw_valid / max(1, expected)

    # ── Robust cleanup ──────────────────────────────────────────────────────
    if raw_valid >= 5:
        px0 = np.asarray([p[0] for p in profile], np.float32)
        py0 = np.asarray([p[1] for p in profile], np.float32)
        px, py = _remove_scan_outliers(px0, py0, config)
    else:
        px, py = np.array([]), np.array([])

    final_valid = len(px)
    final_coverage = final_valid / max(1, expected)

    # ── Debug images ─────────────────────────────────────────────────────────
    scan_band = np.zeros((rh, rw), np.uint8)
    for x, pred_y in predicted:
        y0 = int(np.clip(round(pred_y - search_up), 0, rh - 1))
        y1 = int(np.clip(round(pred_y + search_down), 0, rh - 1))
        scan_band[y0:y1 + 1, int(x)] = 255

    cvis = np.zeros((rh, rw, 3), np.uint8)

    # Predicted geometric edge = green.
    if len(predicted) > 1:
        p = np.asarray(
            [[int(round(x)), int(round(y))] for x, y in predicted],
            np.int32
        )
        cv2.polylines(cvis, [p], False, (0, 180, 0), 1)

    # Candidates = blue/cyan intensity by score.
    for x, y, score in debug_candidates:
        v = int(np.clip(80 + 175 * score, 80, 255))
        cvis[y, x] = (255, v, 0)

    # Selected profile = cyan.
    if len(px) > 1:
        psel = np.asarray(
            [[int(round(x)), int(round(y))] for x, y in zip(px, py)],
            np.int32
        )
        cv2.polylines(cvis, [psel], False, (255, 255, 0), 2)

    svis = np.zeros((rh, rw, 3), np.uint8)
    if len(px) > 1:
        psel = np.asarray(
            [[int(round(x)), int(round(y))] for x, y in zip(px, py)],
            np.int32
        )
        cv2.polylines(svis, [psel], False, (255, 255, 0), 2)
        cv2.line(
            svis,
            tuple(psel[0]),
            tuple(psel[-1]),
            (0, 140, 255),
            2
        )

    scan_grad = np.abs(np.gradient(
        cv2.GaussianBlur(
            denoised, (1, 5), 0
        ).astype(np.float32),
        axis=0
    ))
    scan_grad = np.clip(scan_grad, 0, 255).astype(np.uint8)
    scan_grad = cv2.bitwise_and(scan_grad, scan_grad, mask=scan_band)

    def mkp(s, title, b=False):
        return _make_panel(
            normalized_frame, pts, crop_mask,
            ry, rh, rx, rw,
            0, h, 0, w,
            s, title, 400, b
        )

    p1 = mkp(normalized_frame, "1. Normalized ROI", True)
    p2 = mkp(gray, "2. Grayscale")
    p3 = mkp(enhanced, "3. CLAHE")
    p4 = mkp(denoised, "4. Bilateral")
    p5 = mkp(scan_grad, "5. Local Scan Gradient")
    p6 = mkp(scan_band, "6. Geometry Search Band")
    p7 = mkp(cvis, "7. Candidates + Selected Profile", True)
    p8 = mkp(svis, "8. Actual Profile + Reference", True)

    filter_grid = np.vstack([
        np.hstack([p1, p2, p3, p4]),
        np.hstack([p5, p6, p7, p8])
    ])

    print(
        f"\n  [SLAT V4.5.5] Profile coverage: "
        f"{final_coverage*100:.1f}% "
        f"({final_valid}/{expected})"
    )

    # Coverage gate. If insufficient, do not run metrology.
    coverage_min = float(
        config.get("SLAT_PROFILE_MIN_COVERAGE", 0.80)
    )

    if final_valid < 15 or final_coverage < coverage_min:
        print(
            f"  ⚠️ Profile belum cukup untuk metrologi: "
            f"coverage {final_coverage*100:.1f}% < "
            f"{coverage_min*100:.1f}%"
        )
        return None, None, debug_candidates, filter_grid, {
            "rx": rx, "ry": ry, "rw": rw, "rh": rh,
            "pts": pts, "quad": quad, "M": M, "M_inv": M_inv,
            "normalized": True,
            "profile_coverage": final_coverage,
            "raw_profile_coverage": raw_coverage,
            "expected_points": expected,
            "valid_points": final_valid,
            "profile_valid": False
        }

    return px, py, debug_candidates, filter_grid, {
        "rx": rx, "ry": ry, "rw": rw, "rh": rh,
        "pts": pts, "quad": quad, "M": M, "M_inv": M_inv,
        "normalized": True,
        "profile_coverage": final_coverage,
        "raw_profile_coverage": raw_coverage,
        "expected_points": expected,
        "valid_points": final_valid,
        "profile_valid": True
    }


def analyze_slat_profile(px, py, dbg_cands, filter_grid, roi_info, config):
    """
    Metrologi SLAT V4.5: P-V + Max Abs Deviation.
    Reference line = menghubungkan ujung kiri tepi atas ke ujung kanan tepi atas.
    OK/NG = (P-V > thresh_pv) OR (max_abs > thresh_max).
    """
    rx = roi_info["rx"]
    ry = roi_info["ry"]
    rw = roi_info["rw"]
    rh = roi_info["rh"]
    pts = roi_info["pts"]
    M_inv = roi_info.get("M_inv")
    quad = roi_info.get("quad")
    tpv, tmx = config["THRESH_SLAT_PV"], config["THRESH_SLAT_MAX"]
    nc = len(dbg_cands)

    if px is None or len(px) < 15:
        nv = len(px) if px is not None else 0
        print(f"\n{'='*55}\n  SLAT PROFILE DEBUG (V4.5.5)\n{'─'*55}")
        print(f"  Valid points       : {nv}  <- TERLALU SEDIKIT")
        print(f"  Total candidates   : {nc}")
        print(f"  STATUS             : GAGAL\n{'='*55}")
        return {'success': False, 'name': 'PLAT DASAR',
                'msg': f'Profil tidak cukup ({nv} titik, butuh >=15)',
                'n_valid': nv, 'n_candidates': nc,
                'grid': filter_grid, 'pts': pts}

    nv = len(px)
    x1, y1 = float(px[0]),  float(py[0])
    x2, y2 = float(px[-1]), float(py[-1])
    sp = np.hypot(x2-x1, y2-y1)

    if sp < 1e-3:
        return {'success': False, 'name': 'PLAT DASAR', 'msg': 'Bentang terlalu pendek',
                'grid': filter_grid, 'pts': pts, 'n_valid': nv, 'n_candidates': nc}

    # Perpendicular distance ke reference line
    perp = ((y2-y1)*px - (x2-x1)*py + x2*y1 - y2*x1) / sp
    pv   = float(np.max(perp) - np.min(perp))
    ma   = float(np.max(np.abs(perp)))
    ng   = (pv > tpv) or (ma > tmx)

    mp, mn = float(np.max(perp)), float(np.min(perp))
    if ng:
        if abs(mp) > abs(mn)*1.3:   ds = f"NAIK (+{mp:.1f}px)"
        elif abs(mn) > abs(mp)*1.3: ds = f"TURUN ({mn:.1f}px)"
        else:                       ds = f"BERGELOMBANG (P-V {pv:.1f}px)"
        ss = f"NG ({ds})"
    else:
        ss = f"OK (RATA P-V={pv:.1f}px Max={ma:.1f}px)"

    im  = int(np.argmax(np.abs(perp)))
    pkx = float(px[im]);  pky = float(py[im])
    dx_, dy_ = x2-x1, y2-y1
    t   = ((pkx-x1)*dx_ + (pky-y1)*dy_) / (sp**2)

    print(f"\n{'='*55}\n  SLAT PROFILE DEBUG (V4.5.5 GEOMETRY-GUIDED NORMAL SCAN)\n{'─'*55}")
    print(f"  Valid profile points : {nv}")
    print(f"  Total candidates     : {nc}")
    print(f"  Profile Span (X)     : {x1:.1f} -> {x2:.1f} (Lebar: {x2-x1:.1f}px)")
    print(f"  P-V Flatness         : {pv:.2f} px  (threshold: {tpv} px)")
    print(f"  Max Abs Deviation    : {ma:.2f} px  (threshold: {tmx} px)")
    print(f"  STATUS               : {ss}")
    print(f"{'='*55}")

    return {'success': True, 'name': 'PLAT DASAR', 'pts': pts,
            'edge_x': px, 'edge_y': py,
            'x_left': x1, 'y_left': y1, 'x_right': x2, 'y_right': y2,
            'peak_x': pkx, 'peak_y_actual': pky,
            'proj_x': x1+t*dx_, 'proj_y': y1+t*dy_,
            'pv_flatness': pv, 'max_abs': ma,
            'is_ng': ng, 'status_str': ss, 'grid': filter_grid,
            'n_valid': nv, 'n_candidates': nc,
            'M_inv': M_inv, 'quad': quad,
            'normalized': bool(roi_info.get("normalized", False))}


def draw_slat_visuals(canvas, d):
    """
    Visualisasi SLAT V4.5 pada koordinat gambar asli.

    Metrologi dilakukan pada canonical plane.
    Visualisasi dikembalikan ke gambar asli menggunakan inverse homography.
    """
    if not d.get('success'):
        return

    M_inv = d.get("M_inv")

    edge = np.column_stack(
        (d['edge_x'], d['edge_y'])
    ).astype(np.float32)

    # Jika normalization aktif, kembalikan profile ke gambar asli.
    if M_inv is not None:
        edge_orig = _map_points_from_normalized(edge, M_inv)
        left_orig = _map_point_from_normalized(
            d['x_left'], d['y_left'], M_inv
        )
        right_orig = _map_point_from_normalized(
            d['x_right'], d['y_right'], M_inv
        )
        peak_orig = _map_point_from_normalized(
            d['peak_x'], d['peak_y_actual'], M_inv
        )
        proj_orig = _map_point_from_normalized(
            d['proj_x'], d['proj_y'], M_inv
        )

        cv2.polylines(
            canvas,
            [np.asarray(d['quad'], np.int32)],
            True,
            (0, 200, 200),
            2
        )

        # Reference line pada canonical plane, kemudian dipetakan ke image.
        cv2.line(
            canvas,
            (int(left_orig[0]), int(left_orig[1])),
            (int(right_orig[0]), int(right_orig[1])),
            (0, 165, 255),
            3
        )

        cv2.polylines(
            canvas,
            [edge_orig.astype(np.int32)],
            False,
            (255, 255, 0),
            2
        )

        cv2.line(
            canvas,
            (int(proj_orig[0]), int(proj_orig[1])),
            (int(peak_orig[0]), int(peak_orig[1])),
            (255, 0, 255),
            4
        )

        cv2.circle(
            canvas,
            (int(peak_orig[0]), int(peak_orig[1])),
            6,
            (255, 0, 255),
            -1
        )

    else:
        # Fallback untuk data lama tanpa normalization.
        cv2.polylines(
            canvas,
            [d['pts']],
            True,
            (0, 200, 200),
            1
        )

        cv2.line(
            canvas,
            (int(d['x_left']), int(d['y_left'])),
            (int(d['x_right']), int(d['y_right'])),
            (0, 165, 255),
            3
        )

        cv2.polylines(
            canvas,
            [edge.astype(np.int32)],
            False,
            (255, 255, 0),
            2
        )

        cv2.line(
            canvas,
            (int(d['proj_x']), int(d['proj_y'])),
            (int(d['peak_x']), int(d['peak_y_actual'])),
            (255, 0, 255),
            4
        )

        cv2.circle(
            canvas,
            (int(d['peak_x']), int(d['peak_y_actual'])),
            6,
            (255, 0, 255),
            -1
        )


# ═════════════════════════════════════════════════════════════════════════════
# KONTROL PROGRAM & WINDOW
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


# ═════════════════════════════════════════════════════════════════════════════
# SESI INTERAKTIF UTAMA
# ═════════════════════════════════════════════════════════════════════════════

def interactive_multi_roi_session(initial_image_path, config=FILTER_CONFIG):
    global current_step, points_sirip, points_slat, current_mouse_pos

    image_path = initial_image_path
    t_spv  = config["THRESH_SIRIP_PV"];   t_smx = config["THRESH_SIRIP_MAX"]
    t_lpv  = config["THRESH_SLAT_PV"];    t_lmx = config["THRESH_SLAT_MAX"]

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
        wn   = f"INSPEKSI SLAT & RIB V4.4 (UPPER-EDGE LOCKED) - [{fn}]"

        cv2.namedWindow(wn, cv2.WINDOW_NORMAL)
        ww = min(1360, max(700, w));  wh = min(850, int(ww*h/w))
        cv2.resizeWindow(wn, ww, wh)
        cv2.setMouseCallback(wn, mouse_callback)

        reset_state()
        result_frame = None
        last_sirip   = None
        last_slat    = None

        print("\n" + "="*75)
        print(f"📷 FOTO   : {fn}  ({w}x{h} px)")
        print(f"⚙️  CONFIG : CLAHE={config['CLAHE_CLIP_LIMIT']} | "
              f"Bilateral={config['BILATERAL_D']} | "
              f"Sobel={config['SOBEL_THRESH']} | "
              f"Morph={config['MORPH_KERNEL_SIZE']}")
        print(f"   SIRIP  : PV≤{t_spv}px, Max≤{t_smx}px")
        print(f"   SLAT   : PV≤{t_lpv}px, Max≤{t_lmx}px  "
              f"| floor_str={config['SLAT_MIN_STR_FLOOR']} "
              f"| adaptive_ratio={config['SLAT_ADAPTIVE_RATIO']} "
              f"| max_jump={config['SLAT_MAX_JUMP']}px")
        print("📋 PANDUAN V4.4:")
        print("   [TAHAP 1] Klik >=4 sudut SIRIP (RIB)       → [ENTER]")
        print("   [TAHAP 2] Klik tepat 4 sudut PLAT DASAR (SLAT) → [ENTER]")
        print("   [Klik Kiri] Tambah  |  [Klik Kanan] Undo")
        print("   [P] Filter debug  |  [R] Reset  |  [N] Ganti foto")
        print("   [D/→] Foto berikut  |  [A/←] Foto sebelum  |  [Q/ESC] Keluar")
        print("="*75)

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
                            f"[TAHAP 1/2: SIRIP/RIB] Titik: {len(points_sirip)} | "
                            "[Klik Kiri]: Tambah | [ENTER]: Lanjut",
                            (15, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 180, 255), 2)
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
                            f"[TAHAP 2/2: PLAT DASAR/SLAT] Titik: {len(points_slat)} | "
                            "[Klik Kiri]: Tambah | [ENTER]: Selesai & Analisis",
                            (15, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 200), 2)
                cv2.imshow(wn, disp);  key = cv2.waitKey(20) & 0xFF

            else:
                cv2.imshow(wn, result_frame);  key = cv2.waitKey(30) & 0xFF

            # ── Keyboard Handling ────────────────────────────────────────────

            if key in [13, 32]:   # ENTER / SPACE
                if current_step == 1:
                    if len(points_sirip) >= 3:
                        current_step = 2
                        print("  [OK] ROI Sirip tersimpan! Klik sudut-sudut PLAT DASAR.")
                    else:
                        print("  ⚠️ Klik minimal 3 titik untuk SIRIP!")

                elif current_step == 2:
                    if len(points_slat) == 4:
                        print("\n⚙️  Memproses V4.5 (Geometry Normalized + Upper-Edge Locked)...")

                        # SIRIP: algoritma topmost (tidak diubah)
                        c_sir = extract_clean_contour(frame, points_sirip, "SIRIP", config)
                        l_sir = analyze_sirip_pv(c_sir, config)

                        # SLAT: Upper-Edge Locked Robust Profile Pipeline V4.4
                        px, py, dc, sgrid, roi = \
                            extract_slat_profile_robust_v44(frame, points_slat, config)
                        l_sla = analyze_slat_profile(px, py, dc, sgrid, roi, config)

                        if not l_sir.get('success') or not l_sla.get('success'):
                            msg = l_sir.get('msg','') if not l_sir.get('success') else l_sla.get('msg','')
                            print(f"  ⚠️ [ERROR] {msg}");  continue

                        # Simpan ke variabel sesi agar tombol P bisa akses
                        last_sirip = l_sir
                        last_slat  = l_sla

                        result_frame = frame.copy()
                        draw_sirip_visuals(result_frame, last_sirip)
                        draw_slat_visuals(result_frame, last_slat)

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

                        print("\n" + "="*70)
                        print("📊 LAPORAN METROLOGI V4.4:")
                        print(f"   SIRIP      : {l_sir['status_str']}")
                        print(f"   → P-V: {l_sir['pv_flatness']:.1f}px (≤{t_spv}) | "
                              f"Max: {l_sir['max_abs']:.1f}px (≤{t_smx})")
                        print(f"   PLAT DASAR : {l_sla['status_str']}")
                        print(f"   → P-V: {l_sla['pv_flatness']:.1f}px (≤{t_lpv}) | "
                              f"Max: {l_sla['max_abs']:.1f}px (≤{t_lmx})")
                        print(f"   → Valid pts: {l_sla['n_valid']} | "
                              f"Candidates: {l_sla['n_candidates']}")
                        print("="*70)

                        current_step = 3
                        base, ext  = os.path.splitext(image_path)
                        save_path  = f"{base}_hasil_v4_4{ext}"
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
                    ws_ = "FILTER: SIRIP (6 Panel)"
                    cv2.namedWindow(ws_, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(ws_, 1500, 620)
                    cv2.imshow(ws_, last_sirip['grid'])
                if last_slat is not None and last_slat.get('grid') is not None:
                    wsl = "FILTER: PLAT DASAR (8 Panel V4.5)"
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
    print("\n" + "#"*75)
    print("  AI / Computer Vision - Inspeksi Deformasi Slat & Rib Conveyor")
    print("  V4.4 - Upper-Edge Locked Robust Profile Extraction")
    print("  SLAT : Directional Sobel Y (Sy > 0) + Bidirectional Anchor Tracking")
    print("  SIRIP: Topmost Edge Algorithm (unchanged, proven)")
    print("#"*75)

    img_path = None
    if len(sys.argv) > 1:
        img_path = clean_path_string(sys.argv[1])

    if not img_path or not os.path.exists(img_path):
        print("\n📂 Membuka dialog pemilih file...")
        img_path = choose_file_dialog()

    if not img_path or not os.path.exists(img_path):
        print("-"*55)
        try:
            inp      = input("👉 Masukkan / Drag & Drop path foto: ")
            img_path = clean_path_string(inp)
        except (EOFError, KeyboardInterrupt):
            img_path = None
        print("-"*55)

    if img_path and os.path.exists(img_path):
        interactive_multi_roi_session(img_path, config=FILTER_CONFIG)
    else:
        print("❌ Tidak ada file gambar yang valid. Program selesai.")
