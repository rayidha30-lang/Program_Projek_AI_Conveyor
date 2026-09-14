"""
AI_Slat_Rib_Conveyor_V4_3_Robust_Profile.py
AI / Computer Vision - Inspeksi Deformasi Slat & Rib Conveyor
V4.3 - Robust Physical Profile Extraction

Perubahan utama:
  SLAT : TIDAK lagi menggunakan first-white-pixel dari Morph Close.
          Pipeline baru: gradient -> local maxima candidates -> scoring
          (edge strength + position continuity) -> search band tracking
          -> conservative outlier rejection.
  SIRIP: Topmost-edge algorithm DIPERTAHANKAN (proven, tidak diubah).
  Dual threshold (P-V + Max Abs Deviation) untuk kedua komponen.
  8-panel debug visualization untuk SLAT.
  Semua parameter terpusat di FILTER_CONFIG.
  Semua UI, keyboard, file picker, navigasi foto DIPERTAHANKAN.
"""

import cv2
import numpy as np
import os
import sys

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
    "SOBEL_THRESH":           35,    # Hanya untuk panel Morph Close (debug)
    "MORPH_KERNEL_SIZE":      (13, 5),

    # ── SIRIP (RIB): smoothing & threshold ───────────────────────────────────
    "CORNER_TRIM_PCT":   0.06,
    "MEDIAN_K":          9,
    "SMOOTH_W":          7,
    "THRESH_SIRIP_PV":   15.0,   # batas P-V flatness SIRIP (px)
    "THRESH_SIRIP_MAX":  12.0,   # batas max absolute deviation SIRIP (px)

    # ── SLAT: Robust Profile Extraction ──────────────────────────────────────
    "SLAT_EDGE_MIN_STRENGTH":  30,    # gradient minimum agar jadi kandidat
    "SLAT_SEARCH_RADIUS":      25,    # half-width search band dari prev pos (px)
    "SLAT_MAX_JUMP":           12,    # max step antar kolom sebelum penalti berat (px)
    "SLAT_PROFILE_STEP":       2,     # step sampling kolom (px)
    "SLAT_CANDIDATE_TOPN":     5,     # max kandidat per kolom yang diskor
    "SLAT_INIT_BAND_PCT":      0.40,  # posisi awal = N% dari atas ROI
    "SLAT_MEDIAN_K":           5,     # kernel median outlier (KECIL = jaga deformasi)
    "SLAT_SMOOTH_W":           3,     # window moving avg (KECIL = jaga deformasi)
    "SLAT_CORNER_TRIM_PCT":    0.05,  # potong 5% ujung profil

    # ── SLAT: threshold OK/NG ────────────────────────────────────────────────
    "THRESH_SLAT_PV":   18.0,
    "THRESH_SLAT_MAX":  15.0,
}


# ═════════════════════════════════════════════════════════════════════════════
# UTILITAS
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
# MOUSE CALLBACK
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
    """CLAHE -> Bilateral -> Sobel gradient. Dipakai SIRIP dan SLAT."""
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
    sx   = cv2.Sobel(denoised, cv2.CV_64F, 1, 0, ksize=config["SOBEL_KSIZE"])
    sy   = cv2.Sobel(denoised, cv2.CV_64F, 0, 1, ksize=config["SOBEL_KSIZE"])
    grad = cv2.magnitude(sx, sy)
    grad = np.clip(grad, 0, 255).astype(np.uint8)

    _, thr   = cv2.threshold(grad, config["SOBEL_THRESH"], 255, cv2.THRESH_BINARY)
    ker      = cv2.getStructuringElement(cv2.MORPH_RECT, config["MORPH_KERNEL_SIZE"])
    morph    = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, ker)
    morph_m  = cv2.bitwise_and(morph, morph, mask=crop_mask)
    grad_m   = cv2.bitwise_and(grad, grad, mask=crop_mask)

    return pts, h, w, rx, ry, rw, rh, crop, crop_mask, gray, enhanced, denoised, grad, morph_m, grad_m


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
                cy1, cy2, cx1, cx2, stage, title, pw=500, is_bgr=False):
    cw = max(1, cx2 - cx1)
    ch = max(1, cy2 - cy1)
    sb = cv2.cvtColor(stage, cv2.COLOR_GRAY2BGR) if not is_bgr else stage.copy()
    cv = (frame.astype(np.float32) * 0.45).astype(np.uint8)
    cc = cv[ry:ry+rh, rx:rx+rw]
    cc[crop_mask > 0] = sb[crop_mask > 0]
    cv[ry:ry+rh, rx:rx+rw] = cc
    cv2.polylines(cv, [pts], True, (0, 255, 255), 3)
    sub = cv[cy1:cy2, cx1:cx2].copy()
    gh  = max(200, min(340, int(pw * ch / cw)))
    r   = cv2.resize(sub, (pw, gh))
    cv2.rectangle(r, (0, 0), (pw, 30), (15, 15, 15), -1)
    cv2.putText(r, title, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 2)
    return r


# ═════════════════════════════════════════════════════════════════════════════
# SIRIP (RIB) — Topmost Edge Algorithm (TIDAK DIUBAH)
# ═════════════════════════════════════════════════════════════════════════════

def extract_clean_contour(frame, polygon_pts, name="KOMPONEN", config=FILTER_CONFIG):
    """Profil SIRIP: topmost white pixel dari Morph Close. Algoritma asli dipertahankan."""
    pts, h, w, rx, ry, rw, rh, crop, crop_mask, gray, enhanced, denoised, grad, morph_m, _ = \
        _preprocess_roi(frame, polygon_pts, config)

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
# SLAT (PLAT DASAR) — Robust Physical Profile Extraction V4.3
# ═════════════════════════════════════════════════════════════════════════════

def _find_col_candidates(col_grad, y_min, y_max, min_str, top_n):
    """
    Temukan local maxima gradient di range [y_min, y_max].
    Return list (y, strength) sorted by strength desc, max top_n.
    """
    lo = max(0, y_min)
    hi = min(len(col_grad)-1, y_max)
    if hi - lo < 2:
        return []
    reg  = col_grad[lo:hi+1].astype(np.float32)
    cands = []
    for i in range(1, len(reg)-1):
        v = reg[i]
        if v >= min_str and v >= reg[i-1] and v >= reg[i+1]:
            cands.append((lo+i, float(v)))
    cands.sort(key=lambda c: c[1], reverse=True)
    return cands[:top_n]


def _score_candidate(y, strength, max_str, prev_y, config):
    """
    Score = 55% edge_strength + 45% position_continuity.
    Penalti berat jika loncatan > 2*max_jump.
    Tanpa prev_y: hanya berdasarkan strength.
    """
    sr  = config["SLAT_SEARCH_RADIUS"]
    mj  = config["SLAT_MAX_JUMP"]
    ss  = strength / max_str if max_str > 0 else 0.0
    if prev_y is None:
        return ss
    dist = abs(y - prev_y)
    if dist > mj * 2:
        return ss * 0.05   # penalti berat, bukan reject mutlak
    cs = max(0.0, 1.0 - dist / sr)
    return 0.55 * ss + 0.45 * cs


def _reject_slat_outliers(x_arr, y_arr, config):
    """
    Outlier rejection KONSERVATIF:
    1. Isolated spike removal (jauh dari KEDUA tetangga, tetangga saling dekat).
       Deformasi kontinu (arch, concave) TIDAK dihapus.
    2. Gentle median filter (kernel KECIL).
    3. Light moving average  (window KECIL).
    """
    if len(y_arr) < 5:
        return x_arr, y_arr

    mj, km, ws = config["SLAT_MAX_JUMP"], config["SLAT_MEDIAN_K"], config["SLAT_SMOOTH_W"]

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


def extract_slat_profile_robust(frame, polygon_pts, config=FILTER_CONFIG):
    """
    Robust SLAT Physical Profile Extraction V4.3.

    Pipeline:
      ROI -> CLAHE -> Bilateral -> Sobel gradient
      -> per-column local maxima detection   (BUKAN first white pixel)
      -> candidate scoring: strength + continuity
      -> search band tracking (+/- SLAT_SEARCH_RADIUS)
      -> spike rejection (deformasi kontinu DIPERTAHANKAN)
      -> gentle smoothing (kernel kecil)

    Morph Close dibuat untuk panel debug #6 saja, TIDAK untuk profil.

    Returns:
      profile_x, profile_y  – array profil (koordinat frame)
      dbg_cands             – list (xf, yf, strength) semua kandidat ditemukan
      dbg_rej               – list (xf, yf) kolom tanpa kandidat valid
      filter_grid           – gambar 8-panel debug
      roi_info              – (rx, ry, rw, rh, pts)
    """
    pts, h, w, rx, ry, rw, rh, crop, crop_mask, gray, enhanced, denoised, grad, morph_m, grad_m = \
        _preprocess_roi(frame, polygon_pts, config)

    is_h = (rw >= rh)
    cy1, cy2, cx1, cx2 = _ctx_bounds(frame, ry, rh, rx, rw, is_h)
    PW = 400   # 4 panel per baris -> lebar total 1600px

    def mkp(s, t, b=False):
        return _make_panel(frame, pts, crop_mask, ry, rh, rx, rw, cy1, cy2, cx1, cx2, s, t, PW, b)

    # Guard ROI terlalu kecil
    if rw < 20 or rh < 10:
        d = np.zeros((220, PW, 3), np.uint8)
        cv2.putText(d, "ROI INVALID", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        grid = np.vstack([np.hstack([d]*4), np.hstack([d]*4)])
        return None, None, [], [], grid, (rx, ry, rw, rh, pts)

    # ── Deteksi kandidat per kolom ───────────────────────────────────────────
    min_str  = config["SLAT_EDGE_MIN_STRENGTH"]
    sr       = config["SLAT_SEARCH_RADIUS"]
    step     = config["SLAT_PROFILE_STEP"]
    top_n    = config["SLAT_CANDIDATE_TOPN"]
    init_pct = config["SLAT_INIT_BAND_PCT"]

    init_y  = int(rh * init_pct)   # posisi awal estimasi (crop coords)
    prev_y  = None                  # tracking posisi edge sebelumnya

    raw_x, raw_y = [], []
    dbg_cands    = []   # (xf, yf, strength)
    dbg_rej      = []   # (xf, yf)

    for xc in range(0, rw, step):
        col = grad_m[:, xc].astype(np.float32)

        # Search band
        if prev_y is None:
            ylo = max(0, init_y - sr*2);  yhi = min(rh-1, init_y + sr*2)
        else:
            ylo = max(0, prev_y - sr);    yhi = min(rh-1, prev_y + sr)

        cands = _find_col_candidates(col, ylo, yhi, min_str, top_n)

        if not cands:
            ref = prev_y if prev_y is not None else init_y
            dbg_rej.append((rx+xc, ry+ref))
            continue

        msb = max(c[1] for c in cands)
        best_y, best_sc = None, -1.0

        for (cy_loc, cstr) in cands:
            dbg_cands.append((rx+xc, ry+cy_loc, cstr))
            sc = _score_candidate(cy_loc, cstr, msb, prev_y, config)
            if sc > best_sc:
                best_sc = sc;  best_y = cy_loc

        if best_y is None:
            ref = prev_y if prev_y is not None else init_y
            dbg_rej.append((rx+xc, ry+ref))
            continue

        raw_x.append(rx+xc);  raw_y.append(ry+best_y)
        prev_y = best_y

    # ── Panel debug 7 & 8 ────────────────────────────────────────────────────
    cvis = np.zeros((rh, rw, 3), np.uint8)
    for (xf, yf, cstr) in dbg_cands:
        inten = int(np.clip(cstr/255*255, 60, 255))
        yc, xcc = yf-ry, xf-rx
        if 0 <= yc < rh and 0 <= xcc < rw:
            cv2.circle(cvis, (xcc, yc), 2, (0, inten, inten), -1)
    for (xf, yf) in dbg_rej:
        yc, xcc = yf-ry, xf-rx
        if 0 <= yc < rh and 0 <= xcc < rw:
            cv2.circle(cvis, (xcc, yc), 1, (50, 0, 0), -1)
    cvis = cv2.bitwise_and(cvis, cvis, mask=crop_mask)

    svis = np.zeros((rh, rw, 3), np.uint8)
    if len(raw_x) >= 2:
        psel = np.array([[xi-rx, yi-ry] for xi, yi in zip(raw_x, raw_y)], np.int32)
        cv2.polylines(svis, [psel], False, (0, 255, 255), 2)
    svis = cv2.bitwise_and(svis, svis, mask=crop_mask)

    p1 = mkp(crop,    "1. Foto Asli+ROI (SLAT)", True)
    p2 = mkp(gray,    "2. Grayscale")
    p3 = mkp(enhanced,"3. CLAHE")
    p4 = mkp(denoised,"4. Bilateral")
    p5 = mkp(grad,    "5. Gradient (Sobel)")
    p6 = mkp(morph_m, "6. Morph Close [DEBUG - tdk dipakai]")
    p7 = mkp(cvis,    "7. Candidate Edges (cyan=kuat)", True)
    p8 = mkp(svis,    "8. Selected Profile (cyan=final)", True)

    filter_grid = np.vstack([np.hstack([p1, p2, p3, p4]),
                              np.hstack([p5, p6, p7, p8])])

    if len(raw_x) < 20:
        return None, None, dbg_cands, dbg_rej, filter_grid, (rx, ry, rw, rh, pts)

    rx_a = np.array(raw_x, np.float32)
    ry_a = np.array(raw_y, np.float32)

    # Corner trim
    tr = max(2, int(config["SLAT_CORNER_TRIM_PCT"] * len(rx_a)))
    rx_a, ry_a = rx_a[tr:-tr], ry_a[tr:-tr]

    # Outlier rejection konservatif
    px, py = _reject_slat_outliers(rx_a, ry_a, config)

    return px, py, dbg_cands, dbg_rej, filter_grid, (rx, ry, rw, rh, pts)


def analyze_slat_profile(px, py, dbg_cands, dbg_rej, filter_grid, roi_info, config):
    """
    Metrologi SLAT V4.3: P-V + Max Abs Deviation.
    Reference line = endpoint-to-endpoint profil fisik.
    OK/NG = (P-V > thresh_pv) OR (max_abs > thresh_max).
    """
    rx, ry, rw, rh, pts = roi_info
    tpv, tmx = config["THRESH_SLAT_PV"], config["THRESH_SLAT_MAX"]
    nc, nr    = len(dbg_cands), len(dbg_rej)

    if px is None or len(px) < 15:
        nv = len(px) if px is not None else 0
        print(f"\n{'='*55}\n  SLAT PROFILE DEBUG\n{'─'*55}")
        print(f"  Valid points       : {nv}  <- TERLALU SEDIKIT")
        print(f"  Total candidates   : {nc}")
        print(f"  Rejected cols      : {nr}")
        print(f"  STATUS             : GAGAL\n{'='*55}")
        return {'success': False, 'name': 'PLAT DASAR',
                'msg': f'Profil tidak cukup ({nv} titik, butuh >=15)',
                'n_valid': nv, 'n_candidates': nc, 'n_rejected': nr,
                'grid': filter_grid, 'pts': pts}

    nv = len(px)
    x1, y1 = float(px[0]),  float(py[0])
    x2, y2 = float(px[-1]), float(py[-1])
    sp = np.hypot(x2-x1, y2-y1)

    if sp < 1e-3:
        return {'success': False, 'name': 'PLAT DASAR', 'msg': 'Bentang terlalu pendek',
                'grid': filter_grid, 'pts': pts,
                'n_valid': nv, 'n_candidates': nc, 'n_rejected': nr}

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

    print(f"\n{'='*55}\n  SLAT PROFILE DEBUG\n{'─'*55}")
    print(f"  Valid points       : {nv}")
    print(f"  Total candidates   : {nc}")
    print(f"  Rejected cols      : {nr}")
    print(f"  P-V Flatness       : {pv:.2f} px  (threshold: {tpv} px)")
    print(f"  Max Abs Deviation  : {ma:.2f} px  (threshold: {tmx} px)")
    print(f"  STATUS             : {ss}")
    print(f"{'='*55}")

    return {'success': True, 'name': 'PLAT DASAR', 'pts': pts,
            'edge_x': px, 'edge_y': py,
            'x_left': x1, 'y_left': y1, 'x_right': x2, 'y_right': y2,
            'peak_x': pkx, 'peak_y_actual': pky,
            'proj_x': x1+t*dx_, 'proj_y': y1+t*dy_,
            'pv_flatness': pv, 'max_abs': ma,
            'is_ng': ng, 'status_str': ss, 'grid': filter_grid,
            'n_valid': nv, 'n_candidates': nc, 'n_rejected': nr}


def draw_slat_visuals(canvas, d):
    """
    Visualisasi SLAT V4.3:
      ROI outline    – cyan tipis
      Reference line – ORANGE
      Profile fisik  – KUNING TERANG
      Deviasi max    – MAGENTA (vektor + circle)
    """
    if not d.get('success'):
        return
    cv2.polylines(canvas, [d['pts']], True, (0, 200, 200), 1)
    # Reference line (ORANGE)
    cv2.line(canvas, (int(d['x_left']), int(d['y_left'])),
             (int(d['x_right']), int(d['y_right'])), (0, 165, 255), 3)
    # Profile fisik (KUNING)
    cv2.polylines(canvas,
                  [np.column_stack((d['edge_x'], d['edge_y'])).astype(np.int32)],
                  False, (255, 255, 0), 2)
    # Max deviation (MAGENTA)
    cv2.line(canvas, (int(d['proj_x']), int(d['proj_y'])),
             (int(d['peak_x']), int(d['peak_y_actual'])), (255, 0, 255), 4)
    cv2.circle(canvas, (int(d['peak_x']), int(d['peak_y_actual'])), 6, (255, 0, 255), -1)


# ═════════════════════════════════════════════════════════════════════════════
# KONTROL PROGRAM
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
        wn   = f"INSPEKSI SLAT & RIB V4.3 (ROBUST PROFILE) - [{fn}]"

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
              f"| radius={config['SLAT_SEARCH_RADIUS']}px "
              f"| min_str={config['SLAT_EDGE_MIN_STRENGTH']} "
              f"| max_jump={config['SLAT_MAX_JUMP']}px")
        print("📋 PANDUAN V4.3:")
        print("   [TAHAP 1] Klik >=4 sudut SIRIP (RIB)       → [ENTER]")
        print("   [TAHAP 2] Klik >=4 sudut PLAT DASAR (SLAT) → [ENTER]")
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

            # ── Keyboard ─────────────────────────────────────────────────────

            if key in [13, 32]:   # ENTER / SPACE
                if current_step == 1:
                    if len(points_sirip) >= 3:
                        current_step = 2
                        print("  [OK] ROI Sirip tersimpan! Klik sudut-sudut PLAT DASAR.")
                    else:
                        print("  ⚠️ Klik minimal 3 titik untuk SIRIP!")

                elif current_step == 2:
                    if len(points_slat) >= 3:
                        print("\n⚙️  Memproses V4.3 (Robust Profile)...")

                        # SIRIP: algoritma topmost (tidak diubah)
                        c_sir  = extract_clean_contour(frame, points_sirip, "SIRIP", config)
                        l_sir  = analyze_sirip_pv(c_sir, config)

                        # SLAT: robust candidate-scoring pipeline
                        px, py, dc, dr, sgrid, roi = \
                            extract_slat_profile_robust(frame, points_slat, config)
                        l_sla  = analyze_slat_profile(px, py, dc, dr, sgrid, roi, config)

                        if not l_sir.get('success') or not l_sla.get('success'):
                            msg = l_sir.get('msg','') if not l_sir.get('success') else l_sla.get('msg','')
                            print(f"  ⚠️ [ERROR] {msg}");  continue

                        # Simpan ke variabel sesi agar tombol P bisa akses
                        last_sirip   = l_sir
                        last_slat    = l_sla

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
                        print("📊 LAPORAN METROLOGI V4.3:")
                        print(f"   SIRIP      : {l_sir['status_str']}")
                        print(f"   → P-V: {l_sir['pv_flatness']:.1f}px (≤{t_spv}) | "
                              f"Max: {l_sir['max_abs']:.1f}px (≤{t_smx})")
                        print(f"   PLAT DASAR : {l_sla['status_str']}")
                        print(f"   → P-V: {l_sla['pv_flatness']:.1f}px (≤{t_lpv}) | "
                              f"Max: {l_sla['max_abs']:.1f}px (≤{t_lmx})")
                        print(f"   → Valid pts: {l_sla['n_valid']} | "
                              f"Candidates: {l_sla['n_candidates']} | "
                              f"Rejected cols: {l_sla['n_rejected']}")
                        print("="*70)

                        current_step = 3
                        base, ext  = os.path.splitext(image_path)
                        save_path  = f"{base}_hasil_v4_3{ext}"
                        cv2.imwrite(save_path, result_frame)
                        print(f"  💾 Disimpan: {os.path.basename(save_path)}")
                    else:
                        print("  ⚠️ Klik minimal 3 titik untuk PLAT DASAR!")

            elif key in [ord('p'), ord('P')] and current_step == 3:
                if last_sirip is not None and last_sirip.get('grid') is not None:
                    ws_ = "FILTER: SIRIP (6 Panel)"
                    cv2.namedWindow(ws_, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(ws_, 1500, 620)
                    cv2.imshow(ws_, last_sirip['grid'])
                if last_slat is not None and last_slat.get('grid') is not None:
                    wsl = "FILTER: PLAT DASAR (8 Panel V4.3)"
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
    print("  V4.3 - Robust Physical Profile Extraction")
    print("  SLAT : Gradient Candidate Scoring + Continuity Tracking")
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
