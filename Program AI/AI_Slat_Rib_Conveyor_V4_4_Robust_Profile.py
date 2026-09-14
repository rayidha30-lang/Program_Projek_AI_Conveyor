"""
AI_Slat_Rib_Conveyor_V4_4_Robust_Profile.py
AI / Computer Vision - Inspeksi Deformasi Slat & Rib Conveyor
V4.4 - Upper-Edge Locking & Full-Span Profile Extraction

Pembaruan Utama V4.4:
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
  SIRIP: Topmost-edge algorithm DIPERTAHANKAN 100% (proven, stabil).
  UI   : Manual 2-stage ROI, tombol keyboard, navigasi foto, HUD, dan file picker
         dipertahankan sepenuhnya.
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
# SLAT (PLAT DASAR) — Upper-Edge Locked Robust Profile Extraction V4.4
# ═════════════════════════════════════════════════════════════════════════════

def _find_col_upper_candidates(col_grad, col_sy, col_sx, col_mask, y_lo, y_hi, config):
    """
    Mendeteksi kandidat TEPI ATAS pada satu kolom:
    1. Berada di dalam crop_mask.
    2. Local maxima gradien vertikal.
    3. Sy > 3.0 (polaritas positif: transisi dari celah atas gelap -> plat logam).
       Tepi bawah memiliki Sy < 0 sehingga otomatis TERELIMINASI 100%.
    4. |Sy| >= SLAT_HORIZ_DOMINANCE * |Sx| (menolak goresan vertikal / noise karat).
    5. Kekuatan gradien >= threshold adaptif kolom.
    """
    lo = max(1, y_lo)
    hi = min(len(col_grad)-2, y_hi)
    if hi <= lo:
        return []

    cands_raw = []
    for y in range(lo, hi + 1):
        if col_mask[y] == 0:
            continue
        g   = float(col_grad[y])
        sy  = float(col_sy[y])
        sx  = float(col_sx[y])

        # Kunci tepi atas: Sobel Y harus positif (transisi masuk ke plat)
        if sy <= 3.0:
            continue

        # Dominansi orientasi horizontal
        if abs(sy) < config["SLAT_HORIZ_DOMINANCE"] * (abs(sx) + 1e-5):
            continue

        # Local maxima gradien (terhadap tetangga atas dan bawah)
        if g >= float(col_grad[y-1]) and g >= float(col_grad[y+1]):
            cands_raw.append((y, g, sy))

    if not cands_raw:
        return []

    # Threshold adaptif kolom: mampu menangkap area berkarat di kiri
    max_g = max(c[1] for c in cands_raw)
    col_thr = max(config["SLAT_MIN_STR_FLOOR"], max_g * config["SLAT_ADAPTIVE_RATIO"])

    valid = [c for c in cands_raw if c[1] >= col_thr]
    valid.sort(key=lambda c: c[1], reverse=True)
    return valid[:config["SLAT_CANDIDATE_TOPN"]]


def _score_slat_candidate(y, strength, max_str, prev_y, expected_y, rh, config):
    """
    Scoring kandidat SLAT:
    - 60% kontinuitas (jarak terhadap expected_y) + 40% kekuatan relatif.
    - Loncatan > SLAT_MAX_JUMP -> DISKUALIFIKASI MUTLAK (-1e6).
    """
    sr = config["SLAT_SEARCH_RADIUS"]
    mj = config["SLAT_MAX_JUMP"]
    ss = strength / max_str if max_str > 0 else 0.0

    dist = abs(y - expected_y)
    if dist > mj:
        return -1e6  # Hard reject: dilarang melompat ke tepi lain!

    cs = max(0.0, 1.0 - dist / sr)
    return 0.40 * ss + 0.60 * cs


def _find_best_anchor(col_data, rw, rh, config):
    """
    Mencari kolom jangkar (anchor) dengan tepi atas paling meyakinkan.
    Diutamakan di rentang 20% s/d 80% lebar plat di mana tepi atas biasanya solid.
    """
    best_score = -1.0
    best_xc = rw // 2
    best_cand = None

    x_start = int(rw * 0.15)
    x_end   = int(rw * 0.85)

    for xc in range(x_start, x_end, config["SLAT_PROFILE_STEP"]):
        if xc not in col_data or not col_data[xc]:
            continue
        cands = col_data[xc]
        for (y, g, sy) in cands:
            norm_str = min(1.0, g / 180.0)
            ub = max(0.0, 1.0 - (y / max(1.0, float(rh))))
            score = 0.65 * norm_str + 0.35 * ub
            if score > best_score:
                best_score = score
                best_xc = xc
                best_cand = (y, g, sy)

    if best_cand is None:
        # Fallback: cari dari semua kolom yang memiliki kandidat
        for xc in sorted(col_data.keys()):
            if col_data[xc]:
                best_xc = xc
                best_cand = col_data[xc][0]
                break

    return best_xc, best_cand


def _track_slat_profile_bidirectional(col_data, rw, rh, config):
    """
    Pelacakan dua arah dari anchor:
    - Pass maju: anchor -> kanan (rw)
    - Pass mundur: anchor -> kiri (0)
    Menggunakan proyeksi forward-slope saat dropout untuk mempertahankan kesinambungan.
    Menghasilkan profil 100% bentang plat.
    """
    anchor_xc, anchor_cand = _find_best_anchor(col_data, rw, rh, config)
    if anchor_cand is None:
        return [], None, None

    step = config["SLAT_PROFILE_STEP"]
    max_drop = config["SLAT_MAX_DROPOUT_STEPS"]

    # ── Pass 1: Dari anchor ke kanan ─────────────────────────────────────────
    right_pts = [(anchor_xc, anchor_cand[0])]
    prev_y = float(anchor_cand[0])
    slope = 0.0
    dropout_count = 0

    for xc in range(anchor_xc + step, rw, step):
        cands = col_data.get(xc, [])
        best_cand = None
        best_sc = -1e5
        max_str = max([c[1] for c in cands]) if cands else 1.0
        exp_y = prev_y + slope * step

        for (cy, cg, csy) in cands:
            sc = _score_slat_candidate(cy, cg, max_str, prev_y, exp_y, rh, config)
            if sc > best_sc:
                best_sc = sc
                best_cand = (cy, cg)

        if best_cand is not None and best_sc > 0:
            act_y = float(best_cand[0])
            new_slope = (act_y - prev_y) / float(step)
            slope = 0.7 * slope + 0.3 * np.clip(new_slope, -1.2, 1.2)
            prev_y = act_y
            dropout_count = 0
            right_pts.append((xc, int(round(prev_y))))
        else:
            # Dropout handling: teruskan estimasi lintasan
            dropout_count += 1
            if dropout_count <= max_drop:
                prev_y = float(np.clip(prev_y + slope * step, 0, rh - 1))
                right_pts.append((xc, int(round(prev_y))))
            else:
                break

    # ── Pass 2: Dari anchor ke kiri ──────────────────────────────────────────
    left_pts = []
    prev_y = float(anchor_cand[0])
    slope = 0.0
    dropout_count = 0

    for xc in range(anchor_xc - step, -1, -step):
        cands = col_data.get(xc, [])
        best_cand = None
        best_sc = -1e5
        max_str = max([c[1] for c in cands]) if cands else 1.0
        exp_y = prev_y + slope * (-step)

        for (cy, cg, csy) in cands:
            sc = _score_slat_candidate(cy, cg, max_str, prev_y, exp_y, rh, config)
            if sc > best_sc:
                best_sc = sc
                best_cand = (cy, cg)

        if best_cand is not None and best_sc > 0:
            act_y = float(best_cand[0])
            new_slope = (act_y - prev_y) / float(-step)
            slope = 0.7 * slope + 0.3 * np.clip(new_slope, -1.2, 1.2)
            prev_y = act_y
            dropout_count = 0
            left_pts.append((xc, int(round(prev_y))))
        else:
            dropout_count += 1
            if dropout_count <= max_drop:
                prev_y = float(np.clip(prev_y + slope * (-step), 0, rh - 1))
                left_pts.append((xc, int(round(prev_y))))
            else:
                break

    # Gabungkan: sisi kiri (dibalik) + sisi kanan
    full_path = list(reversed(left_pts)) + right_pts
    return full_path, anchor_xc, anchor_cand


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


def extract_slat_profile_robust_v44(frame, polygon_pts, config=FILTER_CONFIG):
    """
    SLAT Physical Profile Extraction V4.4:
    - Upper-Edge Locking (Sobel Y > 0 & horizontal dominance)
    - Per-column adaptive thresholding
    - Bidirectional Anchor Tracking (100% bentang plat)
    - 8-Panel Visualisasi Debug
    """
    res = _preprocess_roi(frame, polygon_pts, config)
    pts, h, w, rx, ry, rw, rh, crop, crop_mask, gray, enhanced, denoised, grad, morph_m, grad_m, sx_f, sy_f = res

    is_h = (rw >= rh)
    cy1, cy2, cx1, cx2 = _ctx_bounds(frame, ry, rh, rx, rw, is_h)
    PW = 400

    def mkp(s, t, b=False):
        return _make_panel(frame, pts, crop_mask, ry, rh, rx, rw, cy1, cy2, cx1, cx2, s, t, PW, b)

    if rw < 20 or rh < 10:
        d = np.zeros((200, PW, 3), np.uint8)
        cv2.putText(d, "ROI INVALID", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        grid = np.vstack([np.hstack([d]*4), np.hstack([d]*4)])
        return None, None, [], grid, (rx, ry, rw, rh, pts)

    # ── Deteksi Kandidat Upper-Edge di Setiap Kolom ──────────────────────────
    step = config["SLAT_PROFILE_STEP"]
    col_data = {}
    dbg_all_cands = []

    for xc in range(0, rw, step):
        col_g = grad_m[:, xc]
        col_sy = sy_f[:, xc]
        col_sx = sx_f[:, xc]
        col_m  = crop_mask[:, xc]

        cands = _find_col_upper_candidates(col_g, col_sy, col_sx, col_m, 0, rh-1, config)
        col_data[xc] = cands
        for (cy, cg, csy) in cands:
            dbg_all_cands.append((rx + xc, ry + cy, cg))

    # ── Bidirectional Tracking dari Anchor ───────────────────────────────────
    raw_path, anchor_xc, anchor_cand = _track_slat_profile_bidirectional(col_data, rw, rh, config)

    # ── Visualisasi 8-Panel Debug V4.4 ───────────────────────────────────────
    # Panel 5: Sobel Y (hanya polaritas positif = tepi atas)
    sy_pos = np.clip(sy_f, 0, 255).astype(np.uint8)
    sy_pos_m = cv2.bitwise_and(sy_pos, sy_pos, mask=crop_mask)

    # Panel 6: Directional mask (Sy > 3 & dominan horizontal)
    dir_mask = np.zeros((rh, rw), np.uint8)
    cond = (sy_f > 3.0) & (np.abs(sy_f) >= config["SLAT_HORIZ_DOMINANCE"] * (np.abs(sx_f) + 1e-5))
    dir_mask[cond & (crop_mask > 0)] = 255

    # Panel 7: Semua kandidat tepi atas & anchor (merah)
    cvis = np.zeros((rh, rw, 3), np.uint8)
    for (xf, yf, cg) in dbg_all_cands:
        yc, xc = yf - ry, xf - rx
        if 0 <= yc < rh and 0 <= xc < rw:
            cv2.circle(cvis, (xc, yc), 2, (255, 255, 0), -1)  # Cyan kandidat
    if anchor_cand is not None:
        cv2.circle(cvis, (anchor_xc, anchor_cand[0]), 5, (0, 0, 255), -1)  # Merah anchor
    cvis = cv2.bitwise_and(cvis, cvis, mask=crop_mask)

    # Panel 8: Profil fisik terpilih (kuning) & garis acuan (oranye)
    svis = np.zeros((rh, rw, 3), np.uint8)
    if len(raw_path) >= 2:
        psel = np.array([[pt[0], pt[1]] for pt in raw_path], np.int32)
        cv2.polylines(svis, [psel], False, (0, 255, 255), 2)
        # Garis acuan di dalam panel
        cv2.line(svis, (raw_path[0][0], raw_path[0][1]),
                 (raw_path[-1][0], raw_path[-1][1]), (0, 140, 255), 2)
    svis = cv2.bitwise_and(svis, svis, mask=crop_mask)

    p1 = mkp(crop,     "1. Foto Asli+ROI (SLAT)", True)
    p2 = mkp(gray,     "2. Grayscale")
    p3 = mkp(enhanced, "3. CLAHE")
    p4 = mkp(denoised, "4. Bilateral")
    p5 = mkp(sy_pos_m, "5. Sobel Y (Tepi Atas: Sy > 0)")
    p6 = mkp(dir_mask, "6. Mask Upper Dominance")
    p7 = mkp(cvis,     "7. Upper Candidates & Anchor", True)
    p8 = mkp(svis,     "8. Selected Profile & Ref Line", True)

    filter_grid = np.vstack([np.hstack([p1, p2, p3, p4]),
                              np.hstack([p5, p6, p7, p8])])

    if len(raw_path) < 20:
        return None, None, dbg_all_cands, filter_grid, (rx, ry, rw, rh, pts)

    raw_x = np.array([rx + pt[0] for pt in raw_path], np.float32)
    raw_y = np.array([ry + pt[1] for pt in raw_path], np.float32)

    # Corner trim
    tr = max(2, int(config["SLAT_CORNER_TRIM_PCT"] * len(raw_x)))
    raw_x, raw_y = raw_x[tr:-tr], raw_y[tr:-tr]

    # Outlier rejection konservatif
    px, py = _reject_slat_outliers(raw_x, raw_y, config)

    return px, py, dbg_all_cands, filter_grid, (rx, ry, rw, rh, pts)


def analyze_slat_profile(px, py, dbg_cands, filter_grid, roi_info, config):
    """
    Metrologi SLAT V4.4: P-V + Max Abs Deviation.
    Reference line = menghubungkan ujung kiri tepi atas ke ujung kanan tepi atas.
    OK/NG = (P-V > thresh_pv) OR (max_abs > thresh_max).
    """
    rx, ry, rw, rh, pts = roi_info
    tpv, tmx = config["THRESH_SLAT_PV"], config["THRESH_SLAT_MAX"]
    nc = len(dbg_cands)

    if px is None or len(px) < 15:
        nv = len(px) if px is not None else 0
        print(f"\n{'='*55}\n  SLAT PROFILE DEBUG (V4.4)\n{'─'*55}")
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

    print(f"\n{'='*55}\n  SLAT PROFILE DEBUG (V4.4 UPPER-EDGE LOCKED)\n{'─'*55}")
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
            'n_valid': nv, 'n_candidates': nc}


def draw_slat_visuals(canvas, d):
    """
    Visualisasi SLAT V4.4:
      ROI outline    – cyan tipis
      Reference line – ORANGE tebal (sejajar tepi atas plat)
      Profile fisik  – KUNING TERANG
      Deviasi max    – MAGENTA (vektor tegak lurus + circle)
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

            # ── Keyboard Handling ────────────────────────────────────────────

            if key in [13, 32]:   # ENTER / SPACE
                if current_step == 1:
                    if len(points_sirip) >= 3:
                        current_step = 2
                        print("  [OK] ROI Sirip tersimpan! Klik sudut-sudut PLAT DASAR.")
                    else:
                        print("  ⚠️ Klik minimal 3 titik untuk SIRIP!")

                elif current_step == 2:
                    if len(points_slat) >= 3:
                        print("\n⚙️  Memproses V4.4 (Upper-Edge Locked)...")

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
                    else:
                        print("  ⚠️ Klik minimal 3 titik untuk PLAT DASAR!")

            elif key in [ord('p'), ord('P')] and current_step == 3:
                if last_sirip is not None and last_sirip.get('grid') is not None:
                    ws_ = "FILTER: SIRIP (6 Panel)"
                    cv2.namedWindow(ws_, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(ws_, 1500, 620)
                    cv2.imshow(ws_, last_sirip['grid'])
                if last_slat is not None and last_slat.get('grid') is not None:
                    wsl = "FILTER: PLAT DASAR (8 Panel V4.4)"
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
