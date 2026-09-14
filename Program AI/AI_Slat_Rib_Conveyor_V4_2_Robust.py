import cv2
import numpy as np
import os
import sys
import glob

# Try importing Tkinter for native Windows File Dialog
try:
    import tkinter as tka
    from tkinter import filedialog
    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False

# Global state variables
current_step = 1            # 1: Pilih Sirip (Rib), 2: Pilih Plat Dasar (Slat), 3: Selesai Analisis
points_sirip = []           # Koordinat poligon ROI 1 (Sirip)
points_slat = []            # Koordinat poligon ROI 2 (Plat Dasar)
current_mouse_pos = (0, 0)  # Posisi live kursor mouse


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
    """Membuka dialog File Explorer Windows untuk memilih foto."""
    if not HAS_TKINTER:
        return None
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        if initial_dir and os.path.isdir(initial_dir):
            file_path = filedialog.askopenfilename(
                initialdir=initial_dir,
                title="Pilih Foto Slat / Rib Conveyor",
                filetypes=[
                    ("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.webp;*.tiff"),
                    ("Semua File", "*.*")
                ]
            )
        else:
            file_path = filedialog.askopenfilename(
                title="Pilih Foto Slat / Rib Conveyor",
                filetypes=[
                    ("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.webp;*.tiff"),
                    ("Semua File", "*.*")
                ]
            )
        root.destroy()
        return clean_path_string(file_path) if file_path else None
    except Exception as e:
        print(f"  [Info] Dialog GUI tidak dapat dibuka: {e}")
        return None


def get_sibling_images(current_path):
    """Mendapatkan daftar foto yang ada di dalam folder yang sama untuk navigasi cepat."""
    if not current_path or not os.path.exists(current_path):
        return [], -1
    folder = os.path.dirname(os.path.abspath(current_path))
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    try:
        files = [
            os.path.join(folder, f) for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in valid_exts
        ]
        files.sort()
        curr_norm = os.path.abspath(current_path)
        try:
            curr_idx = files.index(curr_norm)
        except ValueError:
            curr_idx = -1
        return files, curr_idx
    except Exception:
        return [], -1


def mouse_callback(event, x, y, flags, param):
    global current_step, points_sirip, points_slat, current_mouse_pos
    current_mouse_pos = (x, y)
    
    # KLIK KIRI: Tambah titik poligon ke ROI yang sedang aktif
    if event == cv2.EVENT_LBUTTONDOWN:
        if current_step == 1:
            points_sirip.append((x, y))
            print(f"  [+] [SIRIP] Titik #{len(points_sirip)}: ({x}, {y})")
        elif current_step == 2:
            points_slat.append((x, y))
            print(f"  [+] [PLAT DASAR] Titik #{len(points_slat)}: ({x}, {y})")
            
    # KLIK KANAN: Undo / Hapus titik terakhir dari ROI yang sedang aktif
    elif event == cv2.EVENT_RBUTTONDOWN:
        if current_step == 1 and len(points_sirip) > 0:
            removed = points_sirip.pop()
            print(f"  [-] [SIRIP] Titik #{len(points_sirip)+1} dihapus: {removed}")
        elif current_step == 2:
            if len(points_slat) > 0:
                removed = points_slat.pop()
                print(f"  [-] [PLAT DASAR] Titik #{len(points_slat)+1} dihapus: {removed}")
            else:
                current_step = 1
                print("  [<--] Kembali ke pengeditan ROI 1 (SIRIP).")



def _select_profile_candidate(candidates, strengths, previous_value=None,
                              expected_value=None, search_radius=18,
                              continuity_limit=12):
    """Memilih kandidat edge berdasarkan kekuatan gradien + kedekatan posisi."""
    if len(candidates) == 0:
        return None

    candidates = np.asarray(candidates, dtype=np.float32)
    strengths = np.asarray(strengths, dtype=np.float32)

    # Normalisasi kekuatan gradien agar dapat digabungkan dengan jarak.
    s_min, s_max = float(strengths.min()), float(strengths.max())
    if s_max > s_min:
        strength_score = (strengths - s_min) / (s_max - s_min)
    else:
        strength_score = np.ones_like(strengths)

    reference = previous_value if previous_value is not None else expected_value
    if reference is None:
        # Fallback: kandidat dengan gradien terkuat.
        return float(candidates[int(np.argmax(strengths))])

    distance = np.abs(candidates - reference)

    # Batasi lompatan yang tidak masuk akal setelah profil mulai terbentuk.
    if previous_value is not None:
        valid = distance <= continuity_limit
        if np.any(valid):
            candidates = candidates[valid]
            strength_score = strength_score[valid]
            distance = distance[valid]
        else:
            return None

    # Kandidat dekat referensi lebih disukai, tetapi edge yang lebih kuat
    # tetap dapat menang jika perbedaannya signifikan.
    radius = max(float(search_radius), 1.0)
    proximity_score = np.clip(1.0 - distance / radius, 0.0, 1.0)
    score = 0.65 * proximity_score + 0.35 * strength_score

    return float(candidates[int(np.argmax(score))])


def _extract_profile_robust(crop, crop_mask, rx, ry, rw, rh, is_horizontal):
    """Ekstraksi profil dengan multi-candidate edge + continuity check."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.bilateralFilter(enhanced, d=7, sigmaColor=50, sigmaSpace=50)

    sobel_x = cv2.Sobel(denoised, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(denoised, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(sobel_x, sobel_y)
    grad_mag_u8 = np.uint8(np.clip(grad_mag, 0, 255))

    _, thresh = cv2.threshold(grad_mag_u8, 35, 255, cv2.THRESH_BINARY)
    # Kernel diperkecil agar detail edge/deformasi kecil tidak terlalu tertutup.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    clean_edges = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    clean_edges_masked = cv2.bitwise_and(clean_edges, clean_edges, mask=crop_mask)

    # Grid visualisasi.
    tw, th = 380, 140

    def prep(img, title):
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()
        r = cv2.resize(bgr, (tw, th))
        cv2.rectangle(r, (0, 0), (tw, 24), (15, 15, 15), -1)
        cv2.putText(r, title, (8, 17), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (0, 255, 255), 1)
        return r

    row1 = np.hstack([
        prep(crop, "1. Crop Asli"),
        prep(gray, "2. Grayscale"),
        prep(enhanced, "3. CLAHE")
    ])
    row2 = np.hstack([
        prep(denoised, "4. Bilateral"),
        prep(grad_mag_u8, "5. Isotropic Gradient"),
        prep(clean_edges_masked, "6. Robust Edge Mask")
    ])
    filter_grid = np.vstack([row1, row2])

    raw_x, raw_y = [], []

    # Pass 1: kumpulkan semua kandidat edge untuk mendapatkan posisi awal.
    scan_records = []
    if is_horizontal:
        for x_col in range(0, rw, 2):
            row = clean_edges_masked[:, x_col]
            idx = np.where(row > 0)[0]
            if len(idx):
                # Kelompokkan edge yang berdekatan; ambil pusat tiap kelompok.
                groups = np.split(idx, np.where(np.diff(idx) > 2)[0] + 1)
                candidates = [int(np.median(g)) for g in groups if len(g)]
                strengths = [float(np.max(grad_mag[g, x_col])) for g in groups if len(g)]
                scan_records.append((x_col, candidates, strengths))
    else:
        for y_row in range(0, rh, 2):
            row = clean_edges_masked[y_row, :]
            idx = np.where(row > 0)[0]
            if len(idx):
                groups = np.split(idx, np.where(np.diff(idx) > 2)[0] + 1)
                candidates = [int(np.median(g)) for g in groups if len(g)]
                strengths = [float(np.max(grad_mag[y_row, g])) for g in groups if len(g)]
                scan_records.append((y_row, candidates, strengths))

    if len(scan_records) < 20:
        return None, None, None, filter_grid

    # Posisi awal dibuat dari median kandidat, sehingga tidak selalu memilih
    # edge pertama seperti versi V4.0.
    all_candidates = np.concatenate([np.asarray(r[1], dtype=np.float32)
                                      for r in scan_records])
    expected_value = float(np.median(all_candidates))

    previous_value = None
    for scan_pos, candidates, strengths in scan_records:
        selected = _select_profile_candidate(
            candidates,
            strengths,
            previous_value=previous_value,
            expected_value=expected_value,
            search_radius=max(18, int(0.08 * max(rw, rh))),
            continuity_limit=max(10, int(0.05 * max(rw, rh)))
        )

        if selected is None:
            continue

        if is_horizontal:
            raw_x.append(rx + scan_pos)
            raw_y.append(ry + selected)
        else:
            raw_x.append(rx + selected)
            raw_y.append(ry + scan_pos)

        previous_value = selected

    raw_x = np.asarray(raw_x, dtype=np.float32)
    raw_y = np.asarray(raw_y, dtype=np.float32)

    if len(raw_x) < 20:
        return None, None, None, filter_grid

    return raw_x, raw_y, filter_grid, None


def extract_clean_contour(frame, polygon_pts, name="KOMPONEN"):
    """
    V4.1/V4.2: Ekstraksi profil untuk foto dengan angle yang belum fixed.

    Tetap menggunakan manual ROI, tetapi pemilihan edge diperbaiki:
    1. CLAHE + bilateral + isotropic gradient.
    2. Multi-candidate edge, bukan edge pertama.
    3. Kandidat dipilih berdasarkan edge strength + proximity.
    4. Continuity check menolak lompatan edge yang tidak realistis.
    5. Median + smoothing tetap digunakan setelah trimming.
    """
    pts = np.asarray(polygon_pts, dtype=np.int32)
    h, w = frame.shape[:2]

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    rx, ry, rw, rh = cv2.boundingRect(pts)
    if rw < 20 or rh < 10:
        return None, None, None, rx, ry, rw, rh, pts, None, True

    crop = frame[ry:ry + rh, rx:rx + rw]
    crop_mask = mask[ry:ry + rh, rx:rx + rw]

    is_horizontal = rw >= rh
    raw_x, raw_y, filter_grid, _ = _extract_profile_robust(
        crop, crop_mask, rx, ry, rw, rh, is_horizontal
    )

    if raw_x is None:
        return None, None, None, rx, ry, rw, rh, pts, filter_grid, is_horizontal

    # Smart Corner Trimming.
    trim = max(2, int(0.06 * len(raw_x)))
    if len(raw_x) <= 2 * trim + 5:
        return None, None, None, rx, ry, rw, rh, pts, filter_grid, is_horizontal

    edge_x = raw_x[trim:-trim]
    edge_y_raw = raw_y[trim:-trim]

    # Median + moving average smoothing.
    k_med = min(9, max(3, len(edge_y_raw) // 5 * 2 + 1))
    if k_med > len(edge_y_raw):
        k_med = len(edge_y_raw) if len(edge_y_raw) % 2 == 1 else len(edge_y_raw) - 1
    k_med = max(3, k_med)

    pad = k_med // 2
    target = edge_y_raw if is_horizontal else edge_x
    padded = np.pad(target, (pad, pad), mode="edge")
    med = np.array([
        np.median(padded[i:i + k_med])
        for i in range(len(target))
    ], dtype=np.float32)

    smooth_w = min(7, len(med))
    if smooth_w % 2 == 0:
        smooth_w -= 1
    smooth_w = max(3, smooth_w)

    kernel_s = np.ones(smooth_w, dtype=np.float32) / smooth_w
    smooth_val = np.convolve(med, kernel_s, mode="same")
    half = smooth_w // 2
    smooth_val[:half] = med[:half]
    smooth_val[-half:] = med[-half:]

    if is_horizontal:
        clean_x = edge_x
        clean_y = smooth_val
    else:
        clean_x = smooth_val
        clean_y = edge_y_raw

    return clean_x, clean_y, edge_y_raw, rx, ry, rw, rh, pts, filter_grid, is_horizontal


def _base_component_result(contour_data, name):
    """Validasi data contour yang dipakai oleh analisis komponen."""
    if contour_data is None or contour_data[0] is None:
        return None, {
            'success': False,
            'name': name,
            'msg': 'Garis tepi tidak cukup terdeteksi'
        }

    clean_x, clean_y, _, rx, ry, rw, rh, pts, grid, is_horizontal = contour_data
    if len(clean_x) < 15:
        return None, {
            'success': False,
            'name': name,
            'msg': 'Garis tepi tidak cukup terdeteksi'
        }

    return (clean_x, clean_y, rx, ry, rw, rh, pts, grid, is_horizontal), None


def analyze_slat_flatness(contour_data, name="PLAT DASAR",
                          threshold_pv=18.0, threshold_max=12.0):
    """
    Analisis SLAT: deformasi diperlakukan sebagai flatness/profile deviation.

    Decision logic:
        NG jika P-V > threshold_pv ATAU Max Absolute Deviation > threshold_max.

    Reference masih dibuat dari endpoint profil saat ini karena kamera/angle
    belum fixed. Ini hanya reference sementara untuk feasibility.
    """
    data, error = _base_component_result(contour_data, name)
    if error:
        return error

    clean_x, clean_y, rx, ry, rw, rh, pts, grid, is_horizontal = data

    x1, y1 = float(clean_x[0]), float(clean_y[0])
    x2, y2 = float(clean_x[-1]), float(clean_y[-1])
    span_len = np.hypot(x2 - x1, y2 - y1)

    if span_len < 1e-3:
        return {'success': False, 'name': name, 'msg': 'Bentang komponen terlalu pendek'}

    perp_dist = (
        (y2 - y1) * clean_x
        - (x2 - x1) * clean_y
        + x2 * y1
        - y2 * x1
    ) / span_len

    max_pos = float(np.max(perp_dist))
    min_neg = float(np.min(perp_dist))
    pv_flatness = max_pos - min_neg
    max_abs = float(np.max(np.abs(perp_dist)))

    is_ng = (pv_flatness > threshold_pv) or (max_abs > threshold_max)

    if is_ng:
        if abs(max_pos) > abs(min_neg) * 1.3:
            dir_str = f"BENGKOK (+{max_pos:.1f}px)"
        elif abs(min_neg) > abs(max_pos) * 1.3:
            dir_str = f"BENGKOK ({min_neg:.1f}px)"
        else:
            dir_str = f"BERGELOMBANG (P-V {pv_flatness:.1f}px)"
        status_str = f"NG ({dir_str})"
    else:
        status_str = f"OK (P-V {pv_flatness:.1f}px)"

    idx_max = int(np.argmax(np.abs(perp_dist)))
    peak_x = float(clean_x[idx_max])
    peak_y_actual = float(clean_y[idx_max])

    dx_line = x2 - x1
    dy_line = y2 - y1
    t = (
        (peak_x - x1) * dx_line
        + (peak_y_actual - y1) * dy_line
    ) / (span_len ** 2)

    proj_x = x1 + t * dx_line
    proj_y = y1 + t * dy_line

    return {
        'success': True,
        'name': name,
        'pts': pts,
        'edge_x': clean_x,
        'edge_y': clean_y,
        'x_left': x1, 'y_left': y1,
        'x_right': x2, 'y_right': y2,
        'peak_x': peak_x,
        'peak_y_actual': peak_y_actual,
        'proj_x': proj_x,
        'proj_y': proj_y,
        'pv_flatness': pv_flatness,
        'max_abs': max_abs,
        'is_ng': is_ng,
        'status_str': status_str,
        'grid': grid,
        'metric_primary': 'P-V Flatness',
        'threshold_pv': threshold_pv,
        'threshold_max': threshold_max
    }


def analyze_rib_geometry(contour_data, name="SIRIP",
                         threshold_max=15.0, threshold_angle=2.0):
    """
    Analisis RIB: line-fit + residual deviation.

    Untuk tahap foto/manual ROI, reference absolut belum digunakan.
    Kita gunakan:
      - robust line fit sebagai baseline lokal,
      - max absolute residual sebagai indikator deformasi,
      - angle hanya sebagai informasi geometri.

    Decision utama tetap residual deviation agar perubahan angle kamera
    tidak otomatis menjadi NG.
    """
    data, error = _base_component_result(contour_data, name)
    if error:
        return error

    clean_x, clean_y, rx, ry, rw, rh, pts, grid, is_horizontal = data
    points = np.column_stack((clean_x, clean_y)).astype(np.float32)

    line = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x0, y0 = [float(v) for v in line.flatten()]

    norm = np.hypot(vx, vy)
    vx, vy = vx / norm, vy / norm

    # Signed perpendicular residual terhadap fitted line.
    residual = (-vy * (clean_x - x0) + vx * (clean_y - y0))
    max_abs = float(np.max(np.abs(residual)))

    angle_deg = float(np.degrees(np.arctan2(vy, vx)))

    t1 = (clean_x[0] - x0) * vx + (clean_y[0] - y0) * vy
    t2 = (clean_x[-1] - x0) * vx + (clean_y[-1] - y0) * vy

    x1, y1 = x0 + t1 * vx, y0 + t1 * vy
    x2, y2 = x0 + t2 * vx, y0 + t2 * vy

    is_ng = max_abs > threshold_max

    if is_ng:
        idx_max = int(np.argmax(np.abs(residual)))
        sign = "+" if residual[idx_max] >= 0 else ""
        status_str = f"NG (DEFORMASI {sign}{residual[idx_max]:.1f}px)"
    else:
        status_str = f"OK (Deviasi {max_abs:.1f}px)"

    idx_max = int(np.argmax(np.abs(residual)))
    peak_x = float(clean_x[idx_max])
    peak_y_actual = float(clean_y[idx_max])

    t = (
        (peak_x - x0) * vx
        + (peak_y_actual - y0) * vy
    )
    proj_x = x0 + t * vx
    proj_y = y0 + t * vy

    return {
        'success': True,
        'name': name,
        'pts': pts,
        'edge_x': clean_x,
        'edge_y': clean_y,
        'x_left': x1, 'y_left': y1,
        'x_right': x2, 'y_right': y2,
        'peak_x': peak_x,
        'peak_y_actual': peak_y_actual,
        'proj_x': proj_x,
        'proj_y': proj_y,
        'pv_flatness': float(np.max(residual) - np.min(residual)),
        'max_abs': max_abs,
        'angle_deg': angle_deg,
        'is_ng': is_ng,
        'status_str': status_str,
        'grid': grid,
        'metric_primary': 'Max Absolute Residual',
        'threshold_max': threshold_max,
        'threshold_angle': threshold_angle
    }


def draw_component_visuals(canvas, comp_data, color_theme):
    """Menggambar garis acuan, kontur deteksi, dan vektor lendutan terbesar."""
    cv2.polylines(canvas, [comp_data['pts']], isClosed=True, color=color_theme, thickness=1)
    
    # Garis acuan ideal
    cv2.line(canvas, (int(comp_data['x_left']), int(comp_data['y_left'])), 
             (int(comp_data['x_right']), int(comp_data['y_right'])), color_theme, 3)
             
    # Kontur tepi fisik terdeteksi
    pts_contour = np.column_stack((comp_data['edge_x'], comp_data['edge_y'])).astype(np.int32)
    cv2.polylines(canvas, [pts_contour], isClosed=False, color=(255, 255, 0), thickness=2)
    
    # Garis vektor lendutan di titik paling kritis
    pt_proj = (int(comp_data['proj_x']), int(comp_data['proj_y']))
    pt_actual = (int(comp_data['peak_x']), int(comp_data['peak_y_actual']))
    cv2.line(canvas, pt_proj, pt_actual, (255, 0, 255), 4)
    cv2.circle(canvas, pt_actual, 5, (255, 0, 255), -1)


def reset_state():
    """Mengosongkan semua state titik ROI dan langkah."""
    global current_step, points_sirip, points_slat
    current_step = 1
    points_sirip = []
    points_slat = []


def safe_destroy_all_windows():
    """Menutup jendela OpenCV dengan tuntas pada sistem operasi Windows."""
    cv2.destroyAllWindows()
    for _ in range(5):
        cv2.waitKey(10)


def interactive_multi_roi_session(initial_image_path, thresh_sirip=15.0, thresh_slat=18.0,
                                  thresh_sirip_angle=2.0):
    global current_step, points_sirip, points_slat, current_mouse_pos

    image_path = initial_image_path

    while True:
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"\n❌ [ERROR] Gambar tidak dapat dibuka: '{image_path}'")
            print("   Mencoba membuka dialog file picker...")
            alt_path = choose_file_dialog()
            if alt_path and os.path.exists(alt_path):
                image_path = alt_path
                continue
            else:
                print("   [Info] Tidak ada file gambar yang dipilih. Menghentikan program.")
                break

        h, w = frame.shape[:2]
        file_name = os.path.basename(image_path)
        window_name = f"INSPEKSI SLAT & RIB V4.2 - [{file_name}]"
        
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # Atur ukuran jendela proporsional dengan layar
        win_w = min(1360, max(700, w))
        win_h = min(850, int(win_w * (h / w)))
        cv2.resizeWindow(window_name, win_w, win_h)
        cv2.setMouseCallback(window_name, mouse_callback)

        reset_state()
        result_frame = None
        last_sirip = None
        last_slat = None

        print("\n" + "="*75)
        print(f"📷 MEMBUKA FOTO: {file_name} ({w}x{h} px)")
        print("📋 PANDUAN PENGUKURAN MULTI-ROI V4.2:")
        print("   TAHAP 1: Klik 4 sudut pada SIRIP (RIB)      -> Tekan [ENTER]")
        print("   TAHAP 2: Klik 4 sudut pada PLAT DASAR (SLAT)-> Tekan [ENTER]")
        print("   KONTROL : [Klik Kiri] Tambah Titik | [Klik Kanan] Undo")
        print("             [P] Lihat Filter   | [R] Reset ROI       | [Q] Keluar")
        print("             [N] Ganti Foto     | [-> / D] Foto Maju  | [<- / A] Foto Mundur")
        print("="*75)

        next_action = None

        while True:
            display = frame.copy()

            if current_step == 1:
                for i, pt in enumerate(points_sirip):
                    cv2.circle(display, pt, 5, (0, 140, 255), -1)
                    cv2.putText(display, f"S{i+1}", (pt[0] + 8, pt[1] - 8), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)

                if len(points_sirip) > 1:
                    for i in range(len(points_sirip) - 1):
                        cv2.line(display, points_sirip[i], points_sirip[i+1], (0, 140, 255), 2)
                    cv2.line(display, points_sirip[-1], current_mouse_pos, (180, 180, 180), 1)

                cv2.rectangle(display, (0, 0), (w, 42), (20, 20, 20), -1)
                cv2.putText(display, f"[TAHAP 1/2: PILIH SIRIP/RIB] Titik: {len(points_sirip)} | [Klik Kiri]: Tambah | [ENTER]: Lanjut ke Plat", 
                            (15, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 180, 255), 2)

                cv2.imshow(window_name, display)
                key = cv2.waitKey(20) & 0xFF

            elif current_step == 2:
                if len(points_sirip) >= 3:
                    cv2.polylines(display, [np.array(points_sirip, dtype=np.int32)], isClosed=True, color=(0, 140, 255), thickness=2)
                    cv2.putText(display, "ROI 1: SIRIP (TERKUNCI)", (points_sirip[0][0], max(30, points_sirip[0][1] - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 140, 255), 2)

                for i, pt in enumerate(points_slat):
                    cv2.circle(display, pt, 5, (0, 255, 255), -1)
                    cv2.putText(display, f"P{i+1}", (pt[0] + 8, pt[1] - 8), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                if len(points_slat) > 1:
                    for i in range(len(points_slat) - 1):
                        cv2.line(display, points_slat[i], points_slat[i+1], (0, 255, 255), 2)
                    cv2.line(display, points_slat[-1], current_mouse_pos, (180, 180, 180), 1)

                cv2.rectangle(display, (0, 0), (w, 42), (20, 20, 20), -1)
                cv2.putText(display, f"[TAHAP 2/2: PILIH PLAT DASAR/SLAT] Titik: {len(points_slat)} | [Klik Kiri]: Tambah | [ENTER]: Selesai", 
                            (15, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2)

                cv2.imshow(window_name, display)
                key = cv2.waitKey(20) & 0xFF

            else:
                cv2.imshow(window_name, result_frame)
                key = cv2.waitKey(30) & 0xFF

            # =================== NAVIGASI & KONTROL KEYBOARD ===================

            # ENTER / SPASI: Lanjut Tahap / Hitung
            if key in [13, 32]:
                if current_step == 1:
                    if len(points_sirip) >= 3:
                        current_step = 2
                        print("  [OK] ROI Sirip tersimpan! Sekarang silakan klik sudut-sudut PLAT DASAR.")
                    else:
                        print("  ⚠️ Klik minimal 3 atau 4 titik untuk menandai area Sirip!")

                elif current_step == 2:
                    if len(points_slat) >= 3:
                        print("\n⚙️ Memproses analisis metrologi V4.2 (Robust Edge + Component-Specific Logic)...")
                        
                        contour_sirip = extract_clean_contour(frame, points_sirip, name="SIRIP")
                        contour_slat = extract_clean_contour(frame, points_slat, name="PLAT DASAR")

                        last_sirip = analyze_rib_geometry(
                            contour_sirip,
                            name="SIRIP",
                            threshold_max=thresh_sirip,
                            threshold_angle=thresh_sirip_angle
                        )
                        last_slat = analyze_slat_flatness(
                            contour_slat,
                            name="PLAT DASAR",
                            threshold_pv=thresh_slat,
                            threshold_max=12.0
                        )

                        if not last_sirip.get('success', False) or not last_slat.get('success', False):
                            msg = last_sirip.get('msg') if not last_sirip.get('success') else last_slat.get('msg')
                            print(f"  ⚠️ [ERROR] {msg}")
                            continue

                        result_frame = frame.copy()
                        draw_component_visuals(result_frame, last_sirip, color_theme=(0, 140, 255))
                        draw_component_visuals(result_frame, last_slat, color_theme=(0, 255, 255))

                        # Status Card HUD Ganda
                        color_sirip = (0, 0, 255) if last_sirip['is_ng'] else (0, 255, 0)
                        txt_sirip = f"SIRIP     : {last_sirip['status_str']}"
                        cv2.rectangle(result_frame, (15, 15), (660, 52), (15, 15, 15), -1)
                        cv2.rectangle(result_frame, (15, 15), (660, 52), color_sirip, 2)
                        cv2.putText(result_frame, txt_sirip, (25, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color_sirip, 2)

                        color_slat = (0, 0, 255) if last_slat['is_ng'] else (0, 255, 0)
                        txt_slat = f"PLAT DASAR: {last_slat['status_str']}"
                        cv2.rectangle(result_frame, (15, 58), (660, 95), (15, 15, 15), -1)
                        cv2.rectangle(result_frame, (15, 58), (660, 95), color_slat, 2)
                        cv2.putText(result_frame, txt_slat, (25, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color_slat, 2)

                        cv2.rectangle(result_frame, (0, h - 35), (w, h), (25, 25, 25), -1)
                        cv2.putText(result_frame, "[P]: Filter | [R]: Ulang | [N]: Ganti Foto | [->]: Next Foto | [Q]: Keluar", 
                                    (20, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

                        print("\n" + "="*70)
                        print("📊 LAPORAN HASIL METROLOGI V4.2:")
                        print(f"   [1] SIRIP (RIB)     : {txt_sirip}")
                        print(f"       -> Max Abs Residual: {last_sirip['max_abs']:.1f} px (Batas: {thresh_sirip} px) | Angle: {last_sirip['angle_deg']:.2f}°")
                        print(f"   [2] PLAT DASAR      : {txt_slat}")
                        print(f"       -> P-V Flatness: {last_slat['pv_flatness']:.1f} px | Max Abs Dev: {last_slat['max_abs']:.1f} px | Batas P-V: {thresh_slat} px | Batas Max: 12.0 px")
                        print("="*70)

                        current_step = 3
                        base_name, ext = os.path.splitext(image_path)
                        save_out = f"{base_name}_hasil_v42_dual{ext}"
                        cv2.imwrite(save_out, result_frame)
                        print(f"  💾 Hasil visual disimpan ke: {os.path.basename(save_out)}")
                    else:
                        print("  ⚠️ Klik minimal 3 atau 4 titik untuk menandai area Plat Dasar!")

            # Tombol 'P': Lihat Filter Preprocessing
            elif key in [ord('p'), ord('P')] and current_step == 3:
                if last_sirip is not None and last_sirip.get('grid') is not None:
                    cv2.imshow("TAHAPAN FILTER: SIRIP", last_sirip['grid'])
                if last_slat is not None and last_slat.get('grid') is not None:
                    cv2.imshow("TAHAPAN FILTER: PLAT DASAR", last_slat['grid'])

            # Tombol 'R': Reset ROI
            elif key in [ord('r'), ord('R')]:
                print("\n🔄 [RESET] Mengulang pemilihan titik Sirip & Plat Dasar...")
                reset_state()
                continue

            # Tombol 'Q' / ESC: Keluar dari Program
            elif key in [ord('q'), ord('Q'), 27]:
                print("\n[INFO] Menutup program inspeksi.")
                safe_destroy_all_windows()
                return

            # Tombol 'N': Ganti Foto Baru (Dialog File Explorer GUI + Fallback Console)
            elif key in [ord('n'), ord('N')]:
                print("\n📂 Membuka dialog pemilihan foto...")
                curr_folder = os.path.dirname(os.path.abspath(image_path))
                new_path = choose_file_dialog(initial_dir=curr_folder)
                
                # Jika pengguna menutup dialog tanpa memilih file, minta lewat terminal
                if not new_path:
                    print("-" * 50)
                    try:
                        inp = input("👉 Atau Masukkan / Drag & Drop path foto BARU di sini: ")
                        new_path = clean_path_string(inp)
                    except (EOFError, KeyboardInterrupt):
                        new_path = None
                    print("-" * 50)

                if new_path and os.path.exists(new_path):
                    image_path = new_path
                    safe_destroy_all_windows()
                    break
                elif new_path:
                    print(f"⚠️ File tidak ditemukan: '{new_path}'")
                else:
                    print("ℹ️ Penggantian foto dibatalkan.")

            # Navigasi Foto Sibling: Foto Selanjutnya [->] atau [D]
            elif key in [ord('d'), ord('D'), 83]:  # 83 adalah Right Arrow pada beberapa platform OpenCV
                siblings, curr_idx = get_sibling_images(image_path)
                if len(siblings) > 1 and curr_idx != -1:
                    next_idx = (curr_idx + 1) % len(siblings)
                    image_path = siblings[next_idx]
                    print(f"\n⏩ [NEXT] Berpindah ke foto ({next_idx + 1}/{len(siblings)}): {os.path.basename(image_path)}")
                    safe_destroy_all_windows()
                    break
                else:
                    print("\nℹ️ Tidak ada foto lain di folder yang sama.")

            # Navigasi Foto Sibling: Foto Sebelumnya [<-] atau [A]
            elif key in [ord('a'), ord('A'), 81]:  # 81 adalah Left Arrow pada beberapa platform OpenCV
                siblings, curr_idx = get_sibling_images(image_path)
                if len(siblings) > 1 and curr_idx != -1:
                    prev_idx = (curr_idx - 1) % len(siblings)
                    image_path = siblings[prev_idx]
                    print(f"\n⏪ [PREV] Berpindah ke foto ({prev_idx + 1}/{len(siblings)}): {os.path.basename(image_path)}")
                    safe_destroy_all_windows()
                    break
                else:
                    print("\nℹ️ Tidak ada foto lain di folder yang sama.")


if __name__ == "__main__":
    print("\n" + "#"*75)
    print("  PROGRAM INSPEKSI DUAL SLAT RIB CONVEYOR (PURE CV - V4.2)")
    print("  Robust Edge Extraction + Component-Specific Metrology")
    print("#"*75)

    img_path = None
    if len(sys.argv) > 1:
        img_path = clean_path_string(sys.argv[1])

    # Jika tidak ada argumen di CLI, buka dialog file picker Windows
    if not img_path or not os.path.exists(img_path):
        print("\n📂 Membuka dialog pemilih file gambar...")
        img_path = choose_file_dialog()

    # Fallback ke terminal jika dialog dibatalkan / gagal
    if not img_path or not os.path.exists(img_path):
        print("-" * 55)
        try:
            inp = input("👉 Masukkan / Drag & Drop path foto gambar di sini: ")
            img_path = clean_path_string(inp)
        except (EOFError, KeyboardInterrupt):
            img_path = None
        print("-" * 55)

    if img_path and os.path.exists(img_path):
        interactive_multi_roi_session(img_path, thresh_sirip=15.0, thresh_slat=18.0)
    else:
        print("❌ Tidak ada file gambar yang valid dipilih. Program selesai.")
