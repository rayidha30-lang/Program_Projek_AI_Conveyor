import cv2
import numpy as np
import os
import sys
import glob

# Try importing Tkinter for native Windows File Dialog
try:
    import tkinter as tk
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
    """Membersihkan string path dari quotes, prefix PowerShell, dan spasi."""
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


def extract_clean_contour(frame, polygon_pts, name="KOMPONEN"):
    """
    Ekstraksi kontur fisik tepi:
    - Gradien Isotropik (Sobel-X + Sobel-Y) agar akurat pada sudut miring/diagonal maupun horizontal
    - Pemindaian adaptif (scan kolom jika horizontal, scan baris jika curam/vertikal)
    - 5 tahap Preprocessing + Smart Corner Trimming (6%)
    """
    pts = np.array(polygon_pts, dtype=np.int32)
    h, w = frame.shape[:2]

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    rx, ry, rw, rh = cv2.boundingRect(pts)
    if rw < 20 or rh < 10:
        return None, None, None, rx, ry, rw, rh, pts, None, True

    is_horizontal = rw >= rh
    crop = frame[ry:ry+rh, rx:rx+rw]
    crop_mask = mask[ry:ry+rh, rx:rx+rw]

    # Preprocessing Pipeline
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.bilateralFilter(enhanced, d=7, sigmaColor=50, sigmaSpace=50)

    # Gradien Isotropik: Sobel-X dan Sobel-Y digabungkan
    sobel_x = cv2.Sobel(denoised, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(denoised, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(sobel_x, sobel_y)
    grad_mag = np.uint8(np.clip(grad_mag, 0, 255))

    _, thresh = cv2.threshold(grad_mag, 35, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5))
    clean_edges = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    clean_edges_masked = cv2.bitwise_and(clean_edges, clean_edges, mask=crop_mask)

    # Context Bounding Box: Sertakan bentang conveyor dan margin vertikal lega di sekitar ROI
    # sehingga benda kerja terlihat BESAR dan JELAS dengan konteks lingkungan sekitarnya
    pad_y = max(90, int(rh * 1.6))
    cy1 = max(0, ry - pad_y)
    cy2 = min(h, ry + rh + pad_y)
    if is_horizontal:
        cx1 = 0
        cx2 = w
    else:
        pad_x = max(90, int(rw * 1.6))
        cx1 = max(0, rx - pad_x)
        cx2 = min(w, rx + rw + pad_x)
        cy1 = 0
        cy2 = h

    context_crop = frame[cy1:cy2, cx1:cx2]
    ch_ctx, cw_ctx = context_crop.shape[:2]

    # Grid Visualisasi Filter Diperbesar (Konteks Lega di Sekitar ROI)
    def make_uncropped_panel(stage_crop, title):
        stage_bgr = cv2.cvtColor(stage_crop, cv2.COLOR_GRAY2BGR) if len(stage_crop.shape) == 2 else stage_crop.copy()
        # Kanvas foto utuh yang sedikit diredupkan (45% kecerahan)
        canvas = (frame.astype(np.float32) * 0.45).astype(np.uint8)
        crop_canvas = canvas[ry:ry+rh, rx:rx+rw]
        crop_canvas[crop_mask > 0] = stage_bgr[crop_mask > 0]
        canvas[ry:ry+rh, rx:rx+rw] = crop_canvas
        # Garis batas poligon ROI tebal
        cv2.polylines(canvas, [pts], isClosed=True, color=(0, 255, 255), thickness=3)
        
        # Ambil area konteks di sekitar ROI agar tampilan plat menjadi BESAR
        sub = canvas[cy1:cy2, cx1:cx2].copy()
        
        # Panel ukuran besar (500 px lebar per panel)
        gw = 500
        gh = int(gw * (ch_ctx / cw_ctx))
        gh = max(240, min(360, gh))
        r = cv2.resize(sub, (gw, gh))
        
        # Banner judul tebal dan jelas (tidak bertumpuk)
        cv2.rectangle(r, (0, 0), (gw, 30), (15, 15, 15), -1)
        cv2.putText(r, title, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)
        return r
    
    p1 = make_uncropped_panel(crop, f"1. Foto Asli + ROI ({name})")
    p2 = make_uncropped_panel(gray, "2. Grayscale")
    p3 = make_uncropped_panel(enhanced, "3. CLAHE (Kontras Diperkuat)")
    p4 = make_uncropped_panel(denoised, "4. Bilateral (Noise Dihaluskan)")
    p5 = make_uncropped_panel(grad_mag, "5. Isotropic Gradient (Sobel)")
    p6 = make_uncropped_panel(clean_edges_masked, "6. Morph Close Edges (Biner)")
    
    row1 = np.hstack([p1, p2, p3])
    row2 = np.hstack([p4, p5, p6])
    filter_grid = np.vstack([row1, row2])
    raw_x = []
    raw_y = []

    if is_horizontal:
        for x_col in range(0, rw, 2):
            col_pixels = clean_edges_masked[:, x_col]
            edge_indices = np.where(col_pixels > 0)[0]
            if len(edge_indices) > 0:
                y_edge = edge_indices[0]
                raw_x.append(rx + x_col)
                raw_y.append(ry + y_edge)
    else:
        for y_row in range(0, rh, 2):
            row_pixels = clean_edges_masked[y_row, :]
            edge_indices = np.where(row_pixels > 0)[0]
            if len(edge_indices) > 0:
                x_edge = edge_indices[0]
                raw_x.append(rx + x_edge)
                raw_y.append(ry + y_row)

    raw_x = np.array(raw_x, dtype=np.float32)
    raw_y = np.array(raw_y, dtype=np.float32)

    if len(raw_x) < 20:
        return None, None, None, rx, ry, rw, rh, pts, filter_grid, is_horizontal

    # Smart Corner Trimming: Potong 6% di kedua ujung bentang
    trim = max(2, int(0.06 * len(raw_x)))
    edge_x = raw_x[trim:-trim]
    edge_y_raw = raw_y[trim:-trim]

    # Median + Convolve Smoothing
    k_med = 9
    pad = k_med // 2
    if is_horizontal:
        padded_target = np.pad(edge_y_raw, (pad, pad), mode='edge')
        med = np.array([np.median(padded_target[i : i + k_med]) for i in range(len(edge_y_raw))])
        smooth_w = 7
        kernel_s = np.ones(smooth_w) / smooth_w
        smooth_val = np.convolve(med, kernel_s, mode='same')
        smooth_val[:smooth_w] = med[:smooth_w]
        smooth_val[-smooth_w:] = med[-smooth_w:]
        clean_x = edge_x
        clean_y = smooth_val
    else:
        padded_target = np.pad(edge_x, (pad, pad), mode='edge')
        med = np.array([np.median(padded_target[i : i + k_med]) for i in range(len(edge_x))])
        smooth_w = 7
        kernel_s = np.ones(smooth_w) / smooth_w
        smooth_val = np.convolve(med, kernel_s, mode='same')
        smooth_val[:smooth_w] = med[:smooth_w]
        smooth_val[-smooth_w:] = med[-smooth_w:]
        clean_x = smooth_val
        clean_y = edge_y_raw

    return clean_x, clean_y, edge_y_raw, rx, ry, rw, rh, pts, filter_grid, is_horizontal


def analyze_single_component_pv(contour_data, name="KOMPONEN", threshold_pv=15.0):
    """
    Analisis Metrologi Flatness (P-V) Invarian Sudut:
    - Garis acuan ideal dihitung dari ujung awal ke ujung akhir bentang komponen.
    - Menghitung JARAK TEGAK LURUS (Perpendicular Euclidean Distance) ke garis acuan.
      Jarak ini invarian terhadap kemiringan kamera / foto potret / diagonal.
    - Peak-to-Valley (P-V Flatness) = max(dev) - min(dev)
    """
    clean_x, clean_y, _, rx, ry, rw, rh, pts, grid, is_horizontal = contour_data
    if clean_x is None or len(clean_x) < 15:
        return {'success': False, 'name': name, 'msg': 'Garis tepi tidak cukup terdeteksi'}

    # Titik ujung garis acuan ideal mandiri
    x1, y1 = float(clean_x[0]), float(clean_y[0])
    x2, y2 = float(clean_x[-1]), float(clean_y[-1])
    span_len = np.hypot(x2 - x1, y2 - y1)
    if span_len < 1e-3:
        return {'success': False, 'name': name, 'msg': 'Bentang komponen terlalu pendek'}

    # Rumus Jarak Perpendicular Bertanda (Signed Perpendicular Distance):
    # d_perp = ((y2 - y1)*x0 - (x2 - x1)*y0 + x2*y1 - y2*x1) / span_len
    perp_dist = ((y2 - y1) * clean_x - (x2 - x1) * clean_y + x2 * y1 - y2 * x1) / span_len

    # Metrologi Peak-to-Valley (P-V Flatness)
    max_pos = float(np.max(perp_dist))
    min_neg = float(np.min(perp_dist))
    pv_flatness = max_pos - min_neg
    max_abs = float(np.max(np.abs(perp_dist)))

    # Kriteria Keputusan Realistis
    is_ng = pv_flatness > threshold_pv

    if is_ng:
        if abs(max_pos) > abs(min_neg) * 1.3:
            dir_str = f"BENGKOK (+{max_pos:.1f}px)"
        elif abs(min_neg) > abs(max_pos) * 1.3:
            dir_str = f"BENGKOK ({min_neg:.1f}px)"
        else:
            dir_str = f"BERGELOMBANG (P-V {pv_flatness:.1f}px)"
        status_str = f"NG ({dir_str})"
    else:
        status_str = f"OK (RATA {pv_flatness:.1f}px)"

    # Titik deviasi absolut terbesar untuk visualisasi vektor
    idx_max = int(np.argmax(np.abs(perp_dist)))
    peak_x = float(clean_x[idx_max])
    peak_y_actual = float(clean_y[idx_max])

    # Proyeksi titik puncak ke garis ideal untuk pangkal garis vektor
    # Parameter t pada garis P(t) = P1 + t*(P2 - P1)
    dx_line = x2 - x1
    dy_line = y2 - y1
    t = ((peak_x - x1) * dx_line + (peak_y_actual - y1) * dy_line) / (span_len ** 2)
    proj_x = x1 + t * dx_line
    proj_y = y1 + t * dy_line

    return {
        'success': True,
        'name': name,
        'pts': pts,
        'edge_x': clean_x, 'edge_y': clean_y,
        'x_left': x1, 'y_left': y1,
        'x_right': x2, 'y_right': y2,
        'peak_x': peak_x, 'peak_y_actual': peak_y_actual,
        'proj_x': proj_x, 'proj_y': proj_y,
        'pv_flatness': pv_flatness, 'max_abs': max_abs,
        'is_ng': is_ng, 'status_str': status_str, 'grid': grid
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


def interactive_multi_roi_session(initial_image_path, thresh_sirip=15.0, thresh_slat=18.0):
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
        window_name = f"INSPEKSI SLAT & RIB V4.0 - [{file_name}]"
        
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
        print("📋 PANDUAN PENGUKURAN MULTI-ROI V4.0:")
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
                        print("\n⚙️ Memproses analisis metrologi V4.0 (Invarian Sudut & Perpendicular Distance)...")
                        
                        contour_sirip = extract_clean_contour(frame, points_sirip, name="SIRIP")
                        contour_slat = extract_clean_contour(frame, points_slat, name="PLAT DASAR")

                        last_sirip = analyze_single_component_pv(contour_sirip, name="SIRIP", threshold_pv=thresh_sirip)
                        last_slat = analyze_single_component_pv(contour_slat, name="PLAT DASAR", threshold_pv=thresh_slat)

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
                        print("📊 LAPORAN HASIL METROLOGI FLATNESS V4.0:")
                        print(f"   [1] SIRIP (RIB)     : {txt_sirip}")
                        print(f"       -> P-V Flatness : {last_sirip['pv_flatness']:.1f} px | Max Abs Dev: {last_sirip['max_abs']:.1f} px (Batas: {thresh_sirip} px)")
                        print(f"   [2] PLAT DASAR      : {txt_slat}")
                        print(f"       -> P-V Flatness : {last_slat['pv_flatness']:.1f} px | Max Abs Dev: {last_slat['max_abs']:.1f} px (Batas: {thresh_slat} px)")
                        print("="*70)

                        current_step = 3
                        base_name, ext = os.path.splitext(image_path)
                        save_out = f"{base_name}_hasil_v4_dual{ext}"
                        cv2.imwrite(save_out, result_frame)
                        print(f"  💾 Hasil visual disimpan ke: {os.path.basename(save_out)}")
                    else:
                        print("  ⚠️ Klik minimal 3 atau 4 titik untuk menandai area Plat Dasar!")

            # Tombol 'P': Lihat Filter Preprocessing (Jendela Besar & Jelas)
            elif key in [ord('p'), ord('P')] and current_step == 3:
                if last_sirip is not None and last_sirip.get('grid') is not None:
                    win_sirip = "TAHAPAN FILTER: SIRIP (FOTO DIPERBESAR)"
                    cv2.namedWindow(win_sirip, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(win_sirip, 1500, 620)
                    cv2.imshow(win_sirip, last_sirip['grid'])
                if last_slat is not None and last_slat.get('grid') is not None:
                    win_slat = "TAHAPAN FILTER: PLAT DASAR (FOTO DIPERBESAR)"
                    cv2.namedWindow(win_slat, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(win_slat, 1500, 620)
                    cv2.imshow(win_slat, last_slat['grid'])

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
    print("  PROGRAM INSPEKSI DUAL SLAT RIB CONVEYOR (PURE CV - V4.0)")
    print("  Angle-Invariant Metrology & Seamless Multi-Photo Switching")
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
