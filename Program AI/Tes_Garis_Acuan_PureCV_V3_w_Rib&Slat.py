import cv2
import numpy as np
import os
import sys

current_step = 1            # 1: Pilih Sirip (Rib), 2: Pilih Plat Dasar (Slat), 3: Selesai Analisis
points_sirip = []           # Koordinat poligon ROI 1 (Sirip)
points_slat = []            # Koordinat poligon ROI 2 (Plat Dasar)
current_mouse_pos = (0, 0)  # Posisi live kursor mouse

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
    Ekstraksi kontur fisik tepi dengan 5 tahap Preprocessing + Smart Corner Trimming
    """
    pts = np.array(polygon_pts, dtype=np.int32)
    h, w = frame.shape[:2]

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    rx, ry, rw, rh = cv2.boundingRect(pts)
    if rw < 30 or rh < 10:
        return None, None, None, rx, ry, rw, rh, pts, None

    crop = frame[ry:ry+rh, rx:rx+rw]
    crop_mask = mask[ry:ry+rh, rx:rx+rw]

    # Preprocessing Pipeline
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.bilateralFilter(enhanced, d=7, sigmaColor=50, sigmaSpace=50)
    sobel_y = cv2.Sobel(denoised, cv2.CV_64F, 0, 1, ksize=3)
    sobel_y = np.uint8(np.absolute(sobel_y))

    _, thresh = cv2.threshold(sobel_y, 35, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 3))
    clean_edges = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    clean_edges_masked = cv2.bitwise_and(clean_edges, clean_edges, mask=crop_mask)

    # Grid Visualisasi Filter
    tw, th = 380, 140
    def prep(img, t):
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()
        r = cv2.resize(bgr, (tw, th))
        cv2.rectangle(r, (0, 0), (tw, 24), (15, 15, 15), -1)
        cv2.putText(r, t, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)
        return r
    
    row1 = np.hstack([prep(crop, f"1. Crop Asli ({name})"), prep(gray, "2. Grayscale"), prep(enhanced, "3. CLAHE")])
    row2 = np.hstack([prep(denoised, "4. Bilateral Denoised"), prep(sobel_y, "5. Sobel-Y Gradient"), prep(clean_edges_masked, "6. Morph Close (13x3)")])
    filter_grid = np.vstack([row1, row2])

    raw_x = []
    raw_y = []
    for x_col in range(0, rw, 2):
        col_pixels = clean_edges_masked[:, x_col]
        edge_indices = np.where(col_pixels > 0)[0]
        if len(edge_indices) > 0:
            y_edge = edge_indices[0]
            raw_x.append(rx + x_col)
            raw_y.append(ry + y_edge)

    raw_x = np.array(raw_x, dtype=np.float32)
    raw_y = np.array(raw_y, dtype=np.float32)

    if len(raw_x) < 25:
        return None, None, None, rx, ry, rw, rh, pts, filter_grid

    # Smart Corner Trimming: Potong 6% di ujung kiri dan kanan
    # Ini secara otomatis membuang chamfer sudut luar sirip & bracket rantai samping
    trim = max(3, int(0.06 * len(raw_x)))
    edge_x = raw_x[trim:-trim]
    edge_y_raw = raw_y[trim:-trim]

    k_med = 11
    pad = k_med // 2
    padded_y = np.pad(edge_y_raw, (pad, pad), mode='edge')
    med_y = np.array([np.median(padded_y[i : i + k_med]) for i in range(len(edge_y_raw))])

    smooth_w = 7
    kernel_s = np.ones(smooth_w) / smooth_w
    smooth_y = np.convolve(med_y, kernel_s, mode='same')
    smooth_y[:smooth_w] = med_y[:smooth_w]
    smooth_y[-smooth_w:] = med_y[-smooth_w:]

    return edge_x, smooth_y, edge_y_raw, rx, ry, rw, rh, pts, filter_grid


def analyze_single_component_pv(contour_data, name="KOMPONEN", threshold_pv=15.0):
    """
    Analisis Flatness Mandiri (Peak-to-Valley Metrology V3.5):
    - Garis acuan ideal dihitung mandiri dari ujung ke ujung bentang efektif.
    - Menghitung deviasi Peak-to-Valley (P-V Flatness) dan Maximum Absolute Deviation.
    """
    edge_x, smooth_y, _, rx, ry, rw, rh, pts, grid = contour_data
    if edge_x is None or len(edge_x) < 20:
        return {'success': False, 'name': name, 'msg': 'Garis tepi tidak terdeteksi'}

    # Garis Acuan Ideal Mandiri (Bentang Efektif Komponen)
    x_left, y_left = edge_x[0], smooth_y[0]
    x_right, y_right = edge_x[-1], smooth_y[-1]

    y_ideal = y_left + ((y_right - y_left) / (x_right - x_left)) * (edge_x - x_left)
    deviations = smooth_y - y_ideal

    # Metrologi Peak-to-Valley (P-V Flatness)
    max_pos = np.max(deviations)   # Puncak lendutan ke bawah (+)
    min_neg = np.min(deviations)   # Puncak lendutan ke atas (-)
    pv_flatness = max_pos - min_neg # Rentang Puncak-ke-Lembah
    max_abs = np.max(np.abs(deviations))

    # Kriteria Keputusan Realistis
    is_ng = pv_flatness > threshold_pv

    if is_ng:
        if abs(max_pos) > abs(min_neg) * 1.3:
            dir_str = f"BENGKOK TURUN (+{max_pos:.1f}px)"
        elif abs(min_neg) > abs(max_pos) * 1.3:
            dir_str = f"BENGKOK NAIK ({min_neg:.1f}px)"
        else:
            dir_str = f"BENGKOK BERGELOMBANG (P-V {pv_flatness:.1f}px)"
        status_str = f"NG ({dir_str})"
    else:
        status_str = f"OK (RATA {pv_flatness:.1f}px)"

    # Titik lokasi deviasi terbesar untuk visualisasi garis vektor
    idx_max = np.argmax(np.abs(deviations))
    peak_x = edge_x[idx_max]
    peak_y_ideal = y_ideal[idx_max]
    peak_y_actual = smooth_y[idx_max]

    return {
        'success': True,
        'name': name,
        'pts': pts,
        'edge_x': edge_x, 'edge_y': smooth_y,
        'x_left': x_left, 'y_left': y_left,
        'x_right': x_right, 'y_right': y_right,
        'peak_x': peak_x, 'peak_y_ideal': peak_y_ideal, 'peak_y_actual': peak_y_actual,
        'pv_flatness': pv_flatness, 'max_abs': max_abs,
        'is_ng': is_ng, 'status_str': status_str, 'grid': grid
    }


def draw_component_visuals(canvas, comp_data, color_theme):
    """Menggambar garis acuan, kontur deteksi, dan vektor lendutan terbesar"""
    cv2.polylines(canvas, [comp_data['pts']], isClosed=True, color=color_theme, thickness=1)
    cv2.line(canvas, (int(comp_data['x_left']), int(comp_data['y_left'])), 
             (int(comp_data['x_right']), int(comp_data['y_right'])), color_theme, 3)
    for i in range(len(comp_data['edge_x']) - 1):
        cv2.line(canvas, (int(comp_data['edge_x'][i]), int(comp_data['edge_y'][i])), 
                 (int(comp_data['edge_x'][i+1]), int(comp_data['edge_y'][i+1])), (255, 255, 0), 2)
    # Garis vektor lendutan di titik paling kritis
    cv2.line(canvas, (int(comp_data['peak_x']), int(comp_data['peak_y_ideal'])), 
             (int(comp_data['peak_x']), int(comp_data['peak_y_actual'])), (255, 0, 255), 4)
    cv2.circle(canvas, (int(comp_data['peak_x']), int(comp_data['peak_y_actual'])), 4, (255, 0, 255), -1)


def interactive_multi_roi_session(image_path, thresh_sirip=15.0, thresh_slat=18.0):
    global current_step, points_sirip, points_slat, current_mouse_pos

    while True:
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"[ERROR] Gambar tidak dapat dibuka: {image_path}")
            break

        h, w = frame.shape[:2]
        window_name = "INSPEKSI MULTI-ROI V3.5: ROBUST CORNER TRIMMING"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(1360, w), min(768, h))
        cv2.setMouseCallback(window_name, mouse_callback)

        current_step = 1
        points_sirip = []
        points_slat = []
        result_frame = None
        last_sirip = None
        last_slat = None

        print("\n" + "="*70)
        print("📋 PANDUAN PENGUKURAN MULTI-ROI V3.5 (ROBUST CORNER TRIMMING):")
        print("   TAHAP 1: Klik 4 sudut pada SIRIP (RIB)      -> Tekan [ENTER]")
        print("   TAHAP 2: Klik 4 sudut pada PLAT DASAR (SLAT)-> Tekan [ENTER]")
        print("   TIPS    : Tandai bidang plat kerja (hindari dinding bracket samping)")
        print("   KONTROL : [Klik Kiri] Tambah | [Klik Kanan] Undo | [P] Lihat Filter")
        print("             [R / SPASI] Reset | [N] Ganti Foto | [Q] Keluar")
        print("="*70)

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

                cv2.rectangle(display, (0, 0), (w, 40), (20, 20, 20), -1)
                cv2.putText(display, f"[TAHAP 1/2: PILIH SIRIP/RIB] Titik: {len(points_sirip)} | [Klik Kiri]: Tambah | [ENTER]: Lanjut ke Plat Dasar", 
                            (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 180, 255), 2)

                cv2.imshow(window_name, display)
                key = cv2.waitKey(20) & 0xFF

            elif current_step == 2:
                if len(points_sirip) >= 3:
                    cv2.polylines(display, [np.array(points_sirip, dtype=np.int32)], isClosed=True, color=(0, 140, 255), thickness=2)
                    cv2.putText(display, "ROI 1: SIRIP (TERKUNCI)", (points_sirip[0][0], max(30, points_sirip[0][1] - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)

                for i, pt in enumerate(points_slat):
                    cv2.circle(display, pt, 5, (0, 255, 255), -1)
                    cv2.putText(display, f"P{i+1}", (pt[0] + 8, pt[1] - 8), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                if len(points_slat) > 1:
                    for i in range(len(points_slat) - 1):
                        cv2.line(display, points_slat[i], points_slat[i+1], (0, 255, 255), 2)
                    cv2.line(display, points_slat[-1], current_mouse_pos, (180, 180, 180), 1)

                cv2.rectangle(display, (0, 0), (w, 40), (20, 20, 20), -1)
                cv2.putText(display, f"[TAHAP 2/2: PILIH PLAT DASAR/SLAT] Titik: {len(points_slat)} | [Klik Kiri]: Tambah | [ENTER]: Selesai & Ukur Semua", 
                            (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)

                cv2.imshow(window_name, display)
                key = cv2.waitKey(20) & 0xFF

            else:
                cv2.imshow(window_name, result_frame)
                key = cv2.waitKey(0) & 0xFF

            # KEYBOARD EVENTS
            if key in [13, 32]:
                if current_step == 1:
                    if len(points_sirip) >= 3:
                        current_step = 2
                        print("  [OK] ROI Sirip tersimpan! Sekarang silakan klik sudut-sudut PLAT DASAR.")
                    else:
                        print("⚠️ Klik minimal 3 atau 4 titik untuk menandai area Sirip!")

                elif current_step == 2:
                    if len(points_slat) >= 3:
                        print("\n⚙️ Memproses analisis metrologi V3.5 (Smart Corner Trimming)...")
                        
                        contour_sirip = extract_clean_contour(frame, points_sirip, name="SIRIP")
                        contour_slat = extract_clean_contour(frame, points_slat, name="PLAT DASAR")

                        last_sirip = analyze_single_component_pv(contour_sirip, name="SIRIP", threshold_pv=thresh_sirip)
                        last_slat = analyze_single_component_pv(contour_slat, name="PLAT DASAR", threshold_pv=thresh_slat)

                        if not last_sirip.get('success', False) or not last_slat.get('success', False):
                            msg = last_sirip.get('msg') if not last_sirip.get('success') else last_slat.get('msg')
                            print(f"⚠️ [ERROR] {msg}")
                            continue

                        result_frame = frame.copy()
                        draw_component_visuals(result_frame, last_sirip, color_theme=(0, 140, 255))
                        draw_component_visuals(result_frame, last_slat, color_theme=(0, 255, 255))

                        # Status Card HUD Ganda
                        color_sirip = (0, 0, 255) if last_sirip['is_ng'] else (0, 255, 0)
                        txt_sirip = f"SIRIP     : {last_sirip['status_str']}"
                        cv2.rectangle(result_frame, (15, 15), (660, 52), (15, 15, 15), -1)
                        cv2.rectangle(result_frame, (15, 15), (660, 52), color_sirip, 2)
                        cv2.putText(result_frame, txt_sirip, (25, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.68, color_sirip, 2)

                        color_slat = (0, 0, 255) if last_slat['is_ng'] else (0, 255, 0)
                        txt_slat = f"PLAT DASAR: {last_slat['status_str']}"
                        cv2.rectangle(result_frame, (15, 58), (660, 95), (15, 15, 15), -1)
                        cv2.rectangle(result_frame, (15, 58), (660, 95), color_slat, 2)
                        cv2.putText(result_frame, txt_slat, (25, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.68, color_slat, 2)

                        cv2.rectangle(result_frame, (0, h - 35), (w, h), (25, 25, 25), -1)
                        cv2.putText(result_frame, "[P]: Lihat Filter | [R / SPASI]: Ulangi | [N]: Ganti Foto | [Q]: Keluar", 
                                    (20, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)

                        print("\n" + "="*68)
                        print("📊 LAPORAN HASIL METROLOGI FLATNESS V3.5:")
                        print(f"   [1] SIRIP (RIB)     : {txt_sirip}")
                        print(f"       -> P-V Flatness : {last_sirip['pv_flatness']:.1f} px | Max Abs Dev: {last_sirip['max_abs']:.1f} px (Batas: {thresh_sirip} px)")
                        print(f"   [2] PLAT DASAR      : {txt_slat}")
                        print(f"       -> P-V Flatness : {last_slat['pv_flatness']:.1f} px | Max Abs Dev: {last_slat['max_abs']:.1f} px (Batas: {thresh_slat} px)")
                        print("="*68)

                        current_step = 3
                        
                        base_name, ext = os.path.splitext(image_path)
                        cv2.imwrite(f"{base_name}_hasil_v3_dual{ext}", result_frame)
                    else:
                        print("⚠️ Klik minimal 3 atau 4 titik untuk menandai area Plat Dasar!")

            # Tombol 'P'
            elif key in [ord('p'), ord('P')] and current_step == 3 and last_sirip is not None and last_slat is not None:
                if last_sirip.get('grid') is not None:
                    cv2.imshow("TAHAPAN FILTER: SIRIP", last_sirip['grid'])
                if last_slat.get('grid') is not None:
                    cv2.imshow("TAHAPAN FILTER: PLAT DASAR", last_slat['grid'])

            # Tombol 'R' / Reset
            elif (key in [ord('r'), ord('R')]) or (current_step == 3 and key == 32):
                print("\n🔄 [RESET] Mengulang pemilihan titik Sirip & Plat Dasar...")
                current_step = 1
                points_sirip = []
                points_slat = []
                continue

            # Tombol 'Q' / ESC -> Keluar
            elif key in [ord('q'), ord('Q'), 27]:
                print("\n[INFO] Menutup program inspeksi.")
                cv2.destroyAllWindows()
                return

            # Tombol 'N' -> Ganti Foto Baru
            elif key in [ord('n'), ord('N')]:
                cv2.destroyAllWindows()
                print("\n" + "-"*50)
                new_path = input("👉 Masukkan path foto BARU: ").strip('\'"')
                print("-"*50)
                if new_path:
                    image_path = os.path.normpath(new_path)
                break


if __name__ == "__main__":
    print("\n" + "#"*70)
    print("  PROGRAM INSPEKSI DUAL SLAT RIB (METODE GARIS ACUAN - V3.5)")
    print("  Metrologi Mandiri (Robust Corner Trimming & Calibrated Thresholds)")
    print("#"*70)

    if len(sys.argv) > 1:
        img_path = sys.argv[1].strip('\'"')
    else:
        print("\n" + "-"*50)
        img_path = input("👉 Masukkan / Drag & Drop path foto gambar di sini: ").strip('\'"')
        print("-"*50)

    if img_path:
        img_path = os.path.normpath(img_path)
        interactive_multi_roi_session(img_path, thresh_sirip=15.0, thresh_slat=18.0)
    else:
        print("Path gambar tidak boleh kosong.")
