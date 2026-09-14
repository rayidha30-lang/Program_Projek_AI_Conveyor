import cv2
import numpy as np
import os
import sys

# Variabel Global untuk Mouse Callback
clicked_points = []
current_mouse_pos = (0, 0)
is_drawing_done = False

def mouse_callback(event, x, y, flags, param):
    global clicked_points, current_mouse_pos
    current_mouse_pos = (x, y)
    
    # Klik Kiri: Tambah Titik Satu per Satu
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        print(f"  [+] Titik #{len(clicked_points)} ditambahkan: ({x}, {y})")
        
    # Klik Kanan: Hapus/Undo Titik Terakhir
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(clicked_points) > 0:
            removed = clicked_points.pop()
            print(f"  [-] Titik #{len(clicked_points)+1} dihapus: {removed}")


def analyze_slat_polygon(frame, polygon_pts, pixel_threshold=15.0):
    """
    Analisis kelengkungan plat dengan 5 tahap Preprocessing Canggih (V2):
    1. CLAHE (Penyetaraan kontras permukaan logam)
    2. Bilateral Filter (Menghilangkan noda oli/karat tanpa mengaburkan tepi)
    3. Sobel-Y Gradient (Hanya mendeteksi garis tepi horizontal)
    4. Morphological Closing (Menyambungkan garis tepi yang terputus)
    5. 1D Signal Smoothing (Menghaluskan kontur & membuang noise sudut)
    """
    pts = np.array(polygon_pts, dtype=np.int32)
    h, w = frame.shape[:2]

    # 1. Buat Masking Poligon
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    # 2. Bounding Box dari Poligon
    rx, ry, rw, rh = cv2.boundingRect(pts)
    if rw < 30 or rh < 10:
        return frame, "Area terlalu kecil"

    crop = frame[ry:ry+rh, rx:rx+rw]
    crop_mask = mask[ry:ry+rh, rx:rx+rw]

    # --- PIPELINE PREPROCESSING CANGGIH ---
    # A. Grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # B. Filter 1: CLAHE (Ratakan Kontras & Cahaya Logam)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # C. Filter 2: Bilateral Filter (Hilangkan Karat/Oli tanpa mengaburkan Tepi Plat)
    denoised = cv2.bilateralFilter(enhanced, d=7, sigmaColor=50, sigmaSpace=50)

    # D. Filter 3: Sobel-Y Gradient (Hanya ambil tepi horizontal, abaikan goresan vertikal)
    sobel_y = cv2.Sobel(denoised, cv2.CV_64F, 0, 1, ksize=3)
    sobel_y = np.uint8(np.absolute(sobel_y))

    # E. Filter 4: Thresholding & Morphological Close (Sambungkan Garis Putus-putus)
    _, thresh = cv2.threshold(sobel_y, 35, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 3))
    clean_edges = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Hanya ambil tepi yang berada di dalam area poligon
    clean_edges = cv2.bitwise_and(clean_edges, clean_edges, mask=crop_mask)

    # 4. Ekstraksi koordinat piksel tepi di sepanjang sumbu X
    raw_edge_x = []
    raw_edge_y = []

    for x_col in range(0, rw, 2):
        col_pixels = clean_edges[:, x_col]
        edge_indices = np.where(col_pixels > 0)[0]
        if len(edge_indices) > 0:
            y_edge = edge_indices[0]  # Ambil tepi teratas
            raw_edge_x.append(rx + x_col)
            raw_edge_y.append(ry + y_edge)

    raw_edge_x = np.array(raw_edge_x, dtype=np.float32)
    raw_edge_y = np.array(raw_edge_y, dtype=np.float32)

    result_frame = frame.copy()

    if len(raw_edge_x) > 30:
        # F. Filter 5: Corner Trimming & 1D Smoothing Filter
        # Buang 3% noise di ujung paling kiri dan kanan yang sering mengenai rangka mesin
        trim_len = max(2, int(0.03 * len(raw_edge_x)))
        edge_x = raw_edge_x[trim_len : -trim_len]
        edge_y_raw = raw_edge_y[trim_len : -trim_len]

        # Moving Average Smoothing (Menghilangkan gerigi halus)
        smooth_window = 7
        kernel_smooth = np.ones(smooth_window) / smooth_window
        edge_y = np.convolve(edge_y_raw, kernel_smooth, mode='same')
        # Rapikan bagian batas ujung smoothing
        edge_y[:smooth_window] = edge_y_raw[:smooth_window]
        edge_y[-smooth_window:] = edge_y_raw[-smooth_window:]

        # 5. Garis Acuan Ideal (Baseline)
        x_left, y_left = edge_x[0], edge_y[0]
        x_right, y_right = edge_x[-1], edge_y[-1]

        y_ideal_line = y_left + ((y_right - y_left) / (x_right - x_left)) * (edge_x - x_left)
        deviations = edge_y - y_ideal_line

        # Area tengah plat (30% - 70%) untuk mengukur lendutan utama
        mid_mask = (edge_x >= (rx + 0.30 * rw)) & (edge_x <= (rx + 0.70 * rw))
        avg_dev = np.mean(deviations[mid_mask]) if np.sum(mid_mask) > 0 else np.mean(deviations)
        abs_dev = abs(avg_dev)

        # 6. Keputusan Status
        is_ng = abs_dev > pixel_threshold
        color = (0, 0, 255) if is_ng else (0, 255, 0)
        direction = "BENGKOK NAIK" if avg_dev < 0 else "BENGKOK TURUN"
        status_text = f"NG ({direction} {abs_dev:.1f}px)" if is_ng else f"OK (RATA - {abs_dev:.1f}px)"

        # --- VISUALISASI ---
        # Poligon ROI Pilihan User (Kuning Tipis)
        cv2.polylines(result_frame, [pts], isClosed=True, color=(0, 255, 255), thickness=1)
        
        # Garis Acuan Ideal (Kuning Tebal)
        cv2.line(result_frame, (int(x_left), int(y_left)), (int(x_right), int(y_right)), (0, 255, 255), 3)

        # Garis Profil Fisik Plat yang Sudah Halus (Cyan Mulus)
        for i in range(len(edge_x) - 1):
            cv2.line(result_frame, (int(edge_x[i]), int(edge_y[i])), 
                     (int(edge_x[i+1]), int(edge_y[i+1])), (255, 255, 0), 2)

        # Garis Lendutan Tengah (Ungu Tebal)
        mid_x = (x_left + x_right) / 2.0
        y_ideal_mid = y_left + ((y_right - y_left) / (x_right - x_left)) * (mid_x - x_left)
        y_actual_mid = y_ideal_mid + avg_dev
        cv2.line(result_frame, (int(mid_x), int(y_ideal_mid)), 
                 (int(mid_x), int(y_actual_mid)), (255, 0, 255), 4)

        # Label Status Box Kontras
        label_y = max(40, ry - 15)
        cv2.rectangle(result_frame, (rx, label_y - 28), (rx + 450, label_y + 6), (0, 0, 0), -1)
        cv2.putText(result_frame, f"STATUS: {status_text}", (rx + 8, label_y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        print("\n" + "="*50)
        print(f"📊 HASIL ANALISIS V2 (DENGAN PREPROCESSING):")
        print(f"   -> Lendutan Rata-rata (ΔY) : {abs_dev:.2f} px")
        print(f"   -> Keputusan Sistem        : {status_text}")
        print("="*50)

    else:
        status_text = "TEPI TIDAK TERDETEKSI"
        cv2.polylines(result_frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
        cv2.putText(result_frame, status_text, (rx + 10, ry + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        print("\n⚠️ [WARNING] Garis tepi plat tidak terdeteksi di dalam ROI.")

    return result_frame, status_text


def interactive_point_by_point_session(image_path, pixel_threshold=15.0):
    global clicked_points, current_mouse_pos

    while True:
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"[ERROR] Gambar tidak dapat dibuka: {image_path}")
            break

        h, w = frame.shape[:2]
        window_name = "INSPEKSI DEFORMASI SLAT V2 (Dengan Preprocessing)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(1280, w), min(720, h))
        cv2.setMouseCallback(window_name, mouse_callback)

        clicked_points = []
        is_analyzed = False
        result_frame = None

        print("\n" + "="*65)
        print("👉 PANDUAN PEMILIHAN ROI SATU-SATU (POLYGON) V2:")
        print("   1. [KLIK KIRI]  : Klik titik sudut plat satu per satu (4 sudut).")
        print("   2. [KLIK KANAN] : Undo / Hapus titik terakhir jika salah klik.")
        print("   3. [ENTER] / [SPASI] : SELESAI memilih & langsung proses hasil.")
        print("   4. [R]          : Reset / Hapus semua titik & mulai ulang.")
        print("   5. [N]          : Ganti Foto Baru.")
        print("   6. [Q] / [ESC]  : Keluar Program.")
        print("="*65)

        # Loop pemilihan titik ROI
        while True:
            display = frame.copy()

            if not is_analyzed:
                # Gambar titik-titik yang sudah diklik user
                for i, pt in enumerate(clicked_points):
                    cv2.circle(display, pt, 5, (0, 255, 255), -1)
                    cv2.putText(display, f"#{i+1}", (pt[0] + 8, pt[1] - 8), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                # Hubungkan garis antar titik
                if len(clicked_points) > 1:
                    for i in range(len(clicked_points) - 1):
                        cv2.line(display, clicked_points[i], clicked_points[i+1], (0, 255, 255), 2)
                    
                    # Garis dinamis dari titik terakhir ke posisi kursor mouse
                    cv2.line(display, clicked_points[-1], current_mouse_pos, (180, 180, 180), 1)

                # Teks petunjuk di pojok atas
                cv2.rectangle(display, (0, 0), (w, 35), (20, 20, 20), -1)
                cv2.putText(display, f"Titik: {len(clicked_points)} | [Klik Kiri]: Tambah | [Klik Kanan]: Undo | [ENTER]: Selesai & Ukur", 
                            (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                cv2.imshow(window_name, display)
                key = cv2.waitKey(20) & 0xFF
            else:
                cv2.imshow(window_name, result_frame)
                key = cv2.waitKey(0) & 0xFF

            # --- RESPON TOMBOL KEYBOARD ---
            # Tekan ENTER / SPASI saat titik >= 3 -> Proses Hasil
            if (key in [13, 32]) and (not is_analyzed):
                if len(clicked_points) >= 3:
                    result_frame, status = analyze_slat_polygon(frame, clicked_points, pixel_threshold)
                    
                    # Footer petunjuk
                    cv2.rectangle(result_frame, (0, h - 35), (w, h), (30, 30, 30), -1)
                    cv2.putText(result_frame, "[R / SPASI]: Reset ROI Baru | [N]: Ganti Foto | [Q / ESC]: Keluar", 
                                (20, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                    
                    is_analyzed = True
                    
                    # Simpan hasil
                    base_name, ext = os.path.splitext(image_path)
                    cv2.imwrite(f"{base_name}_hasil_v2{ext}", result_frame)
                else:
                    print("⚠️ Klik minimal 3 atau 4 titik sudut untuk membentuk area ROI plat!")

            # Tombol 'r' / SPASI setelah dianalisis -> Reset ROI
            elif (key in [ord('r'), ord('R')]) or (is_analyzed and key == 32):
                print("\n🔄 [RESET] Mengulang pemilihan titik pada foto yang sama...")
                clicked_points = []
                is_analyzed = False
                continue

            # Tombol 'q' / ESC -> Keluar
            elif key in [ord('q'), ord('Q'), 27]:
                print("\n[INFO] Menutup program.")
                cv2.destroyAllWindows()
                return

            # Tombol 'n' -> Ganti Foto Baru
            elif key in [ord('n'), ord('N')]:
                cv2.destroyAllWindows()
                print("\n" + "-"*50)
                new_path = input("👉 Masukkan path foto BARU: ").strip('\'"')
                print("-"*50)
                if new_path:
                    image_path = os.path.normpath(new_path)
                break


if __name__ == "__main__":
    print("\n" + "#"*65)
    print("  PROGRAM INSPEKSI DEFORMASI SLAT RIB (METODE GARIS ACUAN - V2)")
    print("  Dilengkapi 5 Preprocessing: CLAHE, Bilateral, Sobel-Y & Smoothing")
    print("#"*65)

    if len(sys.argv) > 1:
        img_path = sys.argv[1].strip('\'"')
    else:
        print("\n" + "-"*50)
        img_path = input("👉 Masukkan / Drag & Drop path foto gambar di sini: ").strip('\'"')
        print("-"*50)

    if img_path:
        img_path = os.path.normpath(img_path)
        interactive_point_by_point_session(img_path, pixel_threshold=15.0)
    else:
        print("Path gambar tidak boleh kosong.")
