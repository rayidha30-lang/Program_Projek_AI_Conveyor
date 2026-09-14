import cv2
import numpy as np
import os
import sys
import time
import glob
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO

# ==============================================================================
# KONFIGURASI DEFAULT
# ==============================================================================
DEFAULT_MODEL_PATH = r"C:\Users\Victus\Downloads\Side_LeftnRight_Conveyor_best (28).pt"
DEFAULT_CONF_THRESHOLD = 0.25  # Nilai default 25% (0.25)
DEFAULT_IOU_THRESHOLD = 0.45   # Nilai IoU / NMS default 45% (0.45)
OUTPUT_DIR = "output_predictions"

# Warna unik untuk tiap kelas (BGR format)
CLASS_COLORS = {
    "bolt_ok": (0, 230, 115),     # Hijau neon terang
    "pin_ok": (255, 178, 50),      # Cyan/Biru muda
    "bolt_ng": (0, 0, 255),       # Merah jika ada varian cacat
    "pin_ng": (0, 140, 255),      # Oranye
}
FALLBACK_COLORS = [
    (0, 255, 127), (255, 191, 0), (0, 165, 255), 
    (255, 0, 255), (0, 255, 255), (147, 20, 255)
]

# Variabel State Global untuk Kontrol Interaktif
current_image_idx = 0
image_list = []
model = None
current_raw_img = None
window_name = "YOLO Inference Tester (Image Mode)"
conf_threshold = DEFAULT_CONF_THRESHOLD
iou_threshold = DEFAULT_IOU_THRESHOLD
show_overlay_info = True


def get_color_for_class(class_name, class_id):
    """Mendapatkan warna BGR untuk kelas tertentu."""
    if class_name in CLASS_COLORS:
        return CLASS_COLORS[class_name]
    return FALLBACK_COLORS[class_id % len(FALLBACK_COLORS)]


def select_file_dialog():
    """Membuka dialog explorer untuk memilih 1 file gambar."""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.askopenfilename(
        title="Pilih Gambar untuk Dites dengan YOLO",
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff"),
            ("All Files", "*.*")
        ]
    )
    root.destroy()
    return file_path


def select_folder_dialog():
    """Membuka dialog explorer untuk memilih folder berisi kumpulan gambar."""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    folder_path = filedialog.askdirectory(
        title="Pilih Folder Berisi Kumpulan Gambar"
    )
    root.destroy()
    return folder_path


def load_images_from_path(target_path):
    """Memuat daftar path gambar dari file tunggal atau direktori."""
    supported_exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp', '*.tiff', 
                      '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.WEBP')
    
    if os.path.isfile(target_path):
        return [target_path]
    elif os.path.isdir(target_path):
        files = []
        for ext in supported_exts:
            files.extend(glob.glob(os.path.join(target_path, ext)))
        files.sort()
        return files
    return []


def draw_custom_detections(image, results, conf_val):
    """
    Menggambar bounding box, label elegan, dan titik pusat deteksi
    dengan desain UI yang rapi dan profesional.
    """
    canvas = image.copy()
    counts = {}
    det_details = []

    for r in results:
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            continue

        for i, box in enumerate(boxes):
            conf = float(box.conf[0].cpu().numpy())
            if conf < conf_val:
                continue

            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = r.names.get(cls_id, f"Class_{cls_id}")
            counts[cls_name] = counts.get(cls_name, 0) + 1

            # Koordinat Bounding Box (x1, y1, x2, y2)
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            w = x2 - x1
            h = y2 - y1
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            det_details.append({
                "id": i + 1,
                "class": cls_name,
                "conf": conf,
                "box": (x1, y1, x2, y2),
                "center": (cx, cy),
                "dim": (w, h)
            })

            color = get_color_for_class(cls_name, cls_id)

            # 1. Bounding Box dengan sudut halus / rounded accent
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
            
            # Corner accents
            corner_len = min(15, w // 4, h // 4)
            if corner_len > 3:
                cv2.line(canvas, (x1, y1), (x1 + corner_len, y1), color, 4, lineType=cv2.LINE_AA)
                cv2.line(canvas, (x1, y1), (x1, y1 + corner_len), color, 4, lineType=cv2.LINE_AA)
                cv2.line(canvas, (x2, y1), (x2 - corner_len, y1), color, 4, lineType=cv2.LINE_AA)
                cv2.line(canvas, (x2, y1), (x2, y1 + corner_len), color, 4, lineType=cv2.LINE_AA)
                cv2.line(canvas, (x1, y2), (x1 + corner_len, y2), color, 4, lineType=cv2.LINE_AA)
                cv2.line(canvas, (x1, y2), (x1, y2 - corner_len), color, 4, lineType=cv2.LINE_AA)
                cv2.line(canvas, (x2, y2), (x2 - corner_len, y2), color, 4, lineType=cv2.LINE_AA)
                cv2.line(canvas, (x2, y2), (x2 - corner_len, y2), color, 4, lineType=cv2.LINE_AA)

            # 2. Titik Pusat (Center dot)
            cv2.circle(canvas, (cx, cy), 4, (0, 0, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), 6, color, 1, lineType=cv2.LINE_AA)

            # 3. Label Badge dengan background gelap semi-transparan
            label_text = f"{cls_name} {conf * 100:.1f}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

            badge_y1 = max(0, y1 - text_h - 10)
            badge_y2 = y1
            badge_x1 = x1
            badge_x2 = x1 + text_w + 12

            # Gambar background pill label
            sub_img = canvas[badge_y1:badge_y2, badge_x1:badge_x2]
            if sub_img.shape[0] > 0 and sub_img.shape[1] > 0:
                bg_color = np.array([20, 20, 20], dtype=np.uint8)
                blend = cv2.addWeighted(sub_img, 0.2, np.full_like(sub_img, bg_color), 0.8, 0)
                canvas[badge_y1:badge_y2, badge_x1:badge_x2] = blend
                cv2.rectangle(canvas, (badge_x1, badge_y1), (badge_x2, badge_y2), color, 1, lineType=cv2.LINE_AA)

            cv2.putText(canvas, label_text, (badge_x1 + 6, badge_y2 - 5), font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)

    return canvas, counts, det_details


def render_hud_overlay(image, filename, img_idx, total_imgs, inference_time_ms, counts, conf_val, iou_val):
    """Menampilkan status bar dan HUD informasi di atas gambar."""
    if not show_overlay_info:
        return image

    h, w = image.shape[:2]
    canvas = image.copy()

    # --- Header Bar (Atas) ---
    header_h = 42
    header_overlay = canvas[0:header_h, 0:w].copy()
    cv2.rectangle(canvas, (0, 0), (w, header_h), (25, 25, 30), -1)
    canvas[0:header_h, 0:w] = cv2.addWeighted(header_overlay, 0.25, canvas[0:header_h, 0:w], 0.75, 0)

    # Info File & Index
    file_info = f"[{img_idx + 1}/{total_imgs}] {Path(filename).name}"
    cv2.putText(canvas, file_info, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # Info Inferensi & Threshold
    right_info = f"Conf: {conf_val*100:.0f}% | IoU: {iou_val*100:.0f}% | Speed: {inference_time_ms:.1f}ms"
    (rw, rh), _ = cv2.getTextSize(right_info, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    cv2.putText(canvas, right_info, (w - rw - 15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 230, 255), 1, cv2.LINE_AA)

    # --- Footer Bar (Bawah) --- Ringkasan Objek Terdeteksi
    footer_h = 36
    footer_overlay = canvas[h - footer_h:h, 0:w].copy()
    cv2.rectangle(canvas, (0, h - footer_h), (w, h), (20, 20, 25), -1)
    canvas[h - footer_h:h, 0:w] = cv2.addWeighted(footer_overlay, 0.25, canvas[h - footer_h:h, 0:w], 0.75, 0)

    total_detected = sum(counts.values())
    if total_detected == 0:
        summary_str = "Status: TIDAK ADA OBJEK TERDETEKSI (Coba turunkan confidence)"
        sum_color = (130, 130, 255)
    else:
        parts = [f"{cls}: {cnt}" for cls, cnt in counts.items()]
        summary_str = f"Deteksi ({total_detected} total): " + " | ".join(parts)
        sum_color = (0, 255, 150)

    cv2.putText(canvas, summary_str, (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, sum_color, 1, cv2.LINE_AA)

    # Shortcut tips di pojok kanan bawah
    tips_str = "[N/Space] Lanjut | [P] Mundur | [S] Simpan | [O] Pilih File | [Q] Keluar"
    (tw, th), _ = cv2.getTextSize(tips_str, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    if w - tw - 15 > 350:
        cv2.putText(canvas, tips_str, (w - tw - 15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

    return canvas


def run_inference_and_display():
    """Menjalankan prediksi YOLO pada gambar aktif dan menampilkan hasilnya ke window."""
    global current_raw_img, current_image_idx, image_list, model, conf_threshold, iou_threshold

    if not image_list or current_image_idx >= len(image_list):
        return

    img_path = image_list[current_image_idx]
    
    # Baca gambar
    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Gagal membaca gambar: {img_path}")
        return

    current_raw_img = img.copy()

    # Jalankan YOLO Predict
    t_start = time.time()
    results = model.predict(
        source=img,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False
    )
    inference_time = (time.time() - t_start) * 1000.0  # milidetik

    # Gambar deteksi
    annotated_img, counts, det_details = draw_custom_detections(img, results, conf_threshold)

    # Render HUD
    final_view = render_hud_overlay(
        annotated_img,
        img_path,
        current_image_idx,
        len(image_list),
        inference_time,
        counts,
        conf_threshold,
        iou_threshold
    )

    cv2.imshow(window_name, final_view)
    return final_view, det_details


def on_conf_trackbar(val):
    """Callback saat slider confidence digeser."""
    global conf_threshold
    conf_threshold = max(0.01, val / 100.0)
    run_inference_and_display()


def on_iou_trackbar(val):
    """Callback saat slider IoU digeser."""
    global iou_threshold
    iou_threshold = max(0.01, val / 100.0)
    run_inference_and_display()


def save_current_result(image_to_save, original_path):
    """Menyimpan hasil anotasi gambar ke direktori output."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    orig_name = Path(original_path).stem
    ext = Path(original_path).suffix or ".jpg"
    out_filename = f"pred_{orig_name}_conf{int(conf_threshold*100)}{ext}"
    out_path = os.path.join(OUTPUT_DIR, out_filename)
    
    cv2.imwrite(out_path, image_to_save)
    print(f"\n[SUKSES] Hasil deteksi berhasil disimpan ke:\n  -> {os.path.abspath(out_path)}")


def print_detection_summary(image_path, det_details):
    """Mencetak detail deteksi ke terminal console."""
    print("=" * 70)
    print(f"FILE: {image_path}")
    print(f"Total Deteksi: {len(det_details)} objek (Conf Thres: {conf_threshold*100:.0f}%)")
    print("-" * 70)
    if not det_details:
        print("  (Tidak ada objek yang lolos ambang batas confidence)")
    else:
        print(f"{'No':<4} | {'Kelas':<12} | {'Confidence':<12} | {'Center (X,Y)':<14} | {'Box (W, H)':<12}")
        print("-" * 70)
        for d in det_details:
            no = d["id"]
            cls = d["class"]
            conf = f"{d['conf']*100:.1f}%"
            center = f"({d['center'][0]}, {d['center'][1]})"
            box_dim = f"({d['dim'][0]}x{d['dim'][1]})"
            print(f"{no:<4} | {cls:<12} | {conf:<12} | {center:<14} | {box_dim:<12}")
    print("=" * 70)


def main():
    global model, image_list, current_image_idx, conf_threshold, iou_threshold, show_overlay_info

    print("\n" + "=" * 70)
    print("       PROGRAM PENGUJIAN MODEL YOLO (IMAGE MODE)")
    print("=" * 70)

    # 1. Tentukan Path Model
    model_path = DEFAULT_MODEL_PATH
    if len(sys.argv) > 1 and sys.argv[1].endswith(('.pt', '.onnx', '.engine')):
        model_path = sys.argv[1]

    if not os.path.exists(model_path):
        print(f"\n[PERINGATAN] File model tidak ditemukan di:\n  {model_path}")
        print("\nSilakan pilih file model YOLO (.pt) secara manual...")
        root = tk.Tk()
        root.withdraw()
        model_path = filedialog.askopenfilename(
            title="Pilih File Bobot YOLO (.pt)",
            filetypes=[("PyTorch YOLO Weights", "*.pt"), ("All Files", "*.*")]
        )
        root.destroy()
        if not model_path or not os.path.exists(model_path):
            print("[ERROR] Model tidak dipilih. Program dibatalkan.")
            return

    # 2. Muat Model YOLO
    print(f"\n[*] Memuat Model YOLO dari:\n    {model_path}")
    try:
        model = YOLO(model_path)
        print("[+] Model berhasil dimuat!")
        print(f"    - Task Model  : {model.task}")
        print(f"    - Daftar Kelas: {model.names}")
    except Exception as e:
        print(f"[ERROR] Gagal memuat model: {e}")
        return

    # 3. Tentukan Gambar Input
    input_path = None
    if len(sys.argv) > 2:
        input_path = sys.argv[2]
    elif len(sys.argv) == 2 and not sys.argv[1].endswith(('.pt', '.onnx', '.engine')):
        input_path = sys.argv[1]

    if input_path and os.path.exists(input_path):
        image_list = load_images_from_path(input_path)
    
    # Jika belum ada gambar yang ditentukan, buka file dialog
    if not image_list:
        print("\n[*] Silakan pilih gambar atau folder gambar yang ingin diuji...")
        selected_file = select_file_dialog()
        if selected_file:
            image_list = [selected_file]
        else:
            # Coba cari gambar contoh di direktori lokal
            local_imgs = load_images_from_path(".")
            if local_imgs:
                image_list = local_imgs
                print(f"[+] Ditemukan {len(local_imgs)} gambar di folder proyek saat ini.")
            else:
                print("[!] Tidak ada gambar yang dipilih. Menunggu gambar via tombol [O]...")

    # 4. Siapkan Window OpenCV & Trackbars
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1080, 720)

    # Trackbar untuk mengatur Confidence dan IoU secara dinamis
    cv2.createTrackbar("Confidence (%)", window_name, int(conf_threshold * 100), 100, on_conf_trackbar)
    cv2.createTrackbar("IoU / NMS (%)", window_name, int(iou_threshold * 100), 100, on_iou_trackbar)

    print("\n" + "-" * 70)
    print("PETUNJUK KONTROL KEYBOARD:")
    print("  [D] / [Right] / [Space] / [N] : Gambar Berikutnya (Next)")
    print("  [A] / [Left]  / [P]           : Gambar Sebelumnya (Prev)")
    print("  [S]                           : Simpan Hasil Deteksi ke Folder")
    print("  [O]                           : Buka File Gambar Baru (File Dialog)")
    print("  [F]                           : Buka Folder Gambar (Batch Mode)")
    print("  [T]                           : Sembunyikan / Tampilkan Info Overlay")
    print("  [R]                           : Reset Trackbar ke Nilai Default")
    print("  [Q] / [ESC]                   : Keluar dari Program")
    print("-" * 70 + "\n")

    current_image_idx = 0
    last_view = None

    if image_list:
        last_view, details = run_inference_and_display()
        print_detection_summary(image_list[current_image_idx], details)

    # Main Loop Interaktif
    while True:
        key = cv2.waitKey(30) & 0xFF

        # Tombol Keluar: Q atau ESC
        if key == ord('q') or key == 27:
            print("\n[INFO] Menutup program.")
            break

        # Tombol Next Image: D, N, Space, atau Arrow Right (kode 83 atau 2555904)
        elif key in (ord('d'), ord('n'), 32, 83):
            if image_list and len(image_list) > 1:
                current_image_idx = (current_image_idx + 1) % len(image_list)
                last_view, details = run_inference_and_display()
                print_detection_summary(image_list[current_image_idx], details)

        # Tombol Prev Image: A, P, atau Arrow Left (kode 81)
        elif key in (ord('a'), ord('p'), 81):
            if image_list and len(image_list) > 1:
                current_image_idx = (current_image_idx - 1 + len(image_list)) % len(image_list)
                last_view, details = run_inference_and_display()
                print_detection_summary(image_list[current_image_idx], details)

        # Tombol Simpan Hasil: S
        elif key == ord('s'):
            if last_view is not None and image_list:
                save_current_result(last_view, image_list[current_image_idx])

        # Tombol Buka File Gambar Baru: O
        elif key == ord('o'):
            f = select_file_dialog()
            if f:
                image_list = [f]
                current_image_idx = 0
                last_view, details = run_inference_and_display()
                print_detection_summary(image_list[current_image_idx], details)

        # Tombol Buka Folder: F
        elif key == ord('f'):
            folder = select_folder_dialog()
            if folder:
                new_list = load_images_from_path(folder)
                if new_list:
                    image_list = new_list
                    current_image_idx = 0
                    print(f"\n[+] Memuat {len(image_list)} gambar dari folder: {folder}")
                    last_view, details = run_inference_and_display()
                    print_detection_summary(image_list[current_image_idx], details)
                else:
                    print(f"\n[!] Tidak ditemukan format gambar yang didukung di folder tersebut.")

        # Tombol Toggle Overlay: T
        elif key == ord('t'):
            show_overlay_info = not show_overlay_info
            last_view, _ = run_inference_and_display()

        # Tombol Reset Trackbars: R
        elif key == ord('r'):
            conf_threshold = DEFAULT_CONF_THRESHOLD
            iou_threshold = DEFAULT_IOU_THRESHOLD
            cv2.setTrackbarPos("Confidence (%)", window_name, int(conf_threshold * 100))
            cv2.setTrackbarPos("IoU / NMS (%)", window_name, int(iou_threshold * 100))
            last_view, details = run_inference_and_display()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
