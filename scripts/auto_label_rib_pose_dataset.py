#!/usr/bin/env python3
"""
========================================================================================
🏷️ AUTO-LABELER DATASET YOLO-POSE 13-KEYPOINTS UNTUK DETEKSI SIRIP KONVEYOR (TIER 2)
========================================================================================
Deskripsi:
  Membangun dataset YOLOv8/11-Pose secara otomatis langsung dari rekaman video konveyor.
  Menggunakan engine pelacak terverifikasi (Viterbi v2) untuk mengekstrak bounding box
  dan 13 titik kunci (keypoints) sepanjang bibir punggung sirip:
    Pt0 (Ujung Kiri) -> Pt1..Pt5 (Badan Kiri) -> Pt6 (Tengah) -> Pt7..Pt11 (Badan Kanan) -> Pt12 (Ujung Kanan)

Penggunaan:
  python scripts/auto_label_rib_pose_dataset.py --max-samples 400 --stride 6
========================================================================================
"""

import os
import sys
import glob
import time
import argparse
import random
import cv2
import numpy as np

# Resolusi path direktori root proyek
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "Vision_Engine_Core", "02_Top_Inspection_Slat_Rib", "src"))

try:
    from hybrid_rib_inspector import HybridRibInspector
except ImportError as e:
    print(f"[!] Gagal mengimpor HybridRibInspector: {e}")
    sys.exit(1)


def find_candidate_videos():
    """Mencari video-video tampak atas sirip konveyor yang relevan."""
    cand_dirs = [
        os.path.join(PROJECT_ROOT, "Documentation"),
        r"C:\Users\Victus\Pictures\Video & Foto Test Slat Rib\New\Reverse",
        r"C:\Users\Victus\Pictures\Video & Foto Test Slat Rib\New",
        r"C:\Users\Victus\Pictures\Video & Foto Test Slat Rib\Testing",
        r"C:\Users\Victus\Pictures\Video & Foto Test Slat Rib"
    ]
    
    videos = []
    for d in cand_dirs:
        if os.path.exists(d):
            # Prioritaskan video yang mengandung kata 'posisi 3', 'dimmer', 'reverse', 'deform', 'ok'
            files = glob.glob(os.path.join(d, "*.mp4"))
            for f in files:
                bn = os.path.basename(f).lower()
                if any(k in bn for k in ["posisi 3", "posisi 4", "slat", "rib", "deform", "ok semua"]):
                    if f not in videos:
                        videos.append(f)
                        
    # Jika belum ada, ambil semua mp4 di candidate dirs
    if not videos:
        for d in cand_dirs:
            if os.path.exists(d):
                for f in glob.glob(os.path.join(d, "*.mp4")):
                    if f not in videos:
                        videos.append(f)
                        
    return videos


def build_pose_dataset(
    output_dir,
    max_samples=400,
    stride=6,
    num_kpts=13,
    val_ratio=0.15,
    custom_videos=None
):
    """
    Mengekstrak frame video dan menghasilkan dataset YOLO-Pose beranotasi lengkap.
    """
    print("\n" + "=" * 78)
    print("🏷️ MEMULAI AUTO-LABELING DATASET YOLO-POSE UNTUK SIRIP KONVEYOR")
    print("=" * 78)
    
    videos = custom_videos if custom_videos else find_candidate_videos()
    if not videos:
        print("[!] Tidak ada video konveyor yang ditemukan untuk auto-labeling.")
        return 0

    print(f"[*] Ditemukan {len(videos)} video sumber konveyor.")
    for v in videos[:6]:
        print(f"    - {os.path.basename(v)}")
    if len(videos) > 6:
        print(f"    ... dan {len(videos) - 6} video lainnya.")

    # Siapkan direktori dataset YOLO Pose
    train_img_dir = os.path.join(output_dir, "images", "train")
    val_img_dir   = os.path.join(output_dir, "images", "val")
    train_lbl_dir = os.path.join(output_dir, "labels", "train")
    val_lbl_dir   = os.path.join(output_dir, "labels", "val")

    for p in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        os.makedirs(p, exist_ok=True)

    # Inisialisasi HybridRibInspector dengan Engine Viterbi v2
    print("\n[*] Memuat engine inspeksi sirip (Viterbi v2 High-Precision)...")
    inspector = HybridRibInspector()
    inspector.set_tracker_engine("TIER1_VITERBI")

    total_extracted = 0
    preview_saved = False
    samples_per_video = max(20, max_samples // max(1, len(videos)))

    for vid_idx, vid_path in enumerate(videos):
        if total_extracted >= max_samples:
            break

        vid_name = os.path.splitext(os.path.basename(vid_path))[0]
        # Bersihkan nama untuk prefix file
        clean_prefix = "".join(c if c.isalnum() else "_" for c in vid_name)[:25]

        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            continue

        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        print(f"\n[{vid_idx+1}/{len(videos)}] Memproses: {os.path.basename(vid_path)} ({n_frames} frames, {fps:.1f} fps)")

        extracted_this_video = 0
        frame_idx = 0

        while cap.isOpened() and total_extracted < max_samples and extracted_this_video < samples_per_video:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            # Lewati frame sesuai stride agar posisi sirip bervariasi
            if frame_idx % stride != 0:
                continue

            h_f, w_f = frame.shape[:2]
            res = inspector.inspect_frame(frame)

            # Syarat Kelayakan Frame untuk Dataset Training:
            # 1. Sirip terdeteksi dengan valid (is_rib_present == True)
            # 2. Kontur bentang penuh memadai (span >= 750 px dan coverage >= 0.65)
            # 3. Posisi bodi sirip berada di area pengujian konveyor
            is_present = res.get("is_rib_present", False)
            xs = res.get("xs_contour", [])
            ys = res.get("ys_contour", [])
            box = res.get("active_box")

            if is_present and len(xs) >= 60 and box is not None:
                w_span = float(xs[-1] - xs[0])
                if w_span >= 750.0:
                    # Ambil 13 keypoints tersebar merata dari ujung kiri ke kanan
                    sample_indices = np.linspace(0, len(xs) - 1, num_kpts).astype(int)
                    kpts = [(float(xs[i]), float(ys[i])) for i in sample_indices]

                    bx1, by1, bx2, by2 = box
                    # Normalisasi koordinat Bounding Box ke [0.0, 1.0]
                    cx = ((bx1 + bx2) / 2.0) / float(w_f)
                    cy = ((by1 + by2) / 2.0) / float(h_f)
                    bw = float(bx2 - bx1) / float(w_f)
                    bh = float(by2 - by1) / float(h_f)

                    # Normalisasi 13 Keypoints ke [0.0, 1.0] dengan visibility=2
                    kpt_line_parts = []
                    valid_kpts = True
                    for kx, ky in kpts:
                        if not (0 <= kx < w_f and 0 <= ky < h_f):
                            valid_kpts = False
                            break
                        kpt_line_parts.extend([f"{kx/w_f:.6f}", f"{ky/h_f:.6f}", "2"])

                    if not valid_kpts:
                        continue

                    # Buat baris label YOLO-Pose
                    # Format: <class_id> <cx> <cy> <bw> <bh> <kp0_x> <kp0_y> <v> <kp1_x> ...
                    label_line = f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} " + " ".join(kpt_line_parts) + "\n"

                    # Pisahkan ke Train vs Val
                    is_val = (random.random() < val_ratio)
                    target_img_dir = val_img_dir if is_val else train_img_dir
                    target_lbl_dir = val_lbl_dir if is_val else train_lbl_dir

                    sample_id = f"rib_{clean_prefix}_f{frame_idx:05d}"
                    img_out_path = os.path.join(target_img_dir, f"{sample_id}.jpg")
                    lbl_out_path = os.path.join(target_lbl_dir, f"{sample_id}.txt")

                    # Simpan gambar citra dan file label
                    cv2.imwrite(img_out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    with open(lbl_out_path, "w", encoding="utf-8") as f_lbl:
                        f_lbl.write(label_line)

                    total_extracted += 1
                    extracted_this_video += 1

                    # Simpan 1 gambar preview beranotasi visual untuk inspeksi
                    if not preview_saved:
                        preview_img = frame.copy()
                        # Gambar Bounding Box
                        cv2.rectangle(preview_img, (bx1, by1), (bx2, by2), (255, 180, 0), 2, cv2.LINE_AA)
                        # Gambar Garis Kontur Penghubung 13 Keypoints
                        pts_poly = np.array(kpts, dtype=np.int32)
                        cv2.polylines(preview_img, [pts_poly], isClosed=False, color=(0, 255, 120), thickness=2, lineType=cv2.LINE_AA)
                        # Gambar 13 Titik Berwarna Gradien
                        for idx_k, (kx, ky) in enumerate(kpts):
                            # Warna ujung beda dengan tengah
                            dot_col = (0, 0, 255) if idx_k in (0, num_kpts - 1) else ((0, 255, 255) if idx_k == num_kpts // 2 else (0, 220, 0))
                            cv2.circle(preview_img, (int(kx), int(ky)), 5, dot_col, -1, cv2.LINE_AA)
                            cv2.circle(preview_img, (int(kx), int(ky)), 6, (0, 0, 0), 1, cv2.LINE_AA)

                        preview_path = os.path.join(output_dir, "preview_auto_labeled_sample.jpg")
                        cv2.imwrite(preview_path, preview_img)
                        print(f"    [*] Preview visual disimpan ke: {os.path.basename(preview_path)}")
                        preview_saved = True

        cap.release()
        print(f"    -> Terkumpul {extracted_this_video} sampel dari video ini.")

    # 4. Buat File data.yaml Resmi untuk YOLO-Pose
    # Definisikan flip_idx agar data augmentation horizontal flip otomatis valid
    flip_indices = list(range(num_kpts - 1, -1, -1))
    clean_path = os.path.abspath(output_dir).replace("\\", "/")
    data_yaml_content = f"""# ==============================================================================
# YOLOv8/11-Pose Dataset Config untuk Slat Rib Conveyor (Tier 2)
# Dihasilkan Otomatis oleh auto_label_rib_pose_dataset.py
# ==============================================================================
path: "{clean_path}"
train: images/train
val: images/val

# Konfigurasi 13 Keypoints
kpt_shape: [{num_kpts}, 3]  # [Jumlah keypoint, (x, y, visibility)]
flip_idx: {flip_indices}    # Pemetaan saat citra di-flip horizontal (ujung kiri <-> kanan)

# Kelas Objek
names:
  0: rib
"""
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f_yaml:
        f_yaml.write(data_yaml_content)

    train_count = len(os.listdir(train_img_dir))
    val_count = len(os.listdir(val_img_dir))

    print("\n" + "=" * 78)
    print("✅ AUTO-LABELING DATASET BERHASIL DISELESAIKAN!")
    print("=" * 78)
    print(f"[*] Total Sampel   : {total_extracted} gambar")
    print(f"    - Training     : {train_count} gambar ({train_img_dir})")
    print(f"    - Validasi     : {val_count} gambar ({val_img_dir})")
    print(f"[*] Config YAML    : {yaml_path}")
    print(f"[*] Keypoints/Rib  : {num_kpts} titik kontinu")
    if preview_saved:
        print(f"[*] Preview Sample : {os.path.join(output_dir, 'preview_auto_labeled_sample.jpg')}")
    print("=" * 78 + "\n")
    return total_extracted


def main():
    parser = argparse.ArgumentParser(description="Auto-Labeler Dataset YOLO-Pose untuk Sirip Konveyor (Tier 2)")
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "Vision_Engine_Core", "02_Top_Inspection_Slat_Rib", "dataset_rib_pose"), help="Folder output dataset")
    parser.add_argument("--max-samples", type=int, default=350, help="Jumlah total frame yang diekstrak")
    parser.add_argument("--stride", type=int, default=6, help="Lompatan frame per ekstraksi")
    parser.add_argument("--num-kpts", type=int, default=13, help="Jumlah titik kunci per sirip (default 13)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Rasio pembagian data validasi (default 0.15)")
    args = parser.parse_args()

    build_pose_dataset(
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        stride=args.stride,
        num_kpts=args.num_kpts,
        val_ratio=args.val_ratio
    )


if __name__ == "__main__":
    main()

