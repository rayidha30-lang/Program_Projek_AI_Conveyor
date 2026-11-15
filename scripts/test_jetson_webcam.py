#!/usr/bin/env python3
"""
Tes cepat kamera USB/webcam di Jetson (V4L2).

Contoh:
  python3 scripts/test_jetson_webcam.py --list
  python3 scripts/test_jetson_webcam.py --cam 0
  python3 scripts/test_jetson_webcam.py --cam /dev/video2 --width 1920 --height 1080 --fps 30
  python3 scripts/test_jetson_webcam.py --cam 0 --headless --seconds 10   # via SSH tanpa monitor

Tombol (mode tampilan):  [S] simpan snapshot   [Q]/[ESC] keluar
"""
import argparse
import glob
import os
import sys
import time

import cv2


def list_devices():
    devices = sorted(glob.glob("/dev/video*"))
    if not devices:
        print("[!] Tidak ada /dev/video* ditemukan. Cek kabel USB atau jalankan: lsusb")
        return
    print("[*] Perangkat video terdeteksi:")
    for dev in devices:
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        ok = cap.isOpened() and cap.read()[0]
        cap.release()
        print(f"    {dev:<14} {'BISA dibaca' if ok else 'tidak menghasilkan frame (mungkin node metadata)'}")
    print("\nDetail format: v4l2-ctl --list-formats-ext -d /dev/videoX")


def open_camera(cam, width, height, fps):
    source = int(cam) if str(cam).isdigit() else cam
    cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None
    # MJPG wajib untuk resolusi tinggi di USB 2.0, tanpa ini biasanya macet di 5 FPS
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def describe(cap):
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4))
    return (
        f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
        f"@ {cap.get(cv2.CAP_PROP_FPS):.0f} FPS | Format: {fourcc_str}"
    )


def main():
    ap = argparse.ArgumentParser(description="Tes webcam USB di Jetson")
    ap.add_argument("--cam", default="0", help="Index (0) atau path (/dev/video0)")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--list", action="store_true", help="Tampilkan semua /dev/video* lalu keluar")
    ap.add_argument("--headless", action="store_true", help="Tanpa jendela (untuk SSH)")
    ap.add_argument("--seconds", type=float, default=10.0, help="Durasi tes mode headless")
    ap.add_argument("--out", default="webcam_snapshots", help="Folder snapshot")
    args = ap.parse_args()

    if args.list:
        list_devices()
        return 0

    cap = open_camera(args.cam, args.width, args.height, args.fps)
    if cap is None:
        print(f"[!] Gagal membuka kamera '{args.cam}'. Coba --list, cek izin: groups (harus ada 'video').")
        return 1

    print(f"[*] Kamera '{args.cam}' terbuka: {describe(cap)}")
    if not args.headless and not os.environ.get("DISPLAY"):
        print("[!] DISPLAY tidak ditemukan, otomatis pindah ke mode --headless.")
        args.headless = True

    os.makedirs(args.out, exist_ok=True)
    t_start = time.time()
    t_prev = t_start
    fps_ema = 0.0
    frames = 0
    drops = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                drops += 1
                if drops >= 30:
                    print("[!] 30 frame gagal berturut-turut. Kamera terputus / bandwidth USB tidak cukup.")
                    break
                continue
            drops = 0
            frames += 1
            now = time.time()
            inst = 1.0 / max(1e-6, now - t_prev)
            fps_ema = inst if fps_ema == 0.0 else 0.9 * fps_ema + 0.1 * inst
            t_prev = now

            if args.headless:
                if frames % 30 == 0:
                    print(f"    frame={frames} FPS={fps_ema:.1f} shape={frame.shape}")
                if now - t_start >= args.seconds:
                    path = os.path.join(args.out, f"snapshot_{int(now)}.jpg")
                    cv2.imwrite(path, frame)
                    print(f"[*] Snapshot terakhir disimpan: {path}")
                    break
                continue

            view = frame.copy()
            h, w = view.shape[:2]
            cv2.putText(view, f"FPS {fps_ema:.1f} | {w}x{h} | [S] snapshot  [Q] keluar",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            # Garis bantu tengah untuk cek kelurusan pemasangan kamera
            cv2.line(view, (w // 2, 0), (w // 2, h), (0, 200, 255), 1)
            cv2.line(view, (0, h // 2), (w, h // 2), (0, 200, 255), 1)
            if w > 1280:
                view = cv2.resize(view, (1280, int(h * 1280 / w)))
            cv2.imshow("Jetson Webcam Test", view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("s"), ord("S")):
                path = os.path.join(args.out, f"snapshot_{int(now)}.jpg")
                cv2.imwrite(path, frame)
                print(f"[*] Snapshot disimpan (resolusi penuh): {path}")
    finally:
        elapsed = max(1e-6, time.time() - t_start)
        cap.release()
        cv2.destroyAllWindows()
        print(f"[*] Selesai: {frames} frame dalam {elapsed:.1f} s (rata-rata {frames / elapsed:.1f} FPS)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
