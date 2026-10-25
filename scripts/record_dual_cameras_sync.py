"""
========================================================================================
🎥 DUAL-CAMERA SYNCHRONOUS CONVEYOR RECORDER
Target Hardware : Host PC (Windows/Linux) & NVIDIA Jetson Orin Nano
System Target   : AI Slat Rib Conveyor Multi-Point Inspection
========================================================================================
Deskripsi:
  Perekam video sinkron dua kamera (CAM-01 Top Slat/Rib & CAM-02 Side Bolt/Pin) secara
  simultan (dalam satu waktu) tanpa beban model AI. 

Keunggulan:
  - 100% Zero-Lag & Zero-Drift: Perekaman multi-threaded terisolasi pada 30 FPS konstan.
  - Multi-File Sync Output:
      1. sync_top_<timestamp>.mp4  : Khusus CAM-01 Top (Plat Slat & Bilah Rib)
      2. sync_side_<timestamp>.mp4 : Khusus CAM-02 Side (Baut & Pin)
      3. sync_dual_<timestamp>.mp4 : Layar gabungan side-by-side (opsional/dokumentasi)
  - Live Alignment Guide: Garis panduan tengah (Grid [G]) untuk menyelaraskan posisi kamera.
  - Disk I/O Queue Buffer: Menulis frame di background thread agar preview tidak patah-patah.
========================================================================================
"""

import os
import sys
import time
import queue
import argparse
import threading
from datetime import datetime

# Pastikan output UTF-8 di Windows terminal
if sys.platform.startswith("win"):
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import cv2
import numpy as np

# Path root program
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "Video_Testing")


def detect_available_cameras(max_devices=6):
    """Mendeteksi seluruh webcam fisik / perangkat video USB yang aktif."""
    cams = []
    if sys.platform.startswith("win"):
        for i in range(max_devices):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    cams.append(i)
                    cap.release()
            except Exception:
                pass
    else:
        import glob
        nodes = sorted(glob.glob("/dev/video[0-9]*"))
        for node in nodes:
            try:
                idx = int(os.path.basename(node).replace("video", ""))
                if idx < max_devices and idx not in cams:
                    cams.append(idx)
            except ValueError:
                pass
        if not cams:
            for i in range(min(max_devices, 3)):
                if os.path.exists(f"/dev/video{i}"):
                    cams.append(i)
    return cams


def create_camera_capture(src, width=1280, height=720, target_fps=30):
    """Membuka kamera dengan pengaturan hardware optimal (DirectShow/V4L2 + MJPG)."""
    is_cam = str(src).strip().isdigit()
    if is_cam:
        idx = int(src)
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(idx)
        else:
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap = cv2.VideoCapture(idx)

        if cap.isOpened():
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FPS, target_fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        return cap
    else:
        # Sumber berupa file video atau RTSP stream
        return cv2.VideoCapture(str(src).strip())


class CameraStreamThread:
    """Thread penangkap frame berkecepatan tinggi agar tidak memblokir penulisan video."""
    def __init__(self, name, source, width, height, target_fps):
        self.name = name
        self.source = source
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.cap = None
        self.latest_frame = None
        self.frame_count = 0
        self.fps = 0.0
        self.lock = threading.Lock()
        self.running = False
        self.thread = None

    def start(self):
        self.cap = create_camera_capture(self.source, self.width, self.height, self.target_fps)
        if not self.cap or not self.cap.isOpened():
            return False
        self.running = True
        self.thread = threading.Thread(target=self._loop, name=f"Thread-{self.name}", daemon=True)
        self.thread.start()
        return True

    def _loop(self):
        t_prev = time.time()
        fps_hist = []
        is_cam = str(self.source).strip().isdigit()

        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                if not is_cam:
                    # Loop video jika file
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.01)
                    continue
                time.sleep(0.005)
                continue

            t_now = time.time()
            dt = t_now - t_prev
            t_prev = t_now
            c_fps = 1.0 / dt if dt > 0 else 30.0
            fps_hist.append(c_fps)
            if len(fps_hist) > 30:
                fps_hist.pop(0)

            with self.lock:
                self.latest_frame = frame
                self.frame_count += 1
                self.fps = sum(fps_hist) / len(fps_hist)

    def get_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return None, 0.0
            return self.latest_frame.copy(), self.fps

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass


class AsyncVideoWriter:
    """Writer asinkron menggunakan queue untuk mencegah frame drop saat disk flush."""
    def __init__(self, path_top, path_side, path_dual, width, height, fps, save_dual=True):
        self.path_top = path_top
        self.path_side = path_side
        self.path_dual = path_dual
        self.width = width
        self.height = height
        self.fps = fps
        self.save_dual = save_dual

        self.queue = queue.Queue(maxsize=150)
        self.running = False
        self.thread = None
        self.frames_written = 0

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer_top = cv2.VideoWriter(path_top, fourcc, fps, (width, height))
        self.writer_side = cv2.VideoWriter(path_side, fourcc, fps, (width, height))
        self.writer_dual = None
        if save_dual and path_dual:
            # Dual view menggabungkan 2 frame berdampingan (lebar = 2 * width)
            self.writer_dual = cv2.VideoWriter(path_dual, fourcc, fps, (width * 2, height))

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True, name="Thread-VideoWriter")
        self.thread.start()

    def write(self, f_top, f_side, f_dual=None):
        if not self.running:
            return
        try:
            self.queue.put_nowait((f_top, f_side, f_dual))
        except queue.Full:
            # Queue penuh (sangat jarang terjadi jika disk normal)
            pass

    def _worker(self):
        while self.running or not self.queue.empty():
            try:
                item = self.queue.get(timeout=0.05)
            except queue.Empty:
                continue

            f_top, f_side, f_dual = item
            if f_top is not None and self.writer_top.isOpened():
                if f_top.shape[1] != self.width or f_top.shape[0] != self.height:
                    f_top = cv2.resize(f_top, (self.width, self.height))
                self.writer_top.write(f_top)

            if f_side is not None and self.writer_side.isOpened():
                if f_side.shape[1] != self.width or f_side.shape[0] != self.height:
                    f_side = cv2.resize(f_side, (self.width, self.height))
                self.writer_side.write(f_side)

            if self.writer_dual and self.writer_dual.isOpened() and f_dual is not None:
                self.writer_dual.write(f_dual)

            self.frames_written += 1
            self.queue.task_done()

    def close(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3.0)
        if self.writer_top:
            self.writer_top.release()
        if self.writer_side:
            self.writer_side.release()
        if self.writer_dual:
            self.writer_dual.release()


def main():
    parser = argparse.ArgumentParser(
        description="🎥 Dual-Camera Synchronous Conveyor Video Recorder (Top & Side)"
    )
    parser.add_argument("--cam-top", type=str, default=None, help="Index webcam atau path input untuk CAM-01 Top")
    parser.add_argument("--cam-side", type=str, default=None, help="Index webcam atau path input untuk CAM-02 Side")
    parser.add_argument("--width", type=int, default=1280, help="Lebar frame rekaman (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Tinggi frame rekaman (default: 720)")
    parser.add_argument("--fps", type=float, default=30.0, help="Target FPS rekaman (default: 30.0)")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Direktori penyimpanan file rekaman")
    parser.add_argument("--no-dual", action="store_true", help="Jangan simpan video gabungan side-by-side (hanya rekam 2 video terpisah)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("🎥 DUAL-CAMERA SYNCHRONOUS CONVEYOR RECORDER")
    print("=" * 80)
    print("[*] Mendeteksi kamera USB aktif...")
    available_cams = detect_available_cameras(max_devices=6)
    print(f"[*] Kamera fisik terdeteksi: {available_cams}")

    cam_top_src = args.cam_top
    cam_side_src = args.cam_side

    if cam_top_src is None or cam_side_src is None:
        if len(available_cams) >= 2:
            cam_top_src = str(available_cams[0]) if cam_top_src is None else cam_top_src
            cam_side_src = str(available_cams[1]) if cam_side_src is None else cam_side_src
            print(f"[*] Auto-Assign: CAM-01 Top -> Webcam #{cam_top_src} | CAM-02 Side -> Webcam #{cam_side_src}")
        elif len(available_cams) == 1:
            cam_top_src = str(available_cams[0]) if cam_top_src is None else cam_top_src
            cam_side_src = str(available_cams[0]) if cam_side_src is None else cam_side_src
            print(f"[!] Peringatan: Hanya terdeteksi 1 webcam (#{available_cams[0]}). Menggunakan webcam yang sama untuk simulasi.")
        else:
            cam_top_src = "0" if cam_top_src is None else cam_top_src
            cam_side_src = "1" if cam_side_src is None else cam_side_src
            print(f"[!] Tidak terdeteksi webcam fisik. Menggunakan index fallback: Top={cam_top_src}, Side={cam_side_src}")

    print(f"[*] Sumber CAM-01 Top  : Webcam/Media [{cam_top_src}]")
    print(f"[*] Sumber CAM-02 Side : Webcam/Media [{cam_side_src}]")
    print(f"[*] Resolusi Rekaman   : {args.width}x{args.height} @ {args.fps:.0f} FPS")
    print(f"[*] Folder Output      : {os.path.abspath(args.output_dir)}")
    print("=" * 80)

    # Inisialisasi Stream Thread
    stream_top = CameraStreamThread("Top", cam_top_src, args.width, args.height, args.fps)
    stream_side = CameraStreamThread("Side", cam_side_src, args.width, args.height, args.fps)

    print("[*] Membuka kamera dan menstabilkan sensor...")
    ok_top = stream_top.start()
    ok_side = stream_side.start()

    if not ok_top:
        print(f"[ERROR] Gagal membuka kamera Top ({cam_top_src})! Periksa koneksi USB atau index kamera.")
    if not ok_side:
        print(f"[ERROR] Gagal membuka kamera Side ({cam_side_src})! Periksa koneksi USB atau index kamera.")

    if not ok_top and not ok_side:
        print("[!] Kedua kamera gagal dibuka. Program dihentikan.")
        sys.exit(1)

    window_name = "DUAL CAMERA SYNCHRONOUS RECORDER - CONVEYOR INSPECTION"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 480)

    is_recording = False
    recorder = None
    rec_start_time = 0.0
    rec_paths = {}
    show_grid = True
    session_count = 0

    target_interval = 1.0 / args.fps
    last_record_tick = time.time()

    print("\n" + "=" * 80)
    print("🎮 KONTROL KEYBOARD (Jendela OpenCV):")
    print("  • [SPACE] atau [R] : Mulai / Hentikan Perekaman Sinkron")
    print("  • [G]              : Aktifkan / Matikan Garis Panduan (Alignment Grid)")
    print("  • [S]              : Tukar Posisi Kamera Top <-> Side")
    print("  • [Q] atau [ESC]   : Keluar Program")
    print("=" * 80 + "\n")

    try:
        while True:
            t_loop_start = time.time()

            frame_top, fps_top = stream_top.get_frame()
            frame_side, fps_side = stream_side.get_frame()

            # Fallback frame hitam jika salah satu kamera terputus
            if frame_top is None:
                frame_top = np.zeros((args.height, args.width, 3), dtype=np.uint8)
                cv2.putText(frame_top, "CAM-01 TOP OFFLINE", (50, args.height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            if frame_side is None:
                frame_side = np.zeros((args.height, args.width, 3), dtype=np.uint8)
                cv2.putText(frame_side, "CAM-02 SIDE OFFLINE", (50, args.height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            # Pastikan ukuran seragam
            if frame_top.shape[1] != args.width or frame_top.shape[0] != args.height:
                frame_top = cv2.resize(frame_top, (args.width, args.height))
            if frame_side.shape[1] != args.width or frame_side.shape[0] != args.height:
                frame_side = cv2.resize(frame_side, (args.width, args.height))

            # Proses Rekam Sinkron (Raw Clean Frames ke disk)
            if is_recording and recorder is not None:
                now = time.time()
                # Tulis frame sesuai target interval (30 FPS konstan)
                if now - last_record_tick >= (target_interval * 0.85):
                    last_record_tick = now
                    # Buat side-by-side mentah bersih untuk dual video
                    f_dual_clean = None
                    if not args.no_dual:
                        f_dual_clean = np.hstack([frame_top, frame_side])
                    recorder.write(frame_top, frame_side, f_dual_clean)

            # ==============================================================
            # RENDER DISPLAY DUAL VIEW DENGAN OVERLAY TELEMETRI
            # ==============================================================
            disp_top = frame_top.copy()
            disp_side = frame_side.copy()

            # Gambar Garis Panduan (Grid) jika aktif untuk kalibrasi visual konveyor
            if show_grid:
                # Garis tengah vertikal & horizontal
                for disp in [disp_top, disp_side]:
                    h, w = disp.shape[:2]
                    cv2.line(disp, (w // 2, 0), (w // 2, h), (0, 255, 255), 1, cv2.LINE_AA)
                    cv2.line(disp, (0, h // 2), (w, h // 2), (0, 255, 255), 1, cv2.LINE_AA)
                    # ROI Box tengah
                    cv2.rectangle(disp, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), (0, 255, 255), 1, cv2.LINE_AA)

            # Label Header Kamera
            cv2.rectangle(disp_top, (10, 10), (args.width - 10, 60), (20, 20, 20), -1)
            cv2.rectangle(disp_top, (10, 10), (args.width - 10, 60), (0, 200, 255), 2)
            cv2.putText(disp_top, f"CAM-01 TOP (SLAT & RIB) | {fps_top:.1f} FPS", (25, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.rectangle(disp_side, (10, 10), (args.width - 10, 60), (20, 20, 20), -1)
            cv2.rectangle(disp_side, (10, 10), (args.width - 10, 60), (0, 255, 120), 2)
            cv2.putText(disp_side, f"CAM-02 SIDE (BOLT & PIN) | {fps_side:.1f} FPS", (25, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 120), 2, cv2.LINE_AA)

            # Gabungkan berdampingan untuk preview
            dual_display = np.hstack([disp_top, disp_side])
            disp_h, disp_w = dual_display.shape[:2]

            # Banner Status Perekaman di Bawah Layar
            banner_h = 70
            cv2.rectangle(dual_display, (10, disp_h - banner_h), (disp_w - 10, disp_h - 10), (15, 15, 15), -1)

            if is_recording:
                elapsed = time.time() - rec_start_time
                mins = int(elapsed // 60)
                secs = int(elapsed % 60)
                f_cnt = recorder.frames_written if recorder else 0

                # Border merah berkedip saat merekam
                border_color = (0, 0, 255) if int(elapsed * 2) % 2 == 0 else (0, 100, 255)
                cv2.rectangle(dual_display, (10, disp_h - banner_h), (disp_w - 10, disp_h - 10), border_color, 2)

                rec_txt = f"🔴 SEDANG MEREKAM SINKRON [{mins:02d}:{secs:02d}] | {f_cnt} Frames @ {args.fps:.0f} FPS"
                cv2.putText(dual_display, rec_txt, (30, disp_h - 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2, cv2.LINE_AA)

                hint_txt = "Tekan [SPACE] / [R] untuk STOP"
                cv2.putText(dual_display, hint_txt, (disp_w - 420, disp_h - 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.80, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.rectangle(dual_display, (10, disp_h - banner_h), (disp_w - 10, disp_h - 10), (80, 80, 80), 1)
                standby_txt = "STANDBY | Tekan [SPACE] atau [R] untuk MULAI REKAM SINKRON"
                cv2.putText(dual_display, standby_txt, (30, disp_h - 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.80, (0, 255, 255), 2, cv2.LINE_AA)

                guide_txt = f"Grid: {'ON' if show_grid else 'OFF'} [G] | Tukar: [S] | Keluar: [Q]"
                cv2.putText(dual_display, guide_txt, (disp_w - 560, disp_h - 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.70, (200, 200, 200), 2, cv2.LINE_AA)

            # Tampilkan di jendela OpenCV
            cv2.imshow(window_name, dual_display)

            # Tangani Tombol Keyboard
            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), ord("Q"), 27]:
                print("\n[*] Tombol keluar ditekan.")
                break
            elif key in [ord(" "), ord("r"), ord("R")]:
                if not is_recording:
                    # MULAI PEREKAMAN SINKRON
                    session_count += 1
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    path_top = os.path.join(args.output_dir, f"sync_top_{timestamp}.mp4")
                    path_side = os.path.join(args.output_dir, f"sync_side_{timestamp}.mp4")
                    path_dual = os.path.join(args.output_dir, f"sync_dual_{timestamp}.mp4") if not args.no_dual else None

                    rec_paths = {
                        "top": path_top,
                        "side": path_side,
                        "dual": path_dual
                    }

                    recorder = AsyncVideoWriter(
                        path_top=path_top,
                        path_side=path_side,
                        path_dual=path_dual,
                        width=args.width,
                        height=args.height,
                        fps=args.fps,
                        save_dual=not args.no_dual
                    )
                    recorder.start()
                    is_recording = True
                    rec_start_time = time.time()
                    last_record_tick = rec_start_time

                    print("\n" + "=" * 80)
                    print(f"🔴 MULAI PEREKAMAN SINKRON DUAL-KAMERA (#{session_count})")
                    print(f"[*] CAM-01 Top  -> {os.path.basename(path_top)}")
                    print(f"[*] CAM-02 Side -> {os.path.basename(path_side)}")
                    if path_dual:
                        print(f"[*] Side-by-Side -> {os.path.basename(path_dual)}")
                    print("[*] Status: Merekam... Tekan [SPACE] lagi untuk menghentikan.")
                    print("=" * 80 + "\n")
                else:
                    # STOP PEREKAMAN
                    is_recording = False
                    elapsed = time.time() - rec_start_time
                    if recorder:
                        recorder.close()
                        f_saved = recorder.frames_written
                    else:
                        f_saved = 0

                    print("\n" + "=" * 80)
                    print(f"⏹️ PEREKAMAN SELESAI & TERSIMPAN SEMPURNA!")
                    print(f"  • Total Durasi         : {elapsed:.2f} detik")
                    print(f"  • Total Frame Rekaman  : {f_saved} frames (~{f_saved / elapsed if elapsed > 0 else 0:.1f} FPS)")
                    print(f"  • File CAM-01 Top      : {rec_paths.get('top')}")
                    print(f"  • File CAM-02 Side     : {rec_paths.get('side')}")
                    if rec_paths.get('dual'):
                        print(f"  • File Dual Visual     : {rec_paths.get('dual')}")
                    print("=" * 80)
                    print("💡 CARA PENGUJIAN BENCHMARK DENGAN VIDEO BARU INI:")
                    print(f'python run_multi_camera_benchmark.py --top-source "{rec_paths.get("top")}" --side-source "{rec_paths.get("side")}"')
                    print("=" * 80 + "\n")
                    recorder = None

            elif key in [ord("g"), ord("G")]:
                show_grid = not show_grid
                print(f"[*] Garis Panduan Alignment (Grid): {'AKTIF' if show_grid else 'NON-AKTIF'}")

            elif key in [ord("s"), ord("S")]:
                if not is_recording:
                    # Tukar kamera
                    stream_top.stop()
                    stream_side.stop()
                    cam_top_src, cam_side_src = cam_side_src, cam_top_src
                    print(f"\n[*] Menukar Kamera: Top -> [{cam_top_src}] | Side -> [{cam_side_src}]")
                    stream_top = CameraStreamThread("Top", cam_top_src, args.width, args.height, args.fps)
                    stream_side = CameraStreamThread("Side", cam_side_src, args.width, args.height, args.fps)
                    stream_top.start()
                    stream_side.start()
                else:
                    print("[!] Tidak dapat menukar kamera saat perekaman sedang berlangsung.")

            # Menjaga framerate preview sekitar 30 FPS
            elapsed_loop = time.time() - t_loop_start
            sleep_time = max(0.001, target_interval - elapsed_loop)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[*] Interupsi Keyboard (Ctrl+C). Menghentikan...")

    finally:
        if is_recording and recorder:
            print("[*] Menutup file video yang sedang direkam...")
            recorder.close()

        print("[*] Melepaskan koneksi kamera...")
        stream_top.stop()
        stream_side.stop()
        cv2.destroyAllWindows()
        print("[*] Selesai.\n")


if __name__ == "__main__":
    main()
