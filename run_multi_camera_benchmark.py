"""
========================================================================================
🚀 MULTI-CAMERA INSPECTION BENCHMARK RUNNER (TOP & SIDE INTEGRATION)
Target Hardware : NVIDIA Jetson Orin Nano (ARM64) & Host PC (Windows/Linux x86_64)
System Target   : AI Slat Rib Conveyor Multi-Point Inspection
========================================================================================
Deskripsi:
  Orchestrator pengujian simultan multi-kamera yang mengeksekus   1. Top Inspection  (CAM-01: Plat Slat Hough Metrology + Bilah Rib Hybrid YOLO)
   2. Side Inspection (CAM-02/CAM-03: Baut & Pin YOLO + Lekukan Bracket PatchCore)

Keunggulan Arsitektur:
  - Multi-Process Isolation: Tiap pipeline berjalan di proses Python terpisah (bebas GIL).
  - True Hardware Benchmark: Mengukur throughput dan latensi murni tanpa saling memblokir.
  - Multi-Display Support:
      * Mode 'dual'     : Menampilkan 2 jendela OpenCV berdampingan otomatis (kiri-kanan).
      * Mode 'headless' : Tanpa GUI desktop, mengukur FPS inferensi murni hardware (Web-ready).
      * Mode 'top-only' / 'side-only' : Pengujian baseline per kamera tunggal.
  - Live Telemetry & CSV Export: Ringkasan FPS per kamera, frame count, dan stabilitas beban.
========================================================================================
"""

import os
import sys
import time
import argparse
import multiprocessing as mp
from datetime import datetime

# Pastikan terminal Windows mendukung UTF-8 output tanpa crash codec charmap/cp1252
if sys.platform.startswith("win"):
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Path direktori utama
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _load_env_file():
    """Membaca file .env jika ada (tanpa ketergantungan library eksternal)"""
    candidate_paths = [
        os.path.join(BASE_DIR, ".env"),
        os.path.abspath(os.path.join(BASE_DIR, "..", ".env")),
        os.path.join(BASE_DIR, "Jetson_Vision_Engine", ".env"),
    ]
    for env_path in candidate_paths:
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v
                break
            except Exception:
                pass

_load_env_file()

# Deteksi lokasi pipeline (utamakan Vision_Engine_Core di PC, lalu Jetson_Vision_Engine jika di-deploy mandiri)
if os.path.exists(os.path.join(BASE_DIR, "Vision_Engine_Core", "02_Top_Inspection_Slat_Rib", "src")):
    TOP_SRC_DIR = os.path.join(BASE_DIR, "Vision_Engine_Core", "02_Top_Inspection_Slat_Rib", "src")
    SIDE_SRC_DIR = os.path.join(BASE_DIR, "Vision_Engine_Core", "01_Side_Inspection_Bolt_Pin_Lekukan", "src")
    VIDEO_DIR = os.path.join(BASE_DIR, "Video_Testing")
elif os.path.exists(os.path.join(BASE_DIR, "Jetson_Vision_Engine", "02_Top_Inspection_Slat_Rib", "src")):
    TOP_SRC_DIR = os.path.join(BASE_DIR, "Jetson_Vision_Engine", "02_Top_Inspection_Slat_Rib", "src")
    SIDE_SRC_DIR = os.path.join(BASE_DIR, "Jetson_Vision_Engine", "01_Side_Inspection_Bolt_Pin", "src")
    VIDEO_DIR = os.path.join(BASE_DIR, "Jetson_Vision_Engine", "Video_Testing")
else:
    TOP_SRC_DIR = os.path.join(BASE_DIR, "02_Top_Inspection_Slat_Rib", "src")
    SIDE_SRC_DIR = os.path.join(BASE_DIR, "01_Side_Inspection_Bolt_Pin", "src")
    VIDEO_DIR = os.path.join(BASE_DIR, "Video_Testing")

def is_stream_or_camera(src):
    """Mengecek apakah input berupa index kamera (misal 0, 1) atau URL stream (RTSP, HTTP, GStreamer)."""
    if src is None:
        return False
    s = str(src).strip()
    if s.isdigit():
        return True
    if s.startswith(("rtsp://", "http://", "https://", "nvarguscamerasrc", "v4l2src")):
        return True
    return False


def resolve_source_path(src, candidates=None):
    """
    Menemukan path media/video atau sumber kamera yang valid secara multi-platform.
    Mendukung index webcam, URL RTSP/CSI, absolute path, dan relative path.
    """
    if not src:
        if candidates:
            for cand in candidates:
                res = resolve_source_path(cand)
                if res:
                    return res
        return ""

    s = str(src).strip()
    if is_stream_or_camera(s):
        return s

    # 1. Cek langsung jika path absolute atau relatif terhadap CWD
    if os.path.exists(s):
        return os.path.abspath(s)

    # 2. Cek relatif terhadap BASE_DIR (Program / root)
    rel_base = os.path.join(BASE_DIR, s)
    if os.path.exists(rel_base):
        return os.path.abspath(rel_base)

    # 3. Cek relatif terhadap VIDEO_DIR (Video_Testing)
    rel_video = os.path.join(VIDEO_DIR, s)
    if os.path.exists(rel_video):
        return os.path.abspath(rel_video)

    # 4. Fallback ke kandidat berikutnya jika ada
    if candidates:
        for cand in candidates:
            if cand != src:
                res = resolve_source_path(cand)
                if res:
                    return res
    return s


# Video Default CAM-01 Top Inspection (Plat Slat & Bilah Rib)
ENV_TOP_VIDEO = os.environ.get("TOP_CAMERA_SOURCE")
CANDIDATE_TOP_VIDEOS = [
    ENV_TOP_VIDEO,
    os.path.join(VIDEO_DIR, "Deform Sedikit.mp4"),
    os.path.join(VIDEO_DIR, "OK Semua.mp4"),
    os.path.join(BASE_DIR, "Jetson_Vision_Engine", "Video_Testing", "OK Semua.mp4"),
    "0",
]
DEFAULT_TOP_VIDEO = resolve_source_path(ENV_TOP_VIDEO, CANDIDATE_TOP_VIDEOS)

# Video Default Side Inspection (Baut, Pin & Bracket - CAM-02 RH / CAM-03 LH)
ENV_SIDE_VIDEO = os.environ.get("SIDE_CAMERA_SOURCE")
CANDIDATE_SIDE_VIDEOS = [
    ENV_SIDE_VIDEO,
    os.path.join(VIDEO_DIR, "Deform Sedikit.mp4"),
    os.path.abspath(os.path.join(BASE_DIR, "..", "Documentation", "WhatsApp Video 2026-08-19 at 15.37.46.mp4")),
    os.path.abspath(os.path.join(BASE_DIR, "Documentation", "WhatsApp Video 2026-08-19 at 15.37.46.mp4")),
    "1",
]
DEFAULT_SIDE_VIDEO = resolve_source_path(ENV_SIDE_VIDEO, CANDIDATE_SIDE_VIDEOS)



def configure_windows_process_performance():
    """
    Optimasi performa proses untuk Windows 10/11:
    1. Menonaktifkan Windows EcoQoS (Power Throttling / Efficiency Mode) agar thread AI tidak
       diturunkan ke E-Core atau di-throttle clock-nya saat jendela GUI kehilangan fokus (unfocused).
    2. Menetapkan prioritas proses ke ABOVE_NORMAL_PRIORITY_CLASS agar scheduler OS tidak
       memangkas kuantum CPU saat operator membuka Web Dashboard di browser.
    3. Mengaktifkan multimedia timer 1ms (timeBeginPeriod) untuk akurasi cv2.waitKey().
    """
    if not sys.platform.startswith("win"):
        return

    # 1. Timer multimedia presisi tinggi 1ms
    try:
        import ctypes
        ctypes.windll.winmm.timeBeginPeriod(1)
    except Exception:
        pass

    # 2. Disable EcoQoS & Boost Priority
    try:
        import ctypes
        import os
        from ctypes import wintypes

        k = ctypes.windll.kernel32
        k.OpenProcess.restype = wintypes.HANDLE
        k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k.CloseHandle.argtypes = [wintypes.HANDLE]

        # PROCESS_SET_INFORMATION = 0x0200, PROCESS_QUERY_INFORMATION = 0x0400
        h_proc = k.OpenProcess(0x0200 | 0x0400, False, os.getpid())
        if h_proc:
            try:
                # Set prioritas ke ABOVE_NORMAL_PRIORITY_CLASS (0x00008000)
                k.SetPriorityClass(h_proc, 0x00008000)

                # Nonaktifkan EcoQoS / Power Throttling
                class PROCESS_POWER_THROTTLING_STATE(ctypes.Structure):
                    _fields_ = [
                        ("Version", wintypes.DWORD),
                        ("ControlMask", wintypes.DWORD),
                        ("StateMask", wintypes.DWORD),
                    ]

                state = PROCESS_POWER_THROTTLING_STATE(1, 0x1, 0)
                k.SetProcessInformation.argtypes = [
                    wintypes.HANDLE,
                    wintypes.DWORD,
                    ctypes.c_void_p,
                    wintypes.DWORD
                ]
                k.SetProcessInformation(h_proc, 4, ctypes.byref(state), ctypes.sizeof(state))
            finally:
                k.CloseHandle(h_proc)
    except Exception:
        pass



def detect_available_cameras(max_devices=6):
    """
    Mendeteksi seluruh webcam fisik / perangkat video USB yang aktif pada PC/Jetson.
    Mengembalikan daftar index kamera yang berhasil dibuka dan membaca frame (misal: [0, 1]).
    """
    import cv2
    cams = []
    for i in range(max_devices):
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(i)
        if cap is not None and cap.isOpened():
            ret, _ = cap.read()
            if ret:
                cams.append(i)
            cap.release()
    return cams


def create_capture(src):
    """
    Membuka sumber video atau webcam dengan konfigurasi performa optimal (DirectShow, MJPG, Buffer=1).
    Mengembalikan tuple: (cap, is_cam, resolved_path).
    """
    import cv2
    is_cam = str(src).strip().isdigit()
    if is_cam:
        idx = int(src)
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        return cap, True, str(idx)
    else:
        path = str(src).strip()
        cap = cv2.VideoCapture(path)
        return cap, False, path


def select_file_dialog(initial_dir=None, title="Pilih File Video"):
    """
    Membuka file dialog explorer Windows (Tkinter) untuk memilih file video/media inspeksi.
    Menjamin jendela dialog muncul paling depan (-topmost) dan tidak memblokir OpenCV.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        # Tentukan direktori awal cerdas
        default_dir = initial_dir
        if not default_dir or not os.path.exists(default_dir):
            candidate_dirs = [
                os.path.join(BASE_DIR, "Jetson_Vision_Engine", "Video_Testing"),
                os.path.abspath(os.path.join(BASE_DIR, "..", "Documentation")),
                os.path.join(BASE_DIR, "Documentation"),
                os.getcwd()
            ]
            for c_dir in candidate_dirs:
                if os.path.exists(c_dir):
                    default_dir = c_dir
                    break

        file_path = filedialog.askopenfilename(
            parent=root,
            title=title,
            initialdir=default_dir,
            filetypes=[
                ("File Video (*.mp4;*.avi;*.mkv;*.mov;*.wmv)", "*.mp4 *.avi *.mkv *.mov *.wmv"),
                ("MP4 Video (*.mp4)", "*.mp4"),
                ("Semua File (*.*)", "*.*")
            ]
        )
        root.update()
        root.destroy()
        return file_path if file_path and os.path.exists(file_path) else None
    except Exception as e:
        print(f"[!] Gagal membuka file dialog: {e}")
        return None


def select_source_dialog(title="Pilih Sumber Kamera / Video", default_source=None):
    """
    Menampilkan dialog visual modern untuk memilih antara Webcam fisik terdeteksi atau File Video.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        cams = detect_available_cameras(max_devices=6)

        root = tk.Tk()
        root.title(title)
        root.attributes("-topmost", True)
        root.geometry("480x370")
        root.resizable(False, False)

        selected_result = {"val": default_source}

        # Header
        lbl_info = tk.Label(
            root,
            text=f"🎥 {title}",
            font=("Segoe UI", 11, "bold"),
            pady=10
        )
        lbl_info.pack()

        frame_options = tk.LabelFrame(root, text="Pilihan Sumber Input", font=("Segoe UI", 10, "bold"), padx=15, pady=10)
        frame_options.pack(fill="both", expand=True, padx=15, pady=5)

        is_default_cam = str(default_source).strip().isdigit() if default_source is not None else False
        var_choice = tk.StringVar(value="webcam" if is_default_cam else "file")
        default_cam_idx = int(default_source) if is_default_cam else (cams[0] if cams else 0)
        var_cam_idx = tk.IntVar(value=default_cam_idx)

        # Radio pilihan Webcam
        if cams:
            rb_cam = tk.Radiobutton(
                frame_options,
                text="Gunakan Perangkat Live Webcam:",
                variable=var_choice,
                value="webcam",
                font=("Segoe UI", 9, "bold")
            )
            rb_cam.pack(anchor="w", pady=(0, 4))

            frame_cams = tk.Frame(frame_options, padx=20)
            frame_cams.pack(fill="x", pady=2)
            for c_idx in cams:
                tk.Radiobutton(
                    frame_cams,
                    text=f"Webcam #{c_idx}  (Terdeteksi Aktif 720p)",
                    variable=var_cam_idx,
                    value=c_idx,
                    command=lambda: var_choice.set("webcam"),
                    font=("Segoe UI", 9)
                ).pack(anchor="w", pady=2)
        else:
            lbl_no = tk.Label(frame_options, text="⚠️ Tidak ada webcam USB yang terdeteksi.", fg="gray")
            lbl_no.pack(anchor="w", pady=2)

        # Radio pilihan File Video
        rb_file = tk.Radiobutton(
            frame_options,
            text="Gunakan File Rekaman Video:",
            variable=var_choice,
            value="file",
            font=("Segoe UI", 9, "bold")
        )
        rb_file.pack(anchor="w", pady=(12, 4))

        lbl_file_path = tk.Label(
            frame_options,
            text=os.path.basename(str(default_source)) if (default_source and not is_default_cam) else "(Belum memilih file)",
            font=("Segoe UI", 8, "italic"),
            fg="#555555"
        )
        lbl_file_path.pack(anchor="w", padx=20)

        file_selected_path = {"path": str(default_source) if (default_source and not is_default_cam) else None}

        def on_browse():
            var_choice.set("file")
            f = filedialog.askopenfilename(
                parent=root,
                title="Pilih File Video Pengujian",
                filetypes=[
                    ("File Video (*.mp4;*.avi;*.mkv;*.mov;*.wmv)", "*.mp4 *.avi *.mkv *.mov *.wmv"),
                    ("Semua File (*.*)", "*.*")
                ]
            )
            if f:
                file_selected_path["path"] = f
                lbl_file_path.config(text=os.path.basename(f), fg="blue")

        btn_browse = tk.Button(
            frame_options,
            text="📁 Buka File Explorer...",
            command=on_browse,
            font=("Segoe UI", 9)
        )
        btn_browse.pack(anchor="w", padx=20, pady=4)

        def on_ok():
            if var_choice.get() == "webcam":
                selected_result["val"] = str(var_cam_idx.get())
            else:
                if file_selected_path["path"]:
                    selected_result["val"] = file_selected_path["path"]
                else:
                    selected_result["val"] = default_source
            root.destroy()

        frame_btn = tk.Frame(root, pady=10)
        frame_btn.pack(side="bottom", fill="x", padx=15)

        btn_apply = tk.Button(
            frame_btn,
            text="Pilih Sumber Ini",
            bg="#007acc",
            fg="white",
            font=("Segoe UI", 9, "bold"),
            width=15,
            command=on_ok
        )
        btn_apply.pack(side="right", padx=5)

        btn_cancel = tk.Button(
            frame_btn,
            text="Batal",
            font=("Segoe UI", 9),
            width=10,
            command=root.destroy
        )
        btn_cancel.pack(side="right")

        root.mainloop()
        return selected_result["val"]
    except Exception as e:
        print(f"[WARN] Dialog error: {e}")
        return default_source


def select_dual_camera_setup_dialog(default_top, default_side, side_cam_id="CAM-02"):
    """
    Dialog visual terpadu untuk mengatur sumber CAM-01 Top dan CAM-02 Side secara simultan.
    Mendukung deteksi otomatis webcam USB dan file explorer video.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        cams = detect_available_cameras(max_devices=6)

        root = tk.Tk()
        root.title("Pengaturan Sumber Dual Kamera (CAM-01 Top & CAM-02 Side)")
        root.attributes("-topmost", True)
        root.geometry("540x510")
        root.resizable(False, False)

        res = {"top": default_top, "side": default_side}

        # Header
        lbl_t = tk.Label(root, text="🎥 PENGATURAN SUMBER DUAL KAMERA", font=("Segoe UI", 12, "bold"), pady=8)
        lbl_t.pack()
        lbl_sub = tk.Label(
            root,
            text=f"Webcam terdeteksi: {len(cams)} perangkat ({', '.join(['#'+str(c) for c in cams]) if cams else 'Tidak ada'})",
            font=("Segoe UI", 9, "italic"),
            fg="#444444"
        )
        lbl_sub.pack(pady=(0, 8))

        # --- PANEL CAM-01 TOP ---
        f_top = tk.LabelFrame(root, text="[CAM-01 TOP] Deteksi Plat Slat & Bilah Rib", font=("Segoe UI", 10, "bold"), padx=12, pady=6)
        f_top.pack(fill="x", padx=15, pady=5)

        is_top_cam = str(default_top).strip().isdigit() if default_top is not None else False
        var_top_choice = tk.StringVar(value="webcam" if (is_top_cam or (cams and not default_top)) else "file")
        var_top_cam = tk.IntVar(value=int(default_top) if is_top_cam else (cams[0] if cams else 0))
        top_file_path = {"path": str(default_top) if (default_top and not is_top_cam) else None}

        f_top_cams = tk.Frame(f_top)
        f_top_cams.pack(fill="x")
        if cams:
            tk.Radiobutton(f_top_cams, text="Webcam:", variable=var_top_choice, value="webcam", font=("Segoe UI", 9, "bold")).pack(side="left")
            for c_idx in cams:
                tk.Radiobutton(f_top_cams, text=f"Cam #{c_idx}", variable=var_top_cam, value=c_idx, command=lambda: var_top_choice.set("webcam"), font=("Segoe UI", 9)).pack(side="left", padx=6)
        else:
            tk.Label(f_top_cams, text="(Tidak ada webcam)", fg="gray").pack(side="left")

        f_top_file = tk.Frame(f_top, pady=4)
        f_top_file.pack(fill="x")
        tk.Radiobutton(f_top_file, text="File Video:", variable=var_top_choice, value="file", font=("Segoe UI", 9, "bold")).pack(side="left")
        lbl_top_fname = tk.Label(f_top_file, text=os.path.basename(str(top_file_path["path"])) if top_file_path["path"] else "(Default Video)", font=("Segoe UI", 8), fg="#0055aa")
        lbl_top_fname.pack(side="left", padx=6)

        def browse_top():
            var_top_choice.set("file")
            f = filedialog.askopenfilename(parent=root, title="Pilih Video CAM-01 Top")
            if f:
                top_file_path["path"] = f
                lbl_top_fname.config(text=os.path.basename(f), fg="blue")

        tk.Button(f_top_file, text="Pilih...", command=browse_top, font=("Segoe UI", 8)).pack(side="right")

        # --- PANEL CAM-02 SIDE ---
        f_side = tk.LabelFrame(root, text=f"[{side_cam_id} SIDE] Deteksi Baut, Pin & Bracket Lekukan", font=("Segoe UI", 10, "bold"), padx=12, pady=6)
        f_side.pack(fill="x", padx=15, pady=8)

        is_side_cam = str(default_side).strip().isdigit() if default_side is not None else False
        var_side_choice = tk.StringVar(value="webcam" if (is_side_cam or (len(cams) >= 2 and not default_side)) else "file")
        default_side_idx = int(default_side) if is_side_cam else (cams[1] if len(cams) >= 2 else (cams[0] if cams else 0))
        var_side_cam = tk.IntVar(value=default_side_idx)
        side_file_path = {"path": str(default_side) if (default_side and not is_side_cam) else None}

        f_side_cams = tk.Frame(f_side)
        f_side_cams.pack(fill="x")
        if cams:
            tk.Radiobutton(f_side_cams, text="Webcam:", variable=var_side_choice, value="webcam", font=("Segoe UI", 9, "bold")).pack(side="left")
            for c_idx in cams:
                tk.Radiobutton(f_side_cams, text=f"Cam #{c_idx}", variable=var_side_cam, value=c_idx, command=lambda: var_side_choice.set("webcam"), font=("Segoe UI", 9)).pack(side="left", padx=6)
        else:
            tk.Label(f_side_cams, text="(Tidak ada webcam)", fg="gray").pack(side="left")

        f_side_file = tk.Frame(f_side, pady=4)
        f_side_file.pack(fill="x")
        tk.Radiobutton(f_side_file, text="File Video:", variable=var_side_choice, value="file", font=("Segoe UI", 9, "bold")).pack(side="left")
        lbl_side_fname = tk.Label(f_side_file, text=os.path.basename(str(side_file_path["path"])) if side_file_path["path"] else "(Default Video)", font=("Segoe UI", 8), fg="#0055aa")
        lbl_side_fname.pack(side="left", padx=6)

        def browse_side():
            var_side_choice.set("file")
            f = filedialog.askopenfilename(parent=root, title=f"Pilih Video {side_cam_id} Side")
            if f:
                side_file_path["path"] = f
                lbl_side_fname.config(text=os.path.basename(f), fg="blue")

        tk.Button(f_side_file, text="Pilih...", command=browse_side, font=("Segoe UI", 8)).pack(side="right")

        # Tombol aksi
        def on_confirm_all():
            if var_top_choice.get() == "webcam":
                res["top"] = str(var_top_cam.get())
            elif top_file_path["path"]:
                res["top"] = top_file_path["path"]

            if var_side_choice.get() == "webcam":
                res["side"] = str(var_side_cam.get())
            elif side_file_path["path"]:
                res["side"] = side_file_path["path"]

            root.destroy()

        f_bottom = tk.Frame(root, pady=15)
        f_bottom.pack(side="bottom", fill="x", padx=15)

        tk.Button(f_bottom, text="Terapkan & Mulai", bg="#007acc", fg="white", font=("Segoe UI", 10, "bold"), width=18, command=on_confirm_all).pack(side="right", padx=5)
        tk.Button(f_bottom, text="Gunakan Bawaan", font=("Segoe UI", 9), width=15, command=root.destroy).pack(side="right")

        root.mainloop()
        return res["top"], res["side"]
    except Exception as e:
        print(f"[WARN] Dialog error: {e}")
        return default_top, default_side


# ======================================================================================
# WORKER 1: TOP INSPECTION PIPELINE (CAM-03 SLAT & RIB)
# ======================================================================================
def _run_top_worker(
    source_path,
    mode,
    rib_mode,
    device,
    enable_db,
    stop_event,
    fps_val,
    frames_val,
    latency_val,
    status_buf
):
    """Worker proses terisolasi untuk CAM-01 Top Inspection (Plat & Rib)."""
    # Bersihkan cache VRAM PyTorch sebelum proses inferensi dimulai
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    # Konfigurasi Anti-Throttling & Prioritas Realtime Windows
    configure_windows_process_performance()

    # 1. Pastikan path impor modul top inspector terdaftar di proses anak
    if TOP_SRC_DIR not in sys.path:
        sys.path.insert(0, TOP_SRC_DIR)

    import cv2
    import numpy as np
    from combined_top_inspector import CombinedTopInspector

    # 2. Inisialisasi Engine Top Inspector
    try:
        inspector = CombinedTopInspector(
            rib_mode=rib_mode,
            device=device,
            enable_db=enable_db
        )
    except Exception as e:
        print(f"[ERROR TOP WORKER] Gagal menginisialisasi CombinedTopInspector: {e}")
        status_buf.value = b"INIT_ERROR"
        return

    # 3. Inisialisasi Sumber Video / Kamera
    is_cam = str(source_path).isdigit()
    cap_src = int(source_path) if is_cam else str(source_path)
    cap = cv2.VideoCapture(cap_src)
    if not cap.isOpened():
        print(f"[ERROR TOP WORKER] Tidak dapat membuka sumber video CAM-01: {source_path}")
        status_buf.value = b"SRC_ERROR"
        return

    win_name = "CAM-01 TOP (SLAT & RIB) | [D]: Toggle DB | [1/V]: Ganti Video | [Q]: Keluar"
    if mode == "dual":
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 920, 518)
        cv2.moveWindow(win_name, 20, 40)

    # Variabel Benchmark FPS & Latency
    frame_count = 0
    fps_history = []
    t_start = time.time()
    t_prev = t_start

    print("[*] Worker CAM-01 Top Inspection AKTIF dan mulai memproses...")

    try:
        while not stop_event.is_set():
            t_frame_start = time.time()

            ret, frame = cap.read()
            if not ret or frame is None:
                # Jika file video habis, putar ulang (looping) dengan re-open agar FFmpeg tidak memory leak
                if not is_cam:
                    cap.release()
                    cap = cv2.VideoCapture(cap_src)
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        time.sleep(0.02)
                        continue
                else:
                    time.sleep(0.01)
                    continue

            # Jalankan inspeksi terpadu plat slat & rib
            inspect_res = inspector.inspect(frame)

            # Hitung latensi inferensi frame ini
            t_frame_end = time.time()
            latency_ms = (t_frame_end - t_frame_start) * 1000.0

            frame_count += 1
            dt = t_frame_end - t_prev
            t_prev = t_frame_end

            curr_fps = 1.0 / dt if dt > 0 else 30.0
            fps_history.append(curr_fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)

            # Update Shared Memory
            fps_val.value = avg_fps
            frames_val.value = frame_count
            latency_val.value = latency_ms
            curr_stat = inspect_res.get("status", "WAIT")
            status_buf.value = curr_stat.encode("utf-8")[:63]

            # Render Visual jika mode dual
            hud = None
            if mode == "dual":
                hud = inspector.render_combined_dashboard(
                    frame=frame,
                    inspect_result=inspect_res,
                    fps=avg_fps,
                    frame_idx=frame_count,
                    total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_cam else 0
                )
                cv2.imshow(win_name, hud)
                key = cv2.waitKey(1) & 0xFF
                if key in [ord("q"), ord("Q"), 27]:  # 27 = ESC
                    print("\n[*] Tombol keluar [Q] ditekan pada jendela CAM-01 Top.")
                    stop_event.set()
                    break
                elif key in [ord("d"), ord("D")]:
                    st = inspector.toggle_db()
                    mode_str = "AKTIF (RECORDING & NOTIFIKASI TEAMS)" if st else "MATI (DRY-RUN BENCHMARK)"
                    print(f"\n[*] [CAM-01 TOP] SAKLAR DATABASE & TEAMS: {mode_str}\n")
                elif key in [ord("r"), ord("R")]:
                    inspector.reset_counters()
                    print("\n[*] [CAM-01 TOP] Counter QC & Cache Deduplikasi di-reset ke 0.\n")
                elif key in [ord("1"), ord("v"), ord("V"), ord("t"), ord("T"), ord("o"), ord("O"), ord("f"), ord("F")]:
                    print("\n[*] [CAM-01 TOP] Membuka File Dialog Explorer...")
                    init_d = os.path.dirname(cap_src) if os.path.exists(str(cap_src)) else os.path.join(BASE_DIR, "Jetson_Vision_Engine", "Video_Testing")
                    new_top = select_file_dialog(
                        initial_dir=init_d,
                        title="[CAM-01 TOP] Pilih File Video Plat Slat & Bilah Rib"
                    )
                    cv2.waitKey(1)  # Refresh window focus
                    if new_top and os.path.exists(new_top):
                        cap.release()
                        cap_src = new_top
                        is_cam = False
                        cap = cv2.VideoCapture(cap_src)
                        frame_count = 0
                        fps_history.clear()
                        print(f"[*] [CAM-01 TOP] Video berhasil dialihkan ke: {os.path.basename(new_top)}")

            # Rekam ke database SQLite dan kirim notifikasi ke MS Teams menggunakan Best-Shot Buffer jika DB aktif
            if inspector and inspector.is_db_enabled():
                inspector.record_if_ng(
                    frame_raw=frame,
                    frame_ann=hud if hud is not None else frame,
                    inspect_res=inspect_res,
                    slat_no=getattr(inspector.slat_inspector, "slat_count", None),
                    rib_no=getattr(inspector.rib_inspector, "rib_count", None)
                )
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        cap.release()
        if mode == "dual":
            cv2.destroyWindow(win_name)
        inspector.close()
        print("[*] Worker CAM-01 Top Inspection selesai dan ditutup.")


# ======================================================================================
# WORKER 2: SIDE INSPECTION PIPELINE (CAM-01 BOLT, PIN & BRACKET)
# ======================================================================================
def _run_side_worker(
    source_path,
    mode,
    side,
    camera_id,
    device,
    enable_db,
    stop_event,
    fps_val,
    frames_val,
    latency_val,
    status_buf
):
    """Worker proses terisolasi untuk Side Inspection (Bolt, Pin, & Bracket)."""
    # Bersihkan cache VRAM PyTorch sebelum proses inferensi dimulai
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    # Konfigurasi Anti-Throttling & Prioritas Realtime Windows
    configure_windows_process_performance()

    # 1. Pastikan path impor modul side inspector dan Database terdaftar di proses anak
    if SIDE_SRC_DIR not in sys.path:
        sys.path.insert(0, SIDE_SRC_DIR)
    # Deteksi lokasi db_bridge (BASE_DIR = Jetson deployment, ../Database = PC original)
    _db_candidates = [
        BASE_DIR,
        os.path.abspath(os.path.join(BASE_DIR, "..", "Database")),
        os.path.abspath(os.path.join(BASE_DIR, "..", "Program AI")),
    ]
    db_dir = next(
        (d for d in _db_candidates if os.path.exists(os.path.join(d, "db_bridge.py"))),
        _db_candidates[1]
    )
    if db_dir not in sys.path:
        sys.path.insert(0, db_dir)

    import cv2
    import numpy as np
    from side_inspector_pipeline import SideConveyorInspector

    # Inisialisasi Database Bridge (SQLite, Captures, & Notifier Teams)
    db_bridge = None
    try:
        from db_bridge import InspectionDBBridge
        db_bridge = InspectionDBBridge(enabled=enable_db, camera_id=camera_id)
    except Exception as e_db:
        print(f"[WARN SIDE WORKER] Gagal menginisialisasi InspectionDBBridge: {e_db}")

    # 2. Inisialisasi Engine Side Inspector
    try:
        inspector = SideConveyorInspector(
            camera_id=camera_id,
            side=side,
            device=device
        )
    except Exception as e:
        print(f"[ERROR SIDE WORKER] Gagal menginisialisasi SideConveyorInspector: {e}")
        status_buf.value = b"INIT_ERROR"
        return

    # 3. Inisialisasi Sumber Video / Kamera
    is_cam = False
    cap = None
    side_video_path = None
    consecutive_cam_fails = 0

    if source_path is not None and os.path.exists(str(source_path)):
        side_video_path = str(source_path)
        cap = cv2.VideoCapture(side_video_path)
        is_cam = False
        print(f"[*] Worker {camera_id} Side Inspection memuat video: {os.path.basename(side_video_path)}")
    elif source_path is None or source_path == "" or str(source_path).isdigit():
        cam_idx = int(source_path) if str(source_path).isdigit() else 0
        # Gunakan DirectShow (CAP_DSHOW) di Windows untuk bypass bug MSMF Error -1072873822
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(cam_idx)
        is_cam = True

        # Jika webcam tidak bisa dibuka, langsung alihkan ke file video testing
        if not cap.isOpened():
            cap.release()
            cap = None
            if DEFAULT_SIDE_VIDEO and os.path.exists(DEFAULT_SIDE_VIDEO):
                side_video_path = DEFAULT_SIDE_VIDEO
                cap = cv2.VideoCapture(side_video_path)
                is_cam = False
                print(f"[INFO SIDE WORKER] Webcam #{cam_idx} tidak terdeteksi. Beralih otomatis ke video testing: {os.path.basename(DEFAULT_SIDE_VIDEO)}")

    win_name = f"{camera_id} SIDE [{camera_id} {side}] | [D]: Toggle DB | [2/O]: Ganti Video | [Q]: Keluar"
    if mode == "dual":
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 920, 518)
        cv2.moveWindow(win_name, 960, 40)

    # Variabel Benchmark FPS & Latency
    frame_count = 0
    fps_history = []
    t_start = time.time()
    t_prev = t_start

    print(f"[*] Worker {camera_id} Side Inspection [{side}] AKTIF dan mulai memproses...")

    try:
        while not stop_event.is_set():
            t_frame_start = time.time()

            if cap is not None and cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    if not is_cam:
                        # Re-open capture untuk continuous loop tanpa memory leak
                        cap.release()
                        target_v = side_video_path or DEFAULT_SIDE_VIDEO
                        cap = cv2.VideoCapture(target_v)
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            time.sleep(0.02)
                            continue
                    else:
                        consecutive_cam_fails += 1
                        if consecutive_cam_fails >= 5:
                            print(f"[WARN SIDE WORKER] Kamera live tidak mengirim frame. Beralih ke video testing...")
                            cap.release()
                            if DEFAULT_SIDE_VIDEO and os.path.exists(DEFAULT_SIDE_VIDEO):
                                side_video_path = DEFAULT_SIDE_VIDEO
                                cap = cv2.VideoCapture(side_video_path)
                                is_cam = False
                            else:
                                cap = None
                        time.sleep(0.02)
                        continue
                else:
                    consecutive_cam_fails = 0
            else:
                # Frame sintetis berdimensi 1280x720 untuk pengujian beban murni jika tidak ada webcam
                frame = np.full((720, 1280, 3), 45, dtype=np.uint8)
                offset = (frame_count * 8) % 1280
                cv2.rectangle(frame, (offset, 250), (offset + 180, 450), (120, 120, 130), -1)
                cv2.circle(frame, (offset + 90, 350), 22, (200, 200, 210), -1)
                time.sleep(0.015)

            # Jalankan inspeksi komponen samping (YOLO Bolt/Pin + PatchCore Bracket)
            display_frame, payload = inspector.process_frame(frame)

            t_frame_end = time.time()
            latency_ms = (t_frame_end - t_frame_start) * 1000.0

            frame_count += 1
            dt = t_frame_end - t_prev
            t_prev = t_frame_end

            curr_fps = 1.0 / dt if dt > 0 else 30.0
            fps_history.append(curr_fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)

            # Update Shared Memory
            fps_val.value = avg_fps
            frames_val.value = frame_count
            latency_val.value = latency_ms
            curr_stat = payload.get("overall_result", "WAIT")
            status_buf.value = str(curr_stat).encode("utf-8")[:63]

            # Rekam ke database SQLite, folder captures, dan kirim notifikasi ke MS Teams jika ada Side NG (Single-Capture per Slat)
            if db_bridge and db_bridge.is_enabled():
                if payload.get("should_emit", False) and payload.get("overall_result") == "NG":
                    emit_raw = payload.get("emit_raw_frame", frame)
                    emit_ann = payload.get("emit_ann_frame", display_frame)
                    db_bridge.emit_side_inspection(
                        frame_raw=emit_raw,
                        frame_ann=emit_ann,
                        side_payload=payload
                    )

            # Render Visual jika mode dual
            if mode == "dual":
                # Tambahkan overlay benchmark banner di bagian bawah frame
                h_d, w_d = display_frame.shape[:2]
                db_state = "ON" if (db_bridge and db_bridge.is_enabled()) else "OFF"
                banner = f"FPS: {avg_fps:4.1f} | Lat: {latency_ms:4.1f}ms | DB:{db_state} | [D]:Toggle DB | [2/O]:Ganti | [Q]:Keluar"
                cv2.rectangle(display_frame, (10, h_d - 40), (620, h_d - 10), (20, 20, 20), -1)
                cv2.putText(display_frame, banner, (18, h_d - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1, cv2.LINE_AA)

                cv2.imshow(win_name, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in [ord("q"), ord("Q"), 27]:
                    print(f"\n[*] Tombol keluar [Q] ditekan pada jendela {camera_id} Side.")
                    stop_event.set()
                    break
                elif key in [ord("d"), ord("D")]:
                    if db_bridge is not None:
                        st = db_bridge.toggle()
                        status_str = "AKTIF (RECORDING & NOTIFIKASI TEAMS)" if st else "MATI (DRY-RUN BENCHMARK)"
                        print(f"\n[*] [{camera_id} SIDE] SAKLAR DATABASE & TEAMS: {status_str}\n")
                elif key in [ord("2"), ord("o"), ord("O"), ord("f"), ord("F"), ord("s"), ord("S")]:
                    print(f"\n[*] [{camera_id} SIDE] Membuka File Dialog Explorer...")
                    candidate_side_dir = os.path.dirname(side_video_path) if side_video_path and os.path.exists(side_video_path) else os.path.abspath(os.path.join(BASE_DIR, "..", "Documentation"))
                    new_side = select_file_dialog(
                        initial_dir=candidate_side_dir,
                        title=f"[{camera_id} SIDE] Pilih File Video Baut, Pin & Bracket"
                    )
                    cv2.waitKey(1)  # Refresh window focus
                    if new_side and os.path.exists(new_side):
                        if cap is not None and cap.isOpened():
                            cap.release()
                        side_video_path = new_side
                        is_cam = False
                        cap = cv2.VideoCapture(side_video_path)
                        frame_count = 0
                        consecutive_cam_fails = 0
                        fps_history.clear()
                        print(f"[*] [CAM-01 SIDE] Video berhasil dialihkan ke: {os.path.basename(new_side)}")
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        if cap is not None and cap.isOpened():
            cap.release()
        if db_bridge is not None:
            db_bridge.close()
        if mode == "dual":
            cv2.destroyWindow(win_name)
        print("[*] Worker CAM-01 Side Inspection selesai dan ditutup.")


# ======================================================================================
# THREADED BENCHMARK ENGINE (SINGLE-PROCESS, LIGHTWEIGHT FOR 8GB RAM & SINGLE GPU)
# ======================================================================================
def _run_threaded_engine(args):
    """
    Runner Terpadu Berbasis Multi-Threading (Single-Process).
    Keunggulan:
    - Hemat memori: Hanya 1 instance Python dan 1 konteks CUDA (~500 MB RAM total vs ~1.3 GB pada multiprocessing).
    - Berbagi CUDA memory pool tanpa kompetisi OOM pada GPU 4GB (RTX 2050).
    - GUI OpenCV berjalan di thread utama sehingga responsif, bebas glitch, dan sinkron.
    - Saklar [D] secara instan mengaktifkan DB & notifikasi Teams pada kedua kamera sekaligus!
    """
    import cv2
    import numpy as np
    import threading

    # 1. Pastikan path impor modul terdaftar
    if TOP_SRC_DIR not in sys.path:
        sys.path.insert(0, TOP_SRC_DIR)
    if SIDE_SRC_DIR not in sys.path:
        sys.path.insert(0, SIDE_SRC_DIR)
    # Deteksi lokasi db_bridge (BASE_DIR = Jetson deployment, ../Database = PC original)
    _db_candidates = [
        BASE_DIR,
        os.path.abspath(os.path.join(BASE_DIR, "..", "Database")),
        os.path.abspath(os.path.join(BASE_DIR, "..", "Program AI")),
    ]
    db_dir = next(
        (d for d in _db_candidates if os.path.exists(os.path.join(d, "db_bridge.py"))),
        _db_candidates[1]
    )
    if db_dir not in sys.path:
        sys.path.insert(0, db_dir)

    from combined_top_inspector import CombinedTopInspector
    from side_inspector_pipeline import SideConveyorInspector
    try:
        from db_bridge import InspectionDBBridge
    except ImportError:
        InspectionDBBridge = None

    # Konfigurasi Anti-Throttling Windows & Prioritas Thread Realtime
    configure_windows_process_performance()

    # Bersihkan cache PyTorch & Batasi alokasi VRAM 85% untuk cegah Shared GPU Memory Paging saat browser aktif
    try:
        import torch
        if torch.cuda.is_available():
            try:
                torch.cuda.set_per_process_memory_fraction(0.85, device=0)
            except Exception:
                pass
            torch.cuda.empty_cache()
    except Exception:
        pass

    # 2. Inisialisasi Top Inspector jika aktif
    top_inspector = None
    cap_top = None
    is_cam_top = False
    top_source_path = args.top_source

    if args.mode in ["dual", "headless", "top-only"]:
        print("[*] Menginisialisasi Top Inspector (CAM-01 Slat & Rib)...")
        try:
            top_inspector = CombinedTopInspector(
                rib_mode=args.rib_mode,
                device=args.device,
                enable_db=args.enable_db
            )
        except Exception as e_top:
            print(f"[ERROR] Gagal inisialisasi CombinedTopInspector: {e_top}")
            return

        cap_top, is_cam_top, top_source_path = create_capture(top_source_path)
        if not cap_top.isOpened():
            print(f"[!] Gagal membuka video CAM-01: {top_source_path}")
            if DEFAULT_TOP_VIDEO and os.path.exists(DEFAULT_TOP_VIDEO):
                top_source_path = DEFAULT_TOP_VIDEO
                cap_top, is_cam_top, top_source_path = create_capture(top_source_path)

    # 3. Inisialisasi Side Inspector & DB Bridge jika aktif
    side_inspector = None
    side_db_bridge = None
    cap_side = None
    is_cam_side = False
    side_source_path = args.side_source

    if args.mode in ["dual", "headless", "side-only"]:
        print(f"[*] Menginisialisasi Side Inspector ({args.side_cam_id} {args.side_name})...")
        try:
            side_inspector = SideConveyorInspector(
                camera_id=args.side_cam_id,
                side=args.side_name,
                device=args.device
            )
        except Exception as e_side:
            print(f"[ERROR] Gagal inisialisasi SideConveyorInspector: {e_side}")
            return

        if InspectionDBBridge is not None:
            try:
                side_db_bridge = InspectionDBBridge(
                    enabled=args.enable_db,
                    camera_id=args.side_cam_id
                )
            except Exception as e_db:
                print(f"[WARN] Gagal inisialisasi Side DB Bridge: {e_db}")

        cap_side, is_cam_side, side_source_path = create_capture(side_source_path)
        if not cap_side.isOpened():
            if DEFAULT_SIDE_VIDEO and os.path.exists(DEFAULT_SIDE_VIDEO):
                side_source_path = DEFAULT_SIDE_VIDEO
                cap_side, is_cam_side, side_source_path = create_capture(side_source_path)
                print(f"[*] Worker {args.side_cam_id} Side memuat video default: {os.path.basename(side_source_path)}")
            else:
                print(f"[!] Gagal membuka sumber video {args.side_cam_id} Side: {side_source_path}")

    # Shared thread states & locks
    stop_event = threading.Event()
    top_lock = threading.Lock()
    side_lock = threading.Lock()

    # Inisialisasi Dedicated CUDA Streams & cuDNN Benchmark untuk mencegah GPU serialization
    top_stream = None
    side_stream = None
    try:
        import torch
        if torch.cuda.is_available() and args.device != "cpu":
            torch.backends.cudnn.benchmark = True
            top_stream = torch.cuda.Stream()
            side_stream = torch.cuda.Stream()
    except Exception:
        pass

    # Pre-Warmup pass untuk mengeliminasi latensi inisialisasi CUDA 5.8s pada frame 0
    print("[*] Melakukan pre-warmup GPU AI Engine (mengeliminasi lonjakan latensi frame pertama)...")
    try:
        if top_inspector is not None and cap_top is not None and cap_top.isOpened():
            ret_w, frame_w = cap_top.read()
            if ret_w and frame_w is not None:
                top_inspector.inspect(frame_w)
                if not is_cam_top:
                    cap_top.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if side_inspector is not None and cap_side is not None and cap_side.isOpened():
            ret_w, frame_w = cap_side.read()
            if ret_w and frame_w is not None:
                side_inspector.process_frame(frame_w)
                if not is_cam_side:
                    cap_side.set(cv2.CAP_PROP_POS_FRAMES, 0)
        print("[*] Pre-warmup selesai. GPU siap pada performa steady-state.")
    except Exception as e_wm:
        print(f"[WARN] Pre-warmup skipped: {e_wm}")

    top_state = {
        "display": None,
        "fps": 0.0,
        "latency": 0.0,
        "status": "STARTING",
        "frames": 0,
    }
    side_state = {
        "display": None,
        "fps": 0.0,
        "latency": 0.0,
        "status": "STARTING",
        "frames": 0,
    }

    # Worker Thread 1: CAM-01 Top Inspector
    def _top_loop():
        nonlocal cap_top, top_source_path, is_cam_top
        t_prev = time.time()
        fps_hist = []
        total_f = int(cap_top.get(cv2.CAP_PROP_FRAME_COUNT)) if (cap_top and not is_cam_top) else 0

        try:
            while not stop_event.is_set():
                if cap_top is None or not cap_top.isOpened():
                    time.sleep(0.02)
                    continue

                t_start = time.time()
                ret, frame = cap_top.read()
                if not ret or frame is None:
                    if not is_cam_top:
                        cap_top.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap_top.read()
                        if not ret or frame is None:
                            time.sleep(0.02)
                            continue
                    else:
                        time.sleep(0.01)
                        continue

                if top_stream is not None:
                    with torch.cuda.stream(top_stream):
                        with torch.inference_mode():
                            inspect_res = top_inspector.inspect(frame)
                else:
                    inspect_res = top_inspector.inspect(frame)
                t_end = time.time()
                lat = (t_end - t_start) * 1000.0

                dt = t_end - t_prev
                t_prev = t_end
                c_fps = 1.0 / dt if dt > 0 else 30.0
                fps_hist.append(c_fps)
                if len(fps_hist) > 30:
                    fps_hist.pop(0)
                avg_f = sum(fps_hist) / len(fps_hist)

                hud = None
                if args.mode == "dual":
                    hud = top_inspector.render_combined_dashboard(
                        frame=frame,
                        inspect_result=inspect_res,
                        fps=avg_f,
                        frame_idx=top_state["frames"] + 1,
                        total_frames=total_f
                    )

                # Rekam ke database SQLite dan kirim notifikasi MS Teams menggunakan Best-Shot Buffer jika DB aktif
                if top_inspector and top_inspector.is_db_enabled():
                    top_inspector.record_if_ng(
                        frame_raw=frame,
                        frame_ann=hud if hud is not None else frame,
                        inspect_res=inspect_res,
                        slat_no=getattr(top_inspector.slat_inspector, "slat_count", None),
                        rib_no=getattr(top_inspector.rib_inspector, "rib_count", None)
                    )

                with top_lock:
                    top_state["frames"] += 1
                    top_state["fps"] = avg_f
                    top_state["latency"] = lat
                    top_state["status"] = inspect_res.get("status", "WAIT")
                    top_state["display"] = hud
        except Exception as e:
            import traceback
            print(f"[ERROR Thread-TopInspector] {e}")
            traceback.print_exc()
            with top_lock:
                top_state["status"] = "ERROR"

    # Worker Thread 2: Side Inspector
    def _side_loop():
        nonlocal cap_side, side_source_path, is_cam_side
        t_prev = time.time()
        fps_hist = []

        try:
            while not stop_event.is_set():
                if cap_side is None or not cap_side.isOpened():
                    time.sleep(0.02)
                    continue

                t_start = time.time()
                ret, frame = cap_side.read()
                if not ret or frame is None:
                    if not is_cam_side:
                        cap_side.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        side_inspector.reset_session()
                        ret, frame = cap_side.read()
                        if not ret or frame is None:
                            time.sleep(0.02)
                            continue
                    else:
                        time.sleep(0.01)
                        continue

                if side_stream is not None:
                    with torch.cuda.stream(side_stream):
                        with torch.inference_mode():
                            display_frame, payload = side_inspector.process_frame(frame)
                else:
                    display_frame, payload = side_inspector.process_frame(frame)
                t_end = time.time()
                lat = (t_end - t_start) * 1000.0

                dt = t_end - t_prev
                t_prev = t_end
                c_fps = 1.0 / dt if dt > 0 else 30.0
                fps_hist.append(c_fps)
                if len(fps_hist) > 30:
                    fps_hist.pop(0)
                avg_f = sum(fps_hist) / len(fps_hist)

                # Rekam ke database SQLite & notifikasi Teams jika terdeteksi Side NG (Single-Capture per Slat)
                if side_db_bridge and side_db_bridge.is_enabled():
                    if payload.get("should_emit", False) and payload.get("overall_result") == "NG":
                        emit_raw = payload.get("emit_raw_frame", frame)
                        emit_ann = payload.get("emit_ann_frame", display_frame)
                        side_db_bridge.emit_side_inspection(
                            frame_raw=emit_raw,
                            frame_ann=emit_ann,
                            side_payload=payload
                        )

                # Gambar overlay banner benchmark di bagian bawah frame Side (DIBESARKAN)
                if args.mode in ["dual", "side-only"] and display_frame is not None:
                    h_d, w_d = display_frame.shape[:2]
                    db_st = "ON" if (side_db_bridge and side_db_bridge.is_enabled()) else "OFF"
                    s_ok = getattr(side_inspector, 'slat_ok_count', getattr(side_inspector, 'stats', {}).get('ok_count', 0))
                    s_ng = getattr(side_inspector, 'slat_ng_count', getattr(side_inspector, 'stats', {}).get('ng_count', 0))
                    b_ng = getattr(side_inspector, 'bolt_ng_count', 0)
                    p_ng = getattr(side_inspector, 'pin_ng_count', 0)
                    c_ng = getattr(side_inspector, 'crack_ng_count', 0)
                    s_cyc = getattr(side_inspector, 'slat_cycle_count', 0)
                    qc_txt = f"Slat #{s_cyc} [OK:{s_ok} NG:{s_ng}]"
                    banner = f"FPS: {avg_f:4.1f} | Lat: {lat:4.1f}ms | [1/V]: Video Top | [2/O]: Video Side | {qc_txt} | [Q]: Keluar"
                    cv2.rectangle(display_frame, (10, h_d - 65), (w_d - 10, h_d - 10), (18, 18, 18), -1)
                    cv2.rectangle(display_frame, (10, h_d - 65), (w_d - 10, h_d - 10), (0, 220, 255), 2, cv2.LINE_AA)
                    cv2.putText(display_frame, banner, (25, h_d - 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.76, (0, 255, 255), 2, cv2.LINE_AA)

                with side_lock:
                    side_state["frames"] += 1
                    side_state["fps"] = avg_f
                    side_state["latency"] = lat
                    side_state["status"] = payload.get("overall_result", "WAIT")
                    side_state["display"] = display_frame
        except Exception as e:
            import traceback
            print(f"[ERROR Thread-SideInspector] {e}")
            traceback.print_exc()
            with side_lock:
                side_state["status"] = "ERROR"

    # Start Worker Threads
    threads = []
    if top_inspector is not None:
        th_top = threading.Thread(target=_top_loop, daemon=True, name="Thread-TopInspector")
        th_top.start()
        threads.append(th_top)
    if side_inspector is not None:
        th_side = threading.Thread(target=_side_loop, daemon=True, name="Thread-SideInspector")
        th_side.start()
        threads.append(th_side)

    # Windows GUI Setup
    win_top = "CAM-01 TOP (SLAT & RIB) | [1]/[V]: Ganti Video TOP | [W]: Dual Setup | [D]: DB | [Q]: Keluar"
    win_side = f"{args.side_cam_id} SIDE [{args.side_cam_id} {args.side_name}] | [2]/[O]: Ganti Video SIDE | [W]: Dual Setup | [D]: DB | [Q]: Keluar"
    if args.mode == "dual":
        if top_inspector is not None:
            cv2.namedWindow(win_top, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win_top, 920, 518)
            cv2.moveWindow(win_top, 20, 40)
        if side_inspector is not None:
            cv2.namedWindow(win_side, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win_side, 920, 518)
            cv2.moveWindow(win_side, 960, 40)

    # Header Telemetri Terminal
    print("-" * 86)
    print(f"{'WAKTU':^8} | {'CAM-01 TOP (FPS)':^18} | {f'{args.side_cam_id} SIDE (FPS)':^18} | {'COMBINED FPS':^14} | {'STATUS':^16}")
    print("-" * 86)

    # Setup CSV jika diaktifkan
    csv_file = None
    if args.record_csv:
        try:
            csv_file = open(args.record_csv, "w", encoding="utf-8")
            csv_file.write("Timestamp,Elapsed_Sec,Top_FPS,Top_Latency_ms,Top_Frames,Side_FPS,Side_Latency_ms,Side_Frames,Combined_Throughput_FPS\n")
        except Exception as e_csv:
            print(f"[!] Gagal membuat file CSV: {e_csv}")
            csv_file = None

    t0 = None
    last_print = 0.0
    last_disp_top_cnt = -1
    last_disp_side_cnt = -1

    try:
        while not stop_event.is_set():
            now = time.time()
            if t0 is None:
                if (top_state["frames"] > 0) or (side_state["frames"] > 0):
                    t0 = now
                else:
                    time.sleep(0.02)
                    continue
            elapsed = now - t0

            if args.duration > 0 and elapsed >= args.duration:
                print(f"\n[*] Durasi benchmark {args.duration}s tercapai. Menghentikan...")
                stop_event.set()
                break

            # Render GUI OpenCV di Main Thread (Hanya render saat frame baru tersedia dari worker thread)
            key = -1
            if args.mode == "dual":
                has_rendered = False
                if top_inspector is not None:
                    with top_lock:
                        disp_top = top_state["display"]
                        cnt_top = top_state["frames"]
                    if disp_top is not None and cnt_top != last_disp_top_cnt:
                        last_disp_top_cnt = cnt_top
                        cv2.imshow(win_top, disp_top)
                        has_rendered = True

                if side_inspector is not None:
                    with side_lock:
                        disp_side = side_state["display"]
                        cnt_side = side_state["frames"]
                    if disp_side is not None and cnt_side != last_disp_side_cnt:
                        last_disp_side_cnt = cnt_side
                        cv2.imshow(win_side, disp_side)
                        has_rendered = True

                if has_rendered:
                    key = cv2.waitKey(1) & 0xFF
                else:
                    # Tetap pompa Win32 window message queue agar jendela tidak dianggap hang / di-throttle oleh DWM
                    key = cv2.waitKey(1) & 0xFF
                    time.sleep(0.001)
            else:
                time.sleep(0.005)

            # Tangani Tombol Keyboard Pintas
            if key in [ord("q"), ord("Q"), 27]:
                print("\n[*] Tombol keluar [Q] ditekan.")
                stop_event.set()
                break
            elif key in [ord("d"), ord("D")]:
                top_st = top_inspector.toggle_db() if top_inspector else False
                side_st = side_db_bridge.toggle() if side_db_bridge else False
                st_text = "AKTIF (RECORDING & NOTIFIKASI TEAMS)" if (top_st or side_st) else "MATI (DRY-RUN BENCHMARK)"
                print(f"\n[DB SWITCH] Status Database Logging beralih ke: {st_text}")
                print(f"[*] [CAM-01 TOP] SAKLAR DATABASE & TEAMS: {st_text}")
                print(f"[*] [{args.side_cam_id} SIDE] SAKLAR DATABASE & TEAMS: {st_text}\n")
            elif key in [ord("r"), ord("R")]:
                if top_inspector:
                    top_inspector.reset_counters()
                if side_inspector:
                    side_inspector.reset_session()
                print("\n[*] Counter QC & Cache Deduplikasi kedua kamera di-reset ke 0.\n")
            elif key in [ord("1"), ord("v"), ord("V"), ord("t"), ord("T")]:
                print("\n[*] [CAM-01 TOP] Membuka File Explorer untuk Memilih Video Baru Plat & Rib...")
                init_top = os.path.dirname(str(top_source_path)) if (top_source_path and os.path.exists(str(top_source_path))) else VIDEO_DIR
                new_top = select_file_dialog(
                    initial_dir=init_top,
                    title="[CAM-01 TOP] Pilih File Video Plat Slat & Bilah Rib"
                )
                if new_top and os.path.exists(new_top) and str(new_top) != str(top_source_path):
                    with top_lock:
                        if cap_top is not None:
                            cap_top.release()
                        cap_top, is_cam_top, top_source_path = create_capture(new_top)
                    if top_inspector:
                        top_inspector.reset_counters()
                    print(f"[*] CAM-01 Top berhasil beralih ke video: {os.path.basename(str(top_source_path))}")
            elif key in [ord("2"), ord("o"), ord("O"), ord("f"), ord("F"), ord("s"), ord("S")]:
                print(f"\n[*] [{args.side_cam_id} SIDE] Membuka File Explorer untuk Memilih Video Baru Baut & Pin...")
                init_side = os.path.dirname(str(side_source_path)) if (side_source_path and os.path.exists(str(side_source_path))) else VIDEO_DIR
                new_side = select_file_dialog(
                    initial_dir=init_side,
                    title=f"[{args.side_cam_id} SIDE] Pilih File Video Baut, Pin & Bracket"
                )
                if new_side and os.path.exists(new_side) and str(new_side) != str(side_source_path):
                    with side_lock:
                        if cap_side is not None:
                            cap_side.release()
                        cap_side, is_cam_side, side_source_path = create_capture(new_side)
                    if side_inspector:
                        side_inspector.reset_session()
                    print(f"[*] {args.side_cam_id} Side berhasil beralih ke video: {os.path.basename(str(side_source_path))}")
            elif key in [ord("w"), ord("W"), ord("c"), ord("C")]:
                print("\n[*] Membuka Dialog Pengaturan Sumber Dual Kamera (Webcam / File Video)...")
                new_top, new_side = select_dual_camera_setup_dialog(
                    default_top=top_source_path,
                    default_side=side_source_path,
                    side_cam_id=args.side_cam_id
                )
                if new_top and str(new_top) != str(top_source_path):
                    with top_lock:
                        if cap_top is not None:
                            cap_top.release()
                        cap_top, is_cam_top, top_source_path = create_capture(new_top)
                    if top_inspector:
                        top_inspector.reset_counters()
                    s_label = f"Webcam #{top_source_path}" if is_cam_top else os.path.basename(str(top_source_path))
                    print(f"[*] CAM-01 Top beralih ke: {s_label}")

                if new_side and str(new_side) != str(side_source_path):
                    with side_lock:
                        if cap_side is not None:
                            cap_side.release()
                        cap_side, is_cam_side, side_source_path = create_capture(new_side)
                    if side_inspector:
                        side_inspector.reset_session()
                    s_label = f"Webcam #{side_source_path}" if is_cam_side else os.path.basename(str(side_source_path))
                    print(f"[*] {args.side_cam_id} Side beralih ke: {s_label}")

            # Cetak Telemetri Terminal Setiap 1 Detik
            if now - last_print >= 1.0:
                last_print = now
                with top_lock:
                    v_top_fps = top_state["fps"]
                    v_top_lat = top_state["latency"]
                    f_top_cnt = top_state["frames"]
                    s_top = top_state["status"]
                with side_lock:
                    v_side_fps = side_state["fps"]
                    v_side_lat = side_state["latency"]
                    f_side_cnt = side_state["frames"]
                    s_side = side_state["status"]

                v_comb_fps = v_top_fps + v_side_fps
                if side_inspector is not None:
                    s_ok = getattr(side_inspector, 'slat_ok_count', getattr(side_inspector, 'stats', {}).get('ok_count', 0))
                    s_ng = getattr(side_inspector, 'slat_ng_count', getattr(side_inspector, 'stats', {}).get('ng_count', 0))
                    stat_summary = f"T:{s_top}|S:{s_side}[OK:{s_ok}/NG:{s_ng}]"
                else:
                    stat_summary = f"T:{s_top}|S:{s_side}"
                time_str = f"{int(elapsed):03d}s"

                print(
                    f"{time_str:^8} | "
                    f"{v_top_fps:5.1f} FPS ({v_top_lat:4.1f}ms) | "
                    f"{v_side_fps:5.1f} FPS ({v_side_lat:4.1f}ms) | "
                    f"{v_comb_fps:6.1f} FPS   | "
                    f"{stat_summary:^16}"
                )

                if csv_file:
                    ts_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    csv_file.write(
                        f"{ts_now},{elapsed:.1f},{v_top_fps:.2f},{v_top_lat:.1f},{f_top_cnt},"
                        f"{v_side_fps:.2f},{v_side_lat:.1f},{f_side_cnt},{v_comb_fps:.2f}\n"
                    )
                    csv_file.flush()

    except KeyboardInterrupt:
        print("\n[*] Interupsi Keyboard (Ctrl+C). Menghentikan sistem...")
        stop_event.set()

    finally:
        stop_event.set()
        for th in threads:
            th.join(timeout=2.0)

        if cap_top is not None:
            cap_top.release()
        if cap_side is not None:
            cap_side.release()
        if top_inspector and hasattr(top_inspector, "db_bridge") and top_inspector.db_bridge:
            top_inspector.db_bridge.close()
        if side_db_bridge is not None:
            side_db_bridge.close()
        if args.mode == "dual":
            cv2.destroyAllWindows()
        if csv_file is not None:
            csv_file.close()

    # Executive Summary Report
    total_elapsed = time.time() - t0
    f_top = top_state["frames"]
    f_side = side_state["frames"]
    avg_top_fps = f_top / total_elapsed if total_elapsed > 0 else 0.0
    avg_side_fps = f_side / total_elapsed if total_elapsed > 0 else 0.0

    print("\n" + "=" * 86)
    print("📊 LAPORAN AKHIR HASIL BENCHMARK MULTI-KAMERA")
    print("=" * 86)
    print(f"  • Total Waktu Pengujian     : {total_elapsed:.1f} detik")
    if args.mode in ["dual", "headless", "top-only"]:
        print(f"  • CAM-01 Top (Slat & Rib)   : {f_top} frames ({avg_top_fps:.1f} rata-rata FPS)")
    if args.mode in ["dual", "headless", "side-only"]:
        print(f"  • {args.side_cam_id} Side (Bolt & Pin)  : {f_side} frames ({avg_side_fps:.1f} rata-rata FPS)")
    if side_inspector is not None:
        s_ok = getattr(side_inspector, 'slat_ok_count', getattr(side_inspector, 'stats', {}).get('ok_count', 0))
        s_ng = getattr(side_inspector, 'slat_ng_count', getattr(side_inspector, 'stats', {}).get('ng_count', 0))
        b_ng = getattr(side_inspector, 'bolt_ng_count', 0)
        p_ng = getattr(side_inspector, 'pin_ng_count', 0)
        c_ng = getattr(side_inspector, 'crack_ng_count', 0)
        s_cyc = getattr(side_inspector, 'slat_cycle_count', 0)
        print(f"  • {args.side_cam_id} QC Fisik Slat     : {s_cyc} Total Slat Melintas | OK: {s_ok} | NG: {s_ng}")
        if s_ng > 0:
            print(f"    - Rincian Defek Sisi      : Baut NG: {b_ng} | Pin NG: {p_ng} | Crack/Lekukan NG: {c_ng}")
    print(f"  • Throughput Gabungan       : {(avg_top_fps + avg_side_fps):.1f} frames/detik")

    target_fps = 20.0
    if avg_top_fps >= target_fps and (args.mode == "top-only" or avg_side_fps >= target_fps):
        verdict = "🟢 SANGAT STABIL (Memenuhi standar real-time conveyor >= 20 FPS)"
    elif avg_top_fps >= 15.0 or (avg_top_fps + avg_side_fps) >= 20.0:
        verdict = "🟡 CUKUP (15 - 20 FPS, disarankan konversi model ke TensorRT FP16 di Jetson)"
    else:
        verdict = "🔴 BUTUH OPTIMASI (< 15 FPS, gunakan mode headless & akselerasi TensorRT)"

    print(f"  • Penilaian Stabilitas Device: {verdict}")
    print("=" * 86)
    print("[*] Selesai.\n")


# ======================================================================================
# MASTER ORCHESTRATOR & BENCHMARK SUPERVISOR
# ======================================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Master Multi-Camera Benchmark Runner (Top Slat/Rib + Side Bolt/Pin)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh Penggunaan:
  1. Pengujian Dual GUI (Standar dengan Video Testing):
     python run_multi_camera_benchmark.py --mode dual

  2. Pilih File Video Interaktif (File Explorer Dialog):
     python run_multi_camera_benchmark.py --browse
     python run_multi_camera_benchmark.py --select-top
     python run_multi_camera_benchmark.py --select-side

  3. Pengujian Headless (Maksimal untuk Jetson Orin Nano / Web-Ready):
     python run_multi_camera_benchmark.py --mode headless --duration 60 --record-csv benchmark.csv

  4. Menentukan file video khusus secara manual:
     python run_multi_camera_benchmark.py --top-source "path/to/top.mp4" --side-source 0

Shortcut Keyboard Saat Berjalan:
  [Q] / [ESC]       : Keluar dari benchmark & tampilkan laporan ringkasan
  [1] / [V] / [T]   : Buka File Explorer untuk mengganti video CAM-01 TOP (Plat Slat & Bilah Rib)
  [2] / [O] / [F]   : Buka File Explorer untuk mengganti video CAM-02 SIDE (Baut, Pin & Bracket)
  [W] / [C]         : Buka dialog visual pengaturan sumber dual kamera (Webcam / File Video)
  [D]               : Saklar Database Logging & Notifikasi Teams (ON / OFF)
  [R]               : Reset counter QC & cache deduplikasi ke 0
        """
    )
    env_mode = os.environ.get("BENCHMARK_MODE", "dual")
    env_side_name = os.environ.get("SIDE_CAMERA_NAME", "RIGHT")
    env_side_cam_id = os.environ.get("SIDE_CAMERA_ID", None)
    env_rib_mode = os.environ.get("RIB_DETECTION_MODE", "HYBRID")
    env_device = os.environ.get("INFERENCE_DEVICE", None)
    try:
        env_duration = int(os.environ.get("BENCHMARK_DURATION", "0"))
    except ValueError:
        env_duration = 0
    env_csv = os.environ.get("RECORD_CSV", None)
    env_db = os.environ.get("ENABLE_DB_LOGGING", "false").lower() in ("true", "1", "yes")
    env_engine = os.environ.get("BENCHMARK_ENGINE", "threads")

    parser.add_argument(
        "--mode",
        choices=["dual", "headless", "top-only", "side-only"],
        default=env_mode,
        help="Mode eksekusi: 'dual' (2 jendela OpenCV berdampingan), 'headless' (tanpa display, murni komputasi AI), atau pengujian kamera tunggal."
    )
    parser.add_argument(
        "--browse",
        action="store_true",
        help="Buka dialog file explorer untuk memilih video CAM-01 Top & CAM-02 Side sebelum memulai benchmark."
    )
    parser.add_argument(
        "--webcam", "--live",
        action="store_true",
        help="Mode Live Webcam: Otomatis mendeteksi dan menggunakan 2 webcam USB untuk CAM-01 Top dan CAM-02 Side."
    )
    parser.add_argument(
        "--select-webcam", "--select-cam",
        action="store_true",
        help="Buka dialog visual untuk memilih index webcam atau file video untuk CAM-01 dan CAM-02 sebelum benchmark dimulai."
    )
    parser.add_argument(
        "--cam-top",
        type=str,
        default=None,
        help="Index webcam untuk CAM-01 Top (misal: 0 atau 1)"
    )
    parser.add_argument(
        "--cam-side",
        type=str,
        default=None,
        help="Index webcam untuk CAM-02 Side (misal: 1 atau 0)"
    )
    parser.add_argument(
        "--select-top",
        action="store_true",
        help="Buka dialog khusus untuk memilih sumber (Webcam/Video) CAM-01 Top sebelum mulai."
    )
    parser.add_argument(
        "--select-side",
        action="store_true",
        help="Buka dialog khusus untuk memilih sumber (Webcam/Video) CAM-02 Side sebelum mulai."
    )
    parser.add_argument(
        "--top-source",
        type=str,
        default=DEFAULT_TOP_VIDEO,
        help="Path video, index webcam (0/1), atau ketik 'browse' untuk membuka dialog sumber CAM-01 Top Slat/Rib"
    )
    default_side_arg = DEFAULT_SIDE_VIDEO if DEFAULT_SIDE_VIDEO else "0"
    parser.add_argument(
        "--side-source",
        type=str,
        default=default_side_arg,
        help=f"Path video, index webcam (0/1), atau ketik 'browse' untuk Side Bolt/Pin (default: {os.path.basename(default_side_arg) if os.path.exists(str(default_side_arg)) else default_side_arg})"
    )
    parser.add_argument("--side-name", type=str, default=env_side_name, choices=["RIGHT", "LEFT"], help="Sisi konveyor (RIGHT / LEFT)")
    parser.add_argument("--side-cam-id", type=str, default=env_side_cam_id, help="ID Kamera samping (default: CAM-02 untuk RIGHT, CAM-03 untuk LEFT)")
    parser.add_argument("--rib-mode", type=str, default=env_rib_mode, choices=["HYBRID", "YOLO_ONLY", "DUAL_POLARITY_ONLY"], help="Mode deteksi sirip")
    parser.add_argument("--device", type=str, default=env_device, help="Device komputasi: 'cuda' atau 'cpu' (default: auto)")
    parser.add_argument("--duration", type=int, default=env_duration, help="Durasi benchmark dalam detik (0 = berjalan terus sampai dihentikan)")
    parser.add_argument("--record-csv", type=str, default=env_csv, help="Path file CSV untuk mencatat log FPS berkala")
    parser.add_argument(
        "--enable-db", "--db",
        action="store_true",
        default=env_db,
        dest="enable_db",
        help="Aktifkan pencatatan SQLite database dan pengiriman card notification ke MS Teams saat terdeteksi abnormalitas (NG)"
    )
    parser.add_argument(
        "--engine",
        choices=["threads", "process"],
        default=env_engine,
        help="Arsitektur runner: 'threads' (default: hemat RAM/VRAM untuk laptop 8GB & single GPU, zero IPC crash), atau 'process' (multiprocessing untuk Jetson Orin multi-core)."
    )

    args = parser.parse_args()


    # Auto-resolve device (CUDA jika tersedia, else CPU)
    if not args.device:
        try:
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            args.device = "cpu"

    # Auto-resolve Camera ID untuk side inspection sesuai standar PRD & Frontend
    if not args.side_cam_id:
        args.side_cam_id = "CAM-02" if str(args.side_name).upper() in ["RIGHT", "RH"] else "CAM-03"

    # 0. Interaktif Dialog Pemilihan Sumber / Webcam
    if args.cam_top is not None:
        args.top_source = str(args.cam_top)
    if args.cam_side is not None:
        args.side_source = str(args.cam_side)

    if args.webcam:
        cams = detect_available_cameras(max_devices=6)
        print(f"[*] Mode Live Dual Webcam diaktifkan. Mendeteksi webcam...")
        if len(cams) >= 2:
            args.top_source = str(cams[0])
            args.side_source = str(cams[1])
            print(f"[*] Berhasil memetakan 2 Webcam:")
            print(f"    - CAM-01 Top  (Plat & Rib) : Webcam #{args.top_source}")
            print(f"    - {args.side_cam_id} Side (Baut & Pin): Webcam #{args.side_source}")
        elif len(cams) == 1:
            args.top_source = str(cams[0])
            print(f"[*] Terdeteksi 1 Webcam (#{cams[0]}). Memetakan ke CAM-01 Top.")
            print(f"    - {args.side_cam_id} Side tetap menggunakan video pengujian: {os.path.basename(str(args.side_source))}")
        else:
            print("[!] Peringatan: Tidak ditemukan webcam fisik yang aktif. Tetap menggunakan input default.")

    if args.select_webcam or args.browse or str(args.top_source).lower() == "browse" or str(args.side_source).lower() == "browse":
        print("\n[*] Menampilkan dialog pengaturan sumber kamera...")
        ch_top, ch_side = select_dual_camera_setup_dialog(
            default_top=args.top_source,
            default_side=args.side_source,
            side_cam_id=args.side_cam_id
        )
        if ch_top:
            args.top_source = ch_top
        if ch_side:
            args.side_source = ch_side
    else:
        if args.select_top or str(args.top_source).lower() in ["select", "dialog", "file"]:
            ch_top = select_source_dialog(title="[CAM-01 TOP] Pilih Sumber Plat & Rib", default_source=args.top_source)
            if ch_top:
                args.top_source = ch_top
        if args.select_side or str(args.side_source).lower() in ["select", "dialog", "file"]:
            ch_side = select_source_dialog(title=f"[{args.side_cam_id} SIDE] Pilih Sumber Baut & Pin", default_source=args.side_source)
            if ch_side:
                args.side_source = ch_side

    # Validasi fallback path jika path manual tidak ditemukan
    if not is_stream_or_camera(args.top_source) and not os.path.exists(str(args.top_source)):
        resolved_top = resolve_source_path(args.top_source)
        if os.path.exists(resolved_top) or is_stream_or_camera(resolved_top):
            args.top_source = resolved_top
        else:
            args.top_source = DEFAULT_TOP_VIDEO

    if not is_stream_or_camera(args.side_source) and not os.path.exists(str(args.side_source)):
        resolved_side = resolve_source_path(args.side_source)
        if os.path.exists(resolved_side) or is_stream_or_camera(resolved_side):
            args.side_source = resolved_side
        else:
            args.side_source = DEFAULT_SIDE_VIDEO

    print("=" * 86)
    print("🚀 MULTI-CAMERA INSPECTION BENCHMARK RUNNER")
    print("=" * 86)
    print(f"[*] Engine Arsitektur  : {'THREADS (Hemat RAM 8GB, Berbagi VRAM Bersama)' if args.engine == 'threads' else 'PROCESS (Multiprocessing Terisolasi)'}")
    print(f"[*] Mode Operasional   : {args.mode.upper()}")
    print(f"[*] CAM-01 Top Source  : {os.path.basename(str(args.top_source)) if os.path.exists(str(args.top_source)) else args.top_source}")
    print(f"[*] {args.side_cam_id} Side Source : {os.path.basename(str(args.side_source)) if os.path.exists(str(args.side_source)) else args.side_source} ({args.side_name})")
    print(f"[*] Rib Model Mode     : {args.rib_mode}")
    print(f"[*] Durasi Uji Coba    : {'Tak Terbatas (Tekan [Q] / Ctrl+C)' if args.duration <= 0 else f'{args.duration} detik'}")
    print(f"[*] Database Logging   : {'AKTIF [RECORDING & TEAMS NOTIFIER]' if args.enable_db else 'OFF (DRY-RUN BENCHMARK - Tekan [D] di jendela untuk aktifkan)'}")
    print("=" * 86)

    # 1. Terapkan konfigurasi Anti-Throttling & Prioritas Windows
    configure_windows_process_performance()

    # 2. Jika mode engine THREADS (default & rekomendasi laptop), eksekusi Single-Process
    if args.engine == "threads":
        _run_threaded_engine(args)
        return

    # Inisialisasi Shared Memory untuk Telemetri Antar-Proses
    stop_event = mp.Event()
    fps_top = mp.Value("d", 0.0)
    fps_side = mp.Value("d", 0.0)
    frames_top = mp.Value("i", 0)
    frames_side = mp.Value("i", 0)
    lat_top = mp.Value("d", 0.0)
    lat_side = mp.Value("d", 0.0)
    stat_top = mp.Array("c", 64)
    stat_side = mp.Array("c", 64)

    stat_top.value = b"STARTING"
    stat_side.value = b"STARTING"

    processes = []

    # 1. Spawn Worker Top
    if args.mode in ["dual", "headless", "top-only"]:
        p_top = mp.Process(
            target=_run_top_worker,
            args=(
                args.top_source,
                "dual" if args.mode == "dual" else "headless",
                args.rib_mode,
                args.device,
                args.enable_db,
                stop_event,
                fps_top,
                frames_top,
                lat_top,
                stat_top
            ),
            name="Worker-TopInspector"
        )
        processes.append(("TOP", p_top))

    # 2. Spawn Worker Side
    if args.mode in ["dual", "headless", "side-only"]:
        p_side = mp.Process(
            target=_run_side_worker,
            args=(
                args.side_source,
                "dual" if args.mode == "dual" else "headless",
                args.side_name,
                args.side_cam_id,
                args.device,
                args.enable_db,
                stop_event,
                fps_side,
                frames_side,
                lat_side,
                stat_side
            ),
            name="Worker-SideInspector"
        )
        processes.append(("SIDE", p_side))

    # Jalankan seluruh proses anak secara paralel
    for name, p in processes:
        p.start()
        print(f"[*] Sub-proses [{name}] berhasil di-spawn | PID: {p.pid}")

    if sys.platform.startswith("win"):
        print("[*] Menginisialisasi konteks CUDA & model AI di masing-masing core CPU (~8-10 detik di Windows)...")

    # Header Telemetri Terminal
    print("-" * 86)
    print(f"{'WAKTU':^8} | {'CAM-01 TOP (FPS)':^18} | {f'{args.side_cam_id} SIDE (FPS)':^18} | {'COMBINED FPS':^14} | {'STATUS':^16}")
    print("-" * 86)

    # Inisialisasi CSV Writer jika diaktifkan
    csv_file = None
    if args.record_csv:
        try:
            csv_file = open(args.record_csv, "w", encoding="utf-8")
            csv_file.write("Timestamp,Elapsed_Sec,Top_FPS,Top_Latency_ms,Top_Frames,Side_FPS,Side_Latency_ms,Side_Frames,Combined_Throughput_FPS\n")
        except Exception as e_csv:
            print(f"[!] Gagal membuat file CSV: {e_csv}")
            csv_file = None

    t_spawn = time.time()
    t0 = None
    last_print = 0.0

    try:
        while not stop_event.is_set():
            now = time.time()

            # Tunggu hingga sub-proses selesai memuat model ke GPU & frame pertama diproses
            if t0 is None:
                if frames_top.value > 0 or frames_side.value > 0:
                    t0 = now
                else:
                    # Cek jika ada proses yang mati saat inisialisasi
                    for name, p in processes:
                        if not p.is_alive() and not stop_event.is_set():
                            print(f"\n[!] Sub-proses [{name}] berhenti saat inisialisasi (Exit code: {p.exitcode})")
                            stop_event.set()
                            break
                    time.sleep(0.1)
                    continue

            elapsed = now - t0

            # Cek batasan durasi
            if args.duration > 0 and elapsed >= args.duration:
                print(f"\n[*] Durasi benchmark {args.duration}s tercapai. Menghentikan seluruh proses...")
                stop_event.set()
                break

            # Cek jika ada proses yang mati secara tidak wajar
            for name, p in processes:
                if not p.is_alive() and not stop_event.is_set():
                    print(f"\n[!] Sub-proses [{name}] berhenti lebih awal (Exit code: {p.exitcode})")
                    stop_event.set()
                    break

            # Cetak ringkasan berkala setiap 1.0 detik
            if now - last_print >= 1.0:
                last_print = now
                v_top_fps = fps_top.value
                v_side_fps = fps_side.value
                v_comb_fps = v_top_fps + v_side_fps

                s_top = stat_top.value.decode("utf-8", errors="replace")
                s_side = stat_side.value.decode("utf-8", errors="replace")

                stat_summary = f"T:{s_top}|S:{s_side}"
                time_str = f"{int(elapsed):03d}s"

                print(
                    f"{time_str:^8} | "
                    f"{v_top_fps:5.1f} FPS ({lat_top.value:4.1f}ms) | "
                    f"{v_side_fps:5.1f} FPS ({lat_side.value:4.1f}ms) | "
                    f"{v_comb_fps:6.1f} FPS   | "
                    f"{stat_summary:^16}"
                )

                if csv_file:
                    ts_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    csv_file.write(
                        f"{ts_now},{elapsed:.1f},{v_top_fps:.2f},{lat_top.value:.1f},{frames_top.value},"
                        f"{v_side_fps:.2f},{lat_side.value:.1f},{frames_side.value},{v_comb_fps:.2f}\n"
                    )
                    csv_file.flush()

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n[*] Interupsi Keyboard (Ctrl+C). Menghentikan sistem...")
        stop_event.set()

    finally:
        # Tunggu semua proses anak selesai dengan aman
        for name, p in processes:
            p.join(timeout=3.0)
            if p.is_alive():
                print(f"[!] Menghentikan paksa sub-proses [{name}]...")
                p.terminate()

        if csv_file:
            csv_file.close()
            print(f"[*] Log hasil benchmark tersimpan di: {os.path.abspath(args.record_csv)}")

    # ==================================================================================
    # LAPORAN AKHIR BENCHMARK (EXECUTIVE SUMMARY)
    # ==================================================================================
    total_elapsed = (time.time() - t0) if t0 is not None else (time.time() - t_spawn)
    f_top = frames_top.value
    f_side = frames_side.value
    avg_top_fps = f_top / total_elapsed if total_elapsed > 0 else 0.0
    avg_side_fps = f_side / total_elapsed if total_elapsed > 0 else 0.0

    print("\n" + "=" * 86)
    print("📊 LAPORAN AKHIR HASIL BENCHMARK MULTI-KAMERA")
    print("=" * 86)
    print(f"  • Total Waktu Pengujian     : {total_elapsed:.1f} detik")
    if args.mode in ["dual", "headless", "top-only"]:
        print(f"  • CAM-01 Top (Slat & Rib)   : {f_top} frames ({avg_top_fps:.1f} rata-rata FPS)")
    if args.mode in ["dual", "headless", "side-only"]:
        print(f"  • {args.side_cam_id} Side (Bolt & Pin)  : {f_side} frames ({avg_side_fps:.1f} rata-rata FPS)")
    print(f"  • Throughput Gabungan       : {(avg_top_fps + avg_side_fps):.1f} frames/detik")

    # Evaluasi Ambang Batas Kelayakan
    target_fps = 20.0  # Standar kelayakan konveyor real-time
    if avg_top_fps >= target_fps and (args.mode == "top-only" or avg_side_fps >= target_fps):
        verdict = "🟢 SANGAT STABIL (Memenuhi standar real-time conveyor >= 20 FPS)"
    elif avg_top_fps >= 15.0:
        verdict = "🟡 CUKUP (15 - 20 FPS, disarankan konversi model ke TensorRT FP16 di Jetson)"
    else:
        verdict = "🔴 BUTUH OPTIMASI (< 15 FPS, gunakan mode headless & akselerasi TensorRT)"

    print(f"  • Penilaian Stabilitas Device: {verdict}")
    print("=" * 86)
    print("[*] Selesai.\n")


if __name__ == "__main__":
    # Wajib menggunakan metode 'spawn' untuk multiprocessing PyTorch & CUDA di Windows & Linux
    mp.set_start_method("spawn", force=True)
    main()
