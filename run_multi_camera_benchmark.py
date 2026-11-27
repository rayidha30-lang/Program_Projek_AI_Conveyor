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
import subprocess
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
                        if k:
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

# Daftarkan direktori modul ke sys.path secara global di level teratas agar dapat diimpor di seluruh proses
for _p in [
    TOP_SRC_DIR,
    SIDE_SRC_DIR,
    os.path.join(BASE_DIR, "Vision_Engine_Core", "02_Top_Inspection_Slat_Rib", "src"),
    os.path.join(BASE_DIR, "Vision_Engine_Core", "01_Side_Inspection_Bolt_Pin_Lekukan", "src"),
    os.path.join(BASE_DIR, "Jetson_Vision_Engine", "02_Top_Inspection_Slat_Rib", "src"),
    os.path.join(BASE_DIR, "Jetson_Vision_Engine", "01_Side_Inspection_Bolt_Pin", "src"),
    os.path.join(BASE_DIR, "02_Top_Inspection_Slat_Rib", "src"),
    os.path.join(BASE_DIR, "01_Side_Inspection_Bolt_Pin", "src"),
    BASE_DIR
]:
    if _p and os.path.exists(_p) and _p not in sys.path:
        sys.path.insert(0, _p)


def _ensure_db_bridge_on_path():
    """Ensure canonical Program/db_bridge.py (or Program/Database fallback) is importable."""
    _db_candidates = [
        BASE_DIR,
        os.path.join(BASE_DIR, "Database"),
    ]
    db_dir = next(
        (d for d in _db_candidates if os.path.exists(os.path.join(d, "db_bridge.py"))),
        _db_candidates[0],
    )
    if db_dir not in sys.path:
        sys.path.insert(0, db_dir)


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


def is_usable_source(src):
    """Return True only for an existing media file, camera index, or stream URL."""
    if src is None or not str(src).strip():
        return False
    return is_stream_or_camera(src) or os.path.exists(str(src))


def resolve_source_path(src, candidates=None):
    """
    Menemukan path media/video atau sumber kamera yang valid secara multi-platform.
    Mendukung index webcam, URL RTSP/CSI, absolute path, dan relative path.
    """
    if not src:
        if candidates:
            for cand in candidates:
                res = resolve_source_path(cand)
                if is_usable_source(res):
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
                if is_usable_source(res):
                    return res
    return ""


def record_top_ng_if_enabled(
    inspector, frame, inspect_result, fps, frame_idx, total_frames, slat_no, rib_no,
):
    """Render and persist NG evidence only when the top verdict is actually NG."""
    if not inspector or inspect_result.get("status") != "NG" or not inspector.is_db_enabled():
        return False
    annotation = inspector.render_combined_dashboard(
        frame=frame,
        inspect_result=inspect_result,
        fps=fps,
        frame_idx=frame_idx,
        total_frames=total_frames,
    )
    inspector.record_if_ng(
        frame_raw=frame,
        frame_ann=annotation,
        inspect_res=inspect_result,
        slat_no=slat_no,
        rib_no=rib_no,
    )
    return True


def rib_result_for_dataset(rib_result, rib_inspector):
    """Supply the live transit-frame count omitted by the current rib result payload."""
    result = dict(rib_result or {})
    result.setdefault("active_rib_frames", int(getattr(rib_inspector, "active_rib_frames", 0) or 0))
    return result


def finalize_pending_rib_dataset(recorder, previous):
    """Persist the last completed candidate when a file loop or run ends."""
    if recorder is None or previous is None:
        return None
    result, rib_number, slat_number = previous
    return recorder.finalize(result, rib_number, slat_number)


def record_rib_dataset_step(
    recorder,
    inspector,
    frame,
    inspect_res,
    avg_fps,
    frame_idx,
    total_frames,
    last_rib_no,
    previous,
    hud_out=None,
):
    """Advance rib dataset state for one inspected frame with lazy annotation rendering."""
    if recorder is None or not inspector or not inspector.is_db_enabled():
        return last_rib_no, previous
    active_profile = getattr(inspector, "active_profile_name", None)
    if active_profile:
        recorder.camera_profile = active_profile
    rib_result = rib_result_for_dataset(
        inspect_res.get("rib_res", {}), inspector.rib_inspector
    )
    current_rib_no = int(rib_result.get("rib_count", 0) or 0)
    current_status = str(rib_result.get("status", "WAIT")).upper()
    slat_number = getattr(inspector.slat_inspector, "slat_count", None)
    if current_rib_no < last_rib_no:
        finalize_pending_rib_dataset(recorder, previous)
        recorder.start_new_session()
        last_rib_no = 0
        previous = None
    if current_rib_no > last_rib_no:
        if last_rib_no > 0 and previous is not None:
            previous_result, previous_rib_no, previous_slat_no = previous
            recorder.finalize(previous_result, previous_rib_no, previous_slat_no)
        last_rib_no = current_rib_no
    if current_status in {"OK", "NG"} and current_rib_no > 0:
        if recorder.should_observe(rib_result, current_rib_no):
            annotated = inspector.render_combined_dashboard(
                frame=frame,
                inspect_result=inspect_res,
                fps=avg_fps,
                frame_idx=frame_idx,
                total_frames=total_frames,
            )
            if hud_out is not None:
                hud_out["frame"] = annotated
            recorder.observe(rib_result, frame, annotated, current_rib_no, slat_number)
        previous = (dict(rib_result), current_rib_no, slat_number)
    return last_rib_no, previous


# Video Default CAM-01 Top Inspection (Plat Slat & Bilah Rib)
ENV_TOP_VIDEO = os.environ.get("TOP_CAMERA_SOURCE")
CANDIDATE_TOP_VIDEOS = [
    ENV_TOP_VIDEO,
    os.path.join(VIDEO_DIR, "Deform Sedikit.mp4"),
    os.path.join(VIDEO_DIR, "OK Semua.mp4"),
    os.path.join(BASE_DIR, "Documentation", "Video Testing", "Deform Sedikit.mp4"),
    os.path.join(BASE_DIR, "Documentation", "Video Testing", "OK Semua.mp4"),
    os.path.join(BASE_DIR, "Jetson_Vision_Engine", "Video_Testing", "OK Semua.mp4"),
]
DEFAULT_TOP_VIDEO = resolve_source_path(ENV_TOP_VIDEO, CANDIDATE_TOP_VIDEOS)

# Video Default Side Inspection (Baut, Pin & Bracket - CAM-02 RH / CAM-03 LH)
ENV_SIDE_VIDEO = os.environ.get("SIDE_CAMERA_SOURCE")
CANDIDATE_SIDE_VIDEOS = [
    ENV_SIDE_VIDEO,
    os.path.join(VIDEO_DIR, "Deform Sedikit.mp4"),
    os.path.join(BASE_DIR, "Documentation", "WhatsApp Video 2026-08-19 at 15.37.46.mp4"),
    os.path.abspath(os.path.join(BASE_DIR, "..", "Documentation", "WhatsApp Video 2026-08-19 at 15.37.46.mp4")),
    os.path.abspath(os.path.join(BASE_DIR, "Documentation", "WhatsApp Video 2026-08-19 at 15.37.46.mp4")),
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
    Mendeteksi seluruh webcam fisik / perangkat video USB yang aktif pada PC/Jetson secara non-destruktif.
    Di Linux (Jetson Ubuntu), mendeteksi node /dev/video* tanpa membuka stream agar tidak mengunci V4L2.
    """
    cams = []
    if sys.platform.startswith("win"):
        import cv2
        for i in range(min(max_devices, 5)):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap is not None and cap.isOpened():
                    cams.append(i)
                    cap.release()
            except Exception:
                pass
        if not cams:
            cams = [0, 1, 2, 3]
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
            for i in range(min(max_devices, 4)):
                if os.path.exists(f"/dev/video{i}"):
                    cams.append(i)
        if not cams:
            cams = [0, 1, 2, 3]
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
            if not cap.isOpened():
                cap = cv2.VideoCapture(idx)
        else:
            # Di Linux / Jetson, utamakan CAP_V4L2 untuk stabilitas driver USB
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                # Cek codec aktif & beri peringatan ramah jika sensor hardware dibatasi DirectShow
                fourcc_val = int(cap.get(cv2.CAP_PROP_FOURCC))
                fourcc_str = "".join([chr((fourcc_val >> 8 * i) & 0xFF) for i in range(4)])
                w_act = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h_act = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if "YUY" in fourcc_str.upper() and w_act >= 1280:
                    print(f"[*] [INFO WEBCAM #{idx}] Format aktif: {fourcc_str} ({w_act}x{h_act}).")
                    print(f"    Webcam laptop menggunakan format YUY2 uncompressed (dibatasi driver DirectShow ke ~10 FPS pada 720p).")
                    print(f"    Untuk 30 FPS penuh, gunakan webcam USB eksternal dengan chip kompresi hardware MJPEG.")
            except Exception:
                pass
        return cap, True, str(idx)
    else:
        path = str(src).strip()
        cap = cv2.VideoCapture(path)
        return cap, False, path


def get_available_videos():
    """Mengambil daftar seluruh file video pengujian yang ada di direktori Video_Testing / Documentation."""
    search_dirs = [
        VIDEO_DIR,
        os.path.join(BASE_DIR, "Video_Testing"),
        os.path.join(BASE_DIR, "Jetson_Vision_Engine", "Video_Testing"),
        os.path.abspath(os.path.join(BASE_DIR, "..", "Documentation")),
        os.path.join(BASE_DIR, "Documentation"),
    ]
    exts = (".mp4", ".avi", ".mkv", ".mov", ".wmv")
    vids = []
    seen = set()
    for d in search_dirs:
        if d and os.path.exists(d):
            for f in sorted(os.listdir(d)):
                if f.lower().endswith(exts):
                    fp = os.path.abspath(os.path.join(d, f))
                    if fp not in seen:
                        seen.add(fp)
                        vids.append(fp)
    return vids


def cycle_next_video(current_src):
    """Beralih ke file video pengujian berikutnya secara instan tanpa dialog GUI (Zero-Crash)."""
    vids = get_available_videos()
    if not vids:
        print("\n[!] Tidak ada file video pengujian ditemukan di folder Video_Testing.")
        return current_src

    cur_norm = os.path.abspath(str(current_src)) if current_src else ""
    try:
        cur_idx = vids.index(cur_norm)
        next_idx = (cur_idx + 1) % len(vids)
    except ValueError:
        next_idx = 0
    next_vid = vids[next_idx]
    print(f"\n[*] [HOTKEY V] Beralih ke video ({next_idx + 1}/{len(vids)}): {os.path.basename(next_vid)}")
    return next_vid


def cycle_next_camera(current_src):
    """Beralih ke perangkat webcam berikutnya secara instan tanpa dialog GUI (Zero-Crash)."""
    cams = detect_available_cameras(max_devices=6)
    if not cams:
        print("\n[!] Tidak ada perangkat webcam fisik USB yang terdeteksi.")
        return current_src

    is_cam = str(current_src).strip().isdigit()
    cur_idx = int(current_src) if is_cam else -1
    try:
        cur_pos = cams.index(cur_idx)
        next_pos = (cur_pos + 1) % len(cams)
    except ValueError:
        next_pos = 0
    next_cam = cams[next_pos]
    print(f"\n[*] [HOTKEY C] Beralih ke Live Webcam: #{next_cam} (daftar aktif: {cams})")
    return str(next_cam)


def _run_isolated_dialog(dialog_mode, **kwargs):
    """
    Menjalankan dialog GUI (Tkinter) di subprocess terpisah (Subprocess Isolation).
    Menghindari tabrakan event loop X11 dengan OpenCV di Linux/Jetson,
    serta mencegah fork error di multiprocessing child processes.
    """
    import subprocess
    dialog_script = os.path.join(BASE_DIR, "scripts", "gui_source_dialog.py")
    if os.path.exists(dialog_script):
        cmd = [sys.executable, dialog_script, "--dialog", dialog_mode]
    else:
        cmd = [sys.executable, os.path.abspath(__file__), "--gui-dialog", dialog_mode]
    for k, v in kwargs.items():
        if v is not None:
            cmd.extend([f"--{k.replace('_', '-')}", str(v)])
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
        if proc.returncode != 0 and proc.stderr:
            sys.stderr.write(f"[!] Warning dialog subprocess ({dialog_mode}):\n{proc.stderr.strip()}\n")
            sys.stderr.flush()
        res = {}
        for line in proc.stdout.strip().splitlines():
            line = line.strip()
            if line.startswith("RESULT_FILE:"):
                f_val = line.split(":", 1)[1].strip()
                res["file"] = f_val
                if "source" not in res:
                    res["source"] = f_val
            elif line.startswith("RESULT_SOURCE:"):
                s_val = line.split(":", 1)[1].strip()
                res["source"] = s_val
                if "file" not in res and os.path.exists(s_val):
                    res["file"] = s_val
            elif line.startswith("RESULT_DUAL_TOP:"):
                res["top"] = line.split(":", 1)[1].strip()
            elif line.startswith("RESULT_DUAL_SIDE:"):
                res["side"] = line.split(":", 1)[1].strip()
            elif line.startswith("RESULT_DUAL_MODE:"):
                res["mode"] = line.split(":", 1)[1].strip()
            elif line.startswith("RESULT_SCHED_MODE:"):
                res.setdefault("schedule", {})["mode"] = line.split(":", 1)[1].strip()
            elif line.startswith("RESULT_SCHED_INTERVAL:"):
                try:
                    res.setdefault("schedule", {})["interval_min"] = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith("RESULT_SCHED_ACTIVE:"):
                try:
                    res.setdefault("schedule", {})["active_min"] = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith("RESULT_SCHED_ANCHOR:"):
                try:
                    res.setdefault("schedule", {})["anchor_min"] = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith("RESULT_SCHED_SLOTS:"):
                res.setdefault("schedule", {})["slots"] = line.split(":", 1)[1].strip()
        return res
    except Exception as e:
        print(f"[!] Gagal mengeksekusi dialog subprocess ({dialog_mode}): {e}")
        return {}


def select_file_dialog(initial_dir=None, title="Pilih File Video"):
    """
    Membuka file dialog explorer secara aman di subprocess terisolasi.
    """
    res = _run_isolated_dialog("file", initial_dir=initial_dir, title=title)
    new_f = res.get("file") or res.get("source")
    if new_f and os.path.exists(new_f):
        return new_f
    return None


def select_source_dialog(title="Pilih Sumber Kamera / Video", default_source=None):
    """
    Menampilkan dialog visual modern untuk memilih Webcam atau File Video secara aman di subprocess terisolasi.
    """
    res = _run_isolated_dialog("source", title=title, default=default_source)
    new_src = res.get("source") or res.get("file")
    if new_src:
        return new_src
    return default_source


def select_dual_camera_setup_dialog(
    default_top,
    default_side,
    side_cam_id="CAM-02",
    default_mode=None,
    include_mode=False,
    default_schedule=None,
    include_schedule=False,
):
    """
    Dialog visual terpadu untuk mengatur sumber CAM-01 Top, CAM-02 Side, Mode Operasional,
    dan opsional Jadwal Deteksi AI (Durasi & Siklus).
    """
    extra_kwargs = {}
    if isinstance(default_schedule, dict):
        if "mode" in default_schedule:
            extra_kwargs["schedule_mode"] = default_schedule["mode"]
        if "interval_min" in default_schedule:
            extra_kwargs["schedule_interval_min"] = default_schedule["interval_min"]
        if "active_min" in default_schedule:
            extra_kwargs["schedule_active_min"] = default_schedule["active_min"]
        if "anchor_min" in default_schedule:
            extra_kwargs["schedule_anchor_min"] = default_schedule["anchor_min"]
        if "slots" in default_schedule:
            extra_kwargs["schedule_slots"] = default_schedule["slots"]

    res = _run_isolated_dialog(
        "dual",
        top=default_top,
        side=default_side,
        side_id=side_cam_id,
        mode=default_mode,
        **extra_kwargs,
    )
    new_top = res.get("top", default_top)
    new_side = res.get("side", default_side)
    new_mode = res.get("mode", default_mode)
    new_sched = res.get("schedule") if "schedule" in res else (dict(default_schedule) if isinstance(default_schedule, dict) else None)
    if include_schedule:
        return new_top, new_side, new_mode, new_sched
    if include_mode:
        return new_top, new_side, new_mode
    return new_top, new_side


def build_schedule_dict_from_args(args):
    """Ekstrak ringkasan konfigurasi jadwal dari Namespace args untuk dikirim ke dialog [W]."""
    return {
        "mode": getattr(args, "schedule_mode", "continuous"),
        "interval_min": float(getattr(args, "schedule_interval_min", 60.0) or 60.0),
        "active_min": float(getattr(args, "schedule_active_min", 15.0) or 15.0),
        "anchor_min": float(getattr(args, "schedule_anchor_min", 0.0) or 0.0),
        "slots": str(getattr(args, "schedule_slots", "") or ""),
    }


def apply_schedule_dict_to_args(args, sched_dict):
    """Terapkan hasil pengaturan jadwal dari dialog [W] ke args dan bangun DetectionScheduleConfig baru."""
    from detection_scheduler import DetectionScheduleConfig, parse_schedule_slots

    if isinstance(sched_dict, dict):
        if sched_dict.get("mode") in ("continuous", "periodic", "slots"):
            args.schedule_mode = sched_dict["mode"]
        if sched_dict.get("interval_min") is not None:
            args.schedule_interval_min = float(sched_dict["interval_min"])
        if sched_dict.get("active_min") is not None:
            args.schedule_active_min = float(sched_dict["active_min"])
        if sched_dict.get("anchor_min") is not None:
            args.schedule_anchor_min = float(sched_dict["anchor_min"])
        if "slots" in sched_dict and sched_dict["slots"] is not None:
            args.schedule_slots = str(sched_dict["slots"])

    return DetectionScheduleConfig(
        mode=getattr(args, "schedule_mode", "continuous"),
        interval_min=float(getattr(args, "schedule_interval_min", 60.0) or 60.0),
        active_min=float(getattr(args, "schedule_active_min", 15.0) or 15.0),
        anchor_minute=float(getattr(args, "schedule_anchor_min", 0.0) or 0.0),
        slots=parse_schedule_slots(getattr(args, "schedule_slots", "")),
        prewarm_sec=float(getattr(args, "schedule_prewarm_sec", 2.0) or 2.0),
        standby_fps=float(getattr(args, "schedule_standby_fps", 20.0) or 20.0),
        motion_quota=bool(getattr(args, "schedule_motion_quota", False)),
    )


from video_nav_utils import (
    SEEK_KEY_SPEC,
    IdleConveyorSkipper,
    compute_seek_target_from_key,
    detect_conveyor_motion_score,
    format_seconds_mmss,
    format_speed_label,
    next_speed_step,
    parse_time_to_seconds,
    prompt_jump_time_dialog,
)
from detection_scheduler import (
    CODE_TO_STATE as SCHED_CODE_TO_STATE,
    STATE_TO_CODE as SCHED_STATE_TO_CODE,
    DetectionScheduler,
)


def make_standby_top_result(schedule_state="STANDBY", badge=""):
    """Build lightweight stub result for CAM-01 when AI is bypassed in STANDBY/PRE_WARMUP."""
    st = str(schedule_state or "STANDBY").upper()
    msg = badge or ("Scheduled Standby (Live View)" if st == "STANDBY" else "GPU Pre-Warmup...")
    return {
        "status": st,
        "schedule_state": st,
        "slat_res": {"display_status": st, "eval_mode": msg, "is_ng": False},
        "rib_res": {"status": st, "is_rib_present": False, "is_ng": False, "message": msg},
        "elapsed_ms": 0.0,
        "is_conveyor_moving": False,
    }


def make_standby_side_payload(camera_id, side, schedule_state="STANDBY"):
    """Build lightweight stub payload for CAM-02/03 when AI is bypassed in STANDBY/PRE_WARMUP."""
    st = str(schedule_state or "STANDBY").upper()
    return {
        "camera_id": camera_id,
        "side": side,
        "overall_result": st,
        "schedule_state": st,
        "should_emit": False,
        "components": [],
        "bolts": [],
        "pins": [],
    }


def draw_schedule_standby_overlay(frame_bgr, badge_text):
    """Draw a minimal standby status banner onto a copy of a live passthrough frame for OpenCV GUI."""
    import cv2
    out = frame_bgr.copy()
    h_f, w_f = out.shape[:2]
    cv2.rectangle(out, (12, 12), (min(w_f - 12, 680), 56), (24, 24, 28), -1)
    cv2.rectangle(out, (12, 12), (min(w_f - 12, 680), 56), (0, 210, 255), 2, cv2.LINE_AA)
    cv2.putText(
        out,
        f"{badge_text} | [W]: Setup | [S]: Override | [Q]: Keluar",
        (24, 41),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 235, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def _handle_isolated_gui_dialog():
    """
    Handler utama proses terisolasi GUI Tkinter.
    Dipanggil saat skrip dijalankan dengan argumen flag: --gui-dialog <file|source|dual>.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Subprocess GUI Dialog Runner")
    parser.add_argument("--gui-dialog", type=str, required=True, choices=["file", "source", "dual"])
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--initial-dir", type=str, default="")
    parser.add_argument("--default", type=str, default="")
    parser.add_argument("--top", type=str, default="")
    parser.add_argument("--side", type=str, default="")
    parser.add_argument("--side-id", type=str, default="CAM-02")
    parser.add_argument("--mode", type=str, default="dual")
    args, _ = parser.parse_known_args()

    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as e:
        sys.stderr.write(f"[!] Tkinter tidak tersedia: {e}\n")
        sys.exit(1)

    cams = detect_available_cameras(max_devices=6)

    # Definisi font lintas-platform (Windows: Segoe UI, Linux Jetson: DejaVu Sans / Helvetica)
    if sys.platform.startswith("win"):
        F_FAMILY = "Segoe UI"
    elif sys.platform.startswith("linux"):
        F_FAMILY = "DejaVu Sans"
    else:
        F_FAMILY = "Helvetica"

    F_TITLE = (F_FAMILY, 11, "bold")
    F_SEC = (F_FAMILY, 10, "bold")
    F_BOLD = (F_FAMILY, 9, "bold")
    F_REG = (F_FAMILY, 9)
    F_SM = (F_FAMILY, 8)
    F_ITALIC = (F_FAMILY, 8, "italic")

    def _apply_x11_safe_window_props(win):
        try:
            win.attributes("-topmost", True)
        except Exception:
            pass
        try:
            win.lift()
            win.focus_force()
        except Exception:
            pass

    if args.gui_dialog == "file":
        try:
            root = tk.Tk()
            root.withdraw()
            _apply_x11_safe_window_props(root)
            init_d = args.initial_dir if (args.initial_dir and os.path.exists(args.initial_dir)) else VIDEO_DIR
            fp = filedialog.askopenfilename(
                parent=root,
                title=args.title or "Pilih File Video",
                initialdir=init_d,
                filetypes=[
                    ("File Video (*.mp4;*.avi;*.mkv;*.mov;*.wmv)", "*.mp4 *.avi *.mkv *.mov *.wmv"),
                    ("Semua File (*.*)", "*.*")
                ]
            )
            try:
                root.update()
                root.destroy()
            except Exception:
                pass
            if fp and os.path.exists(fp):
                print(f"RESULT_FILE:{os.path.abspath(fp)}")
            sys.exit(0)
        except Exception as e:
            sys.stderr.write(f"[!] Error file dialog: {e}\n")
            sys.exit(1)

    elif args.gui_dialog == "source":
        try:
            root = tk.Tk()
            root.title(args.title or "Pilih Sumber Kamera / Video")
            _apply_x11_safe_window_props(root)
            root.geometry("480x400")
            root.minsize(440, 360)

            selected_result = {"val": args.default}

            lbl_info = tk.Label(root, text=f"🎥 {args.title or 'Pilih Sumber Input'}", font=F_TITLE, pady=10)
            lbl_info.pack()

            frame_options = tk.LabelFrame(root, text="Pilihan Sumber Input", font=F_SEC, padx=15, pady=10)
            frame_options.pack(fill="both", expand=True, padx=15, pady=5)

            is_default_cam = str(args.default).strip().isdigit() if args.default else False
            var_choice = tk.StringVar(value="webcam" if is_default_cam else "file")
            default_cam_idx = int(args.default) if is_default_cam else (cams[0] if cams else 0)
            var_cam_idx = tk.IntVar(value=default_cam_idx)

            candidate_cams = sorted(list(set(cams + [0, 1, 2, 3])))
            rb_cam = tk.Radiobutton(frame_options, text="Gunakan Perangkat Live Webcam:", variable=var_choice, value="webcam", font=F_BOLD)
            rb_cam.pack(anchor="w", pady=(0, 4))
            frame_cams = tk.Frame(frame_options, padx=20)
            frame_cams.pack(fill="x", pady=2)
            for c_idx in candidate_cams:
                tag = " (Aktif)" if c_idx in cams else ""
                tk.Radiobutton(frame_cams, text=f"Webcam #{c_idx}{tag}", variable=var_cam_idx, value=c_idx, command=lambda: var_choice.set("webcam"), font=F_REG).pack(anchor="w", pady=2)

            rb_file = tk.Radiobutton(frame_options, text="Gunakan File Rekaman Video:", variable=var_choice, value="file", font=F_BOLD)
            rb_file.pack(anchor="w", pady=(12, 4))

            lbl_file_path = tk.Label(
                frame_options,
                text=os.path.basename(str(args.default)) if (args.default and not is_default_cam) else "(Belum memilih file)",
                font=F_ITALIC,
                fg="#555555"
            )
            lbl_file_path.pack(anchor="w", padx=20)

            file_selected_path = {"path": str(args.default) if (args.default and not is_default_cam) else None}

            def on_browse():
                var_choice.set("file")
                f = filedialog.askopenfilename(
                    parent=root,
                    title="Pilih File Video Pengujian",
                    initialdir=VIDEO_DIR,
                    filetypes=[
                        ("File Video (*.mp4;*.avi;*.mkv;*.mov;*.wmv)", "*.mp4 *.avi *.mkv *.mov *.wmv"),
                        ("Semua File (*.*)", "*.*")
                    ]
                )
                if f:
                    file_selected_path["path"] = f
                    lbl_file_path.config(text=os.path.basename(f), fg="blue")

            btn_browse = tk.Button(frame_options, text="📁 Buka File Explorer...", command=on_browse, font=F_REG)
            btn_browse.pack(anchor="w", padx=20, pady=4)

            def on_ok():
                if var_choice.get() == "webcam":
                    selected_result["val"] = str(var_cam_idx.get())
                else:
                    if file_selected_path["path"]:
                        selected_result["val"] = file_selected_path["path"]
                    else:
                        selected_result["val"] = args.default
                root.destroy()

            frame_btn = tk.Frame(root, pady=10)
            frame_btn.pack(side="bottom", fill="x", padx=15)
            tk.Button(frame_btn, text="Pilih Sumber Ini", bg="#007acc", fg="white", font=F_BOLD, width=15, command=on_ok).pack(side="right", padx=5)
            tk.Button(frame_btn, text="Batal", font=F_REG, width=10, command=root.destroy).pack(side="right")

            root.mainloop()
            if selected_result["val"]:
                print(f"RESULT_SOURCE:{selected_result['val']}")
            sys.exit(0)
        except Exception as e:
            sys.stderr.write(f"[!] Error source dialog: {e}\n")
            sys.exit(1)

    elif args.gui_dialog == "dual":
        try:
            root = tk.Tk()
            root.title("Pengaturan Dual Kamera & Mode (CAM-01 Top & CAM-02 Side)")
            _apply_x11_safe_window_props(root)
            root.geometry("650x690")
            root.minsize(580, 600)

            init_mode = args.mode if args.mode in ("dual", "opencv", "web", "headless") else "dual"
            res = {"top": args.top, "side": args.side, "mode": init_mode}

            lbl_t = tk.Label(root, text="🎥 PENGATURAN SUMBER DUAL KAMERA", font=F_TITLE, pady=8)
            lbl_t.pack()
            lbl_sub = tk.Label(
                root,
                text=f"Webcam terdeteksi: {len(cams)} perangkat ({', '.join(['#'+str(c) for c in cams]) if cams else 'Tidak ada'})",
                font=F_ITALIC,
                fg="#444444"
            )
            lbl_sub.pack(pady=(0, 8))

            # --- PANEL CAM-01 TOP ---
            f_top = tk.LabelFrame(root, text="[CAM-01 TOP] Deteksi Plat Slat & Bilah Rib", font=F_SEC, padx=12, pady=6)
            f_top.pack(fill="x", padx=15, pady=5)

            is_top_cam = str(args.top).strip().isdigit() if args.top else False
            var_top_choice = tk.StringVar(value="webcam" if (is_top_cam or (cams and not args.top)) else "file")
            var_top_cam = tk.IntVar(value=int(args.top) if is_top_cam else (cams[0] if cams else 0))
            top_file_path = {"path": str(args.top) if (args.top and not is_top_cam) else None}

            candidate_cams = sorted(list(set(cams + [0, 1, 2, 3])))
            f_top_cams = tk.Frame(f_top)
            f_top_cams.pack(fill="x")
            tk.Radiobutton(f_top_cams, text="Webcam:", variable=var_top_choice, value="webcam", font=F_BOLD).pack(side="left")
            for c_idx in candidate_cams:
                tk.Radiobutton(f_top_cams, text=f"Cam #{c_idx}", variable=var_top_cam, value=c_idx, command=lambda: var_top_choice.set("webcam"), font=F_REG).pack(side="left", padx=6)

            f_top_file = tk.Frame(f_top, pady=4)
            f_top_file.pack(fill="x")
            tk.Radiobutton(f_top_file, text="File Video:", variable=var_top_choice, value="file", font=F_BOLD).pack(side="left")
            lbl_top_fname = tk.Label(f_top_file, text=os.path.basename(str(top_file_path["path"])) if top_file_path["path"] else "(Default Video)", font=F_SM, fg="#0055aa")
            lbl_top_fname.pack(side="left", padx=6)

            def browse_top():
                var_top_choice.set("file")
                f = filedialog.askopenfilename(parent=root, title="Pilih Video CAM-01 Top", initialdir=VIDEO_DIR)
                if f:
                    top_file_path["path"] = f
                    lbl_top_fname.config(text=os.path.basename(f), fg="blue")

            tk.Button(f_top_file, text="Pilih...", command=browse_top, font=F_SM).pack(side="right")

            # --- PANEL SIDE ---
            f_side = tk.LabelFrame(root, text=f"[{args.side_id} SIDE] Deteksi Baut, Pin & Bracket Lekukan", font=F_SEC, padx=12, pady=6)
            f_side.pack(fill="x", padx=15, pady=8)

            is_side_cam = str(args.side).strip().isdigit() if args.side else False
            var_side_choice = tk.StringVar(value="webcam" if (is_side_cam or (len(cams) >= 2 and not args.side)) else "file")
            default_side_idx = int(args.side) if is_side_cam else (cams[1] if len(cams) >= 2 else (cams[0] if cams else 0))
            var_side_cam = tk.IntVar(value=default_side_idx)
            side_file_path = {"path": str(args.side) if (args.side and not is_side_cam) else None}

            f_side_cams = tk.Frame(f_side)
            f_side_cams.pack(fill="x")
            tk.Radiobutton(f_side_cams, text="Webcam:", variable=var_side_choice, value="webcam", font=F_BOLD).pack(side="left")
            for c_idx in candidate_cams:
                tk.Radiobutton(f_side_cams, text=f"Cam #{c_idx}", variable=var_side_cam, value=c_idx, command=lambda: var_side_choice.set("webcam"), font=F_REG).pack(side="left", padx=6)

            f_side_file = tk.Frame(f_side, pady=4)
            f_side_file.pack(fill="x")
            tk.Radiobutton(f_side_file, text="File Video:", variable=var_side_choice, value="file", font=F_BOLD).pack(side="left")
            lbl_side_fname = tk.Label(f_side_file, text=os.path.basename(str(side_file_path["path"])) if side_file_path["path"] else "(Default Video)", font=F_SM, fg="#0055aa")
            lbl_side_fname.pack(side="left", padx=6)

            def browse_side():
                var_side_choice.set("file")
                f = filedialog.askopenfilename(parent=root, title=f"Pilih Video {args.side_id} Side", initialdir=VIDEO_DIR)
                if f:
                    side_file_path["path"] = f
                    lbl_side_fname.config(text=os.path.basename(f), fg="blue")

            tk.Button(f_side_file, text="Pilih...", command=browse_side, font=F_SM).pack(side="right")

            # --- PANEL MODE OPERASIONAL ---
            f_mode = tk.LabelFrame(root, text="[MODE OPERASIONAL] Tampilan & Arsitektur Streaming", font=F_SEC, padx=12, pady=6)
            f_mode.pack(fill="x", padx=15, pady=6)

            var_mode = tk.StringVar(master=root, value=init_mode)
            tk.Radiobutton(f_mode, text="Dual GUI + Web (dual) — 2 Jendela OpenCV + WebSocket", variable=var_mode, value="dual", font=F_REG, anchor="w").pack(fill="x", pady=1)
            tk.Radiobutton(f_mode, text="OpenCV HUD Only (opencv) — Hanya 2 Jendela OpenCV (Tanpa WebSocket)", variable=var_mode, value="opencv", font=F_REG, anchor="w").pack(fill="x", pady=1)
            tk.Radiobutton(f_mode, text="Web Dashboard Only (web) — Tanpa GUI Desktop (Hybrid + WebSocket 3-Kamera)", variable=var_mode, value="web", font=F_REG, anchor="w").pack(fill="x", pady=1)
            tk.Radiobutton(f_mode, text="Headless (headless) — Tanpa GUI Desktop (Murni Inferensi)", variable=var_mode, value="headless", font=F_REG, anchor="w").pack(fill="x", pady=1)

            def on_confirm_all():
                if var_top_choice.get() == "webcam":
                    res["top"] = str(var_top_cam.get())
                elif top_file_path["path"]:
                    res["top"] = top_file_path["path"]

                if var_side_choice.get() == "webcam":
                    res["side"] = str(var_side_cam.get())
                elif side_file_path["path"]:
                    res["side"] = side_file_path["path"]

                res["mode"] = var_mode.get()
                root.destroy()

            f_bottom = tk.Frame(root, pady=12)
            f_bottom.pack(side="bottom", fill="x", padx=15)
            tk.Button(f_bottom, text="Terapkan & Mulai", bg="#007acc", fg="white", font=F_SEC, width=18, command=on_confirm_all).pack(side="right", padx=5)
            tk.Button(f_bottom, text="Gunakan Bawaan", font=F_REG, width=15, command=root.destroy).pack(side="right")

            root.mainloop()
            if res.get("top"):
                print(f"RESULT_DUAL_TOP:{res['top']}")
            if res.get("side"):
                print(f"RESULT_DUAL_SIDE:{res['side']}")
            if res.get("mode"):
                print(f"RESULT_DUAL_MODE:{res['mode']}")
            sys.exit(0)
        except Exception as e:
            sys.stderr.write(f"[!] Error dual dialog: {e}\n")
            sys.exit(1)


def configure_worker_thread_budget(cv2_threads=2, torch_threads=2):
    """Batasi thread OpenCV & PyTorch per proses agar tidak terjadi oversubscription CPU antar-kamera."""
    try:
        import cv2
        cv2.setNumThreads(max(1, int(cv2_threads)))
    except Exception:
        pass
    try:
        import torch
        torch.set_num_threads(max(1, int(torch_threads)))
    except Exception:
        pass


class _AsyncFramePrefetcher:
    """
    Background reader thread untuk overlap CPU video decode (cap.read) dengan GPU inference.
    - Video file (is_cam=False): bounded FIFO queue (maxsize=2) agar 0 frame terlewat.
    - Live camera (is_cam=True): single-slot latest-frame overwrite agar latensi minimum.
    """

    def __init__(self, cap, is_cam=False, maxsize=2):
        import queue as _q
        import threading as _th

        self._cap = cap
        self._is_cam = bool(is_cam)
        self._stop = _th.Event()
        self._q = _q.Queue(maxsize=max(1, int(maxsize)))
        self._latest_lock = _th.Lock()
        self._latest_ready = _th.Event()
        self._latest_item = None
        self._eof = False
        self._thread = _th.Thread(target=self._loop, daemon=True, name="Thread-AsyncFramePrefetch")
        self._thread.start()

    def _loop(self):
        import queue as _q

        while not self._stop.is_set():
            try:
                ret, frame = self._cap.read()
            except Exception:
                ret, frame = False, None

            if not ret or frame is None:
                if self._is_cam:
                    time.sleep(0.005)
                    continue
                self._eof = True
                while not self._stop.is_set():
                    try:
                        self._q.put((False, None), timeout=0.05)
                        break
                    except _q.Full:
                        continue
                break

            if self._is_cam:
                with self._latest_lock:
                    self._latest_item = (True, frame)
                    self._latest_ready.set()
            else:
                while not self._stop.is_set():
                    try:
                        self._q.put((True, frame), timeout=0.05)
                        break
                    except _q.Full:
                        continue

    def read(self, timeout=1.0):
        import queue as _q

        if self._is_cam:
            if not self._latest_ready.wait(timeout=timeout):
                return False, None
            with self._latest_lock:
                item = self._latest_item
                self._latest_item = None
                self._latest_ready.clear()
            return item if item is not None else (False, None)
        try:
            return self._q.get(timeout=timeout)
        except _q.Empty:
            return False, None

    def stop(self):
        self._stop.set()
        self._latest_ready.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)


class _AsyncTopWSEncoder:
    """
    Single-slot async encoder thread untuk CAM-01 WebSocket 360p clean stream.
    Thread AI cukup menaruh referensi frame & hasil inspeksi (<0.05 ms), sementara
    resize 360p + overlay + JPEG encode berjalan di background tanpa memblokir AI.
    """

    def __init__(self, ws_top_queue, ws_clients_val, ws_overlay_val, ws_hud_mode="clean", ws_fps=0.0, jpeg_quality=65):
        import threading as _th

        self._ws_top_queue = ws_top_queue
        self._ws_clients_val = ws_clients_val
        self._ws_overlay_val = ws_overlay_val
        self._ws_hud_mode = str(ws_hud_mode or "clean").lower()
        self._ws_interval = (1.0 / float(ws_fps)) if (ws_fps and float(ws_fps) > 0.0) else 0.0
        self._jpeg_quality = int(max(30, min(95, int(jpeg_quality))))
        self._lock = _th.Lock()
        self._event = _th.Event()
        self._stop = _th.Event()
        self._pending = None
        self._last_push = 0.0
        self._thread = _th.Thread(target=self._loop, daemon=True, name="Thread-TopWSEncoder")
        self._thread.start()

    def submit(self, frame, inspect_res, avg_fps, slat_no=None, rib_no=None):
        if self._ws_top_queue is None or self._ws_clients_val is None or self._ws_clients_val.value <= 0:
            return
        with self._lock:
            self._pending = (frame, inspect_res, float(avg_fps), slat_no, rib_no)
            self._event.set()

    def _loop(self):
        import cv2
        from ws_stream_server import build_stream_metadata, render_clean_top_360p

        while not self._stop.is_set():
            if not self._event.wait(timeout=0.1):
                continue
            self._event.clear()
            if self._stop.is_set():
                break
            if self._ws_clients_val is None or self._ws_clients_val.value <= 0:
                with self._lock:
                    self._pending = None
                continue

            now = time.perf_counter()
            if self._ws_interval > 0.0 and (now - self._last_push) < self._ws_interval:
                continue

            with self._lock:
                item = self._pending
                self._pending = None
            if item is None:
                continue

            frame, inspect_res, avg_fps, slat_no, rib_no = item
            try:
                show_ov = bool(self._ws_overlay_val.value if self._ws_overlay_val is not None else 1)
                ws_img = render_clean_top_360p(frame, inspect_res, show_overlay=show_ov)
                ok_j, jb = cv2.imencode(".jpg", ws_img, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
                if not ok_j:
                    continue
                self._last_push = time.perf_counter()
                meta_t = build_stream_metadata(
                    "CAM-01",
                    {
                        "result": inspect_res,
                        "fps": avg_fps,
                        "inspector_meta": {
                            "slat_no": slat_no,
                            "rib_no": rib_no,
                        },
                    },
                )
                if self._ws_top_queue.full():
                    try:
                        self._ws_top_queue.get_nowait()
                    except Exception:
                        pass
                self._ws_top_queue.put_nowait((meta_t, jb.tobytes()))
            except Exception:
                pass

    def stop(self):
        self._stop.set()
        self._event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.5)


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
    status_buf,
    gui_fps=22.0,
    start_time=None,
    ws_top_queue=None,
    ws_clients_val=None,
    ws_overlay_val=None,
    ws_db_val=None,
    ws_hud_mode="clean",
    ws_fps=0.0,
    ws_sched_state_val=None,
    standby_fps=20.0,
):
    """Worker proses terisolasi untuk CAM-01 Top Inspection (Plat & Rib)."""
    # Bersihkan cache VRAM PyTorch sebelum proses inferensi dimulai
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    # Konfigurasi Anti-Throttling & Prioritas Realtime Windows + Budget Thread CPU
    configure_windows_process_performance()
    configure_worker_thread_budget(cv2_threads=3, torch_threads=2)

    # 1. Pastikan path impor modul top inspector terdaftar di proses anak
    if TOP_SRC_DIR not in sys.path:
        sys.path.insert(0, TOP_SRC_DIR)

    import cv2
    import numpy as np
    from combined_top_inspector import CombinedTopInspector
    from conveyor_db_path import resolve_db_path
    from dataset_recorder import RibDatasetRecorder
    from dataset_review_store import DatasetReviewStore

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

    try:
        dataset_recorder = RibDatasetRecorder(
            DatasetReviewStore(resolve_db_path()),
            os.path.join(os.path.dirname(resolve_db_path()), "dataset_captures"),
            source_label="CAM-01 TOP",
            camera_profile=getattr(inspector, "active_profile_name", None),
            async_write=True,
        )
    except Exception as e_dataset:
        dataset_recorder = None
        print(f"[WARN TOP WORKER] Perekam dataset rib tidak aktif: {e_dataset}")

    # 3. Inisialisasi Sumber Video / Kamera
    cap, is_cam, cap_src = create_capture(source_path)
    if not cap.isOpened():
        print(f"[ERROR TOP WORKER] Tidak dapat membuka sumber video CAM-01: {source_path}")
        status_buf.value = b"SRC_ERROR"
        return

    win_name = "CAM-01 TOP (SLAT & RIB) | [1]/[V]: Ganti Video TOP | [W]: Dual Setup | [D]: DB | [Q]: Keluar"
    if mode == "dual":
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 920, 518)
        cv2.moveWindow(win_name, 20, 40)

    # Variabel Benchmark FPS & Latency
    frame_count = 0
    fps_history = []
    t_start = time.time()
    t_prev = t_start
    gui_interval = (1.0 / gui_fps) if (gui_fps and gui_fps > 0) else 0.0
    last_gui_render = 0.0
    dataset_last_rib_no = 0
    dataset_previous = None
    total_frames_top = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_cam else 0

    if start_time and (not is_cam) and total_frames_top > 0:
        s_sec = parse_time_to_seconds(start_time)
        if s_sec is not None:
            v_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            if v_fps <= 5.0 or v_fps > 120.0:
                v_fps = 30.0
            init_f = int(np.clip(round(s_sec * v_fps), 0, total_frames_top - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, init_f)

    last_ws_push = 0.0
    ws_interval = (1.0 / ws_fps) if (ws_fps and ws_fps > 0) else 0.0
    ws_encoder = None
    if ws_top_queue is not None and str(ws_hud_mode).lower() != "legacy":
        ws_encoder = _AsyncTopWSEncoder(
            ws_top_queue=ws_top_queue,
            ws_clients_val=ws_clients_val,
            ws_overlay_val=ws_overlay_val,
            ws_hud_mode=ws_hud_mode,
            ws_fps=ws_fps,
        )

    _torch = None
    try:
        import torch as _torch_mod
        _torch = _torch_mod
        if _torch.cuda.is_available() and device != "cpu":
            _torch.backends.cudnn.benchmark = True
    except Exception:
        _torch = None

    prefetcher = _AsyncFramePrefetcher(cap, is_cam=is_cam, maxsize=2)

    def _replace_top_capture(new_cap, new_is_cam):
        nonlocal cap, is_cam, prefetcher, total_frames_top
        if prefetcher is not None:
            prefetcher.stop()
        if cap is not None and cap.isOpened():
            cap.release()
        cap = new_cap
        is_cam = bool(new_is_cam)
        total_frames_top = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_cam else 0
        prefetcher = _AsyncFramePrefetcher(cap, is_cam=is_cam, maxsize=2)

    print("[*] Worker CAM-01 Top Inspection AKTIF dan mulai memproses...")
    last_ws_db = int(ws_db_val.value) if ws_db_val is not None else None
    prev_sched_state = "ACTIVE"
    prewarm_frames_done = 0
    standby_interval = (1.0 / max(1.0, float(standby_fps or 20.0)))

    try:
        while not stop_event.is_set():
            t_loop_top0 = time.perf_counter()
            if ws_db_val is not None:
                cur_ws_db = int(ws_db_val.value)
                if cur_ws_db != last_ws_db:
                    last_ws_db = cur_ws_db
                    if hasattr(inspector, "db_bridge") and inspector.db_bridge is not None:
                        inspector.db_bridge.set_enabled(bool(cur_ws_db))
                        st_lbl = "AKTIF (RECORDING & NOTIFIKASI TEAMS)" if cur_ws_db else "MATI (DRY-RUN)"
                        print(f"\n[*] [CAM-01 TOP] SAKLAR DATABASE (via Web FE): {st_lbl}\n")
            ret, frame = prefetcher.read(timeout=0.5)
            if not ret or frame is None:
                if stop_event.is_set():
                    break
                # Jika file video habis, putar ulang (looping) dengan re-open agar FFmpeg tidak memory leak
                if not is_cam:
                    finalize_pending_rib_dataset(dataset_recorder, dataset_previous)
                    if dataset_recorder is not None:
                        dataset_recorder.start_new_session()
                    dataset_last_rib_no = 0
                    dataset_previous = None
                    inspector.reset_counters()
                    new_cap = cv2.VideoCapture(cap_src)
                    _replace_top_capture(new_cap, False)
                    ret, frame = prefetcher.read(timeout=0.5)
                    if not ret or frame is None:
                        time.sleep(0.02)
                        continue
                else:
                    time.sleep(0.01)
                    continue

            sched_state = (
                SCHED_CODE_TO_STATE.get(int(ws_sched_state_val.value), "ACTIVE")
                if ws_sched_state_val is not None
                else "ACTIVE"
            )
            if sched_state != prev_sched_state:
                if prev_sched_state == "ACTIVE":
                    finalize_pending_rib_dataset(dataset_recorder, dataset_previous)
                    dataset_previous = None
                    inspector.reset_active_tracking()
                elif sched_state == "ACTIVE":
                    inspector.reset_active_tracking()
                if sched_state != "PRE_WARMUP":
                    prewarm_frames_done = 0
                prev_sched_state = sched_state

            if sched_state != "ACTIVE":
                if sched_state == "PRE_WARMUP" and prewarm_frames_done < 2:
                    if _torch is not None:
                        with _torch.inference_mode():
                            inspector.inspect(frame)
                    else:
                        inspector.inspect(frame)
                    inspector.reset_active_tracking()
                    prewarm_frames_done += 1

                inspect_res = make_standby_top_result(sched_state)
                t_frame_end = time.time()
                dt = t_frame_end - t_prev
                t_prev = t_frame_end
                curr_fps = 1.0 / dt if dt > 0 else float(standby_fps or 20.0)
                fps_history.append(curr_fps)
                if len(fps_history) > 30:
                    fps_history.pop(0)
                avg_fps = sum(fps_history) / len(fps_history)
                frame_count += 1
                fps_val.value = avg_fps
                frames_val.value = frame_count
                latency_val.value = 0.0
                status_buf.value = sched_state.encode("utf-8")[:63]
                if ws_encoder is not None:
                    ws_encoder.submit(
                        frame=frame,
                        inspect_res=inspect_res,
                        avg_fps=avg_fps,
                        slat_no=getattr(inspector.slat_inspector, "slat_count", None),
                        rib_no=getattr(inspector.rib_inspector, "rib_count", None),
                    )
                if mode == "dual":
                    t_now = time.time()
                    if (t_now - last_gui_render) >= gui_interval:
                        cv2.imshow(win_name, draw_schedule_standby_overlay(frame, f"SCHED: {sched_state} (LIVE VIEW)"))
                        last_gui_render = t_now
                    key_st = cv2.waitKey(1) & 0xFF
                    if key_st in [ord("q"), ord("Q"), 27]:
                        stop_event.set()
                        break
                rem_stby = standby_interval - (time.perf_counter() - t_loop_top0)
                if rem_stby > 0.001:
                    time.sleep(rem_stby)
                continue

            # Jalankan inspeksi terpadu plat slat & rib (ukur latensi murni pipeline AI)
            t_infer_start = time.time()
            if _torch is not None:
                with _torch.inference_mode():
                    inspect_res = inspector.inspect(frame)
            else:
                inspect_res = inspector.inspect(frame)
            t_frame_end = time.time()
            latency_ms = (t_frame_end - t_infer_start) * 1000.0

            frame_count += 1
            dt = t_frame_end - t_prev
            t_prev = t_frame_end

            curr_fps = 1.0 / dt if dt > 0 else 30.0
            if frame_count > 1:
                fps_history.append(curr_fps)
                if len(fps_history) > 30:
                    fps_history.pop(0)
            avg_fps = (sum(fps_history) / len(fps_history)) if fps_history else curr_fps

            # Update Shared Memory
            fps_val.value = avg_fps
            frames_val.value = frame_count
            latency_val.value = latency_ms
            curr_stat = inspect_res.get("status", "WAIT")
            status_buf.value = curr_stat.encode("utf-8")[:63]

            # Push encoded frame ke WebSocket relay HANYA jika ada client browser aktif (Zero-Client Sleep)
            if ws_encoder is not None:
                ws_encoder.submit(
                    frame=frame,
                    inspect_res=inspect_res,
                    avg_fps=avg_fps,
                    slat_no=getattr(inspector.slat_inspector, "slat_count", None),
                    rib_no=getattr(inspector.rib_inspector, "rib_count", None),
                )
            elif ws_top_queue is not None and ws_clients_val is not None and ws_clients_val.value > 0:
                t_ws_now = time.perf_counter()
                if ws_interval <= 0.0 or (t_ws_now - last_ws_push) >= ws_interval:
                    last_ws_push = t_ws_now
                    try:
                        from ws_stream_server import build_stream_metadata

                        leg_h = inspector.render_combined_dashboard(
                            frame=frame,
                            inspect_result=inspect_res,
                            fps=avg_fps,
                            frame_idx=frame_count,
                            total_frames=total_frames_top,
                        )
                        ws_img = cv2.resize(leg_h, (640, 524), interpolation=cv2.INTER_AREA)
                        ok_j, jb = cv2.imencode(".jpg", ws_img, [cv2.IMWRITE_JPEG_QUALITY, 65])
                        if ok_j:
                            meta_t = build_stream_metadata(
                                "CAM-01",
                                {
                                    "result": inspect_res,
                                    "fps": avg_fps,
                                    "inspector_meta": {
                                        "slat_no": getattr(inspector.slat_inspector, "slat_count", None),
                                        "rib_no": getattr(inspector.rib_inspector, "rib_count", None),
                                    },
                                },
                            )
                            if ws_top_queue.full():
                                try:
                                    ws_top_queue.get_nowait()
                                except Exception:
                                    pass
                            ws_top_queue.put_nowait((meta_t, jb.tobytes()))
                    except Exception:
                        pass

            hud_cache = {}
            dataset_last_rib_no, dataset_previous = record_rib_dataset_step(
                dataset_recorder,
                inspector,
                frame,
                inspect_res,
                avg_fps,
                frame_count,
                total_frames_top,
                dataset_last_rib_no,
                dataset_previous,
                hud_out=hud_cache,
            )

            # Render Visual jika mode dual (Decoupled GUI Rendering)
            hud = hud_cache.get("frame")
            is_ng = inspect_res.get("status") == "NG"
            if mode == "dual":
                t_now = time.time()
                should_render = ((t_now - last_gui_render) >= gui_interval) or is_ng
                if should_render:
                    if hud is None:
                        hud = inspector.render_combined_dashboard(
                            frame=frame,
                            inspect_result=inspect_res,
                            fps=avg_fps,
                            frame_idx=frame_count,
                            total_frames=total_frames_top
                        )
                    cv2.imshow(win_name, hud)
                    if getattr(inspector, "show_rib_unwarp_window", False):
                        rib_unwarp = getattr(inspector.rib_inspector, "last_rib_unwarped_annotated", None)
                        if rib_unwarp is not None:
                            cv2.imshow("CAM-01 TOP: PERSPECTIVE TOP-DOWN CONVEYOR (BIRD'S-EYE)", rib_unwarp)
                    last_gui_render = t_now
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
                elif key in [ord("t"), ord("T"), ord("e"), ord("E")]:
                    if hasattr(inspector, "rib_inspector") and hasattr(inspector.rib_inspector, "toggle_tracker_engine"):
                        new_eng = inspector.rib_inspector.toggle_tracker_engine()
                        print(f"\n[*] [CAM-01 TOP] 🔄 SAKELAR TRACKER ENGINE SIRIP: [{new_eng}]\n")
                elif key in [ord("j"), ord("J")]:
                    if hasattr(inspector, "toggle_rib_unwarp_window"):
                        st = inspector.toggle_rib_unwarp_window()
                        mode_str = "DITAMPILKAN" if st else "DISEMBUNYIKAN"
                        print(f"\n[*] [CAM-01 TOP] 🔍 JENDELA UNWARP BIRD'S-EYE: {mode_str}\n")
                elif key in [ord("b"), ord("B")]:
                    if hasattr(inspector, "toggle_simulate_bending"):
                        st = inspector.toggle_simulate_bending()
                        mode_str = "AKTIF (+20mm BENGKOK - NG TEST)" if st else "NON-AKTIF (NORMAL)"
                        print(f"\n[*] [CAM-01 TOP] ⚡ SIMULASI DEFEK SIRIP BENGKOK: {mode_str}\n")
                elif key in [ord("p"), ord("P")]:
                    if hasattr(inspector, "toggle_profile"):
                        inspector.toggle_profile()
                elif key in [ord("h"), ord("H")]:
                    if hasattr(inspector, "toggle_threshold_source"):
                        inspector.toggle_threshold_source()
                elif key in [ord("u"), ord("U"), ord("0")]:
                    if hasattr(inspector, "restore_default_profile"):
                        inspector.restore_default_profile()
                elif key in [ord("k"), ord("K"), ord("a"), ord("A")]:
                    if hasattr(inspector, "auto_align_rails_instant"):
                        inspector.auto_align_rails_instant(frame)
                elif key in [ord("g"), ord("G")]:
                    if hasattr(inspector, "toggle_alignment_guides"):
                        st = inspector.toggle_alignment_guides()
                        mode_str = "AKTIF (RETICLE ACUAN EMAS)" if st else "NON-AKTIF"
                        print(f"\n[*] [CAM-01 TOP] 📐 GARIS REFERENSI REL & SUMBU KONVEYOR: {mode_str}\n")
                elif key in [ord("l"), ord("L")]:
                    print("\n[*] [CAM-01 TOP] 🚀 Membuka Wizard Kalibrasi Interaktif Rel Ganda...")
                    calib_script = os.path.join(BASE_DIR, "Vision_Engine_Core", "02_Top_Inspection_Slat_Rib", "src", "calibrate_dual_rails.py")
                    subprocess.Popen([sys.executable, calib_script, "--source", str(cap_src)])
                elif key in [ord("v"), ord("V")]:
                    new_src = cycle_next_video(cap_src)
                    if new_src and str(new_src) != str(cap_src):
                        new_cap, new_is_cam, cap_src = create_capture(new_src)
                        if new_cap.isOpened():
                            _replace_top_capture(new_cap, new_is_cam)
                            frame_count = 0
                            fps_history.clear()
                            inspector.reset_counters()
                elif key in [ord("c"), ord("C")]:
                    new_src = cycle_next_camera(cap_src)
                    if new_src and str(new_src) != str(cap_src):
                        new_cap, new_is_cam, cap_src = create_capture(new_src)
                        if new_cap.isOpened():
                            _replace_top_capture(new_cap, new_is_cam)
                            frame_count = 0
                            fps_history.clear()
                            inspector.reset_counters()
                elif key in [ord("w"), ord("W")]:
                    print("\n[*] [CAM-01 TOP] Membuka Dialog Pemilihan Sumber (Webcam / File Video)...")
                    new_src = select_source_dialog(
                        title="[CAM-01 TOP] Pilih Sumber Video / Kamera Plat & Sirip",
                        default_source=cap_src
                    )
                    cv2.waitKey(1)
                    if new_src is not None and str(new_src) != str(cap_src):
                        new_cap, new_is_cam, new_cap_src = create_capture(new_src)
                        if new_cap.isOpened():
                            cap_src = new_cap_src
                            _replace_top_capture(new_cap, new_is_cam)
                            frame_count = 0
                            fps_history.clear()
                            inspector.reset_counters()
                            s_desc = f"Webcam #{cap_src}" if is_cam else os.path.basename(str(cap_src))
                            print(f"[*] [CAM-01 TOP] Sumber input dialihkan ke: {s_desc}")
                        else:
                            print(f"[!] [CAM-01 TOP] Gagal membuka sumber baru: {new_src}, tetap menggunakan sumber sebelumnya.")
                elif key in [ord("t"), ord("T"), ord("e"), ord("E")]:
                    if hasattr(inspector, "rib_inspector"):
                        new_eng = inspector.rib_inspector.toggle_tracker_engine()
                        print(f"\n[*] [CAM-01 TOP] 🔄 SAKELAR TRACKER ENGINE SIRIP: [{new_eng}]\n")
                elif key in [ord("1"), ord("o"), ord("O"), ord("f"), ord("F")]:
                    print("\n[*] [CAM-01 TOP] Membuka File Dialog Explorer...")
                    init_d = os.path.dirname(cap_src) if os.path.exists(str(cap_src)) else VIDEO_DIR
                    new_top = select_file_dialog(
                        initial_dir=init_d,
                        title="[CAM-01 TOP] Pilih File Video Plat Slat & Bilah Rib"
                    )
                    cv2.waitKey(1)  # Refresh window focus
                    if new_top and os.path.exists(new_top):
                        new_cap, new_is_cam, cap_src = create_capture(new_top)
                        if new_cap.isOpened():
                            _replace_top_capture(new_cap, new_is_cam)
                            frame_count = 0
                            fps_history.clear()
                            inspector.reset_counters()
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
        if ws_encoder is not None:
            ws_encoder.stop()
        if prefetcher is not None:
            prefetcher.stop()
        finalize_pending_rib_dataset(dataset_recorder, dataset_previous)
        if dataset_recorder is not None:
            dataset_recorder.close()
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
    status_buf,
    gui_fps=22.0,
    ws_sched_state_val=None,
    standby_fps=20.0,
):
    """Worker proses terisolasi untuk Side Inspection (Bolt, Pin, & Bracket)."""
    # Bersihkan cache VRAM PyTorch sebelum proses inferensi dimulai
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    # Konfigurasi Anti-Throttling & Prioritas Realtime Windows + Budget Thread CPU
    configure_windows_process_performance()
    configure_worker_thread_budget(cv2_threads=3, torch_threads=2)

    # 1. Pastikan path impor modul side inspector dan Database terdaftar di proses anak
    if SIDE_SRC_DIR not in sys.path:
        sys.path.insert(0, SIDE_SRC_DIR)
    _ensure_db_bridge_on_path()

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
        cap, is_cam, _ = create_capture(side_video_path)
        print(f"[*] Worker {camera_id} Side Inspection memuat video: {os.path.basename(side_video_path)}")
        if inspector.auto_preset_mode:
            det_p = inspector.resolve_preset_from_source(side_video_path)
            inspector.set_preset(det_p, verbose=True)
    elif source_path is None or source_path == "" or str(source_path).isdigit():
        cam_idx = int(source_path) if str(source_path).isdigit() else 0
        cap, is_cam, _ = create_capture(cam_idx)

        # Jika webcam tidak bisa dibuka, langsung alihkan ke file video testing
        if not cap.isOpened():
            cap.release()
            cap = None
            if DEFAULT_SIDE_VIDEO and os.path.exists(DEFAULT_SIDE_VIDEO):
                side_video_path = DEFAULT_SIDE_VIDEO
                cap, is_cam, _ = create_capture(side_video_path)
                print(f"[INFO SIDE WORKER] Webcam #{cam_idx} tidak terdeteksi. Beralih otomatis ke video testing: {os.path.basename(DEFAULT_SIDE_VIDEO)}")
                if inspector.auto_preset_mode:
                    det_p = inspector.resolve_preset_from_source(side_video_path)
                    inspector.set_preset(det_p, verbose=True)

    win_name = f"{camera_id} SIDE [{camera_id} {side}] | [2]/[O]: Video | [W]: Setup | [M]: Preset | [D]: DB | [Q]: Keluar"
    if mode == "dual":
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 920, 518)
        cv2.moveWindow(win_name, 960, 40)

    # Variabel Benchmark FPS & Latency
    frame_count = 0
    fps_history = []
    t_start = time.time()
    t_prev = t_start
    gui_interval = (1.0 / gui_fps) if (gui_fps and gui_fps > 0) else 0.0
    last_gui_render_side = 0.0

    print(f"[*] Worker {camera_id} Side Inspection [{side}] AKTIF dan mulai memproses...")
    prev_sched_state = "ACTIVE"
    prewarm_frames_done = 0
    standby_interval = (1.0 / max(1.0, float(standby_fps or 20.0)))

    try:
        while not stop_event.is_set():
            t_loop_side0 = time.perf_counter()
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

            sched_state = (
                SCHED_CODE_TO_STATE.get(int(ws_sched_state_val.value), "ACTIVE")
                if ws_sched_state_val is not None
                else "ACTIVE"
            )
            if sched_state != prev_sched_state:
                if hasattr(inspector, "reset_active_tracking"):
                    inspector.reset_active_tracking()
                if sched_state != "PRE_WARMUP":
                    prewarm_frames_done = 0
                prev_sched_state = sched_state

            if sched_state != "ACTIVE":
                if sched_state == "PRE_WARMUP" and prewarm_frames_done < 2:
                    inspector.process_frame(frame, render_display=False)
                    if hasattr(inspector, "reset_active_tracking"):
                        inspector.reset_active_tracking()
                    prewarm_frames_done += 1
                t_frame_end = time.time()
                dt = t_frame_end - t_prev
                t_prev = t_frame_end
                fps_history.append(1.0 / dt if dt > 0 else float(standby_fps or 20.0))
                if len(fps_history) > 30:
                    fps_history.pop(0)
                avg_fps = sum(fps_history) / len(fps_history)
                frame_count += 1
                fps_val.value = avg_fps
                frames_val.value = frame_count
                latency_val.value = 0.0
                status_buf.value = sched_state.encode("utf-8")[:63]
                if mode == "dual":
                    t_now = time.time()
                    if (t_now - last_gui_render_side) >= gui_interval:
                        cv2.imshow(win_name, draw_schedule_standby_overlay(frame, f"SCHED: {sched_state} (LIVE VIEW)"))
                        last_gui_render_side = t_now
                    key_st = cv2.waitKey(1) & 0xFF
                    if key_st in [ord("q"), ord("Q"), 27]:
                        stop_event.set()
                        break
                rem_stby = standby_interval - (time.perf_counter() - t_loop_side0)
                if rem_stby > 0.001:
                    time.sleep(rem_stby)
                continue

            # Jalankan inspeksi komponen samping (YOLO Bolt/Pin + PatchCore Bracket)
            t_infer_start = time.time()
            need_side_render = (mode == "dual" and (t_infer_start - last_gui_render_side) >= gui_interval) or (db_bridge is not None and db_bridge.is_enabled())
            display_frame, payload = inspector.process_frame(frame, render_display=need_side_render)
            t_frame_end = time.time()
            latency_ms = (t_frame_end - t_infer_start) * 1000.0

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

            # Render Visual jika mode dual (Decoupled GUI Rendering)
            is_side_ng = payload.get("overall_result") == "NG"
            t_now = time.time()
            should_render_side = ((t_now - last_gui_render_side) >= gui_interval) or is_side_ng

            if mode == "dual":
                if should_render_side and display_frame is not None:
                    # Tambahkan overlay benchmark banner di bagian bawah frame Side (Identik dengan Threaded Engine)
                    h_d, w_d = display_frame.shape[:2]
                    s_ok = getattr(inspector, 'slat_ok_count', getattr(inspector, 'stats', {}).get('ok_count', 0))
                    s_ng = getattr(inspector, 'slat_ng_count', getattr(inspector, 'stats', {}).get('ng_count', 0))
                    s_cyc = getattr(inspector, 'slat_cycle_count', 0)
                    qc_txt = f"Slat #{s_cyc} [OK:{s_ok} NG:{s_ng}]"
                    cur_p_label = getattr(inspector, 'preset', 'opsi_1').upper() if inspector else 'OPSI_1'
                    banner = f"FPS: {avg_fps:4.1f} | Lat: {latency_ms:4.1f}ms | [W]: Setup | [2/O]: Video | [M]: {cur_p_label} | {qc_txt} | [Q]: Keluar"
                    cv2.rectangle(display_frame, (10, h_d - 65), (w_d - 10, h_d - 10), (18, 18, 18), -1)
                    cv2.rectangle(display_frame, (10, h_d - 65), (w_d - 10, h_d - 10), (0, 220, 255), 2, cv2.LINE_AA)
                    cv2.putText(display_frame, banner, (25, h_d - 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.76, (0, 255, 255), 2, cv2.LINE_AA)

                    cv2.imshow(win_name, display_frame)
                    last_gui_render_side = t_now
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
                elif key in [ord("m"), ord("M")]:
                    if inspector is not None:
                        cur_p = getattr(inspector, "preset", "opsi_1")
                        next_p = "opsi_2" if cur_p == "opsi_1" else "opsi_1"
                        inspector.set_preset(next_p, verbose=True)
                        inspector.reset_session()
                        print(f"\n[*] [{camera_id} SIDE] Manual Switch Preset: {cur_p.upper()} -> {next_p.upper()}\n")
                elif key in [ord("v"), ord("V")]:
                    new_src = cycle_next_video(side_video_path)
                    if new_src and str(new_src) != str(side_video_path):
                        if cap is not None and cap.isOpened():
                            cap.release()
                        new_cap, is_cam, side_video_path = create_capture(new_src)
                        if new_cap.isOpened():
                            cap = new_cap
                            frame_count = 0
                            consecutive_cam_fails = 0
                            fps_history.clear()
                            if inspector is not None:
                                if getattr(inspector, "auto_preset_mode", True):
                                    det_p = inspector.resolve_preset_from_source(side_video_path)
                                    inspector.set_preset(det_p, verbose=True)
                                inspector.reset_session()
                elif key in [ord("c"), ord("C")]:
                    new_src = cycle_next_camera(side_video_path)
                    if new_src and str(new_src) != str(side_video_path):
                        if cap is not None and cap.isOpened():
                            cap.release()
                        new_cap, is_cam, side_video_path = create_capture(new_src)
                        if new_cap.isOpened():
                            cap = new_cap
                            frame_count = 0
                            consecutive_cam_fails = 0
                            fps_history.clear()
                            if inspector is not None:
                                if getattr(inspector, "auto_preset_mode", True):
                                    det_p = inspector.resolve_preset_from_source(side_video_path)
                                    inspector.set_preset(det_p, verbose=True)
                                inspector.reset_session()
                elif key in [ord("w"), ord("W")]:
                    print(f"\n[*] [{camera_id} SIDE] Membuka Dialog Pemilihan Sumber (Webcam / File Video)...")
                    curr_s = side_video_path if (not is_cam and side_video_path) else "0"
                    new_src = select_source_dialog(
                        title=f"[{camera_id} SIDE] Pilih Sumber Video / Kamera Baut & Pin",
                        default_source=curr_s
                    )
                    cv2.waitKey(1)
                    if new_src is not None and str(new_src) != str(curr_s):
                        if cap is not None and cap.isOpened():
                            cap.release()
                        new_cap, is_cam, side_video_path = create_capture(new_src)
                        if new_cap.isOpened():
                            cap = new_cap
                            frame_count = 0
                            consecutive_cam_fails = 0
                            fps_history.clear()
                            if inspector is not None:
                                if getattr(inspector, "auto_preset_mode", True):
                                    det_p = inspector.resolve_preset_from_source(side_video_path)
                                    inspector.set_preset(det_p, verbose=True)
                                inspector.reset_session()
                            s_desc = f"Webcam #{side_video_path}" if is_cam else os.path.basename(str(side_video_path))
                            print(f"[*] [{camera_id} SIDE] Sumber input dialihkan ke: {s_desc}")
                        else:
                            print(f"[!] [{camera_id} SIDE] Gagal membuka sumber baru: {new_src}, kembali ke video sebelumnya.")
                            cap, is_cam, side_video_path = create_capture(curr_s)
                elif key in [ord("2"), ord("o"), ord("O"), ord("f"), ord("F"), ord("s"), ord("S")]:
                    print(f"\n[*] [{camera_id} SIDE] Membuka File Dialog Explorer...")
                    candidate_side_dir = os.path.dirname(side_video_path) if side_video_path and os.path.exists(side_video_path) else VIDEO_DIR
                    new_side = select_file_dialog(
                        initial_dir=candidate_side_dir,
                        title=f"[{camera_id} SIDE] Pilih File Video Baut, Pin & Bracket"
                    )
                    cv2.waitKey(1)  # Refresh window focus
                    if new_side and os.path.exists(new_side):
                        if cap is not None and cap.isOpened():
                            cap.release()
                        new_cap, is_cam, side_video_path = create_capture(new_side)
                        if new_cap.isOpened():
                            cap = new_cap
                            frame_count = 0
                            consecutive_cam_fails = 0
                            fps_history.clear()
                            if inspector is not None:
                                if getattr(inspector, "auto_preset_mode", True):
                                    det_p = inspector.resolve_preset_from_source(side_video_path)
                                    inspector.set_preset(det_p, verbose=True)
                                inspector.reset_session()
                            print(f"[*] [{camera_id} SIDE] Video berhasil dialihkan ke: {os.path.basename(new_side)}")
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
    _ensure_db_bridge_on_path()

    from combined_top_inspector import CombinedTopInspector
    from side_inspector_pipeline import SideConveyorInspector
    try:
        from db_bridge import InspectionDBBridge
    except ImportError:
        InspectionDBBridge = None
    from conveyor_db_path import resolve_db_path
    from dataset_recorder import RibDatasetRecorder
    from dataset_review_store import DatasetReviewStore

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
    top_dataset_recorder = None
    cap_top = None
    is_cam_top = False
    top_source_path = args.top_source

    if args.mode in ["dual", "opencv", "headless", "web", "top-only"]:
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

        try:
            top_dataset_recorder = RibDatasetRecorder(
                DatasetReviewStore(resolve_db_path()),
                os.path.join(os.path.dirname(resolve_db_path()), "dataset_captures"),
                source_label="CAM-01 TOP",
                camera_profile=getattr(top_inspector, "active_profile_name", None),
                async_write=True,
            )
        except Exception as e_dataset:
            print(f"[WARN] Perekam dataset rib tidak aktif: {e_dataset}")

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

    if args.mode in ["dual", "opencv", "headless", "web", "side-only"]:
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

        # Sinkronkan preset side inspector secara otomatis dengan sumber video awal
        if side_inspector is not None and side_source_path:
            if getattr(side_inspector, "auto_preset_mode", True):
                det_p = side_inspector.resolve_preset_from_source(side_source_path)
                side_inspector.set_preset(det_p, verbose=True)

    # Shared thread states & locks
    stop_event = threading.Event()
    top_lock = threading.Lock()
    side_lock = threading.Lock()
    scheduler = DetectionScheduler.from_env_and_args(args)
    standby_interval = 1.0 / max(1.0, float(scheduler.config.standby_fps or 20.0))

    # Optional Opt-In Binary WebSocket Stream Server (Port 8765) + 3rd Camera (CAM-03)
    ws_hub = None
    ws_server = None
    side_left_inspector = None
    side_left_db_bridge = None

    def _apply_threaded_db_state(val: bool):
        args.enable_db = bool(val)
        if top_inspector is not None and getattr(top_inspector, "db_bridge", None) is not None:
            top_inspector.db_bridge.set_enabled(bool(val))
        if side_db_bridge is not None:
            side_db_bridge.set_enabled(bool(val))
        if side_left_db_bridge is not None:
            side_left_db_bridge.set_enabled(bool(val))
        st_lbl = "AKTIF (RECORDING & NOTIFIKASI TEAMS)" if val else "MATI (DRY-RUN BENCHMARK)"
        print(f"\n[DB SWITCH via Web FE] Status Database Logging seluruh kamera: {st_lbl}\n")

    def _apply_threaded_schedule_override(ovr: str):
        snap_o = scheduler.set_override(ovr)
        print(f"\n[SCHEDULE via Web FE] Override diubah ke '{snap_o.override}' -> {snap_o.hud_badge()}\n")
        return snap_o.to_ws_dict()

    if getattr(args, "ws_stream", False):
        try:
            from ws_stream_server import CameraFrameHub, WebSocketStreamServer

            ws_hub = CameraFrameHub()
            ws_server = WebSocketStreamServer(
                ws_hub,
                port=int(getattr(args, "ws_port", 8765) or 8765),
                max_fps=float(getattr(args, "ws_fps", 0.0) or 0.0),
                hud_mode=str(getattr(args, "ws_hud_mode", "clean") or "clean"),
                initial_db_enabled=bool(args.enable_db),
                on_db_enabled_change=_apply_threaded_db_state,
                initial_schedule=scheduler.evaluate().to_ws_dict(),
                on_schedule_override=_apply_threaded_schedule_override,
            )
            if not ws_server.start():
                ws_server = None
        except Exception as e_ws:
            print(f"[WARN WS-STREAM] Gagal menginisialisasi WebSocket server: {e_ws}")
            ws_hub = None
            ws_server = None

    cap_side_left = None
    is_cam_side_left = False
    side_left_source_path = getattr(args, "side_left_source", None)
    side_left_cam_id = "CAM-03" if str(args.side_cam_id).upper() != "CAM-03" else "CAM-02"
    side_left_side_name = "LEFT" if side_left_cam_id == "CAM-03" else "RIGHT"

    if args.mode in ["dual", "headless", "web"] and (getattr(args, "ws_stream", False) or side_left_source_path):
        print(f"[*] Menginisialisasi Side Inspector ke-2 ({side_left_cam_id} {side_left_side_name})...")
        try:
            side_left_inspector = SideConveyorInspector(
                camera_id=side_left_cam_id,
                side=side_left_side_name,
                device=args.device,
            )
            if InspectionDBBridge is not None:
                try:
                    side_left_db_bridge = InspectionDBBridge(
                        enabled=args.enable_db,
                        camera_id=side_left_cam_id,
                    )
                except Exception as e_db2:
                    print(f"[WARN] Gagal inisialisasi {side_left_cam_id} DB Bridge: {e_db2}")

            cand_left_src = side_left_source_path or side_source_path or DEFAULT_SIDE_VIDEO
            if cand_left_src and not is_usable_source(cand_left_src):
                cand_left_src = resolve_source_path(cand_left_src) or DEFAULT_SIDE_VIDEO
            cap_side_left, is_cam_side_left, side_left_source_path = create_capture(cand_left_src)
            if (not cap_side_left.isOpened()) and DEFAULT_SIDE_VIDEO and os.path.exists(DEFAULT_SIDE_VIDEO):
                side_left_source_path = DEFAULT_SIDE_VIDEO
                cap_side_left, is_cam_side_left, side_left_source_path = create_capture(side_left_source_path)
            if side_left_inspector is not None and side_left_source_path:
                if getattr(side_left_inspector, "auto_preset_mode", True):
                    det_p_l = side_left_inspector.resolve_preset_from_source(side_left_source_path)
                    side_left_inspector.set_preset(det_p_l, verbose=True)
        except Exception as e_sl:
            print(f"[WARN] Gagal inisialisasi {side_left_cam_id} Side Inspector: {e_sl}")
            side_left_inspector = None

    # Inisialisasi Dedicated CUDA Streams & cuDNN Benchmark untuk mencegah GPU serialization
    top_stream = None
    side_stream = None
    side_left_stream = None
    try:
        import torch
        if torch.cuda.is_available() and args.device != "cpu":
            torch.backends.cudnn.benchmark = True
            top_stream = torch.cuda.Stream()
            side_stream = torch.cuda.Stream()
            if side_left_inspector is not None:
                side_left_stream = torch.cuda.Stream()
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
        if side_left_inspector is not None and cap_side_left is not None and cap_side_left.isOpened():
            ret_w, frame_w = cap_side_left.read()
            if ret_w and frame_w is not None:
                side_left_inspector.process_frame(frame_w, render_display=False)
                if not is_cam_side_left:
                    cap_side_left.set(cv2.CAP_PROP_POS_FRAMES, 0)
        print("[*] Pre-warmup selesai. GPU siap pada performa steady-state.")
    except Exception as e_wm:
        print(f"[WARN] Pre-warmup skipped: {e_wm}")

    # Decoupled GUI Rendering Pacing: Pacing tampilan visual GUI desktop tanpa mencekik throughput AI
    gui_fps = getattr(args, "gui_fps", 15.0)
    gui_interval = (1.0 / gui_fps) if (gui_fps and gui_fps > 0) else 0.0

    def _get_cap_fps(cap_obj, is_cam_flag):
        if cap_obj is not None and not is_cam_flag:
            v = float(cap_obj.get(cv2.CAP_PROP_FPS) or 0.0)
            if 5.0 <= v <= 120.0:
                return v
        return 30.0

    total_f_top = int(cap_top.get(cv2.CAP_PROP_FRAME_COUNT)) if (cap_top and not is_cam_top) else 0
    top_video_fps = _get_cap_fps(cap_top, is_cam_top)
    total_f_side = int(cap_side.get(cv2.CAP_PROP_FRAME_COUNT)) if (cap_side and not is_cam_side) else 0
    side_video_fps = _get_cap_fps(cap_side, is_cam_side)

    init_start_frame = None
    if getattr(args, "start_time", None) and cap_top is not None and not is_cam_top and total_f_top > 0:
        s_sec = parse_time_to_seconds(args.start_time)
        if s_sec is not None:
            init_start_frame = int(np.clip(round(s_sec * top_video_fps), 0, total_f_top - 1))
            cap_top.set(cv2.CAP_PROP_POS_FRAMES, init_start_frame)
            print(f"[*] [START-TIME] Melompat awal ke {format_seconds_mmss(s_sec)} (Frame {init_start_frame}/{total_f_top})")
            if cap_side is not None and not is_cam_side and total_f_side > 0:
                side_init_f = int(np.clip(round(s_sec * side_video_fps), 0, total_f_side - 1))
                cap_side.set(cv2.CAP_PROP_POS_FRAMES, side_init_f)

    nav_lock = threading.Lock()
    nav_state = {
        "paused": False,
        "step_once": False,
        "speed": max(1, int(getattr(args, "speed", 1) or 1)),
        "skip_idle": bool(getattr(args, "skip_idle", False)),
        "seek_target_frame": None,
        "side_seek_target_frame": None,
        "cur_pos_frame": init_start_frame or 0,
        "is_scanning_idle": False,
        "motion_energy": 0.0,
    }

    def _request_seek_frames(target_f, sync_side=True):
        if is_cam_top or total_f_top <= 0:
            return
        clamped_f = int(max(0, min(total_f_top - 1, int(target_f))))
        target_sec = float(clamped_f) / max(1e-5, top_video_fps)
        with nav_lock:
            nav_state["seek_target_frame"] = clamped_f
            nav_state["cur_pos_frame"] = clamped_f
            if sync_side and cap_side is not None and not is_cam_side and total_f_side > 0:
                nav_state["side_seek_target_frame"] = int(
                    max(0, min(total_f_side - 1, round(target_sec * side_video_fps)))
                )

    top_state = {
        "display": None,
        "disp_id": 0,
        "fps": 0.0,
        "latency": 0.0,
        "status": "STARTING",
        "frames": 0,
        "pos_frame": init_start_frame or 0,
        "raw_frame": None,
        "inspect_res": None,
        "has_new": False,
    }
    side_state = {
        "display": None,
        "disp_id": 0,
        "fps": 0.0,
        "latency": 0.0,
        "status": "STARTING",
        "frames": 0,
        "has_new": False,
    }

    # Worker Thread 1: CAM-01 Top Inspector
    def _top_loop():
        nonlocal cap_top, top_source_path, is_cam_top, total_f_top, top_video_fps
        t_prev = time.time()
        fps_hist = []
        dataset_last_rib_no = 0
        dataset_previous = None
        idle_skipper = IdleConveyorSkipper(enabled=bool(getattr(args, "skip_idle", False)))
        prev_sched_state = "ACTIVE"
        prewarm_frames_done = 0

        while not stop_event.is_set():
            t_loop_top0 = time.perf_counter()
            try:
                if cap_top is None or not cap_top.isOpened():
                    time.sleep(0.02)
                    continue

                with nav_lock:
                    req_seek = nav_state["seek_target_frame"]
                    if req_seek is not None:
                        nav_state["seek_target_frame"] = None
                    is_paused = nav_state["paused"]
                    do_step = nav_state["step_once"]
                    if do_step:
                        nav_state["step_once"] = False
                    speed_mult = nav_state["speed"]
                    idle_skipper.enabled = bool(nav_state["skip_idle"])

                force_render_seek = False
                if req_seek is not None and not is_cam_top:
                    cap_top.set(cv2.CAP_PROP_POS_FRAMES, req_seek)
                    top_inspector.reset_active_tracking()
                    idle_skipper.reset_scan_state()
                    with nav_lock:
                        nav_state["is_scanning_idle"] = False
                    force_render_seek = True

                if is_paused and not do_step and not force_render_seek:
                    time.sleep(0.02)
                    continue

                # Fast-forward frame skipping saat speed > 1x
                if speed_mult > 1 and not is_cam_top and not is_paused and not force_render_seek:
                    for _ in range(speed_mult - 1):
                        if not cap_top.grab():
                            break
                    if speed_mult >= 4:
                        top_inspector.reset_active_tracking()

                ret, frame = cap_top.read()
                if not ret or frame is None:
                    if not is_cam_top:
                        finalize_pending_rib_dataset(top_dataset_recorder, dataset_previous)
                        if top_dataset_recorder is not None:
                            top_dataset_recorder.start_new_session()
                        dataset_last_rib_no = 0
                        dataset_previous = None
                        top_inspector.reset_counters()
                        cap_top.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        idle_skipper.reset_scan_state()
                        ret, frame = cap_top.read()
                        if not ret or frame is None:
                            time.sleep(0.02)
                            continue
                    else:
                        time.sleep(0.01)
                        continue

                cur_pos_f = int(cap_top.get(cv2.CAP_PROP_POS_FRAMES)) if not is_cam_top else top_state["frames"] + 1

                # Smart Auto-Skip Conveyor Diam ([I] / --skip-idle)
                if idle_skipper.enabled and not is_cam_top and not is_paused and not force_render_seek:
                    skip_action, skip_info = idle_skipper.evaluate(cur_pos_f, frame, top_video_fps, speed_mult)
                    if skip_action == "SKIP_FORWARD":
                        with nav_lock:
                            nav_state["motion_energy"] = skip_info["motion_energy"]
                            nav_state["is_scanning_idle"] = True
                        for _ in range(skip_info["skip_frames"]):
                            if not cap_top.grab():
                                break
                        cur_pos_f = int(cap_top.get(cv2.CAP_PROP_POS_FRAMES))
                        with top_lock:
                            top_state["pos_frame"] = cur_pos_f
                            top_state["raw_frame"] = frame
                            top_state["inspect_res"] = skip_info["stub_result"]
                            top_state["status"] = "SKIP"
                            top_state["has_new"] = True
                        with nav_lock:
                            nav_state["cur_pos_frame"] = cur_pos_f
                        continue
                    elif skip_action == "REWIND_RESUME":
                        rewind_f = skip_info["rewind_frame"]
                        cap_top.set(cv2.CAP_PROP_POS_FRAMES, rewind_f)
                        top_inspector.reset_active_tracking()
                        with nav_lock:
                            nav_state["motion_energy"] = skip_info["motion_energy"]
                            nav_state["is_scanning_idle"] = False
                            nav_state["cur_pos_frame"] = rewind_f
                        print(f"\n[*] [AUTO-SKIP] Gerakan conveyor terdeteksi! Kembali ke 1x di {format_seconds_mmss(rewind_f / top_video_fps)} (Frame {rewind_f}).\n")
                        continue
                    elif "motion_energy" in skip_info:
                        with nav_lock:
                            nav_state["motion_energy"] = skip_info["motion_energy"]

                sched_snap = scheduler.evaluate()
                sched_state = sched_snap.state
                if sched_state != prev_sched_state:
                    if prev_sched_state == "ACTIVE":
                        finalize_pending_rib_dataset(top_dataset_recorder, dataset_previous)
                        dataset_previous = None
                        top_inspector.reset_active_tracking()
                    elif sched_state == "ACTIVE":
                        top_inspector.reset_active_tracking()
                    if sched_state != "PRE_WARMUP":
                        prewarm_frames_done = 0
                    prev_sched_state = sched_state

                if sched_state != "ACTIVE":
                    if sched_state == "PRE_WARMUP" and prewarm_frames_done < 2:
                        if top_stream is not None:
                            with torch.cuda.stream(top_stream), torch.inference_mode():
                                top_inspector.inspect(frame)
                        else:
                            top_inspector.inspect(frame)
                        top_inspector.reset_active_tracking()
                        prewarm_frames_done += 1

                    stub_res = make_standby_top_result(sched_state, sched_snap.hud_badge())
                    t_end = time.time()
                    dt = t_end - t_prev
                    t_prev = t_end
                    fps_hist.append(1.0 / dt if dt > 0 else float(scheduler.config.standby_fps))
                    if len(fps_hist) > 30:
                        fps_hist.pop(0)
                    avg_f = sum(fps_hist) / len(fps_hist)
                    with nav_lock:
                        nav_state["cur_pos_frame"] = cur_pos_f
                    with top_lock:
                        top_state["frames"] += 1
                        top_state["pos_frame"] = cur_pos_f
                        top_state["fps"] = avg_f
                        top_state["latency"] = 0.0
                        top_state["status"] = sched_state
                        top_state["raw_frame"] = frame
                        top_state["inspect_res"] = stub_res
                        top_state["disp_id"] += 1
                        top_state["has_new"] = True
                    if ws_hub is not None:
                        ws_hub.publish(
                            "CAM-01",
                            frame,
                            stub_res,
                            avg_f,
                            inspector_meta={
                                "slat_no": getattr(top_inspector.slat_inspector, "slat_count", None),
                                "rib_no": getattr(top_inspector.rib_inspector, "rib_count", None),
                                "schedule_state": sched_state,
                            },
                        )
                    rem_stby = standby_interval - (time.perf_counter() - t_loop_top0)
                    if rem_stby > 0.001:
                        time.sleep(rem_stby)
                    continue

                t_infer_start = time.time()
                if top_stream is not None:
                    with torch.cuda.stream(top_stream):
                        with torch.inference_mode():
                            inspect_res = top_inspector.inspect(frame)
                else:
                    inspect_res = top_inspector.inspect(frame)
                t_end = time.time()
                lat = (t_end - t_infer_start) * 1000.0
                if scheduler.config.motion_quota:
                    scheduler.record_motion_sample(bool(inspect_res.get("is_conveyor_moving", True)))

                dt = t_end - t_prev
                t_prev = t_end
                c_fps = 1.0 / dt if dt > 0 else 30.0
                fps_hist.append(c_fps)
                if len(fps_hist) > 30:
                    fps_hist.pop(0)
                avg_f = sum(fps_hist) / len(fps_hist)

                # Rekam ke database SQLite dan kirim notifikasi MS Teams menggunakan Best-Shot Buffer jika DB aktif & NG
                is_ng = inspect_res.get("status") == "NG"
                record_top_ng_if_enabled(
                    top_inspector,
                    frame,
                    inspect_res,
                    avg_f,
                    top_state["frames"] + 1,
                    total_f_top,
                    getattr(top_inspector.slat_inspector, "slat_count", None),
                    getattr(top_inspector.rib_inspector, "rib_count", None),
                )

                # Satu bukti raw/annotated untuk setiap rib yang sudah final.
                # Vonis detektor tidak diubah; perekam hanya mengarsipkan hasilnya.
                dataset_last_rib_no, dataset_previous = record_rib_dataset_step(
                    top_dataset_recorder,
                    top_inspector,
                    frame,
                    inspect_res,
                    avg_f,
                    top_state["frames"] + 1,
                    total_f_top,
                    dataset_last_rib_no,
                    dataset_previous,
                )

                # Push data ke top_state untuk dirender asinkron oleh Main GUI Thread (Zero-Overhead di Thread AI)
                with nav_lock:
                    nav_state["cur_pos_frame"] = cur_pos_f
                with top_lock:
                    top_state["frames"] += 1
                    top_state["pos_frame"] = cur_pos_f
                    top_state["fps"] = avg_f
                    top_state["latency"] = lat
                    top_state["status"] = inspect_res.get("status", "WAIT")
                    top_state["raw_frame"] = frame
                    top_state["inspect_res"] = inspect_res
                    top_state["disp_id"] += 1
                    top_state["has_new"] = True

                if ws_hub is not None:
                    leg_frame = None
                    if ws_server is not None and ws_server.hud_mode == "legacy" and ws_server.client_count > 0:
                        leg_frame = top_inspector.render_combined_dashboard(
                            frame=frame,
                            inspect_result=inspect_res,
                            fps=avg_f,
                            frame_idx=cur_pos_f,
                            total_frames=total_f_top,
                        )
                    ws_hub.publish(
                        "CAM-01",
                        frame,
                        inspect_res,
                        avg_f,
                        inspector_meta={
                            "slat_no": getattr(top_inspector.slat_inspector, "slat_count", None),
                            "rib_no": getattr(top_inspector.rib_inspector, "rib_count", None),
                        },
                        legacy_frame=leg_frame,
                    )
            except Exception as e:
                import traceback
                print(f"\n[WARN Thread-TopInspector] Transient exception caught (auto-recovered): {e}")
                traceback.print_exc()
                time.sleep(0.01)

        finalize_pending_rib_dataset(top_dataset_recorder, dataset_previous)
        if top_dataset_recorder is not None:
            top_dataset_recorder.close()

    # Worker Thread 2: Side Inspector
    def _side_loop():
        nonlocal cap_side, side_source_path, is_cam_side
        t_prev = time.time()
        fps_hist = []
        last_render_side = 0.0
        prev_sched_state = "ACTIVE"
        prewarm_frames_done = 0

        while not stop_event.is_set():
            t_loop_side0 = time.perf_counter()
            try:
                if cap_side is None or not cap_side.isOpened():
                    time.sleep(0.02)
                    continue

                with nav_lock:
                    s_seek = nav_state.get("side_seek_target_frame")
                    if s_seek is not None:
                        nav_state["side_seek_target_frame"] = None
                    s_paused = nav_state.get("paused", False)

                if s_seek is not None and not is_cam_side:
                    cap_side.set(cv2.CAP_PROP_POS_FRAMES, s_seek)
                    if side_inspector and hasattr(side_inspector, "reset_session"):
                        side_inspector.reset_session()

                if s_paused and s_seek is None:
                    time.sleep(0.02)
                    continue

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

                sched_snap_s = scheduler.evaluate()
                sched_state_s = sched_snap_s.state
                if sched_state_s != prev_sched_state:
                    if hasattr(side_inspector, "reset_active_tracking"):
                        side_inspector.reset_active_tracking()
                    if sched_state_s != "PRE_WARMUP":
                        prewarm_frames_done = 0
                    prev_sched_state = sched_state_s

                if sched_state_s != "ACTIVE":
                    if sched_state_s == "PRE_WARMUP" and prewarm_frames_done < 2:
                        if side_stream is not None:
                            with torch.cuda.stream(side_stream), torch.inference_mode():
                                side_inspector.process_frame(frame, render_display=False)
                        else:
                            side_inspector.process_frame(frame, render_display=False)
                        if hasattr(side_inspector, "reset_active_tracking"):
                            side_inspector.reset_active_tracking()
                        prewarm_frames_done += 1

                    stub_side = make_standby_side_payload(args.side_cam_id, args.side_name, sched_state_s)
                    t_end = time.time()
                    dt = t_end - t_prev
                    t_prev = t_end
                    fps_hist.append(1.0 / dt if dt > 0 else float(scheduler.config.standby_fps))
                    if len(fps_hist) > 30:
                        fps_hist.pop(0)
                    avg_f = sum(fps_hist) / len(fps_hist)
                    with side_lock:
                        side_state["frames"] += 1
                        side_state["fps"] = avg_f
                        side_state["latency"] = 0.0
                        side_state["status"] = sched_state_s
                        if args.mode not in ("headless", "web") and (t_end - last_render_side) >= gui_interval:
                            side_state["display"] = draw_schedule_standby_overlay(frame, sched_snap_s.hud_badge())
                            side_state["disp_id"] += 1
                            side_state["has_new"] = True
                            last_render_side = t_end
                    if ws_hub is not None:
                        ws_hub.publish(
                            args.side_cam_id,
                            frame,
                            stub_side,
                            avg_f,
                            inspector_meta={"schedule_state": sched_state_s},
                        )
                    rem_stby = standby_interval - (time.perf_counter() - t_loop_side0)
                    if rem_stby > 0.001:
                        time.sleep(rem_stby)
                    continue

                t_infer_start = time.time()
                ws_needs_legacy = bool(
                    ws_server is not None and ws_server.hud_mode == "legacy" and ws_server.client_count > 0
                )
                need_side_render = (
                    (args.mode not in ("headless", "web") and (t_infer_start - last_render_side) >= gui_interval)
                    or (side_db_bridge is not None and side_db_bridge.is_enabled())
                    or ws_needs_legacy
                )
                if side_stream is not None:
                    with torch.cuda.stream(side_stream):
                        with torch.inference_mode():
                            display_frame, payload = side_inspector.process_frame(frame, render_display=need_side_render)
                else:
                    display_frame, payload = side_inspector.process_frame(frame, render_display=need_side_render)
                if need_side_render:
                    last_render_side = t_infer_start
                t_end = time.time()
                lat = (t_end - t_infer_start) * 1000.0

                dt = t_end - t_prev
                t_prev = t_end
                c_fps = 1.0 / dt if dt > 0 else 30.0
                fps_hist.append(c_fps)
                if len(fps_hist) > 30:
                    fps_hist.pop(0)
                avg_f = sum(fps_hist) / len(fps_hist)

                # Rekam ke database SQLite & notifikasi Teams jika terdeteksi Side NG (Single-Capture per Slat)
                is_side_ng = payload.get("overall_result") == "NG"
                if side_db_bridge and side_db_bridge.is_enabled():
                    if payload.get("should_emit", False) and is_side_ng:
                        emit_raw = payload.get("emit_raw_frame", frame)
                        emit_ann = payload.get("emit_ann_frame", display_frame)
                        side_db_bridge.emit_side_inspection(
                            frame_raw=emit_raw,
                            frame_ann=emit_ann,
                            side_payload=payload
                        )

                # Push data ke side_state untuk visualisasi asinkron oleh Main GUI Thread
                with side_lock:
                    side_state["frames"] += 1
                    side_state["fps"] = avg_f
                    side_state["latency"] = lat
                    side_state["status"] = payload.get("overall_result", "WAIT")
                    if need_side_render:
                        side_state["display"] = display_frame
                        side_state["disp_id"] += 1
                        side_state["has_new"] = True

                if ws_hub is not None:
                    ws_hub.publish(
                        args.side_cam_id,
                        frame,
                        payload,
                        avg_f,
                        inspector_meta={
                            "master_roi": getattr(side_inspector, "master_roi", None),
                            "polygon_pts": getattr(side_inspector, "polygon_pts", None),
                        },
                        legacy_frame=display_frame if ws_needs_legacy else None,
                    )
            except Exception as e:
                import traceback
                print(f"\n[WARN Thread-SideInspector] Transient exception caught (auto-recovered): {e}")
                traceback.print_exc()
                time.sleep(0.01)

    # Worker Thread 3 (Opsional): Side Left Inspector (CAM-03) untuk 3-Camera WebSocket Live Stream
    def _side_left_loop():
        t_prev = time.time()
        fps_hist = []
        prev_sched_state = "ACTIVE"
        prewarm_frames_done = 0
        while not stop_event.is_set():
            t_loop_sl0 = time.perf_counter()
            try:
                if cap_side_left is None or not cap_side_left.isOpened():
                    time.sleep(0.02)
                    continue
                with nav_lock:
                    s_paused = nav_state.get("paused", False)
                if s_paused:
                    time.sleep(0.02)
                    continue
                ret, frame = cap_side_left.read()
                if not ret or frame is None:
                    if not is_cam_side_left:
                        cap_side_left.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        side_left_inspector.reset_session()
                        ret, frame = cap_side_left.read()
                        if not ret or frame is None:
                            time.sleep(0.02)
                            continue
                    else:
                        time.sleep(0.01)
                        continue

                sched_snap_l = scheduler.evaluate()
                sched_state_l = sched_snap_l.state
                if sched_state_l != prev_sched_state:
                    if hasattr(side_left_inspector, "reset_active_tracking"):
                        side_left_inspector.reset_active_tracking()
                    if sched_state_l != "PRE_WARMUP":
                        prewarm_frames_done = 0
                    prev_sched_state = sched_state_l

                if sched_state_l != "ACTIVE":
                    if sched_state_l == "PRE_WARMUP" and prewarm_frames_done < 2:
                        if side_left_stream is not None:
                            with torch.cuda.stream(side_left_stream), torch.inference_mode():
                                side_left_inspector.process_frame(frame, render_display=False)
                        else:
                            side_left_inspector.process_frame(frame, render_display=False)
                        if hasattr(side_left_inspector, "reset_active_tracking"):
                            side_left_inspector.reset_active_tracking()
                        prewarm_frames_done += 1
                    stub_l = make_standby_side_payload(side_left_cam_id, side_left_side_name, sched_state_l)
                    t_end = time.time()
                    dt = t_end - t_prev
                    t_prev = t_end
                    fps_hist.append(1.0 / dt if dt > 0 else float(scheduler.config.standby_fps))
                    if len(fps_hist) > 30:
                        fps_hist.pop(0)
                    avg_f = sum(fps_hist) / len(fps_hist)
                    if ws_hub is not None:
                        ws_hub.publish(
                            side_left_cam_id,
                            frame,
                            stub_l,
                            avg_f,
                            inspector_meta={"schedule_state": sched_state_l},
                        )
                    rem_stby = standby_interval - (time.perf_counter() - t_loop_sl0)
                    if rem_stby > 0.001:
                        time.sleep(rem_stby)
                    continue

                ws_needs_legacy = bool(
                    ws_server is not None and ws_server.hud_mode == "legacy" and ws_server.client_count > 0
                )
                need_render = bool(
                    (side_left_db_bridge is not None and side_left_db_bridge.is_enabled()) or ws_needs_legacy
                )
                if side_left_stream is not None:
                    with torch.cuda.stream(side_left_stream), torch.inference_mode():
                        disp_l, payload_l = side_left_inspector.process_frame(frame, render_display=need_render)
                else:
                    disp_l, payload_l = side_left_inspector.process_frame(frame, render_display=need_render)

                t_end = time.time()
                dt = t_end - t_prev
                t_prev = t_end
                fps_hist.append(1.0 / dt if dt > 0 else 30.0)
                if len(fps_hist) > 30:
                    fps_hist.pop(0)
                avg_f = sum(fps_hist) / len(fps_hist)

                if side_left_db_bridge and side_left_db_bridge.is_enabled():
                    if payload_l.get("should_emit", False) and payload_l.get("overall_result") == "NG":
                        side_left_db_bridge.emit_side_inspection(
                            frame_raw=payload_l.get("emit_raw_frame", frame),
                            frame_ann=payload_l.get("emit_ann_frame", disp_l),
                            side_payload=payload_l,
                        )

                if ws_hub is not None:
                    ws_hub.publish(
                        side_left_cam_id,
                        frame,
                        payload_l,
                        avg_f,
                        inspector_meta={
                            "master_roi": getattr(side_left_inspector, "master_roi", None),
                            "polygon_pts": getattr(side_left_inspector, "polygon_pts", None),
                        },
                        legacy_frame=disp_l if ws_needs_legacy else None,
                    )
            except Exception as e:
                import traceback
                print(f"\n[WARN Thread-{side_left_cam_id}] Transient exception caught (auto-recovered): {e}")
                traceback.print_exc()
                time.sleep(0.01)

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
    if side_left_inspector is not None:
        th_side_left = threading.Thread(target=_side_left_loop, daemon=True, name=f"Thread-{side_left_cam_id}")
        th_side_left.start()
        threads.append(th_side_left)

    # Windows GUI Setup
    win_top = "CAM-01 TOP (SLAT & RIB) | [SPACE]: Pause | [,/.]: +-10f | [;/']: +-60f | [[/]]: +-10s | [-/=]: +-60s | [X]: Speed | [I]: Skip-Diam | [Z]: Loncat | [Q]: Keluar"
    win_side = f"{args.side_cam_id} SIDE [{args.side_cam_id} {args.side_name}] | [2]/[O]: Video | [W]: Setup | [M]: Preset | [D]: DB | [Q]: Keluar"
    trackbar_name = "Seek (s)"
    trackbar_state = {"attached": False, "ignore": False}

    def _on_seek_trackbar(sec_val):
        if trackbar_state["ignore"]:
            return
        _request_seek_frames(int(round(float(sec_val) * top_video_fps)), sync_side=True)

    def _sync_top_trackbar_setup():
        if args.mode in ("headless", "web") or top_inspector is None:
            return
        if (not is_cam_top) and total_f_top > 0:
            max_sec = max(1, int(total_f_top / max(1e-5, top_video_fps)))
            if not trackbar_state["attached"]:
                cv2.createTrackbar(trackbar_name, win_top, 0, max_sec, _on_seek_trackbar)
                trackbar_state["attached"] = True
            else:
                try:
                    cv2.setTrackbarMax(trackbar_name, win_top, max_sec)
                except Exception:
                    pass

    def _refresh_top_video_meta():
        nonlocal total_f_top, top_video_fps
        total_f_top = int(cap_top.get(cv2.CAP_PROP_FRAME_COUNT)) if (cap_top and not is_cam_top) else 0
        top_video_fps = _get_cap_fps(cap_top, is_cam_top)
        with nav_lock:
            nav_state["cur_pos_frame"] = 0
            nav_state["seek_target_frame"] = None
            nav_state["is_scanning_idle"] = False
        _sync_top_trackbar_setup()

    if args.mode not in ("headless", "web"):
        if top_inspector is not None:
            cv2.namedWindow(win_top, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win_top, 920, 518)
            cv2.moveWindow(win_top, 20, 40)
            _sync_top_trackbar_setup()
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
    last_gui_render = 0.0
    restart_request = None

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

            # Render GUI OpenCV di Main Thread (Decoupled GUI Rendering Sejati - Bebas GIL dari Thread AI)
            key = -1
            if args.mode not in ("headless", "web"):
                if (now - last_gui_render) >= gui_interval:
                    last_gui_render = now
                    if top_inspector is not None:
                        with top_lock:
                            has_new_top = top_state.get("has_new", False)
                            raw_top = top_state.get("raw_frame")
                            res_top = top_state.get("inspect_res")
                            fps_top = top_state.get("fps", 0.0)
                            f_pos_top = top_state.get("pos_frame", top_state.get("frames", 0))
                            top_state["has_new"] = False

                        with nav_lock:
                            nav_paused = nav_state["paused"]
                            nav_speed = nav_state["speed"]
                            nav_skip_idle = nav_state["skip_idle"]
                            nav_scanning = nav_state["is_scanning_idle"]

                        if (has_new_top or nav_paused) and raw_top is not None and res_top is not None:
                            if trackbar_state["attached"] and (not is_cam_top) and total_f_top > 0:
                                cur_sec = int(max(0, round(float(f_pos_top) / max(1e-5, top_video_fps))))
                                trackbar_state["ignore"] = True
                                try:
                                    cv2.setTrackbarPos(trackbar_name, win_top, cur_sec)
                                except Exception:
                                    pass
                                trackbar_state["ignore"] = False

                            time_str = ""
                            if (not is_cam_top) and total_f_top > 0:
                                time_str = (
                                    f"{format_seconds_mmss(f_pos_top / top_video_fps)}/"
                                    f"{format_seconds_mmss(total_f_top / top_video_fps)}"
                                )
                            if str(res_top.get("status", "")).upper() in ("STANDBY", "PRE_WARMUP"):
                                sched_badge_top = (res_top.get("slat_res") or {}).get("eval_mode") or f"SCHED: {res_top.get('status')}"
                                hud_top = draw_schedule_standby_overlay(raw_top, sched_badge_top)
                            else:
                                hud_top = top_inspector.render_combined_dashboard(
                                    frame=raw_top,
                                    inspect_result=res_top,
                                    fps=fps_top,
                                    is_paused=nav_paused,
                                    frame_idx=f_pos_top if (not is_cam_top and total_f_top > 0) else top_state.get("frames", 0),
                                    total_frames=total_f_top,
                                    video_time_str=time_str,
                                    speed_label=format_speed_label(nav_speed, nav_skip_idle, nav_scanning),
                                )
                            cv2.imshow(win_top, hud_top)
                            if getattr(top_inspector, "show_rib_unwarp_window", False):
                                rib_unwarp = getattr(top_inspector.rib_inspector, "last_rib_unwarped_annotated", None)
                                if rib_unwarp is not None:
                                    cv2.imshow("CAM-01 TOP: PERSPECTIVE TOP-DOWN CONVEYOR (BIRD'S-EYE)", rib_unwarp)

                    if side_inspector is not None:
                        with side_lock:
                            has_new_side = side_state.get("has_new", False)
                            disp_side = side_state.get("display")
                            fps_side = side_state.get("fps", 0.0)
                            lat_side = side_state.get("latency", 0.0)
                            side_state["has_new"] = False

                        if has_new_side and disp_side is not None:
                            # Render banner pada disp_side di Main Thread
                            h_d, w_d = disp_side.shape[:2]
                            s_ok = getattr(side_inspector, 'slat_ok_count', getattr(side_inspector, 'stats', {}).get('ok_count', 0))
                            s_ng = getattr(side_inspector, 'slat_ng_count', getattr(side_inspector, 'stats', {}).get('ng_count', 0))
                            s_cyc = getattr(side_inspector, 'slat_cycle_count', 0)
                            qc_txt = f"Slat #{s_cyc} [OK:{s_ok} NG:{s_ng}]"
                            cur_p_label = getattr(side_inspector, 'preset', 'opsi_1').upper() if side_inspector else 'OPSI_1'
                            banner = f"FPS: {fps_side:4.1f} | Lat: {lat_side:4.1f}ms | [W]: Setup | [2/O]: Video | [M]: {cur_p_label} | {qc_txt} | [Q]: Keluar"
                            cv2.rectangle(disp_side, (10, h_d - 65), (w_d - 10, h_d - 10), (18, 18, 18), -1)
                            cv2.rectangle(disp_side, (10, h_d - 65), (w_d - 10, h_d - 10), (0, 220, 255), 2, cv2.LINE_AA)
                            cv2.putText(disp_side, banner, (25, h_d - 24),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.76, (0, 255, 255), 2, cv2.LINE_AA)
                            cv2.imshow(win_side, disp_side)

                    key = cv2.waitKey(1) & 0xFF
                else:
                    # Pompa event loop window & tidur sejenak untuk melepaskan GIL sepenuhnya ke thread AI
                    key = cv2.waitKey(1) & 0xFF
                    time.sleep(0.005)
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
                new_db_val = bool(top_st or side_st)
                if side_left_db_bridge is not None:
                    side_left_db_bridge.set_enabled(new_db_val)
                args.enable_db = new_db_val
                if ws_server is not None:
                    ws_server.set_db_enabled(new_db_val, notify_callback=False)
                st_text = "AKTIF (RECORDING & NOTIFIKASI TEAMS)" if new_db_val else "MATI (DRY-RUN BENCHMARK)"
                print(f"\n[DB SWITCH] Status Database Logging beralih ke: {st_text}")
                print(f"[*] [CAM-01 TOP] SAKLAR DATABASE & TEAMS: {st_text}")
                print(f"[*] [{args.side_cam_id} SIDE] SAKLAR DATABASE & TEAMS: {st_text}\n")
            elif key in [ord("r"), ord("R")]:
                if top_inspector:
                    top_inspector.reset_counters()
                if side_inspector:
                    side_inspector.reset_session()
                print("\n[*] Counter QC & Cache Deduplikasi kedua kamera di-reset ke 0.\n")
            elif key in [ord("s"), ord("S")]:
                snap_cyc = scheduler.cycle_override()
                if ws_server is not None:
                    ws_server.set_schedule_state(snap_cyc.to_ws_dict(), broadcast=True)
                print(f"\n[*] [SCHEDULE OVERRIDE] Mode -> {snap_cyc.override.upper()} | {snap_cyc.hud_badge()}\n")
            elif key in [ord("p"), ord("P")]:
                if top_inspector and hasattr(top_inspector, "toggle_profile"):
                    with top_lock:
                        top_inspector.toggle_profile()
            elif key in [ord("h"), ord("H")]:
                if top_inspector and hasattr(top_inspector, "toggle_threshold_source"):
                    with top_lock:
                        top_inspector.toggle_threshold_source()
            elif key in [ord("u"), ord("U"), ord("0")]:
                if top_inspector and hasattr(top_inspector, "restore_default_profile"):
                    with top_lock:
                        top_inspector.restore_default_profile()
            elif key in [ord("k"), ord("K"), ord("a"), ord("A")]:
                if top_inspector and hasattr(top_inspector, "auto_align_rails_instant"):
                    with top_lock:
                        cur_raw = top_state.get("raw_frame")
                    if cur_raw is not None:
                        top_inspector.auto_align_rails_instant(cur_raw)
            elif key in [ord("g"), ord("G")]:
                if top_inspector and hasattr(top_inspector, "toggle_alignment_guides"):
                    with top_lock:
                        st = top_inspector.toggle_alignment_guides()
                    mode_str = "AKTIF (RETICLE ACUAN EMAS)" if st else "NON-AKTIF"
                    print(f"\n[*] [CAM-01 TOP] 📐 GARIS REFERENSI REL & SUMBU KONVEYOR: {mode_str}\n")
            elif key in [ord("l"), ord("L")]:
                print("\n[*] [CAM-01 TOP] 🚀 Membuka Wizard Kalibrasi Interaktif Rel Ganda...")
                calib_script = os.path.join(BASE_DIR, "Vision_Engine_Core", "02_Top_Inspection_Slat_Rib", "src", "calibrate_dual_rails.py")
                subprocess.Popen([sys.executable, calib_script, "--source", str(top_source_path)])
            elif key in [ord("v"), ord("V")]:
                new_top = cycle_next_video(top_source_path)
                if new_top and str(new_top) != str(top_source_path):
                    with top_lock:
                        if cap_top is not None:
                            cap_top.release()
                        cap_top, is_cam_top, top_source_path = create_capture(new_top)
                    _refresh_top_video_meta()
                    if top_inspector:
                        top_inspector.reset_counters()
            elif key == 32:  # [SPACE]: Pause / Play
                with nav_lock:
                    nav_state["paused"] = not nav_state["paused"]
                    p_now = nav_state["paused"]
                print(f"\n[*] [VIDEO] {'PAUSED' if p_now else 'RESUMED'}\n")
            elif key in [ord("n"), ord("N")]:
                with nav_lock:
                    nav_state["paused"] = True
                    nav_state["step_once"] = True
            elif key in SEEK_KEY_SPEC:
                if not is_cam_top and total_f_top > 0:
                    with nav_lock:
                        cur_f = nav_state["cur_pos_frame"]
                    tgt_f, lbl = compute_seek_target_from_key(key, cur_f, total_f_top, top_video_fps)
                    if tgt_f is not None:
                        _request_seek_frames(tgt_f, sync_side=True)
                        print(f"\n[*] [SEEK {lbl}] -> Frame {tgt_f}/{total_f_top} ({format_seconds_mmss(tgt_f / top_video_fps)})\n")
            elif key in [ord("x"), ord("X"), ord(">")]:
                with nav_lock:
                    next_sp = next_speed_step(nav_state["speed"])
                    nav_state["speed"] = next_sp
                print(f"\n[*] [SPEED] Kecepatan pemutaran diubah ke: {next_sp}x\n")
            elif key in [ord("i"), ord("I")]:
                with nav_lock:
                    nav_state["skip_idle"] = not nav_state["skip_idle"]
                    nav_state["is_scanning_idle"] = False
                    sk_on = nav_state["skip_idle"]
                print(f"\n[*] [AUTO-SKIP DIAM] Mode skip conveyor diam: {'AKTIF' if sk_on else 'NON-AKTIF'}\n")
            elif key in [ord("z"), ord("Z")]:
                if not is_cam_top and total_f_top > 0:
                    with nav_lock:
                        cur_s = nav_state["cur_pos_frame"] / max(1e-5, top_video_fps)
                    tot_s = total_f_top / max(1e-5, top_video_fps)
                    entered = prompt_jump_time_dialog(format_seconds_mmss(cur_s), format_seconds_mmss(tot_s))
                    tgt_sec = parse_time_to_seconds(entered) if entered else None
                    if tgt_sec is not None:
                        _request_seek_frames(int(round(tgt_sec * top_video_fps)), sync_side=True)
                        print(f"\n[*] [JUMP-TIME] Melompat ke waktu {format_seconds_mmss(tgt_sec)}\n")
            elif key in [ord("m"), ord("M")]:
                if side_inspector:
                    cur_p = getattr(side_inspector, "preset", "opsi_1")
                    next_p = "opsi_2" if cur_p == "opsi_1" else "opsi_1"
                    side_inspector.set_preset(next_p, verbose=True)
                    side_inspector.reset_session()
                    print(f"\n[*] [{args.side_cam_id} SIDE] Manual Switch Preset: {cur_p.upper()} -> {next_p.upper()}\n")
            elif key in [ord("b"), ord("B")]:
                new_side = cycle_next_video(side_source_path)
                if new_side and str(new_side) != str(side_source_path):
                    with side_lock:
                        if cap_side is not None:
                            cap_side.release()
                        cap_side, is_cam_side, side_source_path = create_capture(new_side)
                    if side_inspector:
                        if getattr(side_inspector, "auto_preset_mode", True):
                            det_p = side_inspector.resolve_preset_from_source(side_source_path)
                            side_inspector.set_preset(det_p, verbose=True)
                        side_inspector.reset_session()
            elif key in [ord("c"), ord("C")]:
                new_cam = cycle_next_camera(top_source_path)
                if new_cam and str(new_cam) != str(top_source_path):
                    with top_lock:
                        if cap_top is not None:
                            cap_top.release()
                        cap_top, is_cam_top, top_source_path = create_capture(new_cam)
                    if top_inspector:
                        top_inspector.reset_counters()
            elif key in [ord("t"), ord("T"), ord("e"), ord("E")]:
                if top_inspector and hasattr(top_inspector, "rib_inspector"):
                    with top_lock:
                        new_eng = top_inspector.rib_inspector.toggle_tracker_engine()
                    print(f"\n[*] [CAM-01 TOP] 🔄 SAKELAR TRACKER ENGINE SIRIP: [{new_eng}]\n")
            elif key in [ord("j"), ord("J")]:
                if top_inspector and hasattr(top_inspector, "toggle_rib_unwarp_window"):
                    with top_lock:
                        st = top_inspector.toggle_rib_unwarp_window()
                    mode_str = "DITAMPILKAN" if st else "DISEMBUNYIKAN"
                    print(f"\n[*] [CAM-01 TOP] 🔍 JENDELA UNWARP BIRD'S-EYE: {mode_str}\n")
            elif key in [ord("b"), ord("B")]:
                if top_inspector and hasattr(top_inspector, "toggle_simulate_bending"):
                    with top_lock:
                        st = top_inspector.toggle_simulate_bending()
                    mode_str = "AKTIF (+20mm BENGKOK - NG TEST)" if st else "NON-AKTIF (NORMAL)"
                    print(f"\n[*] [CAM-01 TOP] ⚡ SIMULASI DEFEK SIRIP BENGKOK: {mode_str}\n")
            elif key in [ord("1"), ord("f"), ord("F")]:
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
                    _refresh_top_video_meta()
                    if top_inspector:
                        top_inspector.reset_counters()
                    print(f"[*] CAM-01 Top berhasil beralih ke video: {os.path.basename(str(top_source_path))}")
            elif key in [ord("2"), ord("o"), ord("O")]:
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
                        if getattr(side_inspector, "auto_preset_mode", True):
                            det_p = side_inspector.resolve_preset_from_source(side_source_path)
                            side_inspector.set_preset(det_p, verbose=True)
                        side_inspector.reset_session()
                    print(f"[*] {args.side_cam_id} Side berhasil beralih ke video: {os.path.basename(str(side_source_path))}")
            elif key in [ord("w"), ord("W")]:
                print("\n[*] Membuka Dialog Pengaturan Sumber Dual Kamera, Mode & Jadwal...")
                new_top, new_side, new_mode, new_sched = select_dual_camera_setup_dialog(
                    default_top=top_source_path,
                    default_side=side_source_path,
                    side_cam_id=args.side_cam_id,
                    default_mode=args.mode,
                    include_mode=True,
                    default_schedule=build_schedule_dict_from_args(args),
                    include_schedule=True,
                )
                if new_sched:
                    new_cfg = apply_schedule_dict_to_args(args, new_sched)
                    snap_new = scheduler.reconfigure(new_cfg)
                    standby_interval = 1.0 / max(1.0, float(scheduler.config.standby_fps or 20.0))
                    if ws_server is not None:
                        ws_server.set_schedule_state(snap_new.to_ws_dict(), broadcast=True)
                    print(f"[*] [JADWAL DETEKSI] Diperbarui langsung via [W] -> {snap_new.hud_badge()}")

                norm_mode = str(new_mode).strip().lower() if new_mode else args.mode
                if norm_mode in ("dual", "opencv", "web", "headless") and norm_mode != args.mode:
                    restart_request = {
                        "top_source": new_top or top_source_path,
                        "side_source": new_side or side_source_path,
                        "mode": norm_mode,
                        "schedule": new_sched,
                    }
                    print(f"[*] Mempersiapkan peralihan mode '{args.mode}' -> '{norm_mode}'...")
                    stop_event.set()
                    break

                if new_top and str(new_top) != str(top_source_path):
                    with top_lock:
                        if cap_top is not None:
                            cap_top.release()
                        cap_top, is_cam_top, top_source_path = create_capture(new_top)
                    _refresh_top_video_meta()
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
                        if getattr(side_inspector, "auto_preset_mode", True):
                            det_p = side_inspector.resolve_preset_from_source(side_source_path)
                            side_inspector.set_preset(det_p, verbose=True)
                        side_inspector.reset_session()
                    s_label = f"Webcam #{side_source_path}" if is_cam_side else os.path.basename(str(side_source_path))
                    print(f"[*] {args.side_cam_id} Side beralih ke: {s_label}")

            # Cetak Telemetri Terminal Setiap 1 Detik
            if now - last_print >= 1.0:
                last_print = now
                cur_sched_snap = scheduler.evaluate(force_refresh=True)
                if ws_server is not None:
                    ws_server.set_schedule_state(cur_sched_snap.to_ws_dict(), broadcast=True)
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
        if cap_side_left is not None:
            cap_side_left.release()
        if ws_server is not None:
            ws_server.stop()
        if top_inspector and hasattr(top_inspector, "db_bridge") and top_inspector.db_bridge:
            top_inspector.db_bridge.close()
        if side_db_bridge is not None:
            side_db_bridge.close()
        if side_left_db_bridge is not None:
            side_left_db_bridge.close()
        if args.mode not in ("headless", "web"):
            cv2.destroyAllWindows()
        if csv_file is not None:
            csv_file.close()

    if restart_request is not None:
        return restart_request

    # Executive Summary Report
    total_elapsed = (time.time() - t0) if t0 is not None else 0.0
    f_top = top_state["frames"]
    f_side = side_state["frames"]
    avg_top_fps = f_top / total_elapsed if total_elapsed > 0 else 0.0
    avg_side_fps = f_side / total_elapsed if total_elapsed > 0 else 0.0

    print("\n" + "=" * 86)
    print("📊 LAPORAN AKHIR HASIL BENCHMARK MULTI-KAMERA")
    print("=" * 86)
    print(f"  • Total Waktu Pengujian     : {total_elapsed:.1f} detik")
    if args.mode in ["dual", "opencv", "headless", "web", "top-only"]:
        print(f"  • CAM-01 Top (Slat & Rib)   : {f_top} frames ({avg_top_fps:.1f} rata-rata FPS)")
    if args.mode in ["dual", "opencv", "headless", "web", "side-only"]:
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
# ======================================================================================
# HYBRID 2-PROCESS ENGINE (CAM-01 BEBAS GIL DI PROSES 1 + CAM-02/03 & WS DI PROSES 2)
# ======================================================================================
def _run_hybrid_engine(args):
    """
    Runner Hybrid 2-Proses:
    - Proses 1 (Terisolasi): Menjalankan CAM-01 Top Inspection bebas dari Python GIL agar FPS tetap ~20+.
    - Proses 2 (Koordinator): Menjalankan CAM-02 RH + CAM-03 LH (berbagi VRAM/GIL ringan) dan
      melayani WebSocket Stream Server (:8765) ke Frontend Web.
    """
    import cv2
    import numpy as np
    import queue as _queue
    import threading

    if SIDE_SRC_DIR not in sys.path:
        sys.path.insert(0, SIDE_SRC_DIR)
    _ensure_db_bridge_on_path()

    from side_inspector_pipeline import SideConveyorInspector
    try:
        from db_bridge import InspectionDBBridge
    except ImportError:
        InspectionDBBridge = None

    configure_windows_process_performance()
    configure_worker_thread_budget(cv2_threads=2, torch_threads=2)
    try:
        import torch
        if torch.cuda.is_available():
            try:
                torch.cuda.set_per_process_memory_fraction(0.45, device=0)
            except Exception:
                pass
            torch.cuda.empty_cache()
    except Exception:
        pass

    stop_event = mp.Event()
    fps_top = mp.Value("d", 0.0)
    frames_top = mp.Value("i", 0)
    lat_top = mp.Value("d", 0.0)
    stat_top = mp.Array("c", 64)
    stat_top.value = b"STARTING"

    ws_clients_val = mp.Value("i", 0)
    ws_overlay_val = mp.Value("i", 1)
    ws_db_val = mp.Value("i", 1 if args.enable_db else 0)
    scheduler = DetectionScheduler.from_env_and_args(args)
    init_sched_snap = scheduler.evaluate(force_refresh=True)
    ws_sched_state_val = mp.Value("i", SCHED_STATE_TO_CODE[init_sched_snap.state])
    standby_interval = 1.0 / max(1.0, float(scheduler.config.standby_fps or 20.0))
    ws_top_queue = mp.Queue(maxsize=2) if getattr(args, "ws_stream", False) else None
    side_workers = []

    ws_hub = None
    ws_server = None
    if getattr(args, "ws_stream", False):
        try:
            from ws_stream_server import CameraFrameHub, WebSocketStreamServer

            ws_hub = CameraFrameHub()

            def _on_clients(cnt):
                ws_clients_val.value = int(cnt)

            def _on_overlay(val):
                ws_overlay_val.value = 1 if val else 0

            def _on_db_change(val):
                flag = bool(val)
                args.enable_db = flag
                ws_db_val.value = 1 if flag else 0
                for sw in side_workers:
                    if sw.get("dbb") is not None:
                        sw["dbb"].set_enabled(flag)
                st_lbl = "AKTIF (RECORDING & NOTIFIKASI TEAMS)" if flag else "MATI (DRY-RUN)"
                print(f"\n[DB SWITCH via Web FE] Status Database Logging (Hybrid 3-Kam): {st_lbl}\n")

            def _on_sched_override_hybrid(ovr: str):
                snap_o = scheduler.set_override(ovr)
                ws_sched_state_val.value = SCHED_STATE_TO_CODE[snap_o.state]
                print(f"\n[SCHEDULE via Web FE] Hybrid Override -> '{snap_o.override}' | {snap_o.hud_badge()}\n")
                return snap_o.to_ws_dict()

            ws_server = WebSocketStreamServer(
                ws_hub,
                port=int(getattr(args, "ws_port", 8765) or 8765),
                max_fps=float(getattr(args, "ws_fps", 0.0) or 0.0),
                hud_mode=str(getattr(args, "ws_hud_mode", "clean") or "clean"),
                on_client_count_change=_on_clients,
                on_overlay_change=_on_overlay,
                initial_db_enabled=bool(args.enable_db),
                on_db_enabled_change=_on_db_change,
                initial_schedule=init_sched_snap.to_ws_dict(),
                on_schedule_override=_on_sched_override_hybrid,
            )
            if not ws_server.start():
                ws_server = None
        except Exception as e_ws:
            print(f"[WARN WS-STREAM] Gagal menginisialisasi WebSocket server (hybrid): {e_ws}")
            ws_hub = None
            ws_server = None

    p_top = None
    if args.mode in ["dual", "opencv", "headless", "web", "top-only"]:
        p_top = mp.Process(
            target=_run_top_worker,
            args=(
                args.top_source,
                "dual" if args.mode in ("dual", "opencv") else "headless",
                args.rib_mode,
                args.device,
                args.enable_db,
                stop_event,
                fps_top,
                frames_top,
                lat_top,
                stat_top,
                args.gui_fps,
            ),
            kwargs={
                "start_time": getattr(args, "start_time", None),
                "ws_top_queue": ws_top_queue,
                "ws_clients_val": ws_clients_val,
                "ws_overlay_val": ws_overlay_val,
                "ws_db_val": ws_db_val,
                "ws_hud_mode": getattr(args, "ws_hud_mode", "clean"),
                "ws_fps": float(getattr(args, "ws_fps", 0.0) or 0.0),
                "ws_sched_state_val": ws_sched_state_val,
                "standby_fps": float(scheduler.config.standby_fps),
            },
            name="Worker-TopInspector-Hybrid",
        )
        p_top.start()
        print(f"[*] Sub-proses [TOP CAM-01] terisolasi di-spawn | PID: {p_top.pid}")

    # Inisialisasi CAM-02 & CAM-03 di proses Koordinator
    side_lock = threading.Lock()
    side_state = {"fps": 0.0, "latency": 0.0, "status": "STARTING", "frames": 0, "display": None, "has_new": False}

    def _init_side(cam_id, side_name, raw_src):
        ins = SideConveyorInspector(camera_id=cam_id, side=side_name, device=args.device)
        dbb = InspectionDBBridge(enabled=args.enable_db, camera_id=cam_id) if InspectionDBBridge else None
        cand = raw_src or args.side_source or DEFAULT_SIDE_VIDEO
        if cand and not is_usable_source(cand):
            cand = resolve_source_path(cand) or DEFAULT_SIDE_VIDEO
        cap, is_cam, resolved = create_capture(cand)
        if (not cap.isOpened()) and DEFAULT_SIDE_VIDEO and os.path.exists(DEFAULT_SIDE_VIDEO):
            cap, is_cam, resolved = create_capture(DEFAULT_SIDE_VIDEO)
        if getattr(ins, "auto_preset_mode", True) and resolved:
            ins.set_preset(ins.resolve_preset_from_source(resolved), verbose=True)
        seek_frame = 0
        if getattr(args, "start_time", None) and (not is_cam) and cap.isOpened():
            s_sec = parse_time_to_seconds(args.start_time)
            tot_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            v_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            if s_sec is not None and tot_f > 0:
                seek_frame = int(np.clip(round(s_sec * (v_fps if 5 <= v_fps <= 120 else 30.0)), 0, tot_f - 1))
                cap.set(cv2.CAP_PROP_POS_FRAMES, seek_frame)
        if cap.isOpened():
            ok_w, fw = cap.read()
            if ok_w and fw is not None:
                ins.process_frame(fw, render_display=False)
                if not is_cam:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, seek_frame)
        return {"cam_id": cam_id, "ins": ins, "dbb": dbb, "cap": cap, "is_cam": is_cam}

    if args.mode in ["dual", "opencv", "headless", "web", "side-only"]:
        print(f"[*] Menginisialisasi Side Inspector ({args.side_cam_id} {args.side_name})...")
        side_workers.append(_init_side(args.side_cam_id, args.side_name, args.side_source))
        side_left_src = getattr(args, "side_left_source", None)
        if args.mode in ["dual", "headless", "web"] and (getattr(args, "ws_stream", False) or side_left_src):
            cam3_id = "CAM-03" if str(args.side_cam_id).upper() != "CAM-03" else "CAM-02"
            cam3_side = "LEFT" if cam3_id == "CAM-03" else "RIGHT"
            print(f"[*] Menginisialisasi Side Inspector ke-2 ({cam3_id} {cam3_side})...")
            side_workers.append(_init_side(cam3_id, cam3_side, side_left_src or args.side_source))

    cuda_streams = []
    try:
        import torch
        if torch.cuda.is_available() and args.device != "cpu":
            torch.backends.cudnn.benchmark = True
            cuda_streams = [torch.cuda.Stream() for _ in side_workers]
    except Exception:
        cuda_streams = [None for _ in side_workers]

    gui_interval = (1.0 / args.gui_fps) if (args.gui_fps and args.gui_fps > 0) else 0.0
    side_max_fps = float(getattr(args, "side_max_fps", 0.0) or 0.0)
    side_interval = (1.0 / side_max_fps) if side_max_fps > 0.0 else 0.0

    def _side_thread(idx, w):
        cap = w["cap"]
        ins = w["ins"]
        dbb = w["dbb"]
        is_cam = w["is_cam"]
        cam_id = w["cam_id"]
        cstream = cuda_streams[idx] if idx < len(cuda_streams) else None
        is_primary = (idx == 0)
        t_prev = time.time()
        fps_hist = []
        last_disp = 0.0
        prev_sched_state = "ACTIVE"
        prewarm_frames_done = 0

        while not stop_event.is_set():
            t_loop0 = time.perf_counter()
            try:
                if cap is None or not cap.isOpened():
                    time.sleep(0.02)
                    continue
                ret, frame = cap.read()
                if not ret or frame is None:
                    if not is_cam:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ins.reset_session()
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            time.sleep(0.02)
                            continue
                    else:
                        time.sleep(0.01)
                        continue

                sched_snap_h = scheduler.evaluate()
                sched_state_h = sched_snap_h.state
                if sched_state_h != prev_sched_state:
                    if hasattr(ins, "reset_active_tracking"):
                        ins.reset_active_tracking()
                    if sched_state_h != "PRE_WARMUP":
                        prewarm_frames_done = 0
                    prev_sched_state = sched_state_h

                if sched_state_h != "ACTIVE":
                    if sched_state_h == "PRE_WARMUP" and prewarm_frames_done < 2:
                        if cstream is not None:
                            import torch
                            with torch.cuda.stream(cstream), torch.inference_mode():
                                ins.process_frame(frame, render_display=False)
                        else:
                            ins.process_frame(frame, render_display=False)
                        if hasattr(ins, "reset_active_tracking"):
                            ins.reset_active_tracking()
                        prewarm_frames_done += 1
                    stub_h = make_standby_side_payload(cam_id, getattr(ins, "side", "RIGHT"), sched_state_h)
                    t_end = time.time()
                    dt = t_end - t_prev
                    t_prev = t_end
                    fps_hist.append(1.0 / dt if dt > 0 else float(scheduler.config.standby_fps))
                    if len(fps_hist) > 30:
                        fps_hist.pop(0)
                    avg_f = sum(fps_hist) / len(fps_hist)
                    if is_primary:
                        with side_lock:
                            side_state["frames"] += 1
                            side_state["fps"] = avg_f
                            side_state["latency"] = 0.0
                            side_state["status"] = sched_state_h
                            if args.mode in ("dual", "opencv") and (t_end - last_disp) >= gui_interval:
                                side_state["display"] = draw_schedule_standby_overlay(frame, sched_snap_h.hud_badge())
                                side_state["has_new"] = True
                                last_disp = t_end
                    if ws_hub is not None:
                        ws_hub.publish(
                            cam_id,
                            frame,
                            stub_h,
                            avg_f,
                            inspector_meta={"schedule_state": sched_state_h},
                        )
                    rem_stby = standby_interval - (time.perf_counter() - t_loop0)
                    if rem_stby > 0.001:
                        time.sleep(rem_stby)
                    continue

                t_inf0 = time.time()
                ws_legacy = bool(ws_server is not None and ws_server.hud_mode == "legacy" and ws_server.client_count > 0)
                need_render = (
                    (is_primary and args.mode in ("dual", "opencv") and (t_inf0 - last_disp) >= gui_interval)
                    or (dbb is not None and dbb.is_enabled())
                    or ws_legacy
                )
                if cstream is not None:
                    import torch
                    with torch.cuda.stream(cstream), torch.inference_mode():
                        disp, payload = ins.process_frame(frame, render_display=need_render)
                else:
                    disp, payload = ins.process_frame(frame, render_display=need_render)
                if need_render:
                    last_disp = t_inf0
                t_end = time.time()
                lat_ms = (t_end - t_inf0) * 1000.0
                dt = t_end - t_prev
                t_prev = t_end
                fps_hist.append(1.0 / dt if dt > 0 else 30.0)
                if len(fps_hist) > 30:
                    fps_hist.pop(0)
                avg_f = sum(fps_hist) / len(fps_hist)

                if dbb is not None and dbb.is_enabled() and payload.get("should_emit", False) and payload.get("overall_result") == "NG":
                    dbb.emit_side_inspection(
                        frame_raw=payload.get("emit_raw_frame", frame),
                        frame_ann=payload.get("emit_ann_frame", disp),
                        side_payload=payload,
                    )

                if is_primary:
                    with side_lock:
                        side_state["frames"] += 1
                        side_state["fps"] = avg_f
                        side_state["latency"] = lat_ms
                        side_state["status"] = payload.get("overall_result", "WAIT")
                        if args.mode in ("dual", "opencv") and need_render:
                            side_state["display"] = disp
                            side_state["has_new"] = True

                if ws_hub is not None:
                    ws_hub.publish(
                        cam_id,
                        frame,
                        payload,
                        avg_f,
                        inspector_meta={
                            "master_roi": getattr(ins, "master_roi", None),
                            "polygon_pts": getattr(ins, "polygon_pts", None),
                        },
                        legacy_frame=disp if ws_legacy else None,
                    )

                if side_interval > 0.0:
                    rem = side_interval - (time.perf_counter() - t_loop0)
                    if rem > 0.001:
                        time.sleep(rem)
            except Exception as e_sw:
                print(f"[WARN Hybrid-{cam_id}] {e_sw}")
                time.sleep(0.01)

    def _top_ws_relay():
        while not stop_event.is_set():
            if ws_top_queue is None or ws_hub is None:
                time.sleep(0.05)
                continue
            try:
                meta_t, jpg_b = ws_top_queue.get(timeout=0.1)
                ws_hub.publish_encoded("CAM-01", meta_t, jpg_b)
            except _queue.Empty:
                continue
            except Exception:
                time.sleep(0.01)

    threads = []
    if ws_top_queue is not None:
        th_rel = threading.Thread(target=_top_ws_relay, daemon=True, name="Thread-TopWSRelay")
        th_rel.start()
        threads.append(th_rel)
    for idx, w in enumerate(side_workers):
        th_s = threading.Thread(target=_side_thread, args=(idx, w), daemon=True, name=f"Thread-Hybrid-{w['cam_id']}")
        th_s.start()
        threads.append(th_s)

    win_side = f"{args.side_cam_id} SIDE [HYBRID] | [W]: Setup | [Q]: Keluar"
    if args.mode in ("dual", "opencv") and side_workers:
        cv2.namedWindow(win_side, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_side, 920, 518)
        cv2.moveWindow(win_side, 960, 40)

    print("-" * 86)
    print(f"{'WAKTU':^8} | {'CAM-01 TOP (FPS)':^18} | {f'{args.side_cam_id} SIDE (FPS)':^18} | {'COMBINED FPS':^14} | {'STATUS':^16}")
    print("-" * 86)

    csv_file = None
    if args.record_csv:
        try:
            csv_file = open(args.record_csv, "w", encoding="utf-8")
            csv_file.write("Timestamp,Elapsed_Sec,Top_FPS,Top_Latency_ms,Top_Frames,Side_FPS,Side_Latency_ms,Side_Frames,Combined_Throughput_FPS\n")
        except Exception:
            csv_file = None

    t0 = None
    last_print = 0.0
    t_spawn = time.time()
    restart_request = None

    try:
        while not stop_event.is_set():
            now = time.time()
            if t0 is None:
                if frames_top.value > 0 or side_state["frames"] > 0:
                    t0 = now
                else:
                    if p_top is not None and not p_top.is_alive():
                        stop_event.set()
                        break
                    time.sleep(0.05)
                    continue

            elapsed = now - t0
            if args.duration > 0 and elapsed >= args.duration:
                print(f"\n[*] Durasi benchmark {args.duration}s tercapai. Menghentikan...")
                stop_event.set()
                break

            if p_top is not None and not p_top.is_alive() and not stop_event.is_set():
                stop_event.set()
                break

            if args.mode in ("dual", "opencv") and side_workers:
                with side_lock:
                    has_new = side_state["has_new"]
                    disp_s = side_state["display"]
                    side_state["has_new"] = False
                if has_new and disp_s is not None:
                    cv2.imshow(win_side, disp_s)
                k_hyb = cv2.waitKey(1) & 0xFF
                if k_hyb in [ord("q"), ord("Q"), 27]:
                    stop_event.set()
                    break
                elif k_hyb in [ord("s"), ord("S")]:
                    snap_cyc = scheduler.cycle_override()
                    ws_sched_state_val.value = SCHED_STATE_TO_CODE[snap_cyc.state]
                    if ws_server is not None:
                        ws_server.set_schedule_state(snap_cyc.to_ws_dict(), broadcast=True)
                    print(f"\n[*] [SCHEDULE OVERRIDE] Hybrid -> {snap_cyc.override.upper()} | {snap_cyc.hud_badge()}\n")
                elif k_hyb in [ord("w"), ord("W")]:
                    new_top, new_side, new_mode, new_sched = select_dual_camera_setup_dialog(
                        default_top=args.top_source,
                        default_side=args.side_source,
                        side_cam_id=args.side_cam_id,
                        default_mode=args.mode,
                        include_mode=True,
                        default_schedule=build_schedule_dict_from_args(args),
                        include_schedule=True,
                    )
                    if new_sched:
                        new_cfg = apply_schedule_dict_to_args(args, new_sched)
                        snap_new = scheduler.reconfigure(new_cfg)
                        ws_sched_state_val.value = SCHED_STATE_TO_CODE[snap_new.state]
                        standby_interval = 1.0 / max(1.0, float(scheduler.config.standby_fps or 20.0))
                        if ws_server is not None:
                            ws_server.set_schedule_state(snap_new.to_ws_dict(), broadcast=True)
                        print(f"[*] [JADWAL DETEKSI] Hybrid diperbarui langsung via [W] -> {snap_new.hud_badge()}")
                    norm_mode = str(new_mode).strip().lower() if new_mode else args.mode
                    if (
                        (norm_mode in ("dual", "opencv", "web", "headless") and norm_mode != args.mode)
                        or (new_top and str(new_top) != str(args.top_source))
                        or (new_side and str(new_side) != str(args.side_source))
                    ):
                        restart_request = {
                            "top_source": new_top or args.top_source,
                            "side_source": new_side or args.side_source,
                            "mode": norm_mode if norm_mode in ("dual", "opencv", "web", "headless") else args.mode,
                            "schedule": new_sched,
                        }
                        stop_event.set()
                        break
            else:
                time.sleep(0.05)

            if now - last_print >= 1.0:
                last_print = now
                cur_sched_h = scheduler.evaluate(force_refresh=True)
                ws_sched_state_val.value = SCHED_STATE_TO_CODE[cur_sched_h.state]
                if ws_server is not None:
                    ws_server.set_schedule_state(cur_sched_h.to_ws_dict(), broadcast=True)
                v_top_fps = fps_top.value
                v_top_lat = lat_top.value
                f_top_cnt = frames_top.value
                s_top = stat_top.value.decode("utf-8", errors="replace")
                with side_lock:
                    v_side_fps = side_state["fps"]
                    v_side_lat = side_state["latency"]
                    f_side_cnt = side_state["frames"]
                    s_side = side_state["status"]
                v_comb = v_top_fps + v_side_fps
                print(
                    f"{f'{int(elapsed):03d}s':^8} | "
                    f"{v_top_fps:5.1f} FPS ({v_top_lat:4.1f}ms) | "
                    f"{v_side_fps:5.1f} FPS ({v_side_lat:4.1f}ms) | "
                    f"{v_comb:6.1f} FPS   | "
                    f"{f'T:{s_top}|S:{s_side}':^16}"
                )
                if csv_file:
                    ts_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    csv_file.write(
                        f"{ts_now},{elapsed:.1f},{v_top_fps:.2f},{v_top_lat:.1f},{f_top_cnt},"
                        f"{v_side_fps:.2f},{v_side_lat:.1f},{f_side_cnt},{v_comb:.2f}\n"
                    )
                    csv_file.flush()
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()
        for th in threads:
            th.join(timeout=2.0)
        if p_top is not None:
            p_top.join(timeout=3.0)
            if p_top.is_alive():
                p_top.terminate()
        if ws_server is not None:
            ws_server.stop()
        for w in side_workers:
            if w["cap"] is not None:
                w["cap"].release()
            if w["dbb"] is not None:
                w["dbb"].close()
        if args.mode in ("dual", "opencv"):
            cv2.destroyAllWindows()
        if csv_file is not None:
            csv_file.close()

    if restart_request is not None:
        return restart_request

    total_elapsed = (time.time() - t0) if t0 is not None else (time.time() - t_spawn)
    f_top = frames_top.value
    f_side = side_state["frames"]
    avg_top_fps = f_top / total_elapsed if total_elapsed > 0 else 0.0
    avg_side_fps = f_side / total_elapsed if total_elapsed > 0 else 0.0

    print("\n" + "=" * 86)
    print("📊 LAPORAN AKHIR HASIL BENCHMARK MULTI-KAMERA (HYBRID 2-PROSES)")
    print("=" * 86)
    print(f"  • Total Waktu Pengujian     : {total_elapsed:.1f} detik")
    if args.mode in ["dual", "opencv", "headless", "web", "top-only"]:
        print(f"  • CAM-01 Top (Slat & Rib)   : {f_top} frames ({avg_top_fps:.1f} rata-rata FPS)")
    if args.mode in ["dual", "opencv", "headless", "web", "side-only"]:
        print(f"  • {args.side_cam_id} Side (Bolt & Pin)  : {f_side} frames ({avg_side_fps:.1f} rata-rata FPS)")
    print(f"  • Throughput Gabungan       : {(avg_top_fps + avg_side_fps):.1f} frames/detik")
    print("=" * 86)
    print("[*] Selesai.\n")


def build_arg_parser():
    """Membangun ArgumentParser CLI runner (diekstrak agar dapat diuji unit test)."""
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
  [V]               : Beralih instan ke file video pengujian berikutnya (Zero-Crash, tanpa dialog)
  [B]               : Beralih instan ke video CAM-02 Side berikutnya (pada mode Threaded)
  [C]               : Beralih instan ke perangkat live webcam USB berikutnya (0 -> 1 -> ...)
  [W]               : Buka dialog visual pengaturan sumber dual kamera (Webcam / File Video)
  [T] / [E]         : Sakelar Tracker Engine Sirip TOP (LEGACY <-> TIER 1 VITERBI <-> TIER 2 POSE <-> TIER 3 UNWARP)
  [1] / [F]         : Buka File Explorer dialog untuk mengganti video CAM-01 TOP
  [2] / [O]         : Buka File Explorer dialog untuk mengganti video CAM-02 SIDE
  [D]               : Saklar Database Logging & Notifikasi Teams (ON / OFF)
  [R]               : Reset counter QC & cache deduplikasi ke 0
  [K]               : Jalankan Kalibrasi Otomatis Sudut & Lebar Konveyor (60 frame / ~3 detik)
  [P]               : Beralih Profil Parameter (DEFAULT STABIL <-> HASIL KALIBRASI)
  [U] / [0]         : Pulihkan seketika ke Profil Default Stabil (Golden Baseline)
  [SPACE] / [N]     : Pause / Resume video, atau langkah +1 frame ([N])
  [,] / [.]         : Loncat cepat -10 frame / +10 frame
  [;] / [']         : Loncat cepat -60 frame / +60 frame
  [[] / []]         : Loncat waktu video -10 detik / +10 detik (juga Slider 'Seek (s)')
  [-] / [=]         : Loncat waktu video -60 detik / +60 detik
  [Z]               : Buka dialog Loncat ke Menit:Detik tertentu (MM:SS)
  [X] / [>]         : Putar cepat 1x -> 2x -> 4x -> 8x -> 16x
  [I]               : Auto-Skip segmen conveyor diam (ON / OFF)
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
    env_ws_stream = os.environ.get("ENABLE_WS_STREAM", "false").lower() in ("true", "1", "yes")
    try:
        env_ws_port = int(os.environ.get("WS_STREAM_PORT", "8765"))
    except ValueError:
        env_ws_port = 8765
    try:
        env_ws_fps = float(os.environ.get("WS_MAX_FPS", "0"))
    except ValueError:
        env_ws_fps = 0.0
    try:
        env_side_max_fps = float(os.environ.get("SIDE_MAX_FPS", "0"))
    except ValueError:
        env_side_max_fps = 0.0
    env_ws_hud_mode = os.environ.get("WS_HUD_MODE", "clean").strip().lower()
    if env_ws_hud_mode not in ("clean", "legacy"):
        env_ws_hud_mode = "clean"
    env_side_left_source = os.environ.get("SIDE_LEFT_CAMERA_SOURCE") or None

    parser.add_argument(
        "--mode",
        choices=["dual", "opencv", "headless", "web", "top-only", "side-only"],
        default=env_mode,
        help="Mode eksekusi: 'dual' (2 jendela OpenCV + WS opsional), 'opencv' (HUD OpenCV saja tanpa WebSocket), 'headless' (tanpa display), 'web' (Web Dashboard 3-kam + WebSocket hybrid), atau kamera tunggal."
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
    parser.add_argument(
        "--side-source",
        type=str,
        default=DEFAULT_SIDE_VIDEO,
        help="Path video, index webcam (0/1), atau ketik 'browse' untuk Side Bolt/Pin. Jika kosong, dialog pemilih sumber akan muncul."
    )
    parser.add_argument(
        "--side-left-source",
        type=str,
        default=env_side_left_source,
        help="Path video atau index webcam opsional untuk kamera samping ke-2 (CAM-03 LEFT) saat 3-camera WebSocket stream aktif."
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
        choices=["threads", "process", "hybrid"],
        default=env_engine,
        help="Arsitektur runner: 'threads' (1 proses), 'hybrid' (CAM-01 bebas GIL di proses terpisah + CAM-02/03 & WS), atau 'process' (multiprocessing penuh)."
    )
    try:
        env_bolt_conf = float(os.environ.get("BOLT_CONF_THRESHOLD", "0.55"))
    except ValueError:
        env_bolt_conf = 0.55
    try:
        env_bolt_iou = float(os.environ.get("BOLT_IOU_THRESHOLD", "0.35"))
    except ValueError:
        env_bolt_iou = 0.35
    parser.add_argument("--bolt-conf", type=float, default=env_bolt_conf, help=f"Ambang batas confidence deteksi baut/pin (default dari .env: {env_bolt_conf})")
    parser.add_argument("--bolt-iou", type=float, default=env_bolt_iou, help=f"Ambang batas NMS IoU deteksi baut/pin (default dari .env: {env_bolt_iou})")
    try:
        env_gui_fps = float(os.environ.get("GUI_DISPLAY_FPS", "15.0"))
    except ValueError:
        env_gui_fps = 15.0
    parser.add_argument(
        "--gui-fps",
        type=float,
        default=env_gui_fps,
        help=f"Target refresh rate tampilan visual GUI desktop (default: {env_gui_fps} FPS). Thread AI tetap berjalan di kecepatan penuh hardware (35+ FPS)."
    )
    parser.add_argument(
        "--start-time",
        type=str,
        default=None,
        help="Mulai pemutaran video pada waktu tertentu (contoh: '12:30', '1:05:00', atau '750' detik)."
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=1,
        choices=[1, 2, 4, 8, 16],
        help="Kecepatan awal pemutaran video fast-forward (default: 1x)."
    )
    parser.add_argument(
        "--skip-idle",
        action="store_true",
        help="Aktifkan mode Auto-Skip otomatis melewati bagian video ketika conveyor diam."
    )
    parser.add_argument(
        "--ws-stream",
        action="store_true",
        default=env_ws_stream,
        help="Aktifkan Binary WebSocket Live Stream 3 kamera ke Web Frontend (opt-in)."
    )
    parser.add_argument(
        "--ws-port",
        type=int,
        default=env_ws_port,
        help=f"Port WebSocket Live Stream (default: {env_ws_port})."
    )
    parser.add_argument(
        "--ws-fps",
        type=float,
        default=env_ws_fps,
        help=f"Batas maksimal FPS stream WebSocket per kamera (0 = uncapped, default: {env_ws_fps})."
    )
    parser.add_argument(
        "--side-max-fps",
        type=float,
        default=env_side_max_fps,
        help=f"Batas maksimal FPS pemrosesan kamera samping CAM-02/03 agar CPU stabil untuk CAM-01 (0 = uncapped, default: {env_side_max_fps})."
    )
    parser.add_argument(
        "--ws-hud-mode",
        choices=["clean", "legacy"],
        default=env_ws_hud_mode,
        help=f"Mode visual stream WebSocket: 'clean' (16:9 penuh) atau 'legacy' (HUD OpenCV lama). Default: {env_ws_hud_mode}."
    )
    env_sched_mode = os.environ.get("DETECTION_SCHEDULE_MODE", "continuous").strip().lower()
    if env_sched_mode not in ("continuous", "periodic", "slots"):
        env_sched_mode = "continuous"
    try:
        env_sched_interval = float(os.environ.get("SCHEDULE_INTERVAL_MIN", "60"))
    except ValueError:
        env_sched_interval = 60.0
    try:
        env_sched_active = float(os.environ.get("SCHEDULE_ACTIVE_MIN", "15"))
    except ValueError:
        env_sched_active = 15.0
    try:
        env_sched_anchor = float(os.environ.get("SCHEDULE_ANCHOR_MINUTE", "0"))
    except ValueError:
        env_sched_anchor = 0.0
    env_sched_slots = os.environ.get("SCHEDULE_SLOTS", "")
    try:
        env_sched_prewarm = float(os.environ.get("SCHEDULE_PREWARM_SEC", "2.0"))
    except ValueError:
        env_sched_prewarm = 2.0
    try:
        env_sched_stby_fps = float(os.environ.get("SCHEDULE_STANDBY_FPS", "20.0"))
    except ValueError:
        env_sched_stby_fps = 20.0
    env_sched_motion_quota = os.environ.get("SCHEDULE_MOTION_QUOTA", "false").lower() in ("true", "1", "yes")

    parser.add_argument(
        "--schedule-mode",
        choices=["continuous", "periodic", "slots"],
        default=env_sched_mode,
        help=f"Mode jadwal deteksi AI: 'continuous' (24/7), 'periodic' (mis. 15m tiap 60m), atau 'slots'. Default: {env_sched_mode}."
    )
    parser.add_argument(
        "--schedule-interval-min",
        type=float,
        default=env_sched_interval,
        help=f"Interval siklus periodik dalam menit (default: {env_sched_interval})."
    )
    parser.add_argument(
        "--schedule-active-min",
        type=float,
        default=env_sched_active,
        help=f"Durasi aktif deteksi AI per siklus dalam menit (default: {env_sched_active})."
    )
    parser.add_argument(
        "--schedule-anchor-min",
        type=float,
        default=env_sched_anchor,
        help=f"Menit jangkar dimulainya siklus periodik (0 = awal jam :00, default: {env_sched_anchor})."
    )
    parser.add_argument(
        "--schedule-slots",
        type=str,
        default=env_sched_slots,
        help="Daftar slot jam harian untuk --schedule-mode slots (contoh: '08:00-08:15,09:00-09:15')."
    )
    parser.add_argument(
        "--schedule-prewarm-sec",
        type=float,
        default=env_sched_prewarm,
        help=f"Durasi pre-warmup GPU sebelum jendela aktif dimulai dalam detik (default: {env_sched_prewarm})."
    )
    parser.add_argument(
        "--schedule-standby-fps",
        type=float,
        default=env_sched_stby_fps,
        help=f"Batas FPS live passthrough saat fase STANDBY (default: {env_sched_stby_fps})."
    )
    parser.add_argument(
        "--schedule-motion-quota",
        action="store_true",
        default=env_sched_motion_quota,
        help="Hitung durasi aktif hanya saat conveyor bergerak (opsional untuk mode periodic)."
    )
    return parser


def resolve_mode_defaults(args, argv=None):
    """
    Menyesuaikan default otomatis untuk mode khusus:
    - Mode 'web' otomatis mengaktifkan ws_stream=True, engine 'hybrid' (CAM-01 bebas GIL),
      serta pacing seimbang ws_fps=20 & side_max_fps=30 kecuali user menentukan flag CLI secara eksplisit.
    - Mode 'opencv' menonaktifkan WebSocket (ws_stream=False) agar murni menampilkan HUD OpenCV desktop.
    """
    raw_argv = list(argv) if argv is not None else []
    explicit_engine = any(a == "--engine" or a.startswith("--engine=") for a in raw_argv)
    explicit_ws_fps = any(a == "--ws-fps" or a.startswith("--ws-fps=") for a in raw_argv)
    explicit_side_fps = any(a == "--side-max-fps" or a.startswith("--side-max-fps=") for a in raw_argv)
    cur_mode = str(getattr(args, "mode", "")).lower()
    if cur_mode == "web":
        args.ws_stream = True
        if not explicit_engine:
            args.engine = "hybrid"
        if not explicit_ws_fps and float(getattr(args, "ws_fps", 0.0) or 0.0) <= 0.0:
            args.ws_fps = 20.0
        if not explicit_side_fps and float(getattr(args, "side_max_fps", 0.0) or 0.0) <= 0.0:
            args.side_max_fps = 30.0
    elif cur_mode == "opencv":
        args.ws_stream = False
    return args


def apply_setup_handoff(args, new_top=None, new_side=None, new_mode=None):
    """
    Menerapkan hasil dialog setup [W] (sumber kamera & mode operasional).
    Mengembalikan True jika mode berubah sehingga runner aktif perlu handoff/restart.
    """
    if new_top:
        args.top_source = str(new_top)
    if new_side:
        args.side_source = str(new_side)

    norm_new = str(new_mode).strip().lower() if new_mode else ""
    cur_mode = str(getattr(args, "mode", "")).strip().lower()
    if norm_new in ("dual", "opencv", "web", "headless", "top-only", "side-only") and norm_new != cur_mode:
        args.mode = norm_new
        if norm_new == "web":
            resolve_mode_defaults(args, argv=[])
        elif norm_new == "opencv":
            resolve_mode_defaults(args, argv=[])
            if getattr(args, "engine", "") == "hybrid":
                args.engine = "threads"
        elif norm_new == "dual":
            args.ws_stream = os.environ.get("ENABLE_WS_STREAM", "false").lower() in ("true", "1", "yes")
        return True
    return False


def main():
    if "--gui-dialog" in sys.argv:
        _handle_isolated_gui_dialog()
        return

    import signal
    def _sig_clean(sig, frame):
        print(f"\n[*] Sinyal ({sig}) diterima. Menutup pipeline secara bersih...")
        sys.exit(0)

    try:
        signal.signal(signal.SIGINT, _sig_clean)
        signal.signal(signal.SIGTERM, _sig_clean)
    except Exception:
        pass

    parser = build_arg_parser()

    args = resolve_mode_defaults(parser.parse_args(), sys.argv[1:])

    # Sinkronkan threshold baut ke os.environ agar sub-worker otomatis menggunakannya
    if args.bolt_conf is not None:
        os.environ["BOLT_CONF_THRESHOLD"] = str(args.bolt_conf)
    if args.bolt_iou is not None:
        os.environ["BOLT_IOU_THRESHOLD"] = str(args.bolt_iou)


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

    top_is_active = args.mode in ["dual", "opencv", "headless", "web", "top-only"]
    side_is_active = args.mode in ["dual", "opencv", "headless", "web", "side-only"]
    missing_top_source = top_is_active and not is_usable_source(args.top_source)
    missing_side_source = side_is_active and not is_usable_source(args.side_source)

    if args.select_webcam or args.browse or str(args.top_source).lower() == "browse" or str(args.side_source).lower() == "browse" or (missing_top_source and missing_side_source):
        print("\n[*] Menampilkan dialog pengaturan sumber kamera, mode & jadwal...")
        ch_top, ch_side, ch_mode, ch_sched = select_dual_camera_setup_dialog(
            default_top=args.top_source,
            default_side=args.side_source,
            side_cam_id=args.side_cam_id,
            default_mode=args.mode,
            include_mode=True,
            default_schedule=build_schedule_dict_from_args(args),
            include_schedule=True,
        )
        if ch_sched:
            apply_schedule_dict_to_args(args, ch_sched)
        apply_setup_handoff(args, ch_top, ch_side, ch_mode)
        top_is_active = args.mode in ["dual", "opencv", "headless", "web", "top-only"]
        side_is_active = args.mode in ["dual", "opencv", "headless", "web", "side-only"]
    else:
        if missing_top_source or args.select_top or str(args.top_source).lower() in ["select", "dialog", "file"]:
            ch_top = select_source_dialog(title="[CAM-01 TOP] Pilih Sumber Plat & Rib", default_source=args.top_source)
            if ch_top:
                args.top_source = ch_top
        if missing_side_source or args.select_side or str(args.side_source).lower() in ["select", "dialog", "file"]:
            ch_side = select_source_dialog(title=f"[{args.side_cam_id} SIDE] Pilih Sumber Baut & Pin", default_source=args.side_source)
            if ch_side:
                args.side_source = ch_side

    # Validasi fallback path jika path manual tidak ditemukan
    if not is_usable_source(args.top_source):
        resolved_top = resolve_source_path(args.top_source)
        if is_usable_source(resolved_top):
            args.top_source = resolved_top
        else:
            args.top_source = DEFAULT_TOP_VIDEO

    if not is_usable_source(args.side_source):
        resolved_side = resolve_source_path(args.side_source)
        if is_usable_source(resolved_side):
            args.side_source = resolved_side
        else:
            args.side_source = DEFAULT_SIDE_VIDEO

    if top_is_active and not is_usable_source(args.top_source):
        print("[!] CAM-01 Top belum memiliki sumber video/kamera yang valid. Program dibatalkan.")
        print("    Jalankan kembali dan pilih file video pada dialog, atau gunakan --top-source \"C:\\path\\video.mp4\".")
        return
    if side_is_active and not is_usable_source(args.side_source):
        print(f"[!] {args.side_cam_id} Side belum memiliki sumber video/kamera yang valid. Program dibatalkan.")
        print("    Jalankan kembali dan pilih file video pada dialog, atau gunakan --side-source \"C:\\path\\video.mp4\".")
        return

    def _print_runner_banner():
        print("=" * 86)
        print("🚀 MULTI-CAMERA INSPECTION BENCHMARK RUNNER")
        print("=" * 86)
        engine_lbl = {
            "threads": "THREADS (Hemat RAM 8GB, Berbagi VRAM Bersama)",
            "hybrid": "HYBRID 2-PROSES (CAM-01 Bebas GIL + CAM-02/03 & WebSocket)",
            "process": "PROCESS (Multiprocessing Terisolasi)",
        }.get(args.engine, args.engine.upper())
        print(f"[*] Engine Arsitektur  : {engine_lbl}")
        print(f"[*] Mode Operasional   : {args.mode.upper()}")
        if args.mode not in ("headless", "web"):
            print(f"[*] GUI Render Mode    : DECOUPLED ({args.gui_fps:.1f} FPS GUI Display Pacing, AI Full Speed 35+ FPS)")
        print(f"[*] CAM-01 Top Source  : {os.path.basename(str(args.top_source)) if os.path.exists(str(args.top_source)) else args.top_source}")
        print(f"[*] {args.side_cam_id} Side Source : {os.path.basename(str(args.side_source)) if os.path.exists(str(args.side_source)) else args.side_source} ({args.side_name})")
        print(f"[*] Rib Model Mode     : {args.rib_mode}")
        print(f"[*] Profil Parameter   : DEFAULT_STABIL (Awal Pabrik - Tekan [P] untuk Kalibrasi)")
        print(f"[*] Durasi Uji Coba    : {'Tak Terbatas (Tekan [Q] / Ctrl+C)' if args.duration <= 0 else f'{args.duration} detik'}")
        print(f"[*] Database Logging   : {'AKTIF [RECORDING & TEAMS NOTIFIER]' if args.enable_db else 'OFF (DRY-RUN BENCHMARK - Tekan [D] di jendela untuk aktifkan)'}")
        sched_preview = DetectionScheduler.from_env_and_args(args)
        if sched_preview.is_scheduled:
            snap_p = sched_preview.evaluate(force_refresh=True)
            print(f"[*] Jadwal Deteksi AI  : {sched_preview.config.mode.upper()} ({sched_preview.config.active_min:g}m / {sched_preview.config.interval_min:g}m) | Saat ini: {snap_p.hud_badge()}")
        else:
            print("[*] Jadwal Deteksi AI  : CONTINUOUS (24/7 — Tekan [S] untuk Manual Override)")
        print("=" * 86)

    # 1. Jalankan runner threads/hybrid dengan dukungan handoff mode dinamis dari tombol [W]
    while args.engine in ("threads", "hybrid"):
        _print_runner_banner()
        configure_windows_process_performance()
        if args.engine == "threads":
            restart_req = _run_threaded_engine(args)
        else:
            restart_req = _run_hybrid_engine(args)
        if not restart_req:
            return
        if restart_req.get("schedule"):
            apply_schedule_dict_to_args(args, restart_req.get("schedule"))
        apply_setup_handoff(
            args,
            restart_req.get("top_source"),
            restart_req.get("side_source"),
            restart_req.get("mode"),
        )
        print(f"\n[*] Handoff Setup [W] selesai -> Mode: {args.mode.upper()} | Engine: {args.engine.upper()}")

    # 2. Fallback engine 'process' penuh
    _print_runner_banner()
    configure_windows_process_performance()

    # Inisialisasi Shared Memory untuk Telemetri Antar-Proses
    stop_event = mp.Event()
    proc_scheduler = DetectionScheduler.from_env_and_args(args)
    ws_sched_state_proc = mp.Value("i", SCHED_STATE_TO_CODE[proc_scheduler.evaluate(force_refresh=True).state])
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
    if args.mode in ["dual", "opencv", "headless", "web", "top-only"]:
        p_top = mp.Process(
            target=_run_top_worker,
            args=(
                args.top_source,
                "headless" if args.mode in ("headless", "web") else "dual",
                args.rib_mode,
                args.device,
                args.enable_db,
                stop_event,
                fps_top,
                frames_top,
                lat_top,
                stat_top,
                args.gui_fps
            ),
            kwargs={
                "ws_sched_state_val": ws_sched_state_proc,
                "standby_fps": float(proc_scheduler.config.standby_fps),
            },
            name="Worker-TopInspector"
        )
        processes.append(("TOP", p_top))

    # 2. Spawn Worker Side
    if args.mode in ["dual", "opencv", "headless", "web", "side-only"]:
        p_side = mp.Process(
            target=_run_side_worker,
            args=(
                args.side_source,
                "headless" if args.mode in ("headless", "web") else "dual",
                args.side_name,
                args.side_cam_id,
                args.device,
                args.enable_db,
                stop_event,
                fps_side,
                frames_side,
                lat_side,
                stat_side,
                args.gui_fps
            ),
            kwargs={
                "ws_sched_state_val": ws_sched_state_proc,
                "standby_fps": float(proc_scheduler.config.standby_fps),
            },
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
                ws_sched_state_proc.value = SCHED_STATE_TO_CODE[proc_scheduler.evaluate(force_refresh=True).state]
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
    if args.mode in ["dual", "opencv", "headless", "web", "top-only"]:
        print(f"  • CAM-01 Top (Slat & Rib)   : {f_top} frames ({avg_top_fps:.1f} rata-rata FPS)")
    if args.mode in ["dual", "opencv", "headless", "web", "side-only"]:
        print(f"  • {args.side_cam_id} Side (Bolt & Pin)  : {f_side} frames ({avg_side_fps:.1f} rata-rata FPS)")
    print(f"  • Throughput Gabungan       : {(avg_top_fps + avg_side_fps):.1f} frames/detik")

    # Evaluasi Ambang Batas Kelayakan (Dibaca langsung dari TARGET_MIN_FPS di .env)
    try:
        target_fps = float(os.environ.get("TARGET_MIN_FPS", "20.0"))
    except ValueError:
        target_fps = 20.0
    if avg_top_fps >= target_fps and (args.mode == "top-only" or avg_side_fps >= target_fps):
        verdict = f"🟢 SANGAT STABIL (Memenuhi standar real-time conveyor >= {target_fps:.1f} FPS)"
    elif avg_top_fps >= (target_fps * 0.75):
        verdict = f"🟡 CUKUP ({target_fps * 0.75:.1f} - {target_fps:.1f} FPS, disarankan konversi model ke TensorRT FP16 di Jetson)"
    else:
        verdict = "🔴 BUTUH OPTIMASI (< 15 FPS, gunakan mode headless & akselerasi TensorRT)"

    print(f"  • Penilaian Stabilitas Device: {verdict}")
    print("=" * 86)
    print("[*] Selesai.\n")


if __name__ == "__main__":
    if "--gui-dialog" in sys.argv:
        _handle_isolated_gui_dialog()
        sys.exit(0)

    # Wajib menggunakan metode 'spawn' untuk multiprocessing PyTorch & CUDA di Windows & Linux
    mp.set_start_method("spawn", force=True)
    main()
