"""
==============================================================================
MODUL TELEMETRI & VISUALISASI AKSELERASI HARDWARE / CUDA (CAM-03 & CAM-01)
==============================================================================
Deskripsi:
  Modul ini memantau dan memvisualisasikan status komputasi akselerasi perangkat keras:
    - Status NVIDIA CUDA (PyTorch GPU Runtime, VRAM terpakai & teralokasi, Device Name).
    - Status Hardware NVIDIA Jetson Orin Nano / Tegra via sysfs:
      * GPU Load real-time (%)
      * GPU Engine Frequency (MHz)
      * GPU & CPU Thermal Sensors (°C)
      * Unified Memory LPDDR5 (Used / Total MB & %)
  Menyediakan komponen visual badge HUD berkualitas tinggi dan non-blocking
  untuk ditampilkan pada viewport kamera dan panel dashboard metrologi.
==============================================================================
"""

import os
import time
import cv2
import numpy as np

# Bootstrap Environment CUDA 12.6 Jetson Orin Nano secara otomatis
try:
    import jetson_cuda_env  # noqa: F401
except ImportError:
    pass

# Cek ketersediaan PyTorch secara aman
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False



class HardwareTelemetry:
    """
    Pelacak status akselerasi perangkat keras (CUDA GPU / Jetson Orin Telemetry / CPU).
    Membaca sysfs Tegra dan PyTorch CUDA secara efisien dengan caching temporal
    sehingga tidak pernah membebani loop inspeksi utama.
    """

    def __init__(self, sample_interval_sec=0.25):
        self.sample_interval = sample_interval_sec
        self.last_sample_time = 0.0

        # Cek apakah sistem adalah NVIDIA Jetson (Tegra)
        self.is_jetson = (
            os.path.exists("/sys/devices/platform/bus@0/17000000.gpu") or
            os.path.exists("/sys/devices/gpu.0") or
            os.path.exists("/etc/nv_tegra_release")
        )

        self.cached_telemetry = {
            "cuda_available": False,
            "device_name": "CPU",
            "backend_tag": "CPU",
            "badge_title": "CPU COMPUTATION",
            "status_text": "CPU MODE",
            "gpu_load_pct": 0.0,
            "gpu_freq_mhz": 0,
            "gpu_temp_c": 0.0,
            "cpu_temp_c": 0.0,
            "mem_used_mb": 0.0,
            "mem_total_mb": 0.0,
            "mem_pct": 0.0,
            "vram_used_mb": 0.0,
            "vram_total_mb": 0.0,
            "vram_pct": 0.0,
            "is_jetson": self.is_jetson,
            "theme_color": (0, 210, 255)  # BGR
        }

        self._init_static_hardware()

    def _init_static_hardware(self):
        """Mendeteksi spesifikasi statis perangkat keras saat inisialisasi."""
        cuda_ok = False
        if HAS_TORCH:
            try:
                cuda_ok = torch.cuda.is_available()
            except Exception:
                cuda_ok = False

        if cuda_ok:
            try:
                dev_idx = 0
                dev_name = torch.cuda.get_device_name(dev_idx)
                props = torch.cuda.get_device_properties(dev_idx)
                total_vram = props.total_memory / (1024 * 1024)

                self.cached_telemetry["cuda_available"] = True
                self.cached_telemetry["device_name"] = dev_name
                self.cached_telemetry["backend_tag"] = f"CUDA GPU #{dev_idx}"
                self.cached_telemetry["badge_title"] = "NVIDIA CUDA ACCEL"
                self.cached_telemetry["status_text"] = "CUDA GPU ACTIVE"
                self.cached_telemetry["vram_total_mb"] = total_vram
                self.cached_telemetry["theme_color"] = (0, 230, 100)  # Hijau Emerald
                return
            except Exception:
                pass

        # Jika CUDA PyTorch belum aktif tetapi berjalan di Jetson Orin
        if self.is_jetson:
            self.cached_telemetry["cuda_available"] = False
            self.cached_telemetry["device_name"] = "NVIDIA Jetson Orin Nano"
            self.cached_telemetry["backend_tag"] = "JETSON SOC"
            self.cached_telemetry["badge_title"] = "JETSON HARDWARE"
            self.cached_telemetry["status_text"] = "GPU TELEMETRY ON"
            self.cached_telemetry["theme_color"] = (0, 215, 255)  # Cyan Gold
        else:
            self.cached_telemetry["cuda_available"] = False
            self.cached_telemetry["device_name"] = "Host CPU"
            self.cached_telemetry["backend_tag"] = "CPU"
            self.cached_telemetry["badge_title"] = "CPU COMPUTATION"
            self.cached_telemetry["status_text"] = "CPU MODE"
            self.cached_telemetry["theme_color"] = (200, 200, 200)

    def get_telemetry(self):
        """
        Mengembalikan kamus telemetri terbaru. Menggunakan caching 250ms
        agar tidak membebani I/O disk/sysfs.
        """
        now = time.time()
        if now - self.last_sample_time < self.sample_interval:
            return self.cached_telemetry

        self.last_sample_time = now

        # 1. Baca Jetson GPU Load dari sysfs
        for p in [
            "/sys/devices/platform/bus@0/17000000.gpu/load",
            "/sys/devices/gpu.0/load"
        ]:
            if os.path.exists(p):
                try:
                    with open(p, "r") as f:
                        v = float(f.read().strip())
                        # sysfs mengembalikan nilai permille (0 - 1000)
                        self.cached_telemetry["gpu_load_pct"] = min(100.0, max(0.0, v / 10.0))
                    break
                except Exception:
                    pass

        # 2. Baca Jetson GPU Frequency dari sysfs
        for p in [
            "/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/cur_freq",
            "/sys/devices/gpu.0/devfreq/devfreq0/cur_freq"
        ]:
            if os.path.exists(p):
                try:
                    with open(p, "r") as f:
                        hz = float(f.read().strip())
                        self.cached_telemetry["gpu_freq_mhz"] = int(hz / 1e6)
                    break
                except Exception:
                    pass

        # 3. Baca Sensor Suhu Jetson (Thermal Zones)
        thermal_dir = "/sys/class/thermal"
        if os.path.exists(thermal_dir):
            try:
                for zone in os.listdir(thermal_dir):
                    if zone.startswith("thermal_zone"):
                        zp = os.path.join(thermal_dir, zone)
                        try:
                            with open(os.path.join(zp, "type"), "r") as tf:
                                ztype = tf.read().strip().lower()
                            with open(os.path.join(zp, "temp"), "r") as mf:
                                deg = float(mf.read().strip()) / 1000.0

                            if "gpu" in ztype:
                                self.cached_telemetry["gpu_temp_c"] = deg
                            elif "cpu" in ztype:
                                self.cached_telemetry["cpu_temp_c"] = deg
                            elif "tj" in ztype and self.cached_telemetry["gpu_temp_c"] == 0.0:
                                self.cached_telemetry["gpu_temp_c"] = deg
                        except Exception:
                            # Abaikan zone yang tidak menyediakan data
                            continue
            except Exception:
                pass

        # 4. Baca Unified Memory / RAM dari /proc/meminfo
        if os.path.exists("/proc/meminfo"):
            try:
                mem = {}
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        parts = line.split(":")
                        if len(parts) == 2:
                            mem[parts[0].strip()] = parts[1].strip()
                tot_kb = float(mem.get("MemTotal", "0 kB").split()[0])
                avail_kb = float(mem.get("MemAvailable", "0 kB").split()[0])
                tot_mb = tot_kb / 1024.0
                used_mb = (tot_kb - avail_kb) / 1024.0
                self.cached_telemetry["mem_total_mb"] = tot_mb
                self.cached_telemetry["mem_used_mb"] = used_mb
                self.cached_telemetry["mem_pct"] = (used_mb / tot_mb) * 100.0 if tot_mb > 0 else 0.0
            except Exception:
                pass

        # 5. Baca VRAM jika PyTorch CUDA aktif
        if self.cached_telemetry["cuda_available"] and HAS_TORCH:
            try:
                alloc = torch.cuda.memory_allocated(0) / (1024 * 1024)
                resv = torch.cuda.memory_reserved(0) / (1024 * 1024)
                self.cached_telemetry["vram_used_mb"] = alloc
                if self.cached_telemetry["vram_total_mb"] > 0:
                    self.cached_telemetry["vram_pct"] = (resv / self.cached_telemetry["vram_total_mb"]) * 100.0
            except Exception:
                pass

        return self.cached_telemetry


def draw_camera_badge(canvas, telemetry, x=15, y=12, w=375, h=68):
    """
    Menggambar badge overlay visualisasi CUDA / Akselerasi Hardware pada sudut kamera:
      - Desain semi-transparan dengan border aksen elegan.
      - Menampilkan status CUDA / GPU, load meter grafis, suhu, dan memori.
    """
    if canvas is None or canvas.size == 0:
        return

    c_h, c_w = canvas.shape[:2]
    x2 = min(c_w - 2, x + w)
    y2 = min(c_h - 2, y + h)
    w_box = x2 - x
    h_box = y2 - y
    if w_box <= 20 or h_box <= 20:
        return

    theme = telemetry.get("theme_color", (0, 210, 255))
    is_cuda = telemetry.get("cuda_available", False)
    gpu_load = telemetry.get("gpu_load_pct", 0.0)
    gpu_freq = telemetry.get("gpu_freq_mhz", 0)
    gpu_temp = telemetry.get("gpu_temp_c", 0.0)
    cpu_temp = telemetry.get("cpu_temp_c", 0.0)
    dev_name = telemetry.get("device_name", "Hardware")

    # 1. Buat kotak semi-transparan (Alpha Blending)
    sub = canvas[y:y2, x:x2]
    rect = np.full(sub.shape, (16, 18, 24), dtype=np.uint8)
    cv2.addWeighted(rect, 0.78, sub, 0.22, 0, sub)
    canvas[y:y2, x:x2] = sub

    # Border berbingkai ganda
    cv2.rectangle(canvas, (x, y), (x2, y2), (45, 50, 65), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (x, y), (x2, y2), theme, 2, cv2.LINE_AA)

    # 2. Header Status dengan Icon Vector Petir
    bolt_pts = np.array([
        [x + 14, y + 8],
        [x + 20, y + 8],
        [x + 16, y + 14],
        [x + 21, y + 14],
        [x + 12, y + 22],
        [x + 14, y + 16],
        [x + 11, y + 16]
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [bolt_pts], theme)

    if is_cuda:
        title_text = "NVIDIA CUDA ACCELERATED"
        title_col = (0, 255, 120)
    elif telemetry.get("is_jetson", False):
        title_text = "JETSON ORIN HARDWARE"
        title_col = (0, 225, 255)
    else:
        title_text = "CPU PROCESSING MODE"
        title_col = (200, 220, 240)

    cv2.putText(canvas, title_text, (x + 26, y + 19),
                cv2.FONT_HERSHEY_DUPLEX, 0.42, title_col, 1, cv2.LINE_AA)

    # Tag Status Ringkas di kanan header
    tag_str = "[CUDA #0]" if is_cuda else ("[JETSON HW]" if telemetry.get("is_jetson", False) else "[CPU]")
    cv2.putText(canvas, tag_str, (x2 - 82, y + 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, theme, 1, cv2.LINE_AA)

    # 3. Bar Grafis GPU Load Real-time
    bar_x = x + 12
    bar_y = y + 27
    bar_w = 100
    bar_h = 10
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 45, 55), -1)

    fill_w = int((gpu_load / 100.0) * bar_w)
    if fill_w > 0:
        bar_col = (0, 255, 100) if gpu_load < 75.0 else ((0, 165, 255) if gpu_load < 90.0 else (0, 30, 255))
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_col, -1)
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 85, 100), 1)

    # Keterangan angka di samping bar load
    freq_txt = f"{gpu_freq}MHz" if gpu_freq > 0 else "Active"
    temp_val = gpu_temp if gpu_temp > 0 else cpu_temp
    temp_txt = f"{temp_val:.0f} C" if temp_val > 0 else "Normal"
    cv2.putText(canvas, f"GPU: {gpu_load:4.1f}% | {freq_txt} | {temp_txt}",
                (bar_x + bar_w + 10, bar_y + 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 230, 240), 1, cv2.LINE_AA)

    # 4. Baris 3: Informasi Memori & Perangkat
    if is_cuda:
        v_used = telemetry.get("vram_used_mb", 0.0)
        v_tot = telemetry.get("vram_total_mb", 0.0)
        mem_info = f"VRAM: {v_used:.0f}/{v_tot:.0f} MB | {dev_name[:20]}"
    else:
        m_used = telemetry.get("mem_used_mb", 0.0) / 1024.0
        m_tot = telemetry.get("mem_total_mb", 0.0) / 1024.0
        m_pct = telemetry.get("mem_pct", 0.0)
        c_temp_str = f" | CPU: {cpu_temp:.0f} C" if cpu_temp > 0 else ""
        mem_info = f"RAM: {m_used:.1f}/{m_tot:.1f} GB ({m_pct:.0f}%){c_temp_str} | {dev_name[:16]}"

    cv2.putText(canvas, mem_info, (x + 12, y + 57),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 205, 225), 1, cv2.LINE_AA)


def draw_hud_card(hud, telemetry, x=710, y=12, w=275, h=68):
    """
    Menggambar kartu instrumen Akselerasi Hardware & Telemetri GPU di panel HUD bawah.
    """
    if hud is None or hud.size == 0:
        return

    h_hud, w_hud = hud.shape[:2]
    x2 = min(w_hud - 15, x + w)
    y2 = min(h_hud - 10, y + h)

    theme = telemetry.get("theme_color", (0, 210, 255))
    is_cuda = telemetry.get("cuda_available", False)
    gpu_load = telemetry.get("gpu_load_pct", 0.0)
    gpu_freq = telemetry.get("gpu_freq_mhz", 0)
    gpu_temp = telemetry.get("gpu_temp_c", 0.0)
    cpu_temp = telemetry.get("cpu_temp_c", 0.0)

    # Background kartu
    cv2.rectangle(hud, (x, y), (x2, y2), (24, 25, 32), -1)
    cv2.rectangle(hud, (x, y), (x2, y2), (55, 60, 75), 1, cv2.LINE_AA)
    # Aksen sudut
    cv2.line(hud, (x, y), (x + 18, y), theme, 2, cv2.LINE_AA)
    cv2.line(hud, (x, y), (x, y + 18), theme, 2, cv2.LINE_AA)

    # Icon Vector Petir Mini
    bolt_hud = np.array([
        [x + 11, y + 8],
        [x + 16, y + 8],
        [x + 13, y + 13],
        [x + 17, y + 13],
        [x + 10, y + 20],
        [x + 11, y + 15],
        [x + 9,  y + 15]
    ], dtype=np.int32)
    cv2.fillPoly(hud, [bolt_hud], theme)

    # Judul Kartu
    hdr_txt = "HARDWARE ACCEL"
    cv2.putText(hud, hdr_txt, (x + 22, y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

    # Status Pill Mini di kanan header
    pill_text = "[CUDA ON]" if is_cuda else ("[JETSON ON]" if telemetry.get("is_jetson", False) else "[CPU ON]")
    pill_col = (0, 230, 100) if is_cuda else ((0, 215, 255) if telemetry.get("is_jetson", False) else (180, 180, 180))
    cv2.putText(hud, pill_text, (x2 - 82, y + 18),
                cv2.FONT_HERSHEY_DUPLEX, 0.36, pill_col, 1, cv2.LINE_AA)

    # Bar Grafis Mini GPU
    bar_x = x + 10
    bar_y = y + 26
    bar_w = 80
    bar_h = 9
    cv2.rectangle(hud, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (35, 38, 48), -1)
    fill_w = int((gpu_load / 100.0) * bar_w)
    if fill_w > 0:
        bar_col = (0, 255, 120) if gpu_load < 75.0 else (0, 160, 255)
        cv2.rectangle(hud, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_col, -1)
    cv2.rectangle(hud, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (70, 75, 90), 1)

    t_val = gpu_temp if gpu_temp > 0 else cpu_temp
    cv2.putText(hud, f"GPU: {gpu_load:4.1f}% | {t_val:.0f} C", (bar_x + bar_w + 8, bar_y + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 230, 240), 1, cv2.LINE_AA)

    # Baris 3: Info Memori & Clock
    if is_cuda:
        v_used = telemetry.get("vram_used_mb", 0.0)
        v_tot = telemetry.get("vram_total_mb", 0.0)
        sub_info = f"VRAM: {v_used:.0f}/{v_tot:.0f} MB | {gpu_freq} MHz"
    else:
        m_used = telemetry.get("mem_used_mb", 0.0) / 1024.0
        m_tot = telemetry.get("mem_total_mb", 0.0) / 1024.0
        m_pct = telemetry.get("mem_pct", 0.0)
        freq_str = f"{gpu_freq} MHz" if gpu_freq > 0 else "Dynamic"
        sub_info = f"RAM: {m_used:.1f}/{m_tot:.1f}G ({m_pct:.0f}%) | {freq_str}"

    cv2.putText(hud, sub_info, (x + 10, y + 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (170, 195, 215), 1, cv2.LINE_AA)
