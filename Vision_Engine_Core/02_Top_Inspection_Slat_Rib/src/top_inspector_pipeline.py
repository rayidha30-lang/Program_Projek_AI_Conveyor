"""
==============================================================================
PIPELINE INSPEKSI TAMPAK ATAS KONVEYOR (CAM-03: SLAT & RIB TOP INSPECTION)
==============================================================================
Deskripsi:
  Menyatukan modul inspeksi Deformasi Kelurusan Sirip (Rib Inspector) dan
  Deformasi Kelengkungan Plat & Celah (Slat Gap Inspector) ke dalam satu alur
  inspeksi terpadu sesuai standar PRD AI Conveyor Inspection System.
  
Fitur Utama:
  - Metrologi Kelurusan Sirip (Rib Straightness PV Deflection).
  - Metrologi Celah & Plat (Dual-Edge Gradient + Euler-Bernoulli Beam Curvature).
  - Penentuan Status Terpadu sesuai PRD (NG > INVALID > OK).
  - Interactive ROI Drag & Drop (Penyesuaian titik sudut ROI Sirip & Plat dengan Mouse).
  - Dynamic Video Switching ([O] / [F] hotkey untuk memilih video lain kapan saja).
  - Real-Time Live Graph PIP (Grafik defleksi sirip dan celah plat).
  - Timeline Scrubber Bar & Frame Stepping.
==============================================================================
"""

import os
import sys
import json
import time
import cv2
import numpy as np

# Tambahkan path lokal agar modul dapat diimpor langsung
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from rib_inspector import RibInspector
from slat_gap_inspector import SlatGapInspector


class TopConveyorInspector:
    """
    Sistem Terpadu Inspeksi Tampak Atas Slat & Rib Conveyor (CAM-03).
    """
    def __init__(
        self,
        camera_id="CAM-03",
        config_path=None,
        rib_pv_thresh=15.0,
        slat_delta_limit=14.0
    ):
        self.camera_id = camera_id
        base_dir = os.path.dirname(CURRENT_DIR)

        if config_path is None:
            config_path = os.path.join(base_dir, "configs", "top_inspection_config.json")
        self.config_path = config_path

        # 1. Nilai Default ROI
        self.default_rib_polygon = [
            [0.05, 0.38],
            [0.95, 0.38],
            [0.95, 0.50],
            [0.05, 0.50]
        ]
        self.default_slat_polygon = [
            [0.05, 0.58],
            [0.95, 0.58],
            [0.95, 0.75],
            [0.05, 0.75]
        ]

        # 2. Muat Konfigurasi dari JSON
        self.rib_polygon = [list(p) for p in self.default_rib_polygon]
        self.slat_polygon = [list(p) for p in self.default_slat_polygon]
        self.thresholds = {
            "rib_pv_threshold": rib_pv_thresh,
            "rib_max_dev_threshold": 12.0,
            "slat_delta_limit": slat_delta_limit,
            "slat_max_gap_limit": 45.0,
            "slat_valley_sens": 5.0
        }
        self._load_config()

        # 3. Inisialisasi Detektor
        yolo_model_path = os.path.join(base_dir, "models", "best_rib.pt")
        if not os.path.exists(yolo_model_path):
            yolo_model_path = None
            
        self.rib_inspector = RibInspector(
            model_path=yolo_model_path,
            pv_threshold=self.thresholds["rib_pv_threshold"],
            max_dev_threshold=self.thresholds["rib_max_dev_threshold"],
            use_yolo=False
        )
        self.slat_gap_inspector = SlatGapInspector(
            delta_limit=self.thresholds["slat_delta_limit"],
            max_gap_limit=self.thresholds["slat_max_gap_limit"],
            valley_sens=self.thresholds["slat_valley_sens"]
        )

        # 4. State & Statistik Sesi
        self.simulate_bending = False
        self.bending_amplitude = 22.0  # px defleksi bengkok simulasi
        self.last_rib_result = None
        self.last_slat_result = None
        self.stats = {
            "total_frames": 0,
            "ok_count": 0,
            "ng_count": 0,
            "invalid_count": 0
        }

    def reset_session(self):
        """Mereset state inspeksi dan statistik sesi."""
        self.last_rib_result = None
        self.last_slat_result = None
        self.stats = {
            "total_frames": 0,
            "ok_count": 0,
            "ng_count": 0,
            "invalid_count": 0
        }

    def _load_config(self):
        """Memuat konfigurasi ROI dan threshold dari file JSON."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    data = json.load(f)
                    if "rib_polygon" in data and len(data["rib_polygon"]) >= 4:
                        self.rib_polygon = data["rib_polygon"]
                    if "slat_polygon" in data and len(data["slat_polygon"]) >= 4:
                        self.slat_polygon = data["slat_polygon"]
                    if "thresholds" in data:
                        self.thresholds.update(data["thresholds"])
                print(f"[*] Konfigurasi Top Inspection berhasil dimuat dari: {self.config_path}")
            except Exception as e:
                print(f"[WARN] Gagal membaca config: {e}")

    def save_config(self):
        """Menyimpan konfigurasi ROI dan threshold ke file JSON."""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            data = {
                "camera_id": self.camera_id,
                "rib_polygon": self.rib_polygon,
                "slat_polygon": self.slat_polygon,
                "thresholds": self.thresholds,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(self.config_path, "w") as f:
                json.dump(data, f, indent=2)
            print("[*] Konfigurasi ROI Top Inspection tersimpan otomatis.")
        except Exception as e:
            print(f"[ERROR] Gagal menyimpan konfigurasi: {e}")

    def _inject_simulated_bending(self, frame):
        """
        Menyuntikkan distorsi kelengkungan fisik (Bending Deformation Simulation)
        secara matematis pada area plat untuk memvalidasi algoritma deteksi NG.
        """
        h, w = frame.shape[:2]
        map_x, map_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

        # Hitung profil lengkungan parabolik di sepanjang lebar konveyor
        norm_x = (map_x - 0.10 * w) / (0.80 * w)
        bend_shape = np.sin(np.pi * np.clip(norm_x, 0.0, 1.0))
        bend_px = self.bending_amplitude * bend_shape

        # Pembobotan vertikal pada zona plat
        y_weight = np.clip((map_y - 0.35 * h) / (0.15 * h), 0.0, 1.0) * np.clip((0.85 * h - map_y) / (0.15 * h), 0.0, 1.0)
        map_y_bent = map_y + (bend_px * y_weight).astype(np.float32)

        bent_frame = cv2.remap(frame, map_x, map_y_bent, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return bent_frame

    def process_frame(self, frame, speed_mult=1.0):
        """
        Memproses satu frame citra kamera CAM-03 secara lengkap.
        :param frame: Frame citra BGR (np.ndarray)
        :param speed_mult: Multiplier kecepatan playback (misal: 1.0, 2.0)
        :return: (display_frame, payload)
        """
        if frame is None or frame.size == 0:
            return frame, {"overall_result": "INVALID", "components": []}

        # Jika mode simulasi bending aktif (Tombol [B] ditekan)
        if self.simulate_bending:
            frame = self._inject_simulated_bending(frame)

        h_f, w_f = frame.shape[:2]
        self.stats["total_frames"] += 1
        display_frame = frame.copy()

        # Update threshold detektor jika ada perubahan dinamis
        self.rib_inspector.pv_threshold = self.thresholds["rib_pv_threshold"]
        self.slat_gap_inspector.delta_limit = self.thresholds["slat_delta_limit"]

        # 1. Jalankan Inspeksi Kelurusan Sirip (RIB)
        rib_res = self.rib_inspector.inspect(frame, self.rib_polygon)
        if rib_res.get("is_rib_present", False):
            self.last_rib_result = rib_res
            active_rib_status = rib_res["status"]
            active_rib_pv = rib_res["pv_flatness"]
        elif self.last_rib_result is not None:
            active_rib_status = self.last_rib_result["status"]
            active_rib_pv = self.last_rib_result["pv_flatness"]
        else:
            active_rib_status = "WAIT"
            active_rib_pv = 0.0

        # 2. Jalankan Inspeksi Deformasi Plat & Celah (SLAT / GAP)
        slat_res = self.slat_gap_inspector.inspect(frame, self.slat_polygon)
        if not slat_res.get("is_transition", True):
            self.last_slat_result = slat_res
            active_slat_status = slat_res["status"]
            active_slat_curv = slat_res["curvature"]
            active_slat_gap = slat_res["gap_max"]
        elif self.last_slat_result is not None:
            active_slat_status = self.last_slat_result["status"]
            active_slat_curv = self.last_slat_result["curvature"]
            active_slat_gap = self.last_slat_result["gap_max"]
        else:
            active_slat_status = "WAIT"
            active_slat_curv = 0.0
            active_slat_gap = 0.0

        # 3. Evaluasi Status Komponen & Overall Result (Sesuai Hirarki PRD: NG > INVALID > OK)
        if active_rib_status == "NG" or active_slat_status == "NG":
            overall_result = "NG"
            self.stats["ng_count"] += 1
        elif active_rib_status == "INVALID" or active_slat_status == "INVALID":
            overall_result = "INVALID"
            self.stats["invalid_count"] += 1
        elif active_rib_status == "WAIT" and active_slat_status == "WAIT":
            overall_result = "SCANNING"
            self.stats["ok_count"] += 1
        else:
            overall_result = "OK"
            self.stats["ok_count"] += 1

        payload = {
            "timestamp": time.time(),
            "camera_id": self.camera_id,
            "overall_result": overall_result,
            "components": [
                {
                    "name": "RIB (SIRIP)",
                    "result": active_rib_status,
                    "pv_flatness": active_rib_pv,
                    "is_active": rib_res.get("is_rib_present", False)
                },
                {
                    "name": "SLAT (PLAT)",
                    "result": active_slat_status,
                    "curvature": active_slat_curv,
                    "gap_max": active_slat_gap,
                    "is_active": not slat_res.get("is_transition", True)
                }
            ],
            "rib_details": rib_res if rib_res.get("is_rib_present", False) else self.last_rib_result,
            "slat_details": slat_res if not slat_res.get("is_transition", True) else self.last_slat_result
        }

        # 4. Gambar Overlay Poligon & Garis Inspeksi pada Frame
        self._draw_roi_polygons(display_frame)
        display_frame = self.rib_inspector.draw_overlay_on_frame(display_frame, rib_res)
        display_frame = self.slat_gap_inspector.draw_overlay_on_frame(display_frame, slat_res)

        # 5. Gambar HUD Banner Atas
        self._draw_hud(display_frame, payload, speed_mult=speed_mult)

        # Indikator jika mode simulasi bending aktif
        if self.simulate_bending:
            cv2.rectangle(display_frame, (15, 82), (480, 110), (0, 0, 180), -1)
            cv2.putText(display_frame, "[!] TEST MODE: BENDING DEFECT SIMULATOR (+22px)", (20, 102),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

        # 6. Gambar Mini PIP Live Graphs di Pojok Kanan Atas
        self._draw_pip_graphs(display_frame, rib_res, slat_res)

        return display_frame, payload

    def _draw_roi_polygons(self, img):
        """Menggambar poligon ROI Sirip dan Plat dengan titik sudut interaktif."""
        h_f, w_f = img.shape[:2]

        # 1. Poligon Sirip (Cyan)
        pts_rib_px = np.array([[int(p[0] * w_f), int(p[1] * h_f)] for p in self.rib_polygon], dtype=np.int32)
        cv2.polylines(img, [pts_rib_px], isClosed=True, color=(255, 220, 0), thickness=2, lineType=cv2.LINE_AA)
        for i, (px, py) in enumerate(pts_rib_px):
            cv2.circle(img, (px, py), 6, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (px, py), 7, (20, 20, 20), 1, cv2.LINE_AA)
            cv2.putText(img, f"R{i+1}", (px + 8, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1)

        # 2. Poligon Plat / Slat (Orange)
        pts_slat_px = np.array([[int(p[0] * w_f), int(p[1] * h_f)] for p in self.slat_polygon], dtype=np.int32)
        cv2.polylines(img, [pts_slat_px], isClosed=True, color=(0, 165, 255), thickness=2, lineType=cv2.LINE_AA)
        for i, (px, py) in enumerate(pts_slat_px):
            cv2.circle(img, (px, py), 6, (0, 140, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (px, py), 7, (20, 20, 20), 1, cv2.LINE_AA)
            cv2.putText(img, f"S{i+1}", (px + 8, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 140, 255), 1)

    def _draw_hud(self, img, payload, speed_mult=1.0):
        """Menggambar panel banner HUD di bagian atas layar (Ringan & Hemat Memori)."""
        h, w = img.shape[:2]
        hud_h = 75
        
        # Blend hanya pada area pita 75px teratas
        hud_strip = img[0:hud_h, 0:w].copy()
        overlay = np.full_like(hud_strip, 25)
        cv2.addWeighted(overlay, 0.85, hud_strip, 0.15, 0, img[0:hud_h, 0:w])

        res = payload["overall_result"]
        res_color = (0, 255, 0) if "OK" in res else ((0, 0, 255) if res == "NG" else (0, 200, 255))

        # Judul & Status Utama
        cv2.putText(img, f"CONVEYOR TOP INSPECTION [{self.camera_id} - SLAT & RIB]",
                    (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2)
        cv2.putText(img, f"STATUS: {res}", (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, res_color, 2)

        # Indikator Kecepatan (Speed)
        speed_color = (0, 255, 255) if speed_mult > 1.0 else ((100, 200, 255) if speed_mult < 1.0 else (180, 180, 180))
        speed_tag = " [FAST]" if speed_mult > 1.0 else (" [SLOW]" if speed_mult < 1.0 else "")
        cv2.putText(img, f"SPEED: {speed_mult:.1f}x{speed_tag}", (235, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, speed_color, 2)

        # Rincian Komponen
        c_x = int(w * 0.44)
        for c in payload["components"]:
            c_res = c["result"]
            c_col = (0, 255, 0) if c_res == "OK" else ((0, 0, 255) if c_res == "NG" else (180, 180, 180))
            is_act = c.get("is_active", False)
            act_tag = "*" if is_act else ""
            if "RIB" in c["name"]:
                txt = f"{c['name']}{act_tag}: {c_res} (PV: {c.get('pv_flatness', 0.0):.1f}px)"
            else:
                txt = f"{c['name']}{act_tag}: {c_res} (Curv: {c.get('curvature', 0.0):.1f}px)"
            cv2.putText(img, txt, (c_x, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.58, c_col, 2)
            c_x += 260

    def _draw_pip_graphs(self, img, rib_res, slat_res):
        """Menampilkan mini graph PIP di sudut kanan atas."""
        h_f, w_f = img.shape[:2]
        pip_w, pip_h = 240, 95
        px1 = max(0, w_f - pip_w - 20)
        px2 = px1 + pip_w

        # Mini Graph 1: Rib Profile
        rib_graph = rib_res.get("graph_img")
        if rib_graph is None and self.last_rib_result is not None:
            rib_graph = self.last_rib_result.get("graph_img")
        if rib_graph is not None and isinstance(rib_graph, np.ndarray) and rib_graph.size > 0:
            rg_resized = cv2.resize(rib_graph, (pip_w, pip_h))
            py1 = 85
            py2 = py1 + pip_h
            img[py1:py2, px1:px2] = rg_resized
            cv2.rectangle(img, (px1, py1), (px2, py2), (255, 220, 0), 1)

        # Mini Graph 2: Slat Gap Profile
        slat_graph = slat_res.get("graph_img")
        if slat_graph is None and self.last_slat_result is not None:
            slat_graph = self.last_slat_result.get("graph_img")
        if slat_graph is not None and isinstance(slat_graph, np.ndarray) and slat_graph.size > 0:
            sg_resized = cv2.resize(slat_graph, (pip_w, pip_h))
            py1_s = 85 + pip_h + 10
            py2_s = py1_s + pip_h
            img[py1_s:py2_s, px1:px2] = sg_resized
            cv2.rectangle(img, (px1, py1_s), (px2, py2_s), (0, 165, 255), 1)


def select_video_file(initial_dir=None, title="Pilih File Video Top Inspection"):
    """Membuka dialog file explorer Windows untuk memilih file video."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        default_dir = initial_dir
        if not default_dir or not os.path.exists(default_dir):
            candidate_dirs = [
                r"C:\Users\Victus\Pictures\Video & Foto Test Slat Rib",
                os.path.join(os.path.expanduser("~"), "Pictures"),
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
        root.destroy()
        return file_path if file_path and os.path.exists(file_path) else None
    except Exception as e:
        print(f"[WARN] Dialog file tidak dapat dibuka ({e}).")
        return None


def run_top_inspector_demo(video_path=None, camera_id="CAM-03"):
    """
    Fungsi demo interaktif untuk menguji video konveyor tampak atas (CAM-03).
    Mendukung:
    - Pemilihan file video baru via shortcut [O] / [F]
    - Penyesuaian titik sudut Poligon Sirip (R1-R4) & Plat (S1-S4) via Mouse Drag
    - Trackbar Threshold & Timeline Player
    """
    inspector = TopConveyorInspector(camera_id=camera_id)

    # Validasi path video
    if not video_path or not os.path.exists(video_path):
        print("[*] Membuka dialog pemilih file video Top Inspection...")
        video_path = select_video_file(title="Pilih File Video Top Inspection (CAM-03)")
        if not video_path:
            print("[ERROR] Tidak ada video yang dipilih. Demo dihentikan.")
            return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Gagal membuka video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[*] Membuka Video: {os.path.basename(video_path)} ({total_frames} frame, {fps:.1f} FPS)")

    win_name = f"Conveyor Top Inspector - [{camera_id} SLAT & RIB]"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 720)

    # State Mouse Drag
    mouse_state = {
        "selected_target": None,   # 'rib' atau 'slat'
        "selected_idx": None,      # indeks titik 0..3
        "is_moving_poly": False,
        "drag_start_pos": None,
        "drag_start_poly": None
    }

    def mouse_cb(event, x, y, flags, param):
        w_f, h_f = param
        nx, ny = max(0.01, min(0.99, x / float(w_f))), max(0.01, min(0.99, y / float(h_f)))

        if event == cv2.EVENT_LBUTTONDOWN:
            # 1. Cek Titik Sudut Poligon Sirip (Rib)
            for i, (px_n, py_n) in enumerate(inspector.rib_polygon):
                px, py = int(px_n * w_f), int(py_n * h_f)
                if np.hypot(x - px, y - py) < 18:
                    mouse_state["selected_target"] = "rib"
                    mouse_state["selected_idx"] = i
                    return

            # 2. Cek Titik Sudut Poligon Plat (Slat)
            for i, (px_n, py_n) in enumerate(inspector.slat_polygon):
                px, py = int(px_n * w_f), int(py_n * h_f)
                if np.hypot(x - px, y - py) < 18:
                    mouse_state["selected_target"] = "slat"
                    mouse_state["selected_idx"] = i
                    return

            # 3. Cek Dalam Poligon Sirip (Geser Utuh)
            pts_r = np.array([[int(p[0] * w_f), int(p[1] * h_f)] for p in inspector.rib_polygon], dtype=np.int32)
            if cv2.pointPolygonTest(pts_r, (x, y), False) >= 0:
                mouse_state["selected_target"] = "rib"
                mouse_state["is_moving_poly"] = True
                mouse_state["drag_start_pos"] = (nx, ny)
                mouse_state["drag_start_poly"] = [list(p) for p in inspector.rib_polygon]
                return

            # 4. Cek Dalam Poligon Plat (Geser Utuh)
            pts_s = np.array([[int(p[0] * w_f), int(p[1] * h_f)] for p in inspector.slat_polygon], dtype=np.int32)
            if cv2.pointPolygonTest(pts_s, (x, y), False) >= 0:
                mouse_state["selected_target"] = "slat"
                mouse_state["is_moving_poly"] = True
                mouse_state["drag_start_pos"] = (nx, ny)
                mouse_state["drag_start_poly"] = [list(p) for p in inspector.slat_polygon]
                return

        elif event == cv2.EVENT_MOUSEMOVE:
            # Drag Titik Sudut
            if mouse_state["selected_idx"] is not None:
                if mouse_state["selected_target"] == "rib":
                    inspector.rib_polygon[mouse_state["selected_idx"]] = [nx, ny]
                elif mouse_state["selected_target"] == "slat":
                    inspector.slat_polygon[mouse_state["selected_idx"]] = [nx, ny]

            # Drag Seluruh Poligon
            elif mouse_state["is_moving_poly"] and mouse_state["drag_start_pos"] and mouse_state["drag_start_poly"]:
                dx = nx - mouse_state["drag_start_pos"][0]
                dy = ny - mouse_state["drag_start_pos"][1]
                new_poly = []
                for ox, oy in mouse_state["drag_start_poly"]:
                    new_poly.append([
                        max(0.01, min(0.99, ox + dx)),
                        max(0.01, min(0.99, oy + dy))
                    ])
                if mouse_state["selected_target"] == "rib":
                    inspector.rib_polygon = new_poly
                elif mouse_state["selected_target"] == "slat":
                    inspector.slat_polygon = new_poly

        elif event == cv2.EVENT_LBUTTONUP:
            if mouse_state["selected_idx"] is not None or mouse_state["is_moving_poly"]:
                inspector.save_config()

            mouse_state["selected_target"] = None
            mouse_state["selected_idx"] = None
            mouse_state["is_moving_poly"] = False
            mouse_state["drag_start_pos"] = None
            mouse_state["drag_start_poly"] = None

    # Trackbars
    speed_mult = 1.0

    def on_rib_pv(val):
        inspector.thresholds["rib_pv_threshold"] = float(val)

    def on_slat_curv(val):
        inspector.thresholds["slat_delta_limit"] = float(val)

    def on_timeline(val):
        cap.set(cv2.CAP_PROP_POS_FRAMES, val)

    def on_speed(val):
        nonlocal speed_mult
        speed_mult = max(0.1, val / 10.0)

    cv2.createTrackbar("Rib PV Thresh (px)", win_name, int(inspector.thresholds["rib_pv_threshold"]), 40, on_rib_pv)
    cv2.createTrackbar("Slat Curv Thresh (px)", win_name, int(inspector.thresholds["slat_delta_limit"]), 40, on_slat_curv)
    cv2.createTrackbar("Speed (x0.1)", win_name, 10, 50, on_speed)
    cv2.createTrackbar("Timeline", win_name, 0, max(1, total_frames - 1), on_timeline)

    is_paused = False

    print("\n" + "=" * 75)
    print("   SISTEM INSPEKSI TAMPAK ATAS KONVEYOR (CAM-03 SLAT & RIB)")
    print("=" * 75)
    print("  - [O] / [F]            : PILIH / GANTI FILE VIDEO LAIN (File Explorer)")
    print("  - POLIGON SIRIP (R1-R4): Klik & geser titik kuning untuk zona Sirip (Rib)")
    print("  - POLIGON PLAT (S1-S4) : Klik & geser titik oranye untuk zona Plat (Slat)")
    print("  - [SPACE]              : Pause / Play video")
    print("  - [+]/[W] & [-]/[X]    : Tambah / Kurangi Kecepatan Playback (0.2x - 5.0x)")
    print("  - [1]/[2]/[3]/[4]/[5]  : Preset Kecepatan (1x, 2x, 3x, 4x, 0.5x Slow-Mo)")
    print("  - [D] / [A]            : Maju / Mundur 1 frame saat pause")
    print("  - [B]                  : Toggle Simulasi Bending NG (+22px)")
    print("  - [R]                  : Reset posisi poligon Sirip & Plat ke default")
    print("  - [Q] / [ESC]          : Keluar")
    print("=" * 75 + "\n")

    while True:
        if not is_paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            cur_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cv2.setTrackbarPos("Timeline", win_name, cur_pos)

            # Frame skipping untuk percepatan tinggi (>= 2.0x)
            if speed_mult >= 2.0:
                skip_frames = int(speed_mult) - 1
                if skip_frames > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, min(total_frames - 1, cur_pos + skip_frames))
        else:
            cur_pos = cv2.getTrackbarPos("Timeline", win_name)
            cap.set(cv2.CAP_PROP_POS_FRAMES, cur_pos)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

        h_f, w_f = frame.shape[:2]
        cv2.setMouseCallback(win_name, mouse_cb, (w_f, h_f))

        display_frame, payload = inspector.process_frame(frame, speed_mult=speed_mult)

        # Petunjuk shortcut di bagian bawah frame
        help_bar = f"[SPACE]: Pause | [+]/[-]/[1-5]: Speed ({speed_mult:.1f}x) | [B]: Bending NG | [O]/[F]: Ganti Video | [D]/[A]: Step | [Q]: Keluar"
        cv2.putText(display_frame, help_bar, (15, h_f - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

        cv2.imshow(win_name, display_frame)

        base_delay = max(1, int(1000 / fps))
        calc_delay = max(1, int(base_delay / speed_mult))
        wait_time = calc_delay if not is_paused else 30

        key = cv2.waitKey(wait_time) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            is_paused = not is_paused
        elif key in [ord('+'), ord('='), ord('w'), ord('W')]:
            speed_mult = min(5.0, round(speed_mult + 0.5, 2))
            cv2.setTrackbarPos("Speed (x0.1)", win_name, int(speed_mult * 10))
            print(f"\n[>] Kecepatan Playback dinaikkan: {speed_mult:.1f}x")
        elif key in [ord('-'), ord('_'), ord('x'), ord('X')]:
            speed_mult = max(0.2, round(speed_mult - 0.25, 2))
            cv2.setTrackbarPos("Speed (x0.1)", win_name, int(speed_mult * 10))
            print(f"\n[<] Kecepatan Playback diturunkan: {speed_mult:.1f}x")
        elif key == ord('1'):
            speed_mult = 1.0
            cv2.setTrackbarPos("Speed (x0.1)", win_name, 10)
            print(f"\n[1] Kecepatan Playback Normal: 1.0x")
        elif key == ord('2'):
            speed_mult = 2.0
            cv2.setTrackbarPos("Speed (x0.1)", win_name, 20)
            print(f"\n[2] Kecepatan Playback Cepat: 2.0x")
        elif key == ord('3'):
            speed_mult = 3.0
            cv2.setTrackbarPos("Speed (x0.1)", win_name, 30)
            print(f"\n[3] Kecepatan Playback Sangat Cepat: 3.0x")
        elif key == ord('4'):
            speed_mult = 4.0
            cv2.setTrackbarPos("Speed (x0.1)", win_name, 40)
            print(f"\n[4] Kecepatan Playback Turbo: 4.0x")
        elif key in [ord('5'), ord('0')]:
            speed_mult = 0.5
            cv2.setTrackbarPos("Speed (x0.1)", win_name, 5)
            print(f"\n[5] Kecepatan Playback Lambat (Slow-Mo): 0.5x")
        elif key in [ord('b'), ord('B')]:
            inspector.simulate_bending = not inspector.simulate_bending
            state_str = "AKTIF (BENDING +22px DEFLEKSI)" if inspector.simulate_bending else "NON-AKTIF (NORMAL)"
            print(f"\n[!] Mode Simulasi Deformasi Bending Plat: {state_str}")
        elif key in [ord('o'), ord('O'), ord('f'), ord('F')]:
            cur_dir = os.path.dirname(video_path) if video_path and os.path.exists(os.path.dirname(video_path)) else None
            new_video = select_video_file(initial_dir=cur_dir, title="Pilih File Video Baru untuk Top Inspection")
            if new_video and os.path.exists(new_video):
                new_cap = cv2.VideoCapture(new_video)
                if new_cap.isOpened():
                    cap.release()
                    cap = new_cap
                    video_path = new_video
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cv2.setTrackbarMax("Timeline", win_name, max(1, total_frames - 1))
                    cv2.setTrackbarPos("Timeline", win_name, 0)
                    inspector.reset_session()
                    print(f"\n[✓] Berhasil memuat video baru: {os.path.basename(video_path)} ({total_frames} frame, {fps:.1f} FPS)")
                else:
                    print(f"[ERROR] Gagal membuka video baru: {new_video}")
            else:
                print("[*] Pemilihan file video dibatalkan.")
        elif key == ord('d'):
            cur = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(total_frames - 1, cur + 1))
            cv2.setTrackbarPos("Timeline", win_name, int(cur + 1))
        elif key == ord('a'):
            cur = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, cur - 2))
            cv2.setTrackbarPos("Timeline", win_name, int(max(0, cur - 2)))
        elif key == ord('r'):
            inspector.rib_polygon = [list(p) for p in inspector.default_rib_polygon]
            inspector.slat_polygon = [list(p) for p in inspector.default_slat_polygon]
            inspector.save_config()
            print("[*] Konfigurasi ROI Sirip & Plat di-reset ke default.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Top Conveyor Inspector Pipeline (CAM-03 Slat & Rib)")
    parser.add_argument("video", nargs="?", default=None, help="Path file video inspeksi (opsional)")
    parser.add_argument("--cam", default="CAM-03", help="ID Kamera (default: CAM-03)")
    args = parser.parse_args()

    video_input = args.video
    if not video_input:
        default_video = r"C:\Users\Victus\Pictures\Video & Foto Test Slat Rib\Opsi Posisi 1 - Slat Rib.mp4"
        if os.path.exists(default_video):
            video_input = default_video
        else:
            print("[*] Video default tidak ditemukan, membuka dialog pemilih file...")
            video_input = select_video_file(title="Pilih File Video untuk Top Inspection")

    if video_input and os.path.exists(video_input):
        run_top_inspector_demo(video_input, camera_id=args.cam)
    else:
        print("[!] Tidak ada video yang dipilih. Program dihentikan.")
