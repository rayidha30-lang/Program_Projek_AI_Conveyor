"""
==============================================================================
PIPELINE INSPEKSI SISI SAMPING KONVEYOR (CAM-01 / CAM-02)
==============================================================================
Deskripsi:
  Menyatukan deteksi Baut (Bolt), Pin, dan Lekukan Sisi Ujung (Bracket)
  ke dalam satu alur inspeksi otomatis terkoordinasi (Hybrid YOLO + PatchCore).
  
  Fitur:
  - Master ROI (Kuning) untuk zona deteksi Baut & Pin via YOLO.
  - Static Masked Polygon ROI (Poligon Diam) untuk inspeksi Lekukan Plat via PatchCore:
    * Bentuk poligon mengikuti kontur lengkungan miring plat baja.
    * Di-masking hitam pekat di luar poligon (100% identik dengan dataset training PatchCore).
    * Titik poligon diam di posisi gerbang inspeksi (tidak loncat-loncat mengikuti baut).
    * Diinspeksi otomatis saat baut melintasi garis pemicu (Trigger Gate).
    * Titik poligon dan Master ROI dapat digeser/disesuaikan dengan mouse drag.
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

from bolt_pin_detector import BoltPinDetector
from bracket_anomaly_detector import BracketAnomalyDetector


class SideConveyorInspector:
    """
    Sistem Terpadu Inspeksi Komponen Samping Slat Conveyor dengan Static Masked Polygon ROI.
    """
    def __init__(
        self,
        camera_id="CAM-01",
        side="RIGHT",
        yolo_model_path=None,
        patchcore_model_path=None,
        master_roi_config=None,
        polygon_config=None,
        conf_thresh=0.25,
        patchcore_thresh=None
    ):
        self.camera_id = camera_id
        self.side = side.upper()

        # Path default jika tidak ditentukan
        base_dir = os.path.dirname(CURRENT_DIR)
        if yolo_model_path is None:
            yolo_model_path = os.path.join(base_dir, "models", "best_v2.pt")
        if patchcore_model_path is None:
            patchcore_model_path = os.path.join(base_dir, "models", "patchcore_bracket_model.pkl")
        if master_roi_config is None:
            master_roi_config = os.path.join(base_dir, "configs", "hybrid_master_roi_config.json")
        if polygon_config is None:
            polygon_config = os.path.join(base_dir, "configs", "bracket_polygon_config.json")

        print("=" * 60)
        print(f"[*] Menginisialisasi Side Conveyor Inspector: [{self.camera_id} - {self.side}]")
        print("=" * 60)

        # 1. Inisialisasi Detektor Baut & Pin (YOLO)
        self.bolt_pin_detector = BoltPinDetector(yolo_model_path, conf_threshold=conf_thresh, max_pin_distance=140.0)

        # 2. Inisialisasi Detektor Lekukan (PatchCore)
        self.bracket_detector = BracketAnomalyDetector(patchcore_model_path, default_thresh=0.8500)
        self.custom_patchcore_thresh = patchcore_thresh or 0.8500

        # 3. Muat Konfigurasi Master ROI (Zona YOLO)
        self.master_roi_config_path = master_roi_config
        self.default_master_roi = [0.351, 0.134, 0.541, 0.747]
        self.master_roi = self._load_master_roi()

        # 4. Muat Konfigurasi Poligon Diam (Zona PatchCore)
        self.polygon_config_path = polygon_config
        self.default_polygon = [
            [0.452, 0.303],
            [0.499, 0.340],
            [0.493, 0.467],
            [0.444, 0.446]
        ]
        self.polygon_pts = self._load_polygon_config()

        # 5. Sensor Virtual Photocell (Metode 1: Optical Probe pada Badan Pelat Slat)
        self.photocell_thresh = 75.0           # Ambang batas intensitas logam (Gap < 45, Pelat > 85)
        self.photocell_was_present = False     # Latch deteksi rising/falling edge
        self.slat_settling_delay = 1           # Delay frame awal setelah edge agar plat mapan di poligon
        self.slat_settling_countdown = 0       # Countdown delay settling
        self.slat_audit_countdown = 0          # Countdown frame audit aktif saat plat di poligon
        self.slat_audit_window = 6             # Window audit optimal saat plat mengisi poligon
        self.current_slat_scores = []          # Buffer skor anomali slat aktif
        self.slat_cycle_count = 0              # Akumulator total slat yang melintas
        self.last_slat_verdict = "WAIT"        # Hasil akhir slat terakhir
        self.last_slat_score = 0.0             # Skor maksimum slat terakhir

        # State audit & Latch
        self.last_bracket_info = None
        self.last_audit_frame = -999
        self.last_ng_frame = -999
        self.latched_ng_info = None

        # Statistik Sesi
        self.stats = {
            "total_frames": 0,
            "total_slats": 0,
            "ok_count": 0,
            "ng_count": 0,
            "invalid_count": 0
        }

    def reset_session(self):
        """Mereset state inspeksi dan statistik sesi saat video baru dimuat atau di-restart."""
        self.photocell_was_present = False
        self.slat_settling_countdown = 0
        self.slat_audit_countdown = 0
        self.current_slat_scores = []
        self.slat_cycle_count = 0
        self.last_slat_verdict = "WAIT"
        self.last_slat_score = 0.0
        self.last_bracket_info = None
        self.last_audit_frame = -999
        self.last_ng_frame = -999
        self.latched_ng_info = None
        self.stats = {
            "total_frames": 0,
            "total_slats": 0,
            "ok_count": 0,
            "ng_count": 0,
            "invalid_count": 0
        }


    def _load_master_roi(self):
        """Memuat koordinat Master ROI dari file JSON."""
        if os.path.exists(self.master_roi_config_path):
            try:
                with open(self.master_roi_config_path, "r") as f:
                    data = json.load(f)
                    if "master_roi" in data:
                        return data["master_roi"]
            except Exception as e:
                print(f"[WARN] Gagal membaca Master ROI config: {e}")
        return list(self.default_master_roi)

    def _load_polygon_config(self):
        """Memuat titik poligon lekukan plat dari file JSON."""
        if os.path.exists(self.polygon_config_path):
            try:
                with open(self.polygon_config_path, "r") as f:
                    data = json.load(f)
                    if "polygon_pts" in data and len(data["polygon_pts"]) >= 3:
                        return data["polygon_pts"]
            except Exception as e:
                print(f"[WARN] Gagal membaca Polygon config: {e}")
        return [list(p) for p in self.default_polygon]

    def save_config(self):
        """Menyimpan konfigurasi Master ROI dan Polygon ROI ke file JSON masing-masing."""
        try:
            with open(self.master_roi_config_path, "w") as f:
                json.dump({"master_roi": self.master_roi}, f, indent=2)
            with open(self.polygon_config_path, "w") as f:
                json.dump({"polygon_pts": self.polygon_pts}, f, indent=2)
            print(f"[*] Konfigurasi ROI & Poligon tersimpan otomatis.")
        except Exception as e:
            print(f"[ERROR] Gagal menyimpan konfigurasi: {e}")

    def _get_pixel_roi(self, w, h):
        """Konversi Master ROI ke integer piksel."""
        rx1, ry1, rx2, ry2 = self.master_roi
        return int(rx1 * w), int(ry1 * h), int(rx2 * w), int(ry2 * h)

    def extract_masked_polygon_crop(self, frame):
        """
        Memotong area lekukan plat berbentuk poligon dengan latar belakang luar hitam pekat.
        Format ini identik 100% dengan dataset training PatchCore.
        """
        h, w = frame.shape[:2]
        pts_px = np.array([[int(p[0] * w), int(p[1] * h)] for p in self.polygon_pts], dtype=np.int32)
        bx, by, bw, bh = cv2.boundingRect(pts_px)
        if bw < 10 or bh < 10:
            return None, None, pts_px, (bx, by, bw, bh)

        # Buat masking poligon
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts_px], 255)
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

        # Potong area bounding box dari masked image
        crop_poly = masked_frame[by:by+bh, bx:bx+bw]
        crop_mask = mask[by:by+bh, bx:bx+bw]
        return crop_poly, crop_mask, pts_px, (bx, by, bw, bh)

    def process_frame(self, frame):
        """
        Memproses satu frame untuk inspeksi lengkap Baut, Pin, dan Lekukan Poligon Diam.
        :param frame: Citra BGR (np.ndarray)
        :return: (display_frame, inspection_payload)
        """
        if frame is None or frame.size == 0:
            return frame, {"overall_result": "INVALID", "message": "Frame Kosong"}

        self.stats["total_frames"] += 1
        h_frame, w_frame = frame.shape[:2]
        display_frame = frame.copy()

        # Dapatkan Koordinat Master ROI & Poligon
        rx1, ry1, rx2, ry2 = self._get_pixel_roi(w_frame, h_frame)
        crop_poly, crop_mask, pts_px, (bx, by, bw, bh) = self.extract_masked_polygon_crop(frame)

        # 1. Gambar Batas Master ROI (Zona Kuning - YOLO)
        cv2.rectangle(display_frame, (rx1, ry1), (rx2, ry2), (0, 230, 255), 2)
        cv2.putText(display_frame, f"MASTER ROI ({self.side})", (rx1 + 5, max(20, ry1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 255), 2)

        # 2. Jalankan Deteksi Baut & Pin pada Master ROI via YOLO
        det_res = self.bolt_pin_detector.detect(frame, roi=(rx1, ry1, rx2, ry2))
        bolts = det_res["bolts"]
        pins = det_res["pins"]
        pairs = det_res["pairs"]

        # Evaluasi Bolt & Pin
        bolt_verdict = "OK" if any(not b["is_ng"] for b in bolts) else ("NG" if any(b["is_ng"] for b in bolts) else "EMPTY")
        pin_verdict = "OK" if any(not p["is_ng"] for p in pins) else ("NG" if any(p["is_ng"] for p in pins) else "EMPTY")

        # Cek jika ada komponen yang eksplisit cacat
        for b in bolts:
            if b["is_ng"]:
                bolt_verdict = "NG"

        for p in pins:
            if p["is_ng"]:
                pin_verdict = "NG"

        for pair in pairs:
            if pair["pin"] is not None and pair["pin"]["is_ng"]:
                pin_verdict = "NG"

        # Visualisasi Baut & Pin
        for pair in pairs:
            b = pair["bolt"]
            p = pair["pin"]
            bx_1, by_1, bx_2, by_2 = b["box"]

            b_color = (0, 0, 255) if b["is_ng"] else (255, 180, 0)
            cv2.rectangle(display_frame, (bx_1, by_1), (bx_2, by_2), b_color, 2)
            cv2.putText(display_frame, f"{b['class_name']} {b['confidence']:.2f}",
                        (bx_1, max(15, by_1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, b_color, 2)

            if p is not None:
                px1, py1, px2, py2 = p["box"]
                p_color = (0, 0, 255) if p["is_ng"] else (0, 255, 128)
                cv2.rectangle(display_frame, (px1, py1), (px2, py2), p_color, 2)
                cv2.putText(display_frame, f"{p['class_name']}",
                            (px1, max(15, py1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, p_color, 1)
                cv2.line(display_frame, b["centroid"], p["centroid"], (255, 255, 0), 1)

        # 3. Virtual Photocell / Optical Probe (Metode 1: Sensor Kehadiran Pelat Slat)
        # Menghitung koordinat adaptif sensor probe di samping atas poligon pelat
        min_poly_x = min(p[0] for p in self.polygon_pts)
        min_poly_y = min(p[1] for p in self.polygon_pts)
        px1 = int(max(0.01, min_poly_x - 0.035) * w_frame)
        px2 = int(min_poly_x * w_frame)
        py = int(min(0.99, min_poly_y + 0.025) * h_frame)

        # Mengukur kecerahan refleksi logam pada garis sensor photocell
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        beam_roi = gray_frame[max(0, py - 2):min(h_frame, py + 2), px1:px2]
        beam_val = float(np.mean(beam_roi)) if beam_roi.size > 0 else 0.0
        is_plate_present = (beam_val >= self.photocell_thresh)

        # Deteksi Rising Edge: Pelat baru tiba di gerbang inspeksi
        if is_plate_present and not self.photocell_was_present:
            self.slat_cycle_count += 1
            self.stats["total_slats"] = self.slat_cycle_count
            self.slat_settling_countdown = self.slat_settling_delay
            self.slat_audit_countdown = self.slat_audit_window
            self.current_slat_scores = []

        # Deteksi Falling Edge: Pelat meninggalkan gerbang menuju gap antar slat
        if not is_plate_present and self.photocell_was_present:
            self.slat_settling_countdown = 0
            self.slat_audit_countdown = 0
            if self.current_slat_scores:
                max_score = max(self.current_slat_scores)
                self.last_slat_score = max_score
                self.last_slat_verdict = "NG" if max_score >= self.custom_patchcore_thresh else "OK"

        self.photocell_was_present = is_plate_present

        # Visualisasi Sensor Virtual Photocell pada Layar
        beam_color = (0, 255, 255) if is_plate_present else (100, 100, 100)
        cv2.line(display_frame, (px1, py), (px2, py), beam_color, 2)
        cv2.circle(display_frame, (px1, py), 4, beam_color, -1)
        cv2.circle(display_frame, (px2, py), 4, beam_color, -1)
        sensor_status_text = f"PHOTOCELL: {'ON [PLAT]' if is_plate_present else 'OFF [GAP]'} ({beam_val:.1f})"
        cv2.putText(display_frame, sensor_status_text, (px1 - 20, max(15, py - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, beam_color, 1)

        # 4. Pemicuan & Audit Inspeksi Lekukan Plat (PADA POLIGON DIAM MASKED)
        inspected_bracket_info = None

        if is_plate_present and crop_poly is not None and crop_poly.size > 0:
            if self.slat_settling_countdown > 0:
                self.slat_settling_countdown -= 1
                lekukan_verdict = "WAIT"
                lek_color = (0, 215, 255)
                label_text = "LEKUKAN: MEMASUKI POLIGON"
            elif self.slat_audit_countdown > 0:
                self.slat_audit_countdown -= 1
                # Slat tepat berada di zona inspeksi -> Audit crop poligon
                patch_res = self.bracket_detector.inspect_crop(
                    crop_poly, custom_thresh=self.custom_patchcore_thresh
                )

                raw_score = patch_res["score"]
                raw_status = patch_res["status"]
                self.current_slat_scores.append(raw_score)

                inspected_bracket_info = {
                    "box": [bx, by, bx + bw, by + bh],
                    "score": raw_score,
                    "threshold": patch_res["threshold"],
                    "status": raw_status,
                    "overlay": patch_res.get("overlay")
                }

                self.last_bracket_info = inspected_bracket_info
                self.last_audit_frame = self.stats["total_frames"]

                lekukan_verdict = raw_status
                lek_color = (0, 0, 255) if raw_status == "NG" else (0, 255, 0)
                label_text = f"LEKUKAN: {raw_status} [{raw_score:.3f}] (AUDIT)"
            else:
                # Slat masih melintas setelah window audit selesai -> Tampilkan hasil slat ini
                has_ng = any(s >= self.custom_patchcore_thresh for s in self.current_slat_scores)
                cur_verdict = "NG" if has_ng else "OK"
                cur_score = max(self.current_slat_scores) if self.current_slat_scores else 0.0
                lekukan_verdict = cur_verdict
                lek_color = (0, 0, 255) if cur_verdict == "NG" else (0, 255, 0)
                label_text = f"LEKUKAN: {cur_verdict} [{cur_score:.3f}]"
        else:
            # Di luar gerbang (celah transisi antar slat) -> Tahan hasil slat terakhir
            recent_audit = (self.last_bracket_info is not None and 
                            (self.stats["total_frames"] - self.last_audit_frame) < 25)
            if recent_audit and self.last_slat_verdict != "WAIT":
                lekukan_verdict = self.last_slat_verdict
                lek_color = (0, 0, 255) if lekukan_verdict == "NG" else (0, 255, 0)
                label_text = f"LEKUKAN: {lekukan_verdict} [{self.last_slat_score:.3f}] (LAST)"
            else:
                lekukan_verdict = "WAIT"
                lek_color = (0, 215, 255)  # Cyan standby
                label_text = "LEKUKAN POLIGON [STANDBY]"

        # Gambar Garis Poligon Diam di Layar
        cv2.polylines(display_frame, [pts_px], isClosed=True, color=lek_color, thickness=3)

        # Gambar Titik-titik Sudut Poligon (Interactive Handles)
        for i, p in enumerate(pts_px):
            cv2.circle(display_frame, (int(p[0]), int(p[1])), 5, (0, 255, 255), -1)
            cv2.circle(display_frame, (int(p[0]), int(p[1])), 7, (0, 0, 0), 1)

        # Badge Label di atas Poligon
        cv2.rectangle(display_frame, (bx, max(0, by - 24)), (bx + 260, by), (20, 20, 20), -1)
        cv2.putText(display_frame, label_text, (bx + 5, max(16, by - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, lek_color, 2)

        # 4. Tentukan Overall Verdict Sisi Ini
        # Prioritas: NG > INVALID > OK > WAIT
        if bolt_verdict == "NG" or pin_verdict == "NG" or lekukan_verdict == "NG":
            overall_side = "NG"
        elif bolt_verdict == "EMPTY" and pin_verdict == "EMPTY" and lekukan_verdict == "WAIT":
            overall_side = "WAIT"
        elif (bolt_verdict in ["OK", "EMPTY"]) and (pin_verdict in ["OK", "EMPTY"]) and (lekukan_verdict in ["OK", "WAIT"]):
            overall_side = "OK"
        else:
            overall_side = "INVALID"

        # Update Statistik jika ada inspeksi aktif
        if overall_side in ["OK", "NG", "INVALID"]:
            if overall_side == "OK":
                self.stats["ok_count"] += 1
            elif overall_side == "NG":
                self.stats["ng_count"] += 1
            else:
                self.stats["invalid_count"] += 1

        # 5. Format Payload Sesuai Standar PRD
        payload = {
            "camera_id": self.camera_id,
            "side": self.side,
            "overall_result": overall_side,
            "components": [
                {"name": "BOLT", "result": bolt_verdict},
                {"name": "PIN", "result": pin_verdict},
                {"name": "LEKUKAN", "result": lekukan_verdict}
            ],
            "bracket_details": inspected_bracket_info or self.last_bracket_info
        }

        # 6. Gambar HUD Panel Atas
        self._draw_hud(display_frame, payload)

        # 7. Gambar Mini PIP Heatmap Preview di Pojok Kanan Atas
        if self.last_bracket_info is not None and self.last_bracket_info.get("overlay") is not None:
            ov = self.last_bracket_info["overlay"]
            if ov is not None and ov.size > 0:
                pip_w, pip_h = 160, 160
                pip_resized = cv2.resize(ov, (pip_w, pip_h))
                px1 = max(0, w_frame - pip_w - 20)
                py1 = 85
                px2 = px1 + pip_w
                py2 = py1 + pip_h
                display_frame[py1:py2, px1:px2] = pip_resized
                ov_col = (0, 255, 0) if self.last_bracket_info["status"] == "OK" else (0, 0, 255)
                cv2.rectangle(display_frame, (px1, py1), (px2, py2), ov_col, 2)
                cv2.rectangle(display_frame, (px1, py1 - 20), (px2, py1), (20, 20, 20), -1)
                cv2.putText(display_frame, f"HEATMAP: {self.last_bracket_info['status']}", (px1 + 4, py1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, ov_col, 1)

        return display_frame, payload

    def _draw_hud(self, img, payload):
        """Menggambar banner status HUD di bagian atas layar."""
        h, w = img.shape[:2]
        hud_h = 75
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_h), (25, 25, 28), -1)
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)

        res = payload["overall_result"]
        res_color = (0, 255, 0) if res == "OK" else ((0, 0, 255) if res == "NG" else (0, 200, 255))

        # Judul & Status Utama
        cv2.putText(img, f"CONVEYOR SIDE INSPECTION [{self.camera_id} - {self.side}]",
                    (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(img, f"STATUS: {res}", (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, res_color, 2)

        # Rincian Komponen
        c_x = int(w * 0.45)
        for c in payload["components"]:
            c_col = (0, 255, 0) if c["result"] == "OK" else ((0, 0, 255) if c["result"] == "NG" else (180, 180, 180))
            txt = f"{c['name']}: {c['result']}"
            cv2.putText(img, txt, (c_x, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.65, c_col, 2)
            c_x += 160


def select_video_file(initial_dir=None, title="Pilih File Video Inspeksi"):
    """
    Membuka dialog file explorer Windows/OS untuk memilih file video.
    Menggunakan tkinter secara terisolasi tanpa mengganggu loop GUI OpenCV.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()  # Sembunyikan window utama tkinter
        root.attributes("-topmost", True)  # Pastikan dialog selalu berada di depan

        default_dir = initial_dir
        if not default_dir or not os.path.exists(default_dir):
            candidate_dirs = [
                r"C:\Users\Victus\Pictures\Video & Foto Test Slat Rib",
                os.path.join(os.path.expanduser("~"), "Pictures"),
                os.path.join(os.path.expanduser("~"), "Videos"),
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
                ("AVI Video (*.avi)", "*.avi"),
                ("MKV Video (*.mkv)", "*.mkv"),
                ("Semua File (*.*)", "*.*")
            ]
        )
        root.destroy()
        return file_path if file_path and os.path.exists(file_path) else None
    except Exception as e:
        print(f"[WARN] Dialog file tidak dapat dibuka ({e}). Silakan tentukan file via argumen/konsole.")
        return None


def run_side_inspector_demo(video_path=None, side="RIGHT", camera_id="CAM-01"):
    """
    Fungsi demo interaktif untuk menguji video konveyor sisi samping.
    Mendukung:
    - Pemilihan file video baru kapan saja via shortcut keyboard [O] / [F]
    - Penyesuaian titik sudut Poligon Diam & Master ROI menggunakan Mouse Drag
    """
    inspector = SideConveyorInspector(camera_id=camera_id, side=side)

    # Validasi path video, jika belum ada / salah, buka dialog pemilih file
    if not video_path or not os.path.exists(video_path):
        print("[*] Path video tidak ditemukan. Membuka dialog pemilih file video...")
        video_path = select_video_file(title="Pilih File Video untuk Inspeksi")
        if not video_path:
            print("[ERROR] Tidak ada video yang dipilih. Demo dihentikan.")
            return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Gagal membuka video di: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[*] Membuka Video: {os.path.basename(video_path)} ({total_frames} frame, {fps:.1f} FPS)")

    win_name = f"Conveyor Side Inspector - [{camera_id} {side}]"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 720)

    # State Mouse Interaction
    mouse_state = {
        "selected_poly_idx": None,
        "is_moving_poly": False,
        "drag_start_pos": None,
        "drag_start_poly": None,
        "master_drag_mode": None,   # 'move' atau 'resize'
        "master_start_pos": None,
        "master_start_roi": None
    }

    def mouse_cb(event, x, y, flags, param):
        w_f, h_f = param
        nx, ny = max(0.01, min(0.99, x / float(w_f))), max(0.01, min(0.99, y / float(h_f)))

        mx1_n, my1_n, mx2_n, my2_n = inspector.master_roi
        m_px1, m_py1 = int(mx1_n * w_f), int(my1_n * h_f)
        m_px2, m_py2 = int(mx2_n * w_f), int(my2_n * h_f)

        if event == cv2.EVENT_LBUTTONDOWN:
            # 1. Cek Titik Sudut Poligon (Drag Titik Sudut)
            for i, (px_n, py_n) in enumerate(inspector.polygon_pts):
                px, py = int(px_n * w_f), int(py_n * h_f)
                if np.hypot(x - px, y - py) < 18:
                    mouse_state["selected_poly_idx"] = i
                    return

            # 2. Cek Bagian Dalam Poligon (Geser Seluruh Poligon)
            pts_px = np.array([[int(p[0] * w_f), int(p[1] * h_f)] for p in inspector.polygon_pts], dtype=np.int32)
            if cv2.pointPolygonTest(pts_px, (x, y), False) >= 0:
                mouse_state["is_moving_poly"] = True
                mouse_state["drag_start_pos"] = (nx, ny)
                mouse_state["drag_start_poly"] = [list(p) for p in inspector.polygon_pts]
                return

            # 3. Cek Sudut Kanan-Bawah Master ROI (Resize)
            if np.hypot(x - m_px2, y - m_py2) < 22:
                mouse_state["master_drag_mode"] = "resize"
                return

            # 4. Cek Dalam Master ROI (Geser Master ROI)
            if (m_px1 < x < m_px2) and (m_py1 < y < m_py2):
                mouse_state["master_drag_mode"] = "move"
                mouse_state["master_start_pos"] = (nx, ny)
                mouse_state["master_start_roi"] = list(inspector.master_roi)
                return

        elif event == cv2.EVENT_MOUSEMOVE:
            # Drag Titik Poligon
            if mouse_state["selected_poly_idx"] is not None:
                inspector.polygon_pts[mouse_state["selected_poly_idx"]] = [nx, ny]

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
                inspector.polygon_pts = new_poly

            # Drag Master ROI
            elif mouse_state["master_drag_mode"] == "move" and mouse_state["master_start_pos"]:
                dx = nx - mouse_state["master_start_pos"][0]
                dy = ny - mouse_state["master_start_pos"][1]
                s = mouse_state["master_start_roi"]
                mw = s[2] - s[0]
                mh = s[3] - s[1]
                new_x1 = max(0.01, min(0.99 - mw, s[0] + dx))
                new_y1 = max(0.01, min(0.99 - mh, s[1] + dy))
                inspector.master_roi = [new_x1, new_y1, new_x1 + mw, new_y1 + mh]

            elif mouse_state["master_drag_mode"] == "resize":
                inspector.master_roi[2] = max(inspector.master_roi[0] + 0.05, nx)
                inspector.master_roi[3] = max(inspector.master_roi[1] + 0.05, ny)

        elif event == cv2.EVENT_LBUTTONUP:
            if (mouse_state["selected_poly_idx"] is not None or 
                mouse_state["is_moving_poly"] or 
                mouse_state["master_drag_mode"] is not None):
                inspector.save_config()

            mouse_state["selected_poly_idx"] = None
            mouse_state["is_moving_poly"] = False
            mouse_state["drag_start_pos"] = None
            mouse_state["drag_start_poly"] = None
            mouse_state["master_drag_mode"] = None
            mouse_state["master_start_pos"] = None
            mouse_state["master_start_roi"] = None

    # Trackbar Threshold & Timeline
    def on_thresh(val):
        inspector.custom_patchcore_thresh = val / 1000.0

    init_th = int(inspector.bracket_detector.calibrated_threshold * 1000)
    cv2.createTrackbar("PatchCore Thresh (x1000)", win_name, init_th, 1500, on_thresh)

    def on_timeline(val):
        cap.set(cv2.CAP_PROP_POS_FRAMES, val)

    cv2.createTrackbar("Timeline", win_name, 0, max(1, total_frames - 1), on_timeline)

    is_paused = False
    delay_ms = max(1, int(1000 / fps))

    print("\n" + "=" * 75)
    print("   SISTEM INSPEKSI SISI SAMPING (HYBRID YOLO + STATIC POLYGON MASK)")
    print("=" * 75)
    print("  - [O] / [F]            : PILIH / GANTI FILE VIDEO LAIN (File Explorer)")
    print("  - POLIGON LEKUKAN      : Poligon diam mengikuti lekukan miring plat")
    print("  - DRAG TITIK POLIGON   : Klik & geser titik sudut kuning untuk membentuk lekukan")
    print("  - DRAG DALAM POLIGON   : Klik & geser untuk memindahkan seluruh poligon")
    print("  - [SPACE]              : Pause / Play video")
    print("  - [D] / [A]            : Maju / Mundur 1 frame saat pause")
    print("  - [R]                  : Reset posisi poligon & Master ROI ke default")
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
        else:
            cur_pos = cv2.getTrackbarPos("Timeline", win_name)
            cap.set(cv2.CAP_PROP_POS_FRAMES, cur_pos)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

        h_f, w_f = frame.shape[:2]
        cv2.setMouseCallback(win_name, mouse_cb, (w_f, h_f))

        display_frame, payload = inspector.process_frame(frame)

        # Petunjuk shortcut di bagian bawah frame
        help_bar = "[SPACE]: Pause/Play | [O]/[F]: Ganti Video | [D]/[A]: Frame Step | [R]: Reset ROI | [Q]: Keluar"
        cv2.putText(display_frame, help_bar, (15, h_f - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

        cv2.imshow(win_name, display_frame)

        key = cv2.waitKey(delay_ms if not is_paused else 30) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            is_paused = not is_paused
        elif key in [ord('o'), ord('O'), ord('f'), ord('F')]:
            # Buka dialog pemilih file video baru
            cur_dir = os.path.dirname(video_path) if video_path and os.path.exists(os.path.dirname(video_path)) else None
            new_video = select_video_file(initial_dir=cur_dir, title="Pilih File Video Baru untuk Diinspeksi")
            if new_video and os.path.exists(new_video):
                new_cap = cv2.VideoCapture(new_video)
                if new_cap.isOpened():
                    cap.release()
                    cap = new_cap
                    video_path = new_video
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    delay_ms = max(1, int(1000 / fps))
                    cv2.setTrackbarMax("Timeline", win_name, max(1, total_frames - 1))
                    cv2.setTrackbarPos("Timeline", win_name, 0)
                    inspector.reset_session()
                    print(f"\n[✓] Berhasil memuat video baru: {os.path.basename(video_path)} ({total_frames} frame, {fps:.1f} FPS)")
                else:
                    print(f"[ERROR] Gagal membuka file video baru: {new_video}")
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
            inspector.master_roi = list(inspector.default_master_roi)
            inspector.polygon_pts = [list(p) for p in inspector.default_polygon]
            inspector.save_config()
            print("[*] Konfigurasi ROI & Poligon di-reset ke default.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Side Conveyor Inspector Pipeline (Hybrid YOLO + PatchCore)")
    parser.add_argument("video", nargs="?", default=None, help="Path ke file video inspeksi (opsional)")
    parser.add_argument("--side", default="RIGHT", choices=["LEFT", "RIGHT"], help="Sisi kamera (LEFT / RIGHT)")
    parser.add_argument("--cam", default="CAM-01", help="ID Kamera (default: CAM-01)")
    args = parser.parse_args()

    video_input = args.video
    if not video_input:
        default_video = r"C:\Users\Victus\Pictures\Video & Foto Test Slat Rib\Opsi Posisi 1 - Bolt & Pin.mp4"
        if os.path.exists(default_video):
            video_input = default_video
        else:
            print("[*] Video default tidak ditemukan, membuka dialog pemilih file...")
            video_input = select_video_file(title="Pilih File Video untuk Inspeksi Samping")

    if video_input and os.path.exists(video_input):
        run_side_inspector_demo(video_input, side=args.side, camera_id=args.cam)
    else:
        print("[!] Tidak ada video yang dipilih. Program dihentikan.")

