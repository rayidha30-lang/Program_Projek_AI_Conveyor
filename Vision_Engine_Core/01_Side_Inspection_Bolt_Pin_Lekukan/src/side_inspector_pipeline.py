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


def _load_env_file():
    """Membaca file .env jika ada (tanpa ketergantungan library eksternal)"""
    candidate_paths = [
        os.path.join(CURRENT_DIR, ".env"),
        os.path.abspath(os.path.join(CURRENT_DIR, "..", ".env")),
        os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".env")),
        os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "..", ".env")),
        os.path.join(os.getcwd(), ".env"),
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

from bolt_pin_detector import BoltPinDetector
from bracket_anomaly_detector import BracketAnomalyDetector

# Impor Modul Jembatan Database SQLite & Notifier Teams
DB_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "..", "..", "Database"))
if not os.path.exists(DB_DIR):
    DB_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "..", "Database"))
if DB_DIR not in sys.path:
    sys.path.insert(0, DB_DIR)

try:
    from db_bridge import InspectionDBBridge
except ImportError:
    InspectionDBBridge = None


class SideConveyorInspector:
    """
    Sistem Terpadu Inspeksi Komponen Samping Slat Conveyor dengan Static Masked Polygon ROI.
    """
    def __init__(
        self,
        camera_id="CAM-01",
        side="RIGHT",
        preset="auto",
        yolo_model_path=None,
        patchcore_model_path=None,
        master_roi_config=None,
        polygon_config=None,
        conf_thresh=None,
        iou_thresh=None,
        patchcore_thresh=None,
        enable_db=False,
        device=None,
        enable_missing_check=None
    ):
        # Standar PRD & Frontend:
        # CAM-01: Top (Plat & Rib), CAM-02: Right Side (RH), CAM-03: Left Side (LH)
        side_norm = (side or "RIGHT").upper().strip()
        if not camera_id or camera_id == "CAM-01":
            if side_norm in ["RIGHT", "RH", "KANAN"]:
                camera_id = "CAM-02"
            elif side_norm in ["LEFT", "LH", "KIRI"]:
                camera_id = "CAM-03"
            else:
                camera_id = "CAM-02"
        self.camera_id = camera_id
        self.side = side_norm
        self.device = device

        # Resolusi Preset Lingkungan Kamera (opsi_1 / opsi_2 / auto)
        # Prioritas: Argumen konstruktor eksplisit > Environment variable .env > Default auto
        env_preset = (os.environ.get("SIDE_POSITION_PRESET") or os.environ.get("SIDE_CAMERA_PRESET") or "").strip().lower()
        if preset and str(preset).strip().lower() not in ["auto", ""]:
            preset_raw = str(preset).strip().lower()
        elif env_preset:
            preset_raw = env_preset
        else:
            preset_raw = "auto"

        if preset_raw in ["1", "opsi1", "opsi_1", "posisi1", "posisi_1"]:
            preset_clean = "opsi_1"
            is_auto = False
        elif preset_raw in ["2", "opsi2", "opsi_2", "posisi2", "posisi_2"]:
            preset_clean = "opsi_2"
            is_auto = False
        elif preset_raw == "auto":
            preset_clean = "opsi_1"
            is_auto = True
        else:
            preset_clean = "opsi_1"
            is_auto = False
        self.preset = preset_clean
        self.auto_preset_mode = is_auto

        # Ambang batas deteksi Baut & Pin (Prioritaskan argumen -> .env -> default 0.55 / 0.35)
        if conf_thresh is None:
            try:
                conf_thresh = float(os.environ.get("BOLT_CONF_THRESHOLD", "0.55"))
            except ValueError:
                conf_thresh = 0.55
        if iou_thresh is None:
            try:
                iou_thresh = float(os.environ.get("BOLT_IOU_THRESHOLD", "0.35"))
            except ValueError:
                iou_thresh = 0.35
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        # Saklar Feature Flag Deteksi Baut & Pin Hilang (Default: True, bisa diatur via .env / argumen)
        if enable_missing_check is None:
            env_missing = (os.environ.get("ENABLE_MISSING_BOLT_PIN_CHECK") or os.environ.get("ENABLE_MISSING_CHECK") or "true").strip().lower()
            enable_missing_check = env_missing in ("true", "1", "yes", "on")
        self.enable_missing_check = bool(enable_missing_check)

        # Inisialisasi Database Bridge (SQLite, Captures & MS Teams)
        self.db_bridge = None
        if InspectionDBBridge is not None:
            try:
                self.db_bridge = InspectionDBBridge(enabled=enable_db, camera_id=self.camera_id)
            except Exception as e_bridge:
                print(f"[WARN] Inisialisasi InspectionDBBridge gagal: {e_bridge}")

        # Path default jika tidak ditentukan
        base_dir = os.path.dirname(CURRENT_DIR)
        self.base_dir = base_dir

        if yolo_model_path is None:
            # 1. Cek environment variable MODEL_SIDE_PATH jika didefinisikan di .env
            env_side_model = os.environ.get("MODEL_SIDE_PATH")
            if env_side_model and os.path.exists(env_side_model):
                yolo_model_path = env_side_model
            else:
                # 2. Prioritaskan kandidat model aktif (best_v26_ok)
                for cand_name in [
                    "best_v26_ok.engine", "best_v26_ok.onnx", "best_v26_ok.pt",
                ]:
                    cand_p = os.path.join(base_dir, "models", cand_name)
                    if os.path.exists(cand_p):
                        yolo_model_path = cand_p
                        break
                if yolo_model_path is None:
                    yolo_model_path = os.path.join(base_dir, "models", "best_v26_ok.pt")
        self.yolo_model_path = yolo_model_path

        # Konfigurasi Master ROI (Zona YOLO)
        self.master_roi_config_path = master_roi_config or os.path.join(base_dir, "configs", "hybrid_master_roi_config.json")
        self.default_master_roi = [0.351, 0.134, 0.541, 0.747]
        self.master_roi = self._load_master_roi()

        # Nilai override eksplisit jika pengguna menentukan argumen khusus
        self._user_polygon_config = polygon_config
        self._user_patchcore_model = patchcore_model_path
        self._user_patchcore_thresh = patchcore_thresh

        # Nilai default poligon dasar (Fallback)
        self.default_polygon = [
            [0.452, 0.303],
            [0.499, 0.340],
            [0.493, 0.467],
            [0.444, 0.446]
        ]

        if device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    print("[WARN SideConveyorInspector] CUDA diminta tapi tidak tersedia di PyTorch. Fallback ke CPU.")
                    device = "cpu"
            except Exception:
                device = "cpu"
        elif device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        self.device = device

        print("=" * 60)
        print(f"[*] Menginisialisasi Side Conveyor Inspector: [{self.camera_id} - {self.side}]")
        print("=" * 60)

        # 1. Inisialisasi Detektor Baut & Pin (YOLO)
        self.bolt_pin_detector = BoltPinDetector(
            self.yolo_model_path,
            conf_threshold=self.conf_thresh,
            iou_threshold=self.iou_thresh,
            max_pin_distance=140.0,
            agnostic_nms=True,
            device=self.device
        )

        # 2. Inisialisasi Preset (Polygon, PatchCore Model, & Photocell Thresh)
        self.set_preset(self.preset, verbose=True)

        # 5. Sensor Virtual Photocell (Hysteresis Schmitt Trigger & Debounce Persistence Filter)
        self.gap_debounce_frames = 2           # Debounce gap: intensitas celah bertahan minimal 2 frame
        self.plat_debounce_frames = 2          # Debounce plat: intensitas plat masuk bertahan minimal 2 frame
        self.max_slat_duration_frames = 45     # Watchdog anti-stuck: auto-release ke GAP jika plat tertahan >= 45 frame
        self.consecutive_gap_frames = 0        # Counter persistensi celah gelap
        self.consecutive_plat_frames = 0       # Counter persistensi plat masuk
        self.min_audit_samples_required = 2    # Minimal sampel audit sweet spot agar vonis sah dan kebal noise parsial
        self.photocell_state = False           # State sensor stabil (True = PLAT, False = GAP)
        self.slat_settling_delay = 2           # Settling delay 2 frame setelah pemicuan photocell agar braket tepat di tengah poligon
        self.slat_settling_countdown = 0       # Countdown delay settling
        self.slat_audit_countdown = 0          # Countdown frame audit aktif saat plat di poligon
        self.slat_audit_window = 4             # Window audit 4 frame optimal saat plat mengisi poligon (sweet spot)
        self.current_slat_scores = []          # Buffer skor anomali slat aktif
        self.current_slat_samples = []         # Buffer snapshot audit (score, status, raw_frame, ann_frame, bracket_info)
        self.slat_evaluated = False            # Latch vonis selesai untuk slat aktif
        self.sweet_spot_sample = None          # Frame sweet spot terbaik dari jendela audit
        self.slat_cycle_count = 0              # Akumulator total slat yang melintas
        self.last_slat_verdict = "WAIT"        # Hasil akhir slat terakhir
        self.last_slat_score = 0.0             # Skor representatif slat terakhir
        self.logged_slat_cycle = -1            # Entity latch: 1 slat fisik NG hanya di-emit 1 kali!
        self.last_emit_frame = -999            # Frame timestamp emisi terakhir untuk transit cooldown
        self.min_slat_transit_frames = 15      # Minimal transit frames antar-emisi (deduplikasi fisik)
        self.last_slat_trigger_frame = -999    # Frame timestamp saat pemicuan plat terakhir
        self.min_slat_interval_frames = 15     # Minimal frame interval antar plat fisik (cegah noise bolt ulir)
        self.has_seen_gap = False              # Synchronizer: abaikan plat terpotong pada awal start video/stream
        self.last_frame_time = None            # Timestamp presisi untuk kalkulasi inter-frame delta
        self.inter_frame_dt = 0.033            # Delta waktu antar-frame aktual (deteksi jitter)

        # State audit & Latch
        self.last_bracket_info = None
        self.last_audit_frame = -999
        self.last_ng_frame = -999
        self.latched_ng_info = None
        self.last_counted_slat_cycle = -1      # Latch penghitungan entitas fisik slat (Single Entity Latch)
        self.current_slat_has_bolt_ng = False  # Akumulator defek baut pada slat aktif
        self.current_slat_has_pin_ng = False   # Akumulator defek pin pada slat aktif
        self.current_slat_has_crack_ng = False # Akumulator defek lekukan/crack pada slat aktif
        self.current_slat_bolt_seen = False    # Penanda visual komponen baut terlihat di ROI
        self.current_slat_pin_seen = False     # Penanda visual komponen pin terlihat di ROI
        self.last_slat_bolt_status = "WAIT"    # Hasil akhir komponen baut slat sebelumnya
        self.last_slat_pin_status = "WAIT"     # Hasil akhir komponen pin slat sebelumnya

        # Statistik Sesi (Per-Slat Entity QC Counters)
        self.stats = {
            "total_frames": 0,
            "total_slats": 0,
            "ok_count": 0,
            "ng_count": 0,
            "frame_ok_count": 0,
            "frame_ng_count": 0,
            "invalid_count": 0,
            "bolt_ng": 0,
            "pin_ng": 0,
            "crack_ng": 0
        }
        self._bolt_ng_count = 0
        self._pin_ng_count = 0
        self._crack_ng_count = 0

    def reset_session(self):
        """Mereset state inspeksi dan statistik sesi saat video baru dimuat atau di-restart."""
        self.photocell_state = False
        self.consecutive_gap_frames = 0
        self.consecutive_plat_frames = 0
        self.slat_settling_countdown = 0
        self.slat_audit_countdown = 0
        self.current_slat_scores = []
        self.current_slat_samples = []
        self.slat_evaluated = False
        self.sweet_spot_sample = None
        self.slat_cycle_count = 0
        self.last_slat_verdict = "WAIT"
        self.last_slat_score = 0.0
        self.logged_slat_cycle = -1
        self.last_emit_frame = -999
        self.last_slat_trigger_frame = -999
        self.has_seen_gap = False
        self.last_frame_time = None
        self.inter_frame_dt = 0.033
        self.last_bracket_info = None
        self.last_audit_frame = -999
        self.last_ng_frame = -999
        self.latched_ng_info = None
        self.last_counted_slat_cycle = -1
        self.current_slat_has_bolt_ng = False
        self.current_slat_has_pin_ng = False
        self.current_slat_has_crack_ng = False
        self.current_slat_bolt_seen = False
        self.current_slat_pin_seen = False
        self.last_slat_bolt_status = "WAIT"
        self.last_slat_pin_status = "WAIT"
        self.stats = {
            "total_frames": 0,
            "total_slats": 0,
            "ok_count": 0,
            "ng_count": 0,
            "frame_ok_count": 0,
            "frame_ng_count": 0,
            "invalid_count": 0,
            "bolt_ng": 0,
            "pin_ng": 0,
            "crack_ng": 0
        }
        self._bolt_ng_count = 0
        self._pin_ng_count = 0
        self._crack_ng_count = 0

    def _finalize_slat_qc(self):
        """
        Finalisasi evaluasi kualitas untuk 1 entitas plat slat fisik secara tunggal (Single Entity Slat Latch).
        Memastikan 1 slat fisik hanya dihitung tepat 1 kali ke dalam statistik (OK atau NG).
        """
        if self.slat_cycle_count <= 0:
            return
        if self.slat_cycle_count == self.last_counted_slat_cycle:
            return

        # Pastikan evaluasi PatchCore (lekukan/crack) tuntas jika ada sampel skor
        if self.current_slat_scores and not self.slat_evaluated:
            rep_score = min(self.current_slat_scores)
            self.last_slat_score = rep_score
            self.last_slat_verdict = "NG" if rep_score >= self.custom_patchcore_thresh else "OK"
            self.slat_evaluated = True

        # Validasi ketiadaan komponen Baut & Pin (Missing Check) jika saklar diaktifkan
        if self.enable_missing_check:
            if not self.current_slat_bolt_seen:
                self.current_slat_has_bolt_ng = True
                self.last_slat_bolt_status = "NG"
            else:
                self.last_slat_bolt_status = "OK"

            if not self.current_slat_pin_seen:
                self.current_slat_has_pin_ng = True
                self.last_slat_pin_status = "NG"
            else:
                self.last_slat_pin_status = "OK"

        is_crack_ng = self.current_slat_has_crack_ng or (self.last_slat_verdict == "NG")
        is_bolt_ng = self.current_slat_has_bolt_ng
        is_pin_ng = self.current_slat_has_pin_ng

        is_slat_ng = (is_bolt_ng or is_pin_ng or is_crack_ng)

        if is_slat_ng:
            self.stats["ng_count"] += 1
            if is_bolt_ng:
                self.stats["bolt_ng"] = self.stats.get("bolt_ng", 0) + 1
                self._bolt_ng_count = self.stats["bolt_ng"]
            if is_pin_ng:
                self.stats["pin_ng"] = self.stats.get("pin_ng", 0) + 1
                self._pin_ng_count = self.stats["pin_ng"]
            if is_crack_ng:
                self.stats["crack_ng"] = self.stats.get("crack_ng", 0) + 1
                self._crack_ng_count = self.stats["crack_ng"]
        else:
            self.stats["ok_count"] += 1

        self.last_counted_slat_cycle = self.slat_cycle_count

    def close(self):
        """Menutup inspector, memfinalisasi slat aktif jika belum, dan membersihkan resource."""
        if self.slat_cycle_count > 0 and self.slat_cycle_count != self.last_counted_slat_cycle:
            self._finalize_slat_qc()
        if self.db_bridge is not None:
            self.db_bridge.close()

    @property
    def slat_ok_count(self):
        return self.stats.get("ok_count", 0)

    @property
    def slat_ng_count(self):
        return self.stats.get("ng_count", 0)

    @property
    def bolt_ng_count(self):
        return self.stats.get("bolt_ng", getattr(self, "_bolt_ng_count", 0))

    @property
    def pin_ng_count(self):
        return self.stats.get("pin_ng", getattr(self, "_pin_ng_count", 0))

    @property
    def crack_ng_count(self):
        return self.stats.get("crack_ng", getattr(self, "_crack_ng_count", 0))

    def toggle_db(self):
        """Membalik status saklar database logging & Teams notification (ON <-> OFF)."""
        if self.db_bridge is not None:
            return self.db_bridge.toggle()
        return False

    def is_db_enabled(self):
        """Mengecek apakah database logging aktif."""
        if self.db_bridge is not None:
            return self.db_bridge.is_enabled()
        return False

    def record_if_ng(self, frame_raw, frame_ann, side_payload):
        """
        Merekam frame ke SQLite, folder captures, dan mengirim notifikasi MS Teams jika ada defek side (NG).
        Menerapkan deduplikasi internal agar tidak spamming notifikasi (single-capture per slat cycle).
        """
        if self.db_bridge is None or not self.db_bridge.is_enabled():
            return False

        if side_payload and side_payload.get("should_emit", False) and side_payload.get("overall_result") == "NG":
            emit_raw = side_payload.get("emit_raw_frame", frame_raw)
            emit_ann = side_payload.get("emit_ann_frame", frame_ann)
            return self.db_bridge.emit_side_inspection(
                frame_raw=emit_raw,
                frame_ann=emit_ann,
                side_payload=side_payload
            )
        return False

    def close(self):
        """Membersihkan resource background database bridge."""
        if self.db_bridge is not None:
            try:
                self.db_bridge.close()
            except Exception:
                pass



    def resolve_preset_from_source(self, video_source):
        """Mendeteksi preset posisi (opsi_1 / opsi_2) dari nama/path video atau input camera."""
        if not video_source:
            return self.preset
        src_str = str(video_source).lower()
        if any(term in src_str for term in ["posisi 2", "posisi_2", "opsi 2", "opsi_2", "pos2"]):
            return "opsi_2"
        elif any(term in src_str for term in ["posisi 1", "posisi_1", "opsi 1", "opsi_1", "pos1"]):
            return "opsi_1"
        return self.preset

    def set_preset(self, preset_name, verbose=False):
        """
        Beralih preset posisi kamera (opsi_1 <-> opsi_2) secara dinamis.
        Menyesuaikan file konfigurasi poligon, bobot PatchCore, dan ambang photocell otomatis.
        """
        norm_p = "opsi_2" if str(preset_name).lower() in ["2", "opsi2", "posisi2", "posisi_2", "opsi_2"] else "opsi_1"
        self.preset = norm_p

        # 1. Tentukan Path Poligon ROI Sesuai Preset
        if self._user_polygon_config:
            self.polygon_config_path = self._user_polygon_config
        else:
            if norm_p == "opsi_2":
                self.polygon_config_path = os.path.join(self.base_dir, "configs", "bracket_polygon_config_pos2.json")
            else:
                self.polygon_config_path = os.path.join(self.base_dir, "configs", "bracket_polygon_config_pos1.json")
                if not os.path.exists(self.polygon_config_path):
                    self.polygon_config_path = os.path.join(self.base_dir, "configs", "bracket_polygon_config.json")
        self.polygon_pts = self._load_polygon_config()

        # 2. Tentukan Ambang Photocell Sesuai Pencahayaan Preset
        if norm_p == "opsi_2":
            th_h = os.environ.get("PHOTOCELL_POS2_HIGH") or os.environ.get("PHOTOCELL_THRESH_HIGH")
            th_l = os.environ.get("PHOTOCELL_POS2_LOW") or os.environ.get("PHOTOCELL_THRESH_LOW")
            self.photocell_thresh_high = float(th_h) if th_h else 105.0
            self.photocell_thresh_low = float(th_l) if th_l else 85.0
            default_patch_th = 0.8500
        else:
            th_h = os.environ.get("PHOTOCELL_POS1_HIGH") or os.environ.get("PHOTOCELL_THRESH_HIGH")
            th_l = os.environ.get("PHOTOCELL_POS1_LOW") or os.environ.get("PHOTOCELL_THRESH_LOW")
            self.photocell_thresh_high = float(th_h) if th_h else 72.0
            self.photocell_thresh_low = float(th_l) if th_l else 62.0
            default_patch_th = 0.8650

        # 3. Tentukan Model PatchCore & Anomaly Threshold
        self.custom_patchcore_thresh = self._user_patchcore_thresh or default_patch_th
        if self._user_patchcore_model:
            model_path = self._user_patchcore_model
        else:
            if norm_p == "opsi_2":
                cand_pos2 = os.path.join(self.base_dir, "models", "patchcore_bracket_model_pos2.pkl")
                model_path = cand_pos2 if os.path.exists(cand_pos2) else os.path.join(self.base_dir, "models", "patchcore_bracket_model.pkl")
            else:
                model_path = os.path.join(self.base_dir, "models", "patchcore_bracket_model.pkl")

        # Inisialisasi Bracket Detector
        self.bracket_detector = BracketAnomalyDetector(
            model_path,
            default_thresh=self.custom_patchcore_thresh,
            device=self.device
        )

        if verbose:
            print(f"[*] [SideConveyorInspector] Preset Aktif: [{self.preset.upper()}]")
            print(f"    - Polygon Config   : {os.path.basename(self.polygon_config_path)}")
            print(f"    - PatchCore Model  : {os.path.basename(model_path)}")
            print(f"    - Photocell Thresh : High={self.photocell_thresh_high:.1f}, Low={self.photocell_thresh_low:.1f}")
            print(f"    - PatchCore Thresh : {self.custom_patchcore_thresh:.4f}")

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

        # Cek jika ada komponen yang eksplisit cacat atau anomali pasangan
        for b in bolts:
            if b["is_ng"]:
                bolt_verdict = "NG"

        for p in pins:
            if p["is_ng"]:
                pin_verdict = "NG"

        for pair in pairs:
            if pair["pin"] is not None and pair["pin"]["is_ng"]:
                pin_verdict = "NG"

        # Visualisasi Baut & Pin (Seragam Warna Hijau, Status Evaluasi di Panel HUD Atas)
        box_color = (0, 255, 0)
        drawn_pins = set()
        for pair in pairs:
            b = pair["bolt"]
            p = pair["pin"]
            bx_1, by_1, bx_2, by_2 = b["box"]

            cv2.rectangle(display_frame, (bx_1, by_1), (bx_2, by_2), box_color, 2)
            cv2.putText(display_frame, f"{b['class_name']} {b['confidence']:.2f}",
                        (bx_1, max(15, by_1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, box_color, 2)

            if p is not None:
                drawn_pins.add(id(p))
                px1, py1, px2, py2 = p["box"]
                cv2.rectangle(display_frame, (px1, py1), (px2, py2), box_color, 2)
                cv2.putText(display_frame, f"{p['class_name']} {p['confidence']:.2f}",
                            (px1, max(15, py1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.44, box_color, 2)
                cv2.line(display_frame, b["centroid"], p["centroid"], (0, 255, 180), 1)

        # Gambar pin yang belum berpasangan (jika ada)
        for p in pins:
            if id(p) not in drawn_pins:
                px1, py1, px2, py2 = p["box"]
                cv2.rectangle(display_frame, (px1, py1), (px2, py2), box_color, 2)
                cv2.putText(display_frame, f"{p['class_name']} {p['confidence']:.2f}",
                            (px1, max(15, py1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.44, box_color, 2)

        # 3. Virtual Photocell / Optical Probe (Sensor Kehadiran Pelat Slat di Gerbang Masuk Poligon)
        # Menghitung koordinat adaptif sensor probe tepat di area masuk / leher poligon braket
        # Posisi ini berada tepat di zona pelat braket melintas dan kebal terhadap interferensi bolt ulir di sebelah kiri mesin
        min_poly_x = min(p[0] for p in self.polygon_pts)
        max_poly_x = max(p[0] for p in self.polygon_pts)
        min_poly_y = min(p[1] for p in self.polygon_pts)

        px1 = int((min_poly_x + 0.016) * w_frame)
        px2 = int((max_poly_x - 0.009) * w_frame)
        py = int((min_poly_y + 0.057) * h_frame)

        # Mengukur kecerahan refleksi logam pada garis sensor photocell (konversi ROI strip saja, hemat CPU)
        py1_b = max(0, py - 2)
        py2_b = min(h_frame, py + 2)
        px1_b = max(0, px1)
        px2_b = min(w_frame, px2)
        if py2_b > py1_b and px2_b > px1_b:
            beam_bgr = frame[py1_b:py2_b, px1_b:px2_b]
            beam_roi = cv2.cvtColor(beam_bgr, cv2.COLOR_BGR2GRAY)
            beam_val = float(np.mean(beam_roi))
        else:
            beam_val = 0.0

        # Ukur delta waktu antar-frame untuk mendeteksi frame jump / lag sesaat
        t_now = time.time()
        if self.last_frame_time is not None:
            self.inter_frame_dt = max(0.001, t_now - self.last_frame_time)
        self.last_frame_time = t_now
        is_frame_jump = (self.inter_frame_dt > 0.065)  # Terjadi lag/drop > 65ms (setara lompatan 2+ frame pada 30 FPS)

        # Logika Sensor Hysteresis Schmitt Trigger & Debounce Persistence Filter
        if not self.photocell_state:
            # Dari keadaan GAP menuju PLAT: butuh beam_val >= photocell_thresh_high
            # Debounce terpenuhi jika bertahan plat_debounce_frames berturut-turut,
            # ATAU jika terjadi frame jump/drop dan intensitas sangat kuat (>= thresh * 1.15)
            if beam_val >= self.photocell_thresh_high:
                self.consecutive_plat_frames += 1
                debounce_ok = (self.consecutive_plat_frames >= self.plat_debounce_frames) or \
                              (is_frame_jump and beam_val >= (self.photocell_thresh_high * 1.15))
                if debounce_ok:
                    current_f = self.stats["total_frames"]
                    # Proteksi interval: abaikan pemicuan kilap sesaat dari slat yang sama (misal refleksi bolt ulir)
                    if 0 <= (current_f - self.last_slat_trigger_frame) < self.min_slat_interval_frames:
                        pass
                    else:
                        self.photocell_state = True
                        self.consecutive_gap_frames = 0
                        self.last_slat_trigger_frame = current_f

                        # RISING EDGE TERKONFIRMASI: Pelat baru tiba di gerbang inspeksi
                        if not self.has_seen_gap:
                            # Video/stream baru mulai saat plat sudah di tengah jalan; lewati plat parsial ini
                            pass
                        else:
                            # Pastikan slat sebelumnya difinalisasi jika belum sempat lewat falling edge
                            if self.slat_cycle_count > 0 and self.slat_cycle_count != self.last_counted_slat_cycle:
                                self._finalize_slat_qc()

                            self.slat_cycle_count += 1
                            self.stats["total_slats"] = self.slat_cycle_count
                            self.slat_settling_countdown = self.slat_settling_delay
                            self.slat_audit_countdown = self.slat_audit_window
                            self.current_slat_scores = []
                            self.current_slat_samples = []
                            self.slat_evaluated = False
                            self.sweet_spot_sample = None
                            self.last_slat_verdict = "WAIT"
                            self.last_slat_score = 0.0
                            self.current_slat_has_bolt_ng = False
                            self.current_slat_has_pin_ng = False
                            self.current_slat_has_crack_ng = False
                            self.current_slat_bolt_seen = False
                            self.current_slat_pin_seen = False
            else:
                self.consecutive_plat_frames = 0
                if beam_val < self.photocell_thresh_low:
                    self.consecutive_gap_frames += 1
                    if self.consecutive_gap_frames >= self.gap_debounce_frames:
                        self.has_seen_gap = True
        else:
            # Dari keadaan PLAT menuju GAP: butuh beam_val < photocell_thresh_low minimal gap_debounce_frames berturut-turut
            # ATAU Watchdog timeout: jika plat tertahan >= max_slat_duration_frames, paksa auto-release ke GAP
            current_f = self.stats["total_frames"]
            slat_duration = current_f - self.last_slat_trigger_frame
            watchdog_triggered = (self.last_slat_trigger_frame > 0 and slat_duration >= self.max_slat_duration_frames)

            if beam_val < self.photocell_thresh_low or watchdog_triggered:
                self.consecutive_gap_frames += 1
                if self.consecutive_gap_frames >= self.gap_debounce_frames or watchdog_triggered:
                    self.photocell_state = False
                    self.consecutive_plat_frames = 0
                    self.has_seen_gap = True

                    # FALLING EDGE TERKONFIRMASI: Pelat sah meninggalkan gerbang menuju gap antar slat
                    self.slat_settling_countdown = 0
                    self.slat_audit_countdown = 0
                    if self.current_slat_scores and not self.slat_evaluated:
                        # Evaluasi skor: kebal terhadap frame drop (Anti False-OK)
                        if len(self.current_slat_scores) >= self.min_audit_samples_required:
                            rep_score = min(self.current_slat_scores)
                            self.last_slat_score = rep_score
                            self.last_slat_verdict = "NG" if rep_score >= self.custom_patchcore_thresh else "OK"
                        elif len(self.current_slat_scores) == 1:
                            # Kompensasi frame drop / lag: tetap evaluasi sampel tunggal agar NG tidak terlewat
                            rep_score = self.current_slat_scores[0]
                            self.last_slat_score = rep_score
                            self.last_slat_verdict = "NG" if rep_score >= self.custom_patchcore_thresh else "OK"
                        else:
                            self.last_slat_verdict = "WAIT"
                        self.slat_evaluated = True


                    self._finalize_slat_qc()
            else:
                self.consecutive_gap_frames = 0

        is_plate_present = self.photocell_state

        # Visualisasi Sensor Virtual Photocell pada Layar
        beam_color = (0, 255, 255) if is_plate_present else (100, 100, 100)
        cv2.line(display_frame, (px1, py), (px2, py), beam_color, 2)
        cv2.circle(display_frame, (px1, py), 4, beam_color, -1)
        cv2.circle(display_frame, (px2, py), 4, beam_color, -1)
        sensor_status_text = f"PHOTOCELL: {'ON [PLAT]' if is_plate_present else 'OFF [GAP]'} ({beam_val:.1f} | Th:{self.photocell_thresh_high:.0f}/{self.photocell_thresh_low:.0f})"
        font_scale = 0.58
        font_thick = 2
        (t_w, t_h), _ = cv2.getTextSize(sensor_status_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
        bg_x1 = max(5, px1 - 20)
        bg_y1 = max(5, py - t_h - 12)
        bg_x2 = min(w_frame - 5, bg_x1 + t_w + 10)
        bg_y2 = bg_y1 + t_h + 8
        cv2.rectangle(display_frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (20, 20, 20), -1)
        cv2.rectangle(display_frame, (bg_x1, bg_y1), (bg_x2, bg_y2), beam_color, 1)
        cv2.putText(display_frame, sensor_status_text, (bg_x1 + 5, bg_y2 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, beam_color, font_thick)

        # 4. Pemicuan & Audit Inspeksi Lekukan Plat (PADA POLIGON DIAM MASKED)
        inspected_bracket_info = None

        if is_plate_present and crop_poly is not None and crop_poly.size > 0:
            if self.slat_settling_countdown > 0:
                self.slat_settling_countdown -= 1
                lekukan_verdict = "WAIT"
                lek_color = (0, 215, 255)
                label_text = "BRACKET CRACK: MEMASUKI POLIGON"
            elif self.slat_audit_countdown > 0 or (len(self.current_slat_scores) < self.min_audit_samples_required and (current_f - self.last_slat_trigger_frame) <= (self.slat_settling_delay + self.slat_audit_window + 2)):
                if self.slat_audit_countdown > 0:
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

                # Simpan snapshot sweet-spot yang memiliki anotasi poligon & badge terpasang
                sample_ann_frame = display_frame.copy()
                sample_color = (0, 0, 255) if raw_status == "NG" else (0, 255, 0)
                cv2.polylines(sample_ann_frame, [pts_px], isClosed=True, color=sample_color, thickness=3)
                for p_pt in pts_px:
                    cv2.circle(sample_ann_frame, (int(p_pt[0]), int(p_pt[1])), 5, (0, 255, 255), -1)
                    cv2.circle(sample_ann_frame, (int(p_pt[0]), int(p_pt[1])), 7, (0, 0, 0), 1)
                cv2.rectangle(sample_ann_frame, (bx, max(0, by - 24)), (bx + 280, by), (20, 20, 20), -1)
                cv2.putText(sample_ann_frame, f"BRACKET CRACK: {raw_status} [{raw_score:.3f}]",
                            (bx + 5, max(16, by - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, sample_color, 2)

                self.current_slat_samples.append({
                    "score": raw_score,
                    "status": raw_status,
                    "raw_frame": frame.copy(),
                    "ann_frame": sample_ann_frame,
                    "bracket_info": inspected_bracket_info
                })

                self.last_bracket_info = inspected_bracket_info
                self.last_audit_frame = self.stats["total_frames"]

                # Real-time visual feedback: responsif dan akurat langsung di layar
                lekukan_verdict = raw_status
                lek_color = (0, 0, 255) if raw_status == "NG" else (0, 255, 0)
                label_text = f"BRACKET CRACK: {raw_status} [{raw_score:.3f}] (AUDIT)"
            else:
                # Slat masih melintas setelah window audit selesai -> Tampilkan hasil slat ini
                if self.current_slat_scores and not self.slat_evaluated:
                    rep_score = min(self.current_slat_scores)
                    self.last_slat_score = rep_score
                    self.last_slat_verdict = "NG" if rep_score >= self.custom_patchcore_thresh else "OK"
                    self.slat_evaluated = True

                lekukan_verdict = self.last_slat_verdict

                lek_color = (0, 0, 255) if lekukan_verdict == "NG" else (0, 255, 0)
                label_text = f"BRACKET CRACK: {lekukan_verdict} [{self.last_slat_score:.3f}]"
        else:
            # Di luar gerbang (celah transisi antar slat) -> Tahan hasil slat terakhir
            if self.current_slat_scores and not self.slat_evaluated:
                rep_score = min(self.current_slat_scores)
                self.last_slat_score = rep_score
                self.last_slat_verdict = "NG" if rep_score >= self.custom_patchcore_thresh else "OK"
                self.slat_evaluated = True

            recent_audit = (self.last_bracket_info is not None and 
                            (self.stats["total_frames"] - self.last_audit_frame) < 25)
            if recent_audit and self.last_slat_verdict != "WAIT":
                lekukan_verdict = self.last_slat_verdict
                lek_color = (0, 0, 255) if lekukan_verdict == "NG" else (0, 255, 0)
                label_text = f"BRACKET CRACK: {lekukan_verdict} [{self.last_slat_score:.3f}] (LAST)"
            else:
                lekukan_verdict = "WAIT"
                lek_color = (0, 215, 255)  # Cyan standby
                label_text = "BRACKET CRACK [STANDBY]"

        # Gambar Garis Poligon Diam di Layar
        cv2.polylines(display_frame, [pts_px], isClosed=True, color=lek_color, thickness=3)

        # Gambar Titik-titik Sudut Poligon (Interactive Handles)
        for i, p in enumerate(pts_px):
            cv2.circle(display_frame, (int(p[0]), int(p[1])), 5, (0, 255, 255), -1)
            cv2.circle(display_frame, (int(p[0]), int(p[1])), 7, (0, 0, 0), 1)

        # Badge Label di atas Poligon
        cv2.rectangle(display_frame, (bx, max(0, by - 24)), (bx + 280, by), (20, 20, 20), -1)
        cv2.putText(display_frame, label_text, (bx + 5, max(16, by - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, lek_color, 2)

        # Akumulasi temuan keberadaan dan defek komponen sepanjang transit plat slat aktif
        if is_plate_present:
            if any(not b.get("is_ng", False) for b in bolts):
                self.current_slat_bolt_seen = True
            if any(not p.get("is_ng", False) for p in pins):
                self.current_slat_pin_seen = True

            if bolt_verdict == "NG":
                self.current_slat_has_bolt_ng = True
            if pin_verdict == "NG":
                self.current_slat_has_pin_ng = True
            if lekukan_verdict == "NG":
                self.current_slat_has_crack_ng = True

        # Evaluasi status visual Baut & Pin untuk Tampilan HUD & Telemetri
        if is_plate_present:
            if self.current_slat_bolt_seen or any(not b.get("is_ng", False) for b in bolts):
                bolt_display = "OK"
            elif self.enable_missing_check and self.slat_audit_countdown == 0 and len(self.current_slat_scores) >= self.min_audit_samples_required:
                bolt_display = "NG"
            else:
                bolt_display = bolt_verdict if bolt_verdict != "EMPTY" else "WAIT"

            if self.current_slat_pin_seen or any(not p.get("is_ng", False) for p in pins):
                pin_display = "OK"
            elif self.enable_missing_check and self.slat_audit_countdown == 0 and len(self.current_slat_scores) >= self.min_audit_samples_required:
                pin_display = "NG"
            else:
                pin_display = pin_verdict if pin_verdict != "EMPTY" else "WAIT"
        else:
            if self.enable_missing_check and self.last_slat_bolt_status != "WAIT":
                bolt_display = self.last_slat_bolt_status
            else:
                bolt_display = bolt_verdict

            if self.enable_missing_check and self.last_slat_pin_status != "WAIT":
                pin_display = self.last_slat_pin_status
            else:
                pin_display = pin_verdict

        # 4. Tentukan Overall Verdict Sisi Ini
        # Prioritas visual live: NG > INVALID > OK > WAIT
        if bolt_display == "NG" or pin_display == "NG" or lekukan_verdict == "NG":
            overall_side = "NG"
        elif bolt_display == "WAIT" or pin_display == "WAIT" or lekukan_verdict == "WAIT":
            overall_side = "WAIT"
        elif (bolt_display in ["OK", "EMPTY"]) and (pin_display in ["OK", "EMPTY"]) and (lekukan_verdict in ["OK", "WAIT"]):
            overall_side = "OK"
        else:
            overall_side = "INVALID"

        # Single-Capture per Slat Entity Latch:
        # Emit ke database SQLite & Teams HANYA 1 KALI PER SIKLUS SLAT setelah evaluasi tuntas dan terkonfirmasi NG
        should_emit = False
        emit_raw = frame
        emit_ann = display_frame

        current_f = self.stats["total_frames"]
        transit_cooldown_ok = (current_f - self.last_emit_frame) >= self.min_slat_transit_frames or (current_f < self.last_emit_frame)

        # Evaluasi apakah slat ini berstatus NG
        is_bolt_ng = self.current_slat_has_bolt_ng or (bolt_display == "NG")
        is_pin_ng = self.current_slat_has_pin_ng or (pin_display == "NG")
        is_lek_ng = self.current_slat_has_crack_ng or (self.last_slat_verdict == "NG")
        is_ng_slat = (is_bolt_ng or is_pin_ng or is_lek_ng)

        # Latch Entity: Emit HANYA 1 kali per siklus slat fisik begitu status NG terkonfirmasi secara tuntas
        if (self.slat_evaluated and is_ng_slat and 
            self.slat_cycle_count > 0 and 
            self.slat_cycle_count != self.logged_slat_cycle and 
            transit_cooldown_ok and self.current_slat_samples and
            len(self.current_slat_scores) >= 1):
            
            self.logged_slat_cycle = self.slat_cycle_count
            self.last_emit_frame = current_f
            should_emit = True

            # Ambil sampel sweet-spot terbaik (frame di mana braket terpusat optimal pada poligon, bukan trailing-edge keluar)
            sweet_idx = len(self.current_slat_samples) // 2
            best_sample = self.current_slat_samples[sweet_idx]
            for s in self.current_slat_samples:
                if abs(s.get("score", 0.0) - self.last_slat_score) < 1e-4:
                    best_sample = s
                    break
            emit_raw = best_sample.get("raw_frame", frame)
            emit_ann = best_sample.get("ann_frame", display_frame)

        # Update Statistik Frame (telemetri live per-frame untuk analisis throughput)
        if overall_side in ["OK", "NG", "INVALID"]:
            if overall_side == "OK":
                self.stats["frame_ok_count"] = self.stats.get("frame_ok_count", 0) + 1
            elif overall_side == "NG":
                self.stats["frame_ng_count"] = self.stats.get("frame_ng_count", 0) + 1
            else:
                self.stats["invalid_count"] = self.stats.get("invalid_count", 0) + 1

        # 5. Format Payload Sesuai Standar PRD
        max_b_conf = max([b["confidence"] for b in bolts], default=0.0)
        max_p_conf = max([p["confidence"] for p in pins], default=0.0)

        payload = {
            "camera_id": self.camera_id,
            "side": self.side,
            "overall_result": overall_side,
            "should_emit": should_emit,
            "slat_number": self.slat_cycle_count,
            "emit_raw_frame": emit_raw,
            "emit_ann_frame": emit_ann,
            "components": [
                {"name": "BOLT", "result": bolt_display, "confidence": max_b_conf},
                {"name": "PIN", "result": pin_display, "confidence": max_p_conf},
                {"name": "LEKUKAN", "result": lekukan_verdict, "confidence": self.last_slat_score}
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
        if h >= hud_h and w > 0:
            sub = img[0:hud_h, 0:w]
            dark = np.full_like(sub, 25)
            cv2.addWeighted(dark, 0.85, sub, 0.15, 0, sub)

        res = payload["overall_result"]
        res_color = (0, 255, 0) if res == "OK" else ((0, 0, 255) if res == "NG" else (0, 200, 255))

        # Judul & Status Utama
        preset_badge = f"[{self.preset.upper()}]" if hasattr(self, "preset") else ""
        cv2.putText(img, f"CONVEYOR SIDE INSPECTION [{self.camera_id} - {self.side}] {preset_badge}",
                    (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
        cv2.putText(img, f"STATUS: {res}", (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, res_color, 2)

        # Rincian Komponen dengan Confidence
        c_x = int(w * 0.42)
        for c in payload["components"]:
            c_col = (0, 255, 0) if c["result"] == "OK" else ((0, 0, 255) if "NG" in str(c["result"]) else ((0, 215, 255) if c["result"] == "WAIT" else (180, 180, 180)))
            conf_val = c.get("confidence", 0.0)
            if c["name"] == "LEKUKAN":
                conf_str = f" [{conf_val:.3f}]" if conf_val > 0 else ""
            else:
                conf_str = f" [{conf_val:.2f}]" if conf_val > 0 else ""
            txt = f"{c['name']}: {c['result']}{conf_str}"
            cv2.putText(img, txt, (c_x, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.58, c_col, 2)
            c_x += int(w * 0.19)


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
        try:
            root.lift()
            root.focus_force()
        except Exception:
            pass

        default_dir = initial_dir
        if not default_dir or not os.path.exists(default_dir):
            candidate_dirs = [
                os.path.abspath(os.path.join(base_dir, "..", "Video_Testing")),
                os.path.abspath(os.path.join(base_dir, "..", "..", "Video_Testing")),
                os.path.abspath(os.path.join(base_dir, "..", "..", "..", "Video_Testing")),
                os.path.join(os.path.expanduser("~"), "Videos"),
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


def run_side_inspector_demo(video_path=None, side="RIGHT", camera_id="CAM-01", preset="auto", enable_db=False, enable_missing_check=None):
    """
    Fungsi demo interaktif untuk menguji video konveyor sisi samping.
    Mendukung:
    - Saklar Database SQLite & Notifikasi MS Teams via shortcut [D] atau flag --db
    - Pemilihan file video baru kapan saja via shortcut keyboard [O] / [F]
    - Toggle Preset Posisi Kamera (Opsi 1 / Opsi 2) via shortcut [M]
    - Penyesuaian titik sudut Poligon Diam & Master ROI menggunakan Mouse Drag
    """
    inspector = SideConveyorInspector(camera_id=camera_id, side=side, preset=preset, enable_db=enable_db, enable_missing_check=enable_missing_check)

    # Validasi path video, jika belum ada / salah, buka dialog pemilih file
    if not video_path or not os.path.exists(video_path):
        print("[*] Path video tidak ditemukan. Membuka dialog pemilih file video...")
        video_path = select_video_file(title="Pilih File Video untuk Inspeksi")
        if not video_path:
            print("[ERROR] Tidak ada video yang dipilih. Demo dihentikan.")
            return

    # Auto-resolusi preset dari nama video jika mode auto aktif
    if inspector.auto_preset_mode and video_path:
        det_p = inspector.resolve_preset_from_source(video_path)
        inspector.set_preset(det_p, verbose=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Gagal membuka video di: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[*] Membuka Video: {os.path.basename(video_path)} ({total_frames} frame, {fps:.1f} FPS)")

    win_name = f"Conveyor Side Inspector - [{camera_id} {side}] | [D]: Toggle DB"
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

    frame_dim = [1280, 720]

    def mouse_cb(event, x, y, flags, param):
        w_f, h_f = frame_dim[0], frame_dim[1]
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
    cv2.setMouseCallback(win_name, mouse_cb)

    is_paused = False
    delay_ms = max(1, int(1000 / fps))

    print("\n" + "=" * 75)
    print("   SISTEM INSPEKSI SISI SAMPING (HYBRID YOLO + STATIC POLYGON MASK)")
    print("=" * 75)
    print("  - [D]                  : TOGGLE DATABASE LOGGING & TEAMS NOTIFICATION (ON/OFF)")
    print("  - [O] / [F]            : PILIH / GANTI FILE VIDEO LAIN (File Explorer)")
    print("  - POLIGON LEKUKAN      : Poligon diam mengikuti lekukan miring plat")
    print("  - DRAG TITIK POLIGON   : Klik & geser titik sudut kuning untuk membentuk lekukan")
    print("  - DRAG DALAM POLIGON   : Klik & geser untuk memindahkan seluruh poligon")
    print("  - [SPACE]              : Pause / Play video")
    print("  - [N] / [P] / [A]      : Maju / Mundur 1 frame saat pause")
    print("  - [R]                  : Reset posisi poligon & Master ROI ke default")
    print("  - [Q] / [ESC]          : Keluar")
    print("=" * 75 + "\n")

    try:
        while True:
            t_loop_start = time.time()
            if not is_paused:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                cur_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                if cur_pos % 5 == 0 or cur_pos >= total_frames - 1:
                    cv2.setTrackbarPos("Timeline", win_name, cur_pos)
            else:
                cur_pos = cv2.getTrackbarPos("Timeline", win_name)
                cap.set(cv2.CAP_PROP_POS_FRAMES, cur_pos)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

            h_f, w_f = frame.shape[:2]
            frame_dim[0], frame_dim[1] = w_f, h_f

            display_frame, payload = inspector.process_frame(frame)

            # Rekam ke database SQLite, folder captures, dan kirim notifikasi Teams jika status NG dan DB aktif
            if payload.get("overall_result") == "NG":
                inspector.record_if_ng(
                    frame_raw=frame,
                    frame_ann=display_frame,
                    side_payload=payload
                )

            # Petunjuk shortcut di bagian bawah frame
            db_state = "ON" if inspector.is_db_enabled() else "OFF"
            help_bar = f"[SPACE]: Pause | [M]: Preset({inspector.preset.upper()}) | [D]: DB({db_state}) | [W]/[O]: Media | [R]: Reset | [Q]: Keluar"
            banner_color = (0, 255, 255) if inspector.is_db_enabled() else (220, 220, 220)
            cv2.putText(display_frame, help_bar, (15, h_f - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.44, banner_color, 1)

            cv2.imshow(win_name, display_frame)

            if not is_paused:
                elapsed_ms = (time.time() - t_loop_start) * 1000.0
                dynamic_delay = max(1, int(delay_ms - elapsed_ms))
                key = cv2.waitKey(dynamic_delay) & 0xFF
            else:
                key = cv2.waitKey(30) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord(' '):
                is_paused = not is_paused
            elif key in [ord('m'), ord('M')]:
                next_p = "opsi_2" if inspector.preset == "opsi_1" else "opsi_1"
                inspector.set_preset(next_p, verbose=True)
                init_th = int(inspector.bracket_detector.calibrated_threshold * 1000)
                cv2.setTrackbarPos("PatchCore Thresh (x1000)", win_name, init_th)
                inspector.reset_session()
                print(f"\n[✓] Preset kamera dialihkan ke: {inspector.preset.upper()}\n")
            elif key in [ord('d'), ord('D')]:
                st = inspector.toggle_db()
                status_str = "AKTIF (RECORDING & NOTIFIKASI TEAMS)" if st else "MATI (DRY-RUN BENCHMARK)"
                print(f"\n[*] [{inspector.camera_id} SIDE] SAKLAR DATABASE & TEAMS: {status_str}\n")
            elif key in [ord('w'), ord('W'), ord('o'), ord('O'), ord('f'), ord('F')]:
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

                        if inspector.auto_preset_mode:
                            det_p = inspector.resolve_preset_from_source(video_path)
                            inspector.set_preset(det_p, verbose=True)
                            init_th = int(inspector.bracket_detector.calibrated_threshold * 1000)
                            cv2.setTrackbarPos("PatchCore Thresh (x1000)", win_name, init_th)

                        inspector.reset_session()
                        print(f"\n[✓] Berhasil memuat video baru: {os.path.basename(video_path)} ({total_frames} frame, {fps:.1f} FPS)")
                    else:
                        print(f"[ERROR] Gagal membuka file video baru: {new_video}")
                else:
                    print("[*] Pemilihan file video dibatalkan.")
            elif key in [ord('n'), ord('N')]:
                cur = cap.get(cv2.CAP_PROP_POS_FRAMES)
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(total_frames - 1, cur + 1))
                cv2.setTrackbarPos("Timeline", win_name, int(cur + 1))
            elif key in [ord('p'), ord('P'), ord('a'), ord('A')]:
                cur = cap.get(cv2.CAP_PROP_POS_FRAMES)
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, cur - 2))
                cv2.setTrackbarPos("Timeline", win_name, int(max(0, cur - 2)))
            elif key in [ord('r'), ord('R')]:
                inspector.master_roi = list(inspector.default_master_roi)
                inspector.polygon_pts = [list(p) for p in inspector.default_polygon]
                inspector.save_config()
                print("[*] Konfigurasi ROI & Poligon di-reset ke default.")
    finally:
        cap.release()
        inspector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Side Conveyor Inspector Pipeline (Hybrid YOLO + PatchCore)")
    parser.add_argument("video", nargs="?", default=None, help="Path ke file video inspeksi (opsional)")
    parser.add_argument("--side", default="RIGHT", choices=["LEFT", "RIGHT"], help="Sisi kamera (LEFT / RIGHT)")
    parser.add_argument("--cam", default="CAM-01", help="ID Kamera (default: CAM-01)")
    parser.add_argument("--preset", default="auto", choices=["auto", "opsi_1", "opsi_2", "1", "2"], help="Preset posisi kamera (opsi_1 / opsi_2 / auto)")
    parser.add_argument("--db", action="store_true", help="Aktifkan pencatatan Database SQLite & Notifikasi MS Teams")
    parser.add_argument("--no-missing-check", action="store_true", help="Nonaktifkan deteksi baut & pin hilang (Default: Aktif)")
    args = parser.parse_args()

    video_input = args.video
    if not video_input:
        # 1. Cek dari .env jika ada
        env_side = os.environ.get("SIDE_CAMERA_SOURCE")
        if env_side:
            if env_side.isdigit() or env_side.startswith(("rtsp://", "http://", "https://", "nvarguscamerasrc")):
                video_input = env_side
            elif os.path.exists(env_side):
                video_input = os.path.abspath(env_side)

        # 2. Cek kandidat video bawaan
        if not video_input:
            default_candidates = [
                os.path.abspath(os.path.join(base_dir, "..", "Video_Testing", "Deform Sedikit.mp4")),
                os.path.abspath(os.path.join(base_dir, "..", "..", "Video_Testing", "Deform Sedikit.mp4")),
                os.path.abspath(os.path.join(base_dir, "..", "..", "..", "Video_Testing", "Deform Sedikit.mp4")),
            ]
            video_input = next((v for v in default_candidates if os.path.exists(v)), None)

        # 3. Fallback dialog file explorer
        if not video_input:
            print("[*] Video default tidak ditemukan, membuka dialog pemilih file...")
            video_input = select_video_file(title="Pilih File Video untuk Inspeksi Samping")

    if video_input and (os.path.exists(video_input) or str(video_input).isdigit() or str(video_input).startswith(("rtsp://", "http://"))):
        run_side_inspector_demo(video_input, side=args.side, camera_id=args.cam, preset=args.preset, enable_db=args.db, enable_missing_check=not args.no_missing_check)
    else:
        print("[!] Tidak ada video yang dipilih. Program dihentikan.")

