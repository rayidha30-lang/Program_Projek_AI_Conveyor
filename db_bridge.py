"""
==============================================================================
INSPECTION DATABASE BRIDGE (SQLITE NON-BLOCKING ASYNC LOGGER)
Target: AI Slat Rib Conveyor (NVIDIA Jetson Orin Nano & Local PC)
==============================================================================
Deskripsi:
  Modul jembatan penghubung antara Vision Engine (combined_top_inspector.py)
  dengan database SQLite (AI_Slat_Rib_Conveyor.db) dan folder captures.

Fitur Unggulan:
  1. Non-Blocking Worker: Menjalankan penyimpanan gambar (JPEG imwrite) dan 
     insert SQLite di background thread antrean (queue.Queue) agar FPS kamera
     tidak drop sama sekali (Zero Latency Impact).
  2. Dry-Run / Toggle Nyala-Mati: Default status adalah OFF.
     Bebas tuning parameter tanpa mengotori database. Tekan [D] untuk aktivasi.
  3. Latch Cooldown / Debounce: Mencegah spam multi-frame untuk 1 bilah slat
     yang sama (cooldown default 2.0 detik per jenis cacat).
  4. Penyimpanan Gambar Terstruktur: Folder otomatis DD-MM (tanggal-bulan)
     dengan subfolder raw/ dan annotated/.
==============================================================================
"""

import os
import sys
import time
import queue
import sqlite3
import threading
from datetime import datetime

from collections import deque
import cv2

# Impor modul Notifikasi MS Teams via Node-RED
PROGRAM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Program"))
if PROGRAM_DIR not in sys.path:
    sys.path.insert(0, PROGRAM_DIR)

try:
    from teams_notifier import NodeRedTeamsNotifier
except ImportError:
    NodeRedTeamsNotifier = None


class InspectionDBBridge:
    def __init__(
        self,
        db_path=None,
        captures_dir=None,
        enabled=False,
        cooldown_sec=2.0,
        camera_id="CAM-01"  # CAM-01 (Overhead Plat/Rib di FE) atau CAM-03 di Jetson
    ):
        # 1. Tentukan lokasi database file
        if db_path is None:
            env_db = os.environ.get("SQLITE_DB_PATH")
            if env_db:
                db_path = os.path.abspath(env_db)
            else:
                base_dir = os.path.dirname(os.path.abspath(__file__))
                candidate = os.path.join(base_dir, "AI_Slat_Rib_Conveyor.db")
                if not os.path.exists(candidate):
                    # Cari di parent direktori jika diimpor dari subfolder lain
                    search_roots = [
                        os.path.abspath(os.path.join(base_dir, "AI_Slat_Rib_Conveyor.db")),
                        os.path.abspath(os.path.join(base_dir, "..", "Database", "AI_Slat_Rib_Conveyor.db")),
                        os.path.abspath(os.path.join(base_dir, "..", "..", "Database", "AI_Slat_Rib_Conveyor.db")),
                    ]
                    for r in search_roots:
                        if os.path.exists(r):
                            candidate = r
                            break
                db_path = candidate
        self.db_path = db_path

        # 2. Tentukan lokasi direktori captures
        if captures_dir is None:
            env_captures = os.environ.get("CAPTURES_DIR")
            if env_captures:
                captures_dir = os.path.abspath(env_captures)
            else:
                captures_dir = os.path.join(os.path.dirname(self.db_path), "captures")
        self.captures_dir = captures_dir

        # 3. Status saklar on/off (cek parameter atau environment variable)
        env_enable_db = os.environ.get("ENABLE_DB_LOGGING", "").lower() in ("true", "1", "yes")
        self.enabled = bool(enabled or env_enable_db)
        self.cooldown_sec = float(cooldown_sec)
        self.camera_id = camera_id

        # 4. Anti-spam tracker (Timestamp deteksi terakhir per komponen & Entity Cache)
        self._last_log_time = {
            "PLAT": 0.0,
            "RIB": 0.0,
            "BOLT": 0.0,
            "PIN": 0.0,
        }
        self.logged_entity_keys = deque(maxlen=60)

        # 5. Non-blocking queue & background worker thread
        self._queue = queue.Queue(maxsize=100)
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="DBBridgeWorker"
        )
        self._worker_thread.start()

        # 6. Notifier MS Teams via Node-RED
        self.teams_notifier = None
        if NodeRedTeamsNotifier is not None:
            try:
                teams_endpoint = os.environ.get("NODE_RED_WEBHOOK_URL") or os.environ.get("NODE_RED_ENDPOINT") or "http://127.0.0.1:1880/ai-inspection"
                self.teams_notifier = NodeRedTeamsNotifier(endpoint_url=teams_endpoint, enabled=True)
            except Exception as e_tn:
                print(f"[TEAMS INIT WARN] {e_tn}")

        status_str = "ACTIVE (RECORDING)" if self.enabled else "OFF (DRY-RUN TUNING)"
        print(f"[*] InspectionDBBridge diinisialisasi | Status: {status_str} | DB: {os.path.basename(self.db_path)}")

        # Pastikan skema tabel SQLite terbuat otomatis jika database masih baru
        self._ensure_schema()

        # Pastikan SQLite WAL mode dan busy timeout diaktifkan untuk konkurensi multi-proses
        try:
            init_conn = sqlite3.connect(self.db_path, timeout=10.0)
            init_conn.execute("PRAGMA journal_mode = WAL;")
            init_conn.execute("PRAGMA busy_timeout = 30000;")
            init_conn.execute("PRAGMA synchronous = NORMAL;")
            init_conn.close()
        except Exception as e_wal:
            print(f"[DB WAL WARN] Gagal set WAL mode: {e_wal}")

    def _ensure_schema(self):
        """Memastikan tabel SQLite dan skema awal tersedia (Self-healing)."""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
            conn = sqlite3.connect(self.db_path, timeout=15.0)
            with conn:
                conn.executescript("""
                PRAGMA foreign_keys = ON;
                CREATE TABLE IF NOT EXISTS Inspections (
                    InspectionID INTEGER PRIMARY KEY AUTOINCREMENT,
                    TraceabilityCode TEXT NOT NULL UNIQUE,
                    SlatNumber INTEGER,
                    InspectionTime DATETIME NOT NULL DEFAULT (datetime('now', 'localtime')),
                    OverallResult TEXT NOT NULL CHECK (OverallResult IN ('OK', 'NG', 'INVALID')),
                    OverallSeverity TEXT NOT NULL DEFAULT 'NONE' CHECK (OverallSeverity IN ('NONE', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
                    StatusLifecycle TEXT NOT NULL DEFAULT 'UNACKNOWLEDGED' CHECK (StatusLifecycle IN ('NORMAL', 'UNACKNOWLEDGED', 'ACKNOWLEDGED', 'RESOLVED')),
                    PICName TEXT,
                    MaintenanceNote TEXT,
                    ResolvedAt DATETIME,
                    CreatedAt DATETIME NOT NULL DEFAULT (datetime('now', 'localtime'))
                );
                CREATE TABLE IF NOT EXISTS Inspection_Detections (
                    DetectionID INTEGER PRIMARY KEY AUTOINCREMENT,
                    InspectionID INTEGER NOT NULL,
                    CameraID TEXT NOT NULL CHECK (CameraID IN ('CAM-01', 'CAM-02', 'CAM-03')),
                    TargetArea TEXT NOT NULL,
                    ComponentType TEXT NOT NULL CHECK (ComponentType IN ('PLAT', 'RIB', 'BOLT', 'PIN', 'LEKUKAN')),
                    Side TEXT NOT NULL DEFAULT 'NONE' CHECK (Side IN ('RH', 'LH', 'NONE')),
                    Result TEXT NOT NULL CHECK (Result IN ('OK', 'NG', 'INVALID')),
                    StatusDetail TEXT NOT NULL CHECK (StatusDetail IN ('NORMAL', 'BENDING', 'MISSING', 'CRACK')),
                    Confidence REAL,
                    PVFlatness REAL,
                    MaxDeviation REAL,
                    RawImagePath TEXT,
                    AnnotatedPath TEXT,
                    CreatedAt DATETIME NOT NULL DEFAULT (datetime('now', 'localtime')),
                    FOREIGN KEY (InspectionID) REFERENCES Inspections (InspectionID) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS Maintenance_Logs (
                    LogID INTEGER PRIMARY KEY AUTOINCREMENT,
                    InspectionID INTEGER NOT NULL,
                    Action TEXT NOT NULL CHECK (Action IN ('ACKNOWLEDGED', 'RESOLVED')),
                    PICName TEXT NOT NULL,
                    Notes TEXT,
                    LoggedAt DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (InspectionID) REFERENCES Inspections (InspectionID) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS Master_Components (
                    ComponentType TEXT PRIMARY KEY,
                    ComponentName TEXT NOT NULL,
                    DefaultCamera TEXT NOT NULL,
                    DefaultSide TEXT NOT NULL
                );
                INSERT OR IGNORE INTO Master_Components (ComponentType, ComponentName, DefaultCamera, DefaultSide) VALUES
                ('PLAT', 'Plat / Slat Conveyor', 'CAM-01', 'NONE'),
                ('RIB', 'Rib / Sirip Conveyor', 'CAM-01', 'NONE'),
                ('BOLT', 'Bolt Penahan Sisi', 'CAM-02', 'RH'),
                ('PIN', 'Pin Sambungan Rantai', 'CAM-02', 'RH'),
                ('LEKUKAN', 'Lekukan Ujung Sisi', 'CAM-02', 'RH');
                CREATE INDEX IF NOT EXISTS idx_inspections_time ON Inspections (InspectionTime DESC);
                CREATE INDEX IF NOT EXISTS idx_inspections_result ON Inspections (OverallResult);
                CREATE INDEX IF NOT EXISTS idx_detections_lookup ON Inspection_Detections (ComponentType, Side, Result);
                """)
            conn.close()
        except Exception as e_schema:
            print(f"[DB SCHEMA WARN] Gagal memastikan skema SQLite: {e_schema}")

    def _get_connection(self):
        """Membuka koneksi SQLite dengan konfigurasi WAL dan busy timeout tinggi (30 detik)."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        try:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA busy_timeout = 30000;")
            conn.execute("PRAGMA synchronous = NORMAL;")
        except Exception:
            pass
        return conn

    def reset_entity_cache(self):
        """Mereset riwayat deduplikasi entitas (misal saat video di-loop atau counter di-reset)."""
        self.logged_entity_keys.clear()
        self._last_log_time = {"PLAT": 0.0, "RIB": 0.0, "BOLT": 0.0, "PIN": 0.0}

    def toggle(self):
        """Membalik status saklar pencatatan (ON <-> OFF)."""
        self.enabled = not self.enabled
        status_str = "ACTIVE (RECORDING)" if self.enabled else "OFF (DRY-RUN TUNING)"
        print(f"\n[DB SWITCH] Status Database Logging beralih ke: {status_str}\n")
        return self.enabled

    def set_enabled(self, val: bool):
        """Mengatur status saklar secara eksplisit."""
        self.enabled = bool(val)
        return self.enabled

    def is_enabled(self) -> bool:
        """Mengecek apakah database logging sedang aktif."""
        return self.enabled

    def _get_today_capture_dirs(self):
        """Menyiapkan direktori captures/DD-MM/raw dan captures/DD-MM/annotated."""
        today_folder = datetime.now().strftime("%d-%m")
        raw_dir = os.path.join(self.captures_dir, today_folder, "raw")
        ann_dir = os.path.join(self.captures_dir, today_folder, "annotated")
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(ann_dir, exist_ok=True)
        return raw_dir, ann_dir

    def emit_top_inspection(
        self,
        frame_raw,
        frame_ann,
        slat_result,
        rib_result,
        slat_number=None,
        rib_number=None,
        force=False
    ):
        """
        Menerima hasil inspeksi tampak atas (Plat & Rib) dari combined_top_inspector.
        Fungsi ini 100% NON-BLOCKING: frame dikirim ke antrean worker thread.
        HANYA memproses jika status saklar ON (atau force=True) dan terdapat NG!
        Dilengkapi Deduplikasi Entity ID: 1 objek fisik NG hanya dicatat tepat 1 kali.
        """
        if not self.enabled and not force:
            return False

        # Verifikasi apakah ada NG
        is_plat_ng = bool(slat_result.get("is_ng", False)) if slat_result else False
        is_rib_ng = bool(rib_result.get("is_ng", False)) if rib_result else False

        if not is_plat_ng and not is_rib_ng:
            # Sesuai instruksi pengguna: Hanya mencatat data NG saja!
            return False

        # Entity-based deduplication
        entity_keys = []
        if is_plat_ng and slat_number is not None:
            entity_keys.append(f"PLAT_{slat_number}")
        if is_rib_ng:
            r_num = rib_number if rib_number is not None else slat_number
            if r_num is not None:
                entity_keys.append(f"RIB_{r_num}")

        # Jika seluruh komponen NG pada frame ini sudah pernah dicatat dalam cache, abaikan!
        if entity_keys and not force:
            if all(k in self.logged_entity_keys for k in entity_keys):
                return False

        # Cek Cooldown Debounce (mencegah spamming multi-frame jika tidak ada ID objek)
        now = time.time()
        need_process = False
        if is_plat_ng and (now - self._last_log_time["PLAT"] > self.cooldown_sec or entity_keys):
            self._last_log_time["PLAT"] = now
            need_process = True
        if is_rib_ng and (now - self._last_log_time["RIB"] > self.cooldown_sec or entity_keys):
            self._last_log_time["RIB"] = now
            need_process = True

        if not need_process and not force:
            return False

        for k in entity_keys:
            if k not in self.logged_entity_keys:
                self.logged_entity_keys.append(k)

        # Clone image secara aman sebelum dikirim ke queue agar tidak tertimpa frame baru
        raw_copy = frame_raw.copy() if frame_raw is not None else None
        ann_copy = frame_ann.copy() if frame_ann is not None else None

        payload = {
            "time": datetime.now(),
            "raw_frame": raw_copy,
            "ann_frame": ann_copy,
            "slat_res": slat_result,
            "rib_res": rib_result,
            "slat_number": slat_number,
            "rib_number": rib_number,
        }

        try:
            self._queue.put_nowait(payload)
            return True
        except queue.Full:
            print("[DB WARN] Antrean database penuh! Frame NG dilewati.")
            return False

    def emit_side_inspection(
        self,
        frame_raw,
        frame_ann,
        side_payload,
        force=False
    ):
        """
        Menerima hasil inspeksi samping (Baut, Pin & Lekukan Bracket).
        100% NON-BLOCKING: frame dikirim ke antrean worker thread untuk disimpan ke SQLite & Captures.
        """
        if not self.enabled and not force:
            return False

        if not side_payload or side_payload.get("overall_result") != "NG":
            return False

        components = side_payload.get("components", [])
        ng_components = [c for c in components if str(c.get("result", "")).upper() == "NG"]
        if not ng_components and not force:
            return False

        # Entity-based deduplication per slat cycle untuk mencegah duplicate capture
        slat_num = side_payload.get("slat_number", None)
        side_str = side_payload.get("side", "RIGHT")
        cam_str = side_payload.get("camera_id", self.camera_id)
        if slat_num is not None and not force:
            entity_key = f"{cam_str}_{side_str}_SLAT_{slat_num}"
            if entity_key in self.logged_entity_keys:
                return False
            self.logged_entity_keys.append(entity_key)

        # Debounce per detik agar tidak spam setiap frame
        now = time.time()
        if (now - self._last_log_time.get("BOLT", 0.0)) < self.cooldown_sec and not force:
            return False
        self._last_log_time["BOLT"] = now

        raw_copy = frame_raw.copy() if frame_raw is not None else None
        ann_copy = frame_ann.copy() if frame_ann is not None else None

        item = {
            "is_side": True,
            "time": datetime.now(),
            "raw_frame": raw_copy,
            "ann_frame": ann_copy,
            "side_payload": side_payload
        }
        try:
            self._queue.put_nowait(item)
            return True
        except queue.Full:
            print("[DB WARN] Antrean database penuh! Frame Side NG dilewati.")
            return False

    def _worker_loop(self):
        """Worker thread latar belakang untuk kompresi file dan transaksi database."""
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                self._save_to_database_and_disk(item)
            except Exception as e:
                print(f"[DB ERROR] Gagal menyimpan data NG: {e}")
            finally:
                self._queue.task_done()

    def _save_to_database_and_disk(self, item):
        dt = item["time"]
        raw_img = item["raw_frame"]
        ann_img = item["ann_frame"]
        raw_dir, ann_dir = self._get_today_capture_dirs()
        ts_str = dt.strftime("%Y%m%d_%H%M%S_%f")[:19]
        dt_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        today_date_str = dt.strftime("%Y%m%d")

        # ------------------------------------------------------------------
        # KASUS A: DATA INSPEKSI SAMPING (SIDE BOLT, PIN & BRACKET)
        # ------------------------------------------------------------------
        if item.get("is_side", False):
            side_pl = item.get("side_payload", {})
            raw_side = str(side_pl.get("side", "RIGHT")).upper().strip()
            if raw_side in ["RIGHT", "RH", "KANAN"]:
                db_side = "RH"
                default_cam = "CAM-02"
            elif raw_side in ["LEFT", "LH", "KIRI"]:
                db_side = "LH"
                default_cam = "CAM-03"
            else:
                db_side = "NONE"
                default_cam = "CAM-02"

            raw_cam = str(side_pl.get("camera_id", self.camera_id)).upper().strip()
            # Standardisasi: jika camera_id masih CAM-01 atau tidak valid, sesuaikan ke default side cam (CAM-02 / CAM-03)
            if raw_cam not in ["CAM-02", "CAM-03"]:
                cam_id = default_cam
            else:
                cam_id = raw_cam

            components = side_pl.get("components", [])
            bracket_details = side_pl.get("bracket_details", {})

            tag = f"{cam_id}_{db_side}_SIDE"
            raw_path = None
            ann_path = None
            if raw_img is not None:
                raw_fname = f"{tag}_{ts_str}_RAW.jpg"
                raw_path = os.path.join(raw_dir, raw_fname)
                cv2.imwrite(raw_path, raw_img, [cv2.IMWRITE_JPEG_QUALITY, 88])

            if ann_img is not None:
                ann_fname = f"{tag}_{ts_str}_ANN.jpg"
                ann_path = os.path.join(ann_dir, ann_fname)
                cv2.imwrite(ann_path, ann_img, [cv2.IMWRITE_JPEG_QUALITY, 88])
            else:
                ann_path = raw_path

            # Buka koneksi database dengan proteksi try/finally & WAL
            conn = self._get_connection()
            trace_code = None
            try:
                with conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT COUNT(1) FROM Inspections WHERE date(InspectionTime) = date('now', 'localtime')"
                    )
                    daily_count = cursor.fetchone()[0] + 1
                    trace_code = f"ABN-{today_date_str}-{daily_count:05d}"
                    while True:
                        cursor.execute("SELECT 1 FROM Inspections WHERE TraceabilityCode = ?", (trace_code,))
                        if cursor.fetchone():
                            daily_count += 1
                            trace_code = f"ABN-{today_date_str}-{daily_count:05d}"
                        else:
                            break

                    cursor.execute(
                        """
                        INSERT INTO Inspections
                        (TraceabilityCode, SlatNumber, InspectionTime, OverallResult, OverallSeverity, StatusLifecycle, CreatedAt)
                        VALUES (?, NULL, ?, 'NG', 'HIGH', 'UNACKNOWLEDGED', ?)
                        """,
                        (trace_code, dt_str, dt_str)
                    )
                    inspection_id = cursor.lastrowid

                    for c in components:
                        c_raw_name = str(c.get("name", "")).upper().strip()
                        c_res = str(c.get("result", "")).upper().strip()
                        if c_res == "NG":
                            # Sesuai CHECK constraint di init_db.sql:
                            # ComponentType IN ('PLAT', 'RIB', 'BOLT', 'PIN', 'LEKUKAN')
                            # StatusDetail IN ('NORMAL', 'BENDING', 'MISSING', 'CRACK')
                            if "BOLT" in c_raw_name:
                                db_comp = "BOLT"
                                db_stat = "MISSING"
                            elif "PIN" in c_raw_name:
                                db_comp = "PIN"
                                db_stat = "MISSING"
                            elif "LEKUKAN" in c_raw_name or "BRACKET" in c_raw_name:
                                db_comp = "LEKUKAN"
                                db_stat = "CRACK"
                            else:
                                db_comp = "LEKUKAN"
                                db_stat = "CRACK"

                            cursor.execute(
                                """
                                INSERT INTO Inspection_Detections
                                (InspectionID, CameraID, TargetArea, ComponentType, Side, Result, StatusDetail, Confidence, PVFlatness, MaxDeviation, RawImagePath, AnnotatedPath, CreatedAt)
                                VALUES (?, ?, 'CHAIN_SIDE', ?, ?, 'NG', ?, 95.0, 0.0, 0.0, ?, ?, ?)
                                """,
                                (inspection_id, cam_id, db_comp, db_side, db_stat, raw_path, ann_path, dt_str)
                            )
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

            print(f"\n[DB SAVED SIDE] Berhasil mencatat Side NG ke SQLite | ID: {trace_code} | {cam_id} ({db_side})")
            for c in components:
                if str(c.get("result", "")).upper() == "NG":
                    print(f"  -> {c.get('name')}: NG")

            # Memicu notifikasi MS Teams via Node-RED
            if hasattr(self, "teams_notifier") and self.teams_notifier and self.teams_notifier.is_enabled():
                try:
                    self.teams_notifier.send_side_inspection_ng(
                        camera_id=cam_id,
                        side=db_side,
                        components=components,
                        bracket_details=bracket_details,
                        inspection_id=trace_code,
                        evidence_url=f"http://localhost:5173/?trace={trace_code}",
                        image_url=f"http://localhost:5000/api/image?path={ann_path}" if ann_path else None,
                        force=True
                    )
                except Exception as e_teams:
                    print(f"[TEAMS NOTICE WARN] Gagal mengirim side ke Teams: {e_teams}")
            return

        # ------------------------------------------------------------------
        # KASUS B: DATA INSPEKSI TAMPAK ATAS (TOP SLAT & RIB)
        # ------------------------------------------------------------------
        slat_res = item["slat_res"] or {}
        rib_res = item["rib_res"] or {}
        slat_no = item.get("slat_number")

        raw_dir, ann_dir = self._get_today_capture_dirs()

        # Simpan file gambar
        ts_str = dt.strftime("%Y%m%d_%H%M%S_%f")[:19]
        slat_tag = f"SLAT{slat_no}" if slat_no else "SLAT_TOP"

        raw_path = None
        ann_path = None

        if raw_img is not None:
            raw_fname = f"{self.camera_id}_{slat_tag}_{ts_str}_RAW.jpg"
            raw_path = os.path.join(raw_dir, raw_fname)
            cv2.imwrite(raw_path, raw_img, [cv2.IMWRITE_JPEG_QUALITY, 88])

        if ann_img is not None:
            ann_fname = f"{self.camera_id}_{slat_tag}_{ts_str}_ANN.jpg"
            ann_path = os.path.join(ann_dir, ann_fname)
            cv2.imwrite(ann_path, ann_img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        else:
            ann_path = raw_path

        # Tentukan Severity
        # Plat/Rib Bending: MEDIUM (deviasi tinggi) atau LOW
        overall_severity = "MEDIUM"
        is_plat_ng = bool(slat_res.get("is_ng", False))
        is_rib_ng = bool(rib_res.get("is_ng", False))

        max_dev_val = 0.0
        if is_rib_ng:
            rib_dev = float(rib_res.get("max_dev", rib_res.get("dev", 0.0)) or 0.0)
            max_dev_val = max(max_dev_val, rib_dev)
        if is_plat_ng:
            slat_pv = float(slat_res.get("pv_flatness", slat_res.get("pv", 0.0)) or 0.0)
            max_dev_val = max(max_dev_val, slat_pv)

        if max_dev_val > 25.0:
            overall_severity = "HIGH"

        raw_top_cam = str(self.camera_id).upper().strip()
        top_cam_id = raw_top_cam if raw_top_cam in ["CAM-01", "CAM-02", "CAM-03"] else "CAM-03"

        # Buka koneksi database dengan proteksi try/finally & WAL
        conn = self._get_connection()
        trace_code = None
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT COUNT(1) FROM Inspections WHERE date(InspectionTime) = date('now', 'localtime')"
                )
                daily_count = cursor.fetchone()[0] + 1
                trace_code = f"ABN-{today_date_str}-{daily_count:05d}"
                while True:
                    cursor.execute("SELECT 1 FROM Inspections WHERE TraceabilityCode = ?", (trace_code,))
                    if cursor.fetchone():
                        daily_count += 1
                        trace_code = f"ABN-{today_date_str}-{daily_count:05d}"
                    else:
                        break

                dt_str = dt.strftime("%Y-%m-%d %H:%M:%S")

                # 1. INSERT Tabel Master: Inspections
                cursor.execute(
                    """
                    INSERT INTO Inspections
                    (TraceabilityCode, SlatNumber, InspectionTime, OverallResult, OverallSeverity, StatusLifecycle, CreatedAt)
                    VALUES (?, ?, ?, 'NG', ?, 'UNACKNOWLEDGED', ?)
                    """,
                    (trace_code, slat_no, dt_str, overall_severity, dt_str)
                )
                inspection_id = cursor.lastrowid

                # 2. INSERT Tabel Detail: Inspection_Detections
                # A. Komponen RIB (jika NG)
                if is_rib_ng:
                    rib_conf = float(rib_res.get("confidence", 95.0) or 95.0)
                    rib_dev = float(rib_res.get("max_dev", rib_res.get("dev", max_dev_val)) or max_dev_val)
                    rib_pv = float(rib_res.get("pv_flatness", 0.0) or 0.0)
                    cursor.execute(
                        """
                        INSERT INTO Inspection_Detections
                        (InspectionID, CameraID, TargetArea, ComponentType, Side, Result, StatusDetail, Confidence, PVFlatness, MaxDeviation, RawImagePath, AnnotatedPath, CreatedAt)
                        VALUES (?, ?, 'PLAT_RIB', 'RIB', 'NONE', 'NG', 'BENDING', ?, ?, ?, ?, ?, ?)
                        """,
                        (inspection_id, top_cam_id, rib_conf, rib_pv, rib_dev, raw_path, ann_path, dt_str)
                    )

                # B. Komponen PLAT (jika NG)
                if is_plat_ng:
                    slat_conf = float(slat_res.get("confidence", 92.0) or 92.0)
                    slat_pv = float(slat_res.get("pv_flatness", slat_res.get("pv", max_dev_val)) or max_dev_val)
                    cursor.execute(
                        """
                        INSERT INTO Inspection_Detections
                        (InspectionID, CameraID, TargetArea, ComponentType, Side, Result, StatusDetail, Confidence, PVFlatness, MaxDeviation, RawImagePath, AnnotatedPath, CreatedAt)
                        VALUES (?, ?, 'PLAT_RIB', 'PLAT', 'NONE', 'NG', 'BENDING', ?, ?, ?, ?, ?, ?)
                        """,
                        (inspection_id, top_cam_id, slat_conf, slat_pv, slat_pv, raw_path, ann_path, dt_str)
                    )
        finally:
            try:
                conn.close()
            except Exception:
                pass

        scale_k = 0.515
        print(f"\n[DB SAVED] Berhasil mencatat NG ke SQLite | ID: {trace_code} | Severity: {overall_severity}")
        if is_rib_ng:
            print(f"  -> RIB BENDING (Dev: {max_dev_val * scale_k:.1f}mm [{max_dev_val:.1f}px])")
        if is_plat_ng:
            slat_pv_val = float(slat_res.get("pv_flatness", slat_res.get("pv", max_dev_val)) or max_dev_val)
            print(f"  -> PLAT BENDING (PV: {slat_pv_val * scale_k:.1f}mm [{slat_pv_val:.1f}px])")

        # 3. Memicu notifikasi MS Teams via Node-RED
        if hasattr(self, "teams_notifier") and self.teams_notifier and self.teams_notifier.is_enabled():
            try:
                evidence_url = f"http://localhost:5173/?trace={trace_code}"
                image_url = f"http://localhost:5000/api/image?path={ann_path}" if ann_path else None
                self.teams_notifier.send_top_inspection_ng(
                    slat_res=slat_res,
                    rib_res=rib_res,
                    slat_number=slat_no,
                    rib_number=item.get("rib_number"),
                    inspection_id=trace_code,
                    evidence_url=evidence_url,
                    image_url=image_url,
                    force=True
                )
            except Exception as e_teams:
                print(f"[TEAMS NOTICE WARN] Gagal mengirim ke Teams: {e_teams}")

    def close(self):
        """Menutup background worker thread secara bersih."""
        self._stop_event.set()
        if hasattr(self, "teams_notifier") and self.teams_notifier:
            try:
                self.teams_notifier.close()
            except Exception:
                pass
        if hasattr(self, "_worker_thread") and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
        print("[*] InspectionDBBridge ditutup.")
