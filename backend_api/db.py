import sqlite3
import os
import sys
from datetime import datetime, timedelta

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Path default ke file database SQLite
DB_PATH = r"c:\File Rayidha\Internship\Project\5. Project AI Slat Rib Conveyor\Database\AI_Slat_Rib_Conveyor.db"


def resolve_db_path():
    """
    Mencari lokasi file database SQLite dengan prioritas fleksibel untuk Windows & Linux Jetson:
    1. Environment variable CONVEYOR_DB_PATH
    2. Path folder ../../Database/AI_Slat_Rib_Conveyor.db (struktur project lokal & Jetson)
    3. Path folder ../Database/AI_Slat_Rib_Conveyor.db (jika diletakkan di dalam Program)
    4. Path lokal folder saat ini AI_Slat_Rib_Conveyor.db
    5. Path default Windows DB_PATH
    """
    env_path = os.environ.get("CONVEYOR_DB_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    candidates = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Database", "AI_Slat_Rib_Conveyor.db")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Database", "AI_Slat_Rib_Conveyor.db")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "AI_Slat_Rib_Conveyor.db")),
        DB_PATH
    ]

    for p in candidates:
        if os.path.exists(p):
            return p

    # Fallback default aman jika belum ada (memastikan direktori dibuat)
    default_target = candidates[0]
    os.makedirs(os.path.dirname(default_target), exist_ok=True)
    return default_target


def get_connection():
    """
    Membuat koneksi ke database SQLite AI_Slat_Rib_Conveyor.db.
    """
    target = resolve_db_path()
    return sqlite3.connect(target, timeout=5.0)


def map_plat_rib_status(is_ng, status_str=""):
    """
    Memetakan status Plat / Rib:
    - Normal -> 'NORMAL'
    - Cacat -> 'BENDING'
    """
    if not is_ng:
        return "NORMAL"
    return "BENDING"


def determine_severity_and_result(detections):
    """
    Menghitung OverallResult dan OverallSeverity:
    - NORMAL / OK_PRESENT -> NONE
    - PLAT / RIB BENDING -> MEDIUM
    - BOLT MISSING -> HIGH
    - PIN MISSING -> CRITICAL (karena resiko rantai putus)
    """
    severity_rank = {
        "NONE": 1,
        "LOW": 2,
        "MEDIUM": 3,
        "HIGH": 4,
        "CRITICAL": 5
    }

    max_severity = "NONE"
    is_any_ng = False

    for d in detections:
        res = d.get("Result", "OK")
        comp = d.get("ComponentType", "")
        stat = d.get("StatusDetail", "NORMAL")

        if res == "NG":
            is_any_ng = True

            comp_sev = "MEDIUM"
            if comp in ("PLAT", "RIB"):
                comp_sev = "MEDIUM"
            elif comp == "BOLT" and stat == "MISSING":
                comp_sev = "HIGH"
            elif comp == "PIN" and stat == "MISSING":
                comp_sev = "CRITICAL"

            if severity_rank[comp_sev] > severity_rank[max_severity]:
                max_severity = comp_sev

    overall_result = "NG" if is_any_ng else "OK"
    return overall_result, max_severity


def generate_traceability_code(cursor, is_ng=False):
    """
    Menghasilkan TraceabilityCode yang unik dan urut per hari,
    contoh: ABN-20261009-00001 (untuk NG) atau INSP-20261009-00001 (untuk OK).
    """
    today_str = datetime.now().strftime("%Y%m%d")
    prefix = "ABN" if is_ng else "INSP"

    # Hitung jumlah inspeksi hari ini di SQLite
    cursor.execute(
        "SELECT COUNT(1) FROM Inspections WHERE date(InspectionTime) = date('now', 'localtime')"
    )
    daily_count = cursor.fetchone()[0] + 1
    return f"{prefix}-{today_str}-{daily_count:05d}"


def save_inspection_to_db(image_path, l_sir=None, l_sla=None, detections=None, slat_number=None, shift=None, annotated_path=None):
    """
    Menyimpan hasil inspeksi ke tabel Inspections (Master) dan
    Inspection_Detections (Detail).

    Mendukung pemanggilan langsung dari program CAM-01 (l_sir & l_sla)
    maupun gabungan seluruh kamera CAM-01, CAM-02, CAM-03 via list detections.

    Parameters
    ----------
    image_path : str
        Path file gambar utama (misal dari CAM-01).
    l_sir : dict, optional
        Hasil analisis SIRIP/RIB (CAM-01).
    l_sla : dict, optional
        Hasil analisis PLAT/SLAT (CAM-01).
    detections : list of dict, optional
        Daftar deteksi manual dari CAM-01, CAM-02, dan CAM-03.
    slat_number : int, optional
        Nomor urutan slat (1-150).
    shift : int, optional
        Shift kerja (1, 2, atau 3).
    annotated_path : str, optional
        Path gambar hasil anotasi/overlay AI.
    """
    conn = None

    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Susun daftar deteksi
        all_detections = []

        # 1. Parsing jika dikirim dari format l_sir & l_sla (CAM-01)
        if l_sir is not None:
            is_ng_rib = bool(l_sir.get("is_ng", False))
            status_rib = map_plat_rib_status(is_ng_rib, l_sir.get("status_str", ""))
            all_detections.append({
                "CameraID": "CAM-01",
                "TargetArea": "PLAT_RIB",
                "ComponentType": "RIB",
                "Side": "NONE",
                "Result": "NG" if is_ng_rib else "OK",
                "StatusDetail": status_rib,
                "Confidence": None,
                "PVFlatness": float(l_sir.get("pv_flatness", 0.0)) if l_sir.get("pv_flatness") is not None else None,
                "MaxDeviation": float(l_sir.get("max_abs", 0.0)) if l_sir.get("max_abs") is not None else None,
                "RawImagePath": image_path,
                "AnnotatedPath": annotated_path
            })

        if l_sla is not None:
            is_ng_sla = bool(l_sla.get("is_ng", False))
            status_sla = map_plat_rib_status(is_ng_sla, l_sla.get("status_str", ""))
            all_detections.append({
                "CameraID": "CAM-01",
                "TargetArea": "PLAT_RIB",
                "ComponentType": "PLAT",
                "Side": "NONE",
                "Result": "NG" if is_ng_sla else "OK",
                "StatusDetail": status_sla,
                "Confidence": None,
                "PVFlatness": float(l_sla.get("pv_flatness", 0.0)) if l_sla.get("pv_flatness") is not None else None,
                "MaxDeviation": float(l_sla.get("max_abs", 0.0)) if l_sla.get("max_abs") is not None else None,
                "RawImagePath": image_path,
                "AnnotatedPath": annotated_path
            })

        # 2. Tambahkan jika ada deteksi tambahan (misal CAM-02 & CAM-03)
        if detections and isinstance(detections, list):
            all_detections.extend(detections)

        # 3. Hitung Overall Result & Overall Severity
        overall_result, overall_severity = determine_severity_and_result(all_detections)
        status_lifecycle = "UNACKNOWLEDGED" if overall_result == "NG" else "NORMAL"
        shift_val = shift if shift is not None else determine_shift()

        # 4. Generate Traceability Code unik
        traceability_code = generate_traceability_code(cursor, is_ng=(overall_result == "NG"))

        # 5. INSERT ke tabel Master: Inspections
        cursor.execute(
            """
            INSERT INTO Inspections
            (
                TraceabilityCode,
                SlatNumber,
                InspectionTime,
                OverallResult,
                OverallSeverity,
                StatusLifecycle
            )
            VALUES
            (
                ?,
                ?,
                CURRENT_TIMESTAMP,
                ?,
                ?,
                ?
            )
            """,
            (
                traceability_code,
                slat_number,
                overall_result,
                overall_severity,
                status_lifecycle
            )
        )

        inspection_id = cursor.lastrowid

        # 6. INSERT ke tabel Detail: Inspection_Detections
        insert_query = """
            INSERT INTO Inspection_Detections
            (
                InspectionID,
                CameraID,
                TargetArea,
                ComponentType,
                Side,
                Result,
                StatusDetail,
                Confidence,
                PVFlatness,
                MaxDeviation,
                RawImagePath,
                AnnotatedPath
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        for d in all_detections:
            cursor.execute(
                insert_query,
                inspection_id,
                d.get("CameraID", "CAM-01"),
                d.get("TargetArea", "PLAT_RIB"),
                d.get("ComponentType"),
                d.get("Side", "NONE"),
                d.get("Result", "OK"),
                d.get("StatusDetail"),
                d.get("Confidence"),
                d.get("PVFlatness"),
                d.get("MaxDeviation"),
                d.get("RawImagePath"),
                d.get("AnnotatedPath")
            )

        # Commit transaksi
        conn.commit()

        print("\n" + "=" * 65)
        print("DATABASE - CONVEYOR INSPECTION")
        print("=" * 65)
        print(f"Inspection ID     : {inspection_id}")
        print(f"Traceability Code : {traceability_code}")
        print(f"Shift             : Shift {shift_val}")
        print(f"Overall Result    : {overall_result}")
        print(f"Overall Severity  : {overall_severity}")
        print(f"Status Lifecycle  : {status_lifecycle}")
        print(f"Total Detections  : {len(all_detections)} items")
        print("=" * 65)

        return inspection_id

    except Exception as e:
        if conn is not None:
            conn.rollback()
        print(f"\n❌ Gagal menyimpan data ke SQL Server: {e}")
        return None

    finally:
        if conn is not None:
            conn.close()



def purge_expired_inspections(retention_days=7, custom_cutoff_str=None):
    """
    Menghapus data rekaman inspeksi dan file fisik citra yang telah berumur lebih dari `retention_days` hari.
    Mencakup:
    1. Pencarian path gambar (RawImagePath, AnnotatedPath) dari detail deteksi yang kedaluwarsa.
    2. Penghapusan file fisik citra (.jpg) dan pembersihan folder kosong di disk (Database/captures/<DD-MM>).
    3. Penghapusan baris relasional dari database SQLite (Inspections dan cascade detail).
    
    Returns:
        dict: Statistik eksekusi {deleted_inspections, deleted_detections, deleted_files, deleted_dirs, cutoff}
    """
    stats = {
        "deleted_inspections": 0,
        "deleted_detections": 0,
        "deleted_files": 0,
        "deleted_dirs": 0,
        "cutoff": None,
        "errors": []
    }

    if custom_cutoff_str:
        cutoff_dt = datetime.strptime(custom_cutoff_str, "%Y-%m-%d %H:%M:%S")
    else:
        # Menyimpan tepat `retention_days` hari terakhir (termasuk hari ini sebagai hari ke-1).
        # Contoh jika hari ini tgl 20 dan retention 7 hari: data tgl 14 s/d 20 disimpan (7 hari),
        # sehingga cutoff adalah tgl 14 pukul 00:00:00 (semua rekaman tgl 13 dan sebelumnya dihapus).
        today_midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff_dt = today_midnight - timedelta(days=max(1, retention_days - 1))

    cutoff_str = cutoff_dt.strftime("%Y-%m-%d %H:%M:%S")
    stats["cutoff"] = cutoff_str

    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON;")

        # 1. Ambil semua path file gambar yang akan dihapus
        query_files = """
            SELECT d.RawImagePath, d.AnnotatedPath, i.InspectionID
            FROM Inspection_Detections d
            JOIN Inspections i ON d.InspectionID = i.InspectionID
            WHERE i.InspectionTime < ?
        """
        cursor.execute(query_files, (cutoff_str,))
        rows = cursor.fetchall()

        files_to_remove = set()
        dirs_to_check = set()
        for raw_p, ann_p, _ in rows:
            if raw_p and os.path.exists(raw_p):
                files_to_remove.add(os.path.abspath(raw_p))
                dirs_to_check.add(os.path.dirname(os.path.abspath(raw_p)))
            if ann_p and os.path.exists(ann_p):
                files_to_remove.add(os.path.abspath(ann_p))
                dirs_to_check.add(os.path.dirname(os.path.abspath(ann_p)))

        # 2. Hapus file fisik citra dari hard disk
        for fpath in files_to_remove:
            try:
                os.remove(fpath)
                stats["deleted_files"] += 1
            except Exception as e:
                stats["errors"].append(f"Gagal menghapus file {fpath}: {e}")

        # Bersihkan direktori kosong (misal Database/captures/09-10/raw dan /09-10)
        for dir_path in sorted(dirs_to_check, key=lambda p: len(p), reverse=True):
            try:
                if os.path.exists(dir_path) and not os.listdir(dir_path):
                    os.rmdir(dir_path)
                    stats["deleted_dirs"] += 1
                    # Cek juga parent directory-nya
                    parent_dir = os.path.dirname(dir_path)
                    if os.path.exists(parent_dir) and not os.listdir(parent_dir):
                        os.rmdir(parent_dir)
                        stats["deleted_dirs"] += 1
            except Exception:
                pass

        # 3. Hitung jumlah record yang akan dihapus
        cursor.execute("SELECT COUNT(1) FROM Inspections WHERE InspectionTime < ?", (cutoff_str,))
        stats["deleted_inspections"] = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(1) FROM Inspection_Detections d
            JOIN Inspections i ON d.InspectionID = i.InspectionID
            WHERE i.InspectionTime < ?
        """, (cutoff_str,))
        stats["deleted_detections"] = cursor.fetchone()[0]

        # 4. Hapus dari database (Maintenance_Logs, Inspection_Detections, Inspections)
        cursor.execute("""
            DELETE FROM Maintenance_Logs WHERE InspectionID IN (
                SELECT InspectionID FROM Inspections WHERE InspectionTime < ?
            )
        """, (cutoff_str,))
        cursor.execute("""
            DELETE FROM Inspection_Detections WHERE InspectionID IN (
                SELECT InspectionID FROM Inspections WHERE InspectionTime < ?
            )
        """, (cutoff_str,))
        cursor.execute("DELETE FROM Inspections WHERE InspectionTime < ?", (cutoff_str,))

        conn.commit()
        print(f"[RETENTION PURGE] Berhasil membersihkan data sebelum {cutoff_str}:")
        print(f"   - Inspections dihapus: {stats['deleted_inspections']}")
        print(f"   - Detections dihapus: {stats['deleted_detections']}")
        print(f"   - File gambar dihapus: {stats['deleted_files']}")
        print(f"   - Folder kosong dibersihkan: {stats['deleted_dirs']}")

    except Exception as e:
        if conn is not None:
            conn.rollback()
        stats["errors"].append(str(e))
        print(f"[RETENTION PURGE ERROR] Gagal melakukan purge: {e}")
    finally:
        if conn is not None:
            conn.close()

    return stats


# ==========================================================
# TEST FUNCTION
# ==========================================================
if __name__ == "__main__":
    print("--- Test 1: CAM-01 saja (Plat & Rib) ---")
    test_img = r"C:\File Rayidha\Internship\Project\5. Project AI Slat Rib Conveyor\Dataset\IMG_TEST.jpg"

    dummy_sirip = {
        "is_ng": False,
        "pv_flatness": 5.4,
        "max_abs": 2.8,
        "status_str": "OK (RATA)"
    }

    dummy_slat = {
        "is_ng": True,
        "pv_flatness": 21.3,
        "max_abs": 16.5,
        "status_str": "NG (NAIK +16.5px)"
    }

    # Simpan hasil CAM-01
    save_inspection_to_db(test_img, dummy_sirip, dummy_slat, slat_number=42)

    print("\n--- Test 2: Multi-Camera Lengkap (CAM-01, CAM-02, CAM-03) ---")
    multi_cam_detections = [
        # CAM-01
        {
            "CameraID": "CAM-01",
            "TargetArea": "PLAT_RIB",
            "ComponentType": "PLAT",
            "Side": "NONE",
            "Result": "OK",
            "StatusDetail": "NORMAL",
            "Confidence": None,
            "PVFlatness": 8.2,
            "MaxDeviation": 4.1,
            "RawImagePath": test_img,
            "AnnotatedPath": None
        },
        {
            "CameraID": "CAM-01",
            "TargetArea": "PLAT_RIB",
            "ComponentType": "RIB",
            "Side": "NONE",
            "Result": "OK",
            "StatusDetail": "NORMAL",
            "Confidence": None,
            "PVFlatness": 6.1,
            "MaxDeviation": 3.0,
            "RawImagePath": test_img,
            "AnnotatedPath": None
        },
        # CAM-02 (Kanan)
        {
            "CameraID": "CAM-02",
            "TargetArea": "RH_SIDE",
            "ComponentType": "BOLT",
            "Side": "RH",
            "Result": "OK",
            "StatusDetail": "OK_PRESENT",
            "Confidence": 98.5,
            "PVFlatness": None,
            "MaxDeviation": None,
            "RawImagePath": test_img,
            "AnnotatedPath": None
        },
        {
            "CameraID": "CAM-02",
            "TargetArea": "RH_SIDE",
            "ComponentType": "PIN",
            "Side": "RH",
            "Result": "OK",
            "StatusDetail": "OK_PRESENT",
            "Confidence": 96.2,
            "PVFlatness": None,
            "MaxDeviation": None,
            "RawImagePath": test_img,
            "AnnotatedPath": None
        },
        # CAM-03 (Kiri - Contoh Pin Hilang)
        {
            "CameraID": "CAM-03",
            "TargetArea": "LH_SIDE",
            "ComponentType": "BOLT",
            "Side": "LH",
            "Result": "OK",
            "StatusDetail": "OK_PRESENT",
            "Confidence": 97.8,
            "PVFlatness": None,
            "MaxDeviation": None,
            "RawImagePath": test_img,
            "AnnotatedPath": None
        },
        {
            "CameraID": "CAM-03",
            "TargetArea": "LH_SIDE",
            "ComponentType": "PIN",
            "Side": "LH",
            "Result": "NG",
            "StatusDetail": "MISSING",
            "Confidence": 92.4,
            "PVFlatness": None,
            "MaxDeviation": None,
            "RawImagePath": test_img,
            "AnnotatedPath": None
        }
    ]

    save_inspection_to_db(
        image_path=test_img,
        detections=multi_cam_detections,
        slat_number=11
    )