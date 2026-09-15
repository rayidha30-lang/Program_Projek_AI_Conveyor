import pyodbc
import os
from datetime import datetime


def get_connection():
    """
    Membuat koneksi ke database SQL Server ConveyorInspection.
    """
    return pyodbc.connect(
        "DRIVER={ODBC Driver 18 for SQL Server};"
        "SERVER=localhost\\MSSQLSERVER01;"
        "DATABASE=ConveyorInspection;"
        "Trusted_Connection=yes;"
        "TrustServerCertificate=yes;"
    )


def determine_shift():
    """
    Menentukan shift kerja saat ini:
    - Shift 1: 07:00 - 15:00
    - Shift 2: 15:00 - 23:00
    - Shift 3: 23:00 - 07:00
    """
    hour = datetime.now().hour
    if 7 <= hour < 15:
        return 1
    elif 15 <= hour < 23:
        return 2
    else:
        return 3


def map_plat_rib_status(is_ng, status_str=""):
    """
    Memetakan status Plat / Rib ke nilai baku CHECK constraint database:
    - 'NORMAL'
    - 'BEND_UP'
    - 'BEND_DOWN'
    """
    if not is_ng:
        return "NORMAL"
    
    st_upper = str(status_str).upper()
    if "TURUN" in st_upper or "DOWN" in st_upper or "-" in st_upper:
        return "BEND_DOWN"
    else:
        # Default jika bengkok ke atas atau bergelombang
        return "BEND_UP"


def determine_severity_and_result(detections):
    """
    Menghitung OverallResult dan OverallSeverity berdasarkan seluruh deteksi:
    - NORMAL / OK_PRESENT -> NONE
    - PLAT / RIB BEND_UP / BEND_DOWN -> MEDIUM
    - BOLT MISSING -> HIGH
    - PIN MISSING -> CRITICAL (karena resiko rantai putus)
    """
    severity_rank = {
        "NONE": 1,
        "MEDIUM": 2,
        "HIGH": 3,
        "CRITICAL": 4
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
    contoh: ABN-20260915-00001 (untuk NG) atau INSP-20260915-00001 (untuk OK).
    """
    today_str = datetime.now().strftime("%Y%m%d")
    prefix = "ABN" if is_ng else "INSP"

    # Hitung jumlah inspeksi hari ini untuk nomor urut
    cursor.execute(
        """
        SELECT COUNT(1) 
        FROM Inspections 
        WHERE CAST(InspectionTime AS DATE) = CAST(GETDATE() AS DATE)
        """
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
                Shift,
                OverallResult,
                OverallSeverity,
                StatusLifecycle
            )
            OUTPUT INSERTED.InspectionID
            VALUES
            (
                ?,
                ?,
                GETDATE(),
                ?,
                ?,
                ?,
                ?
            )
            """,
            traceability_code,
            slat_number,
            shift_val,
            overall_result,
            overall_severity,
            status_lifecycle
        )

        inspection_id = cursor.fetchone()[0]

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