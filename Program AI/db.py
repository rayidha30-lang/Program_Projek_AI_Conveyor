import pyodbc
import os


def get_connection():
    """
    Membuat koneksi ke SQL Server.
    """
    return pyodbc.connect(
        "DRIVER={ODBC Driver 18 for SQL Server};"
        "SERVER=localhost\\MSSQLSERVER01;"
        "DATABASE=ConveyorInspection;"
        "Trusted_Connection=yes;"
        "TrustServerCertificate=yes;"
    )


def save_inspection_to_db(image_path, l_sir, l_sla):
    """
    Menyimpan hasil inspeksi Plat dan Rib ke SQL Server.

    Parameters
    ----------
    image_path : str
        Path gambar yang sedang diinspeksi.

    l_sir : dict
        Hasil inspeksi SIRIP/RIB dari program vision.

    l_sla : dict
        Hasil inspeksi PLAT/SLAT dari program vision.
    """

    conn = None

    try:
        # ==========================================================
        # 1. Membuat koneksi
        # ==========================================================
        conn = get_connection()
        cursor = conn.cursor()

        # ==========================================================
        # 2. Menentukan nama file
        # ==========================================================
        image_name = os.path.basename(image_path)

        # ==========================================================
        # 3. Menentukan Overall Result
        # ==========================================================
        overall_result = (
            "NG"
            if l_sir["is_ng"] or l_sla["is_ng"]
            else "OK"
        )

        # ==========================================================
        # 4. INSERT ke tabel Inspection
        # ==========================================================
        cursor.execute(
            """
            INSERT INTO Inspection
            (
                ImageName,
                InspectionTime,
                OverallResult
            )
            OUTPUT INSERTED.InspectionID
            VALUES
            (
                ?,
                GETDATE(),
                ?
            )
            """,
            image_name,
            overall_result
        )

        # Mengambil InspectionID yang baru dibuat
        inspection_id = cursor.fetchone()[0]

        # ==========================================================
        # 5. INSERT hasil RIB
        # ==========================================================
        rib_result = "NG" if l_sir["is_ng"] else "OK"

        cursor.execute(
            """
            INSERT INTO Detection
            (
                InspectionID,
                ObjectType,
                Side,
                Result,
                Confidence,
                PVFlatness,
                MaxDeviation,
                StatusDetail
            )
            VALUES
            (
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?
            )
            """,
            inspection_id,
            "RIB",
            None,
            rib_result,
            None,
            l_sir["pv_flatness"],
            l_sir["max_abs"],
            l_sir["status_str"]
        )

        # ==========================================================
        # 6. INSERT hasil PLAT
        # ==========================================================
        plat_result = "NG" if l_sla["is_ng"] else "OK"

        cursor.execute(
            """
            INSERT INTO Detection
            (
                InspectionID,
                ObjectType,
                Side,
                Result,
                Confidence,
                PVFlatness,
                MaxDeviation,
                StatusDetail
            )
            VALUES
            (
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?
            )
            """,
            inspection_id,
            "PLAT",
            None,
            plat_result,
            None,
            l_sla["pv_flatness"],
            l_sla["max_abs"],
            l_sla["status_str"]
        )

        # ==========================================================
        # 7. Commit transaksi
        # ==========================================================
        conn.commit()

        print("\n" + "=" * 60)
        print("DATABASE")
        print("=" * 60)
        print("Data inspeksi berhasil disimpan!")
        print(f"Inspection ID : {inspection_id}")
        print(f"Image         : {image_name}")
        print(f"Overall       : {overall_result}")
        print(f"RIB           : {rib_result}")
        print(f"PLAT          : {plat_result}")
        print("=" * 60)

        return inspection_id

    except Exception as e:

        # Jika terjadi error, batalkan transaksi
        if conn is not None:
            conn.rollback()

        print("\nGagal menyimpan data ke SQL Server.")
        print(f"Error: {e}")

        return None

    finally:

        if conn is not None:
            conn.close()

# ==========================================================
# TEST FUNCTION
# ==========================================================

if __name__ == "__main__":

    # Data dummy dengan struktur yang sama
    # seperti hasil program Plat & Rib

    test_image_path = r"C:\File Rayidha\Internship\Project\5. Project AI Slat Rib Conveyor\Dataset\IMG_TEST.jpg"

    test_l_sir = {
        "is_ng": False,
        "pv_flatness": 4.2,
        "max_abs": 2.1,
        "status_str": "OK"
    }

    test_l_sla = {
        "is_ng": True,
        "pv_flatness": 8.7,
        "max_abs": 5.3,
        "status_str": "BEND RIGHT"
    }

    # Simpan ke database
    inspection_id = save_inspection_to_db(
        test_image_path,
        test_l_sir,
        test_l_sla
    )

    print(f"\nTest selesai. Inspection ID = {inspection_id}")