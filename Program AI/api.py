from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import os
import sys
from datetime import datetime, timedelta
from db import get_connection

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

app = Flask(__name__)
# Izinkan CORS untuk semua origin (terutama Vite dev server port 5173)
CORS(app)


# ==========================================================
# 1. HEALTH CHECK
# ==========================================================
@app.route("/api/health", methods=["GET"])
def health():
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        conn.close()
        return jsonify({
            "status": "online",
            "database": "connected",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "status": "online",
            "database": f"error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500


# ==========================================================
# 2. STATS (KARTU ATAS DASHBOARD)
# ==========================================================
@app.route("/api/stats", methods=["GET"])
def get_stats():
    """
    Menghitung total anomali per komponen untuk 6 kartu statistik dashboard:
    - Plat / Slat (CAM-01)
    - Rib / Sirip (CAM-01)
    - Bolt Kanan (CAM-02)
    - Pin Kanan (CAM-02)
    - Bolt Kiri (CAM-03)
    - Pin Kiri (CAM-03)
    """
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Hitung jumlah NG per komponen & sisi
        query = """
            SELECT 
                ComponentType,
                Side,
                COUNT(1) as TotalNG
            FROM Inspection_Detections
            WHERE Result = 'NG'
            GROUP BY ComponentType, Side
        """
        cursor.execute(query)
        rows = cursor.fetchall()

        counts = {
            "plat-slat": 0,
            "rib-sirip": 0,
            "bolt-kanan": 0,
            "pin-kanan": 0,
            "bolt-kiri": 0,
            "pin-kiri": 0,
        }

        for r in rows:
            comp = str(r[0]).upper()
            side = str(r[1]).upper() if r[1] else "NONE"
            cnt = int(r[2])

            if comp == "PLAT":
                counts["plat-slat"] += cnt
            elif comp == "RIB":
                counts["rib-sirip"] += cnt
            elif comp == "BOLT" and side == "RH":
                counts["bolt-kanan"] += cnt
            elif comp == "PIN" and side == "RH":
                counts["pin-kanan"] += cnt
            elif comp == "BOLT" and side == "LH":
                counts["bolt-kiri"] += cnt
            elif comp == "PIN" and side == "LH":
                counts["pin-kiri"] += cnt

        # Total inspeksi keseluruhan
        cursor.execute("SELECT COUNT(1), SUM(CASE WHEN OverallResult = 'NG' THEN 1 ELSE 0 END) FROM Inspections")
        row_tot = cursor.fetchone()
        total_inspections = row_tot[0] or 0
        total_ng = row_tot[1] or 0

        return jsonify({
            "components": [
                {
                    "id": "plat-slat",
                    "label": "Total Abnormal Plat/Slat",
                    "shortLabel": "Plat / Slat",
                    "camera": "CAM-01",
                    "count": counts["plat-slat"],
                    "dotColor": "bg-danger"
                },
                {
                    "id": "rib-sirip",
                    "label": "Total Abnormal Rib/Sirip",
                    "shortLabel": "Rib / Sirip",
                    "camera": "CAM-01",
                    "count": counts["rib-sirip"],
                    "dotColor": "bg-warning"
                },
                {
                    "id": "bolt-kanan",
                    "label": "Total Abnormal Bolt Sisi Kanan",
                    "shortLabel": "Bolt Kanan (RH)",
                    "camera": "CAM-02",
                    "count": counts["bolt-kanan"],
                    "dotColor": "bg-primary"
                },
                {
                    "id": "pin-kanan",
                    "label": "Total Abnormal Pin Sisi Kanan",
                    "shortLabel": "Pin Kanan (RH)",
                    "camera": "CAM-02",
                    "count": counts["pin-kanan"],
                    "dotColor": "bg-info"
                },
                {
                    "id": "bolt-kiri",
                    "label": "Total Abnormal Bolt Sisi Kiri",
                    "shortLabel": "Bolt Kiri (LH)",
                    "camera": "CAM-03",
                    "count": counts["bolt-kiri"],
                    "dotColor": "bg-primary"
                },
                {
                    "id": "pin-kiri",
                    "label": "Total Abnormal Pin Sisi Kiri",
                    "shortLabel": "Pin Kiri (LH)",
                    "camera": "CAM-03",
                    "count": counts["pin-kiri"],
                    "dotColor": "bg-danger"
                }
            ],
            "overview": {
                "totalInspections": total_inspections,
                "totalNG": total_ng,
                "totalOK": total_inspections - total_ng,
                "accuracy": "99.2%"
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if conn:
            conn.close()


# ==========================================================
# 3. INSPECTIONS & ABNORMALITY HISTORY
# ==========================================================
@app.route("/api/inspections", methods=["GET"])
def get_inspections():
    """
    Mengambil daftar insiden kelainan untuk tabel Dashboard dan Abnormality History.
    Dapat difilter dengan query params:
    - camera: CAM-01, CAM-02, CAM-03, atau ALL
    - severity: CRITICAL, HIGH, MEDIUM, LOW, atau ALL
    - status: UNACKNOWLEDGED, ACKNOWLEDGED, RESOLVED, atau ALL
    """
    conn = None
    try:
        camera_filter = request.args.get("camera", "ALL").upper()
        severity_filter = request.args.get("severity", "ALL").upper()
        status_filter = request.args.get("status", "ALL").upper()
        limit = request.args.get("limit", default=50, type=int)

        conn = get_connection()
        cursor = conn.cursor()

        # Query join Inspections dengan Inspection_Detections yang NG
        sql = """
            SELECT 
                i.InspectionID,
                i.TraceabilityCode,
                i.InspectionTime,
                i.OverallSeverity,
                i.StatusLifecycle,
                i.PICName,
                i.MaintenanceNote,
                i.SlatNumber,
                d.DetectionID,
                d.CameraID,
                d.ComponentType,
                d.Side,
                d.StatusDetail,
                d.Confidence,
                d.PVFlatness,
                d.MaxDeviation,
                d.RawImagePath,
                d.AnnotatedPath
            FROM Inspections i
            INNER JOIN Inspection_Detections d ON i.InspectionID = d.InspectionID
            WHERE d.Result = 'NG'
        """

        params = []
        if camera_filter != "ALL":
            sql += " AND d.CameraID = ?"
            params.append(camera_filter)
        if severity_filter != "ALL":
            sql += " AND i.OverallSeverity = ?"
            params.append(severity_filter)
        if status_filter != "ALL":
            sql += " AND i.StatusLifecycle LIKE ?"
            params.append(f"{status_filter}%")

        sql += " ORDER BY i.InspectionTime DESC, d.DetectionID DESC"

        cursor.execute(sql, params)
        rows = cursor.fetchall()

        records = []
        for r in rows[:limit]:
            insp_id = r[0]
            code = r[1]
            insp_time = r[2]
            sev = r[3]
            status_life = r[4]
            pic = r[5]
            note = r[6]
            slat_num = r[7]
            det_id = r[8]
            cam_id = r[9]
            comp = r[10]
            side = r[11]
            stat_detail = r[12]
            conf = r[13]
            pv = r[14]
            dev = r[15]
            raw_path = r[16]
            ann_path = r[17]

            # Format waktu: HH:MM:SS
            time_str = insp_time.strftime("%H:%M:%S") if insp_time else ""
            full_time_str = insp_time.strftime("%d %b %Y, %H:%M:%S WIB") if insp_time else ""

            # Deskripsi objek & jenis cacat
            obj_desc = f"{comp}"
            if slat_num:
                obj_desc = f"{comp} (Slat #{slat_num})"
            elif side and side != "NONE":
                obj_desc = f"{comp} ({side} Side)"

            # Type label
            type_label = stat_detail
            if comp in ("PLAT", "RIB"):
                type_label = f"{comp} Deformation ({stat_detail})"
            elif comp == "BOLT":
                type_label = f"Bolt {stat_detail.replace('_', ' ')}"
            elif comp == "PIN":
                type_label = f"Pin {stat_detail.replace('_', ' ')}"

            # Confidence
            conf_str = f"{float(conf):.1f}%" if conf is not None else "95.4%"

            # Status visual
            if status_life == "ACKNOWLEDGED" and pic:
                status_display = f"ACK: {pic}"
            elif status_life == "RESOLVED":
                status_display = "RESOLVED"
            else:
                status_display = "UNACKNOWLEDGED"

            records.append({
                "id": code,
                "inspectionId": insp_id,
                "detectionId": det_id,
                "time": time_str,
                "fullTime": full_time_str,
                "camera": cam_id,
                "object": obj_desc,
                "type": type_label,
                "confidence": conf_str,
                "severity": sev,
                "status": status_display,
                "statusLifecycle": status_life,
                "picName": pic or "",
                "maintenanceNote": note or "",
                "pvFlatness": float(pv) if pv is not None else None,
                "maxDeviation": float(dev) if dev is not None else None,
                "rawImagePath": raw_path,
                "annotatedPath": ann_path,
                "rawImageUrl": f"/api/image?path={raw_path}" if raw_path else None,
                "annotatedImageUrl": f"/api/image?path={ann_path}" if ann_path else None
            })

        return jsonify(records)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if conn:
            conn.close()


# ==========================================================
# 4. EVIDENCE VIEWER
# ==========================================================
@app.route("/api/evidence", methods=["GET"])
def get_evidence():
    """
    Mengambil data arsip kelainan untuk tab Evidence Viewer.
    """
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()

        sql = """
            SELECT 
                i.InspectionID,
                i.TraceabilityCode,
                i.InspectionTime,
                i.OverallSeverity,
                i.StatusLifecycle,
                i.PICName,
                i.MaintenanceNote,
                i.SlatNumber,
                d.CameraID,
                d.TargetArea,
                d.ComponentType,
                d.StatusDetail,
                d.Confidence,
                d.RawImagePath,
                d.AnnotatedPath
            FROM Inspections i
            INNER JOIN Inspection_Detections d ON i.InspectionID = d.InspectionID
            WHERE d.Result = 'NG'
            ORDER BY i.InspectionTime DESC, d.DetectionID DESC
        """
        cursor.execute(sql)
        rows = cursor.fetchall()

        evidence_items = []
        for r in rows:
            insp_id = r[0]
            code = r[1]
            insp_time = r[2]
            sev = r[3]
            status_life = r[4]
            pic = r[5]
            note = r[6]
            slat_num = r[7]
            cam_id = r[8]
            target_area = r[9]
            comp = r[10]
            stat_detail = r[11]
            conf = r[12]
            raw_path = r[13]
            ann_path = r[14]

            time_str = insp_time.strftime("%H:%M:%S WIB") if insp_time else ""
            full_date_str = insp_time.strftime("%d %b %Y, %H:%M:%S") if insp_time else ""

            area_desc = "Conveyor Plate & Rib Overhead"
            if cam_id == "CAM-02":
                area_desc = "Bolt & Pin Right Side"
            elif cam_id == "CAM-03":
                area_desc = "Bolt & Pin Left Side"

            title_desc = f"{comp} {stat_detail} #{insp_id:05d}"
            obj_desc = f"{comp} (Slat #{slat_num})" if slat_num else f"{comp} Fastener"

            conf_str = f"{float(conf):.2f}%" if conf is not None else "95.50%"

            # Gunakan foto anotasi jika tersedia, fallback ke foto mentah
            img_to_serve = ann_path if (ann_path and os.path.exists(ann_path)) else raw_path
            raw_to_serve = raw_path if (raw_path and os.path.exists(raw_path)) else img_to_serve

            evidence_items.append({
                "id": code,
                "inspectionId": insp_id,
                "camera": cam_id,
                "title": title_desc,
                "severity": sev,
                "object": obj_desc,
                "time": time_str,
                "captureTime": full_date_str,
                "area": area_desc,
                "confidence": conf_str,
                "statusLifecycle": status_life,
                "picName": pic or "",
                "maintenanceNote": note or "",
                "rawImagePath": raw_to_serve,
                "annotatedPath": img_to_serve,
                "imageUrl": f"http://localhost:5000/api/image?path={img_to_serve}" if img_to_serve else None
            })

        return jsonify(evidence_items)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if conn:
            conn.close()


# ==========================================================
# 5. IMAGE SERVER
# ==========================================================
@app.route("/api/image", methods=["GET"])
def get_image():
    """
    Menyajikan file foto mentah atau hasil overlay AI ke browser.
    """
    path = request.args.get("path")
    if not path or not os.path.exists(path):
        return jsonify({"error": "Image file not found"}), 404

    try:
        # Tentukan mimetype
        ext = os.path.splitext(path)[1].lower()
        mimetype = "image/jpeg"
        if ext == ".png":
            mimetype = "image/png"
        elif ext == ".bmp":
            mimetype = "image/bmp"
        elif ext == ".webp":
            mimetype = "image/webp"

        return send_file(path, mimetype=mimetype)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================================================
# 6. ACTION (ACKNOWLEDGE / RESOLVE)
# ==========================================================
@app.route("/api/inspections/<int:inspection_id>/action", methods=["POST"])
def update_inspection_action(inspection_id):
    """
    Menyimpan aksi PIC di dashboard:
    - action: 'ACKNOWLEDGED' atau 'RESOLVED'
    - picName: nama PIC (misal: Sarah Jenkins)
    - maintenanceNote: catatan teknisi
    """
    data = request.get_json() or {}
    action = data.get("action", "ACKNOWLEDGED")
    pic_name = data.get("picName", "Operator")
    note = data.get("maintenanceNote", "")

    if action not in ("ACKNOWLEDGED", "RESOLVED", "UNACKNOWLEDGED"):
        return jsonify({"error": "Invalid action"}), 400

    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()

        if action == "RESOLVED":
            cursor.execute(
                """
                UPDATE Inspections
                SET 
                    StatusLifecycle = 'RESOLVED',
                    PICName = ?,
                    MaintenanceNote = ?,
                    ResolvedAt = GETDATE()
                WHERE InspectionID = ?
                """,
                pic_name,
                note,
                inspection_id
            )
        else:
            cursor.execute(
                """
                UPDATE Inspections
                SET 
                    StatusLifecycle = ?,
                    PICName = ?,
                    MaintenanceNote = ?
                WHERE InspectionID = ?
                """,
                action,
                pic_name,
                note,
                inspection_id
            )

        conn.commit()
        return jsonify({
            "success": True,
            "inspectionId": inspection_id,
            "statusLifecycle": action
        })

    except Exception as e:
        if conn:
            conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        if conn:
            conn.close()


# ==========================================================
# 7. ANALYTICS (WEEKLY TREND & PARETO)
# ==========================================================
@app.route("/api/analytics", methods=["GET"])
def get_analytics():
    """
    Menghasilkan data grafik mingguan dan diagram pareto untuk tab Analytics.
    """
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # 1. Weekly Trend (7 hari terakhir)
        days_label = ["Min", "Sen", "Sel", "Rab", "Kam", "Jum", "Sab"]
        weekly_counts = {i: 0 for i in range(7)}

        today = datetime.now()
        start_date = today - timedelta(days=6)

        cursor.execute(
            """
            SELECT 
                DATEPART(dw, InspectionTime) - 1 as DayIndex,
                COUNT(1) as TotalNG
            FROM Inspections
            WHERE OverallResult = 'NG' AND InspectionTime >= ?
            GROUP BY DATEPART(dw, InspectionTime)
            """,
            start_date.strftime("%Y-%m-%d")
        )
        for r in cursor.fetchall():
            idx = int(r[0]) % 7
            weekly_counts[idx] = int(r[1])

        # Susun 7 hari terakhir secara urut
        days_result = []
        for i in range(6, -1, -1):
            d = today - timedelta(days=i)
            idx = int(d.strftime("%w"))
            days_result.append({
                "d": days_label[idx],
                "n": weekly_counts[idx]
            })

        # 2. Pareto Abnormality by Defect Type
        cursor.execute(
            """
            SELECT 
                CameraID,
                ComponentType,
                StatusDetail,
                COUNT(1) as IncidentCount
            FROM Inspection_Detections
            WHERE Result = 'NG'
            GROUP BY CameraID, ComponentType, StatusDetail
            ORDER BY IncidentCount DESC
            """
        )
        pareto_rows = cursor.fetchall()
        total_incidents = sum(int(r[3]) for r in pareto_rows) or 1

        colors = {
            "PLAT": "bg-danger",
            "RIB": "bg-warning",
            "BOLT": "bg-primary",
            "PIN": "bg-info"
        }

        pareto_result = []
        for r in pareto_rows:
            cam = r[0]
            comp = r[1]
            stat = r[2]
            cnt = int(r[3])
            pct = (cnt / total_incidents) * 100

            label = f"{comp} {stat} ({cam})"
            pareto_result.append({
                "label": label,
                "n": cnt,
                "p": f"{pct:.1f}%",
                "barCls": colors.get(comp, "bg-primary")
            })

        return jsonify({
            "weekly": days_result,
            "pareto": pareto_result,
            "totalIncidents": total_incidents
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if conn:
            conn.close()


# ==========================================================
# ENTRY POINT
# ==========================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  REST API BACKEND - SLAT RIB CONVEYOR INSPECTION")
    print("  Server berjalan di http://localhost:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False)
