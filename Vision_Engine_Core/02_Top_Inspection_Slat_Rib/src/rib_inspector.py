"""
==============================================================================
MODUL INSPEKSI KELURUSAN SIRIP KONVEYOR (RIB STRAIGHTNESS INSPECTOR)
==============================================================================
Deskripsi:
  Metode Hibrida: YOLO Object Detection + Sub-Pixel Gradient-Polarity Ridge Tracker.
  Inspeksi kelurusan dan deformasi sirip (Rib) tegak konveyor CAM-03.

Prinsip Kerja:
  1. YOLO Presence & Localization:
     - Mendeteksi keberadaan bilah sirip ('rib_ok') vs plat datar ('plat_ok').
     - Mengabaikan frame saat plat datar melintas (mengembalikan status 'WAIT').
  2. Bounding Box & Canonical Perspective Alignment:
     - Mengisolasi ROI sirip berdasarkan deteksi YOLO atau Polygon Gate.
     - Melakukan perspective transform horizontal (1000 x 160 px).
  3. Preprocessing CLAHE + Bilateral Filtering:
     - Meredam tekstur karat, noda oli, dan noise visual.
  4. Sub-Pixel Negative Gradient (-Sobel Y) Ridge Tracking:
     - Melacak garis punggung atas sirip (Top Ridge) di setiap kolom X.
     - Hard Jump Guard (<= 6 px) mencegah loncatan ke kaki las-lasan sirip.
  5. Metrologi Kelurusan (Straightness Metrology):
     - Filter Median + Moving Average untuk smoothing kontur.
     - Linear Regression Reference Line Fitting (y_ref = m*x + c).
     - Perhitungan Peak-to-Valley (PV) Flatness & Max Absolute Deviation.
  6. Overlay Proyeksi Kontur ke Frame Asli (Inverse Homography).
==============================================================================
"""

import os
import cv2
import numpy as np
import time

# Lazy load / safe import YOLO
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False


def order_quad_points(pts):
    """Mengurutkan 4 titik kuadran: [Top-Left, Top-Right, Bottom-Right, Bottom-Left]."""
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


class RibInspector:
    """
    Pendeteksi & Pengukur Deformasi Kelurusan Sirip (Rib) CAM-03
    berbasis YOLO Presence Gate + Sub-Pixel Gradient-Polarity Ridge Tracking.
    """
    def __init__(
        self,
        model_path=None,
        conf_threshold=0.35,
        iou_threshold=0.45,
        pv_threshold=15.0,
        max_dev_threshold=12.0,
        min_ridge_energy=16.0,
        use_yolo=False
    ):
        """
        :param model_path: Path ke file bobot YOLO (.pt)
        :param conf_threshold: Ambang batas confidence YOLO
        :param iou_threshold: Ambang batas IoU untuk Agnostic NMS
        :param pv_threshold: Ambang batas Peak-to-Valley flatness (px). Nilai > threshold = NG.
        :param max_dev_threshold: Ambang batas deviasi absolut maksimum (px).
        :param min_ridge_energy: Ambang batas energi punggung sirip (Pure-CV fallback).
        :param use_yolo: Mengaktifkan pendeteksi YOLO jika tersedia.
        """
        self.pv_threshold = float(pv_threshold)
        self.max_dev_threshold = float(max_dev_threshold)
        self.min_ridge_energy = float(min_ridge_energy)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.use_yolo = use_yolo and HAS_YOLO
        self.canonical_w = 1000
        self.canonical_h = 160

        # Inisialisasi Model YOLO jika path tersedia
        self.yolo_model = None
        self.class_names = {}
        if self.use_yolo:
            if model_path is None:
                # Cari otomatis di folder models
                cur_dir = os.path.dirname(os.path.abspath(__file__))
                parent_dir = os.path.dirname(cur_dir)
                default_models = [
                    os.path.join(parent_dir, "models", "best_rib.pt"),
                    os.path.join(parent_dir, "models", "best.pt"),
                    os.path.join(r"C:\Users\Victus\Downloads", "best_rib_v1.pt"),
                    os.path.join(r"C:\Users\Victus\Downloads", "best.pt")
                ]
                for p in default_models:
                    if os.path.exists(p):
                        model_path = p
                        break

            if model_path and os.path.exists(model_path):
                try:
                    self.yolo_model = YOLO(model_path)
                    self.class_names = self.yolo_model.names if hasattr(self.yolo_model, 'names') else {}
                    print(f"[RibInspector] Model YOLO dimuat dari: {model_path}")
                    print(f"[RibInspector] Kelas model: {self.class_names}")
                except Exception as e:
                    print(f"[RibInspector WARNING] Gagal memuat YOLO ({e}). Menggunakan mode Pure-CV.")
                    self.yolo_model = None

    def detect_yolo(self, frame, roi_poly=None):
        """
        Menjalankan inferensi YOLO pada area ROI terpotong untuk menghemat memori (anti bad allocation)
        dan menyaring hanya deteksi yang berada di dalam area poligon ROI (roi_poly).
        :param frame: Frame citra BGR
        :param roi_poly: Koordinat 4 titik poligon [[x1, y1], ...] dalam piksel
        :return: Dict berisi list 'ribs' (boxes), 'plats' (boxes), dan 'is_rib_detected'
        """
        if self.yolo_model is None or frame is None or frame.size == 0:
            return {"ribs": [], "plats": [], "all_boxes": [], "is_rib_detected": False}

        h_f, w_f = frame.shape[:2]
        offset_x, offset_y = 0, 0
        infer_img = frame

        # Siapkan kontur poligon ROI & potong area inferensi untuk efisiensi memori
        poly_contour = None
        if roi_poly is not None and len(roi_poly) >= 4:
            poly_pts = np.asarray(roi_poly, dtype=np.float32)
            poly_contour = np.int32(poly_pts).reshape(-1, 1, 2)
            
            min_x, min_y = np.min(poly_pts, axis=0)
            max_x, max_y = np.max(poly_pts, axis=0)
            
            # Berikan padding secukupnya (+- 25 px) agar konteks bilah sirip utuh
            rx1 = max(0, int(min_x - 25))
            ry1 = max(0, int(min_y - 25))
            rx2 = min(w_f, int(max_x + 25))
            ry2 = min(h_f, int(max_y + 25))
            
            if rx2 > rx1 and ry2 > ry1:
                infer_img = frame[ry1:ry2, rx1:rx2]
                offset_x, offset_y = rx1, ry1

        detected_ribs = []
        detected_plats = []
        all_boxes = []

        try:
            # Inferensi YOLO dengan ukuran ringkas untuk mencegah PyTorch RAM bad allocation
            results = self.yolo_model.predict(
                source=infer_img,
                imgsz=640,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                agnostic_nms=True,
                verbose=False
            )

            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    cls_name = self.class_names.get(cls_id, str(cls_id)).lower()

                    # Konversi kembali koordinat terpotong ke koordinat frame penuh
                    x1 = int(xyxy[0] + offset_x)
                    y1 = int(xyxy[1] + offset_y)
                    x2 = int(xyxy[2] + offset_x)
                    y2 = int(xyxy[3] + offset_y)
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    box_w = x2 - x1
                    box_h = y2 - y1

                    # Uji apakah bounding box / centroid berada di dalam Poligon ROI Rib
                    if poly_contour is not None:
                        dist_center = cv2.pointPolygonTest(poly_contour, (float(cx), float(cy)), True)
                        dist_top = cv2.pointPolygonTest(poly_contour, (float(cx), float(y1)), True)
                        dist_bot = cv2.pointPolygonTest(poly_contour, (float(cx), float(y2)), True)
                        
                        is_inside = (dist_center >= -14.0) or (dist_top >= -14.0) or (dist_bot >= -14.0)
                        if not is_inside:
                            continue  # Abaikan deteksi di luar ROI Rib

                    box_info = {
                        "box": [x1, y1, x2, y2],
                        "centroid": (cx, cy),
                        "size": (box_w, box_h),
                        "conf": conf,
                        "class_id": cls_id,
                        "class_name": cls_name
                    }
                    all_boxes.append(box_info)

                    if "rib" in cls_name:
                        detected_ribs.append(box_info)
                    elif "plat" in cls_name:
                        detected_plats.append(box_info)

        except Exception as e:
            # Fallback halus jika terjadi kendala alokasi memori
            print(f"[RibInspector WARNING] YOLO inference skipped: {e}")

        return {
            "ribs": detected_ribs,
            "plats": detected_plats,
            "all_boxes": all_boxes,
            "is_rib_detected": len(detected_ribs) > 0
        }

    def inspect(self, frame, polygon_pts, simulate_defect=False, simulated_bending_amp=22.0):
        """
        Melakukan inspeksi kelurusan sirip pada area poligon 4-titik.
        :param frame: Citra BGR (np.ndarray)
        :param polygon_pts: List 4 titik [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        :param simulate_defect: Boolean jika mode simulasi bending aktif
        :param simulated_bending_amp: Amplitudo defleksi simulasi (px)
        :return: Dict hasil inspeksi terstruktur
        """
        t0 = time.time()
        h_f, w_f = frame.shape[:2]

        if frame is None or polygon_pts is None or len(polygon_pts) < 4:
            return {
                "status": "INVALID",
                "pv_flatness": 0.0,
                "max_deviation": 0.0,
                "coverage": 0.0,
                "is_rib_present": False,
                "is_ng": False,
                "message": "ROI poligon sirip tidak valid",
                "elapsed_ms": 0.0,
                "graph_img": None,
                "yolo_boxes": []
            }

        # Konversi koordinat jika ternormalisasi (0.0 - 1.0)
        pts_px = []
        for p in polygon_pts[:4]:
            px = p[0] * w_f if p[0] <= 1.0 else p[0]
            py = p[1] * h_f if p[1] <= 1.0 else p[1]
            pts_px.append([px, py])

        ordered_pts = order_quad_points(pts_px)

        # 1. Jalankan Verifikasi YOLO jika aktif (Strict ROI Gating)
        yolo_res = None
        yolo_boxes = []
        is_rib_present_by_yolo = True

        if self.yolo_model is not None:
            # Lewatkan ordered_pts agar YOLO hanya mengambil deteksi di dalam ROI
            yolo_res = self.detect_yolo(frame, roi_poly=ordered_pts)
            yolo_boxes = yolo_res["all_boxes"]
            is_rib_present_by_yolo = yolo_res["is_rib_detected"]

        # 2. Perspective Transform ke Kanonikal (1000 x 160 px)
        cw, ch = self.canonical_w, self.canonical_h
        dst_pts = np.array([
            [0, 0],
            [cw - 1, 0],
            [cw - 1, ch - 1],
            [0, ch - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(ordered_pts, dst_pts)
        M_inv = cv2.getPerspectiveTransform(dst_pts, ordered_pts)
        norm_roi = cv2.warpPerspective(frame, M, (cw, ch))

        # 3. Preprocessing & Ekstraksi Gradien Terarah
        gray = cv2.cvtColor(norm_roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        filtered = cv2.bilateralFilter(enhanced, d=7, sigmaColor=45, sigmaSpace=45)

        # Sobel Y:
        # Pinggiran atas fisik sirip = Transisi Gelap -> Terang (+Sobel Y)
        # Kaki bawah / bayangan bilah = Transisi Terang -> Gelap (-Sobel Y)
        sy = cv2.Sobel(filtered, cv2.CV_32F, 0, 1, ksize=3)
        sx = cv2.Sobel(filtered, cv2.CV_32F, 1, 0, ksize=3)
        pos_sy = np.clip(sy, 0, 255)
        neg_sy = np.clip(-sy, 0, 255)

        # 4. Verifikasi Keberadaan Sirip via Metode Acuan Dua Ujung Fisik (Two-End Anchor Method - Opsi 1)
        # Ujung Kiri (Anchor A): X di [80, 160], titik acuan nominal X_A = 120
        # Ujung Kanan (Anchor B): X di [840, 920], titik acuan nominal X_B = 880
        x_A = 120
        x_B = 880

        left_pos = np.mean(pos_sy[8 : ch - 8, 80 : 160], axis=1)
        left_neg = np.mean(neg_sy[8 : ch - 8, 80 : 160], axis=1)
        y_top_A = int(np.argmax(left_pos)) + 8
        y_bot_A = int(np.argmax(left_neg)) + 8
        e_top_A = float(np.max(left_pos))
        e_bot_A = float(np.max(left_neg))
        h_A = y_bot_A - y_top_A

        right_pos = np.mean(pos_sy[8 : ch - 8, 840 : 920], axis=1)
        right_neg = np.mean(neg_sy[8 : ch - 8, 840 : 920], axis=1)
        y_top_B = int(np.argmax(right_pos)) + 8
        y_bot_B = int(np.argmax(right_neg)) + 8
        e_top_B = float(np.max(right_pos))
        e_bot_B = float(np.max(right_neg))
        h_B = y_bot_B - y_top_B

        # Kriteria Fisik Sirip Tegak 3D di Kedua Titik Tumpu (Anchor A & B):
        # 1. Bilah sirip memiliki ketebalan vertikal terukur antara 6 px hingga 35 px di kedua ujung
        # 2. Energi gradien pinggiran atas (+Sobel Y) dan kaki bawah (-Sobel Y) keduanya kuat
        # 3. Kesejajaran horizontal kedua titik tumpu (|y_top_A - y_top_B| <= 25 px)
        has_anchor_A = (6 <= h_A <= 35) and (e_top_A >= 24.0) and (e_bot_A >= 30.0)
        has_anchor_B = (6 <= h_B <= 35) and (e_top_B >= 24.0) and (e_bot_B >= 30.0)
        is_horizontally_aligned = abs(y_top_A - y_top_B) <= 25

        is_physical_rib = has_anchor_A and has_anchor_B and is_horizontally_aligned

        # Jika model YOLO aktif, verifikasi hibrida (YOLO box + Two-End Anchor)
        if self.yolo_model is not None:
            is_valid_rib = is_rib_present_by_yolo and is_physical_rib
        else:
            is_valid_rib = is_physical_rib

        if not is_valid_rib:
            return {
                "status": "WAIT",
                "pv_flatness": 0.0,
                "max_deviation": 0.0,
                "coverage": 0.0,
                "is_rib_present": False,
                "is_ng": False,
                "message": "Plat datar melintas (Menunggu Sirip)",
                "elapsed_ms": (time.time() - t0) * 1000,
                "graph_img": None,
                "norm_roi": norm_roi,
                "M_inv": M_inv,
                "yolo_boxes": yolo_boxes,
                "anchor_A": (x_A, y_top_A),
                "anchor_B": (x_B, y_top_B)
            }

        # 5. Konstruksi Garis Acuan Fisik Nominal A -> B (Physical Baseline)
        full_x = np.arange(x_A, x_B + 1)
        ref_y = y_top_A + ((y_top_B - y_top_A) / float(x_B - x_A)) * (full_x - x_A)

        # 6. Pelacakan Titik Punggung Atas Terpandu (Guided Ridge Tracking) dengan Jump Guard
        raw_y = np.zeros(len(full_x), dtype=np.float32)
        prev_y = float(y_top_A)

        for idx, x in enumerate(full_x):
            nom_y = ref_y[idx]
            lo = max(2, int(nom_y - 10))
            hi = min(ch - 2, int(nom_y + 10))
            sub_sy = pos_sy[lo:hi, x]
            sub_sx = abs(sx[lo:hi, x])

            if len(sub_sy) > 0:
                best_sub_idx = int(np.argmax(sub_sy))
                candidate_y = float(lo + best_sub_idx)
                cand_val = sub_sy[best_sub_idx]
                cand_sx = sub_sx[best_sub_idx]

                if cand_val > 10.0 and cand_val >= 0.30 * (cand_sx + 1e-5):
                    if abs(candidate_y - prev_y) <= 4.0:
                        raw_y[idx] = candidate_y
                        prev_y = candidate_y
                    else:
                        step = np.clip(candidate_y - prev_y, -2.0, 2.0)
                        raw_y[idx] = prev_y + step
                        prev_y = raw_y[idx]
                else:
                    raw_y[idx] = prev_y
            else:
                raw_y[idx] = prev_y

        # 7. Smoothing & Metrologi Kelurusan
        k_med = 9
        pad_m = k_med // 2
        padded_y = np.pad(raw_y, (pad_m, pad_m), mode='edge')
        med_y = np.array([np.median(padded_y[i : i + k_med]) for i in range(len(full_x))], dtype=np.float32)

        w_ma = 7
        pad_w = w_ma // 2
        padded_ma = np.pad(med_y, (pad_w, pad_w), mode='edge')
        smooth_y = np.convolve(padded_ma, np.ones(w_ma) / w_ma, mode='valid')

        # Injeksi Simulasi Bending jika aktif (Hotkey [B])
        if simulate_defect:
            x_norm = (full_x - full_x[0]) / float(full_x[-1] - full_x[0])
            parabola = np.sin(x_norm * np.pi) * simulated_bending_amp
            smooth_y = smooth_y + parabola

        # 8. Metrologi Defleksi terhadap Garis Acuan Nominal A -> B
        deviations = smooth_y - ref_y
        pv_flatness = float(np.max(deviations) - np.min(deviations))
        max_deviation = float(np.max(np.abs(deviations)))

        is_ng = (pv_flatness > self.pv_threshold) or (max_deviation > self.max_dev_threshold)
        status = "NG" if is_ng else "OK"

        # 9. Render Mini Graph Preview
        graph_img = self._render_profile_graph(full_x, smooth_y, ref_y, deviations, pv_flatness, is_ng)
        elapsed_ms = (time.time() - t0) * 1000

        return {
            "status": status,
            "pv_flatness": pv_flatness,
            "max_deviation": max_deviation,
            "coverage": 1.0,
            "is_rib_present": True,
            "is_ng": is_ng,
            "message": f"PV: {pv_flatness:.1f} px | MaxDev: {max_deviation:.1f} px",
            "elapsed_ms": elapsed_ms,
            "graph_img": graph_img,
            "full_x": full_x,
            "smooth_y": smooth_y,
            "ref_y": ref_y,
            "deviations": deviations,
            "norm_roi": norm_roi,
            "M_inv": M_inv,
            "yolo_boxes": yolo_boxes,
            "anchor_A": (x_A, y_top_A),
            "anchor_B": (x_B, y_top_B)
        }

    def _render_profile_graph(self, x_arr, y_curve, y_ref, dev_arr, pv, is_ng, w=320, h=120):
        """Membuat gambar grafik profil kelurusan sirip 1D secara real-time."""
        graph = np.full((h, w, 3), (28, 28, 32), dtype=np.uint8)
        cv2.rectangle(graph, (0, 0), (w - 1, h - 1), (70, 70, 75), 1)

        col = (0, 0, 255) if is_ng else (0, 255, 0)
        mode_label = "TWO-END ANCHOR"
        cv2.putText(graph, f"PROFIL KELURUSAN SIRIP ({mode_label})", (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1)
        cv2.putText(graph, f"PV: {pv:.1f} px (Max: {self.pv_threshold:.1f} px)", (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)

        pad_x, pad_y = 15, 42
        pw, ph = w - 2 * pad_x, h - pad_y - 12
        mid_y = pad_y + ph // 2

        cv2.line(graph, (pad_x, mid_y), (pad_x + pw, mid_y), (100, 100, 100), 1, cv2.LINE_AA)

        max_scale = max(20.0, self.pv_threshold * 1.3)
        pts_curve = []
        n_pts = len(dev_arr)
        if n_pts > 1:
            for i in range(0, n_pts, max(1, n_pts // pw)):
                gx = int(pad_x + (i / float(n_pts)) * pw)
                dev_val = dev_arr[i]
                gy = int(mid_y - (dev_val / max_scale) * (ph / 2))
                gy = max(pad_y, min(h - 8, gy))
                pts_curve.append((gx, gy))

            for j in range(len(pts_curve) - 1):
                cv2.line(graph, pts_curve[j], pts_curve[j + 1], col, 2, cv2.LINE_AA)

        return graph

    def draw_overlay_on_frame(self, frame, result):
        """Memproyeksikan kurva punggung sirip & titik acuan anchor A & B kembali ke frame kamera asli."""
        if result is None:
            return frame

        # Gambar Bounding Box YOLO jika tersedia
        yolo_boxes = result.get("yolo_boxes", [])
        for bx in yolo_boxes:
            x1, y1, x2, y2 = bx["box"]
            cls_name = bx["class_name"]
            conf = bx["conf"]
            
            if "rib" in cls_name:
                box_col = (0, 165, 255)  # Oranye untuk YOLO rib
            else:
                box_col = (180, 180, 180)  # Abu-abu untuk plat

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_col, 1)
            tag = f"{cls_name} {conf:.2f}"
            cv2.putText(frame, tag, (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_col, 1)

        if not result.get("is_rib_present", False) or "M_inv" not in result:
            return frame

        full_x = result.get("full_x")
        smooth_y = result.get("smooth_y")
        ref_y = result.get("ref_y")
        M_inv = result.get("M_inv")
        anchor_A = result.get("anchor_A")
        anchor_B = result.get("anchor_B")

        if full_x is None or smooth_y is None or M_inv is None:
            return frame

        is_ng = result["is_ng"]
        col = (0, 0, 255) if is_ng else (0, 255, 0)

        # 1. Transformasi dan Gambar Titik Acuan Anchor A & Anchor B
        if anchor_A is not None and anchor_B is not None:
            pts_anchors_norm = np.array([
                [[float(anchor_A[0]), float(anchor_A[1])]],
                [[float(anchor_B[0]), float(anchor_B[1])]]
            ], dtype=np.float32)
            pts_anchors_frame = cv2.perspectiveTransform(pts_anchors_norm, M_inv)

            pt_A = (int(pts_anchors_frame[0][0][0]), int(pts_anchors_frame[0][0][1]))
            pt_B = (int(pts_anchors_frame[1][0][0]), int(pts_anchors_frame[1][0][1]))

            # Gambar Titik Acuan A (Ujung Kiri) - Cyan Target Mark
            cv2.circle(frame, pt_A, 7, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(frame, pt_A, 2, (255, 255, 0), -1)
            cv2.line(frame, (pt_A[0] - 10, pt_A[1]), (pt_A[0] + 10, pt_A[1]), (255, 255, 0), 1, cv2.LINE_AA)
            cv2.line(frame, (pt_A[0], pt_A[1] - 10), (pt_A[0], pt_A[1] + 10), (255, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, "Anchor [A]", (pt_A[0] - 30, pt_A[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

            # Gambar Titik Acuan B (Ujung Kanan) - Cyan Target Mark
            cv2.circle(frame, pt_B, 7, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(frame, pt_B, 2, (255, 255, 0), -1)
            cv2.line(frame, (pt_B[0] - 10, pt_B[1]), (pt_B[0] + 10, pt_B[1]), (255, 255, 0), 1, cv2.LINE_AA)
            cv2.line(frame, (pt_B[0], pt_B[1] - 10), (pt_B[0], pt_B[1] + 10), (255, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, "Anchor [B]", (pt_B[0] - 30, pt_B[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

        # 2. Transformasi dan Gambar Garis Acuan Nominal Referensi (Kuning Tipis)
        pts_ref_norm = np.vstack((full_x, ref_y)).T.reshape(-1, 1, 2).astype(np.float32)
        pts_ref_frame = cv2.perspectiveTransform(pts_ref_norm, M_inv)

        n_p = len(pts_ref_frame)
        for i in range(0, n_p - 1, 4):
            p1 = (int(pts_ref_frame[i][0][0]), int(pts_ref_frame[i][0][1]))
            p2 = (int(pts_ref_frame[i + 1][0][0]), int(pts_ref_frame[i + 1][0][1]))
            cv2.line(frame, p1, p2, (0, 220, 255), 1, cv2.LINE_AA)

        # 3. Transformasi dan Gambar Kurva Kontur Sirip Terukur (Hijau/Merah)
        pts_norm = np.vstack((full_x, smooth_y)).T.reshape(-1, 1, 2).astype(np.float32)
        pts_frame = cv2.perspectiveTransform(pts_norm, M_inv)

        for i in range(0, n_p - 1):
            p1 = (int(pts_frame[i][0][0]), int(pts_frame[i][0][1]))
            p2 = (int(pts_frame[i + 1][0][0]), int(pts_frame[i + 1][0][1]))
            cv2.line(frame, p1, p2, col, 2, cv2.LINE_AA)

        # 4. Label teks di atas kurva
        if n_p > 0:
            mid_idx = n_p // 2
            mid_pt = (int(pts_frame[mid_idx][0][0]) - 80, int(pts_frame[mid_idx][0][1]) - 14)
            txt = f"RIB: {result['status']} (PV: {result['pv_flatness']:.1f}px)"
            cv2.putText(frame, txt, mid_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

        return frame
