"""
==============================================================================
MODUL INSPEKSI CELAH & KERATAAN PLAT KONVEYOR (SLAT GAP & PROFILE INSPECTOR)
==============================================================================
Deskripsi:
  Modul metrologi celah dan kelengkungan sambungan plat konveyor CAM-03
  menggunakan Metode Dual-Edge Gradient Profiling & Euler-Bernoulli Beam Modeling.
  
Prinsip Kerja:
  1. Melokalisasi lembah celah lokal menggunakan Black Top-Hat & Row Energy Projection.
  2. Mengekstrak pasangan transisi tepi fisik vertikal (Sobel-Y):
     - Tepi Atas: Transisi Terang -> Gelap (Gradien Negatif Kuat).
     - Tepi Bawah: Transisi Gelap -> Terang (Gradien Positif Kuat).
  3. Memodelkan kelengkungan defleksi batang menggunakan Quadratic Beam Model (D=2).
  4. Menghitung profil ketebalan fisik H_gap(x) = Y_bot_fit(x) - Y_top_fit(x).
  5. Linear Detrending untuk mengeliminasi sudut kemiringan kamera.
  6. Evaluasi Curvature Defleksi dan Maksimum Lebar Celah.
  7. Proyeksi kontur celah kembali ke koordinat frame asli (Inverse Homography).
==============================================================================
"""

import cv2
import numpy as np
import time


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


class SlatGapInspector:
    """
    Pendeteksi & Pengukur Deformasi Plat & Celah Sambungan Antar-Slat Konveyor.
    """
    def __init__(self, delta_limit=14.0, max_gap_limit=45.0, valley_sens=5.0):
        """
        :param delta_limit: Ambang batas kelengkungan / curvature deviasi celah (px). Nilai > limit = NG.
        :param max_gap_limit: Ambang batas celah renggang maksimum antar-plat (px).
        :param valley_sens: Sensitivitas deteksi lembah celah aktif.
        """
        self.delta_limit = float(delta_limit)
        self.max_gap_limit = float(max_gap_limit)
        self.valley_sens = float(valley_sens)
        self.canonical_w = 1000
        self.canonical_h = 140

    def inspect(self, frame, polygon_pts):
        """
        Melakukan inspeksi celah dan kelurusan plat pada area poligon 4-titik.
        :param frame: Citra BGR (np.ndarray)
        :param polygon_pts: List 4 titik [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        :return: Dict hasil inspeksi terstruktur
        """
        t0 = time.time()
        h_f, w_f = frame.shape[:2]

        if frame is None or polygon_pts is None or len(polygon_pts) < 4:
            return {
                "status": "INVALID",
                "curvature": 0.0,
                "gap_mean": 0.0,
                "gap_max": 0.0,
                "is_transition": True,
                "is_ng": False,
                "message": "ROI poligon slat tidak valid",
                "elapsed_ms": 0.0,
                "graph_img": None
            }

        # Konversi titik jika ternormalisasi (0.0 - 1.0)
        pts_px = []
        for p in polygon_pts[:4]:
            px = p[0] * w_f if p[0] <= 1.0 else p[0]
            py = p[1] * h_f if p[1] <= 1.0 else p[1]
            pts_px.append([px, py])

        ordered_pts = order_quad_points(pts_px)

        # 1. Perspective Transform ke Kanonikal (1000 x 140 px)
        cw, ch = self.canonical_w, self.canonical_h
        dst_pts = np.array([
            [0, 0],
            [cw - 1, 0],
            [cw - 1, ch - 1],
            [0, ch - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(ordered_pts, dst_pts)
        M_inv = cv2.getPerspectiveTransform(dst_pts, ordered_pts)
        norm_seam = cv2.warpPerspective(frame, M, (cw, ch))

        # 2. Filtering & Dual-Edge Profiling
        gray = cv2.cvtColor(norm_seam, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, d=7, sigmaColor=45, sigmaSpace=45)

        # Morphological Black Top-Hat vertikal untuk isolasi lembah celah
        k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 65))
        bhat = cv2.morphologyEx(filtered, cv2.MORPH_BLACKHAT, k_v)

        sx = cv2.Sobel(filtered, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(filtered, cv2.CV_32F, 0, 1, ksize=3)

        # 3. Deteksi Lembah Celah Aktif (Row Energy Projection)
        margin = int(cw * 0.10)
        row_energy = np.sum(bhat[:, margin : cw - margin], axis=1)

        lo, hi = 20, 120
        seam_y_peak = int(np.argmax(row_energy[lo:hi])) + lo
        peak_energy = row_energy[seam_y_peak] / float(cw - 2 * margin)

        # Jika energi lembah terlalu rendah (saat badan plat penuh mengisi zona inspeksi)
        if peak_energy < self.valley_sens:
            return {
                "status": "WAIT",
                "curvature": 0.0,
                "gap_mean": 0.0,
                "gap_max": 0.0,
                "is_transition": True,
                "is_ng": False,
                "message": "Menunggu celah sambungan plat melintas...",
                "elapsed_ms": (time.time() - t0) * 1000,
                "graph_img": None,
                "norm_seam": norm_seam,
                "M_inv": M_inv
            }

        # 4. Ekstraksi Pasangan Tepi Fisik (Top & Bottom Edge) Cepat (Vectorized)
        local_lo = max(2, seam_y_peak - 20)
        local_hi = min(ch - 3, seam_y_peak + 20)

        raw_valley_center = np.argmax(bhat[local_lo:local_hi, :], axis=0) + local_lo
        pad_c = 7
        p_c = np.pad(raw_valley_center, (pad_c, pad_c), mode='edge')
        center_y_smooth = np.array([np.median(p_c[i : i + 15]) for i in range(cw)], dtype=np.float32)

        # Precompute fast contrast maps via box filter (sub-millisecond)
        radius = 5
        filt_f = filtered.astype(np.float32)
        k_box = np.ones((radius, 1), dtype=np.float32) / float(radius)
        above_map = cv2.filter2D(filt_f, -1, k_box, anchor=(0, radius - 1))
        below_map = cv2.filter2D(filt_f, -1, k_box, anchor=(0, 0))
        contrast_t = np.maximum(0.0, above_map - below_map)
        contrast_b = np.maximum(0.0, below_map - above_map)

        valid_grad_mask_t = (sy <= -2.0) & (np.abs(sy) >= 0.40 * (np.abs(sx) + 1e-5)) & (contrast_t >= 1.0)
        valid_grad_mask_b = (sy >= 2.0) & (np.abs(sy) >= 0.40 * (np.abs(sx) + 1e-5)) & (contrast_b >= 1.0)

        score_matrix_t = np.where(valid_grad_mask_t, -sy * 0.4 + contrast_t * 0.6, -1e6).astype(np.float32)
        score_matrix_b = np.where(valid_grad_mask_b, sy * 0.4 + contrast_b * 0.6, -1e6).astype(np.float32)

        top_y_raw = np.zeros(cw, dtype=np.float32)
        bot_y_raw = np.zeros(cw, dtype=np.float32)

        for x in range(cw):
            cy = int(round(center_y_smooth[x]))

            # Tepi Atas: Gradien Negatif Kuat (-Sy)
            y_t_start = max(radius, cy - 25)
            y_t_end = min(ch - 1 - radius, cy + 4)
            if y_t_end > y_t_start:
                sub_scores_t = score_matrix_t[y_t_start:y_t_end, x]
                best_t_idx = int(np.argmax(sub_scores_t))
                if sub_scores_t[best_t_idx] > -1e5:
                    top_y_raw[x] = float(y_t_start + best_t_idx)
                else:
                    top_y_raw[x] = float(max(y_t_start, cy - 10))
            else:
                top_y_raw[x] = float(max(radius, cy - 10))

            # Tepi Bawah: Gradien Positif Kuat (+Sy)
            y_b_start = max(radius, cy - 4)
            y_b_end = min(ch - 1 - radius, cy + 25)
            if y_b_end > y_b_start:
                sub_scores_b = score_matrix_b[y_b_start:y_b_end, x]
                best_b_idx = int(np.argmax(sub_scores_b))
                if sub_scores_b[best_b_idx] > -1e5:
                    bot_y_raw[x] = float(y_b_start + best_b_idx)
                else:
                    bot_y_raw[x] = float(min(y_b_end, cy + 10))
            else:
                bot_y_raw[x] = float(min(ch - 1 - radius, cy + 10))

        # 5. Quadratic Beam Modeling (Euler-Bernoulli D=2)
        x_axis = np.arange(cw)
        trim_w = int(cw * 0.05)
        fit_range = slice(trim_w, cw - trim_w)

        poly_top = np.polyfit(x_axis[fit_range], top_y_raw[fit_range], deg=2)
        poly_bot = np.polyfit(x_axis[fit_range], bot_y_raw[fit_range], deg=2)

        top_y_fit = np.polyval(poly_top, x_axis)
        bot_y_fit = np.polyval(poly_bot, x_axis)

        # Pastikan bot_y >= top_y
        bot_y_fit = np.maximum(bot_y_fit, top_y_fit + 1.0)
        h_gap = bot_y_fit - top_y_fit

        # 6. Linear Detrending & Metrik Kelengkungan (Curvature)
        poly_linear = np.polyfit(x_axis[fit_range], bot_y_fit[fit_range], deg=1)
        detrended_curve = bot_y_fit[fit_range] - np.polyval(poly_linear, x_axis[fit_range])

        p95 = np.percentile(detrended_curve, 95)
        p5 = np.percentile(detrended_curve, 5)
        curvature_gap = float(p95 - p5)

        gap_mean = float(np.mean(h_gap[fit_range]))
        gap_max = float(np.max(h_gap[fit_range]))
        gap_min = float(np.min(h_gap[fit_range]))

        # Evaluasi Status NG / OK
        is_ng = (curvature_gap > self.delta_limit) or (gap_max > self.max_gap_limit)
        status = "NG" if is_ng else "OK"

        # 7. Render Mini Live Graph
        graph_img = self._render_gap_graph(x_axis[fit_range], detrended_curve, h_gap[fit_range], curvature_gap, gap_max, is_ng)

        elapsed_ms = (time.time() - t0) * 1000

        return {
            "status": status,
            "curvature": curvature_gap,
            "gap_mean": gap_mean,
            "gap_max": gap_max,
            "gap_min": gap_min,
            "is_transition": False,
            "is_ng": is_ng,
            "message": f"Curv: {curvature_gap:.1f} px | MaxGap: {gap_max:.1f} px",
            "elapsed_ms": elapsed_ms,
            "graph_img": graph_img,
            "top_y_fit": top_y_fit,
            "bot_y_fit": bot_y_fit,
            "x_axis": x_axis,
            "norm_seam": norm_seam,
            "M_inv": M_inv
        }

    def _render_gap_graph(self, x_arr, detrended_curv, gap_arr, curvature, max_gap, is_ng, w=320, h=120):
        """Membuat grafik visualisasi defleksi kelengkungan dan ketebalan celah."""
        graph = np.full((h, w, 3), (28, 28, 32), dtype=np.uint8)
        cv2.rectangle(graph, (0, 0), (w - 1, h - 1), (70, 70, 75), 1)

        col = (0, 0, 255) if is_ng else (0, 255, 0)
        cv2.putText(graph, "PROFIL CELAH & DEFLEKSI PLAT", (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)
        cv2.putText(graph, f"Curv: {curvature:.1f}px (Lim: {self.delta_limit:.1f}) | Gap: {max_gap:.1f}px", 
                    (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1)

        pad_x, pad_y = 15, 42
        pw, ph = w - 2 * pad_x, h - pad_y - 12
        mid_y = pad_y + ph // 2

        cv2.line(graph, (pad_x, mid_y), (pad_x + pw, mid_y), (100, 100, 100), 1, cv2.LINE_AA)

        max_scale = max(20.0, self.delta_limit * 1.3)
        pts_curve = []
        n_pts = len(detrended_curv)
        if n_pts > 1:
            for i in range(0, n_pts, max(1, n_pts // pw)):
                gx = int(pad_x + (i / float(n_pts)) * pw)
                curv_val = detrended_curv[i]
                gy = int(mid_y - (curv_val / max_scale) * (ph / 2))
                gy = max(pad_y, min(h - 8, gy))
                pts_curve.append((gx, gy))

            for j in range(len(pts_curve) - 1):
                cv2.line(graph, pts_curve[j], pts_curve[j + 1], col, 2, cv2.LINE_AA)

        return graph

    def draw_overlay_on_frame(self, frame, result):
        """Memproyeksikan kurva celah plat kembali ke koordinat frame kamera asli."""
        if result is None or result.get("is_transition", True) or "M_inv" not in result:
            return frame

        top_y_fit = result.get("top_y_fit")
        bot_y_fit = result.get("bot_y_fit")
        x_axis = result.get("x_axis")
        M_inv = result.get("M_inv")

        if top_y_fit is None or bot_y_fit is None or M_inv is None:
            return frame

        is_ng = result["is_ng"]
        col = (0, 0, 255) if is_ng else (0, 255, 0)

        # Proyeksikan garis tepi atas dan tepi bawah ke frame asli
        pts_top = np.vstack((x_axis, top_y_fit)).T.reshape(-1, 1, 2).astype(np.float32)
        pts_bot = np.vstack((x_axis, bot_y_fit)).T.reshape(-1, 1, 2).astype(np.float32)

        pts_top_frame = cv2.perspectiveTransform(pts_top, M_inv)
        pts_bot_frame = cv2.perspectiveTransform(pts_bot, M_inv)

        # Gambar kurva tepi atas (kuning cyan) dan tepi bawah (hijau / merah)
        n_p = len(pts_top_frame)
        for i in range(0, n_p - 1):
            p1_t = (int(pts_top_frame[i][0][0]), int(pts_top_frame[i][0][1]))
            p2_t = (int(pts_top_frame[i + 1][0][0]), int(pts_top_frame[i + 1][0][1]))
            cv2.line(frame, p1_t, p2_t, (0, 220, 255), 1, cv2.LINE_AA)

            p1_b = (int(pts_bot_frame[i][0][0]), int(pts_bot_frame[i][0][1]))
            p2_b = (int(pts_bot_frame[i + 1][0][0]), int(pts_bot_frame[i + 1][0][1]))
            cv2.line(frame, p1_b, p2_b, col, 2, cv2.LINE_AA)

        # Label teks di titik awal
        if n_p > 0:
            start_pt = (int(pts_bot_frame[0][0][0]), int(pts_bot_frame[0][0][1]) + 20)
            txt = f"SLAT: {result['status']} (Curv: {result['curvature']:.1f}px, MaxGap: {result['gap_max']:.1f}px)"
            cv2.putText(frame, txt, start_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

        return frame
