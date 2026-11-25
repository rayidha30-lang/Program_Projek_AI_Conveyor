"""
================================================================================
ALAT KALIBRASI INTERAKTIF REL GANDA (DUAL-RAIL INTERACTIVE CALIBRATION WIZARD)
STANDAR ACUAN EMAS: Kamera posisi 3 dimmer 2 - Reverse / WIN_20261005_09_01_08_Pro
================================================================================
Fitur:
1. Menampilkan 2 garis vertikal interaktif (Rel Kiri & Rel Kanan) yang dapat
   ditarik langsung dengan mouse pada bibir fisik konveyor dari sembarang sudut pandang (angle).
2. Live Split-Screen Preview real-time:
   [KIRI]: Citra input angle baru dengan garis yang Anda sesuaikan.
   [KANAN]: Hasil pelurusan (rectified) yang langsung dikunci ke Garis Acuan Emas:
            - Rel Kiri: X = 188 px
            - Rel Kanan: X = 1759 px
            - Sumbu Tengah: X = 960 px
            - Lebar: 1571 px
3. Auto-Snap [A]: Deteksi tepi rel otomatis menggunakan gradien Sobel-X.
4. Mode Transformasi [M]: Keystone Perspective 4-Titik atau Affine 2D.
5. Simpan [S]: Menyimpan matriks kalibrasi langsung ke top_inspection_config.calibrated.json.
================================================================================
"""

import os
import sys
import json
import math
import argparse
from datetime import datetime
import cv2
import numpy as np

# Tambahkan path ke modul sistem
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.abspath(os.path.join(SRC_DIR, "..", "configs"))

DEFAULT_CONFIG_PATH = os.path.join(CONFIG_DIR, "top_inspection_config.json")
CALIBRATED_CONFIG_PATH = os.path.join(CONFIG_DIR, "top_inspection_config.calibrated.json")
DEFAULT_STABIL_CONFIG_PATH = os.path.join(CONFIG_DIR, "top_inspection_config.default.json")

# Default Video / Foto
DEFAULT_GOLDEN_VIDEO = r"C:\Users\Victus\Pictures\Video & Foto Test Slat Rib\New\Reverse\Kamera posisi 3 dimmer 2 - Reverse.mp4"
DEFAULT_GOLDEN_IMAGE = r"C:\Users\Victus\Pictures\Video & Foto Test Slat Rib\05-10-2026\WIN_20261005_09_01_08_Pro.jpg"
DEFAULT_TEST_VIDEO = r"C:\Users\Victus\Pictures\Video & Foto Test Slat Rib\New\Reverse\Kamera posisi 4 - Reverse.mp4"


class DualRailCalibrator:
    """
    Kelas pengelola kalibrasi interaktif rel ganda konveyor.
    """

    def __init__(self, config_path=DEFAULT_CONFIG_PATH):
        self.config_path = config_path
        self.calibrated_config_path = CALIBRATED_CONFIG_PATH
        self.default_config_path = DEFAULT_STABIL_CONFIG_PATH

        # Nilai Target Acuan Emas (Golden Reference - Trapesium Perspektif Posisi 3)
        self.ref_left_rail = 188.0
        self.ref_right_rail = 1759.0
        self.ref_center_x = 960.0
        self.ref_width = 1571.0
        self.ref_tilt_deg = 0.90
        self.ref_y_top = 216.0
        self.ref_y_bot = 864.0
        self.ref_top_left_rail = 278.0
        self.ref_top_right_rail = 1630.0
        self.ref_bot_left_rail = 104.0
        self.ref_bot_right_rail = 1808.0

        self.load_golden_reference()

        # 4 Titik Kendali Rel pada Angle Saat Ini (Koordinat Citra Asli 1080p)
        # Point 0: Left Top, Point 1: Left Bottom
        # Point 2: Right Top, Point 3: Right Bottom
        self.y_top = self.ref_y_top
        self.y_bot = self.ref_y_bot
        self.pts = [
            np.array([self.ref_top_left_rail, self.y_top]),    # 0: Left Top
            np.array([self.ref_bot_left_rail, self.y_bot]),    # 1: Left Bottom
            np.array([self.ref_top_right_rail, self.y_top]),   # 2: Right Top
            np.array([self.ref_bot_right_rail, self.y_bot]),   # 3: Right Bottom
        ]

        # State Mouse Interaction
        self.selected_point_idx = None
        self.hover_point_idx = None
        self.drag_line_idx = None  # 0: Left line, 1: Right line
        self.drag_start_mouse = None
        self.drag_start_pts = None

        # Mode: 'PERSPECTIVE' (4-Point Keystone) atau 'AFFINE' (2D Rigid/Scale)
        self.mode = "PERSPECTIVE"
        self.is_saved = False
        self.save_feedback_timer = 0
        self.last_matrix = None
        self.calib_frame_w = 1920
        self.calib_frame_h = 1080
        self.metrology = {}

    def load_golden_reference(self):
        """Memuat acuan emas dari konfigurasi jika tersedia."""
        target_path = self.default_config_path if os.path.exists(self.default_config_path) else self.config_path
        if os.path.exists(target_path):
            try:
                with open(target_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                cp = cfg.get("calibration_profile", {})
                gref = cp.get("golden_reference", {})
                self.ref_left_rail = float(gref.get("left_rail_px", cp.get("left_rail_px", self.ref_left_rail)))
                self.ref_right_rail = float(gref.get("right_rail_px", cp.get("right_rail_px", self.ref_right_rail)))
                self.ref_center_x = float(gref.get("target_cx", cp.get("conveyor_center_x", self.ref_center_x)))
                self.ref_tilt_deg = float(gref.get("target_tilt_deg", cp.get("bracket_tilt_deg", self.ref_tilt_deg)))
                self.ref_width = float(gref.get("conveyor_width_px", self.ref_right_rail - self.ref_left_rail))
                self.ref_y_top = float(gref.get("top_y_px", self.ref_y_top))
                self.ref_y_bot = float(gref.get("bot_y_px", self.ref_y_bot))
                self.ref_top_left_rail = float(gref.get("top_left_rail_px", self.ref_top_left_rail))
                self.ref_top_right_rail = float(gref.get("top_right_rail_px", self.ref_top_right_rail))
                self.ref_bot_left_rail = float(gref.get("bot_left_rail_px", self.ref_bot_left_rail))
                self.ref_bot_right_rail = float(gref.get("bot_right_rail_px", self.ref_bot_right_rail))
            except Exception as e:
                print(f"[!] Warning load golden reference: {e}")

        # Muat mode preferensi terakhir jika ada di config aktif
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f_act:
                    cfg_act = json.load(f_act)
                method = cfg_act.get("calibration_profile", {}).get("calibration_method", "")
                if "AFFINE" in method.upper():
                    self.mode = "AFFINE"
                elif "PERSPECTIVE" in method.upper():
                    self.mode = "PERSPECTIVE"
            except Exception:
                pass

    def get_golden_rail_x(self, side, y_val, w_f=1920, h_f=1080):
        """Interpolasi linier posisi X rel kiri/kanan Acuan Emas Posisi 3 pada tinggi y_val."""
        sx = float(w_f) / 1920.0
        sy = float(h_f) / 1080.0
        y_top_s = self.ref_y_top * sy
        y_bot_s = self.ref_y_bot * sy
        denom = max(1e-5, y_bot_s - y_top_s)
        alpha = (float(y_val) - y_top_s) / denom
        if side == "left":
            x_top = self.ref_top_left_rail * sx
            x_bot = self.ref_bot_left_rail * sx
        else:
            x_top = self.ref_top_right_rail * sx
            x_bot = self.ref_bot_right_rail * sx
        return float(x_top + alpha * (x_bot - x_top))

    def get_golden_trapezoid_pts(self, w_f=1920, h_f=1080):
        """Mengembalikan 4 titik sudut trapesium Acuan Emas [Top-Left, Top-Right, Bottom-Right, Bottom-Left]."""
        sy = float(h_f) / 1080.0
        y_top_s = self.ref_y_top * sy
        y_bot_s = self.ref_y_bot * sy
        return np.array([
            [self.get_golden_rail_x("left", y_top_s, w_f, h_f), y_top_s],
            [self.get_golden_rail_x("right", y_top_s, w_f, h_f), y_top_s],
            [self.get_golden_rail_x("right", y_bot_s, w_f, h_f), y_bot_s],
            [self.get_golden_rail_x("left", y_bot_s, w_f, h_f), y_bot_s],
        ], dtype=np.float32)

    @staticmethod
    def _find_top_rail_x(prof_top, primary_bounds, fallback_bounds, min_peak=12.0):
        """Cari puncak Sobel-X bibir rel dalam pada rentang utama dengan fallback rentang lebar."""
        p_min, p_max = primary_bounds
        idx_rel = int(np.argmax(prof_top[p_min:p_max]))
        peak_val = float(prof_top[p_min + idx_rel])
        if peak_val >= min_peak:
            return int(p_min + idx_rel)
        f_min, f_max = fallback_bounds
        return int(f_min + np.argmax(prof_top[f_min:f_max]))

    @staticmethod
    def _track_bottom_rail_x(
        sobx,
        x_top_val,
        y_top,
        y_bot,
        sample_ys,
        nom_dx,
        dx_min,
        dx_max,
        x_min_bound,
        x_max_bound,
        outward_sign,
        min_abs_dx,
    ):
        """Lacak garis tepi rel dari titik atas ke bawah dengan batasan sudut perspektif keluar."""
        best_score = -1e9
        best_xb = int(np.clip(x_top_val + nom_dx, x_min_bound, x_max_bound))
        span_y = max(1e-5, y_bot - y_top)
        h_f, w_f = sobx.shape[:2]
        for dx in range(int(dx_min), int(dx_max) + 1):
            if dx * outward_sign <= int(min_abs_dx):
                continue
            xb_cand = int(x_top_val + dx)
            if xb_cand < x_min_bound or xb_cand > x_max_bound:
                continue
            vals = []
            for sy in sample_ys:
                alpha = (sy - y_top) / span_y
                cx = int(round(x_top_val + alpha * dx))
                y0, y1 = max(0, int(sy) - 10), min(h_f, int(sy) + 10)
                x0, x1 = max(0, cx - 4), min(w_f, cx + 5)
                vals.append(float(np.mean(sobx[y0:y1, x0:x1])))
            score = float(np.median(vals)) - 0.15 * abs(dx - nom_dx)
            if score > best_score:
                best_score = score
                best_xb = xb_cand
        return best_xb

    def auto_snap_rails(self, frame_bgr, rib_boxes=None):
        """
        Mendeteksi posisi rel kiri & kanan konveyor secara vektorisasi berbasis pencarian garis miring
        trapesium perspektif, menolak pilar dinding vertikal latar (dx ~ 0) dan distraktor di atas sabuk,
        serta memanfaatkan jangkar ujung kotak rib YOLO bila tersedia.
        """
        if frame_bgr is None:
            return
        h_f, w_f = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        sobx = np.abs(cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3))
        sobx_s = cv2.boxFilter(sobx, -1, (7, 15), normalize=True)

        # Deteksi kontinuitas vertikal murni (pilar dinding / tiang rangka stasioner pada Y=16%..46%)
        strip_hi = np.mean(sobx_s[int(0.16 * h_f):max(int(0.16 * h_f) + 1, int(0.28 * h_f)), :], axis=0)
        strip_mid = np.mean(sobx_s[int(0.32 * h_f):max(int(0.32 * h_f) + 1, int(0.46 * h_f)), :], axis=0)
        vert_continuity = np.minimum(strip_hi, strip_mid)

        self.y_top = 0.20 * h_f
        self.y_bot = 0.80 * h_f
        span_y = max(1.0, self.y_bot - self.y_top)
        sample_ys = np.linspace(0.20 * h_f, 0.78 * h_f, 13)
        alphas = ((sample_ys - self.y_top) / span_y)[:, None, None]
        sy_idx = np.clip(np.round(sample_ys).astype(np.int32), 0, h_f - 1)[:, None, None]

        sx = float(w_f) / 1920.0
        min_abs_dx = max(12, int(round(0.055 * w_f)))
        max_abs_dx = max(min_abs_dx + 20, int(round(0.22 * w_f)))
        step_x = max(2, int(round(w_f / 480.0)))
        nom_abs_dx = 0.091 * w_f

        # Ekstrak referensi rib opsional untuk validasi jarak rel kiri/kanan terhadap ujung bilah
        rib_anchors = []
        if rib_boxes:
            for rb in rib_boxes:
                box = rb.get("box") if isinstance(rb, dict) else rb
                if box is not None and len(box) >= 4:
                    bx1, by1, bx2, by2 = [float(v) for v in box[:4]]
                    if (bx2 - bx1) >= 0.24 * w_f:
                        rib_anchors.append((bx1, bx2, 0.5 * (by1 + by2)))

        def _search_slanted_rail(is_left):
            if is_left:
                xts = np.arange(int(0.09 * w_f), int(0.34 * w_f) + 1, step_x, dtype=np.float32)
                dxs = np.arange(-max_abs_dx, -min_abs_dx + 1, step_x, dtype=np.float32)
            else:
                xts = np.arange(int(0.66 * w_f), int(0.91 * w_f) + 1, step_x, dtype=np.float32)
                dxs = np.arange(min_abs_dx, max_abs_dx + 1, step_x, dtype=np.float32)

            xt_grid = xts[None, :, None]
            dx_grid = dxs[None, None, :]
            xb_grid = xt_grid + dx_grid
            valid = (xb_grid >= 0.015 * w_f) & (xb_grid <= 0.985 * w_f)

            if rib_anchors:
                min_gap = 0.015 * w_f
                max_gap = 0.145 * w_f
                for r_x1, r_x2, r_cy in rib_anchors:
                    a_r = (r_cy - self.y_top) / span_y
                    x_at_r = xt_grid + a_r * dx_grid
                    gap = (r_x1 - x_at_r) if is_left else (x_at_r - r_x2)
                    # Hanya aktifkan filter bila setidaknya ada kandidat valid dalam envelope rel fisik
                    anchor_ok = (gap >= min_gap) & (gap <= max_gap)
                    if np.any(valid & anchor_ok):
                        valid = valid & anchor_ok

            cx = np.round(xt_grid + alphas * dx_grid).astype(np.int32)
            cx_c = np.clip(cx, 0, w_f - 1)
            vals = sobx_s[sy_idx, cx_c]

            med = np.median(vals, axis=0)
            p25 = np.percentile(vals, 25, axis=0)
            p75 = np.percentile(vals, 75, axis=0)
            slope_pen = 0.08 * np.abs(np.abs(dx_grid[0]) - nom_abs_dx) / max(1.0, float(step_x))
            xt_idx = np.clip(np.round(xts).astype(np.int32), 0, w_f - 1)
            pillar_pen = 0.35 * vert_continuity[xt_idx][:, None]
            score = 0.50 * med + 0.30 * p25 + 0.25 * p75 - slope_pen - pillar_pen
            score = np.where(valid[0], score, -1e9)
            flat_idx = int(np.argmax(score))
            ix, id_ = divmod(flat_idx, len(dxs))
            xt_b = float(xts[ix])
            xb_b = float(xt_b + dxs[id_])
            return int(round(xt_b)), int(round(xb_b))

        lt_x, lb_x = _search_slanted_rail(True)
        rt_x, rb_x = _search_slanted_rail(False)

        self.pts[0] = np.array([float(lt_x), self.y_top])
        self.pts[1] = np.array([float(lb_x), self.y_bot])
        self.pts[2] = np.array([float(rt_x), self.y_top])
        self.pts[3] = np.array([float(rb_x), self.y_bot])
        self.is_saved = False
        print(f"[*] Auto-Snap Rel Berhasil: Kiri=[{lt_x}, {lb_x}], Kanan=[{rt_x}, {rb_x}]")

    def compute_rectification(self, w_f=1920, h_f=1080):
        """
        Menghitung matriks rektifikasi (Perspective 3x3 atau Affine 2x3)
        berdasarkan 2 garis rel saat ini terhadap Trapesium Acuan Emas Posisi 3.
        """
        p_lt, p_lb, p_rt, p_rb = self.pts

        # Hitung kemiringan masing-masing rel
        theta_l = np.degrees(np.arctan2(p_lb[0] - p_lt[0], p_lb[1] - p_lt[1]))
        theta_r = np.degrees(np.arctan2(p_rb[0] - p_rt[0], p_rb[1] - p_rt[1]))
        theta_mean = float((theta_l + theta_r) / 2.0)

        mid_y = h_f / 2.0
        den_l = max(1e-5, p_lb[1] - p_lt[1])
        den_r = max(1e-5, p_rb[1] - p_rt[1])
        x_l_mid = float(p_lt[0] + (mid_y - p_lt[1]) * (p_lb[0] - p_lt[0]) / den_l)
        x_r_mid = float(p_rt[0] + (mid_y - p_rt[1]) * (p_rb[0] - p_rt[0]) / den_r)

        cur_width = max(10.0, x_r_mid - x_l_mid)
        cur_cx = (x_l_mid + x_r_mid) / 2.0

        ref_left_mid = self.get_golden_rail_x("left", mid_y, w_f, h_f)
        ref_right_mid = self.get_golden_rail_x("right", mid_y, w_f, h_f)
        ref_width_mid = max(10.0, ref_right_mid - ref_left_mid)
        ref_cx_mid = (ref_left_mid + ref_right_mid) / 2.0

        scale_x = ref_width_mid / cur_width
        delta_x = ref_cx_mid - cur_cx
        cur_top_w = max(10.0, float(p_rt[0] - p_lt[0]))
        ref_top_w = max(
            10.0,
            self.get_golden_rail_x("right", p_rt[1], w_f, h_f)
            - self.get_golden_rail_x("left", p_lt[1], w_f, h_f),
        )
        top_width_ratio = cur_top_w / ref_top_w
        pitch_width_ratio = min(cur_width / ref_width_mid, top_width_ratio)
        if pitch_width_ratio < 0.90:
            rec_rib_y_top_frac = round(max(0.18, min(0.28, 0.28 * pitch_width_ratio)), 3)
            rec_prime_min_y = int(round(max(225, rec_rib_y_top_frac * 1080.0 + 18)))
            rec_min_span_px = float(round(max(440.0, min(680.0, 0.56 * cur_top_w)), 1))
        else:
            rec_rib_y_top_frac = 0.28
            rec_prime_min_y = 315
            rec_min_span_px = 680.0

        self.calib_frame_w = int(w_f)
        self.calib_frame_h = int(h_f)
        self.metrology = {
            "calib_frame_w": int(w_f),
            "calib_frame_h": int(h_f),
            "theta_l": theta_l,
            "theta_r": theta_r,
            "measured_tilt_deg": theta_mean,
            "angle_correction_deg": -theta_mean,
            "cur_width_px": cur_width,
            "cur_center_x": cur_cx,
            "scale_x": scale_x,
            "lateral_offset_px": delta_x,
            "ref_width_px": ref_width_mid,
            "ref_center_x": ref_cx_mid,
            "ref_left_px": ref_left_mid,
            "ref_right_px": ref_right_mid,
            "top_width_ratio": round(top_width_ratio, 3),
            "pitch_width_ratio": round(pitch_width_ratio, 3),
            "recommended_rib_y_top_frac": rec_rib_y_top_frac,
            "recommended_prime_min_y": rec_prime_min_y,
            "recommended_min_span_px": rec_min_span_px,
            "inner_rail_trapezoid": {
                "y_top": float(p_lt[1]),
                "y_bot": float(p_lb[1]),
                "lt_x": float(p_lt[0]),
                "lb_x": float(p_lb[0]),
                "rt_x": float(p_rt[0]),
                "rb_x": float(p_rb[0]),
            },
        }

        src_pts = np.array([p_lt, p_rt, p_rb, p_lb], dtype=np.float32)
        dst_pts = np.array([
            [self.get_golden_rail_x("left", p_lt[1], w_f, h_f), p_lt[1]],
            [self.get_golden_rail_x("right", p_rt[1], w_f, h_f), p_rt[1]],
            [self.get_golden_rail_x("right", p_rb[1], w_f, h_f), p_rb[1]],
            [self.get_golden_rail_x("left", p_lb[1], w_f, h_f), p_lb[1]],
        ], dtype=np.float32)

        if self.mode == "PERSPECTIVE":
            M_persp = cv2.getPerspectiveTransform(src_pts, dst_pts)
            self.last_matrix = M_persp
            return M_persp, "PERSPECTIVE"
        else:
            # Mode Affine 2D: 4-point least-squares affine fit ke Trapesium Acuan Emas
            X_aug = np.hstack([src_pts, np.ones((4, 1), dtype=np.float32)])
            M_aff = np.linalg.lstsq(X_aug, dst_pts, rcond=None)[0].T.astype(np.float32)
            self.last_matrix = M_aff
            return M_aff, "AFFINE"

    def apply_warp(self, frame_bgr):
        """Menerapkan transformasi rektifikasi pada citra input."""
        if frame_bgr is None:
            return None
        h_f, w_f = frame_bgr.shape[:2]
        M, m_type = self.compute_rectification(w_f, h_f)

        if m_type == "PERSPECTIVE":
            warped = cv2.warpPerspective(
                frame_bgr, M, (w_f, h_f),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )
        else:
            warped = cv2.warpAffine(
                frame_bgr, M, (w_f, h_f),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )
        return warped

    def save_calibration_to_config(self):
        """Menyimpan matriks hasil kalibrasi ke file JSON config."""
        if self.last_matrix is None:
            return False, "Matriks kalibrasi belum dihitung."

        cfg_data = {}
        # Prioritaskan load default config agar batas QC tetap terjaga
        if os.path.exists(self.default_config_path):
            try:
                with open(self.default_config_path, "r", encoding="utf-8") as f:
                    cfg_data = json.load(f)
            except Exception:
                pass
        if not cfg_data and os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    cfg_data = json.load(f)
            except Exception:
                pass

        cfg_data["profile_name"] = "HASIL_KALIBRASI"
        cfg_data["description"] = f"Profil Hasil Kalibrasi Interaktif Rel Ganda ({self.mode})"
        cfg_data["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cfg_data["rectification_matrix"] = self.last_matrix.tolist()

        if "thresholds" not in cfg_data:
            cfg_data["thresholds"] = {}
        cfg_data["thresholds"]["bracket_tilt_deg"] = round(self.metrology.get("measured_tilt_deg", 0.0), 3)

        if "calibration_profile" not in cfg_data:
            cfg_data["calibration_profile"] = {}

        cfg_data["calibration_profile"].update({
            "calibrated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "calibration_method": f"INTERACTIVE_DUAL_RAILS_{self.mode}",
            "calib_frame_w": int(self.metrology.get("calib_frame_w", getattr(self, "calib_frame_w", 1920))),
            "calib_frame_h": int(self.metrology.get("calib_frame_h", getattr(self, "calib_frame_h", 1080))),
            "golden_reference": {
                "source_video": "Kamera posisi 3 dimmer 2 - Reverse.mp4",
                "source_image": "WIN_20261005_09_01_08_Pro.jpg",
                "target_tilt_deg": self.ref_tilt_deg,
                "target_cx": self.ref_center_x,
                "left_rail_px": int(self.ref_left_rail),
                "right_rail_px": int(self.ref_right_rail),
                "conveyor_width_px": int(self.ref_width),
                "top_y_px": int(self.ref_y_top),
                "bot_y_px": int(self.ref_y_bot),
                "top_left_rail_px": int(self.ref_top_left_rail),
                "top_right_rail_px": int(self.ref_top_right_rail),
                "bot_left_rail_px": int(self.ref_bot_left_rail),
                "bot_right_rail_px": int(self.ref_bot_right_rail),
            },
            "measured_tilt_deg": round(self.metrology.get("measured_tilt_deg", 0.0), 3),
            "target_tilt_deg": self.ref_tilt_deg,
            "angle_correction_deg": round(self.metrology.get("angle_correction_deg", 0.0), 3),
            "conveyor_center_x": round(self.metrology.get("cur_center_x", 960.0), 1),
            "lateral_offset_px": round(self.metrology.get("lateral_offset_px", 0.0), 1),
            "measured_width_px": round(self.metrology.get("cur_width_px", 1571.0), 1),
            "scale_x": round(self.metrology.get("scale_x", 1.0), 3),
            "pitch_width_ratio": self.metrology.get("pitch_width_ratio", 1.0),
            "recommended_rib_y_top_frac": self.metrology.get("recommended_rib_y_top_frac", 0.28),
            "recommended_prime_min_y": self.metrology.get("recommended_prime_min_y", 315),
            "recommended_min_span_px": self.metrology.get("recommended_min_span_px", 680.0),
            "left_rail_pts": [self.pts[0].tolist(), self.pts[1].tolist()],
            "right_rail_pts": [self.pts[2].tolist(), self.pts[3].tolist()]
        })

        # Sesuaikan batas atas zona rib saat rel menyempit (kamera agak mendongak ke atas)
        rec_top_y = float(self.metrology.get("recommended_rib_y_top_frac", 0.28))
        rec_prime_min = int(self.metrology.get("recommended_prime_min_y", 315))
        if rec_top_y < 0.28 and isinstance(cfg_data.get("rib_polygon"), list) and len(cfg_data["rib_polygon"]) >= 4:
            updated_poly = [list(pt) for pt in cfg_data["rib_polygon"]]
            updated_poly[0][1] = min(float(updated_poly[0][1]), rec_top_y)
            updated_poly[1][1] = min(float(updated_poly[1][1]), rec_top_y)
            cfg_data["rib_polygon"] = updated_poly
            if "rib_prime_min_y" in cfg_data["thresholds"]:
                cfg_data["thresholds"]["rib_prime_min_y"] = min(
                    int(cfg_data["thresholds"]["rib_prime_min_y"]), rec_prime_min
                )
            else:
                cfg_data["thresholds"]["rib_prime_min_y"] = rec_prime_min

        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(cfg_data, f, indent=2)
            with open(self.calibrated_config_path, "w", encoding="utf-8") as f:
                json.dump(cfg_data, f, indent=2)

            self.is_saved = True
            self.save_feedback_timer = 90
            msg = f"Kalibrasi Berhasil Disimpan ke {os.path.basename(self.calibrated_config_path)}!"
            print(f"\n[*] [SUCCESS] {msg}")
            return True, msg
        except Exception as e:
            err = f"Gagal menyimpan config: {e}"
            print(f"[!] {err}")
            return False, err


def run_interactive_dual_rail_calibration(source_path=None, config_path=DEFAULT_CONFIG_PATH):
    """
    Fungsi utama menjalankan GUI Wizard Kalibrasi Interaktif Rel Ganda.
    """
    calibrator = DualRailCalibrator(config_path=config_path)

    # Tentukan sumber input
    if source_path is None or not os.path.exists(source_path):
        source_path = DEFAULT_TEST_VIDEO if os.path.exists(DEFAULT_TEST_VIDEO) else DEFAULT_GOLDEN_VIDEO

    is_image = False
    cap = None
    if source_path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        is_image = True
        frame_raw = cv2.imread(source_path)
    else:
        cap = cv2.VideoCapture(source_path)
        ret, frame_raw = cap.read()
        if not ret:
            print(f"[!] Gagal membaca video: {source_path}")
            return

    h_f, w_f = frame_raw.shape[:2]
    calibrator.auto_snap_rails(frame_raw)

    win_name = "WIZARD KALIBRASI INTERAKTIF REL GANDA (Tarik Garis Objek Rel Konveyor)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1600, 850)

    # Variabel tampilan split screen
    # Kanvas kiri: 720 x 480, Kanvas kanan: 720 x 480 (Total lebar 1480 x 600 + HUD)
    vw, vh = 720, 480
    pad_x, pad_y = 20, 60

    # Skala koordinat dari citra asli (w_f, h_f) ke viewport kiri
    scale_vx = float(vw) / float(w_f)
    scale_vy = float(vh) / float(h_f)

    # Mouse Callback Handler
    def on_mouse(event, x, y, flags, param):
        nonlocal vw, vh, pad_x, pad_y, scale_vx, scale_vy

        # Cek apakah mouse berada di dalam viewport kiri
        in_left_vp = (pad_x <= x <= pad_x + vw) and (pad_y <= y <= pad_y + vh)
        mx_orig = (x - pad_x) / scale_vx
        my_orig = (y - pad_y) / scale_vy

        # Hitung jarak mouse ke 4 handle titik kendali
        hit_radius = 16.0  # pixel tampilan
        hover_idx = None
        for i, pt in enumerate(calibrator.pts):
            px_disp = pad_x + pt[0] * scale_vx
            py_disp = pad_y + pt[1] * scale_vy
            dist = math.hypot(x - px_disp, y - py_disp)
            if dist <= hit_radius:
                hover_idx = i
                break

        calibrator.hover_point_idx = hover_idx

        if event == cv2.EVENT_LBUTTONDOWN:
            if hover_idx is not None:
                calibrator.selected_point_idx = hover_idx
                calibrator.drag_start_mouse = (mx_orig, my_orig)
                calibrator.drag_start_pts = [p.copy() for p in calibrator.pts]
            elif in_left_vp:
                # Cek apakah klik dekat garis rel kiri atau rel kanan untuk drag garis utuh
                # Garis kiri (idx 0 -> 1), Garis kanan (idx 2 -> 3)
                p_lt, p_lb, p_rt, p_rb = calibrator.pts
                mid_l = (p_lt + p_lb) / 2.0
                mid_r = (p_rt + p_rb) / 2.0
                dist_l = abs(mx_orig - mid_l[0])
                dist_r = abs(mx_orig - mid_r[0])
                if dist_l < 35.0:
                    calibrator.drag_line_idx = 0  # Drag garis kiri
                    calibrator.drag_start_mouse = (mx_orig, my_orig)
                    calibrator.drag_start_pts = [p.copy() for p in calibrator.pts]
                elif dist_r < 35.0:
                    calibrator.drag_line_idx = 1  # Drag garis kanan
                    calibrator.drag_start_mouse = (mx_orig, my_orig)
                    calibrator.drag_start_pts = [p.copy() for p in calibrator.pts]

        elif event == cv2.EVENT_MOUSEMOVE:
            if calibrator.selected_point_idx is not None:
                idx = calibrator.selected_point_idx
                # Batasi koordinat dalam batas citra
                new_x = float(np.clip(mx_orig, 10.0, w_f - 10.0))
                new_y = float(np.clip(my_orig, 20.0, h_f - 20.0))
                calibrator.pts[idx][0] = new_x
                calibrator.pts[idx][1] = new_y
                calibrator.is_saved = False
            elif calibrator.drag_line_idx is not None and calibrator.drag_start_mouse is not None:
                dx = mx_orig - calibrator.drag_start_mouse[0]
                if calibrator.drag_line_idx == 0:
                    calibrator.pts[0][0] = float(np.clip(calibrator.drag_start_pts[0][0] + dx, 10.0, w_f - 10.0))
                    calibrator.pts[1][0] = float(np.clip(calibrator.drag_start_pts[1][0] + dx, 10.0, w_f - 10.0))
                elif calibrator.drag_line_idx == 1:
                    calibrator.pts[2][0] = float(np.clip(calibrator.drag_start_pts[2][0] + dx, 10.0, w_f - 10.0))
                    calibrator.pts[3][0] = float(np.clip(calibrator.drag_start_pts[3][0] + dx, 10.0, w_f - 10.0))
                calibrator.is_saved = False

        elif event == cv2.EVENT_LBUTTONUP:
            calibrator.selected_point_idx = None
            calibrator.drag_line_idx = None
            calibrator.drag_start_mouse = None
            calibrator.drag_start_pts = None

    cv2.setMouseCallback(win_name, on_mouse)

    is_paused = True
    frame_idx = 0
    save_msg = ""

    print("\n" + "=" * 75)
    print("  WIZARD KALIBRASI INTERAKTIF REL GANDA AKTIF!")
    print("  [DRAG MOUSE]: Tarik titik/garis pada bibir fisik rel kiri dan kanan konveyor.")
    print("  [A]         : Auto-Snap deteksi tepi rel otomatis.")
    print("  [M]         : Toggle Mode Transformasi (PERSPECTIVE 4-Titik <-> AFFINE 2D).")
    print("  [SPACE]     : Pause / Play video.")
    print("  [N] / [P]   : Maju / Mundur 1 frame.")
    print("  [S]         : SIMPAN matriks kalibrasi ke file konfigurasi.")
    print("  [Q] / [ESC] : Selesai dan keluar.")
    print("=" * 75 + "\n")

    while True:
        if not is_paused and cap is not None:
            ret, frame_read = cap.read()
            if ret and frame_read is not None:
                frame_raw = frame_read
                frame_idx += 1
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # 1. Siapkan Kanvas UI (Lebar: 1500, Tinggi: 750)
        canvas = np.zeros((750, 1500, 3), dtype=np.uint8)
        canvas[:] = (20, 22, 28)

        # 2. Viewport Kiri: Citra Input Asli + Garis Rel yang Ditarik
        view_left = cv2.resize(frame_raw, (vw, vh))

        # Render Garis Rel Kiri (0 -> 1)
        p_lt_d = (int(round(calibrator.pts[0][0] * scale_vx)), int(round(calibrator.pts[0][1] * scale_vy)))
        p_lb_d = (int(round(calibrator.pts[1][0] * scale_vx)), int(round(calibrator.pts[1][1] * scale_vy)))
        cv2.line(view_left, p_lt_d, p_lb_d, (0, 235, 255), 2, cv2.LINE_AA)

        # Render Garis Rel Kanan (2 -> 3)
        p_rt_d = (int(round(calibrator.pts[2][0] * scale_vx)), int(round(calibrator.pts[2][1] * scale_vy)))
        p_rb_d = (int(round(calibrator.pts[3][0] * scale_vx)), int(round(calibrator.pts[3][1] * scale_vy)))
        cv2.line(view_left, p_rt_d, p_rb_d, (0, 235, 255), 2, cv2.LINE_AA)

        # Render Garis Tengah (Midline)
        mid_t_d = ((p_lt_d[0] + p_rt_d[0]) // 2, (p_lt_d[1] + p_rt_d[1]) // 2)
        mid_b_d = ((p_lb_d[0] + p_rb_d[0]) // 2, (p_lb_d[1] + p_rb_d[1]) // 2)
        cv2.line(view_left, mid_t_d, mid_b_d, (255, 200, 0), 1, cv2.LINE_AA)

        # Render 4 Titik Kendali (Handles)
        handle_labels = ["REL KIRI (TOP)", "REL KIRI (BOT)", "REL KANAN (TOP)", "REL KANAN (BOT)"]
        for i, pt in enumerate(calibrator.pts):
            px_d = int(round(pt[0] * scale_vx))
            py_d = int(round(pt[1] * scale_vy))

            is_hover = (calibrator.hover_point_idx == i)
            is_active = (calibrator.selected_point_idx == i)
            h_col = (0, 255, 0) if (is_hover or is_active) else (0, 220, 255)
            h_rad = 8 if (is_hover or is_active) else 6

            cv2.circle(view_left, (px_d, py_d), h_rad, h_col, -1, cv2.LINE_AA)
            cv2.circle(view_left, (px_d, py_d), h_rad + 3, (255, 255, 255), 1, cv2.LINE_AA)

        # Tempel Viewport Kiri ke Kanvas
        canvas[pad_y:pad_y + vh, pad_x:pad_x + vw] = view_left
        cv2.rectangle(canvas, (pad_x, pad_y), (pad_x + vw, pad_y + vh), (80, 85, 95), 2)

        # Label Header Viewport Kiri
        cv2.putText(canvas, "[INPUT ANGLE] Tarik Garis pada Bibir Rel Konveyor", (pad_x + 10, pad_y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 235, 255), 2, cv2.LINE_AA)

        # 3. Viewport Kanan: Hasil Pelurusan (Rectified Preview) Real-Time
        warped_frame = calibrator.apply_warp(frame_raw)
        view_right = cv2.resize(warped_frame, (vw, vh))

        # Skala Trapesium Acuan Emas Posisi 3 ke Viewport Kanan
        ref_lx_top_d = int(round(calibrator.get_golden_rail_x("left", 0.0, w_f, h_f) * scale_vx))
        ref_lx_bot_d = int(round(calibrator.get_golden_rail_x("left", float(h_f), w_f, h_f) * scale_vx))
        ref_rx_top_d = int(round(calibrator.get_golden_rail_x("right", 0.0, w_f, h_f) * scale_vx))
        ref_rx_bot_d = int(round(calibrator.get_golden_rail_x("right", float(h_f), w_f, h_f) * scale_vx))
        ref_cx_d = int(round(calibrator.ref_center_x * scale_vx))

        # Overlay Garis Trapesium Acuan Emas pada Viewport Kanan
        # Rel Kiri Emas (Neon Green)
        cv2.line(view_right, (ref_lx_top_d, 0), (ref_lx_bot_d, vh), (0, 255, 100), 2, cv2.LINE_AA)
        cv2.putText(view_right, f"ACUAN KIRI ({int(calibrator.ref_top_left_rail)}->{int(calibrator.ref_bot_left_rail)}px)", (max(4, ref_lx_bot_d + 6), vh - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 100), 1, cv2.LINE_AA)

        # Rel Kanan Emas (Neon Green)
        cv2.line(view_right, (ref_rx_top_d, 0), (ref_rx_bot_d, vh), (0, 255, 100), 2, cv2.LINE_AA)
        cv2.putText(view_right, f"ACUAN KANAN ({int(calibrator.ref_top_right_rail)}->{int(calibrator.ref_bot_right_rail)}px)", (max(10, ref_rx_bot_d - 165), vh - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 100), 1, cv2.LINE_AA)

        # Sumbu Simetri Tengah Emas (Cyan)
        for y_dash in range(0, vh, 20):
            cv2.line(view_right, (ref_cx_d, y_dash), (ref_cx_d, min(vh, y_dash + 10)), (255, 200, 0), 1, cv2.LINE_AA)
        cv2.putText(view_right, f"SUMBU TENGAH ({int(calibrator.ref_center_x)}px)", (ref_cx_d - 65, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 200, 0), 1, cv2.LINE_AA)

        # Tempel Viewport Kanan ke Kanvas
        right_pad_x = pad_x + vw + 20
        canvas[pad_y:pad_y + vh, right_pad_x:right_pad_x + vw] = view_right
        cv2.rectangle(canvas, (right_pad_x, pad_y), (right_pad_x + vw, pad_y + vh), (0, 200, 100), 2)

        # Label Header Viewport Kanan
        cv2.putText(canvas, "[RECTIFIED] Diluruskan Real-Time ke Standar Acuan Emas", (right_pad_x + 10, pad_y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 100), 2, cv2.LINE_AA)

        # 4. Panel Informasi Metrologi di Bawah Viewport (Y: 560 s.d. 660)
        panel_y = pad_y + vh + 15
        cv2.rectangle(canvas, (pad_x, panel_y), (pad_x + vw * 2 + 20, panel_y + 80), (28, 32, 40), -1)
        cv2.rectangle(canvas, (pad_x, panel_y), (pad_x + vw * 2 + 20, panel_y + 80), (55, 60, 75), 1)

        m = calibrator.metrology
        tilt_val = m.get("measured_tilt_deg", 0.0)
        w_val = m.get("cur_width_px", 1571.0)
        scale_val = m.get("scale_x", 1.0)
        offset_val = m.get("lateral_offset_px", 0.0)

        # Baris 1: Status Metrologi Real-Time
        col_tilt = (0, 255, 100) if abs(tilt_val) < 1.0 else (0, 220, 255)
        txt_row1 = (
            f"KEMIRINGAN: {tilt_val:+.2f} deg (Koreksi {-tilt_val:+.2f} deg)  |  "
            f"LEBAR REL: {w_val:.1f}px (Target: {calibrator.ref_width:.0f}px | Skala: {scale_val:.3f}x)  |  "
            f"GESER PUSAT X: {offset_val:+.1f}px"
        )
        cv2.putText(canvas, txt_row1, (pad_x + 15, panel_y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, col_tilt, 1, cv2.LINE_AA)

        # Baris 2: Mode Transformasi & Feedback Penyimpanan
        badge_mode_col = (0, 200, 255) if calibrator.mode == "PERSPECTIVE" else (255, 180, 0)
        txt_row2 = f"MODE: [{calibrator.mode} 4-TITIK]  |  Target Acuan: Kamera posisi 3 dimmer 2  |  Status: "
        if calibrator.save_feedback_timer > 0:
            calibrator.save_feedback_timer -= 1
            txt_status = "BERHASIL DISIMPAN KE CONFIG! (PROFIL AKTIF)"
            col_status = (0, 255, 100)
        else:
            txt_status = "SIAP DISIMPAN [S]" if not calibrator.is_saved else "TERKALIBRASI"
            col_status = (200, 200, 200)

        cv2.putText(canvas, txt_row2, (pad_x + 15, panel_y + 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(canvas, txt_status, (pad_x + 480, panel_y + 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, col_status, 2, cv2.LINE_AA)

        # 5. Footer Panduan Tombol Keyboard (Y: 665 s.d. 735)
        foot_y = panel_y + 92
        cv2.rectangle(canvas, (pad_x, foot_y), (pad_x + vw * 2 + 20, foot_y + 52), (18, 20, 25), -1)
        cv2.rectangle(canvas, (pad_x, foot_y), (pad_x + vw * 2 + 20, foot_y + 52), (40, 45, 55), 1)

        txt_keys1 = "[DRAG MOUSE]: Tarik Handle Titik Rel  |  [A]: Auto-Snap Sobel  |  [M]: Ganti Mode (Perspective/Affine)  |  [S]: SIMPAN HASIL"
        txt_keys2 = "[SPACE]: Pause/Play  |  [N]/[P]: Step Frame  |  [O]/[F]: Buka Video/Foto Baru  |  [Q]/[ESC]: Selesai & Keluar"
        cv2.putText(canvas, txt_keys1, (pad_x + 15, foot_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 230, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, txt_keys2, (pad_x + 15, foot_y + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 185, 195), 1, cv2.LINE_AA)

        cv2.imshow(win_name, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key in [ord("q"), ord("Q"), 27]:
            print("\n[*] Menutup Wizard Kalibrasi Interaktif Rel Ganda.")
            break
        elif key in [ord("s"), ord("S")]:
            calibrator.compute_rectification(w_f, h_f)
            ok, msg = calibrator.save_calibration_to_config()
            save_msg = msg
        elif key in [ord("a"), ord("A")]:
            calibrator.auto_snap_rails(frame_raw)
        elif key in [ord("m"), ord("M")]:
            calibrator.mode = "AFFINE" if calibrator.mode == "PERSPECTIVE" else "PERSPECTIVE"
            calibrator.is_saved = False
            print(f"[*] Beralih ke Mode Transformasi: {calibrator.mode}")
        elif key in [ord(" ")]:
            is_paused = not is_paused
        elif key in [ord("n"), ord("N")]:
            if cap is not None:
                ret, frame_read = cap.read()
                if ret and frame_read is not None:
                    frame_raw = frame_read
                    frame_idx += 1
        elif key in [ord("p"), ord("P")]:
            if cap is not None:
                cur_pos = max(0, frame_idx - 2)
                cap.set(cv2.CAP_PROP_POS_FRAMES, cur_pos)
                ret, frame_read = cap.read()
                if ret and frame_read is not None:
                    frame_raw = frame_read
                    frame_idx = cur_pos + 1
        elif key in [ord("o"), ord("O"), ord("f"), ord("F")]:
            try:
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk()
                root.withdraw()
                root.attributes("-topmost", True)
                chosen = filedialog.askopenfilename(
                    title="Pilih Video / Foto Angle Konveyor Baru",
                    filetypes=[("Media Files", "*.mp4 *.avi *.mkv *.jpg *.png *.jpeg *.bmp"), ("All Files", "*.*")]
                )
                root.destroy()
                cv2.waitKey(1)
                if chosen and os.path.exists(chosen):
                    if cap is not None:
                        cap.release()
                    source_path = chosen
                    if source_path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                        is_image = True
                        cap = None
                        frame_raw = cv2.imread(source_path)
                    else:
                        is_image = False
                        cap = cv2.VideoCapture(source_path)
                        ret, frame_raw = cap.read()
                    if frame_raw is not None:
                        h_f, w_f = frame_raw.shape[:2]
                        calibrator.auto_snap_rails(frame_raw)
                        print(f"[*] Beralih ke media baru: {os.path.basename(source_path)}")
            except Exception as e:
                print(f"[!] Error open file: {e}")

    if cap is not None:
        cap.release()
    cv2.destroyWindow(win_name)


def main():
    parser = argparse.ArgumentParser(description="Alat Kalibrasi Interaktif Rel Ganda Konveyor (Dual-Rail Calibration Wizard)")
    parser.add_argument("--source", type=str, default=None, help="Path ke video atau foto sudut kamera baru")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path ke file konfigurasi target JSON")
    args = parser.parse_args()

    run_interactive_dual_rail_calibration(source_path=args.source, config_path=args.config)


if __name__ == "__main__":
    main()

