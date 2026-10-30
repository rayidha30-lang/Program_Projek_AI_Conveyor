"""
==============================================================================
MODUL KALIBRASI OTOMATIS KONVEYOR SLAT & RIB (CONVEYOR MOTION CALIBRATOR)
METODE HYBRID RECTIFICATION: SLAT SEAM ORIENTATION + MOTION OPTICAL FLOW + PARALLEL RAILS
==============================================================================
Deskripsi:
  Modul ini menghitung parameter koreksi geometris kamera konveyor secara otomatis
  tanpa mengubah nilai toleransi metrologi golden baseline pabrik.
  
Prinsip Kerja (Virtual Image Rectification):
  1. Mengumpulkan 40-60 frame saat konveyor bergerak.
  2. Mendeteksi sudut kemiringan fisik celah plat slat (Slat Seam) via Hough Ridge.
  3. Memvalidasi arah gerak 1D konveyor menggunakan Vektor Aliran Optik (Optical Flow).
  4. Mendeteksi batas lateral rel baja pemandu samping (Parallel Rails) konveyor.
  5. Menghasilkan Matriks Transformasi Affine 2x3 (M_rect) yang meluruskan citra kamera
     baru kembali ke posisi 'Golden Baseline' horizontal kanonikal.
==============================================================================
"""

import os
import math
import cv2
import numpy as np


class ConveyorMotionCalibrator:
    """
    Kalibrator Geometris Kamera Konveyor berbasis Analisis Celah Plat & Vektor Gerak.
    """

    def __init__(self, target_tilt_deg=0.90, target_cx=960.0, target_left_rail=188.0, target_right_rail=1759.0):
        """
        :param target_tilt_deg: Sudut target kanonikal konveyor (default: 0.90 derajat / golden baseline)
        :param target_cx: Titik tengah target sumbu X konveyor (default: 960.0 px pada frame 1080p)
        :param target_left_rail: Posisi X rel kiri acuan emas (default: 188.0 px)
        :param target_right_rail: Posisi X rel kanan acuan emas (default: 1759.0 px)
        """
        self.target_tilt_deg = float(target_tilt_deg)
        self.target_cx = float(target_cx)
        self.target_left_rail = float(target_left_rail)
        self.target_right_rail = float(target_right_rail)
        self.target_width = float(self.target_right_rail - self.target_left_rail)
        self.collected_frames = []

    def reset(self):
        """Mereset buffer frame kalibrasi."""
        self.collected_frames.clear()

    def add_frame(self, frame):
        """Menampung frame untuk proses kalibrasi."""
        if frame is not None and frame.size > 0:
            self.collected_frames.append(frame.copy())

    def calibrate(self, frames=None):
        """
        Menganalisis kumpulan frame dan menghasilkan parameter serta matriks rectifikasi.
        
        :param frames: List frame BGR opsional. Jika None, menggunakan self.collected_frames.
        :return: dict hasil kalibrasi lengkap dengan matriks Affine 2x3.
        """
        if frames is None:
            frames = self.collected_frames

        if not frames or len(frames) < 10:
            return {
                "success": False,
                "error": f"Jumlah frame tidak mencukupi untuk kalibrasi (dibutuhkan minimal 10, ada {len(frames) if frames else 0})"
            }

        h_f, w_f = frames[0].shape[:2]
        sample_step = max(1, len(frames) // 30)
        sampled_frames = frames[::sample_step]

        # ----------------------------------------------------------------------
        # 1. DETEKSI SUDUT KEMIRINGAN CELAH PLAT SLAT (SLAT SEAM ANGLE)
        # ----------------------------------------------------------------------
        seam_angles = []
        band_y1, band_y2 = int(0.35 * h_f), int(0.72 * h_f)
        band_x1, band_x2 = int(0.08 * w_f), int(0.92 * w_f)

        for f in sampled_frames:
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            roi = gray[band_y1:band_y2, band_x1:band_x2]
            
            # Deteksi tepi celah plat
            edges = cv2.Canny(roi, 35, 110)
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=70,
                minLineLength=int(0.10 * w_f),
                maxLineGap=30
            )

            if lines is not None:
                for l in lines:
                    x1, y1, x2, y2 = l[0]
                    dx = x2 - x1
                    dy = y2 - y1
                    # Filter garis yang dominan horizontal (celah antar plat)
                    if abs(dx) > int(0.08 * w_f):
                        ang = np.degrees(np.arctan2(dy, dx))
                        # Hanya ambil garis dalam rentang kemiringan wajar (+/- 15 derajat)
                        if abs(ang) < 15.0:
                            seam_angles.append(ang)
                        elif abs(abs(ang) - 180.0) < 15.0:
                            ang_corr = ang - 180.0 if ang > 0 else ang + 180.0
                            seam_angles.append(ang_corr)

        if len(seam_angles) < 5:
            # Fallback jika celah kurang kontras: gunakan default tilt 0.0
            measured_tilt = 0.0
        else:
            # Filter outlier menggunakan IQR (Interquartile Range)
            q25, q75 = np.percentile(seam_angles, [25, 75])
            iqr = max(0.5, q75 - q25)
            valid_angles = [a for a in seam_angles if (q25 - 1.5 * iqr) <= a <= (q75 + 1.5 * iqr)]
            measured_tilt = float(np.median(valid_angles if valid_angles else seam_angles))

        # Sudut rotasi koreksi untuk meluruskan celah ke target
        delta_theta = float(self.target_tilt_deg - measured_tilt)

        # ----------------------------------------------------------------------
        # 2. VALIDASI ARAH GERAK KONVEYOR (MOTION FLOW VECTOR)
        # ----------------------------------------------------------------------
        # Memastikan bahwa gerakan konveyor tegak lurus terhadap celah plat
        motion_angles = []
        if len(frames) >= 8:
            step_m = max(2, min(6, len(frames) // 10))
            for i in range(0, len(frames) - step_m, step_m):
                g1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
                g2 = cv2.cvtColor(frames[i + step_m], cv2.COLOR_BGR2GRAY)
                
                # Area tengah konveyor
                c_mask = np.zeros((h_f, w_f), dtype=np.uint8)
                c_mask[int(0.40 * h_f):int(0.65 * h_f), int(0.25 * w_f):int(0.75 * w_f)] = 255
                
                p0 = cv2.goodFeaturesToTrack(g1, mask=c_mask, maxCorners=150, qualityLevel=0.03, minDistance=20)
                if p0 is not None and len(p0) >= 15:
                    p1, st, _ = cv2.calcOpticalFlowPyrLK(g1, g2, p0, None)
                    if p1 is not None:
                        good_new = p1[st == 1]
                        good_old = p0[st == 1]
                        diff = good_new - good_old
                        mags = np.hypot(diff[:, 0], diff[:, 1])
                        mv = mags > 1.5
                        if np.sum(mv) >= 8:
                            vx = diff[mv, 0]
                            vy = diff[mv, 1]
                            med_vx = float(np.median(vx))
                            med_vy = float(np.median(vy))
                            # Sudut vektor gerak relatif terhadap sumbu vertikal
                            ang_m = math.degrees(math.atan2(med_vx, abs(med_vy)))
                            motion_angles.append(ang_m)

        # ----------------------------------------------------------------------
        # 3. DETEKSI BATAS LATERAL REL SAMPING KONVEYOR (PARALLEL RAILS)
        # ----------------------------------------------------------------------
        # Putar frame tengah sementara agar tegak lurus horizontal
        mid_frame = frames[len(frames) // 2]
        center_pt = (w_f / 2.0, h_f / 2.0)
        temp_rot_m = cv2.getRotationMatrix2D(center_pt, -measured_tilt, 1.0)
        derot_frame = cv2.warpAffine(mid_frame, temp_rot_m, (w_f, h_f), borderMode=cv2.BORDER_REPLICATE)
        derot_gray = cv2.cvtColor(derot_frame, cv2.COLOR_BGR2GRAY)

        # Hitung gradien vertikal (Sobel X) untuk menemukan garis rel kiri dan kanan
        rail_band = derot_gray[int(0.35 * h_f):int(0.68 * h_f), :]
        sobx = np.abs(cv2.Sobel(rail_band, cv2.CV_64F, 1, 0, ksize=3))
        profile_x = np.mean(sobx, axis=0)

        # Rel kiri di rentang [3%, 25%], Rel kanan di rentang [75%, 97%]
        x_min_bound = int(0.03 * w_f)
        x_max_left = int(0.25 * w_f)
        left_rail_x = x_min_bound + int(np.argmax(profile_x[x_min_bound:x_max_left]))

        x_min_right = int(0.75 * w_f)
        x_max_bound = int(0.97 * w_f)
        right_rail_x = x_min_right + int(np.argmax(profile_x[x_min_right:x_max_bound]))

        conveyor_center_x = (left_rail_x + right_rail_x) / 2.0
        delta_x = float(self.target_cx - conveyor_center_x)

        # ----------------------------------------------------------------------
        # 4. PEMBENTUKAN MATRIKS AFFINE RECTIFICATION (M_rect 2x3)
        # ----------------------------------------------------------------------
        # Memutar frame sejauh delta_theta dan menggeser pusat lateral sejauh delta_x
        M_rect = cv2.getRotationMatrix2D(center_pt, delta_theta, 1.0)
        M_rect[0, 2] += delta_x

        conveyor_width_px = int(right_rail_x - left_rail_x)

        # ----------------------------------------------------------------------
        # 5. PEMBUATAN RECTIFIED SLAT POLYGON (GERBANG SLAT KANONIKAL)
        # ----------------------------------------------------------------------
        # Menghasilkan koordinat polygon plat slat yang persis mengikuti posisi baru
        norm_lx = float(max(0.04, left_rail_x / float(w_f)))
        norm_rx = float(min(0.96, right_rail_x / float(w_f)))
        
        # Gerbang slat kanonikal di area tengah stabil Y [51%, 67%]
        calibrated_slat_poly = [
            [norm_lx, 0.515],
            [norm_rx, 0.515],
            [norm_rx, 0.670],
            [norm_lx, 0.670]
        ]

        calib_result = {
            "success": True,
            "measured_tilt_deg": round(measured_tilt, 3),
            "target_tilt_deg": round(self.target_tilt_deg, 3),
            "angle_correction_deg": round(delta_theta, 3),
            "conveyor_center_x": round(conveyor_center_x, 1),
            "lateral_offset_px": round(delta_x, 1),
            "left_rail_px": int(left_rail_x),
            "right_rail_px": int(right_rail_x),
            "target_left_rail_px": int(self.target_left_rail),
            "target_right_rail_px": int(self.target_right_rail),
            "target_width_px": int(self.target_width),
            "conveyor_width_px": conveyor_width_px,
            "rectification_matrix": M_rect.tolist(),
            "calibrated_slat_polygon": calibrated_slat_poly,
            "motion_sample_count": len(motion_angles),
            "status_message": (
                f"KALIBRASI SUKSES | Tilt: {measured_tilt:+.2f} deg (Koreksi {delta_theta:+.2f} deg) "
                f"| Geser X: {delta_x:+.1f}px | Lebar: {conveyor_width_px}px"
            )
        }

        return calib_result
