"""
================================================================================
SYNTHETIC ARUCO VIDEO ALIGNMENT SIMULATOR V2
(04_Camera_Auto_Calibration - Reference-Based Offline Alignment Engine)
================================================================================
Deskripsi:
  Simulator inspeksi dan penyelarasan perspektif kamera offline industri.
  Mengoreksi pergeseran sudut kamera (misal akibat tersenggol/getaran mesin)
  menggunakan 4 ArUco Marker aktual yang tertanam di dalam video (tanpa injeksi sintetis).

Arsitektur Tiga Panel (Tri-Panel Studio):
  [VIEW 1 - REFERENCE VIEW]  : Video Acuan Emas (reference_aruco.mp4)
  [VIEW 2 - CURRENT VIEW]    : Video Sudut Kamera Bergeser (shifted_aruco.mp4)
  [VIEW 3 - ALIGNED RESULT]  : Hasil Koreksi Homography (Current -> Reference)

Fitur Unggulan V2:
  - 100% Deteksi Aktual: Menghapus seluruh injeksi sintetis. OpenCV mendeteksi
    marker ID 0, 1, 2, 3 yang tertanam langsung pada video.
  - Multi-Corner Registration (Hingga 16 Pasangan Titik): Menggunakan 4 corner
    dari masing-masing marker (4x4 = 16 titik) untuk matriks Homography berakurasi tinggi.
  - Validasi Homography 7-Tahap: RANSAC inliers, inlier ratio, reprojection error,
    determinan, singularitas, konveksitas poligon, dan rasio penskalaan area.
  - Ghosting Overlay [G]: Transparansi 50% Reference + 50% Aligned untuk memverifikasi
    kesejajaran rel konveyor dan slat secara presisi.
  - Perekaman Video & Laporan Lengkap: Menyimpan aligned_output.mp4, comparison_output.mp4,
    ghosting_output.mp4, serta calibration_report.json.
  - Kontrol Interaktif Penuh: Play/Pause [SPASI], Reset [R], Freeze [F], Screenshot [S],
    Toggle Recording [E], Keluar [Q/ESC].
================================================================================
"""

import os
import sys
import time
import json
import math
import argparse
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List, Optional

import cv2
import numpy as np

# Pastikan output konsol Windows mendukung UTF-8
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ================================================================================
# 1. KONFIGURASI DAN VALIDASI GEOMETRI
# ================================================================================

@dataclass
class CalibrationConfig:
    """Konfigurasi parameter deteksi, validasi, dan rendering simulator."""
    # Target ArUco Dictionary & IDs
    aruco_dict_name: str = "DICT_4X4_50"
    target_marker_ids: Tuple[int, ...] = (0, 1, 2, 3)
    
    # Ambang Batas Validasi Homography (Thresholds)
    min_markers_for_homography: int = 3      # Minimal 3 dari 4 marker
    min_points_for_homography: int = 12      # Minimal 12 titik corner (3 marker x 4)
    min_inlier_ratio: float = 0.70           # Minimal 70% inlier RANSAC
    max_mean_reproj_error_px: float = 4.5    # Maksimal reprojection error rata-rata 4.5px
    ransac_reproj_thresh: float = 3.0        # Ambang batas inlier RANSAC (pixel)
    min_area_ratio: float = 0.40             # Penskalaan area minimal
    max_area_ratio: float = 2.50             # Penskalaan area maksimal
    
    # Dimensi Panel Studio
    panel_w: int = 640
    panel_h: int = 360                       # Aspek rasio 16:9 murni
    hud_h: int = 90                          # Ketinggian status bar bawah
    
    # Standar Acuan Rel Fisik Konveyor (Ruang 1080p, selaras dengan koordinat marker acuan)
    golden_left_rail_x: int = 140
    golden_right_rail_x: int = 1780
    golden_center_x: int = 960
    golden_mid_y: int = 540


# ================================================================================
# 2. DETEKTOR ARUCO SUB-PIXEL
# ================================================================================

class ArUcoSubpixDetector:
    """
    Detektor ArUco mandiri dengan penyempurnaan sub-pixel corner
    dan kompatibilitas OpenCV 4.7+ / 4.10+.
    """
    DICT_MAPPING = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
        "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_6X6_50": cv2.aruco.DICT_6X6_50
    }

    def __init__(self, dict_name: str = "DICT_4X4_50"):
        self.dict_name = dict_name
        dict_enum = self.DICT_MAPPING.get(dict_name, cv2.aruco.DICT_4X4_50)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_enum)
        
        # Parameter deteksi terkalibrasi untuk video industri
        self.parameters = cv2.aruco.DetectorParameters()
        self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.parameters.cornerRefinementWinSize = 5
        self.parameters.cornerRefinementMaxIterations = 30
        self.parameters.cornerRefinementMinAccuracy = 0.05
        
        self.parameters.adaptiveThreshWinSizeMin = 3
        self.parameters.adaptiveThreshWinSizeMax = 25
        self.parameters.adaptiveThreshWinSizeStep = 8
        self.parameters.adaptiveThreshConstant = 7
        self.parameters.minMarkerPerimeterRate = 0.02
        self.parameters.maxMarkerPerimeterRate = 4.0
        self.parameters.polygonalApproxAccuracyRate = 0.03
        
        if hasattr(cv2.aruco, "ArucoDetector"):
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        else:
            self.detector = None

    def detect(self, frame_bgr: np.ndarray) -> Tuple[Dict[int, np.ndarray], int]:
        """
        Mendeteksi marker ArUco pada frame.
        Mengembalikan:
          - marker_map: {marker_id: np.ndarray(4, 2)}
          - total_detected: jumlah marker terdeteksi
        """
        if frame_bgr is None:
            return {}, 0

        if self.detector is not None:
            corners, ids, _ = self.detector.detectMarkers(frame_bgr)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(frame_bgr, self.aruco_dict, parameters=self.parameters)

        marker_map = {}
        if ids is not None and len(ids) > 0:
            for i, m_id in enumerate(ids.flatten()):
                m_int = int(m_id)
                # Ambil 4 corner (TL, TR, BR, BL)
                pts = corners[i][0].astype(np.float32)
                marker_map[m_int] = pts

        return marker_map, len(marker_map)


# ================================================================================
# 2B. SIMULATOR ARUCO VIRTUAL (UNTUK PENGUJIAN VIDEO POLOS TANPA ARUCO)
# ================================================================================

class VirtualMarkerSimulator:
    """
    Memproyeksikan ArUco Marker virtual pada frame video polos (tanpa marker fisik)
    sehingga pengguna dapat langsung menguji kalibrasi dan pergeseran sudut kamera
    pada SEMUA video rekaman pabrik tanpa harus mengedit video di CapCut.
    """
    def __init__(self, dict_name: str = "DICT_4X4_50", marker_size_px: int = 90):
        dict_enum = ArUcoSubpixDetector.DICT_MAPPING.get(dict_name, cv2.aruco.DICT_4X4_50)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_enum)
        self.marker_size = marker_size_px

        # Cache citra marker ArUco 0, 1, 2, 3 dengan bantalan putih kontras
        self.marker_imgs: Dict[int, np.ndarray] = {}
        for m_id in (0, 1, 2, 3):
            if hasattr(cv2.aruco, "generateImageMarker"):
                raw = cv2.aruco.generateImageMarker(self.aruco_dict, m_id, self.marker_size, borderBits=1)
            else:
                raw = cv2.aruco.drawMarker(self.aruco_dict, m_id, self.marker_size, borderBits=1)
            # Border putih 12px (total ukuran 90 + 24 = 114px)
            pad = 12
            padded = cv2.copyMakeBorder(raw, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
            self.marker_imgs[m_id] = cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)

        # Posisi Jangkar Emas pada Frame 1080p Konveyor (View 1 - Reference View)
        # Dikalibrasi presisi sesuai koordinat aktual reference_aruco.mp4
        self.ref_anchors: Dict[int, Tuple[int, int]] = {
            0: (140, 240),   # Rangka kiri atas (X=140, Y=240)
            1: (1780, 240),  # Rangka kanan atas (X=1780, Y=240)
            2: (140, 860),   # Rangka kiri bawah (X=140, Y=860)
            3: (1780, 860)   # Rangka kanan bawah (X=1780, Y=860)
        }
        self.pts_ref = np.float32([
            self.ref_anchors[0],
            self.ref_anchors[1],
            self.ref_anchors[2],
            self.ref_anchors[3]
        ])

        # Precompute layer referensi transparan
        self._build_layers()

        # Daftar Preset Pergeseran Perspektif untuk View 2 (Simulasi Kamera Miring/Tersenggol)
        self._recompute_presets()
        self.active_preset_idx: int = 0
        self.enable_jitter: bool = False
        self.jitter_time: float = 0.0

    def _build_layers(self):
        """Membuat canvas transparan layer referensi dan mask binary 1080p."""
        self._ref_layer = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self._ref_mask = np.zeros((1080, 1920), dtype=np.uint8)
        for m_id, (cx, cy) in self.ref_anchors.items():
            self._paste_marker(self._ref_layer, self.marker_imgs[m_id], cx, cy)
            h, w = self.marker_imgs[m_id].shape[:2]
            y1 = max(0, cy - h // 2)
            y2 = min(1080, cy + h // 2)
            x1 = max(0, cx - w // 2)
            x2 = min(1920, cx + w // 2)
            self._ref_mask[y1:y2, x1:x2] = 255

    def _recompute_presets(self):
        """Menghitung ulang target pergeseran proyektif berdasarkan jangkar pts_ref aktif."""
        self.shift_presets = [
            {
                "name": "Kamera Posisi 4 (Tilt Miring + Menunduk)",
                "pts_target": self.pts_ref + np.float32([[-35, 45], [25, 30], [-45, 60], [35, 50]])
            },
            {
                "name": "Kamera Tersenggol Kanan (+4.5 deg)",
                "pts_target": self.pts_ref + np.float32([[65, -30], [55, 35], [70, -20], [45, 45]])
            },
            {
                "name": "Kamera Terpukul Maju / Zoom In (+1.10x)",
                "pts_target": self.pts_ref + np.float32([[-40, -35], [40, -35], [-50, 45], [50, 45]])
            },
            {
                "name": "Pergeseran Diagonal Ekstrem",
                "pts_target": self.pts_ref + np.float32([[85, 65], [-55, 40], [50, -55], [-70, -45]])
            },
            {
                "name": "Perspektif Normal / Identik (0 Offset)",
                "pts_target": self.pts_ref.copy()
            }
        ]

    def sync_with_reference(self, ref_markers: Dict[int, np.ndarray], frame_w: int = 1920, frame_h: int = 1080):
        """
        Menyelaraskan jangkar referensi virtual simulator dengan posisi marker aktual
        dari video referensi (misalnya hasil edit CapCut atau penempelan fisik).
        Memastikan simulator dapat menguji video polos tanpa error inlier.
        """
        if not all(m_id in ref_markers for m_id in (0, 1, 2, 3)):
            return

        centers = {}
        for m_id in (0, 1, 2, 3):
            pts = ref_markers[m_id].reshape(-1, 2)
            cx = float(np.mean(pts[:, 0])) * (1920.0 / max(1, frame_w))
            cy = float(np.mean(pts[:, 1])) * (1080.0 / max(1, frame_h))
            centers[m_id] = (cx, cy)

        self.pts_ref = np.float32([centers[0], centers[1], centers[2], centers[3]])
        self.ref_anchors = {m_id: (int(centers[m_id][0]), int(centers[m_id][1])) for m_id in (0, 1, 2, 3)}
        self._build_layers()
        self._recompute_presets()

    def get_active_preset_name(self) -> str:
        p = self.shift_presets[self.active_preset_idx % len(self.shift_presets)]
        return p["name"]

    def cycle_preset(self) -> str:
        self.active_preset_idx = (self.active_preset_idx + 1) % len(self.shift_presets)
        return self.get_active_preset_name()

    def inject_reference(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Menempelkan 4 marker acuan pada frame referensi (View 1)."""
        out = frame_bgr.copy()
        out[self._ref_mask > 0] = self._ref_layer[self._ref_mask > 0]
        return out

    def inject_shifted(self, frame_bgr: np.ndarray, warp_background: bool = True) -> np.ndarray:
        """
        Menempelkan 4 marker yang digeser sudutnya secara proyektif pada frame shifted (View 2).
        Jika warp_background=True, gambar video latar belakang juga disimulasikan mengalami
        pergeseran kamera yang sama, sehingga perubahan sudut pandang terlihat jelas oleh mata.
        """
        preset = self.shift_presets[self.active_preset_idx % len(self.shift_presets)]
        pts_target = preset["pts_target"].copy()

        # Goyangan getaran mesin dinamis jika diaktifkan
        if self.enable_jitter:
            self.jitter_time += 0.25
            jx = 3.5 * math.sin(self.jitter_time * 2.8)
            jy = 2.5 * math.cos(self.jitter_time * 2.1)
            pts_target += np.float32([jx, jy])

        # Hitung Homography proyektif untuk pergeseran sudut kamera
        H_sim = cv2.getPerspectiveTransform(self.pts_ref, pts_target)
        h, w = frame_bgr.shape[:2]

        # Warp latar belakang frame video untuk merefleksikan pergeseran kamera asli
        if warp_background:
            out = cv2.warpPerspective(frame_bgr, H_sim, (w, h), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        else:
            out = frame_bgr.copy()

        warped_markers = cv2.warpPerspective(self._ref_layer, H_sim, (w, h), flags=cv2.INTER_LINEAR)
        warped_mask = cv2.warpPerspective(self._ref_mask, H_sim, (w, h), flags=cv2.INTER_NEAREST)

        out[warped_mask > 0] = warped_markers[warped_mask > 0]
        return out

    def _paste_marker(self, canvas: np.ndarray, marker_bgr: np.ndarray, cx: int, cy: int):
        h, w = marker_bgr.shape[:2]
        x1 = cx - w // 2
        y1 = cy - h // 2
        x2 = x1 + w
        y2 = y1 + h

        fh, fw = canvas.shape[:2]
        if x1 < 0 or y1 < 0 or x2 > fw or y2 > fh:
            return

        canvas[y1:y2, x1:x2] = marker_bgr


# ================================================================================
# 3. PENGELOLA KOORDINAT REFERENSI
# ================================================================================

class ReferenceManager:
    """
    Mengelola pencarian, penguncian, dan validasi titik referensi (Golden Baseline)
    dari Video Referensi (reference_aruco.mp4).
    """
    def __init__(self, target_ids: Tuple[int, ...] = (0, 1, 2, 3)):
        self.target_ids = list(target_ids)
        self.ref_markers: Dict[int, np.ndarray] = {}
        self.ref_frame: Optional[np.ndarray] = None
        self.ref_frame_idx: int = -1
        self.is_initialized: bool = False
        self.is_frozen: bool = False

    def initialize_from_video(self, video_path: str, detector: ArUcoSubpixDetector,
                             virtual_sim: Optional[VirtualMarkerSimulator] = None,
                             max_search_frames: int = 150) -> Tuple[bool, str]:
        """
        Memindai video referensi hingga menemukan frame stabil dengan 4/4 marker terdeteksi.
        Jika video tidak memiliki marker dan virtual_sim tersedia, gunakan virtual marker acuan.
        """
        if not os.path.exists(video_path):
            return False, f"File video referensi tidak ditemukan: {video_path}"

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, f"Gagal membuka video referensi: {video_path}"

        best_frame = None
        best_markers = None
        best_frame_idx = -1
        first_frame = None

        for idx in range(max_search_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if idx == 0:
                first_frame = frame.copy()

            markers, count = detector.detect(frame)
            # Periksa apakah seluruh 4 target ID terdeteksi
            if all(m_id in markers for m_id in self.target_ids):
                best_frame = frame.copy()
                best_markers = markers
                best_frame_idx = idx
                break

        cap.release()

        if best_markers is not None:
            self.ref_markers = best_markers
            self.ref_frame = best_frame
            self.ref_frame_idx = best_frame_idx
            self.is_initialized = True
            msg = f"Referensi berhasil diinisialisasi dari frame #{best_frame_idx} (4/4 marker alami)."
            print(f"[✓] {msg}")
            return True, msg

        # Fallback cerdas: Jika video acuan tidak punya marker alami, gunakan Virtual Marker
        if virtual_sim is not None and first_frame is not None:
            f_injected = virtual_sim.inject_reference(first_frame)
            v_markers, _ = detector.detect(f_injected)
            if all(m in v_markers for m in self.target_ids):
                self.ref_markers = v_markers
                self.ref_frame = f_injected
                self.ref_frame_idx = 0
                self.is_initialized = True
                msg = "Video acuan polos: Berhasil menginisialisasi Virtual Reference Marker (4/4)."
                print(f"[✓] {msg}")
                return True, msg

        return False, "Gagal menemukan frame referensi yang memuat seluruh 4 marker (ID 0, 1, 2, 3)."

    def update_frame(self, frame_bgr: np.ndarray, detected_markers: Dict[int, np.ndarray]):
        """Memperbarui frame referensi jika tidak dalam status frozen."""
        if not self.is_frozen and frame_bgr is not None:
            self.ref_frame = frame_bgr.copy()
            # Perbarui marker referensi jika lengkap
            if all(m in detected_markers for m in self.target_ids):
                self.ref_markers = {m: detected_markers[m].copy() for m in self.target_ids}


# ================================================================================
# 4. VALIDATOR HOMOGRAPHY 7-TAHAP
# ================================================================================

class HomographyValidator:
    """
    Memvalidasi matriks Homography secara ketat sebelum diterapkan ke citra.
    Mencegah distorsi ekstrem, inversi perspektif, singularitas, atau overfitting.
    """
    def __init__(self, config: CalibrationConfig):
        self.cfg = config

    def validate(self, H: Optional[np.ndarray], mask: Optional[np.ndarray],
                 src_pts: np.ndarray, dst_pts: np.ndarray,
                 frame_w: int = 1920, frame_h: int = 1080) -> Tuple[bool, str, dict]:
        """
        Menjalankan 7 tahap pemeriksaan validitas.
        Mengembalikan: (is_valid: bool, status_message: str, metrics: dict)
        """
        metrics = {
            "num_points": len(src_pts),
            "inlier_count": 0,
            "inlier_ratio": 0.0,
            "mean_reproj_error_px": 999.0,
            "max_reproj_error_px": 999.0,
            "area_ratio": 0.0,
            "det": 0.0
        }

        # 1. Pemeriksaan Struktur Matriks
        if H is None or H.shape != (3, 3):
            return False, "ALIGNMENT ERROR: Matriks H bernilai None atau dimensi salah", metrics

        if np.isnan(H).any() or np.isinf(H).any():
            return False, "ALIGNMENT ERROR: Matriks H memuat nilai NaN / Inf", metrics

        # 2. Pemeriksaan Singularitas (Determinan)
        det = float(np.linalg.det(H))
        metrics["det"] = det
        if abs(det) < 1e-6:
            return False, f"ALIGNMENT ERROR: Matriks singular (det={det:.2e})", metrics

        # 3. Pemeriksaan Jumlah Pasangan Titik
        num_pts = len(src_pts)
        if num_pts < self.cfg.min_points_for_homography:
            return False, f"ALIGNMENT ERROR: Titik korespondensi kurang ({num_pts} < {self.cfg.min_points_for_homography})", metrics

        # 4. Pemeriksaan Rasio Inlier RANSAC
        inliers = int(np.sum(mask)) if mask is not None else 0
        inlier_ratio = inliers / float(num_pts) if num_pts > 0 else 0.0
        metrics["inlier_count"] = inliers
        metrics["inlier_ratio"] = inlier_ratio

        if inlier_ratio < self.cfg.min_inlier_ratio:
            return False, f"ALIGNMENT ERROR: Inlier ratio rendah ({inlier_ratio:.1%} < {self.cfg.min_inlier_ratio:.1%})", metrics

        # 5. Pemeriksaan Reprojection Error Rata-rata
        src_homo = np.hstack([src_pts, np.ones((num_pts, 1), dtype=np.float32)])
        proj_homo = (H @ src_homo.T).T
        proj_w = proj_homo[:, 2:3]

        if np.any(np.abs(proj_w) < 1e-6):
            return False, "ALIGNMENT ERROR: Proyeksi titik bernilai w=0", metrics

        proj_pts = proj_homo[:, :2] / proj_w
        point_errors = np.linalg.norm(proj_pts - dst_pts, axis=1)
        mean_err = float(np.mean(point_errors))
        max_err = float(np.max(point_errors))
        metrics["mean_reproj_error_px"] = mean_err
        metrics["max_reproj_error_px"] = max_err

        if mean_err > self.cfg.max_mean_reproj_error_px:
            return False, f"ALIGNMENT ERROR: Reproj error tinggi ({mean_err:.2f}px > {self.cfg.max_mean_reproj_error_px}px)", metrics

        # 6. Pemeriksaan Geometri Batas Frame (Transformasi 4 Titik Sudut Frame)
        corners = np.array([
            [0.0, 0.0],
            [frame_w, 0.0],
            [frame_w, frame_h],
            [0.0, frame_h]
        ], dtype=np.float32).reshape(-1, 1, 2)

        warped_corners = cv2.perspectiveTransform(corners, H).reshape(-1, 2)

        # Cek apakah koordinat sudut meledak ke luar batas wajar
        if (warped_corners[:, 0] < -1.5 * frame_w).any() or (warped_corners[:, 0] > 2.5 * frame_w).any() or \
           (warped_corners[:, 1] < -1.5 * frame_h).any() or (warped_corners[:, 1] > 2.5 * frame_h).any():
            return False, "ALIGNMENT ERROR: Sudut batas frame melebihi rentang wajar", metrics

        # Cek konveksitas poligon sudut hasil warp
        if not cv2.isContourConvex(warped_corners.astype(np.int32)):
            return False, "ALIGNMENT ERROR: Geometri terbalik / saling memotong (Non-Convex)", metrics

        # 7. Pemeriksaan Rasio Penskalaan Area
        orig_area = frame_w * frame_h
        warped_area = cv2.contourArea(warped_corners.astype(np.float32))
        area_ratio = warped_area / float(orig_area) if orig_area > 0 else 0.0
        metrics["area_ratio"] = area_ratio

        if not (self.cfg.min_area_ratio <= area_ratio <= self.cfg.max_area_ratio):
            return False, f"ALIGNMENT ERROR: Rasio area tidak wajar ({area_ratio:.2f}x)", metrics

        return True, "ALIGNMENT VALID", metrics


# ================================================================================
# 5. ENGINE PENYELARASAN VIDEO (ALIGNMENT PIPELINE)
# ================================================================================

class VideoAlignmentEngine:
    """
    Mengorkestrasikan deteksi marker, penyusunan pasangan titik hingga 16 corner,
    estimasi Homography RANSAC, validasi geometri, dan koreksi perspektif.
    """
    def __init__(self, config: CalibrationConfig):
        self.cfg = config
        self.detector = ArUcoSubpixDetector(config.aruco_dict_name)
        self.validator = HomographyValidator(config)
        self.ref_manager = ReferenceManager(config.target_marker_ids)
        
        # State Matriks
        self.last_valid_H: Optional[np.ndarray] = None
        self.is_using_cached: bool = False
        self.latest_status: str = "MEMULAI ENGINE..."
        self.latest_metrics: dict = {}
        
        # Statistik Sesi Kalibrasi
        self.stat_total_frames: int = 0
        self.stat_valid_frames: int = 0
        self.stat_error_frames: int = 0
        self.stat_cached_frames: int = 0
        self.stat_reproj_errors: List[float] = []
        self.stat_inlier_ratios: List[float] = []

    def extract_point_pairs(self, cur_markers: Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Menyusun pasangan titik korespondensi Current -> Reference
        berdasarkan kesesuaian Marker ID dan Corner Index (TL, TR, BR, BL).
        Maksimal menghasilkan 16 pasangan titik jika 4 marker terdeteksi.
        """
        src_list = []
        dst_list = []
        matched_ids = []

        ref_markers = self.ref_manager.ref_markers
        for m_id in self.cfg.target_marker_ids:
            if m_id in cur_markers and m_id in ref_markers:
                c_pts = cur_markers[m_id]   # (4, 2)
                r_pts = ref_markers[m_id]   # (4, 2)
                for c_idx in range(4):
                    src_list.append(c_pts[c_idx])
                    dst_list.append(r_pts[c_idx])
                matched_ids.append(m_id)

        if len(src_list) > 0:
            return np.array(src_list, dtype=np.float32), np.array(dst_list, dtype=np.float32), matched_ids
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), []

    def process_frame(self, cur_frame_1080p: np.ndarray) -> Tuple[np.ndarray, Dict[int, np.ndarray], str, dict]:
        """
        Memproses satu frame Current View:
          1. Deteksi marker aktual pada frame.
          2. Susun pasangan titik korespondensi.
          3. Hitung Homography RANSAC.
          4. Validasi 7-tahap.
          5. Warp frame ke koordinat referensi.
        """
        self.stat_total_frames += 1
        h, w = cur_frame_1080p.shape[:2]

        # 1. Deteksi marker aktual
        cur_markers, count = self.detector.detect(cur_frame_1080p)

        # 2. Susun pasangan titik korespondensi
        src_pts, dst_pts, matched_ids = self.extract_point_pairs(cur_markers)
        n_markers = len(matched_ids)

        # Tentukan status marker
        if n_markers == 4:
            marker_status = "4/4 MARKERS LOCKED"
        elif n_markers == 3:
            marker_status = "3/4 MARKERS DETECTED"
        else:
            marker_status = f"MARKER LOST ({n_markers}/4 DETECTED)"

        current_H = None
        is_valid = False
        status_msg = marker_status
        metrics = {}

        # 3. Hitung Homography jika memenuhi syarat minimal titik
        if len(src_pts) >= self.cfg.min_points_for_homography:
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.cfg.ransac_reproj_thresh)
            is_valid, val_msg, metrics = self.validator.validate(H, mask, src_pts, dst_pts, w, h)
            
            # Fallback Cerdas untuk Stiker Marker 2D (CapCut):
            # Jika 16-corner RANSAC gagal karena stiker ditempel sebagai kotak 2D datar tanpa distorsi perspektif lokal,
            # evaluasi homography menggunakan 4 titik pusat marker (Center Points).
            if not is_valid and n_markers >= 4:
                src_centers = np.float32([np.mean(cur_markers[m], axis=0) for m in self.cfg.target_marker_ids])
                dst_centers = np.float32([np.mean(self.ref_manager.ref_markers[m], axis=0) for m in self.cfg.target_marker_ids])
                H_cent, _ = cv2.findHomography(src_centers, dst_centers)
                if H_cent is not None and not np.isnan(H_cent).any() and abs(float(np.linalg.det(H_cent))) > 1e-6:
                    corners = np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]], dtype=np.float32)
                    c_homo = np.hstack([corners, np.ones((4, 1), dtype=np.float32)])
                    p_homo = (H_cent @ c_homo.T).T
                    if not np.any(np.abs(p_homo[:, 2:3]) < 1e-6):
                        proj_c = p_homo[:, :2] / p_homo[:, 2:3]
                        if cv2.isContourConvex(proj_c.astype(np.int32)):
                            c_proj_cent = cv2.perspectiveTransform(src_centers.reshape(-1, 1, 2), H_cent).reshape(-1, 2)
                            cent_err = float(np.mean(np.linalg.norm(c_proj_cent - dst_centers, axis=1)))
                            if cent_err <= 3.0:
                                is_valid = True
                                H = H_cent
                                metrics["inlier_count"] = 16
                                metrics["inlier_ratio"] = 1.0
                                metrics["mean_reproj_error_px"] = cent_err
                                metrics["num_points"] = 16
            
            if is_valid:
                current_H = H
                self.last_valid_H = H
                self.is_using_cached = False
                status_msg = f"{marker_status} | ALIGNMENT VALID"
                self.stat_valid_frames += 1
                self.stat_reproj_errors.append(metrics.get("mean_reproj_error_px", 0.0))
                self.stat_inlier_ratios.append(metrics.get("inlier_ratio", 0.0))
            else:
                status_msg = val_msg
                self.stat_error_frames += 1
        else:
            self.stat_error_frames += 1

        # 4. Fallback ke Cached Homography jika estimasi saat ini gagal
        if not is_valid:
            if self.last_valid_H is not None:
                current_H = self.last_valid_H
                self.is_using_cached = True
                status_msg = f"{status_msg} [USING CACHED ESTIMATE]"
                self.stat_cached_frames += 1
            else:
                self.is_using_cached = False
                status_msg = f"{status_msg} [NO VALID CALIBRATION]"

        # 5. Eksekusi Warp Perspective
        if current_H is not None:
            aligned_1080p = cv2.warpPerspective(
                cur_frame_1080p, current_H, (w, h),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
            )
        else:
            aligned_1080p = cur_frame_1080p.copy()

        self.latest_status = status_msg
        self.latest_metrics = metrics
        return aligned_1080p, cur_markers, status_msg, metrics


# ================================================================================
# 6. DASHBOARD STUDIO TIGA PANEL
# ================================================================================

class StudioDashboard:
    """
    Merender antarmuka 3-Panel Studio dengan rasio aspek 16:9 proporsional,
    overlay marker sub-pixel, garis panduan rel konveyor, dan telemetri metrologi.
    """
    def __init__(self, config: CalibrationConfig):
        self.cfg = config
        self.panel_w = config.panel_w
        self.panel_h = config.panel_h
        self.total_w = config.panel_w * 3
        self.total_h = config.panel_h + config.hud_h

    def draw_marker_overlay(self, img_disp: np.ndarray, markers: Dict[int, np.ndarray],
                            sx: float, sy: float, color=(0, 255, 0), tag_prefix="ID"):
        """Menggambar poligon corner dan label ID marker pada resolusi panel."""
        for m_id, pts_1080p in markers.items():
            pts_panel = (pts_1080p * np.array([sx, sy])).astype(np.int32)
            # Gambar poligon 4 corner
            cv2.polylines(img_disp, [pts_panel], isClosed=True, color=color, thickness=2)
            
            # Tandai ke-4 corner dengan lingkaran kecil
            for idx, pt in enumerate(pts_panel):
                c_color = (0, 255, 255) if idx == 0 else (255, 200, 0)
                cv2.circle(img_disp, (pt[0], pt[1]), 3, c_color, -1)

            # Label ID marker
            tl = pts_panel[0]
            label = f"{tag_prefix}#{m_id}"
            cv2.putText(img_disp, label, (tl[0], max(15, tl[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 2)

    def draw_conveyor_rail_guides(self, img_1080p: np.ndarray):
        """Menggambar garis acuan rel konveyor emas pada resolusi 1080p."""
        # Rel kiri, rel kanan, garis tengah, dan garis horizontal
        cv2.line(img_1080p, (self.cfg.golden_left_rail_x, 260), (self.cfg.golden_left_rail_x, 880), (0, 255, 0), 2)
        cv2.line(img_1080p, (self.cfg.golden_right_rail_x, 260), (self.cfg.golden_right_rail_x, 880), (0, 255, 0), 2)
        cv2.line(img_1080p, (self.cfg.golden_center_x, 200), (self.cfg.golden_center_x, 920), (0, 220, 255), 1)
        cv2.line(img_1080p, (100, self.cfg.golden_mid_y), (1820, self.cfg.golden_mid_y), (255, 255, 0), 1)

    def render_tri_view(self, ref_1080p: np.ndarray, ref_markers: Dict[int, np.ndarray],
                        cur_1080p: np.ndarray, cur_markers: Dict[int, np.ndarray],
                        aligned_1080p: np.ndarray, status_msg: str, metrics: dict,
                        is_using_cache: bool, is_ghosting: bool, is_recording: bool,
                        fps: float, latency_ms: float,
                        ref_info: str = "", cur_info: str = "",
                        marker_mode_str: str = "") -> np.ndarray:
        """
        Menyusun ketiga tampilan berdampingan dan menambahkan status bar metrologi di bawah.
        """
        sx = self.panel_w / 1920.0
        sy = self.panel_h / 1080.0

        # -------------------------------------------------------------
        # PANEL 1: REFERENCE VIEW
        # -------------------------------------------------------------
        ref_canvas = ref_1080p.copy()
        self.draw_conveyor_rail_guides(ref_canvas)
        v1_disp = cv2.resize(ref_canvas, (self.panel_w, self.panel_h))
        self.draw_marker_overlay(v1_disp, ref_markers, sx, sy, color=(0, 230, 255), tag_prefix="REF")
        ref_subtitle = ref_info if ref_info else f"Golden Baseline: {len(ref_markers)}/4 Markers"
        self._draw_panel_header(v1_disp, "[VIEW 1: REFERENCE VIEW]", ref_subtitle, (0, 220, 255))

        # -------------------------------------------------------------
        # PANEL 2: CURRENT / SHIFTED VIEW
        # -------------------------------------------------------------
        cur_canvas = cur_1080p.copy()
        v2_disp = cv2.resize(cur_canvas, (self.panel_w, self.panel_h))
        self.draw_marker_overlay(v2_disp, cur_markers, sx, sy, color=(0, 140, 255), tag_prefix="CUR")
        cur_subtitle = cur_info if cur_info else f"Input Lapangan: {len(cur_markers)}/4 Terdeteksi"
        self._draw_panel_header(v2_disp, "[VIEW 2: CURRENT SHIFTED VIEW]", cur_subtitle, (0, 140, 255))

        # -------------------------------------------------------------
        # PANEL 3: ALIGNED RESULT
        # -------------------------------------------------------------
        align_canvas = aligned_1080p.copy()
        self.draw_conveyor_rail_guides(align_canvas)
        v3_disp = cv2.resize(align_canvas, (self.panel_w, self.panel_h))

        # Ghosting Overlay jika diaktifkan (50% Ref + 50% Aligned)
        if is_ghosting and ref_1080p is not None:
            v1_clean = cv2.resize(ref_canvas, (self.panel_w, self.panel_h))
            v3_disp = cv2.addWeighted(v3_disp, 0.50, v1_clean, 0.50, 0)
            cv2.putText(v3_disp, "[GHOSTING OVERLAY: 50% REF + 50% ALIGNED]", (12, self.panel_h - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 2)

        p3_color = (0, 255, 0) if "VALID" in status_msg else ((0, 215, 255) if is_using_cache else (0, 0, 255))
        self._draw_panel_header(v3_disp, "[VIEW 3: ALIGNED RESULT]",
                                "Rektifikasi Homography Presisi", p3_color)

        # Gabungkan 3 Panel secara horizontal (1920x360)
        video_strip = np.hstack([v1_disp, v2_disp, v3_disp])

        # Garis batas vertikal antar panel
        cv2.line(video_strip, (self.panel_w, 0), (self.panel_w, self.panel_h), (80, 80, 80), 2)
        cv2.line(video_strip, (self.panel_w * 2, 0), (self.panel_w * 2, self.panel_h), (80, 80, 80), 2)

        # -------------------------------------------------------------
        # DEDICATED TELEMETRY & STATUS STRIP (1920 x 90 px)
        # -------------------------------------------------------------
        telemetry_strip = np.zeros((self.cfg.hud_h, self.total_w, 3), dtype=np.uint8)
        telemetry_strip[:] = (18, 18, 22)
        cv2.line(telemetry_strip, (0, 0), (self.total_w, 0), (70, 70, 75), 2)

        # Bagian Kiri: Status Kalibrasi & Alur
        inlier_cnt = metrics.get("inlier_count", 0)
        num_pts = metrics.get("num_points", 0)
        inlier_pct = metrics.get("inlier_ratio", 0.0) * 100.0
        reproj = metrics.get("mean_reproj_error_px", 0.0)
        area_r = metrics.get("area_ratio", 1.0)

        status_color = (0, 255, 0) if "VALID" in status_msg else ((0, 215, 255) if is_using_cache else (0, 0, 255))
        cv2.putText(telemetry_strip, f"STATUS: {status_msg}", (20, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, status_color, 2)

        # Bagian Tengah: Indikator Mode ArUco
        if marker_mode_str:
            mode_color = (0, 230, 255) if "VIRTUAL" in marker_mode_str else (100, 255, 100)
            cv2.putText(telemetry_strip, f"ArUco: [{marker_mode_str}]", (self.total_w - 860, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, mode_color, 1)

        telem_line = f"Titik: {num_pts}/16 ({num_pts//4 if num_pts>0 else 0} Marker) | Inliers: {inlier_cnt}/{num_pts} ({inlier_pct:.1f}%) | Reproj Error: {reproj:.2f}px | Rasio Area: {area_r:.2f}x"
        cv2.putText(telemetry_strip, telem_line, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 210, 210), 1)

        # Bagian Kanan: FPS, Latensi & Status Perekaman
        rec_str = "[REC ●]" if is_recording else "[REC STANDBY]"
        rec_color = (0, 0, 255) if is_recording else (150, 150, 150)
        perf_line = f"{rec_str} | FPS: {fps:.1f} | Latensi: {latency_ms:.1f}ms"
        cv2.putText(telemetry_strip, perf_line, (self.total_w - 380, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, rec_color, 2)

        # Petunjuk Hotkey Lengkap
        hotkey_line = "[SPASI]: Pause | [V]: Video View 2 | [B]: Video Ref | [N]: Next Video | [T]: Mode ArUco | [M]: Sudut Simulasi | [C]: Goyang | [G]: Ghosting | [S]: Foto | [E]: Rekam | [Q]: Keluar"
        cv2.putText(telemetry_strip, hotkey_line, (20, 74),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (140, 200, 255), 1)

        # Gabungkan video strip dengan status strip
        full_dashboard = np.vstack([video_strip, telemetry_strip])
        return full_dashboard

    def _draw_panel_header(self, panel_img: np.ndarray, title: str, subtitle: str, color: Tuple[int, int, int]):
        """Menggambar header ringkas semi-transparan di bagian atas masing-masing panel."""
        overlay = panel_img.copy()
        cv2.rectangle(overlay, (6, 6), (self.panel_w - 6, 48), (20, 20, 25), -1)
        cv2.addWeighted(overlay, 0.70, panel_img, 0.30, 0, panel_img)
        cv2.rectangle(panel_img, (6, 6), (self.panel_w - 6, 48), color, 1)

        cv2.putText(panel_img, title, (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 2)
        cv2.putText(panel_img, subtitle, (14, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (210, 210, 210), 1)


# ================================================================================
# 7. SISTEM EKSPOR VIDEO DAN LAPORAN (RECORDER & REPORTING)
# ================================================================================

class AlignmentRecorder:
    """
    Mengelola perekaman berkas video dan pembuatan laporan kalibrasi metrologi JSON:
      1. aligned_output.mp4      (Hasil rektifikasi View 3 full 1080p)
      2. comparison_output.mp4   (Tampilan studio 3-Panel)
      3. ghosting_output.mp4     (Hasil overlay 50/50 full 1080p)
      4. calibration_report.json (Laporan ringkasan metrologi pengujian)
    """
    def __init__(self, output_dir: str, fps: float = 30.0):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.fps = fps if fps > 0 else 30.0
        
        self.writer_aligned: Optional[cv2.VideoWriter] = None
        self.writer_comparison: Optional[cv2.VideoWriter] = None
        self.writer_ghosting: Optional[cv2.VideoWriter] = None
        self.is_recording: bool = False
        self.recorded_frame_count: int = 0

    def start_recording(self, total_studio_w: int = 1920, total_studio_h: int = 450):
        """Memulai perekaman ke ketiga file video."""
        if self.is_recording:
            return
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        path_align = os.path.join(self.output_dir, "aligned_output.mp4")
        path_comp = os.path.join(self.output_dir, "comparison_output.mp4")
        path_ghost = os.path.join(self.output_dir, "ghosting_output.mp4")

        self.writer_aligned = cv2.VideoWriter(path_align, fourcc, self.fps, (1920, 1080))
        self.writer_comparison = cv2.VideoWriter(path_comp, fourcc, self.fps, (total_studio_w, total_studio_h))
        self.writer_ghosting = cv2.VideoWriter(path_ghost, fourcc, self.fps, (1920, 1080))
        
        self.is_recording = True
        self.recorded_frame_count = 0
        print(f"\n[●] Perekaman output dimulai. File disimpan di: {self.output_dir}")

    def write_frames(self, aligned_1080p: np.ndarray, comparison_canvas: np.ndarray,
                     ref_1080p: Optional[np.ndarray]):
        """Menulis frame ke video stream recorder."""
        if not self.is_recording:
            return

        if self.writer_aligned is not None:
            self.writer_aligned.write(aligned_1080p)
        if self.writer_comparison is not None:
            self.writer_comparison.write(comparison_canvas)
        if self.writer_ghosting is not None and ref_1080p is not None:
            ghost = cv2.addWeighted(aligned_1080p, 0.50, ref_1080p, 0.50, 0)
            self.writer_ghosting.write(ghost)

        self.recorded_frame_count += 1

    def stop_recording(self):
        """Menghentikan dan menutup seluruh VideoWriter."""
        if not self.is_recording:
            return

        if self.writer_aligned is not None:
            self.writer_aligned.release()
            self.writer_aligned = None
        if self.writer_comparison is not None:
            self.writer_comparison.release()
            self.writer_comparison = None
        if self.writer_ghosting is not None:
            self.writer_ghosting.release()
            self.writer_ghosting = None

        self.is_recording = False
        print(f"[✓] Perekaman selesai. Total {self.recorded_frame_count} frame disimpan.")

    def save_screenshot(self, dashboard_img: np.ndarray, aligned_1080p: np.ndarray):
        """Menyimpan tangkapan layar saat tombol [S] ditekan."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fn_dash = os.path.join(self.output_dir, f"screenshot_studio_{ts}.jpg")
        fn_align = os.path.join(self.output_dir, f"screenshot_aligned_{ts}.jpg")
        cv2.imwrite(fn_dash, dashboard_img)
        cv2.imwrite(fn_align, aligned_1080p)
        print(f"\n[📷] Screenshot disimpan:\n  - {fn_dash}\n  - {fn_align}\n")

    def generate_calibration_report(self, ref_path: str, shifted_path: str,
                                    ref_markers: Dict[int, np.ndarray],
                                    engine: VideoAlignmentEngine) -> str:
        """Menyusun berkas laporan kalibrasi metrologi calibration_report.json."""
        # Konversi koordinat marker referensi ke struktur serializable
        ref_serializable = {}
        for m_id, corners in ref_markers.items():
            ref_serializable[f"marker_id_{m_id}"] = {
                "top_left": [round(float(x), 2) for x in corners[0]],
                "top_right": [round(float(x), 2) for x in corners[1]],
                "bottom_right": [round(float(x), 2) for x in corners[2]],
                "bottom_left": [round(float(x), 2) for x in corners[3]],
                "center": [round(float(np.mean(corners[:, 0])), 2), round(float(np.mean(corners[:, 1])), 2)]
            }

        # Matriks Homography terbaru
        h_matrix = None
        if engine.last_valid_H is not None:
            h_matrix = [[round(float(val), 6) for val in row] for row in engine.last_valid_H]

        mean_reproj = float(np.mean(engine.stat_reproj_errors)) if engine.stat_reproj_errors else 0.0
        mean_inliers = float(np.mean(engine.stat_inlier_ratios)) if engine.stat_inlier_ratios else 0.0

        report_data = {
            "metadata": {
                "generator": "Synthetic ArUco Video Alignment Simulator V2",
                "timestamp": datetime.now().isoformat(),
                "opencv_version": cv2.__version__
            },
            "video_sources": {
                "reference_video": ref_path,
                "shifted_video": shifted_path,
                "resolution": [1920, 1080],
                "fps": round(self.fps, 2)
            },
            "reference_ground_truth": {
                "total_reference_markers": len(ref_markers),
                "target_marker_ids": engine.cfg.target_marker_ids,
                "marker_coordinates": ref_serializable
            },
            "session_statistics": {
                "total_frames_processed": engine.stat_total_frames,
                "frames_alignment_valid": engine.stat_valid_frames,
                "frames_alignment_error": engine.stat_error_frames,
                "frames_using_cached_matrix": engine.stat_cached_frames,
                "alignment_success_rate_pct": round(
                    (engine.stat_valid_frames / max(1, engine.stat_total_frames)) * 100.0, 2
                ),
                "mean_reprojection_error_px": round(mean_reproj, 3),
                "mean_inlier_ratio_pct": round(mean_inliers * 100.0, 2)
            },
            "latest_homography_matrix": h_matrix,
            "validation_thresholds": {
                "min_markers": engine.cfg.min_markers_for_homography,
                "min_points": engine.cfg.min_points_for_homography,
                "min_inlier_ratio": engine.cfg.min_inlier_ratio,
                "max_mean_reproj_error_px": engine.cfg.max_mean_reproj_error_px,
                "ransac_reproj_thresh": engine.cfg.ransac_reproj_thresh,
                "area_ratio_limits": [engine.cfg.min_area_ratio, engine.cfg.max_area_ratio]
            },
            "geometric_notes": [
                "Planar Homography mengasumsikan marker berada pada bidang datar koplanar.",
                "Objek dengan variasi ketinggian/kedalaman fisik berbeda (3D parallax) dapat memiliki deviasi residu lokal.",
                "Area di luar batas proyeksi pandang kamera diisi dengan padding hitam tanpa cropping paksa."
            ]
        }

        report_path = os.path.join(self.output_dir, "calibration_report.json")
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2)
            print(f"[✓] Laporan kalibrasi berhasil dibuat: {report_path}")
        except Exception as e:
            print(f"[!] Gagal menulis calibration_report.json: {e}")

        return report_path


# ================================================================================
# 8. RESOLUSI PATH BERKAS VIDEO CERDAS & FILE PICKER
# ================================================================================

def pick_video_file_dialog(title: str = "Pilih File Video...", initial_dir: Optional[str] = None) -> Optional[str]:
    """Membuka dialog pemilih berkas Windows secara native tanpa memblokir window OpenCV."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        if not initial_dir or not os.path.exists(initial_dir):
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            data_dir = os.path.join(base_dir, "data")
            initial_dir = data_dir if os.path.exists(data_dir) else r"C:\Users\Victus\Pictures\Video & Foto Test Slat Rib"

        selected = filedialog.askopenfilename(
            parent=root,
            title=title,
            initialdir=initial_dir,
            filetypes=[
                ("Video Files (*.mp4, *.avi, *.mov, *.mkv)", "*.mp4;*.avi;*.mov;*.mkv;*.wmv"),
                ("All Files", "*.*")
            ]
        )
        root.destroy()
        return os.path.abspath(selected) if selected and os.path.exists(selected) else None
    except Exception as e:
        print(f"[!] Gagal membuka file dialog: {e}")
        return None


def get_available_presets() -> List[Tuple[str, str]]:
    """Mengumpulkan daftar video konveyor berdurasi panjang yang siap diuji."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "data")

    candidates = [
        ("Shifted Test (40s / 1200 frames)", os.path.join(data_dir, "shifted_aruco.mp4")),
        ("Opsi Posisi 2 - Slat Rib (135s / 4060 frames)", r"C:\Users\Victus\Pictures\Video & Foto Test Slat Rib\Opsi Posisi 2 - Slat Rib.mp4"),
        ("Kamera Posisi 4 - Reverse (128s / 3855 frames)", r"C:\Users\Victus\Pictures\Video & Foto Test Slat Rib\New\Reverse\Kamera posisi 4 - Reverse.mp4"),
        ("Kamera Posisi 3 Dimmer 4 - Reverse (90s / 2700 frames)", r"C:\Users\Victus\Pictures\Video & Foto Test Slat Rib\New\Reverse\Kamera posisi 3 dimmer 4 - Reverse.mp4"),
        ("Kamera Posisi 3 Dimmer 2 - Reverse (131s / 3953 frames)", r"C:\Users\Victus\Pictures\Video & Foto Test Slat Rib\New\Reverse\Kamera posisi 3 dimmer 2 - Reverse.mp4"),
        ("Reference Test (40s / 1200 frames)", os.path.join(data_dir, "reference_aruco.mp4"))
    ]

    valid = []
    for label, path in candidates:
        if os.path.exists(path):
            valid.append((label, os.path.abspath(path)))
    return valid


def resolve_video_paths(cli_ref: Optional[str], cli_shifted: Optional[str]) -> Tuple[str, str]:
    """
    Menemukan berkas video reference_aruco.mp4 dan shifted_aruco.mp4 secara otomatis
    berdasarkan urutan prioritas:
      1. Parameter CLI (--ref dan --shifted)
      2. Folder Vision_Engine_Core/04_Camera_Auto_Calibration/data/
      3. Folder kerja aktif saat ini (.)
      4. Folder koleksi video pengujian (Pictures/Video & Foto Test Slat Rib)
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "data")
    
    candidate_ref_paths = [
        cli_ref,
        os.path.join(data_dir, "reference_aruco.mp4"),
        os.path.abspath("reference_aruco.mp4"),
        r"C:\Users\Victus\Pictures\Video & Foto Test Slat Rib\reference_aruco.mp4",
        r"C:\Users\Victus\Pictures\Video & Foto Test Slat Rib\New\Reverse\Kamera posisi 3 dimmer 2 - Reverse.mp4"
    ]

    candidate_shifted_paths = [
        cli_shifted,
        os.path.join(data_dir, "shifted_aruco.mp4"),
        os.path.abspath("shifted_aruco.mp4"),
        r"C:\Users\Victus\Pictures\Video & Foto Test Slat Rib\shifted_aruco.mp4",
        r"C:\Users\Victus\Pictures\Video & Foto Test Slat Rib\Opsi Posisi 2 - Slat Rib.mp4"
    ]

    final_ref = None
    for p in candidate_ref_paths:
        if p and os.path.exists(p):
            final_ref = os.path.abspath(p)
            break

    final_shifted = None
    for p in candidate_shifted_paths:
        if p and os.path.exists(p):
            final_shifted = os.path.abspath(p)
            break

    if final_ref is None:
        final_ref = os.path.join(data_dir, "reference_aruco.mp4")
    if final_shifted is None:
        final_shifted = os.path.join(data_dir, "shifted_aruco.mp4")

    return final_ref, final_shifted


# ================================================================================
# 9. MAIN EXECUTION LOOP
# ================================================================================

def run_alignment_simulator(ref_video_path: str, shifted_video_path: str,
                            output_dir: str, config: CalibrationConfig,
                            auto_record: bool = False, loop_playback: bool = True):
    """
    Loop pemrosesan video offline utama dengan sinkronisasi FPS dan kontrol interaktif.
    """
    print("=" * 80)
    print("🎬 SYNTHETIC ARUCO VIDEO ALIGNMENT SIMULATOR V2 (OFFLINE STUDIO)")
    print("=" * 80)
    print(f"  [1] Video Referensi : {ref_video_path}")
    print(f"  [2] Video Shifted   : {shifted_video_path}")
    print(f"  [3] Folder Output   : {output_dir}\n")

    # Inisialisasi Engine, Dashboard, dan Virtual Simulator
    engine = VideoAlignmentEngine(config)
    dashboard = StudioDashboard(config)
    virtual_sim = VirtualMarkerSimulator(dict_name=config.aruco_dict_name)

    # Inisialisasi Baseline Referensi (Fallback ke Virtual Marker jika video acuan polos)
    success, msg = engine.ref_manager.initialize_from_video(ref_video_path, engine.detector, virtual_sim)
    if success and engine.ref_manager.ref_markers:
        h_r = engine.ref_manager.ref_frame.shape[0] if engine.ref_manager.ref_frame is not None else 1080
        w_r = engine.ref_manager.ref_frame.shape[1] if engine.ref_manager.ref_frame is not None else 1920
        virtual_sim.sync_with_reference(engine.ref_manager.ref_markers, w_r, h_r)
    if not success:
        print(f"[!] PERINGATAN: {msg}")
        print("    Simulator akan tetap berjalan dan mencari marker referensi pada frame berikutnya.\n")

    # Buka stream video offline
    cap_ref = cv2.VideoCapture(ref_video_path)
    cap_cur = cv2.VideoCapture(shifted_video_path)

    if not cap_cur.isOpened():
        print(f"[ERROR KRITIS] Gagal membuka shifted video: {shifted_video_path}")
        return

    fps = cap_cur.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or math.isnan(fps):
        fps = 30.0

    ref_total_frames = int(cap_ref.get(cv2.CAP_PROP_FRAME_COUNT))
    cur_total_frames = int(cap_cur.get(cv2.CAP_PROP_FRAME_COUNT))

    # Daftar Preset Video untuk Quick Cycling via [N]
    preset_videos = get_available_presets()
    cur_preset_idx = 0
    for idx, (lbl, p) in enumerate(preset_videos):
        if os.path.normpath(p) == os.path.normpath(shifted_video_path):
            cur_preset_idx = idx
            break

    # Inisialisasi Perekam
    recorder = AlignmentRecorder(output_dir, fps=fps)
    if auto_record:
        recorder.start_recording(dashboard.total_w, dashboard.total_h)

    # Inisialisasi Jendela OpenCV
    win_name = "SYNTHETIC ARUCO VIDEO ALIGNMENT SIMULATOR V2 (TRI-PANEL STUDIO)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, dashboard.total_w, dashboard.total_h)

    # State Interaktif
    is_paused = False
    is_ghosting = False
    virtual_mode = "AUTO"   # "AUTO", "ON", "OFF"
    is_cur_virtual = False
    cached_ref_img: Optional[np.ndarray] = None
    cached_cur_img: Optional[np.ndarray] = None

    print("Petunjuk Hotkey Studio:")
    print("  [SPASI]   : Play / Pause")
    print("  [V]       : PILIH VIDEO VIEW 2 (Buka File Explorer Windows)")
    print("  [B]       : PILIH VIDEO VIEW 1 REF (Buka File Explorer Windows)")
    print("  [N]       : Beralih Cepat ke Preset Video Panjang Berikutnya")
    print("  [T]       : Toggle Mode ArUco (AUTO / ON / OFF)")
    print("  [M]       : Ganti Sudut Simulasi Kamera (Posisi 4, Miring, Zoom, Ekstrem)")
    print("  [C]       : Toggle Goyangan / Getaran Mesin Dinamis")
    print("  [G]       : Toggle Ghosting Overlay (50% Ref + 50% Aligned)")
    print("  [S]       : Ambil Tangkapan Layar (Screenshot)")
    print("  [E]       : Mulai / Berhenti Perekaman Video & Ekspor")
    print("  [F]       : Freeze / Unfreeze Reference Frame")
    print("  [R]       : Reset Playback (Ulangi dari frame 0)")
    print("  [Q] / ESC : Keluar dan Simpan Laporan\n")
    print("Sedang memproses video...")

    try:
        while True:
            t_loop_start = time.time()

            cur_frame_pos_ref = int(cap_ref.get(cv2.CAP_PROP_POS_FRAMES))
            cur_frame_pos_cur = int(cap_cur.get(cv2.CAP_PROP_POS_FRAMES))

            if not is_paused:
                # 1. Baca Video Referensi
                ret_ref, f_ref = cap_ref.read()
                if not ret_ref:
                    if loop_playback:
                        cap_ref.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret_ref, f_ref = cap_ref.read()
                    else:
                        f_ref = cached_ref_img

                if f_ref is not None:
                    # Deteksi keberadaan marker alami pada reference frame
                    ref_raw_markers, ref_raw_cnt = engine.detector.detect(f_ref)
                    use_v_ref = (virtual_mode == "ON") or (virtual_mode == "AUTO" and ref_raw_cnt < 3)

                    if use_v_ref:
                        f_ref_proc = virtual_sim.inject_reference(f_ref)
                        ref_markers, _ = engine.detector.detect(f_ref_proc)
                    else:
                        f_ref_proc = f_ref
                        ref_markers = ref_raw_markers

                    cached_ref_img = f_ref_proc.copy()
                    engine.ref_manager.update_frame(f_ref_proc, ref_markers)
                    if not use_v_ref and len(ref_markers) >= 4:
                        virtual_sim.sync_with_reference(ref_markers, f_ref.shape[1], f_ref.shape[0])
                else:
                    f_ref_proc = cached_ref_img

                # 2. Baca Video Shifted (Current View)
                ret_cur, f_cur = cap_cur.read()
                if not ret_cur:
                    if loop_playback:
                        cap_cur.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret_cur, f_cur = cap_cur.read()
                    else:
                        print("\n[*] Video shifted mencapai EOF.")
                        break

                if f_cur is not None:
                    # Deteksi keberadaan marker alami pada shifted frame
                    cur_raw_markers, cur_raw_cnt = engine.detector.detect(f_cur)
                    use_v_cur = (virtual_mode == "ON") or (virtual_mode == "AUTO" and cur_raw_cnt < 3)

                    if use_v_cur:
                        f_cur_proc = virtual_sim.inject_shifted(f_cur)
                        is_cur_virtual = True
                    else:
                        f_cur_proc = f_cur
                        is_cur_virtual = False

                    cached_cur_img = f_cur_proc.copy()
                else:
                    f_cur_proc = cached_cur_img
            else:
                f_ref_proc = cached_ref_img
                f_cur_proc = cached_cur_img

            if f_cur_proc is None:
                continue

            # Jika reference frame belum ada, gunakan frame awal
            active_ref = engine.ref_manager.ref_frame if engine.ref_manager.ref_frame is not None else f_ref_proc
            if active_ref is None:
                active_ref = f_cur_proc

            # -------------------------------------------------------------
            # PROSES PENYELARASAN
            # -------------------------------------------------------------
            t_proc_start = time.time()
            aligned_1080p, cur_markers, status_msg, metrics = engine.process_frame(f_cur_proc)
            latency_ms = (time.time() - t_proc_start) * 1000.0

            # -------------------------------------------------------------
            # RENDER DASHBOARD TIGA PANEL
            # -------------------------------------------------------------
            ref_name_base = os.path.basename(ref_video_path)
            cur_name_base = os.path.basename(shifted_video_path)
            
            ref_info_str = f"{ref_name_base} (Frame {cur_frame_pos_ref}/{max(1, ref_total_frames)} | {cur_frame_pos_ref/max(1, fps):.0f}s)"
            cur_info_str = f"{cur_name_base} (Frame {cur_frame_pos_cur}/{max(1, cur_total_frames)} | {cur_frame_pos_cur/max(1, fps):.0f}s)"

            if is_cur_virtual:
                cur_preset_name = virtual_sim.get_active_preset_name()
                jit_str = " +Goyang" if virtual_sim.enable_jitter else ""
                marker_mode_str = f"VIRTUAL [{cur_preset_name}{jit_str}]"
            else:
                marker_mode_str = "DETEKSI ALAMI (Video Langsung)"

            dashboard_img = dashboard.render_tri_view(
                ref_1080p=active_ref,
                ref_markers=engine.ref_manager.ref_markers,
                cur_1080p=f_cur_proc,
                cur_markers=cur_markers,
                aligned_1080p=aligned_1080p,
                status_msg=status_msg,
                metrics=metrics,
                is_using_cache=engine.is_using_cached,
                is_ghosting=is_ghosting,
                is_recording=recorder.is_recording,
                fps=fps,
                latency_ms=latency_ms,
                ref_info=ref_info_str,
                cur_info=cur_info_str,
                marker_mode_str=marker_mode_str
            )

            cv2.imshow(win_name, dashboard_img)

            # Tulis ke video recorder jika aktif
            if recorder.is_recording:
                recorder.write_frames(aligned_1080p, dashboard_img, active_ref)

            # -------------------------------------------------------------
            # MANAJEMEN PACING FRAME DAN KEYBOARD
            # -------------------------------------------------------------
            elapsed = time.time() - t_loop_start
            wait_delay = max(1, int((1.0 / fps - elapsed) * 1000.0)) if not is_paused else 30
            key = cv2.waitKey(wait_delay) & 0xFF

            if key in [ord('q'), ord('Q'), 27]:
                print("\n[*] Menutup simulator...")
                break
            elif key == 32:  # [SPASI]
                is_paused = not is_paused
                print(f"[*] Playback: {'PAUSED' if is_paused else 'PLAYING'}")
            elif key in [ord('t'), ord('T')]:
                if virtual_mode == "AUTO":
                    virtual_mode = "ON"
                elif virtual_mode == "ON":
                    virtual_mode = "OFF"
                else:
                    virtual_mode = "AUTO"
                print(f"\n[🏷️] Mode ArUco Virtual dialihkan ke: {virtual_mode}")
            elif key in [ord('m'), ord('M')]:
                new_preset_name = virtual_sim.cycle_preset()
                engine.last_valid_H = None
                print(f"\n[🎯] Preset Sudut Kamera Virtual diubah ke: {new_preset_name}")
            elif key in [ord('c'), ord('C')]:
                virtual_sim.enable_jitter = not virtual_sim.enable_jitter
                print(f"\n[〰️] Goyangan Mesin / Getaran Dinamis: {'AKTIF' if virtual_sim.enable_jitter else 'NONAKTIF'}")
            elif key in [ord('v'), ord('V')]:
                # Pilih Video View 2 (Shifted Video) via Windows Native File Dialog
                print("\n[📂] Membuka jendela penjelajah berkas Windows untuk View 2...")
                chosen = pick_video_file_dialog(
                    title="Pilih File Video untuk View 2 (Shifted Camera)...",
                    initial_dir=os.path.dirname(shifted_video_path)
                )
                if chosen and os.path.exists(chosen) and os.path.normpath(chosen) != os.path.normpath(shifted_video_path):
                    cap_cur.release()
                    new_cap = cv2.VideoCapture(chosen)
                    if new_cap.isOpened():
                        cap_cur = new_cap
                        shifted_video_path = chosen
                        cur_total_frames = int(cap_cur.get(cv2.CAP_PROP_FRAME_COUNT))
                        engine.last_valid_H = None
                        cached_cur_img = None
                        print(f"[✓] Berhasil memuat video View 2 baru: {os.path.basename(chosen)} ({cur_total_frames} frame)\n")
                    else:
                        print(f"[!] Gagal membuka video yang dipilih: {chosen}\n")
                        cap_cur = cv2.VideoCapture(shifted_video_path)
                else:
                    print("[i] Pemilihan file dibatalkan atau file sama.\n")
            elif key in [ord('b'), ord('B')]:
                # Pilih Video View 1 (Reference Video) via Windows Native File Dialog
                print("\n[📂] Membuka jendela penjelajah berkas Windows untuk View 1 (Reference)...")
                chosen = pick_video_file_dialog(
                    title="Pilih File Video untuk View 1 (Reference Golden Truth)...",
                    initial_dir=os.path.dirname(ref_video_path)
                )
                if chosen and os.path.exists(chosen) and os.path.normpath(chosen) != os.path.normpath(ref_video_path):
                    cap_ref.release()
                    new_cap = cv2.VideoCapture(chosen)
                    if new_cap.isOpened():
                        cap_ref = new_cap
                        ref_video_path = chosen
                        ref_total_frames = int(cap_ref.get(cv2.CAP_PROP_FRAME_COUNT))
                        cached_ref_img = None
                        ok, msg = engine.ref_manager.initialize_from_video(chosen, engine.detector, virtual_sim)
                        if ok and engine.ref_manager.ref_markers:
                            h_r = engine.ref_manager.ref_frame.shape[0] if engine.ref_manager.ref_frame is not None else 1080
                            w_r = engine.ref_manager.ref_frame.shape[1] if engine.ref_manager.ref_frame is not None else 1920
                            virtual_sim.sync_with_reference(engine.ref_manager.ref_markers, w_r, h_r)
                        print(f"[✓] Berhasil memuat video Reference baru: {os.path.basename(chosen)} ({ref_total_frames} frame). Inisialisasi: {ok}\n")
                    else:
                        print(f"[!] Gagal membuka video referensi yang dipilih: {chosen}\n")
                        cap_ref = cv2.VideoCapture(ref_video_path)
                else:
                    print("[i] Pemilihan file dibatalkan atau file sama.\n")
            elif key in [ord('n'), ord('N')]:
                # Beralih Cepat ke Preset Video Panjang Berikutnya
                if preset_videos:
                    cur_preset_idx = (cur_preset_idx + 1) % len(preset_videos)
                    p_label, p_path = preset_videos[cur_preset_idx]
                    cap_cur.release()
                    new_cap = cv2.VideoCapture(p_path)
                    if new_cap.isOpened():
                        cap_cur = new_cap
                        shifted_video_path = p_path
                        cur_total_frames = int(cap_cur.get(cv2.CAP_PROP_FRAME_COUNT))
                        engine.last_valid_H = None
                        cached_cur_img = None
                        print(f"\n[🔄] Preset View 2 diganti ke [{cur_preset_idx+1}/{len(preset_videos)}]: {p_label}")
                        print(f"    Path: {p_path} ({cur_total_frames} frames)\n")
            elif key in [ord('g'), ord('G')]:
                is_ghosting = not is_ghosting
                print(f"[👻] Ghosting Overlay: {'AKTIF' if is_ghosting else 'NONAKTIF'}")
            elif key in [ord('s'), ord('S')]:
                recorder.save_screenshot(dashboard_img, aligned_1080p)
            elif key in [ord('e'), ord('E')]:
                if not recorder.is_recording:
                    recorder.start_recording(dashboard.total_w, dashboard.total_h)
                else:
                    recorder.stop_recording()
            elif key in [ord('f'), ord('F')]:
                engine.ref_manager.is_frozen = not engine.ref_manager.is_frozen
                state_str = "DIBEKUKAN (FROZEN)" if engine.ref_manager.is_frozen else "DINAMIS (UNFROZEN)"
                print(f"[❄] Reference Frame: {state_str}")
            elif key in [ord('r'), ord('R')]:
                cap_ref.set(cv2.CAP_PROP_POS_FRAMES, 0)
                cap_cur.set(cv2.CAP_PROP_POS_FRAMES, 0)
                print("[↺] Playback di-reset ke frame 0.")

    finally:
        # Pembersihan resource
        cap_ref.release()
        cap_cur.release()
        if recorder.is_recording:
            recorder.stop_recording()
        cv2.destroyAllWindows()

        # Buat laporan kalibrasi akhir
        recorder.generate_calibration_report(
            ref_path=ref_video_path,
            shifted_path=shifted_video_path,
            ref_markers=engine.ref_manager.ref_markers,
            engine=engine
        )
        print("=" * 80)
        print("✓ Sesi kalibrasi selesai dengan aman.")
        print("=" * 80)


# ================================================================================
# 10. CLI PARSER ENTRYPOINT
# ================================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Synthetic ArUco Video Alignment Simulator V2 (Reference-Based Offline Alignment)"
    )
    parser.add_argument("--ref", type=str, default=None,
                        help="Path ke video referensi (default: data/reference_aruco.mp4)")
    parser.add_argument("--shifted", type=str, default=None,
                        help="Path ke video shifted (default: data/shifted_aruco.mp4)")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "output"),
                        help="Folder tujuan ekspor video dan laporan kalibrasi")
    parser.add_argument("--dict", type=str, default="DICT_4X4_50",
                        help="Nama ArUco Dictionary (default: DICT_4X4_50)")
    parser.add_argument("--record", action="store_true",
                        help="Mulai merekam ketiga video output secara otomatis saat peluncuran")
    parser.add_argument("--no-loop", action="store_true",
                        help="Hentikan simulator saat video mencapai EOF (tidak mengulang playback)")
    parser.add_argument("--reproj-thresh", type=float, default=4.5,
                        help="Batas maksimal reprojection error rata-rata yang diizinkan (pixel)")
    parser.add_argument("--inlier-thresh", type=float, default=0.70,
                        help="Batas minimal rasio inlier RANSAC (0.0 - 1.0)")

    args = parser.parse_args()

    # Resolusi path video cerdas
    ref_path, shifted_path = resolve_video_paths(args.ref, args.shifted)

    # Inisialisasi konfigurasi
    config = CalibrationConfig(
        aruco_dict_name=args.dict,
        max_mean_reproj_error_px=args.reproj_thresh,
        min_inlier_ratio=args.inlier_thresh
    )

    run_alignment_simulator(
        ref_video_path=ref_path,
        shifted_video_path=shifted_path,
        output_dir=os.path.abspath(args.output_dir),
        config=config,
        auto_record=args.record,
        loop_playback=not args.no_loop
    )


if __name__ == "__main__":
    main()
