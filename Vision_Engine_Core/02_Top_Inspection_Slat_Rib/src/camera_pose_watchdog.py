"""
Hybrid GPU + ArUco Continuous Camera Pose & Holder Vibration Watchdog.

Priority 1: Fiducial Marker Lock via cv2.aruco (DICT_4X4_50, IDs 0..3) when >= 2 markers are present.
Priority 2: Markerless Stationary 4-Quadrant Side-Rail Structural Edge Registration running in CUDA VRAM
            (9-frame temporal median ring buffer + Sobel edge zero-padded cuFFT Normalized Cross-Correlation
             with rigid-pose outlier rejection across 4 rail quadrants).

3-Zone Geometric Policy (normalized to 720p = 1280px reference width):
  - GREEN  (STABLE)       : |dx| <= 6 px, |d_theta| <= 0.6 deg -> zero extra warp
  - AMBER  (MICRO_DRIFT)  : 6 < |dx| <= 24 px or 0.6 < |d_theta| <= 2.5 deg -> smooth EMA 3x3 corrective warp
  - RED    (CAMERA_SHIFT) : |dx| > 28 px or |d_theta| > 3.0 deg or |dw/w| > 12% -> INVALID: CAMERA_SHIFT
"""

from __future__ import annotations

import math
import threading
from typing import Any

import cv2
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    torch = None
    F = None
    _HAS_TORCH = False


def detect_aruco_landmarks(frame_bgr: np.ndarray | None) -> dict[int, list[float]]:
    """Detect DICT_4X4_50 ArUco marker centers (IDs 0..15) returning {marker_id: [cx, cy]}."""
    if frame_bgr is None or frame_bgr.size == 0 or not hasattr(cv2, "aruco"):
        return {}
    try:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        params = cv2.aruco.DetectorParameters()
        if hasattr(cv2.aruco, "ArucoDetector"):
            detector = cv2.aruco.ArucoDetector(aruco_dict, params)
            corners, ids, _ = detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
        if ids is None or len(ids) == 0:
            return {}
        out: dict[int, list[float]] = {}
        for c_arr, m_id_arr in zip(corners, ids):
            m_id = int(m_id_arr[0])
            pts = np.asarray(c_arr, dtype=np.float32).reshape(-1, 2)
            cx = float(np.mean(pts[:, 0]))
            cy = float(np.mean(pts[:, 1]))
            out[m_id] = [round(cx, 2), round(cy, 2)]
        return out
    except Exception:
        return {}


class CameraPoseWatchdog:
    """Periodic GPU VRAM + ArUco camera drift & knock detector with 3-zone response."""

    THUMB_W = 320
    THUMB_H = 180
    RING_SIZE = 9
    FFT_PAD = 128
    # 4 stationary rail/frame quadrants on 320x180 thumbnail: (x1, y1, x2, y2) -> each 80x64
    QUAD_ROIS = (
        (2, 22, 82, 86),     # TL
        (238, 22, 318, 86),  # TR
        (238, 88, 318, 152), # BR
        (2, 88, 82, 152),    # BL
    )
    MAX_LAG_X = 22  # +-22 thumb px = +-88 px @720p
    MAX_LAG_Y = 14  # +-14 thumb px = +-56 px @720p

    def __init__(
        self,
        device: str | None = "auto",
        *,
        sample_every: int = 4,
        eval_every: int = 30,
        confirm_checks_amber: int = 2,
        confirm_checks_red: int = 3,
        green_dx_px_720: float = 6.0,
        amber_max_dx_px_720: float = 24.0,
        red_dx_px_720: float = 28.0,
        green_tilt_deg: float = 0.6,
        amber_max_tilt_deg: float = 2.5,
        red_tilt_deg: float = 3.0,
        red_width_ratio: float = 0.12,
        min_ncc_conf: float = 0.34,
        async_eval: bool = False,
    ):
        use_cuda = False
        if _HAS_TORCH and str(device or "auto").lower() != "cpu" and torch.cuda.is_available():
            use_cuda = True
        self.use_cuda = use_cuda
        self.torch_device = torch.device("cuda:0") if use_cuda else None
        self.async_eval = bool(async_eval)
        self._state_lock = threading.Lock()
        self._ring_lock = threading.Lock()
        self._generation = 0
        self._eval_running = False
        self._eval_thread: threading.Thread | None = None
        self._closed = False

        self.sample_every = max(1, int(sample_every))
        self.eval_every = max(5, int(eval_every))
        self.confirm_checks_amber = max(1, int(confirm_checks_amber))
        self.confirm_checks_red = max(2, int(confirm_checks_red))

        self.green_dx_px_720 = float(green_dx_px_720)
        self.amber_max_dx_px_720 = float(amber_max_dx_px_720)
        self.red_dx_px_720 = float(red_dx_px_720)
        self.green_tilt_deg = float(green_tilt_deg)
        self.amber_max_tilt_deg = float(amber_max_tilt_deg)
        self.red_tilt_deg = float(red_tilt_deg)
        self.red_width_ratio = float(red_width_ratio)
        self.min_ncc_conf = float(min_ncc_conf)

        if self.use_cuda:
            kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=self.torch_device).view(1, 1, 3, 3)
            ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=self.torch_device).view(1, 1, 3, 3)
            self._sobel_xy = torch.cat([kx, ky], dim=0)
            self._ring_gpu = torch.zeros((self.RING_SIZE, self.THUMB_H, self.THUMB_W), dtype=torch.float32, device=self.torch_device)
        else:
            self._sobel_xy = None
            self._ring_cpu = np.zeros((self.RING_SIZE, self.THUMB_H, self.THUMB_W), dtype=np.float32)

        self.configured_aruco_ref: dict[int, list[float]] = {}
        self.configured_aruco_size: tuple[int, int] = (1280, 720)
        self.reset_reference()

    def set_aruco_reference(self, landmarks: dict[Any, Any] | None, calib_w: int = 1280, calib_h: int = 720) -> None:
        """Load optional fiducial landmark coordinates from calibration_profile['aruco_landmarks']."""
        self.configured_aruco_ref = {}
        if isinstance(landmarks, dict):
            for k, v in landmarks.items():
                try:
                    if isinstance(v, (list, tuple)) and len(v) >= 2:
                        self.configured_aruco_ref[int(k)] = [float(v[0]), float(v[1])]
                except (TypeError, ValueError):
                    continue
        self.configured_aruco_size = (max(1, int(calib_w)), max(1, int(calib_h)))
        self._session_aruco_ref = dict(self.configured_aruco_ref) if len(self.configured_aruco_ref) >= 2 else None
        self._session_aruco_size = self.configured_aruco_size

    def reset_reference(self) -> None:
        """Reset locked session baseline pose and corrective homography (e.g. after [K] recalibration)."""
        with self._state_lock:
            self._generation += 1
            self._frame_counter = 0
            with self._ring_lock:
                self._ring_write_idx = 0
                self._ring_count = 0
            self._ref_quad_fft = None
            self._ref_quad_norms = None
            self._session_aruco_ref: dict[int, list[float]] | None = (
                dict(self.configured_aruco_ref) if len(self.configured_aruco_ref) >= 2 else None
            )
            self._session_aruco_size: tuple[int, int] = self.configured_aruco_size
            self._amber_streak = 0
            self._red_streak = 0
            self.corrective_matrix: np.ndarray | None = None
            self._corrective_size: tuple[int, int] | None = None
            self.last_status: dict[str, Any] = {
                "zone": "STANDBY",
                "mode": "STANDBY",
                "is_invalid": False,
                "invalid_code": None,
                "invalid_reason": "",
                "dx_px_720": 0.0,
                "dy_px_720": 0.0,
                "d_theta_deg": 0.0,
                "width_ratio_delta": 0.0,
                "has_correction": False,
            }

    def wait_pending_eval(self, timeout: float = 2.0) -> None:
        """Wait for an in-flight asynchronous pose evaluation thread if active."""
        th = self._eval_thread
        if th is not None and th.is_alive():
            th.join(timeout=timeout)

    def close(self) -> None:
        """Stop accepting new async evaluations and wait briefly for any running check."""
        self._closed = True
        self.wait_pending_eval(timeout=1.0)

    def get_scaled_corrective_matrix(self, w_f: int, h_f: int) -> np.ndarray | None:
        """Return 3x3 corrective matrix scaled to (w_f, h_f) when MICRO_DRIFT compensation is active."""
        with self._state_lock:
            cmat = self.corrective_matrix
            csize = self._corrective_size
        if cmat is None or csize is None:
            return None
        cw, ch = csize
        if int(w_f) == int(cw) and int(h_f) == int(ch):
            return cmat
        sx = float(w_f) / float(max(1, cw))
        sy = float(h_f) / float(max(1, ch))
        S = np.diag([sx, sy, 1.0])
        S_inv = np.diag([1.0 / sx, 1.0 / sy, 1.0])
        return (S @ cmat.astype(np.float64) @ S_inv).astype(np.float32)

    def _push_ring_sample(self, frame_bgr: np.ndarray, gpu_thumb=None) -> None:
        with self._ring_lock:
            slot = self._ring_write_idx % self.RING_SIZE
            if self.use_cuda:
                with torch.inference_mode():
                    if gpu_thumb is not None and getattr(gpu_thumb, "is_cuda", False):
                        luma = 0.114 * gpu_thumb[0, 0] + 0.587 * gpu_thumb[0, 1] + 0.299 * gpu_thumb[0, 2]
                    else:
                        h_f, w_f = frame_bgr.shape[:2]
                        sy = max(1, h_f // 360)
                        sx = max(1, w_f // 640)
                        sub = np.ascontiguousarray(frame_bgr[::sy, ::sx])
                        t = torch.from_numpy(sub).to(self.torch_device, non_blocking=True).permute(2, 0, 1).unsqueeze(0).float()
                        th = F.interpolate(t, size=(self.THUMB_H, self.THUMB_W), mode="bilinear", align_corners=False)
                        luma = 0.114 * th[0, 0] + 0.587 * th[0, 1] + 0.299 * th[0, 2]
                    self._ring_gpu[slot].copy_(luma)
            else:
                small = cv2.resize(frame_bgr, (self.THUMB_W, self.THUMB_H), interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
                self._ring_cpu[slot] = gray
            self._ring_write_idx += 1
            self._ring_count = min(self.RING_SIZE, self._ring_count + 1)

    @staticmethod
    def _subpixel_parabola(vm1: float, v0: float, vp1: float) -> float:
        denom = 2.0 * (vm1 - 2.0 * v0 + vp1)
        if abs(denom) < 1e-6:
            return 0.0
        return float(max(-0.5, min(0.5, (vm1 - vp1) / denom)))

    def _extract_quad_fft_and_norms(self) -> tuple[Any, np.ndarray] | None:
        """Compute structural Sobel edge map of temporal median and return zero-padded 4-quadrant rfft2 + L2 norms."""
        if self._ring_count < 5:
            return None

        pad_sz = self.FFT_PAD
        if self.use_cuda:
            with torch.inference_mode():
                med_luma = torch.median(self._ring_gpu[: self._ring_count], dim=0).values
                grads = F.conv2d(med_luma[None, None], self._sobel_xy, padding=1)[0]
                # Suppress pixel grain (< 42 Sobel mag) so only structural rails/brackets drive registration
                edge = F.relu(torch.sqrt(grads[0] * grads[0] + grads[1] * grads[1] + 1e-6) - 42.0)
                edge = F.avg_pool2d(edge[None, None], kernel_size=3, stride=1, padding=1)[0, 0]

                padded = torch.zeros((4, pad_sz, pad_sz), dtype=torch.float32, device=self.torch_device)
                norms = np.zeros(4, dtype=np.float32)
                for idx, (x1, y1, x2, y2) in enumerate(self.QUAD_ROIS):
                    q = edge[y1:y2, x1:x2]
                    q_zm = q - q.mean()
                    n_val = float(torch.linalg.norm(q_zm).item())
                    norms[idx] = n_val
                    if n_val > 80.0:
                        padded[idx, : (y2 - y1), : (x2 - x1)] = q_zm
                if int(np.sum(norms > 80.0)) < 3:
                    return None
                q_fft = torch.fft.rfft2(padded)
                return q_fft, norms
        else:
            med_luma_np = np.median(self._ring_cpu[: self._ring_count], axis=0).astype(np.float32)
            gx = cv2.Sobel(med_luma_np, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(med_luma_np, cv2.CV_32F, 0, 1, ksize=3)
            edge_np = np.maximum(0.0, np.sqrt(gx * gx + gy * gy + 1e-6) - 42.0)
            edge_np = cv2.boxFilter(edge_np, -1, (3, 3), normalize=True)

            padded_np = np.zeros((4, pad_sz, pad_sz), dtype=np.float32)
            norms = np.zeros(4, dtype=np.float32)
            for idx, (x1, y1, x2, y2) in enumerate(self.QUAD_ROIS):
                q = edge_np[y1:y2, x1:x2]
                q_zm = q - float(np.mean(q))
                n_val = float(np.linalg.norm(q_zm))
                norms[idx] = n_val
                if n_val > 80.0:
                    padded_np[idx, : (y2 - y1), : (x2 - x1)] = q_zm
            if int(np.sum(norms > 80.0)) < 3:
                return None
            q_fft_np = np.fft.rfft2(padded_np)
            return q_fft_np, norms

    def _match_quadrants_to_reference(self, curr_fft, curr_norms: np.ndarray) -> list[tuple[int, float, float, float]]:
        """Match 4 stationary side-rail quadrants via zero-padded linear NCC; returns [(idx, dx_thumb, dy_thumb, ncc)]."""
        pad_sz = self.FFT_PAD
        if self.use_cuda:
            with torch.inference_mode():
                corr_t = torch.fft.irfft2(curr_fft * torch.conj(self._ref_quad_fft), s=(pad_sz, pad_sz))
                corr_np = np.fft.fftshift(corr_t.cpu().numpy(), axes=(1, 2))
        else:
            corr_raw = np.fft.irfft2(curr_fft * np.conj(self._ref_quad_fft), s=(pad_sz, pad_sz)).astype(np.float32)
            corr_np = np.fft.fftshift(corr_raw, axes=(1, 2))

        c0 = pad_sz // 2
        y_lo, y_hi = c0 - self.MAX_LAG_Y, c0 + self.MAX_LAG_Y + 1
        x_lo, x_hi = c0 - self.MAX_LAG_X, c0 + self.MAX_LAG_X + 1

        matches: list[tuple[int, float, float, float]] = []
        for idx in range(4):
            denom = float(curr_norms[idx]) * float(self._ref_quad_norms[idx])
            if denom <= 6400.0:
                continue
            win = corr_np[idx, y_lo:y_hi, x_lo:x_hi] / denom
            flat_i = int(np.argmax(win))
            py, px = divmod(flat_i, win.shape[1])
            pk_ncc = float(win[py, px])
            if pk_ncc < self.min_ncc_conf:
                continue
            sub_x = 0.0
            sub_y = 0.0
            if 0 < px < (win.shape[1] - 1):
                sub_x = self._subpixel_parabola(float(win[py, px - 1]), pk_ncc, float(win[py, px + 1]))
            if 0 < py < (win.shape[0] - 1):
                sub_y = self._subpixel_parabola(float(win[py - 1, px]), pk_ncc, float(win[py + 1, px]))
            dx_th = float(px - self.MAX_LAG_X) + sub_x
            dy_th = float(py - self.MAX_LAG_Y) + sub_y
            matches.append((idx, dx_th, dy_th, pk_ncc))
        return matches

    def _apply_3zone_policy(
        self,
        *,
        mode: str,
        dx_720: float,
        dy_720: float,
        d_theta_deg: float,
        width_ratio_delta: float,
        cand_H_full: np.ndarray | None,
        w_f: int,
        h_f: int,
        immediate_red: bool = False,
    ) -> dict[str, Any]:
        abs_dx = max(abs(dx_720), abs(dy_720))
        abs_th = abs(d_theta_deg)
        abs_wr = abs(width_ratio_delta)

        is_red_cand = (
            abs_dx > self.red_dx_px_720
            or abs_th > self.red_tilt_deg
            or abs_wr > self.red_width_ratio
        )
        is_amber_cand = (
            not is_red_cand
            and (abs_dx > self.green_dx_px_720 or abs_th > self.green_tilt_deg)
            and abs_dx <= self.amber_max_dx_px_720
            and abs_th <= self.amber_max_tilt_deg
        )

        if is_red_cand:
            self._red_streak += 1
            self._amber_streak = 0
        elif is_amber_cand:
            self._amber_streak += 1
            self._red_streak = 0
        else:
            self._amber_streak = 0
            self._red_streak = 0
            if self.corrective_matrix is not None and abs_dx <= (0.5 * self.green_dx_px_720) and abs_th <= 0.3:
                self.corrective_matrix = None
                self._corrective_size = None

        req_red = max(1, self.confirm_checks_red - 1) if immediate_red else self.confirm_checks_red
        if self._red_streak >= req_red:
            zone = "CAMERA_SHIFT"
            is_invalid = True
            inv_code = "CAMERA_SHIFT"
            inv_reason = (
                f"INVALID: Posisi Kamera Bergeser (Geser={dx_720:+.1f}px, Sudut={d_theta_deg:+.2f}deg) "
                f"- Cek Holder / Tekan [K]"
            )
        elif self._amber_streak >= self.confirm_checks_amber and cand_H_full is not None:
            zone = "MICRO_DRIFT"
            is_invalid = False
            inv_code = None
            inv_reason = ""
            if self.corrective_matrix is None or self._corrective_size != (w_f, h_f):
                self.corrective_matrix = cand_H_full.astype(np.float32)
                self._corrective_size = (w_f, h_f)
            else:
                self.corrective_matrix = (0.55 * self.corrective_matrix + 0.45 * cand_H_full).astype(np.float32)
                self.corrective_matrix /= max(1e-6, float(self.corrective_matrix[2, 2]))
        else:
            zone = "MICRO_DRIFT" if self.corrective_matrix is not None else "STABLE"
            is_invalid = False
            inv_code = None
            inv_reason = ""

        self.last_status = {
            "zone": zone,
            "mode": mode,
            "is_invalid": is_invalid,
            "invalid_code": inv_code,
            "invalid_reason": inv_reason,
            "dx_px_720": round(dx_720, 2),
            "dy_px_720": round(dy_720, 2),
            "d_theta_deg": round(d_theta_deg, 2),
            "width_ratio_delta": round(width_ratio_delta, 4),
            "has_correction": self.corrective_matrix is not None,
        }
        return self.last_status

    def update(self, frame_bgr: np.ndarray | None, gpu_thumb=None, force_eval: bool = False) -> dict[str, Any]:
        """Feed one raw camera frame; evaluates pose periodically every eval_every frames."""
        if frame_bgr is None or frame_bgr.size == 0:
            return self.last_status

        h_f, w_f = frame_bgr.shape[:2]
        self._frame_counter += 1

        if (self._frame_counter % self.sample_every) == 1 or self.sample_every == 1 or force_eval:
            self._push_ring_sample(frame_bgr, gpu_thumb=gpu_thumb)

        if not force_eval and (self._frame_counter % self.eval_every) != 0:
            return self.last_status

        if self.async_eval and not force_eval:
            with self._state_lock:
                if self._eval_running or self._closed:
                    return self.last_status
                self._eval_running = True
                gen = self._generation
            frame_snap = frame_bgr.copy()

            def _bg_worker():
                try:
                    self._evaluate_pose_core(frame_snap, h_f, w_f, expected_gen=gen)
                except Exception:
                    pass
                finally:
                    with self._state_lock:
                        self._eval_running = False

            th = threading.Thread(target=_bg_worker, daemon=True, name="Thread-PoseWatchdog")
            self._eval_thread = th
            th.start()
            return self.last_status

        return self._evaluate_pose_core(frame_bgr, h_f, w_f, expected_gen=self._generation)

    def _evaluate_pose_core(self, frame_bgr: np.ndarray, h_f: int, w_f: int, expected_gen: int) -> dict[str, Any]:
        # Priority 1: ArUco Fiducial Landmarks (when markers exist in frame or config)
        markers = detect_aruco_landmarks(frame_bgr)
        with self._state_lock:
            if expected_gen != self._generation:
                return self.last_status
        if len(markers) >= 2:
            if self._session_aruco_ref is None:
                with self._state_lock:
                    if expected_gen != self._generation:
                        return self.last_status
                    self._session_aruco_ref = dict(markers)
                    self._session_aruco_size = (w_f, h_f)
                    self.last_status = {
                        **self.last_status,
                        "zone": "STABLE",
                        "mode": "ARUCO_FIDUCIAL",
                        "is_invalid": False,
                        "invalid_code": None,
                        "invalid_reason": "",
                    }
                return self.last_status

            ref_w, ref_h = self._session_aruco_size
            sx_r = float(w_f) / float(max(1, ref_w))
            sy_r = float(h_f) / float(max(1, ref_h))
            common_ids = sorted(set(markers.keys()) & set(self._session_aruco_ref.keys()))
            if len(common_ids) >= 2:
                src_pts = np.array([markers[i] for i in common_ids], dtype=np.float32)
                dst_pts = np.array(
                    [[self._session_aruco_ref[i][0] * sx_r, self._session_aruco_ref[i][1] * sy_r] for i in common_ids],
                    dtype=np.float32,
                )
                diffs = src_pts - dst_pts
                norm_720 = 1280.0 / float(max(1, w_f))
                dx_720 = float(np.median(diffs[:, 0])) * norm_720
                dy_720 = float(np.median(diffs[:, 1])) * norm_720

                v_src = src_pts[-1] - src_pts[0]
                v_dst = dst_pts[-1] - dst_pts[0]
                ang_s = math.degrees(math.atan2(float(v_src[1]), float(v_src[0])))
                ang_d = math.degrees(math.atan2(float(v_dst[1]), float(v_dst[0])))
                d_theta = ((ang_s - ang_d + 180.0) % 360.0) - 180.0
                dist_s = float(np.hypot(v_src[0], v_src[1]))
                dist_d = max(1.0, float(np.hypot(v_dst[0], v_dst[1])))
                w_delta = (dist_s - dist_d) / dist_d

                cand_H = None
                if len(common_ids) >= 4:
                    cand_H, _ = cv2.findHomography(src_pts, dst_pts, method=0)
                else:
                    M_aff, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
                    if M_aff is not None:
                        cand_H = np.eye(3, dtype=np.float32)
                        cand_H[:2, :] = M_aff.astype(np.float32)
                with self._state_lock:
                    if expected_gen != self._generation:
                        return self.last_status
                    return self._apply_3zone_policy(
                        mode="ARUCO_FIDUCIAL",
                        dx_720=dx_720,
                        dy_720=dy_720,
                        d_theta_deg=d_theta,
                        width_ratio_delta=w_delta,
                        cand_H_full=cand_H,
                        w_f=w_f,
                        h_f=h_f,
                        immediate_red=True,
                    )

        # Priority 2: GPU VRAM 4-Quadrant Structural-Edge Registration
        mode_name = "GPU_RAIL_TRAPEZOID" if self.use_cuda else "CPU_RAIL_TRAPEZOID"
        with self._ring_lock:
            quad_pack = self._extract_quad_fft_and_norms()
        if quad_pack is None:
            return self.last_status

        q_fft, q_norms = quad_pack
        if self._ref_quad_fft is None:
            with self._state_lock:
                if expected_gen != self._generation:
                    return self.last_status
                self._ref_quad_fft = q_fft
                self._ref_quad_norms = q_norms
                self.last_status = {
                    **self.last_status,
                    "zone": "STABLE",
                    "mode": mode_name,
                    "is_invalid": False,
                    "invalid_code": None,
                    "invalid_reason": "",
                }
            return self.last_status

        matches = self._match_quadrants_to_reference(q_fft, q_norms)
        if len(matches) < 3:
            return self.last_status

        sx_full = float(w_f) / float(self.THUMB_W)
        sy_full = float(h_f) / float(self.THUMB_H)
        scale_thumb_to_720 = 1280.0 / float(self.THUMB_W)

        dst_pts = []
        src_pts = []
        dxs_720 = []
        dys_720 = []
        for idx, dx_th, dy_th, _ncc in matches:
            x1, y1, x2, y2 = self.QUAD_ROIS[idx]
            cx_f = 0.5 * float(x1 + x2) * sx_full
            cy_f = 0.5 * float(y1 + y2) * sy_full
            dst_pts.append([cx_f, cy_f])
            src_pts.append([cx_f + dx_th * sx_full, cy_f + dy_th * sy_full])
            dxs_720.append(dx_th * scale_thumb_to_720)
            dys_720.append(dy_th * scale_thumb_to_720)

        src_arr = np.asarray(src_pts, dtype=np.float32)
        dst_arr = np.asarray(dst_pts, dtype=np.float32)

        # Rigid/similarity consistency check via RANSAC/LMEDS so non-rigid belt clutter never fakes a camera shift
        M_sim, inliers = cv2.estimateAffinePartial2D(
            src_arr,
            dst_arr,
            method=cv2.RANSAC,
            ransacReprojThreshold=max(4.5, 5.5 * (float(w_f) / 1280.0)),
        )
        if M_sim is None or inliers is None or int(np.sum(inliers)) < 3:
            return self.last_status

        inl_mask = inliers.reshape(-1).astype(bool)
        dx_720 = float(np.median(np.asarray(dxs_720)[inl_mask]))
        dy_720 = float(np.median(np.asarray(dys_720)[inl_mask]))

        # Extract rotation & scale from similarity matrix M_sim (maps src -> dst, so camera shift angle is -atan2)
        cos_a = float(M_sim[0, 0])
        sin_a = float(M_sim[1, 0])
        sim_scale = math.hypot(cos_a, sin_a)
        d_theta = -math.degrees(math.atan2(sin_a, cos_a))
        w_delta = (1.0 / max(1e-4, sim_scale)) - 1.0

        cand_H = np.eye(3, dtype=np.float32)
        cand_H[:2, :] = M_sim.astype(np.float32)

        with self._state_lock:
            if expected_gen != self._generation:
                return self.last_status
            return self._apply_3zone_policy(
                mode=mode_name,
                dx_720=dx_720,
                dy_720=dy_720,
                d_theta_deg=d_theta,
                width_ratio_delta=w_delta,
                cand_H_full=cand_H,
                w_f=w_f,
                h_f=h_f,
                immediate_red=False,
            )
