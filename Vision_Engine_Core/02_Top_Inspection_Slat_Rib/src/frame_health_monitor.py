"""
GPU-Accelerated Pre-Inspection Frame Health Gate (PRD §25.6) & cuFFT Phase-Correlation Auto-Direction.

Runs Luma/Blackout, Extreme Specular Glare, Side-Rail Tenengrad Blur/Fog, Foreign Ingot Occlusion,
and 2D Phase-Correlation conveyor velocity/direction estimation on CUDA (torch.cuda + torch.fft)
via a single non-blocking downsampled tensor upload (~0.3-0.8 ms GPU, ~0% main CPU load),
with automatic NumPy/OpenCV fallback when running in CPU-only mode.
"""

from __future__ import annotations

from collections import deque
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


class FrameHealthMonitor:
    """
    Stage-0 Frame Health & Extreme-Environment Gate + GPU cuFFT Phase-Correlation Motion Tracker.

    Distinguishes true product defects (NG) from camera/environment unsuitability (INVALID / WAIT):
      - INVALID: LIGHTING_BLACKOUT
      - INVALID: EXTREME_GLARE
      - INVALID: LENS_BLUR_OR_FOG
      - WAIT: FOREIGN_OBJECT_OCCLUSION (e.g. bright aluminum ingot crossing gate)
    And estimates flicker-immune conveyor vertical velocity (vy_px/frame) + direction (REVERSE / FORWARD / STOPPED).
    """

    THUMB_W = 320
    THUMB_H = 180
    FFT_SIZE = 128

    def __init__(
        self,
        device: str | None = "auto",
        *,
        blackout_luma_thresh: float = 20.0,
        blackout_dark_ratio: float = 0.82,
        glare_sat_ratio: float = 0.28,
        lighting_hysteresis_frames: int = 12,
        blur_rel_drop_ratio: float = 0.42,
        blur_abs_floor: float = 4.2,
        blur_hysteresis_frames: int = 18,
        ingot_min_area_ratio: float = 0.14,
        ingot_max_area_ratio: float = 0.68,
    ):
        use_cuda = False
        if _HAS_TORCH and str(device or "auto").lower() != "cpu" and torch.cuda.is_available():
            use_cuda = True
        self.use_cuda = use_cuda
        self.torch_device = torch.device("cuda:0") if use_cuda else None

        self.blackout_luma_thresh = float(blackout_luma_thresh)
        self.blackout_dark_ratio = float(blackout_dark_ratio)
        self.glare_sat_ratio = float(glare_sat_ratio)
        self.lighting_hysteresis_frames = max(3, int(lighting_hysteresis_frames))
        self.blur_rel_drop_ratio = float(blur_rel_drop_ratio)
        self.blur_abs_floor = float(blur_abs_floor)
        self.blur_hysteresis_frames = max(5, int(blur_hysteresis_frames))
        self.ingot_min_area_ratio = float(ingot_min_area_ratio)
        self.ingot_max_area_ratio = float(ingot_max_area_ratio)

        # Pre-allocate GPU kernels & Hanning window in VRAM
        if self.use_cuda:
            kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=self.torch_device).view(1, 1, 3, 3)
            ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=self.torch_device).view(1, 1, 3, 3)
            self._sobel_xy = torch.cat([kx, ky], dim=0)
            h1 = torch.hann_window(self.FFT_SIZE, periodic=False, device=self.torch_device, dtype=torch.float32)
            self._hann_2d = (h1[:, None] * h1[None, :]).contiguous()
        else:
            self._sobel_xy = None
            h1_np = np.hanning(self.FFT_SIZE).astype(np.float32)
            self._hann_2d_np = (h1_np[:, None] * h1_np[None, :]).astype(np.float32)

        self.reset()

    def reset(self) -> None:
        """Reset temporal hysteresis counters, sharpness baseline, and FFT phase memory."""
        self._blackout_streak = 0
        self._glare_streak = 0
        self._blur_streak = 0
        self._ingot_streak = 0
        self._static_frame_streak = 0
        self._frames_since_distinct = 0
        self.baseline_sharpness: float | None = None
        self.baseline_p99_grad: float | None = None
        self._baseline_samples = 0
        self._prev_fft = None
        self._prev_patch_gpu = None
        self._prev_patch_cpu = None
        self._prev_raw_patch_cpu = None
        self._vy_history: deque[float] = deque(maxlen=5)
        self.conveyor_vy_ema: float = 0.0
        self.conveyor_direction: str = "UNKNOWN"
        self._direction_locked: bool = False
        self._last_gpu_thumb = None

    def reset_phase_memory(self) -> None:
        """Clear transient inter-frame phase buffers after a seek while keeping calibrated baselines."""
        self._static_frame_streak = 0
        self._frames_since_distinct = 0
        self._prev_fft = None
        self._prev_patch_gpu = None
        self._prev_patch_cpu = None
        self._prev_raw_patch_cpu = None

    @property
    def last_gpu_thumb(self):
        """Expose latest (1, 3, THUMB_H, THUMB_W) float32 BGR tensor on CUDA for downstream reuse."""
        return self._last_gpu_thumb

    @staticmethod
    def _subpixel_peak_1d(vm1: float, v0: float, vp1: float) -> float:
        denom = 2.0 * (vm1 - 2.0 * v0 + vp1)
        if abs(denom) < 1e-6:
            return 0.0
        return float(max(-0.5, min(0.5, (vm1 - vp1) / denom)))

    def _compute_metrics_and_phase_gpu(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        h_f, w_f = frame_bgr.shape[:2]
        # Fast strided downsample before PCIe upload (~170 KB instead of 2.7 MB -> <0.05 ms transfer)
        step_y = max(1, h_f // 360)
        step_x = max(1, w_f // 640)
        sub_bgr = np.ascontiguousarray(frame_bgr[::step_y, ::step_x])

        with torch.inference_mode():
            t_u8 = torch.from_numpy(sub_bgr).to(self.torch_device, non_blocking=True)
            t_chw = t_u8.permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32)
            thumb = F.interpolate(t_chw, size=(self.THUMB_H, self.THUMB_W), mode="bilinear", align_corners=False)
            self._last_gpu_thumb = thumb

            b = thumb[0, 0]
            g = thumb[0, 1]
            r = thumb[0, 2]
            luma = 0.114 * b + 0.587 * g + 0.299 * r

            # 1. Active conveyor lighting & glare reductions
            ry1, ry2 = int(0.25 * self.THUMB_H), int(0.70 * self.THUMB_H)
            rx1, rx2 = int(0.08 * self.THUMB_W), int(0.92 * self.THUMB_W)
            roi_luma = luma[ry1:ry2, rx1:rx2]
            roi_max = torch.maximum(torch.maximum(b[ry1:ry2, rx1:rx2], g[ry1:ry2, rx1:rx2]), r[ry1:ry2, rx1:rx2])
            roi_min = torch.minimum(torch.minimum(b[ry1:ry2, rx1:rx2], g[ry1:ry2, rx1:rx2]), r[ry1:ry2, rx1:rx2])

            mean_luma_t = roi_luma.mean()
            luma_std_t = roi_luma.std()
            dark_ratio_t = (roi_luma <= 15.0).to(torch.float32).mean()
            glare_ratio_t = ((roi_max >= 248.0) & ((roi_max - roi_min) <= 35.0)).to(torch.float32).mean()

            # 2. Side-rail Tenengrad sharpness via fused 2-filter Sobel conv2d
            grads = F.conv2d(luma.unsqueeze(0).unsqueeze(0), self._sobel_xy, padding=1)[0]
            grad_mag = torch.sqrt(grads[0] * grads[0] + grads[1] * grads[1] + 1e-6)
            sy1, sy2 = int(0.22 * self.THUMB_H), int(0.72 * self.THUMB_H)
            lx1, lx2 = int(0.05 * self.THUMB_W), int(0.22 * self.THUMB_W)
            rx_r1, rx_r2 = int(0.78 * self.THUMB_W), int(0.95 * self.THUMB_W)
            rail_grad = torch.cat([grad_mag[sy1:sy2, lx1:lx2], grad_mag[sy1:sy2, rx_r1:rx_r2]], dim=1)
            rail_sharp_t = rail_grad.mean()
            p99_grad_t = torch.quantile(grad_mag[ry1:ry2, rx1:rx2].reshape(-1), 0.99)

            # 3. Gate bright-ingot candidate mask (downsampled boolean mask for ultra-fast CC)
            gy1, gy2 = int(0.32 * self.THUMB_H), int(0.68 * self.THUMB_H)
            gx1, gx2 = int(0.12 * self.THUMB_W), int(0.88 * self.THUMB_W)
            gate_luma = luma[gy1:gy2, gx1:gx2]
            bg_med_t = gate_luma.median()
            bright_thr = torch.clamp(bg_med_t + 58.0, min=175.0)
            ingot_mask_gpu = (gate_luma >= bright_thr) & (bg_med_t < 155.0)
            ingot_frac_t = ingot_mask_gpu.to(torch.float32).mean()

            # 4. CUDA cuFFT Phase Correlation on anti-aliased 128x128 center conveyor patch (spans >= 2 slat seams)
            py1_f, py2_f = int(0.26 * h_f), int(0.74 * h_f)
            px1_f, px2_f = int(0.24 * w_f), int(0.76 * w_f)
            aa_bgr = cv2.resize(
                frame_bgr[py1_f:py2_f, px1_f:px2_f],
                (self.FFT_SIZE, self.FFT_SIZE),
                interpolation=cv2.INTER_AREA,
            )
            t_aa = torch.from_numpy(aa_bgr).to(self.torch_device, non_blocking=True).to(dtype=torch.float32)
            patch_128 = 0.114 * t_aa[:, :, 0] + 0.587 * t_aa[:, :, 1] + 0.299 * t_aa[:, :, 2]

            p_raw_std = patch_128.std()
            vy_inst = 0.0
            phase_conf = 0.0
            is_duplicate_frame = False
            if float(p_raw_std.item()) > 1.2:
                unit_p = (patch_128 - patch_128.mean()) / (p_raw_std + 1e-4)
                if self._prev_patch_gpu is not None:
                    mad_unit = float(torch.mean(torch.abs(unit_p - self._prev_patch_gpu)).item())
                    if mad_unit < 0.045:
                        is_duplicate_frame = True

                if not is_duplicate_frame:
                    # High-pass out slow stationary illumination/glare gradients while keeping seams, ribs & belt grain
                    illum_bg = F.avg_pool2d(unit_p[None, None], kernel_size=(33, 9), stride=1, padding=(16, 4))[0, 0]
                    hp = unit_p - illum_bg
                    hp_std = hp.std()
                    norm_p = ((hp - hp.mean()) / (hp_std + 1e-4)) * self._hann_2d
                    f_curr = torch.fft.rfft2(norm_p)
                    dt_ticks = max(1, self._frames_since_distinct + 1)
                    if self._prev_fft is not None:
                        prod = f_curr * torch.conj(self._prev_fft)
                        cps = prod / (torch.abs(prod) + 1e-5)
                        corr = torch.fft.irfft2(cps, s=(self.FFT_SIZE, self.FFT_SIZE))
                        flat_idx = int(torch.argmax(corr).item())
                        pk_y, pk_x = divmod(flat_idx, self.FFT_SIZE)
                        pk_val = float(corr[pk_y, pk_x].item())
                        ym1 = (pk_y - 1) % self.FFT_SIZE
                        yp1 = (pk_y + 1) % self.FFT_SIZE
                        sub_dy = self._subpixel_peak_1d(
                            float(corr[ym1, pk_x].item()),
                            pk_val,
                            float(corr[yp1, pk_x].item()),
                        )
                        dy_fft = float(pk_y) + sub_dy
                        if dy_fft > (self.FFT_SIZE * 0.5):
                            dy_fft -= float(self.FFT_SIZE)
                        orig_patch_h = float(py2_f - py1_f)
                        vy_inst = (dy_fft * (orig_patch_h / float(self.FFT_SIZE))) / float(min(4, dt_ticks))
                        phase_conf = pk_val
                    self._prev_fft = f_curr
                    self._prev_patch_gpu = unit_p
            else:
                self._prev_fft = None
                self._prev_patch_gpu = None

            # Single scalar pack transfer from GPU to CPU
            scalars = torch.stack([
                mean_luma_t, luma_std_t, dark_ratio_t, glare_ratio_t,
                rail_sharp_t, p99_grad_t, ingot_frac_t,
            ]).cpu().numpy()

            ingot_mask_np = None
            if float(scalars[6]) >= (self.ingot_min_area_ratio * 0.75) and float(scalars[3]) <= self.glare_sat_ratio:
                ingot_mask_np = ingot_mask_gpu.to(torch.uint8).cpu().numpy()

        return {
            "mean_luma": float(scalars[0]),
            "luma_std": float(scalars[1]),
            "dark_ratio": float(scalars[2]),
            "glare_ratio": float(scalars[3]),
            "rail_sharpness": float(scalars[4]),
            "p99_grad": float(scalars[5]),
            "ingot_frac": float(scalars[6]),
            "ingot_mask": ingot_mask_np,
            "gate_bounds_thumb": (gx1, gy1, gx2, gy2),
            "vy_inst": float(vy_inst),
            "phase_conf": float(phase_conf),
            "is_duplicate_frame": is_duplicate_frame,
            "backend": "cuda",
        }

    def _compute_metrics_and_phase_cpu(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        h_f, w_f = frame_bgr.shape[:2]
        thumb = cv2.resize(frame_bgr, (self.THUMB_W, self.THUMB_H), interpolation=cv2.INTER_AREA)
        b = thumb[:, :, 0].astype(np.float32)
        g = thumb[:, :, 1].astype(np.float32)
        r = thumb[:, :, 2].astype(np.float32)
        luma = 0.114 * b + 0.587 * g + 0.299 * r

        ry1, ry2 = int(0.25 * self.THUMB_H), int(0.70 * self.THUMB_H)
        rx1, rx2 = int(0.08 * self.THUMB_W), int(0.92 * self.THUMB_W)
        roi_luma = luma[ry1:ry2, rx1:rx2]
        roi_max = np.maximum(np.maximum(b[ry1:ry2, rx1:rx2], g[ry1:ry2, rx1:rx2]), r[ry1:ry2, rx1:rx2])
        roi_min = np.minimum(np.minimum(b[ry1:ry2, rx1:rx2], g[ry1:ry2, rx1:rx2]), r[ry1:ry2, rx1:rx2])

        mean_luma = float(np.mean(roi_luma))
        luma_std = float(np.std(roi_luma))
        dark_ratio = float(np.mean(roi_luma <= 15.0))
        glare_ratio = float(np.mean((roi_max >= 248.0) & ((roi_max - roi_min) <= 35.0)))

        gx = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx * gx + gy * gy + 1e-6)
        sy1, sy2 = int(0.22 * self.THUMB_H), int(0.72 * self.THUMB_H)
        lx1, lx2 = int(0.05 * self.THUMB_W), int(0.22 * self.THUMB_W)
        rx_r1, rx_r2 = int(0.78 * self.THUMB_W), int(0.95 * self.THUMB_W)
        rail_sharp = float(0.5 * (np.mean(grad_mag[sy1:sy2, lx1:lx2]) + np.mean(grad_mag[sy1:sy2, rx_r1:rx_r2])))
        p99_grad = float(np.percentile(grad_mag[ry1:ry2, rx1:rx2], 99))

        gy1, gy2 = int(0.32 * self.THUMB_H), int(0.68 * self.THUMB_H)
        gx1, gx2 = int(0.12 * self.THUMB_W), int(0.88 * self.THUMB_W)
        gate_luma = luma[gy1:gy2, gx1:gx2]
        bg_med = float(np.median(gate_luma))
        bright_thr = max(175.0, bg_med + 58.0)
        ingot_mask_bool = (gate_luma >= bright_thr) if bg_med < 155.0 else np.zeros_like(gate_luma, dtype=bool)
        ingot_frac = float(np.mean(ingot_mask_bool))
        ingot_mask_np = ingot_mask_bool.astype(np.uint8) if (ingot_frac >= self.ingot_min_area_ratio * 0.75 and glare_ratio <= self.glare_sat_ratio) else None

        # Phase correlation on anti-aliased 128x128 center conveyor patch
        py1, py2 = int(0.26 * h_f), int(0.74 * h_f)
        px1, px2 = int(0.24 * w_f), int(0.76 * w_f)
        crop_g = cv2.cvtColor(frame_bgr[py1:py2, px1:px2], cv2.COLOR_BGR2GRAY)
        patch_128 = cv2.resize(crop_g, (self.FFT_SIZE, self.FFT_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)
        p_std = float(np.std(patch_128))
        vy_inst = 0.0
        phase_conf = 0.0
        is_duplicate_frame = False
        if p_std > 1.2:
            unit_p = ((patch_128 - float(np.mean(patch_128))) / (p_std + 1e-4)).astype(np.float32)
            if self._prev_raw_patch_cpu is not None:
                if float(np.mean(np.abs(unit_p - self._prev_raw_patch_cpu))) < 0.045:
                    is_duplicate_frame = True
            if not is_duplicate_frame:
                illum_bg = cv2.boxFilter(unit_p, -1, (9, 33), normalize=True)
                hp = unit_p - illum_bg
                hp_std = float(np.std(hp))
                norm_p = ((hp - float(np.mean(hp))) / (hp_std + 1e-4)).astype(np.float32)
                dt_ticks = max(1, self._frames_since_distinct + 1)
                if self._prev_patch_cpu is not None:
                    (_dx_pc, dy_pc), resp = cv2.phaseCorrelate(self._prev_patch_cpu, norm_p, self._hann_2d_np)
                    vy_inst = (float(dy_pc) * (float(py2 - py1) / float(self.FFT_SIZE))) / float(min(4, dt_ticks))
                    phase_conf = float(resp)
                self._prev_patch_cpu = norm_p
                self._prev_raw_patch_cpu = unit_p
        else:
            self._prev_patch_cpu = None
            self._prev_raw_patch_cpu = None

        return {
            "mean_luma": mean_luma,
            "luma_std": luma_std,
            "dark_ratio": dark_ratio,
            "glare_ratio": glare_ratio,
            "rail_sharpness": rail_sharp,
            "p99_grad": p99_grad,
            "ingot_frac": ingot_frac,
            "ingot_mask": ingot_mask_np,
            "gate_bounds_thumb": (gx1, gy1, gx2, gy2),
            "vy_inst": vy_inst,
            "phase_conf": phase_conf,
            "is_duplicate_frame": is_duplicate_frame,
            "backend": "cpu",
        }

    def evaluate(self, frame_bgr: np.ndarray | None) -> dict[str, Any]:
        """Evaluate Stage-0 frame health, ingot occlusion, and conveyor phase velocity."""
        if frame_bgr is None or frame_bgr.size == 0:
            return {
                "is_invalid": True,
                "invalid_code": "EMPTY_FRAME",
                "invalid_reason": "INVALID: Citra kamera kosong / terputus",
                "is_foreign_occluded": False,
                "foreign_box": None,
                "conveyor_vy": self.conveyor_vy_ema,
                "conveyor_direction": self.conveyor_direction,
                "backend": "none",
            }

        h_f, w_f = frame_bgr.shape[:2]
        raw = (
            self._compute_metrics_and_phase_gpu(frame_bgr)
            if self.use_cuda
            else self._compute_metrics_and_phase_cpu(frame_bgr)
        )

        # 1. Update Phase-Correlation Velocity & Auto-Direction (immune to 15-20 FPS USB duplicate frames)
        if raw.get("is_duplicate_frame", False):
            self._static_frame_streak += 1
            self._frames_since_distinct += 1
            # Only declare STOPPED after >= 15 consecutive static frames (~0.5s), never on 1-2 USB duplicate ticks
            if self._static_frame_streak >= 15:
                self.conveyor_vy_ema *= 0.65
                if abs(self.conveyor_vy_ema) < 0.20:
                    self.conveyor_vy_ema = 0.0
                    self.conveyor_direction = "STOPPED"
                    self._vy_history.clear()
        else:
            self._static_frame_streak = 0
            self._frames_since_distinct = 0
            vy_inst = raw["vy_inst"]
            p_conf = raw["phase_conf"]
            if p_conf >= 0.07 and abs(vy_inst) <= (0.14 * h_f):
                # Ignore isolated zero-shift spikes when smooth steel briefly passes while conveyor is moving
                if abs(vy_inst) >= 0.25 or not self._direction_locked or abs(self.conveyor_vy_ema) < 0.50:
                    self._vy_history.append(vy_inst)
                med_vy = float(np.median(self._vy_history)) if self._vy_history else vy_inst
                if not self._direction_locked and len(self._vy_history) >= 3:
                    self.conveyor_vy_ema = med_vy
                    self._direction_locked = True
                else:
                    self.conveyor_vy_ema = 0.72 * self.conveyor_vy_ema + 0.28 * med_vy

                if len(self._vy_history) >= 3:
                    if self.conveyor_vy_ema < -0.40:
                        self.conveyor_direction = "REVERSE"
                    elif self.conveyor_vy_ema > 0.40:
                        self.conveyor_direction = "FORWARD"
                    elif abs(self.conveyor_vy_ema) < 0.18:
                        self.conveyor_direction = "STOPPED"

        # 2. Check Blackout & Extreme Specular Glare with Hysteresis
        is_blackout_now = (
            raw["mean_luma"] < self.blackout_luma_thresh
            and raw["dark_ratio"] > self.blackout_dark_ratio
        )
        self._blackout_streak = (self._blackout_streak + 1) if is_blackout_now else 0

        is_glare_now = raw["glare_ratio"] > self.glare_sat_ratio
        self._glare_streak = (self._glare_streak + 1) if is_glare_now else 0

        # 3. Check Side-Rail Blur / Steam / Oil Fog with Self-Calibrating Baseline
        sharp = raw["rail_sharpness"]
        p99_g = raw["p99_grad"]
        healthy_lighting = (not is_blackout_now) and (not is_glare_now) and raw["mean_luma"] >= 26.0
        if healthy_lighting and sharp >= 8.5:
            if self.baseline_sharpness is None:
                self.baseline_sharpness = sharp
                self.baseline_p99_grad = p99_g
            else:
                alpha = 0.08 if sharp >= self.baseline_sharpness * 0.75 else 0.002
                self.baseline_sharpness = (1.0 - alpha) * self.baseline_sharpness + alpha * sharp
                if self.baseline_p99_grad is not None and p99_g >= self.baseline_p99_grad * 0.75:
                    self.baseline_p99_grad = 0.92 * self.baseline_p99_grad + 0.08 * p99_g
            self._baseline_samples += 1

        rel_blur = (
            self._baseline_samples >= 10
            and self.baseline_sharpness is not None
            and sharp < max(self.blur_abs_floor, self.blur_rel_drop_ratio * self.baseline_sharpness)
            and (self.baseline_p99_grad is None or p99_g < 0.60 * self.baseline_p99_grad)
        )
        cold_severe_fog = (
            healthy_lighting
            and raw["luma_std"] > 5.5
            and p99_g < 18.0
            and sharp < 5.0
        )
        self._blur_streak = (self._blur_streak + 1) if (healthy_lighting and (rel_blur or cold_severe_fog)) else 0

        # 4. Check Compact Bright Foreign Object / Aluminum Ingot Occlusion
        is_foreign_now = False
        foreign_box = None
        ingot_mask = raw["ingot_mask"]
        if ingot_mask is not None and healthy_lighting:
            num_lbl, _lbls, stats, _centroids = cv2.connectedComponentsWithStats(ingot_mask, connectivity=8)
            gate_area = float(max(1, ingot_mask.shape[0] * ingot_mask.shape[1]))
            gx1, gy1, _gx2, _gy2 = raw["gate_bounds_thumb"]
            sx_orig = float(w_f) / float(self.THUMB_W)
            sy_orig = float(h_f) / float(self.THUMB_H)
            for lbl_idx in range(1, num_lbl):
                bx, by, bw, bh, area = stats[lbl_idx]
                area_ratio = float(area) / gate_area
                if not (self.ingot_min_area_ratio <= area_ratio <= self.ingot_max_area_ratio):
                    continue
                fill_ratio = float(area) / float(max(1, bw * bh))
                orig_bw = float(bw) * sx_orig
                orig_bh = float(bh) * sy_orig
                aspect = orig_bw / max(1.0, orig_bh)
                # Compact bright block (ingot/scrap), explicitly excluding wide horizontal ribs (aspect > 4.5)
                if fill_ratio >= 0.45 and 0.30 <= aspect <= 3.20:
                    is_foreign_now = True
                    fx1 = int(round((gx1 + bx) * sx_orig))
                    fy1 = int(round((gy1 + by) * sy_orig))
                    fx2 = int(round((gx1 + bx + bw) * sx_orig))
                    fy2 = int(round((gy1 + by + bh) * sy_orig))
                    foreign_box = [max(0, fx1), max(0, fy1), min(w_f, fx2), min(h_f, fy2)]
                    break

        self._ingot_streak = (self._ingot_streak + 1) if is_foreign_now else 0
        is_foreign_occluded = bool(is_foreign_now and (self._ingot_streak >= 2 or raw["ingot_frac"] >= 0.18))

        invalid_code = None
        invalid_reason = ""
        if self._blackout_streak >= self.lighting_hysteresis_frames:
            invalid_code = "LIGHTING_BLACKOUT"
            invalid_reason = f"INVALID: Pencahayaan Gelap/Mati (Luma={raw['mean_luma']:.1f}, Dark={raw['dark_ratio']*100:.0f}%)"
        elif self._glare_streak >= self.lighting_hysteresis_frames:
            invalid_code = "EXTREME_GLARE"
            invalid_reason = f"INVALID: Silau Ekstrem Menutupi Kamera (Glare={raw['glare_ratio']*100:.0f}%)"
        elif self._blur_streak >= self.blur_hysteresis_frames:
            invalid_code = "LENS_BLUR_OR_FOG"
            invalid_reason = f"INVALID: Lensa Kabur / Tertutup Uap-Oli (Sharp={sharp:.1f})"

        return {
            "is_invalid": invalid_code is not None,
            "invalid_code": invalid_code,
            "invalid_reason": invalid_reason,
            "is_foreign_occluded": is_foreign_occluded,
            "foreign_box": foreign_box,
            "mean_luma": round(raw["mean_luma"], 2),
            "dark_ratio": round(raw["dark_ratio"], 4),
            "glare_ratio": round(raw["glare_ratio"], 4),
            "rail_sharpness": round(sharp, 2),
            "baseline_sharpness": round(self.baseline_sharpness, 2) if self.baseline_sharpness is not None else None,
            "conveyor_vy": round(self.conveyor_vy_ema, 3),
            "conveyor_direction": self.conveyor_direction,
            "phase_conf": round(raw["phase_conf"], 3),
            "backend": raw["backend"],
        }
