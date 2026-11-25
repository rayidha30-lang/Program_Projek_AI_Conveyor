"""
Shared video navigation, time-jump, frame-skip, speed, and idle-conveyor auto-skip utilities
used by both combined_top_inspector.py and run_multi_camera_benchmark.py.
"""

from __future__ import annotations

import subprocess
import sys

import cv2
import numpy as np


SPEED_STEPS: tuple[int, ...] = (1, 2, 4, 8, 16)

# Unified hotkey table: key_code -> (delta_amount, unit ['s'|'f'], label)
# Note: '>' is intentionally reserved for playback speed cycling ([X]/[>]).
SEEK_KEY_SPEC: dict[int, tuple[float, str, str]] = {
    ord("["): (-10.0, "s", "-10s"),
    ord("]"): (+10.0, "s", "+10s"),
    ord("-"): (-60.0, "s", "-60s"),
    ord("_"): (-60.0, "s", "-60s"),
    ord("{"): (-60.0, "s", "-60s"),
    ord("="): (+60.0, "s", "+60s"),
    ord("+"): (+60.0, "s", "+60s"),
    ord("}"): (+60.0, "s", "+60s"),
    ord(","): (-10.0, "f", "-10f"),
    ord("<"): (-10.0, "f", "-10f"),
    ord("."): (+10.0, "f", "+10f"),
    ord(";"): (-60.0, "f", "-60f"),
    ord(":"): (-60.0, "f", "-60f"),
    ord("'"): (+60.0, "f", "+60f"),
    ord('"'): (+60.0, "f", "+60f"),
}


def next_speed_step(current_speed: int) -> int:
    """Return the next playback speed multiplier in SPEED_STEPS."""
    if current_speed in SPEED_STEPS:
        return SPEED_STEPS[(SPEED_STEPS.index(current_speed) + 1) % len(SPEED_STEPS)]
    return SPEED_STEPS[0]


def parse_time_to_seconds(time_str):
    """Convert a time string ('MM:SS', 'HH:MM:SS', or numeric seconds) into float seconds."""
    if time_str is None:
        return None
    raw = str(time_str).strip().lower().rstrip("s")
    if not raw:
        return None
    try:
        if ":" in raw:
            parts = [float(p.strip()) for p in raw.split(":")]
            if len(parts) == 2:
                m, s = parts
                return max(0.0, m * 60.0 + s)
            if len(parts) == 3:
                h, m, s = parts
                return max(0.0, h * 3600.0 + m * 60.0 + s)
            return None
        return max(0.0, float(raw))
    except Exception:
        return None


def format_seconds_mmss(sec):
    """Format seconds as a compact 'MM:SS' or 'H:MM:SS' string."""
    total_s = max(0, int(round(float(sec))))
    h, rem = divmod(total_s, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def compute_seek_target_from_key(
    key: int,
    cur_frame: int,
    total_frames: int,
    fps: float,
) -> tuple[int | None, str | None]:
    """Resolve a seek hotkey into a clamped (target_frame, human_label), or (None, None)."""
    spec = SEEK_KEY_SPEC.get(int(key))
    if spec is None or total_frames <= 0:
        return None, None
    amount, unit, label = spec
    safe_fps = float(fps) if (fps and float(fps) > 1e-5) else 30.0
    delta_frames = amount * safe_fps if unit == "s" else amount
    raw_target = int(round(float(cur_frame) + delta_frames))
    clamped = max(0, min(int(total_frames) - 1, raw_target))
    return clamped, label


def format_speed_label(
    speed_mult: int,
    skip_idle: bool = False,
    is_scanning_idle: bool = False,
    sync_realtime: bool = True,
) -> str:
    """Format speed & skip-idle badge text for the combined dashboard HUD."""
    parts = [f"Speed: {max(1, int(speed_mult))}x"]
    if speed_mult == 1 and not sync_realtime:
        parts[0] = "Speed: TURBO"
    if is_scanning_idle:
        parts.append("SKIP-DIAM [SCANNING]")
    elif skip_idle:
        parts.append("Skip-Diam: ON")
    return " | ".join(parts)


def detect_conveyor_motion_score(prev_gray_roi, frame_bgr):
    """Compute lightweight belt motion energy (<0.2 ms) on the center corridor ROI."""
    if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
        return 0.0, prev_gray_roi
    small = cv2.resize(frame_bgr, (240, 135), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    # Focus on center belt & inner rail corridor so outside shadows don't cause false motion
    roi = gray[30:108, 42:198]
    if prev_gray_roi is None or prev_gray_roi.shape != roi.shape:
        return 0.0, roi
    diff = cv2.absdiff(roi, prev_gray_roi)
    return float(np.mean(diff)), roi


def prompt_jump_time_dialog(default_time_str="00:00", total_time_str=""):
    """Prompt for target MM:SS timestamp in an isolated subprocess (X11/OpenCV safe)."""
    prompt_lbl = "Masukkan target waktu video (contoh: 14:30 atau 870):"
    if total_time_str:
        prompt_lbl = f"Masukkan target waktu video (Durasi {total_time_str}, contoh 14:30):"
    script = (
        "import tkinter as tk; from tkinter import simpledialog; "
        "r = tk.Tk(); r.withdraw(); r.attributes('-topmost', True); "
        f"val = simpledialog.askstring('Loncat ke Waktu Video', {prompt_lbl!r}, initialvalue={default_time_str!r}, parent=r); "
        "r.destroy(); print('JUMP_VAL:' + (val or ''))"
    )
    try:
        proc = subprocess.run([sys.executable, "-c", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
        for line in proc.stdout.splitlines():
            if line.startswith("JUMP_VAL:"):
                val = line.split(":", 1)[1].strip()
                return val or None
    except Exception:
        pass
    return None


class IdleConveyorSkipper:
    """Stateful detector that fast-skips stationary conveyor segments and rewinds when motion resumes."""

    def __init__(
        self,
        enabled: bool = False,
        motion_thresh: float = 2.2,
        idle_trigger_sec: float = 1.2,
        skip_step_sec: float = 0.35,
        rewind_sec: float = 1.5,
        sample_interval_frames: int = 2,
    ):
        self.enabled = bool(enabled)
        self.motion_thresh = float(motion_thresh)
        self.idle_trigger_sec = float(idle_trigger_sec)
        self.skip_step_sec = float(skip_step_sec)
        self.rewind_sec = float(rewind_sec)
        self.sample_interval_frames = max(1, int(sample_interval_frames))
        self.reset_all()

    def reset_scan_state(self) -> None:
        """Reset active idle/motion scan counters while keeping enabled flag intact."""
        self.prev_motion_roi = None
        self.idle_streak_sec = 0.0
        self.motion_streak = 0
        self.last_motion_sample_frame = -1
        self.is_scanning_idle = False
        self.motion_energy = 0.0

    def reset_all(self) -> None:
        """Alias for resetting scan state."""
        self.reset_scan_state()

    def toggle(self) -> bool:
        """Toggle Skip-Diam enabled state and reset transient scan state."""
        self.enabled = not self.enabled
        self.reset_scan_state()
        return self.enabled

    def evaluate(
        self,
        cur_pos_f: int,
        frame_bgr: np.ndarray | None,
        video_fps: float,
        speed_mult: int = 1,
    ) -> tuple[str, dict]:
        """
        Evaluate whether to continue normal inspection, fast-skip ahead, or rewind on motion resume.
        Returns (action, payload) where action in {'NORMAL', 'SKIP_FORWARD', 'REWIND_RESUME'}.
        """
        if not self.enabled or frame_bgr is None:
            return "NORMAL", {}

        safe_fps = float(video_fps) if (video_fps and float(video_fps) > 1e-5) else 30.0
        if (
            self.last_motion_sample_frame >= 0
            and abs(int(cur_pos_f) - self.last_motion_sample_frame) < self.sample_interval_frames
        ):
            return "NORMAL", {}

        m_score, self.prev_motion_roi = detect_conveyor_motion_score(self.prev_motion_roi, frame_bgr)
        self.last_motion_sample_frame = int(cur_pos_f)
        self.motion_energy = m_score
        was_scanning = self.is_scanning_idle
        step_dt = float(max(1, int(speed_mult))) / safe_fps

        if m_score < self.motion_thresh:
            self.idle_streak_sec += self.skip_step_sec if was_scanning else step_dt
            self.motion_streak = 0
        else:
            self.idle_streak_sec = 0.0
            self.motion_streak += 1

        if (self.idle_streak_sec >= self.idle_trigger_sec or was_scanning) and self.motion_streak < 2:
            self.is_scanning_idle = True
            skip_frames = max(6, int(round(self.skip_step_sec * safe_fps)))
            stub_result = {
                "status": "WAIT",
                "slat_res": {"display_status": "WAIT", "eval_mode": f"SKIP DIAM (E={m_score:.1f})"},
                "rib_res": {"status": "WAIT", "is_rib_present": False, "message": "Mencari segmen conveyor bergerak..."},
                "elapsed_ms": 0.0,
                "is_conveyor_moving": False,
            }
            return "SKIP_FORWARD", {
                "skip_frames": skip_frames,
                "motion_energy": m_score,
                "stub_result": stub_result,
            }

        if was_scanning and self.motion_streak >= 2:
            rewind_f = max(0, int(cur_pos_f) - int(round(self.rewind_sec * safe_fps)))
            self.idle_streak_sec = 0.0
            self.motion_streak = 0
            self.is_scanning_idle = False
            self.prev_motion_roi = None
            return "REWIND_RESUME", {
                "rewind_frame": rewind_f,
                "motion_energy": m_score,
            }

        return "NORMAL", {"motion_energy": m_score}
