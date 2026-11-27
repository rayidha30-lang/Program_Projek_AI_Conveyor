"""
Scheduled / Duty-Cycle Detection Controller for AI Slat Rib Conveyor.

Designed for zero-FPS-drop hot-loop gating (<0.15 us/frame via monotonic TTL cache)
and Keep-Warm Resident GPU operation (models remain loaded in VRAM during STANDBY,
with an automatic T-2s PRE_WARMUP window to wake the GPU from P8 -> P0 cleanly).

Supported modes:
  - 'continuous' : Always ACTIVE (backward-compatible default when not configured).
  - 'periodic'   : Fixed duty cycle, e.g. active 15 min every 60 min anchored at :00
                   (08:00-08:15, 09:00-09:15, ...).
  - 'slots'      : Explicit daily HH:MM-HH:MM windows (supports midnight wrap, e.g. 23:50-00:10).
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Literal

ScheduleMode = Literal["continuous", "periodic", "slots"]
ScheduleState = Literal["ACTIVE", "PRE_WARMUP", "STANDBY"]
OverrideMode = Literal["auto", "force_active", "force_standby"]

STATE_TO_CODE: dict[str, int] = {"STANDBY": 0, "PRE_WARMUP": 1, "ACTIVE": 2}
CODE_TO_STATE: dict[int, ScheduleState] = {0: "STANDBY", 1: "PRE_WARMUP", 2: "ACTIVE"}

OVERRIDE_TO_CODE: dict[str, int] = {"auto": 0, "force_active": 1, "force_standby": 2}
CODE_TO_OVERRIDE: dict[int, OverrideMode] = {0: "auto", 1: "force_active", 2: "force_standby"}

OVERRIDE_CYCLE: tuple[OverrideMode, ...] = ("auto", "force_active", "force_standby")
SECONDS_PER_DAY = 86400


def parse_schedule_slots(raw: str | None) -> tuple[tuple[int, int], ...]:
    """
    Parse comma-separated 'HH:MM-HH:MM' daily slot definitions into tuples of
    `(start_sec_of_day, end_sec_of_day)`. Midnight-wrapping slots (end <= start)
    are preserved as `(start, end)` and evaluated with wrap-around semantics.
    """
    if not raw:
        return ()
    parsed: list[tuple[int, int]] = []
    for token in str(raw).split(","):
        tok = token.strip()
        if not tok or "-" not in tok:
            continue
        left, right = tok.split("-", 1)
        s_sec = _parse_hhmm_to_sec(left)
        e_sec = _parse_hhmm_to_sec(right)
        if s_sec is None or e_sec is None or s_sec == e_sec:
            continue
        parsed.append((s_sec, e_sec))
    parsed.sort(key=lambda item: item[0])
    return tuple(parsed)


def _parse_hhmm_to_sec(token: str) -> int | None:
    parts = token.strip().split(":")
    if len(parts) not in (2, 3):
        return None
    try:
        hh = int(parts[0])
        mm = int(parts[1])
        ss = int(parts[2]) if len(parts) == 3 else 0
    except ValueError:
        return None
    if hh == 24 and mm == 0 and ss == 0:
        return SECONDS_PER_DAY
    if not (0 <= hh <= 23 and 0 <= mm <= 59 and 0 <= ss <= 59):
        return None
    return hh * 3600 + mm * 60 + ss


def format_mmss(seconds: float) -> str:
    """Format remaining seconds as 'MM:SS' or 'HH:MM:SS'."""
    total = max(0, int(round(float(seconds))))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _format_hhmm_from_sec(sec_of_day: float) -> str:
    sec_norm = int(round(sec_of_day)) % SECONDS_PER_DAY
    hh, rem = divmod(sec_norm, 3600)
    mm = rem // 60
    return f"{hh:02d}:{mm:02d}"


@dataclass(frozen=True)
class DetectionScheduleConfig:
    mode: ScheduleMode = "continuous"
    interval_min: float = 60.0
    active_min: float = 15.0
    anchor_minute: float = 0.0
    slots: tuple[tuple[int, int], ...] = ()
    prewarm_sec: float = 2.0
    standby_fps: float = 20.0
    motion_quota: bool = False
    ttl_sec: float = 0.5

    def normalized(self) -> "DetectionScheduleConfig":
        raw_mode = str(self.mode or "continuous").strip().lower()
        mode: ScheduleMode = raw_mode if raw_mode in ("continuous", "periodic", "slots") else "continuous"
        interval_min = float(self.interval_min) if self.interval_min is not None else 60.0
        active_min = float(self.active_min) if self.active_min is not None else 15.0
        anchor_minute = float(self.anchor_minute) if self.anchor_minute is not None else 0.0
        prewarm_sec = max(0.0, min(30.0, float(self.prewarm_sec or 0.0)))
        standby_fps = max(1.0, min(60.0, float(self.standby_fps or 20.0)))
        ttl_sec = max(0.05, min(5.0, float(self.ttl_sec or 0.5)))

        if interval_min <= 0.0:
            interval_min = 60.0
        if active_min <= 0.0:
            active_min = 15.0

        slots = tuple(self.slots or ())
        if mode == "slots" and not slots:
            mode = "continuous"

        return DetectionScheduleConfig(
            mode=mode,
            interval_min=interval_min,
            active_min=active_min,
            anchor_minute=anchor_minute,
            slots=slots,
            prewarm_sec=prewarm_sec,
            standby_fps=standby_fps,
            motion_quota=bool(self.motion_quota),
            ttl_sec=ttl_sec,
        )


@dataclass(frozen=True)
class ScheduleSnapshot:
    mode: ScheduleMode
    state: ScheduleState
    is_detecting: bool
    is_prewarm: bool
    window_id: str
    remaining_sec: float
    next_start_hhmm: str
    override: OverrideMode
    active_min: float
    interval_min: float

    def to_ws_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "state": self.state,
            "override": self.override,
            "remaining_sec": int(max(0, round(self.remaining_sec))),
            "next_start": self.next_start_hhmm,
            "active_min": round(self.active_min, 2),
            "interval_min": round(self.interval_min, 2),
            "window_id": self.window_id,
        }

    def hud_badge(self) -> str:
        if self.mode == "continuous" and self.override == "auto":
            return "SCHED: CONTINUOUS"
        ovr_tag = ""
        if self.override == "force_active":
            ovr_tag = " [MANUAL ON]"
        elif self.override == "force_standby":
            ovr_tag = " [MANUAL STANDBY]"
        if self.state == "ACTIVE":
            rem = format_mmss(self.remaining_sec) if self.override == "auto" else "MANUAL"
            return f"SCHED: ACTIVE ({rem}){ovr_tag}"
        if self.state == "PRE_WARMUP":
            return f"SCHED: WARMUP ({int(max(1, round(self.remaining_sec)))}s)"
        return f"SCHED: STANDBY -> {self.next_start_hhmm} ({format_mmss(self.remaining_sec)}){ovr_tag}"


class DetectionScheduler:
    """
    Thread-safe, monotonic-cached detection window scheduler.

    Calling `evaluate()` in a 30+ FPS frame loop refreshes wall-clock state only once
    every `ttl_sec` (default 0.5s), keeping per-frame overhead around ~0.10 us.
    """

    def __init__(self, config: DetectionScheduleConfig | None = None) -> None:
        self.config = (config or DetectionScheduleConfig()).normalized()
        self._lock = threading.Lock()
        self._override: OverrideMode = "auto"
        self._next_refresh_mono: float = 0.0
        self._cached_snapshot: ScheduleSnapshot | None = None

        # Motion-quota tracking (only active when config.motion_quota=True in periodic mode)
        self._motion_cycle_id: str = ""
        self._motion_accum_sec: float = 0.0
        self._motion_last_mono: float | None = None

    @property
    def is_scheduled(self) -> bool:
        return self.config.mode != "continuous"

    @property
    def override(self) -> OverrideMode:
        with self._lock:
            return self._override

    def set_override(self, mode: str) -> ScheduleSnapshot:
        norm = str(mode or "auto").strip().lower()
        if norm not in ("auto", "force_active", "force_standby"):
            norm = "auto"
        with self._lock:
            self._override = norm  # type: ignore[assignment]
            self._next_refresh_mono = 0.0
        return self.evaluate(force_refresh=True)

    def reconfigure(self, config: DetectionScheduleConfig) -> ScheduleSnapshot:
        """Apply a new DetectionScheduleConfig live without restarting the runner."""
        norm_cfg = config.normalized()
        with self._lock:
            self.config = norm_cfg
            self._override = "auto"
            self._next_refresh_mono = 0.0
            self._cached_snapshot = None
            self._motion_cycle_id = ""
            self._motion_accum_sec = 0.0
            self._motion_last_mono = None
        return self.evaluate(force_refresh=True)

    def cycle_override(self) -> ScheduleSnapshot:
        with self._lock:
            idx = OVERRIDE_CYCLE.index(self._override)
            self._override = OVERRIDE_CYCLE[(idx + 1) % len(OVERRIDE_CYCLE)]
            self._next_refresh_mono = 0.0
        return self.evaluate(force_refresh=True)

    def record_motion_sample(self, is_moving: bool, now_mono: float | None = None) -> None:
        """Accumulate conveyor-moving seconds for optional motion_quota mode."""
        if not self.config.motion_quota or self.config.mode != "periodic":
            return
        cur_m = time.monotonic() if now_mono is None else float(now_mono)
        with self._lock:
            if self._motion_last_mono is None:
                self._motion_last_mono = cur_m
                return
            dt = max(0.0, min(2.0, cur_m - self._motion_last_mono))
            self._motion_last_mono = cur_m
            if is_moving:
                self._motion_accum_sec += dt

    def evaluate(
        self,
        now_dt: datetime | None = None,
        now_mono: float | None = None,
        force_refresh: bool = False,
    ) -> ScheduleSnapshot:
        # Fast path: no lock when cached snapshot is fresh and using real clock
        if now_dt is None and not force_refresh:
            cur_m = time.monotonic() if now_mono is None else float(now_mono)
            cached = self._cached_snapshot
            if cached is not None and cur_m < self._next_refresh_mono:
                return cached
        else:
            cur_m = time.monotonic() if now_mono is None else float(now_mono)

        dt = datetime.now() if now_dt is None else now_dt
        with self._lock:
            snap = self._compute_snapshot_locked(dt)
            if now_dt is None:
                self._cached_snapshot = snap
                self._next_refresh_mono = cur_m + self.config.ttl_sec
            return snap

    def _compute_snapshot_locked(self, dt: datetime) -> ScheduleSnapshot:
        cfg = self.config
        ovr = self._override

        if ovr == "force_active":
            return ScheduleSnapshot(
                mode=cfg.mode,
                state="ACTIVE",
                is_detecting=True,
                is_prewarm=False,
                window_id=dt.strftime("%Y%m%d_%H%M_MANUAL"),
                remaining_sec=0.0,
                next_start_hhmm=dt.strftime("%H:%M"),
                override=ovr,
                active_min=cfg.active_min,
                interval_min=cfg.interval_min,
            )

        if ovr == "force_standby":
            return ScheduleSnapshot(
                mode=cfg.mode,
                state="STANDBY",
                is_detecting=False,
                is_prewarm=False,
                window_id="MANUAL_STANDBY",
                remaining_sec=0.0,
                next_start_hhmm="--:--",
                override=ovr,
                active_min=cfg.active_min,
                interval_min=cfg.interval_min,
            )

        if cfg.mode == "continuous":
            return ScheduleSnapshot(
                mode="continuous",
                state="ACTIVE",
                is_detecting=True,
                is_prewarm=False,
                window_id=dt.strftime("%Y%m%d_CONT"),
                remaining_sec=0.0,
                next_start_hhmm=dt.strftime("%H:%M"),
                override=ovr,
                active_min=cfg.active_min,
                interval_min=cfg.interval_min,
            )

        if cfg.mode == "periodic":
            return self._compute_periodic_locked(dt)

        return self._compute_slots_locked(dt)

    def _compute_periodic_locked(self, dt: datetime) -> ScheduleSnapshot:
        cfg = self.config
        interval_sec = max(1.0, cfg.interval_min * 60.0)
        active_sec = max(1.0, cfg.active_min * 60.0)
        anchor_sec = (cfg.anchor_minute * 60.0) % interval_sec

        # Active >= Interval means always ACTIVE (safe degenerate clamping)
        if active_sec >= interval_sec:
            return ScheduleSnapshot(
                mode="periodic",
                state="ACTIVE",
                is_detecting=True,
                is_prewarm=False,
                window_id=dt.strftime("%Y%m%d_%H%M"),
                remaining_sec=interval_sec,
                next_start_hhmm=dt.strftime("%H:%M"),
                override=self._override,
                active_min=cfg.active_min,
                interval_min=cfg.interval_min,
            )

        sec_of_day = dt.hour * 3600.0 + dt.minute * 60.0 + dt.second + (dt.microsecond / 1e6)
        shifted = sec_of_day - anchor_sec
        cycle_idx = int(shifted // interval_sec)
        pos_in_cycle = shifted - (cycle_idx * interval_sec)

        cycle_start_sec = anchor_sec + (cycle_idx * interval_sec)
        cycle_start_dt = dt.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(seconds=cycle_start_sec)
        next_start_sec = cycle_start_sec + interval_sec
        next_start_hhmm = _format_hhmm_from_sec(next_start_sec)
        window_id = cycle_start_dt.strftime("%Y%m%d_%H%M")

        if cfg.motion_quota:
            if window_id != self._motion_cycle_id:
                self._motion_cycle_id = window_id
                self._motion_accum_sec = 0.0
                self._motion_last_mono = None
            hard_deadline = max(active_sec, interval_sec - max(cfg.prewarm_sec + 5.0, 60.0))
            if self._motion_accum_sec < active_sec and pos_in_cycle < hard_deadline:
                rem = max(0.0, active_sec - self._motion_accum_sec)
                return ScheduleSnapshot(
                    mode="periodic",
                    state="ACTIVE",
                    is_detecting=True,
                    is_prewarm=False,
                    window_id=window_id,
                    remaining_sec=rem,
                    next_start_hhmm=next_start_hhmm,
                    override=self._override,
                    active_min=cfg.active_min,
                    interval_min=cfg.interval_min,
                )
            rem_to_next = max(0.0, interval_sec - pos_in_cycle)
            if cfg.prewarm_sec > 0.0 and rem_to_next <= cfg.prewarm_sec:
                return ScheduleSnapshot(
                    mode="periodic",
                    state="PRE_WARMUP",
                    is_detecting=False,
                    is_prewarm=True,
                    window_id=window_id,
                    remaining_sec=rem_to_next,
                    next_start_hhmm=next_start_hhmm,
                    override=self._override,
                    active_min=cfg.active_min,
                    interval_min=cfg.interval_min,
                )
            return ScheduleSnapshot(
                mode="periodic",
                state="STANDBY",
                is_detecting=False,
                is_prewarm=False,
                window_id=window_id,
                remaining_sec=rem_to_next,
                next_start_hhmm=next_start_hhmm,
                override=self._override,
                active_min=cfg.active_min,
                interval_min=cfg.interval_min,
            )

        if pos_in_cycle < active_sec:
            return ScheduleSnapshot(
                mode="periodic",
                state="ACTIVE",
                is_detecting=True,
                is_prewarm=False,
                window_id=window_id,
                remaining_sec=max(0.0, active_sec - pos_in_cycle),
                next_start_hhmm=next_start_hhmm,
                override=self._override,
                active_min=cfg.active_min,
                interval_min=cfg.interval_min,
            )

        rem_to_next = max(0.0, interval_sec - pos_in_cycle)
        if cfg.prewarm_sec > 0.0 and rem_to_next <= cfg.prewarm_sec:
            return ScheduleSnapshot(
                mode="periodic",
                state="PRE_WARMUP",
                is_detecting=False,
                is_prewarm=True,
                window_id=window_id,
                remaining_sec=rem_to_next,
                next_start_hhmm=next_start_hhmm,
                override=self._override,
                active_min=cfg.active_min,
                interval_min=cfg.interval_min,
            )

        return ScheduleSnapshot(
            mode="periodic",
            state="STANDBY",
            is_detecting=False,
            is_prewarm=False,
            window_id=window_id,
            remaining_sec=rem_to_next,
            next_start_hhmm=next_start_hhmm,
            override=self._override,
            active_min=cfg.active_min,
            interval_min=cfg.interval_min,
        )

    def _compute_slots_locked(self, dt: datetime) -> ScheduleSnapshot:
        cfg = self.config
        sec_of_day = dt.hour * 3600.0 + dt.minute * 60.0 + dt.second + (dt.microsecond / 1e6)

        # 1. Check if currently inside any slot (including midnight-wrapping slots)
        for s_sec, e_sec in cfg.slots:
            if s_sec < e_sec:
                if s_sec <= sec_of_day < e_sec:
                    rem = e_sec - sec_of_day
                    w_id = f"{dt.strftime('%Y%m%d')}_{_format_hhmm_from_sec(s_sec).replace(':', '')}"
                    next_s, _ = self._find_next_slot_start(sec_of_day, cfg.slots)
                    return ScheduleSnapshot(
                        mode="slots",
                        state="ACTIVE",
                        is_detecting=True,
                        is_prewarm=False,
                        window_id=w_id,
                        remaining_sec=rem,
                        next_start_hhmm=_format_hhmm_from_sec(next_s),
                        override=self._override,
                        active_min=round((e_sec - s_sec) / 60.0, 2),
                        interval_min=cfg.interval_min,
                    )
            else:
                # Midnight wrap: e.g. 23:50 (85800) -> 00:10 (600)
                if sec_of_day >= s_sec:
                    rem = (SECONDS_PER_DAY - sec_of_day) + e_sec
                    w_id = f"{dt.strftime('%Y%m%d')}_{_format_hhmm_from_sec(s_sec).replace(':', '')}"
                    next_s, _ = self._find_next_slot_start(sec_of_day, cfg.slots)
                    return ScheduleSnapshot(
                        mode="slots",
                        state="ACTIVE",
                        is_detecting=True,
                        is_prewarm=False,
                        window_id=w_id,
                        remaining_sec=rem,
                        next_start_hhmm=_format_hhmm_from_sec(next_s),
                        override=self._override,
                        active_min=round(((SECONDS_PER_DAY - s_sec) + e_sec) / 60.0, 2),
                        interval_min=cfg.interval_min,
                    )
                if sec_of_day < e_sec:
                    rem = e_sec - sec_of_day
                    prev_day = (dt - timedelta(days=1)).strftime("%Y%m%d")
                    w_id = f"{prev_day}_{_format_hhmm_from_sec(s_sec).replace(':', '')}"
                    next_s, _ = self._find_next_slot_start(sec_of_day, cfg.slots)
                    return ScheduleSnapshot(
                        mode="slots",
                        state="ACTIVE",
                        is_detecting=True,
                        is_prewarm=False,
                        window_id=w_id,
                        remaining_sec=rem,
                        next_start_hhmm=_format_hhmm_from_sec(next_s),
                        override=self._override,
                        active_min=round(((SECONDS_PER_DAY - s_sec) + e_sec) / 60.0, 2),
                        interval_min=cfg.interval_min,
                    )

        # 2. Outside all slots: find closest upcoming slot start
        next_start_sec, wait_sec = self._find_next_slot_start(sec_of_day, cfg.slots)
        next_hhmm = _format_hhmm_from_sec(next_start_sec)

        if cfg.prewarm_sec > 0.0 and wait_sec <= cfg.prewarm_sec:
            return ScheduleSnapshot(
                mode="slots",
                state="PRE_WARMUP",
                is_detecting=False,
                is_prewarm=True,
                window_id=f"PRE_{next_hhmm.replace(':', '')}",
                remaining_sec=wait_sec,
                next_start_hhmm=next_hhmm,
                override=self._override,
                active_min=cfg.active_min,
                interval_min=cfg.interval_min,
            )

        return ScheduleSnapshot(
            mode="slots",
            state="STANDBY",
            is_detecting=False,
            is_prewarm=False,
            window_id="STANDBY",
            remaining_sec=wait_sec,
            next_start_hhmm=next_hhmm,
            override=self._override,
            active_min=cfg.active_min,
            interval_min=cfg.interval_min,
        )

    @staticmethod
    def _find_next_slot_start(sec_of_day: float, slots: tuple[tuple[int, int], ...]) -> tuple[int, float]:
        best_start = slots[0][0]
        best_wait = float(SECONDS_PER_DAY)
        for s_sec, _ in slots:
            delta = (s_sec - sec_of_day) % SECONDS_PER_DAY
            if delta == 0.0:
                delta = float(SECONDS_PER_DAY)
            if delta < best_wait:
                best_wait = delta
                best_start = s_sec
        return best_start, best_wait

    @classmethod
    def from_env_and_args(cls, args: Any = None) -> "DetectionScheduler":
        """Build a DetectionScheduler from parsed CLI args (with .env fallback)."""
        mode = getattr(args, "schedule_mode", None) or os.environ.get("DETECTION_SCHEDULE_MODE", "continuous")
        interval_min = getattr(args, "schedule_interval_min", None)
        if interval_min is None:
            interval_min = _env_float("SCHEDULE_INTERVAL_MIN", 60.0)
        active_min = getattr(args, "schedule_active_min", None)
        if active_min is None:
            active_min = _env_float("SCHEDULE_ACTIVE_MIN", 15.0)
        anchor_min = getattr(args, "schedule_anchor_min", None)
        if anchor_min is None:
            anchor_min = _env_float("SCHEDULE_ANCHOR_MINUTE", 0.0)
        slots_raw = getattr(args, "schedule_slots", None)
        if slots_raw is None:
            slots_raw = os.environ.get("SCHEDULE_SLOTS", "")
        prewarm_sec = getattr(args, "schedule_prewarm_sec", None)
        if prewarm_sec is None:
            prewarm_sec = _env_float("SCHEDULE_PREWARM_SEC", 2.0)
        standby_fps = getattr(args, "schedule_standby_fps", None)
        if standby_fps is None:
            standby_fps = _env_float("SCHEDULE_STANDBY_FPS", 20.0)
        motion_quota = getattr(args, "schedule_motion_quota", None)
        if motion_quota is None:
            motion_quota = os.environ.get("SCHEDULE_MOTION_QUOTA", "false").lower() in ("true", "1", "yes")

        cfg = DetectionScheduleConfig(
            mode=str(mode).strip().lower(),  # type: ignore[arg-type]
            interval_min=float(interval_min),
            active_min=float(active_min),
            anchor_minute=float(anchor_min),
            slots=parse_schedule_slots(slots_raw),
            prewarm_sec=float(prewarm_sec),
            standby_fps=float(standby_fps),
            motion_quota=bool(motion_quota),
        )
        return cls(cfg)


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default
