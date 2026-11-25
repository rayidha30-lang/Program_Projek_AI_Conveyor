"""
WebSocket Live Stream Server & Single-Slot Frame Hub for 3-Camera Conveyor Inspection.

Designed to run as an opt-in daemon thread inside the Vision Engine process without
altering or blocking AI inference threads or existing OpenCV desktop GUIs.
"""

from __future__ import annotations

import json
import struct
import threading
from typing import Any

import cv2
import numpy as np

STREAM_WIDTH = 640
STREAM_HEIGHT = 360
_CANONICAL_SLAT_DST = np.array(
    [[0.0, 0.0], [999.0, 0.0], [999.0, 139.0], [0.0, 139.0]],
    dtype=np.float32,
)


def encode_ws_binary_packet(meta: dict[str, Any], jpeg_bytes: bytes) -> bytes:
    """Pack JSON metadata and JPEG bytes into `[4-byte uint32 BE len][JSON][JPEG]`."""
    meta_bytes = json.dumps(meta, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return struct.pack("!I", len(meta_bytes)) + meta_bytes + jpeg_bytes


def decode_ws_binary_packet(packet: bytes) -> tuple[dict[str, Any], bytes]:
    """Unpack `[4-byte uint32 BE len][JSON][JPEG]` into `(meta_dict, jpeg_bytes)`."""
    if len(packet) < 4:
        raise ValueError("Packet too short to contain header length")
    meta_len = struct.unpack("!I", packet[:4])[0]
    if len(packet) < 4 + meta_len:
        raise ValueError("Packet truncated inside metadata header")
    meta = json.loads(packet[4 : 4 + meta_len].decode("utf-8"))
    return meta, packet[4 + meta_len :]


class CameraFrameHub:
    """Thread-safe single-slot atomic frame buffer per camera (overwrites stale frames)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._slots: dict[str, dict[str, Any]] = {}

    def publish(
        self,
        cam_id: str,
        raw_frame: np.ndarray | None,
        result: dict[str, Any] | None,
        fps: float,
        inspector_meta: dict[str, Any] | None = None,
        legacy_frame: np.ndarray | None = None,
    ) -> None:
        """Store latest frame reference and metadata non-blockingly for `cam_id`."""
        if raw_frame is None:
            return
        with self._lock:
            prev_seq = self._slots.get(cam_id, {}).get("seq", 0)
            self._slots[cam_id] = {
                "seq": prev_seq + 1,
                "raw_frame": raw_frame,
                "result": result or {},
                "fps": float(fps),
                "inspector_meta": inspector_meta or {},
                "legacy_frame": legacy_frame,
            }

    def snapshot_if_newer(self, cam_id: str, last_seq: int) -> tuple[int, dict[str, Any] | None]:
        """Return `(new_seq, slot_dict)` if `cam_id` has a newer frame than `last_seq`."""
        with self._lock:
            slot = self._slots.get(cam_id)
            if slot is None or slot["seq"] == last_seq:
                return last_seq, None
            return slot["seq"], dict(slot)

    def camera_ids(self) -> list[str]:
        """Return ordered list of registered camera IDs."""
        with self._lock:
            return sorted(self._slots.keys())


def _status_color_bgr(status: str) -> tuple[int, int, int]:
    st = str(status or "WAIT").upper()
    if st == "NG":
        return (0, 30, 255)
    if st == "OK":
        return (0, 230, 0)
    if st == "HOLD":
        return (0, 220, 255)
    return (0, 165, 255)


def render_clean_top_360p(
    frame_bgr: np.ndarray,
    inspect_res: dict[str, Any] | None,
    show_overlay: bool = True,
) -> np.ndarray:
    """Resize CAM-01 frame to 640x360 (16:9) and draw core slat/rib detection geometry."""
    h_f, w_f = frame_bgr.shape[:2]
    canvas = cv2.resize(frame_bgr, (STREAM_WIDTH, STREAM_HEIGHT), interpolation=cv2.INTER_LINEAR)
    if not show_overlay or not inspect_res or h_f <= 0 or w_f <= 0:
        return canvas

    sx = STREAM_WIDTH / float(w_f)
    sy = STREAM_HEIGHT / float(h_f)

    slat_res = inspect_res.get("slat_res") or {}
    rib_res = inspect_res.get("rib_res") or {}

    poly_px = slat_res.get("polygon_px")
    if poly_px is not None and len(poly_px) >= 3:
        poly_arr = np.asarray(poly_px, dtype=np.float32)
        poly_s = (poly_arr * [sx, sy]).astype(np.int32)
        slat_col = _status_color_bgr(slat_res.get("display_status", "WAIT"))
        cv2.polylines(canvas, [poly_s], True, slat_col, 2, cv2.LINE_AA)

        xs_s = slat_res.get("xs_contour")
        ys_s = slat_res.get("ys_contour")
        if xs_s is not None and ys_s is not None and len(xs_s) > 10 and len(poly_arr) >= 4:
            try:
                m_inv = cv2.getPerspectiveTransform(_CANONICAL_SLAT_DST, poly_arr[:4])
                unwarp_pts = np.column_stack([xs_s, ys_s]).reshape(-1, 1, 2).astype(np.float32)
                cam_pts = cv2.perspectiveTransform(unwarp_pts, m_inv).reshape(-1, 2)
                cam_pts_s = (cam_pts * [sx, sy]).astype(np.int32)
                cv2.polylines(canvas, [cam_pts_s], False, slat_col, 2, cv2.LINE_AA)
            except Exception:
                pass

    box = rib_res.get("active_box")
    if rib_res.get("is_rib_present") and box is not None and len(box) >= 4:
        rx1, ry1 = int(box[0] * sx), int(box[1] * sy)
        rx2, ry2 = int(box[2] * sx), int(box[3] * sy)
        rib_col = (0, 30, 255) if rib_res.get("is_ng") else (255, 170, 0)
        cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), rib_col, 2)

    xs_r = rib_res.get("xs_contour")
    ys_r = rib_res.get("ys_contour")
    if xs_r is not None and ys_r is not None and len(xs_r) > 4:
        line_col = (0, 30, 255) if rib_res.get("is_ng") else (0, 230, 0)
        step = max(1, len(xs_r) // 60)
        pts_r = np.column_stack([xs_r[::step], ys_r[::step]]).astype(np.float32)
        pts_r_s = (pts_r * [sx, sy]).astype(np.int32)
        cv2.polylines(canvas, [pts_r_s], False, line_col, 2, cv2.LINE_AA)

    return canvas


def render_clean_side_360p(
    frame_bgr: np.ndarray,
    payload: dict[str, Any] | None,
    master_roi: list[float] | None,
    polygon_pts: list[list[float]] | None,
    show_overlay: bool = True,
) -> np.ndarray:
    """Resize CAM-02/03 frame to 640x360 (16:9) and draw ROI, polygon, and component boxes."""
    h_f, w_f = frame_bgr.shape[:2]
    canvas = cv2.resize(frame_bgr, (STREAM_WIDTH, STREAM_HEIGHT), interpolation=cv2.INTER_LINEAR)
    if not show_overlay or h_f <= 0 or w_f <= 0:
        return canvas

    sx = STREAM_WIDTH / float(w_f)
    sy = STREAM_HEIGHT / float(h_f)

    if master_roi is not None and len(master_roi) >= 4:
        rx1 = int(master_roi[0] * STREAM_WIDTH)
        ry1 = int(master_roi[1] * STREAM_HEIGHT)
        rx2 = int(master_roi[2] * STREAM_WIDTH)
        ry2 = int(master_roi[3] * STREAM_HEIGHT)
        cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), (0, 220, 255), 1)

    status = str((payload or {}).get("overall_result", "WAIT")).upper()
    if polygon_pts is not None and len(polygon_pts) >= 3:
        pts_s = np.array(
            [[int(p[0] * STREAM_WIDTH), int(p[1] * STREAM_HEIGHT)] for p in polygon_pts],
            dtype=np.int32,
        )
        poly_col = _status_color_bgr(status)
        cv2.polylines(canvas, [pts_s], True, poly_col, 2, cv2.LINE_AA)

    if payload:
        for key in ("bolts", "pins"):
            for item in payload.get(key) or []:
                box = item.get("box")
                if box is not None and len(box) >= 4:
                    bx1, by1 = int(box[0] * sx), int(box[1] * sy)
                    bx2, by2 = int(box[2] * sx), int(box[3] * sy)
                    col = (0, 30, 255) if item.get("is_ng") else (0, 230, 0)
                    cv2.rectangle(canvas, (bx1, by1), (bx2, by2), col, 2)

    return canvas
