"""
WebSocket Live Stream Server & Single-Slot Frame Hub for 3-Camera Conveyor Inspection.

Designed to run as an opt-in daemon thread inside the Vision Engine process without
altering or blocking AI inference threads or existing OpenCV desktop GUIs.
"""

from __future__ import annotations

import asyncio
import json
import socket
import struct
import threading
import time
from typing import Any

import cv2
import numpy as np

try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    websockets = None
    HAS_WEBSOCKETS = False

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


def _build_stream_metadata(cam_id: str, slot: dict[str, Any]) -> dict[str, Any]:
    """Build compact JSON telemetry header for a camera frame."""
    res = slot.get("result") or {}
    ins_meta = slot.get("inspector_meta") or {}
    fps_val = round(float(slot.get("fps", 0.0)), 1)

    if cam_id == "CAM-01":
        status = str(res.get("status", "WAIT")).upper()
        slat_res = res.get("slat_res") or {}
        rib_res = res.get("rib_res") or {}
        slat_no = ins_meta.get("slat_no")
        rib_no = ins_meta.get("rib_no")
        rib_conf = float(rib_res.get("confidence", 0.0) or 0.0)
        slat_conf = float(slat_res.get("confidence", slat_res.get("coverage", 0.0)) or 0.0)
        conf = round(max(rib_conf, slat_conf), 3) if (rib_conf > 0 or slat_conf > 0) else None
        if rib_res.get("is_ng"):
            obj = f"Rib #{rib_no} Deform" if rib_no else "Rib Deformation"
        elif slat_res.get("is_ng"):
            obj = f"Slat #{slat_no} Deform" if slat_no else "Plate Deformation"
        elif rib_res.get("is_rib_present"):
            obj = f"Slat #{slat_no or '-'} / Rib #{rib_no or '-'}"
        else:
            obj = f"Slat #{slat_no}" if slat_no else "Conveyor Plate / Rib"
        severity = "MEDIUM" if status == "NG" else "NONE"
    else:
        status = str(res.get("overall_result", "WAIT")).upper()
        comps = res.get("components") or []
        slat_no = res.get("slat_number")
        ng_names = [str(c.get("name", "")) for c in comps if str(c.get("result", "")).upper() == "NG"]
        confs = [float(c.get("confidence", 0.0) or 0.0) for c in comps if float(c.get("confidence", 0.0) or 0.0) > 0]
        conf = round(max(confs), 3) if confs else None
        if "PIN" in ng_names:
            obj = "Pin Missing / NG"
            severity = "CRITICAL"
        elif "BOLT" in ng_names:
            obj = "Bolt Missing / NG"
            severity = "HIGH"
        elif "LEKUKAN" in ng_names:
            obj = "Bracket Crack / NG"
            severity = "MEDIUM"
        else:
            obj = f"Slat #{slat_no} Fasteners" if slat_no else "Bolt / Pin / Bracket"
            severity = "HIGH" if status == "NG" else "NONE"

    return {
        "cam": cam_id,
        "status": status,
        "fps": fps_val,
        "object": obj,
        "conf": conf,
        "severity": severity,
        "slatNo": slat_no,
        "ts": round(time.time(), 3),
    }


class WebSocketStreamServer:
    """Asyncio Binary WebSocket server running in a dedicated daemon thread."""

    def __init__(
        self,
        hub: CameraFrameHub,
        host: str = "0.0.0.0",
        port: int = 8765,
        max_fps: float = 0.0,
        hud_mode: str = "clean",
        jpeg_quality: int = 65,
    ) -> None:
        self.hub = hub
        self.host = host
        self._requested_port = int(port)
        self._bound_port = int(port)
        self.max_fps = max(0.0, float(max_fps))
        self.hud_mode = "legacy" if str(hud_mode).lower() == "legacy" else "clean"
        self.jpeg_quality = int(max(30, min(95, int(jpeg_quality))))

        self._overlay_lock = threading.Lock()
        self._show_overlay = True
        self._clients: set[Any] = set()
        self._encoded_frame_count = 0

        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_async: asyncio.Event | None = None
        self._ready_event = threading.Event()
        self._start_ok = False
        self._stop_flag = False

    @property
    def port(self) -> int:
        return self._bound_port

    @property
    def client_count(self) -> int:
        return len(self._clients)

    @property
    def encoded_frame_count(self) -> int:
        return self._encoded_frame_count

    @property
    def show_overlay(self) -> bool:
        with self._overlay_lock:
            return self._show_overlay

    @show_overlay.setter
    def show_overlay(self, value: bool) -> None:
        with self._overlay_lock:
            self._show_overlay = bool(value)

    def start(self) -> bool:
        """Start the WebSocket server thread. Returns False without raising on error."""
        if not HAS_WEBSOCKETS:
            print("[WARN WS-STREAM] Paket 'websockets' belum terpasang. Live stream WebSocket dinonaktifkan.")
            return False
        if self._thread is not None and self._thread.is_alive():
            return True

        self._stop_flag = False
        self._start_ok = False
        self._ready_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="Thread-WSBroadcaster",
        )
        self._thread.start()
        self._ready_event.wait(timeout=4.0)
        return self._start_ok

    def stop(self) -> None:
        """Stop the WebSocket server and join its daemon thread."""
        self._stop_flag = True
        loop = self._loop
        stop_ev = self._stop_async
        if loop is not None and stop_ev is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(stop_ev.set)
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._serve_forever())
        except Exception as exc:
            if not self._ready_event.is_set():
                print(f"[WARN WS-STREAM] WebSocket server gagal berjalan di {self.host}:{self._requested_port} ({exc}).")
                self._start_ok = False
                self._ready_event.set()
        finally:
            try:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                loop.close()
            except Exception:
                pass
            self._loop = None

    async def _handle_client(self, websocket: Any) -> None:
        self._clients.add(websocket)
        try:
            sock = websocket.transport.get_extra_info("socket") if websocket.transport else None
            if sock is not None:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception:
            pass
        try:
            async for msg in websocket:
                if isinstance(msg, str):
                    try:
                        cmd = json.loads(msg)
                        if isinstance(cmd, dict) and "show_overlay" in cmd:
                            self.show_overlay = bool(cmd["show_overlay"])
                    except Exception:
                        pass
        except Exception:
            pass
        finally:
            self._clients.discard(websocket)

    def _encode_slot(self, cam_id: str, slot: dict[str, Any]) -> bytes | None:
        raw_frame = slot.get("raw_frame")
        if raw_frame is None:
            return None

        show_ov = self.show_overlay
        res = slot.get("result") or {}
        ins_meta = slot.get("inspector_meta") or {}

        if self.hud_mode == "legacy" and slot.get("legacy_frame") is not None:
            leg = slot["legacy_frame"]
            h_l, w_l = leg.shape[:2]
            if w_l > 0 and h_l > 0:
                target_h = int(round(STREAM_WIDTH * (h_l / float(w_l))))
                img = cv2.resize(leg, (STREAM_WIDTH, max(180, target_h)), interpolation=cv2.INTER_AREA)
            else:
                img = cv2.resize(raw_frame, (STREAM_WIDTH, STREAM_HEIGHT), interpolation=cv2.INTER_LINEAR)
        elif cam_id == "CAM-01":
            img = render_clean_top_360p(raw_frame, res, show_overlay=show_ov)
        else:
            img = render_clean_side_360p(
                raw_frame,
                res,
                ins_meta.get("master_roi"),
                ins_meta.get("polygon_pts"),
                show_overlay=show_ov,
            )

        ok, jb = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            return None
        self._encoded_frame_count += 1
        meta = _build_stream_metadata(cam_id, slot)
        return encode_ws_binary_packet(meta, jb.tobytes())

    async def _send_safe(self, ws: Any, packet: bytes) -> None:
        try:
            await asyncio.wait_for(ws.send(packet), timeout=0.25)
        except Exception:
            self._clients.discard(ws)

    async def _broadcast_loop(self, stop_ev: asyncio.Event) -> None:
        last_seqs: dict[str, int] = {}
        interval = (1.0 / self.max_fps) if self.max_fps > 0 else 0.0
        next_tick = time.perf_counter()

        while not stop_ev.is_set() and not self._stop_flag:
            if not self._clients:
                await asyncio.sleep(0.15)
                next_tick = time.perf_counter()
                continue

            now = time.perf_counter()
            if interval > 0 and now < next_tick:
                await asyncio.sleep(min(0.01, next_tick - now))
                continue
            if interval > 0:
                next_tick = now + interval

            sent_any = False
            for cam_id in self.hub.camera_ids():
                new_seq, slot = self.hub.snapshot_if_newer(cam_id, last_seqs.get(cam_id, 0))
                if slot is None:
                    continue
                last_seqs[cam_id] = new_seq
                packet = self._encode_slot(cam_id, slot)
                if packet is None:
                    continue
                sent_any = True
                clients_snapshot = list(self._clients)
                if clients_snapshot:
                    await asyncio.gather(
                        *(self._send_safe(ws, packet) for ws in clients_snapshot),
                        return_exceptions=True,
                    )

            if not sent_any:
                await asyncio.sleep(0.003)

    async def _serve_forever(self) -> None:
        stop_ev = asyncio.Event()
        self._stop_async = stop_ev
        if self._stop_flag:
            stop_ev.set()
            self._start_ok = False
            self._ready_event.set()
            return

        try:
            server = await websockets.serve(
                self._handle_client,
                self.host,
                self._requested_port,
                max_size=2 * 1024 * 1024,
                ping_interval=20,
                ping_timeout=20,
            )
        except Exception as exc:
            print(f"[WARN WS-STREAM] Gagal bind port WebSocket {self.host}:{self._requested_port}: {exc}")
            self._start_ok = False
            self._ready_event.set()
            return

        try:
            for sock in getattr(server, "sockets", []) or []:
                self._bound_port = int(sock.getsockname()[1])
                break
            self._start_ok = True
            self._ready_event.set()
            print(
                f"[*] [WS-STREAM] Binary WebSocket aktif di ws://{self.host}:{self._bound_port} "
                f"(mode={self.hud_mode}, max_fps={self.max_fps or 'uncapped'})"
            )
            broadcaster = asyncio.create_task(self._broadcast_loop(stop_ev))
            await stop_ev.wait()
            broadcaster.cancel()
            await asyncio.gather(broadcaster, return_exceptions=True)
        finally:
            server.close()
            await server.wait_closed()
