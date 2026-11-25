"""Capture one reviewable evidence pair for each finalized rib."""

from __future__ import annotations

import json
import queue
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import cv2

from dataset_review_store import DatasetReviewStore


class RibDatasetRecorder:
    """Buffers the highest-quality frame until one physical rib is final."""

    def __init__(
        self,
        store: DatasetReviewStore,
        capture_root: str | Path,
        *,
        session_id: str | None = None,
        source_label: str = "CAM-01 TOP",
        camera_profile: str | None = None,
        pipeline_version: str = "root-runner",
        min_active_frames: int = 15,
        async_write: bool = False,
    ):
        self.store = store
        self.store.ensure_schema()
        self.capture_root = Path(capture_root)
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
        self.source_label = source_label
        self.camera_profile = camera_profile
        self.pipeline_version = pipeline_version
        self.min_active_frames = min_active_frames
        self.async_write = bool(async_write)
        self._candidates: dict[int | None, dict[str, Any]] = {}
        self._finalized: dict[int | None, int | None] = {}
        self._queue: queue.Queue[dict[str, Any]] | None = None
        self._stop_event: threading.Event | None = None
        self._worker_thread: threading.Thread | None = None
        if self.async_write:
            self._queue = queue.Queue(maxsize=128)
            self._stop_event = threading.Event()
            self._worker_thread = threading.Thread(
                target=self._worker_loop, daemon=True, name="RibDatasetRecorderWorker"
            )
            self._worker_thread.start()

    def start_new_session(self, session_id: str | None = None) -> None:
        """Reset in-memory deduplication after the operator resets the rib counter."""
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
        self._candidates.clear()
        self._finalized.clear()

    @staticmethod
    def _model_label(rib_result: Mapping[str, Any]) -> str | None:
        status = str(rib_result.get("status", "")).upper()
        if status in {"OK", "NG"}:
            return status
        return None

    @staticmethod
    def _quality_key(rib_result: Mapping[str, Any]) -> tuple[int, float, float]:
        return (
            1 if rib_result.get("final_consensus_ready") else 0,
            float(rib_result.get("coverage", 0.0) or 0.0),
            float(rib_result.get("confidence", 0.0) or 0.0),
        )

    @staticmethod
    def _is_observable_frame(rib_result: Mapping[str, Any]) -> bool:
        """Reject Transit Hysteresis buffer frames that lack an active rib box or contour."""
        reason = str(rib_result.get("detection_reason", "") or "")
        if "transit hysteresis" in reason.lower():
            return False
        if "active_box" in rib_result and rib_result.get("active_box") is None:
            return False
        contour = rib_result.get("ys_contour")
        if contour is None or len(contour) == 0:
            return False
        return True

    def should_observe(self, rib_result: Mapping[str, Any], rib_number: int | None) -> bool:
        """Return True only when this frame improves on the buffered candidate."""
        if rib_number in self._finalized:
            return False
        label = self._model_label(rib_result)
        active_frames = int(rib_result.get("active_rib_frames", 0) or 0)
        if label is None or active_frames < self.min_active_frames:
            return False
        if not self._is_observable_frame(rib_result):
            return False
        current = self._candidates.get(rib_number)
        return current is None or self._quality_key(rib_result) > current["quality"]

    def observe(
        self,
        rib_result: Mapping[str, Any],
        raw_frame: Any,
        annotated_frame: Any,
        rib_number: int | None,
        slat_number: int | None,
    ) -> None:
        if raw_frame is None or annotated_frame is None or not self.should_observe(rib_result, rib_number):
            return
        self._candidates[rib_number] = {
            "quality": self._quality_key(rib_result),
            "result": dict(rib_result),
            "slat_number": slat_number,
            "raw_frame": raw_frame.copy(),
            "annotated_frame": annotated_frame.copy(),
        }

    def finalize(
        self,
        rib_result: Mapping[str, Any],
        rib_number: int | None,
        slat_number: int | None,
    ) -> int | None:
        if rib_number in self._finalized:
            return self._finalized[rib_number]
        candidate = self._candidates.pop(rib_number, None)
        label = self._model_label(rib_result)
        active_frames = int(rib_result.get("active_rib_frames", 0) or 0)
        if candidate is None or label is None or active_frames < self.min_active_frames:
            return None

        # Keep the geometry metrics from the best-frame candidate so DB values
        # always match the overlay baked into annotated_frame; only sync final
        # verdict and total active frame count.
        final_result = dict(candidate["result"])
        final_result["status"] = label
        best_frames = int(final_result.get("active_rib_frames", 0) or 0)
        final_result["active_rib_frames"] = max(best_frames, active_frames)
        resolved_slat = candidate["slat_number"] if candidate.get("slat_number") is not None else slat_number

        payload = {
            "sample_values": self._sample_values(
                final_result, rib_number, resolved_slat, raw_frame=candidate["raw_frame"]
            ),
            "raw_frame": candidate["raw_frame"],
            "annotated_frame": candidate["annotated_frame"],
            "session_id": self.session_id,
        }
        if self.async_write and self._queue is not None:
            try:
                self._queue.put_nowait(payload)
                self._finalized[rib_number] = None
            except queue.Full:
                pass
            return None

        dataset_id = self._persist_payload(payload)
        if dataset_id is not None:
            self._finalized[rib_number] = dataset_id
        return dataset_id

    def _persist_payload(self, payload: Mapping[str, Any]) -> int | None:
        dataset_id = self.store.insert_sample(payload["sample_values"])
        session_dir = self.capture_root / str(payload["session_id"])
        session_dir.mkdir(parents=True, exist_ok=True)
        raw_path = session_dir / f"rib_{dataset_id:06d}_raw.jpg"
        annotated_path = session_dir / f"rib_{dataset_id:06d}_annotated.jpg"
        if not (
            cv2.imwrite(str(raw_path), payload["raw_frame"])
            and cv2.imwrite(str(annotated_path), payload["annotated_frame"])
        ):
            self.store.delete_sample(dataset_id)
            return None
        self.store.attach_evidence(
            dataset_id,
            {
                "session_id": str(payload["session_id"]),
                "source_label": self.source_label,
                "raw_image_path": str(raw_path.resolve()),
                "annotated_image_path": str(annotated_path.resolve()),
                "camera_profile": self.camera_profile,
                "pipeline_version": self.pipeline_version,
            },
        )
        return dataset_id

    def _worker_loop(self) -> None:
        assert self._queue is not None and self._stop_event is not None
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                payload = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._persist_payload(payload)
            except Exception as error:
                print(f"[WARN RibDatasetRecorder] Gagal menyimpan bukti rib: {error}")
            finally:
                self._queue.task_done()

    def flush(self) -> None:
        """Wait until all queued async evidence writes finish."""
        if self._queue is not None:
            self._queue.join()

    def close(self) -> None:
        """Drain pending writes and stop the background worker cleanly."""
        if self._stop_event is not None:
            self._stop_event.set()
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)

    def _sample_values(
        self,
        result: Mapping[str, Any],
        rib_number: int | None,
        slat_number: int | None,
        raw_frame: Any = None,
    ) -> dict[str, Any]:
        contour = result.get("ys_contour")
        profile_json = None
        if contour is not None:
            try:
                profile_json = json.dumps([round(float(value), 2) for value in list(contour)[::4]])
            except (TypeError, ValueError):
                profile_json = None
        frame_h = result.get("frame_height")
        if frame_h is None and raw_frame is not None and hasattr(raw_frame, "shape") and len(raw_frame.shape) >= 2:
            frame_h = int(raw_frame.shape[0])
        res_scale = result.get("res_scale")
        if res_scale is None and frame_h is not None and float(frame_h) > 0:
            res_scale = float(min(1.80, max(0.75, 1080.0 / float(max(240, int(frame_h))))))
        span_px = result.get("raw_span_px")
        metrics_scaled = 1 if ("res_scale" in result or "raw_span_px" in result) else 0
        return {
            "rib_number": rib_number,
            "slat_number": slat_number,
            "label": self._model_label(result),
            "tracker_engine": result.get("tracker_engine", "UNKNOWN"),
            "pv_flatness": result.get("pv_flatness", 0.0),
            "sagitta": result.get("sagitta", 0.0),
            "inner_dev": result.get("inner_dev", 0.0),
            "tip_dev": result.get("tip_dev", 0.0),
            "coverage": result.get("coverage", 0.0),
            "active_frames": result.get("active_rib_frames", 0),
            "confidence": result.get("confidence", 0.0),
            "profile_json": profile_json,
            "camera_profile": self.camera_profile,
            "frame_height": int(frame_h) if frame_h is not None else None,
            "span_px": float(span_px) if span_px is not None else None,
            "res_scale": float(res_scale) if res_scale is not None else None,
            "metrics_scaled": metrics_scaled,
        }
