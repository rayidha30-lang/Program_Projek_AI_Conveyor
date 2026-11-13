"""Capture one reviewable evidence pair for each finalized rib."""

from __future__ import annotations

import json
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
    ):
        self.store = store
        self.capture_root = Path(capture_root)
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
        self.source_label = source_label
        self.camera_profile = camera_profile
        self.pipeline_version = pipeline_version
        self.min_active_frames = min_active_frames
        self._candidates: dict[tuple[int | None, int | None], dict[str, Any]] = {}
        self._finalized: dict[tuple[int | None, int | None], int] = {}

    @staticmethod
    def _model_label(rib_result: Mapping[str, Any]) -> str | None:
        status = str(rib_result.get("status", "")).upper()
        if status in {"OK", "NG"}:
            return status
        return None

    def observe(
        self,
        rib_result: Mapping[str, Any],
        raw_frame: Any,
        annotated_frame: Any,
        rib_number: int | None,
        slat_number: int | None,
    ) -> None:
        label = self._model_label(rib_result)
        active_frames = int(rib_result.get("active_rib_frames", 0) or 0)
        if label is None or active_frames < self.min_active_frames or raw_frame is None or annotated_frame is None:
            return
        key = (rib_number, slat_number)
        quality = (float(rib_result.get("coverage", 0.0) or 0.0), float(rib_result.get("confidence", 0.0) or 0.0))
        current = self._candidates.get(key)
        if current is not None and quality <= current["quality"]:
            return
        self._candidates[key] = {
            "quality": quality,
            "result": dict(rib_result),
            "raw_frame": raw_frame.copy(),
            "annotated_frame": annotated_frame.copy(),
        }

    def finalize(
        self,
        rib_result: Mapping[str, Any],
        rib_number: int | None,
        slat_number: int | None,
    ) -> int | None:
        key = (rib_number, slat_number)
        if key in self._finalized:
            return self._finalized[key]
        candidate = self._candidates.pop(key, None)
        label = self._model_label(rib_result)
        active_frames = int(rib_result.get("active_rib_frames", 0) or 0)
        if candidate is None or label is None or active_frames < self.min_active_frames:
            return None
        final_result = dict(candidate["result"])
        final_result.update({key: value for key, value in rib_result.items() if value is not None})
        dataset_id = self.store.insert_sample(self._sample_values(final_result, rib_number, slat_number))
        session_dir = self.capture_root / self.session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        raw_path = session_dir / f"rib_{dataset_id:06d}_raw.jpg"
        annotated_path = session_dir / f"rib_{dataset_id:06d}_annotated.jpg"
        if not (cv2.imwrite(str(raw_path), candidate["raw_frame"]) and cv2.imwrite(str(annotated_path), candidate["annotated_frame"])):
            self.store.delete_sample(dataset_id)
            return None
        self.store.attach_evidence(
            dataset_id,
            {
                "session_id": self.session_id,
                "source_label": self.source_label,
                "raw_image_path": str(raw_path.resolve()),
                "annotated_image_path": str(annotated_path.resolve()),
                "camera_profile": self.camera_profile,
                "pipeline_version": self.pipeline_version,
            },
        )
        self._finalized[key] = dataset_id
        return dataset_id

    def _sample_values(self, result: Mapping[str, Any], rib_number: int | None, slat_number: int | None) -> dict[str, Any]:
        contour = result.get("ys_contour")
        profile_json = None
        if contour is not None:
            try:
                profile_json = json.dumps([round(float(value), 2) for value in list(contour)[::4]])
            except (TypeError, ValueError):
                profile_json = None
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
        }
