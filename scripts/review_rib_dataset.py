"""Review evidence-backed rib samples locally using OpenCV keyboard controls."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from dataset_review_store import DatasetReviewStore


def _fit(image: np.ndarray, target_height: int = 520) -> np.ndarray:
    height, width = image.shape[:2]
    if height == target_height:
        return image
    return cv2.resize(image, (max(1, int(width * target_height / height)), target_height))


def _render(sample: dict) -> np.ndarray | None:
    raw = cv2.imread(sample["raw_image_path"])
    annotated = cv2.imread(sample["annotated_image_path"])
    if raw is None or annotated is None:
        return None
    raw, annotated = _fit(raw), _fit(annotated)
    canvas = cv2.hconcat((raw, annotated))
    header = np.zeros((118, canvas.shape[1], 3), dtype=np.uint8)
    metrics = (
        f"PV {sample['pv_flatness']:.2f} | SAG {sample['sagitta']:.2f} | "
        f"INNER {sample['inner_dev']:.2f} | TIP {sample['tip_dev']:.2f} | "
        f"COVERAGE {sample['coverage']:.2f} | CONF {sample['confidence']:.2f}"
    )
    cv2.putText(header, f"Dataset #{sample['dataset_id']} | Model: {sample['model_label']} | Rib: {sample['rib_number']}", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(header, metrics, (18, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(header, "[O] OK   [N] NG   [I] INVALID   [S] Skip   [Q] Quit", (18, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (140, 255, 140), 1, cv2.LINE_AA)
    return cv2.vconcat((header, canvas))


def main() -> int:
    parser = argparse.ArgumentParser(description="Review captured rib evidence on a local PC.")
    parser.add_argument("--db", required=True, help="Path to AI_Slat_Rib_Conveyor.db")
    parser.add_argument("--reviewer", required=True, help="Name or ID of the QA reviewer")
    args = parser.parse_args()
    store = DatasetReviewStore(args.db)
    window = "Rib Dataset Review"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    try:
        while True:
            sample = store.next_unreviewed()
            if sample is None:
                print("Tidak ada sampel dengan dua file bukti yang belum direview.")
                return 0
            display = _render(sample)
            if display is None:
                print(f"Lewati DatasetID {sample['dataset_id']}: gambar tidak dapat dibaca.")
                continue
            cv2.imshow(window, display)
            key = cv2.waitKey(0) & 0xFF
            labels = {ord("o"): "OK", ord("n"): "NG", ord("i"): "INVALID"}
            if key in (ord("q"), 27):
                return 0
            if key in (ord("s"),):
                continue
            if key in labels:
                note = input(f"Catatan untuk DatasetID {sample['dataset_id']} (opsional): ").strip()
                saved = store.save_review(sample["dataset_id"], labels[key], args.reviewer, note)
                print(f"DatasetID {sample['dataset_id']}: {'tersimpan' if saved else 'sudah direview'}.")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
