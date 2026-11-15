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


def default_database_path() -> Path:
    """Return the project-local review database used by direct launches."""
    return PROJECT_ROOT / "Database" / "AI_Slat_Rib_Conveyor.db"


def resolve_review_config(db_argument, reviewer_argument, prompt_reviewer) -> tuple[Path, str]:
    """Resolve optional CLI values without making the Run button depend on a terminal."""
    database_path = Path(db_argument).expanduser().resolve() if db_argument else default_database_path()
    reviewer = str(reviewer_argument or "").strip()
    if not reviewer:
        reviewer = str(prompt_reviewer() or "").strip()
    return database_path, reviewer


def prompt_text(title: str, prompt: str, initial_value: str = "") -> str | None:
    """Request text in a native popup so direct execution needs no terminal input."""
    try:
        import tkinter as tk
        from tkinter import simpledialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        root.update()
        try:
            return simpledialog.askstring(title, prompt, initialvalue=initial_value, parent=root)
        finally:
            root.destroy()
    except Exception as error:
        print(f"[!] Popup input tidak dapat dibuka: {error}")
        return None


def prompt_reviewer() -> str | None:
    return prompt_text("Rib Dataset Review", "Masukkan nama atau ID reviewer:")


def _fit(image: np.ndarray, target_height: int = 520) -> np.ndarray:
    height, width = image.shape[:2]
    if height == target_height:
        return image
    return cv2.resize(image, (max(1, int(width * target_height / height)), target_height))


def _render(
    sample: dict,
    progress: dict[str, int] | None = None,
    pending_note: str = "",
) -> np.ndarray | None:
    raw = cv2.imread(sample["raw_image_path"])
    annotated = cv2.imread(sample["annotated_image_path"])
    if raw is None or annotated is None:
        return None
    raw, annotated = _fit(raw), _fit(annotated)
    canvas = cv2.hconcat((raw, annotated))
    header = np.zeros((132, canvas.shape[1], 3), dtype=np.uint8)
    progress_text = f" | Progres: {progress['reviewed']}/{progress['total']}" if progress else ""
    undo_text = f" | [UNDO: {sample['manual_label']}]" if sample.get("manual_label") else ""
    title = (
        f"Dataset #{sample['dataset_id']} | Model: {sample['model_label']} | "
        f"Rib: {sample['rib_number']}{progress_text}{undo_text}"
    )
    metrics = (
        f"PV {sample['pv_flatness']:.2f} | SAG {sample['sagitta']:.2f} | "
        f"INNER {sample['inner_dev']:.2f} | TIP {sample['tip_dev']:.2f} | "
        f"COVERAGE {sample['coverage']:.2f} | CONF {sample['confidence']:.2f}"
    )
    cv2.putText(header, title, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(header, metrics, (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1, cv2.LINE_AA)
    active_note = pending_note or sample.get("note") or ""
    if active_note:
        cv2.putText(header, f"Catatan: {active_note[:90]}", (18, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 220, 140), 1, cv2.LINE_AA)
    cv2.putText(
        header,
        "[O] OK  [N] NG  [I] INVALID  [C] Catatan  [S] Skip  [U] Undo  [Q] Quit",
        (18, 114),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (140, 255, 140),
        1,
        cv2.LINE_AA,
    )
    return cv2.vconcat((header, canvas))


class ReviewSession:
    """Tracks in-session skip exclusions, undo history, and optional pending notes."""

    def __init__(self, store: DatasetReviewStore, reviewer: str):
        self.store = store
        self.reviewer = reviewer
        self.skipped_ids: set[int] = set()
        self.history: list[int] = []
        self.undo_id: int | None = None
        self.pending_note: str = ""

    def current_sample(self) -> dict | None:
        if self.undo_id is not None:
            sample = self.store.fetch_sample(self.undo_id)
            if sample is not None:
                return sample
            self.undo_id = None
        return self.store.next_unreviewed(exclude_ids=self.skipped_ids)

    def skip(self, dataset_id: int) -> None:
        if self.undo_id == dataset_id:
            self.undo_id = None
        else:
            self.skipped_ids.add(int(dataset_id))
        self.pending_note = ""

    def undo(self) -> bool:
        if not self.history:
            return False
        self.undo_id = self.history.pop()
        self.pending_note = ""
        return True

    def record_label(self, dataset_id: int, label: str) -> bool:
        is_edit = self.undo_id == dataset_id
        note = self.pending_note
        if is_edit and not note:
            existing = self.store.fetch_sample(dataset_id)
            note = str((existing or {}).get("note") or "")
        saved = self.store.save_review(
            dataset_id,
            label,
            self.reviewer,
            note,
            allow_update=is_edit,
        )
        if saved:
            self.history.append(int(dataset_id))
        self.undo_id = None
        self.pending_note = ""
        return saved


def main() -> int:
    parser = argparse.ArgumentParser(description="Review captured rib evidence on a local PC.")
    parser.add_argument("--db", help="Path to AI_Slat_Rib_Conveyor.db (default: Database proyek)")
    parser.add_argument("--reviewer", help="Name atau ID QA reviewer (default: diminta lewat popup)")
    args = parser.parse_args()
    database_path, reviewer = resolve_review_config(args.db, args.reviewer, prompt_reviewer)
    if not reviewer:
        print("Review dibatalkan: nama reviewer belum diisi.")
        return 0

    store = DatasetReviewStore(database_path)
    session = ReviewSession(store, reviewer)
    window = "Rib Dataset Review"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    try:
        while True:
            sample = session.current_sample()
            if sample is None:
                if session.skipped_ids:
                    print(f"Semua sampel tersisa telah di-skip dalam sesi ini ({len(session.skipped_ids)} sampel).")
                else:
                    print("Tidak ada sampel dengan dua file bukti yang belum direview.")
                return 0
            display = _render(sample, store.review_progress(), session.pending_note)
            if display is None:
                print(f"Lewati DatasetID {sample['dataset_id']}: gambar tidak dapat dibaca.")
                session.skip(sample["dataset_id"])
                continue
            cv2.imshow(window, display)
            key = cv2.waitKey(0) & 0xFF
            labels = {ord("o"): "OK", ord("n"): "NG", ord("i"): "INVALID"}
            if key in (ord("q"), 27):
                return 0
            if key == ord("s"):
                session.skip(sample["dataset_id"])
                continue
            if key == ord("u"):
                if not session.undo():
                    print("Tidak ada riwayat review di sesi ini untuk di-undo.")
                continue
            if key == ord("c"):
                note = prompt_text(
                    "Catatan Judgment (Opsional)",
                    f"Catatan untuk DatasetID {sample['dataset_id']} (lalu tekan O/N/I):",
                    initial_value=session.pending_note or str(sample.get("note") or ""),
                )
                if note is not None:
                    session.pending_note = note.strip()
                continue
            if key in labels:
                saved = session.record_label(sample["dataset_id"], labels[key])
                print(f"DatasetID {sample['dataset_id']}: {'tersimpan' if saved else 'gagal disimpan'}.")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
