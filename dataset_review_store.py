"""SQLite storage for evidence-backed rib dataset review."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping


MANUAL_LABELS = {"OK", "NG", "INVALID"}
MODEL_LABELS = {"OK", "NG"}


class DatasetReviewStore:
    """Owns the review tables while keeping the existing model label intact."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(Path(db_path).expanduser().resolve())

    @contextmanager
    def _connection(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA busy_timeout = 30000")
            conn.execute("PRAGMA synchronous = NORMAL")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def ensure_schema(self) -> None:
        with self._connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS Rib_Dataset (
                    DatasetID INTEGER PRIMARY KEY AUTOINCREMENT,
                    RecordedAt DATETIME NOT NULL DEFAULT (datetime('now','localtime')),
                    RibNumber INTEGER,
                    SlatNumber INTEGER,
                    Label TEXT NOT NULL CHECK (Label IN ('OK', 'NG')),
                    TrackerEngine TEXT,
                    PVFlatness REAL,
                    Sagitta REAL,
                    InnerDev REAL,
                    TipDev REAL,
                    Coverage REAL,
                    ActiveFrames INTEGER,
                    Confidence REAL,
                    ProfileJson TEXT,
                    CameraProfile TEXT
                );
                CREATE TABLE IF NOT EXISTS Rib_Dataset_Evidence (
                    EvidenceID INTEGER PRIMARY KEY AUTOINCREMENT,
                    DatasetID INTEGER NOT NULL UNIQUE,
                    SessionID TEXT NOT NULL,
                    SourceLabel TEXT,
                    RawImagePath TEXT NOT NULL,
                    AnnotatedImagePath TEXT NOT NULL,
                    CameraProfile TEXT,
                    PipelineVersion TEXT,
                    CapturedAt DATETIME NOT NULL DEFAULT (datetime('now','localtime')),
                    FOREIGN KEY (DatasetID) REFERENCES Rib_Dataset(DatasetID) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS Rib_Dataset_Review (
                    ReviewID INTEGER PRIMARY KEY AUTOINCREMENT,
                    DatasetID INTEGER NOT NULL UNIQUE,
                    ManualLabel TEXT NOT NULL CHECK (ManualLabel IN ('OK', 'NG', 'INVALID')),
                    Reviewer TEXT NOT NULL,
                    Note TEXT,
                    ReviewedAt DATETIME NOT NULL DEFAULT (datetime('now','localtime')),
                    FOREIGN KEY (DatasetID) REFERENCES Rib_Dataset(DatasetID) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_rib_dataset_review_lookup
                    ON Rib_Dataset_Review (DatasetID);
                """
            )

    def insert_sample(self, sample: Mapping[str, Any]) -> int:
        label = str(sample["label"]).upper()
        if label not in MODEL_LABELS:
            raise ValueError("model label must be OK or NG")
        self.ensure_schema()
        columns = (
            "RibNumber, SlatNumber, Label, TrackerEngine, PVFlatness, Sagitta, "
            "InnerDev, TipDev, Coverage, ActiveFrames, Confidence, ProfileJson, CameraProfile"
        )
        values = (
            sample.get("rib_number"), sample.get("slat_number"), label,
            sample.get("tracker_engine"), sample.get("pv_flatness"), sample.get("sagitta"),
            sample.get("inner_dev"), sample.get("tip_dev"), sample.get("coverage"),
            sample.get("active_frames"), sample.get("confidence"), sample.get("profile_json"),
            sample.get("camera_profile"),
        )
        with self._connection() as conn:
            cursor = conn.execute(
                f"INSERT INTO Rib_Dataset ({columns}) VALUES ({', '.join('?' for _ in values)})",
                values,
            )
            return int(cursor.lastrowid)

    def attach_evidence(self, dataset_id: int, evidence: Mapping[str, Any]) -> None:
        self.ensure_schema()
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO Rib_Dataset_Evidence (
                    DatasetID, SessionID, SourceLabel, RawImagePath, AnnotatedImagePath,
                    CameraProfile, PipelineVersion
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(DatasetID) DO UPDATE SET
                    SessionID=excluded.SessionID,
                    SourceLabel=excluded.SourceLabel,
                    RawImagePath=excluded.RawImagePath,
                    AnnotatedImagePath=excluded.AnnotatedImagePath,
                    CameraProfile=excluded.CameraProfile,
                    PipelineVersion=excluded.PipelineVersion
                """,
                (
                    dataset_id, evidence["session_id"], evidence.get("source_label"),
                    evidence["raw_image_path"], evidence["annotated_image_path"],
                    evidence.get("camera_profile"), evidence.get("pipeline_version"),
                ),
            )

    def delete_sample(self, dataset_id: int) -> None:
        """Remove a row when its required evidence files could not be written."""
        self.ensure_schema()
        with self._connection() as conn:
            conn.execute("DELETE FROM Rib_Dataset WHERE DatasetID = ?", (dataset_id,))

    def evidence_for(self, dataset_id: int) -> dict[str, Any] | None:
        self.ensure_schema()
        with self._connection() as conn:
            row = conn.execute(
                "SELECT DatasetID, RawImagePath, AnnotatedImagePath FROM Rib_Dataset_Evidence WHERE DatasetID = ?",
                (dataset_id,),
            ).fetchone()
        if row is None:
            return None
        return {"dataset_id": row[0], "raw_image_path": row[1], "annotated_image_path": row[2]}

    def sample_count(self) -> int:
        self.ensure_schema()
        with self._connection() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM Rib_Dataset").fetchone()[0])

    def evidence_count(self) -> int:
        self.ensure_schema()
        with self._connection() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM Rib_Dataset_Evidence").fetchone()[0])

    def save_review(self, dataset_id: int, manual_label: str, reviewer: str, note: str = "") -> bool:
        label = str(manual_label).upper()
        if label not in MANUAL_LABELS:
            raise ValueError("manual label must be OK, NG, or INVALID")
        if not str(reviewer).strip():
            raise ValueError("reviewer is required")
        self.ensure_schema()
        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO Rib_Dataset_Review (DatasetID, ManualLabel, Reviewer, Note)
                VALUES (?, ?, ?, ?)
                """,
                (dataset_id, label, reviewer.strip(), note.strip()),
            )
            return cursor.rowcount == 1
