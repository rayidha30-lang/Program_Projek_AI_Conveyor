"""SQLite storage for evidence-backed rib dataset review."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping


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

    _SAMPLE_SELECT = """
        SELECT d.DatasetID, d.RecordedAt, d.RibNumber, d.SlatNumber, d.Label,
               d.PVFlatness, d.Sagitta, d.InnerDev, d.TipDev, d.Coverage,
               d.ActiveFrames, d.Confidence, d.CameraProfile,
               e.RawImagePath, e.AnnotatedImagePath,
               r.ManualLabel, r.Note
        FROM Rib_Dataset AS d
        INNER JOIN Rib_Dataset_Evidence AS e ON e.DatasetID = d.DatasetID
        LEFT JOIN Rib_Dataset_Review AS r ON r.DatasetID = d.DatasetID
    """

    @staticmethod
    def _row_to_sample(row: tuple[Any, ...]) -> dict[str, Any]:
        keys = (
            "dataset_id", "recorded_at", "rib_number", "slat_number", "model_label",
            "pv_flatness", "sagitta", "inner_dev", "tip_dev", "coverage",
            "active_frames", "confidence", "camera_profile", "raw_image_path",
            "annotated_image_path", "manual_label", "note",
        )
        return dict(zip(keys, row))

    @staticmethod
    def _has_both_evidence_files(sample: Mapping[str, Any]) -> bool:
        return Path(sample["raw_image_path"]).is_file() and Path(sample["annotated_image_path"]).is_file()

    def next_unreviewed(self, exclude_ids: Iterable[int] = ()) -> dict[str, Any] | None:
        """Return the oldest reviewable row, excluding missing files, skipped IDs, and prior reviews."""
        self.ensure_schema()
        excluded = {int(item) for item in exclude_ids}
        with self._connection() as conn:
            rows = conn.execute(
                f"""
                {self._SAMPLE_SELECT}
                WHERE r.DatasetID IS NULL
                ORDER BY d.RecordedAt ASC, d.DatasetID ASC
                """
            ).fetchall()
        for row in rows:
            item = self._row_to_sample(row)
            if item["dataset_id"] in excluded:
                continue
            if self._has_both_evidence_files(item):
                return item
        return None

    def fetch_sample(self, dataset_id: int) -> dict[str, Any] | None:
        """Return one evidence-backed sample by ID, including any existing review."""
        self.ensure_schema()
        with self._connection() as conn:
            row = conn.execute(
                f"{self._SAMPLE_SELECT} WHERE d.DatasetID = ?",
                (int(dataset_id),),
            ).fetchone()
        if row is None:
            return None
        item = self._row_to_sample(row)
        return item if self._has_both_evidence_files(item) else None

    def review_progress(self) -> dict[str, int]:
        """Return counts of evidence-backed samples and completed reviews."""
        self.ensure_schema()
        with self._connection() as conn:
            total = int(conn.execute("SELECT COUNT(*) FROM Rib_Dataset_Evidence").fetchone()[0])
            reviewed = int(
                conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM Rib_Dataset_Review AS r
                    INNER JOIN Rib_Dataset_Evidence AS e ON e.DatasetID = r.DatasetID
                    """
                ).fetchone()[0]
            )
        return {"total": total, "reviewed": reviewed, "remaining": max(0, total - reviewed)}

    def evaluation_rows(self) -> list[dict[str, Any]]:
        """Return reviewed model/manual label pairs with metrology metrics."""
        self.ensure_schema()
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT d.DatasetID, d.Label, r.ManualLabel, d.CameraProfile,
                       d.PVFlatness, d.Sagitta, d.InnerDev, d.TipDev,
                       d.Coverage, d.ActiveFrames, d.Confidence,
                       d.RibNumber, d.SlatNumber, r.Note
                FROM Rib_Dataset AS d
                INNER JOIN Rib_Dataset_Review AS r ON r.DatasetID = d.DatasetID
                ORDER BY r.ReviewedAt ASC, d.DatasetID ASC
                """
            ).fetchall()
        return [
            {
                "dataset_id": row[0],
                "model_label": row[1],
                "manual_label": row[2],
                "camera_profile": row[3] or "UNSPECIFIED",
                "pv_flatness": row[4],
                "sagitta": row[5],
                "inner_dev": row[6],
                "tip_dev": row[7],
                "coverage": row[8],
                "active_frames": row[9],
                "confidence": row[10],
                "rib_number": row[11],
                "slat_number": row[12],
                "note": row[13] or "",
            }
            for row in rows
        ]

    def save_review(
        self,
        dataset_id: int,
        manual_label: str,
        reviewer: str,
        note: str = "",
        *,
        allow_update: bool = False,
    ) -> bool:
        label = str(manual_label).upper()
        if label not in MANUAL_LABELS:
            raise ValueError("manual label must be OK, NG, or INVALID")
        if not str(reviewer).strip():
            raise ValueError("reviewer is required")
        self.ensure_schema()
        with self._connection() as conn:
            evidence = conn.execute(
                "SELECT RawImagePath, AnnotatedImagePath FROM Rib_Dataset_Evidence WHERE DatasetID = ?",
                (dataset_id,),
            ).fetchone()
            if evidence is None or not Path(evidence[0]).is_file() or not Path(evidence[1]).is_file():
                return False
            if allow_update:
                cursor = conn.execute(
                    """
                    INSERT INTO Rib_Dataset_Review (DatasetID, ManualLabel, Reviewer, Note)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(DatasetID) DO UPDATE SET
                        ManualLabel = excluded.ManualLabel,
                        Reviewer = excluded.Reviewer,
                        Note = excluded.Note,
                        ReviewedAt = datetime('now', 'localtime')
                    """,
                    (dataset_id, label, reviewer.strip(), note.strip()),
                )
            else:
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO Rib_Dataset_Review (DatasetID, ManualLabel, Reviewer, Note)
                    VALUES (?, ?, ?, ?)
                    """,
                    (dataset_id, label, reviewer.strip(), note.strip()),
                )
            return cursor.rowcount >= 1
