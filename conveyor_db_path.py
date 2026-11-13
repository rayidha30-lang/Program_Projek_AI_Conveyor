"""Canonical location of the shared conveyor SQLite database."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_db_path() -> str:
    """Prefer the current setting, then legacy setting, then project Database."""
    configured = os.environ.get("CONVEYOR_DB_PATH") or os.environ.get("SQLITE_DB_PATH")
    if configured:
        return os.path.abspath(configured)
    return str(Path(__file__).resolve().parent / "Database" / "AI_Slat_Rib_Conveyor.db")
