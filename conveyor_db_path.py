"""Canonical location of the shared conveyor SQLite database and captures."""

from __future__ import annotations

import os
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def load_project_env() -> None:
    """Load Program/.env into os.environ without overwriting existing variables."""
    env_path = _project_root() / ".env"
    if not env_path.is_file():
        return
    try:
        with env_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k:
                    os.environ.setdefault(k, v)
    except Exception:
        pass


def resolve_db_path() -> str:
    """Prefer the current setting, then legacy setting, then project Database."""
    project_root = _project_root()
    configured = os.environ.get("CONVEYOR_DB_PATH") or os.environ.get("SQLITE_DB_PATH")
    if configured:
        cfg_path = Path(configured)
        if not cfg_path.is_absolute():
            cfg_path = project_root / cfg_path
        return str(cfg_path.resolve())
    return str(project_root / "Database" / "AI_Slat_Rib_Conveyor.db")


def resolve_captures_dir(db_path: str | None = None) -> str:
    """Resolve captures directory relative to Program root instead of CWD."""
    project_root = _project_root()
    configured = os.environ.get("CAPTURES_DIR")
    if configured:
        cfg_path = Path(configured)
        if not cfg_path.is_absolute():
            cfg_path = project_root / cfg_path
        return str(cfg_path.resolve())
    target_db = Path(db_path) if db_path else Path(resolve_db_path())
    return str((target_db.resolve().parent / "captures"))
