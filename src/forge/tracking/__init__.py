"""Built-in experiment tracking — zero-config SQLite backend.

Public API:
    get_db()               → ForgeDB singleton
    create_experiment(...)  → experiment ID
    finish_experiment(...)  → marks complete
"""

from __future__ import annotations

from forge.tracking.db import ForgeDB

# Module-level singleton
_db: ForgeDB | None = None


def get_db(db_path: str | None = None) -> ForgeDB:
    """Get or create the global ForgeDB instance.

    Default location: .forge/experiments.db (project-local).
    """
    global _db
    if _db is None:
        _db = ForgeDB(db_path)
    return _db


def create_experiment(
    name: str,
    config: dict | None = None,
    tags: list[str] | None = None,
    db_path: str | None = None,
) -> str:
    """Create a new experiment and return its ID."""
    db = get_db(db_path)
    return db.create_experiment(name=name, config=config, tags=tags)


def finish_experiment(
    experiment_id: str,
    status: str = "completed",
    db_path: str | None = None,
) -> None:
    """Mark an experiment as finished."""
    db = get_db(db_path)
    db.finish_experiment(experiment_id, status=status)


def reset_db() -> None:
    """Reset the singleton (for testing)."""
    global _db
    if _db is not None:
        _db.close()
    _db = None
