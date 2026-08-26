"""SQLite experiment database — zero-config, WAL mode, thread-safe.

Schema:
    experiments — run metadata, config snapshot, status
    metrics     — scalar time series (step → value)
    artifacts   — output files (adapters, checkpoints, exports)

Usage:
    db = ForgeDB()                        # .forge/experiments.db
    eid = db.create_experiment("run-1", config={...})
    db.log_metric(eid, "loss", 2.31, step=10)
    db.finish_experiment(eid)
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS experiments (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    config_json TEXT,
    status      TEXT NOT NULL DEFAULT 'running',
    tags        TEXT,          -- JSON array
    created_at  TEXT NOT NULL,
    finished_at TEXT,
    notes       TEXT
);

CREATE TABLE IF NOT EXISTS metrics (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    step          INTEGER NOT NULL,
    key           TEXT NOT NULL,
    value         REAL NOT NULL,
    wall_time     REAL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS artifacts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    name          TEXT NOT NULL,
    path          TEXT NOT NULL,
    artifact_type TEXT DEFAULT 'file',
    created_at    TEXT NOT NULL,
    metadata_json TEXT,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_metrics_exp_key
    ON metrics(experiment_id, key);
CREATE INDEX IF NOT EXISTS idx_metrics_exp_step
    ON metrics(experiment_id, step);
CREATE INDEX IF NOT EXISTS idx_artifacts_exp
    ON artifacts(experiment_id);
CREATE INDEX IF NOT EXISTS idx_experiments_status
    ON experiments(status);
CREATE INDEX IF NOT EXISTS idx_experiments_created
    ON experiments(created_at);
"""


class ForgeDB:
    """Thread-safe SQLite experiment database.

    Uses WAL mode for concurrent read/write (training loop writes
    while the dashboard reads).
    """

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            db_dir = Path(".forge")
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(db_dir / "experiments.db")

        self._db_path = db_path
        self._lock = threading.Lock()

        # Create schema on first connect
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        """Thread-safe connection context manager."""
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @property
    def db_path(self) -> str:
        """Return the database file path."""
        return self._db_path

    # --- Experiments ----------------------------------------------------------

    def create_experiment(
        self,
        name: str,
        config: dict | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
    ) -> str:
        """Create a new experiment and return its ID."""
        experiment_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()

        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO experiments (id, name, config_json, status, tags, created_at, notes)
                   VALUES (?, ?, ?, 'running', ?, ?, ?)""",
                (
                    experiment_id,
                    name,
                    json.dumps(config) if config else None,
                    json.dumps(tags) if tags else None,
                    now,
                    notes,
                ),
            )
        return experiment_id

    def finish_experiment(
        self,
        experiment_id: str,
        status: str = "completed",
    ) -> None:
        """Mark an experiment as finished."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE experiments SET status = ?, finished_at = ? WHERE id = ?",
                (status, now, experiment_id),
            )

    def get_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        """Get a single experiment by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
            ).fetchone()
        return self._row_to_experiment(row) if row else None

    def list_experiments(
        self,
        status: str | None = None,
        tag: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List experiments with optional filters."""
        query = "SELECT * FROM experiments"
        params: list[Any] = []
        conditions: list[str] = []

        if status:
            conditions.append("status = ?")
            params.append(status)
        if tag:
            conditions.append("tags LIKE ?")
            params.append(f'%"{tag}"%')

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_experiment(row) for row in rows]

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment and all its data."""
        with self._lock, self._connect() as conn:
            cursor = conn.execute("DELETE FROM experiments WHERE id = ?", (experiment_id,))
            return cursor.rowcount > 0

    def count_experiments(self, status: str | None = None) -> int:
        """Count experiments, optionally filtered by status."""
        if status:
            query = "SELECT COUNT(*) FROM experiments WHERE status = ?"
            params: tuple = (status,)
        else:
            query = "SELECT COUNT(*) FROM experiments"
            params = ()
        with self._connect() as conn:
            return conn.execute(query, params).fetchone()[0]

    # --- Metrics --------------------------------------------------------------

    def log_metric(
        self,
        experiment_id: str,
        key: str,
        value: float,
        step: int,
        wall_time: float | None = None,
    ) -> None:
        """Log a single metric value."""
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO metrics (experiment_id, key, value, step, wall_time)
                   VALUES (?, ?, ?, ?, ?)""",
                (experiment_id, key, value, step, wall_time),
            )

    def log_metrics_batch(
        self,
        experiment_id: str,
        metrics: list[tuple[str, float, int, float | None]],
    ) -> None:
        """Log multiple metrics in a single transaction.

        Each tuple: (key, value, step, wall_time).
        """
        with self._lock, self._connect() as conn:
            conn.executemany(
                """INSERT INTO metrics (experiment_id, key, value, step, wall_time)
                   VALUES (?, ?, ?, ?, ?)""",
                [(experiment_id, k, v, s, w) for k, v, s, w in metrics],
            )

    def get_metric_history(
        self,
        experiment_id: str,
        key: str,
    ) -> list[dict[str, Any]]:
        """Get full time series for a metric."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT step, value, wall_time FROM metrics
                   WHERE experiment_id = ? AND key = ?
                   ORDER BY step ASC""",
                (experiment_id, key),
            ).fetchall()
        return [{"step": r["step"], "value": r["value"], "wall_time": r["wall_time"]} for r in rows]

    def get_metric_keys(self, experiment_id: str) -> list[str]:
        """List all metric keys logged for an experiment."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT key FROM metrics WHERE experiment_id = ? ORDER BY key",
                (experiment_id,),
            ).fetchall()
        return [r["key"] for r in rows]

    def get_latest_metrics(self, experiment_id: str) -> dict[str, float]:
        """Get the latest value of each metric for an experiment."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT key, value FROM metrics
                   WHERE experiment_id = ?
                   AND step = (
                       SELECT MAX(step) FROM metrics m2
                       WHERE m2.experiment_id = metrics.experiment_id
                       AND m2.key = metrics.key
                   )""",
                (experiment_id,),
            ).fetchall()
        return {r["key"]: r["value"] for r in rows}

    # --- Artifacts ------------------------------------------------------------

    def log_artifact(
        self,
        experiment_id: str,
        name: str,
        path: str,
        artifact_type: str = "file",
        metadata: dict | None = None,
    ) -> None:
        """Log an artifact (output file, checkpoint, etc.)."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO artifacts (experiment_id, name, path, artifact_type, created_at, metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    experiment_id,
                    name,
                    path,
                    artifact_type,
                    now,
                    json.dumps(metadata) if metadata else None,
                ),
            )

    def get_artifacts(self, experiment_id: str) -> list[dict[str, Any]]:
        """Get all artifacts for an experiment."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM artifacts WHERE experiment_id = ? ORDER BY created_at",
                (experiment_id,),
            ).fetchall()
        return [
            {
                "name": r["name"],
                "path": r["path"],
                "type": r["artifact_type"],
                "created_at": r["created_at"],
                "metadata": json.loads(r["metadata_json"]) if r["metadata_json"] else None,
            }
            for r in rows
        ]

    # --- Helpers --------------------------------------------------------------

    @staticmethod
    def _row_to_experiment(row: sqlite3.Row) -> dict[str, Any]:
        """Convert a sqlite3.Row to a clean experiment dict."""
        return {
            "id": row["id"],
            "name": row["name"],
            "config": json.loads(row["config_json"]) if row["config_json"] else None,
            "status": row["status"],
            "tags": json.loads(row["tags"]) if row["tags"] else [],
            "created_at": row["created_at"],
            "finished_at": row["finished_at"],
            "notes": row["notes"],
        }

    def close(self) -> None:
        """No-op — connections are opened/closed per operation."""
        pass
