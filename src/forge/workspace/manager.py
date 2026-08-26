"""Team Workspace Manager.

Handles Role-Based Access Control (RBAC) and team configuration
using a local SQLite database for demonstration.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()


class WorkspaceManager:
    """Manages team workspaces and RBAC locally."""

    def __init__(self, db_path: str | Path | None = None):
        if db_path is None:
            db_path = Path.home() / ".forge" / "workspace.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the SQLite schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS teams (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS members (
                    team_id TEXT NOT NULL,
                    email TEXT NOT NULL,
                    role TEXT NOT NULL,
                    added_at TEXT NOT NULL,
                    PRIMARY KEY (team_id, email),
                    FOREIGN KEY (team_id) REFERENCES teams(id)
                )
            """)

    def init_team(self, team_name: str) -> str:
        """Create a new team workspace."""
        team_id = team_name.lower().replace(" ", "-")
        now = datetime.now(timezone.utc).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            # Check if exists
            cur = conn.execute("SELECT id FROM teams WHERE id = ?", (team_id,))
            if cur.fetchone():
                raise ValueError(f"Team '{team_name}' already exists.")

            conn.execute(
                "INSERT INTO teams (id, name, created_at) VALUES (?, ?, ?)",
                (team_id, team_name, now),
            )

            # Add creator as owner
            current_user = "local-admin@forge.local"
            conn.execute(
                "INSERT INTO members (team_id, email, role, added_at) VALUES (?, ?, ?, ?)",
                (team_id, current_user, "owner", now),
            )

        return team_id

    def invite_member(self, team_id: str, email: str, role: str = "contributor") -> None:
        """Invite a member to a team with a specific role."""
        valid_roles = {"owner", "maintainer", "contributor", "viewer"}
        if role not in valid_roles:
            raise ValueError(f"Invalid role: {role}. Must be one of {valid_roles}")

        now = datetime.now(timezone.utc).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT id FROM teams WHERE id = ?", (team_id,))
            if not cur.fetchone():
                raise ValueError(f"Team '{team_id}' does not exist.")

            # Upsert logic (since sqlite3 syntax varies, try update then insert)
            cur = conn.execute(
                "SELECT role FROM members WHERE team_id = ? AND email = ?", (team_id, email)
            )
            if cur.fetchone():
                conn.execute(
                    "UPDATE members SET role = ? WHERE team_id = ? AND email = ?",
                    (role, team_id, email),
                )
            else:
                conn.execute(
                    "INSERT INTO members (team_id, email, role, added_at) VALUES (?, ?, ?, ?)",
                    (team_id, email, role, now),
                )

    def list_members(self, team_id: str) -> list[dict[str, str]]:
        """List all members of a team."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                "SELECT email, role, added_at FROM members WHERE team_id = ? ORDER BY role",
                (team_id,),
            )
            return [dict(row) for row in cur.fetchall()]
