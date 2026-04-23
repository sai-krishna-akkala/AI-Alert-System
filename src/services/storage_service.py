"""
Storage service – persists and loads review results.

Backends:
  • SQLite  (default, demo-friendly, shares DB between bot and dashboard)
  • PostgreSQL / Supabase  (production, same interface)
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ── Schema DDL ───────────────────────────────────────────────────────────────

_SQLITE_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS reviews (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    repo            TEXT NOT NULL,
    pr_number       INTEGER NOT NULL,
    pr_title        TEXT,
    pr_author       TEXT,
    branch          TEXT,
    commit_sha      TEXT,
    summary         TEXT,
    overall_assessment TEXT,
    risk_score      INTEGER,
    risk_level      TEXT,
    decision        TEXT,
    reasoning       TEXT,
    files           TEXT,           -- JSON array
    cross_file_impact TEXT,         -- JSON array
    issues          TEXT,           -- JSON array
    good_improvements TEXT,         -- JSON array
    bad_regressions TEXT,           -- JSON array
    recommended_actions TEXT,       -- JSON array
    created_at      TEXT NOT NULL
);
"""

_PG_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS reviews (
    id                  SERIAL PRIMARY KEY,
    repo                TEXT NOT NULL,
    pr_number           INTEGER NOT NULL,
    pr_title            TEXT,
    pr_author           TEXT,
    branch              TEXT,
    commit_sha          TEXT,
    summary             TEXT,
    overall_assessment  TEXT,
    risk_score          INTEGER,
    risk_level          TEXT,
    decision            TEXT,
    reasoning           TEXT,
    files               JSONB,
    cross_file_impact   JSONB,
    issues              JSONB,
    good_improvements   JSONB,
    bad_regressions     JSONB,
    recommended_actions JSONB,
    created_at          TEXT NOT NULL
);
"""


# ── Helper to serialise JSON fields ─────────────────────────────────────────

_JSON_FIELDS = (
    "files",
    "cross_file_impact",
    "issues",
    "good_improvements",
    "bad_regressions",
    "recommended_actions",
)


def _encode_json_fields(row: dict[str, Any]) -> dict[str, Any]:
    """Ensure list/dict fields are JSON-encoded strings for SQLite."""
    out = dict(row)
    for key in _JSON_FIELDS:
        val = out.get(key)
        if val is not None and not isinstance(val, str):
            out[key] = json.dumps(val, default=str)
    return out


def _decode_json_fields(row: dict[str, Any]) -> dict[str, Any]:
    """Parse JSON-encoded strings back into lists/dicts."""
    out = dict(row)
    for key in _JSON_FIELDS:
        val = out.get(key)
        if isinstance(val, str):
            try:
                out[key] = json.loads(val)
            except json.JSONDecodeError:
                pass
    return out


# ── Abstract interface ──────────────────────────────────────────────────────

class StorageService:
    """Unified interface for saving and loading reviews."""

    def save_review_result(self, review: dict[str, Any]) -> None:
        raise NotImplementedError

    def load_review_results(self, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        raise NotImplementedError


# ── SQLite backend ──────────────────────────────────────────────────────────

class SQLiteStorage(StorageService):
    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or os.getenv("SQLITE_DB_PATH", "reviews.db")
        self._ensure_table()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_table(self) -> None:
        with self._conn() as conn:
            conn.execute(_SQLITE_CREATE_TABLE)
            conn.commit()

    def save_review_result(self, review: dict[str, Any]) -> None:
        row = _encode_json_fields(review)
        row.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        cols = [
            "repo", "pr_number", "pr_title", "pr_author", "branch", "commit_sha",
            "summary", "overall_assessment", "risk_score", "risk_level", "decision",
            "reasoning", "files", "cross_file_impact", "issues", "good_improvements",
            "bad_regressions", "recommended_actions", "created_at",
        ]
        placeholders = ", ".join("?" for _ in cols)
        col_names = ", ".join(cols)
        values = [row.get(c) for c in cols]
        try:
            with self._conn() as conn:
                conn.execute(f"INSERT INTO reviews ({col_names}) VALUES ({placeholders})", values)
                conn.commit()
            logger.info("Review saved for %s PR #%s", row.get("repo"), row.get("pr_number"))
        except Exception:
            logger.exception("Failed to save review result")
            raise

    def load_review_results(self, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM reviews WHERE 1=1"
        params: list[Any] = []
        if filters:
            if filters.get("repo"):
                query += " AND repo = ?"
                params.append(filters["repo"])
            if filters.get("risk_level"):
                query += " AND risk_level = ?"
                params.append(filters["risk_level"])
            if filters.get("decision"):
                query += " AND decision = ?"
                params.append(filters["decision"])
            if filters.get("date_from"):
                query += " AND created_at >= ?"
                params.append(filters["date_from"])
            if filters.get("date_to"):
                query += " AND created_at <= ?"
                params.append(filters["date_to"])
        query += " ORDER BY created_at DESC"
        try:
            with self._conn() as conn:
                rows = conn.execute(query, params).fetchall()
            return [_decode_json_fields(dict(r)) for r in rows]
        except Exception:
            logger.exception("Failed to load review results")
            return []


# ── PostgreSQL backend ──────────────────────────────────────────────────────

class PostgresStorage(StorageService):
    def __init__(self, dsn: str | None = None):
        self.dsn = dsn or os.getenv("DATABASE_URL", "")
        self._ensure_table()

    def _conn(self):  # type: ignore[override]
        import psycopg2
        import psycopg2.extras
        conn = psycopg2.connect(self.dsn)
        return conn

    def _ensure_table(self) -> None:
        try:
            conn = self._conn()
            with conn.cursor() as cur:
                cur.execute(_PG_CREATE_TABLE)
            conn.commit()
            conn.close()
        except Exception:
            logger.exception("Could not ensure PG table")

    def save_review_result(self, review: dict[str, Any]) -> None:
        import psycopg2.extras  # noqa: F811

        row = dict(review)
        row.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        # For Postgres JSONB, keep dicts/lists as-is; psycopg2 Json adapter handles them
        for key in _JSON_FIELDS:
            val = row.get(key)
            if val is not None and not isinstance(val, str):
                import psycopg2.extras as _ex
                row[key] = _ex.Json(val)

        cols = [
            "repo", "pr_number", "pr_title", "pr_author", "branch", "commit_sha",
            "summary", "overall_assessment", "risk_score", "risk_level", "decision",
            "reasoning", "files", "cross_file_impact", "issues", "good_improvements",
            "bad_regressions", "recommended_actions", "created_at",
        ]
        placeholders = ", ".join(f"%({c})s" for c in cols)
        col_names = ", ".join(cols)
        try:
            conn = self._conn()
            with conn.cursor() as cur:
                cur.execute(f"INSERT INTO reviews ({col_names}) VALUES ({placeholders})", row)
            conn.commit()
            conn.close()
            logger.info("Review saved (PG) for %s PR #%s", row.get("repo"), row.get("pr_number"))
        except Exception:
            logger.exception("Failed to save review result (PG)")
            raise

    def load_review_results(self, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM reviews WHERE TRUE"
        params: list[Any] = []
        if filters:
            if filters.get("repo"):
                query += " AND repo = %s"
                params.append(filters["repo"])
            if filters.get("risk_level"):
                query += " AND risk_level = %s"
                params.append(filters["risk_level"])
            if filters.get("decision"):
                query += " AND decision = %s"
                params.append(filters["decision"])
            if filters.get("date_from"):
                query += " AND created_at >= %s"
                params.append(filters["date_from"])
            if filters.get("date_to"):
                query += " AND created_at <= %s"
                params.append(filters["date_to"])
        query += " ORDER BY created_at DESC"
        try:
            conn = self._conn()
            import psycopg2.extras  # noqa: F811
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
            conn.close()
            return [_decode_json_fields(dict(r)) for r in rows]
        except Exception:
            logger.exception("Failed to load review results (PG)")
            return []


# ── Factory ─────────────────────────────────────────────────────────────────

def get_storage() -> StorageService:
    """Return the configured storage backend."""
    backend = os.getenv("STORAGE_BACKEND", "sqlite").lower()
    if backend == "postgres":
        return PostgresStorage()
    return SQLiteStorage()
