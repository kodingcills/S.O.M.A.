"""Event log — SOMA Task 1.1.

Append-only SQLite event store. THE communication law (ARCHITECTURE.md):
every component talks via write_event() / get_events_since(); no direct
component-to-component calls outside the three named exceptions.

Invariants enforced here:
  - WAL MODE INVARIANT: PRAGMA journal_mode=WAL on EVERY sqlite3.connect().
  - C7: payload is stored as serialized JSON, returned as a dict.
  - Accepted column kwargs are ONLY agent_id, parent_id, sim_id, region,
    error_before, error_after — everything else goes into payload.

IMPORT LAW: stdlib only. Zero imports from backend/ or third parties.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Coroutine

# SPEC DEVIATION: `os` added to the import list — required for the
# SOMA_DB_PATH env override mandated by Task 1.1. Still stdlib-only,
# so the ARCHITECTURE.md import law holds.

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp    REAL    NOT NULL,
    event_type   TEXT    NOT NULL,
    agent_id     TEXT,
    parent_id    TEXT,
    sim_id       TEXT,
    region       TEXT,
    error_before REAL,
    error_after  REAL,
    payload      TEXT    NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_timestamp  ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_agent_id   ON events(agent_id);
CREATE INDEX IF NOT EXISTS idx_event_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_sim_id     ON events(sim_id);
"""

_COLUMN_KWARGS = frozenset(
    {"agent_id", "parent_id", "sim_id", "region", "error_before", "error_after"}
)

# Module-level state: resolved DB path + registered broadcast callback.
# Path resolution order: explicit init_db(db_path=...) arg
#                       > SOMA_DB_PATH env var
#                       > backend/data/events.db (module-relative, cwd-proof)
_DEFAULT_DB_PATH = Path(__file__).resolve().parent / "data" / "events.db"
_db_path: Path = _DEFAULT_DB_PATH
_initialized: bool = False
_broadcast_callback: Callable[[dict], Coroutine[Any, Any, None]] | None = None


def _resolve_db_path(explicit: str | Path | None) -> Path:
    if explicit is not None:
        return Path(explicit)
    env_path = os.environ.get("SOMA_DB_PATH")
    if env_path:
        return Path(env_path)
    return _DEFAULT_DB_PATH


def _connect(path: Path) -> sqlite3.Connection:
    # WAL MODE INVARIANT: executed per-connection, never once globally.
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str | Path | None = None) -> None:
    """Create schema + indexes. Idempotent; safe to call multiple times."""
    global _db_path, _initialized
    _db_path = _resolve_db_path(db_path)
    _db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = _connect(_db_path)
    try:
        conn.executescript(_SCHEMA)
        conn.commit()
    finally:
        conn.close()
    _initialized = True


def _ensure_initialized() -> None:
    # Safety net for components that skipped startup; init_db is idempotent.
    if not _initialized:
        init_db()


def _row_to_event(row: sqlite3.Row) -> dict[str, Any]:
    event = dict(row)
    # C7: payload is ALWAYS a dict on the way out, never a raw JSON string.
    event["payload"] = json.loads(event["payload"])
    return event


async def _safe_broadcast(
    callback: Callable[[dict], Coroutine[Any, Any, None]], event: dict[str, Any]
) -> None:
    # Fire-and-forget: a failing consumer must never break the writer.
    try:
        await callback(event)
    except Exception:
        pass


def write_event(event_type: str, **kwargs: Any) -> int:
    """Insert one event synchronously; returns the new row id.

    Column kwargs: agent_id, parent_id, sim_id, region, error_before,
    error_after. All other kwargs are serialized into payload JSON.
    If a broadcast callback is registered and an asyncio loop is running,
    dispatches callback(event_dict) as a fire-and-forget task.
    """
    _ensure_initialized()

    columns: dict[str, Any] = {k: kwargs[k] for k in _COLUMN_KWARGS if k in kwargs}
    payload: dict[str, Any] = {k: v for k, v in kwargs.items() if k not in _COLUMN_KWARGS}

    timestamp = time.time()
    conn = _connect(_db_path)
    try:
        cursor = conn.execute(
            """
            INSERT INTO events (
                timestamp, event_type, agent_id, parent_id, sim_id,
                region, error_before, error_after, payload
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                event_type,
                columns.get("agent_id"),
                columns.get("parent_id"),
                columns.get("sim_id"),
                columns.get("region"),
                columns.get("error_before"),
                columns.get("error_after"),
                json.dumps(payload),
            ),
        )
        conn.commit()
        event_id = int(cursor.lastrowid)
    finally:
        conn.close()

    callback = _broadcast_callback
    if callback is not None:
        event_dict: dict[str, Any] = {
            "id": event_id,
            "timestamp": timestamp,
            "event_type": event_type,
            "agent_id": columns.get("agent_id"),
            "parent_id": columns.get("parent_id"),
            "sim_id": columns.get("sim_id"),
            "region": columns.get("region"),
            "error_before": columns.get("error_before"),
            "error_after": columns.get("error_after"),
            "payload": payload,
        }
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return event_id  # no running loop — skip broadcast silently
        loop.create_task(_safe_broadcast(callback, event_dict))

    return event_id


def get_events_since(timestamp: float = 0.0, limit: int = 1000) -> list[dict[str, Any]]:
    """Events with ts strictly after `timestamp`, oldest first.

    Strict `>` keeps REST polling pagination duplicate-free; since=0.0
    still replays the full history (all timestamps are > 0).
    `id` is the deterministic tiebreaker for same-microsecond writes.
    """
    _ensure_initialized()
    conn = _connect(_db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM events WHERE timestamp > ? "
            "ORDER BY timestamp ASC, id ASC LIMIT ?",
            (timestamp, limit),
        ).fetchall()
    finally:
        conn.close()
    return [_row_to_event(row) for row in rows]


def get_events_for_agent(agent_id: str) -> list[dict[str, Any]]:
    """Full subtree for an agent: rows where agent_id OR parent_id matches."""
    _ensure_initialized()
    conn = _connect(_db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM events WHERE agent_id = ? OR parent_id = ? "
            "ORDER BY timestamp ASC, id ASC",
            (agent_id, agent_id),
        ).fetchall()
    finally:
        conn.close()
    return [_row_to_event(row) for row in rows]


def set_broadcast_callback(
    fn: Callable[[dict], Coroutine[Any, Any, None]] | None,
) -> None:
    """Register (or clear with None) the async broadcast sink.

    Called exactly once during startup by api.py with manager.broadcast.
    """
    global _broadcast_callback
    _broadcast_callback = fn
