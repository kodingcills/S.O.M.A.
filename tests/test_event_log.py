"""Tests for backend/event_log.py — SOMA Task 1.1.

Named tests per ARCHITECTURE.md TESTING PROTOCOL:
  test_wal_mode                            — WAL MODE INVARIANT — MUST PASS
  test_write_read_roundtrip_payload_is_dict — C7 contract (payload always dict)
  test_column_kwargs_route_to_columns      — accepted-kwargs law
  test_events_for_agent_subtree            — agent_id OR parent_id match
  test_broadcast_callback_called           — fire-and-forget dispatch
"""

import asyncio
import sqlite3

from backend.event_log import (
    get_events_for_agent,
    get_events_since,
    init_db,
    set_broadcast_callback,
    write_event,
)


# ---------------------------------------------------------------------------
# test:event_log:wal — WAL MODE INVARIANT (ARCHITECTURE.md)
# ---------------------------------------------------------------------------

def test_wal_mode(tmp_path):
    # A brand-new connection (not the one init_db used) must observe 'wal':
    # proves the pragma runs per-connection, not once globally.
    db = tmp_path / "events.db"
    init_db(db_path=db)

    conn = sqlite3.connect(db)
    try:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        conn.close()
    assert mode == "wal"


# ---------------------------------------------------------------------------
# test:event_log:roundtrip — C7: payload deserialized to dict, never raw string
# ---------------------------------------------------------------------------

def test_write_read_roundtrip_payload_is_dict(tmp_path):
    init_db(db_path=tmp_path / "events.db")
    event_id = write_event("system_ready", test_value=42, nested={"a": 1})

    events = get_events_since(0.0)
    assert len(events) == 1
    event = events[0]
    assert event["id"] == event_id
    assert event["event_type"] == "system_ready"
    assert isinstance(event["payload"], dict), "C7 violated: payload not a dict"
    assert event["payload"]["test_value"] == 42
    assert event["payload"]["nested"] == {"a": 1}


def test_column_kwargs_route_to_columns(tmp_path):
    # Only agent_id/parent_id/sim_id/region/error_before/error_after are
    # columns; everything else lands in payload.
    init_db(db_path=tmp_path / "events.db")
    write_event(
        "agent_completed",
        agent_id="exp_abc123",
        region="upper_left",
        error_before=0.5,
        error_after=0.2,
        samples_collected=100,
    )

    (event,) = get_events_since(0.0)
    assert event["agent_id"] == "exp_abc123"
    assert event["region"] == "upper_left"
    assert event["error_before"] == 0.5
    assert event["error_after"] == 0.2
    assert event["payload"] == {"samples_collected": 100}

    # Absent optional columns come back as None, not missing keys.
    assert event["parent_id"] is None
    assert event["sim_id"] is None


# ---------------------------------------------------------------------------
# test:event_log:agent_subtree
# ---------------------------------------------------------------------------

def test_events_for_agent_subtree(tmp_path):
    init_db(db_path=tmp_path / "events.db")
    write_event("agent_spawned", agent_id="exp_parent", parent_id="orchestrator")
    write_event("agent_step", agent_id="exp_child", parent_id="exp_parent", step=1)
    write_event("agent_step", agent_id="unrelated", step=2)

    subtree = get_events_for_agent("exp_parent")
    agent_ids = {e["agent_id"] for e in subtree}
    assert agent_ids == {"exp_parent", "exp_child"}
    assert all(e["agent_id"] != "unrelated" for e in subtree)

    timestamps = [e["timestamp"] for e in subtree]
    assert timestamps == sorted(timestamps)


# ---------------------------------------------------------------------------
# test:event_log:broadcast — fire-and-forget dispatch when loop is running
# ---------------------------------------------------------------------------

async def test_broadcast_callback_called(tmp_path):
    init_db(db_path=tmp_path / "events.db")
    received: list[dict] = []

    async def callback(event: dict) -> None:
        received.append(event)

    set_broadcast_callback(callback)
    write_event("system_ready", hello="world")

    await asyncio.sleep(0.05)  # yield so the fire-and-forget task executes
    assert len(received) == 1
    assert received[0]["event_type"] == "system_ready"
    assert received[0]["payload"] == {"hello": "world"}


async def test_broadcast_callback_exception_is_swallowed(tmp_path):
    # A raising consumer must never break write_event's caller.
    init_db(db_path=tmp_path / "events.db")

    async def bad_callback(event: dict) -> None:
        raise RuntimeError("consumer exploded")

    set_broadcast_callback(bad_callback)
    event_id = write_event("system_ready")  # must not raise
    await asyncio.sleep(0.05)
    assert event_id > 0


def test_broadcast_skipped_without_running_loop(tmp_path):
    # Synchronous context (no asyncio loop): SQLite write still happens,
    # dispatch is silently skipped.
    init_db(db_path=tmp_path / "events.db")

    async def callback(event: dict) -> None:  # pragma: no cover - never runs
        raise AssertionError("callback must not run without a loop")

    set_broadcast_callback(callback)
    event_id = write_event("system_ready")
    assert get_events_since(0.0)[0]["id"] == event_id
