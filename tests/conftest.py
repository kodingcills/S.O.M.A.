"""Shared test isolation for SOMA Task 1.1.

Every test gets a throwaway SQLite DB via SOMA_DB_PATH (resolved by
event_log.init_db) and a cleared broadcast callback, so no test ever
touches backend/data/events.db or leaks callbacks into another test.
"""

import pytest

from backend import event_log


@pytest.fixture(autouse=True)
def isolated_event_db(tmp_path, monkeypatch):
    monkeypatch.setenv("SOMA_DB_PATH", str(tmp_path / "events.db"))
    # Re-point module state EVERY test: event_log caches _db_path/_initialized
    # as module globals; without this, later tests write into the first
    # test's DB file (cross-test event pollution breaks ordering asserts).
    event_log.init_db()
    event_log.set_broadcast_callback(None)
    yield
    event_log.set_broadcast_callback(None)
