"""Tests for backend/main.py + scripts/golden_signal.py — SOMA Task 1.9.

Fast full-system boot test (<60s): the two multi-minute CPU phases
(data collection, initial training) are monkeypatched to instant stubs;
the two SurRoL envs are REAL (~4s) so the smoke actually exercises the
lifespan wiring, task registry, and STATE bridge.

Named tests per ARCHITECTURE.md TESTING PROTOCOL:
  test_health_lifecycle            — boot → /health 'ok' → live telemetry events
  test_golden_signal_allow_empty   — script exits 0 on empty db with flag
  test_golden_signal_detects_improvement — delta math + PASS path
  test_golden_signal_fails_on_regression — FAIL path without flag
"""

import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from backend import event_log, main as soma_main
from backend.event_log import get_events_since
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_SCRIPT = REPO_ROOT / "scripts" / "golden_signal.py"


@pytest.fixture
def fast_boot(monkeypatch):
    """Stub the long CPU phases referenced from backend.main's namespace."""
    for _name, _stub in [
        ("_count_training_samples", lambda: 999_999),
        ("train_prediction_network", lambda *a, **k: {"epochs_run": 1, "val_tissue_mse": 0.5}),
        ("collect_training_data", lambda *a, **k: 0),
    ]:
        if hasattr(soma_main, _name):
            monkeypatch.setattr(soma_main, _name, _stub)


def _wait_until(predicate, timeout_s: float, poll_s: float = 0.25) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll_s)
    return False


def _events_of_types(types: set[str]) -> list[dict]:
    return [e for e in get_events_since(0.0) if e["event_type"] in types]


# ---------------------------------------------------------------------------
# Full lifespan boot (real envs, stubbed training)
# ---------------------------------------------------------------------------

def test_health_lifecycle(fast_boot):
    with TestClient(soma_main.app) as client:
        # Startup → bridge task flips status to 'ok' within one tick.
        assert _wait_until(
            lambda: client.get("/health").json()["status"] == "ok", 30
        ), f"health never reached 'ok': {client.get('/health').json()}"

        # Craftax substrate writes system_ready immediately and belief_snapshot on interval.
        assert _wait_until(
            lambda: bool(_events_of_types({"system_ready"})), 10
        ), "no system_ready event within 10s"

        # belief_snapshot_loop writes its first snapshot immediately.
        assert _wait_until(
            lambda: bool(_events_of_types({"belief_snapshot"})), 25
        ), "no belief_snapshot event within 25s"

        body = client.get("/health").json()
        assert set(body.keys()) == {
            "status", "world_model_version", "global_error",
            "active_agents", "primary_sim_step", "training_progress",
        }
        assert isinstance(body["global_error"], float)
        assert body["world_model_version"] >= 0
    # Context exit ran shutdown: tasks cancelled, sims closed, no raise.


# ---------------------------------------------------------------------------
# golden_signal.py contract
# ---------------------------------------------------------------------------

def _run_golden(db: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(GOLDEN_SCRIPT), "--db", str(db), *extra],
        capture_output=True, text=True, timeout=60,
    )


def _seed_completed(db: Path, rows: list[tuple[str, float, float]]) -> None:
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS events ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " timestamp REAL NOT NULL, event_type TEXT NOT NULL,"
            " agent_id TEXT, error_before REAL, error_after REAL)"
        )
        conn.executemany(
            "INSERT INTO events (timestamp, event_type, agent_id,"
            " error_before, error_after) VALUES (?, 'agent_completed', ?, ?, ?)",
            [(time.time(), aid, eb, ea) for aid, eb, ea in rows],
        )
        conn.commit()
    finally:
        conn.close()


def test_golden_signal_allow_empty(tmp_path):
    result = _run_golden(tmp_path / "events.db", "--allow-empty")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS" in result.stdout


def test_golden_signal_fails_without_allow_empty(tmp_path):
    result = _run_golden(tmp_path / "missing.db")
    assert result.returncode == 1
    assert "FAIL" in result.stdout


def test_golden_signal_detects_improvement(tmp_path):
    db = tmp_path / "events.db"
    _seed_completed(db, [
        ("exp_aaa", 0.400, 0.250),   # improved (delta 0.150)
        ("exp_bbb", 0.300, 0.100),   # improved (delta 0.200)
    ])
    result = _run_golden(db)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2/2 (100%)" in result.stdout
    assert "PASS" in result.stdout


def test_golden_signal_fails_on_regression(tmp_path):
    db = tmp_path / "events.db"
    _seed_completed(db, [
        ("exp_aaa", 0.200, 0.400),   # worsened
        ("exp_bbb", 0.300, 0.500),   # worsened
    ])
    result = _run_golden(db)
    assert result.returncode == 1
    assert "FAIL" in result.stdout
