"""Task 1.3b — training data collector + quality gate.

Tests the three contracts:
  1. init_training_db: idempotent, WAL mode, exact 7-column schema.
  2. collect_training_data: real env smoke run — row count, blob
     round-trip shapes/dtypes, per-episode step tracking.
  3. heuristic_action: RISKS D-8 guided policy — structure + bleeding
     branch determinism.

Every test uses a throwaway DB (tmp_path) and closes its env.
"""

import sqlite3

import numpy as np
import pytest

from backend.data_collector import (
    collect_training_data,
    heuristic_action,
    init_training_db,
)
from backend.simulation import SimConfig, SurROLTissueEnv

EXPECTED_COLUMNS = {
    "id",
    "state",
    "action",
    "next_state",
    "reward",
    "done",
    "episode_step",
}


@pytest.fixture
def env():
    """Real collection-config env (DATA COLLECTION SEED=0), closed after."""
    e = SurROLTissueEnv(SimConfig(seed=0, dof=1, n_vessels=2))
    yield e
    e.close()


def test_init_training_db_idempotent_wal_schema(tmp_path):
    db = tmp_path / "training.db"
    init_training_db(db)
    init_training_db(db)  # second call must not raise or wipe schema

    conn = sqlite3.connect(db)
    try:
        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert journal_mode == "wal"
        columns = {row[1] for row in conn.execute("PRAGMA table_info(samples)")}
        assert columns == EXPECTED_COLUMNS
    finally:
        conn.close()


def test_collect_smoke(env, tmp_path):
    db = tmp_path / "training.db"
    written = collect_training_data(env, n_samples=120, db_path=db)
    assert written == 120

    conn = sqlite3.connect(db)
    try:
        (count,) = conn.execute("SELECT COUNT(*) FROM samples").fetchone()
        rows = conn.execute(
            "SELECT state, action, next_state, reward, done, episode_step "
            "FROM samples"
        ).fetchall()
    finally:
        conn.close()

    assert count == 120
    assert len(rows) == 120
    for state_blob, action_blob, next_blob, reward, done, episode_step in rows:
        state = np.frombuffer(state_blob, dtype=np.float32)
        action = np.frombuffer(action_blob, dtype=np.float32)
        next_state = np.frombuffer(next_blob, dtype=np.float32)
        assert state.shape == (806,)
        assert action.shape == (12,)
        assert next_state.shape == (806,)
        assert isinstance(reward, float)
        assert done in (0, 1)
        assert episode_step >= 0


def test_heuristic_action():
    rng = np.random.RandomState(0)

    # Bleeding branch: RandomState(0).uniform() first draw ≈ 0.55 ≥ 0.25,
    # so the coverage roll deterministically takes the GUIDED branch.
    # Cells set in [512:768] (bleeding mask) → CAUTERIZE → vec[9] == 1.
    bleeding_vec = np.zeros(806, dtype=np.float32)
    bleeding_vec[512:520] = 1.0
    cauterize = heuristic_action(bleeding_vec, rng)
    assert cauterize.shape == (12,)
    assert cauterize.dtype == np.float32
    assert cauterize[9] == 1.0
    assert float(cauterize[7:11].sum()) == 1.0  # exactly one mode hot

    # Guided MOVE branch: no bleeding → vec[7] == 1, magnitude in [0, 1].
    move_vec = np.zeros(806, dtype=np.float32)
    move_vec[768:770] = [0.2, 0.2]  # EE xy
    move_vec[776:778] = [0.8, 0.8]  # target col,row
    move = heuristic_action(move_vec, rng)
    assert move.shape == (12,)
    assert move.dtype == np.float32
    assert move[7] == 1.0
    assert int(np.argmax(move[7:11])) == 0  # MOVE slot
    assert 0.0 <= move[11] <= 1.0
