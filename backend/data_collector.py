"""Task 1.3b — training data collector + RISKS D-8 guided heuristic policy.

Produces backend/data/training.db. Schema and procedure follow
docs/specs/SIMULATION.md "DATA COLLECTION"; the heuristic policy exists
because pure-random policies terminate episodes too early and fail the
D-8 quality gate (early-episode state dominance).
"""

import sqlite3
from collections.abc import Callable
from pathlib import Path

import numpy as np

from backend.simulation import SurROLTissueEnv

DEFAULT_DB_PATH = Path("backend/data/training.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    state        BLOB NOT NULL,
    action       BLOB NOT NULL,
    next_state   BLOB NOT NULL,
    reward       REAL NOT NULL,
    done         INTEGER NOT NULL,
    episode_step INTEGER NOT NULL DEFAULT 0
);
"""

_INSERT = (
    "INSERT INTO samples (state, action, next_state, reward, done, "
    "episode_step) VALUES (?, ?, ?, ?, ?, ?)"
)

# policy(state_vec, rng) -> action12 float32
ActionPolicy = Callable[[np.ndarray, np.random.RandomState], np.ndarray]


def _connect(db_path: str | Path) -> sqlite3.Connection:
    # WAL MODE INVARIANT: pragma on EVERY connection, not once globally.
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_training_db(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """Create schema + WAL mode. Idempotent — safe to call repeatedly."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = _connect(path)
    try:
        conn.executescript(_SCHEMA)
        conn.commit()
    finally:
        conn.close()


def heuristic_action(
    state_vec: np.ndarray, rng: np.random.RandomState
) -> np.ndarray:
    """D-8 guided policy: mostly move-to-target, cauterize on bleeding,
    25% pure-random coverage actions. Threaded RandomState only — never
    the global numpy seed."""
    if rng.uniform() < 0.25:
        vec = np.zeros(12, dtype=np.float32)
        vec[7 + rng.randint(0, 4)] = 1.0  # one-hot mode into [7:11]
        vec[0:3] = rng.uniform(-1, 1, 3)
        vec[3:6] = rng.uniform(-1, 1, 3)
        vec[6] = rng.uniform(0, 1)
        vec[11] = rng.uniform(0, 1)
        return vec

    vec = np.zeros(12, dtype=np.float32)
    if float(state_vec[512:768].sum()) > 0:
        vec[9] = 1.0  # CAUTERIZE
        vec[11] = 1.0
        return vec

    d = state_vec[776:778] - state_vec[768:770]
    unit = d / (np.linalg.norm(d) + 1e-8)
    vec[7] = 1.0  # MOVE
    vec[0:2] = np.clip(unit * 0.6 + rng.normal(0, 0.15, 2), -1, 1)
    vec[11] = rng.uniform(0.4, 0.9)
    return vec


def collect_training_data(
    env: SurROLTissueEnv,
    n_samples: int = 50_000,
    db_path: str | Path = DEFAULT_DB_PATH,
    policy: ActionPolicy | None = None,
) -> int:
    """Step `env` until n_samples transitions are written to training.db.

    Records (state, action, next_state, reward, done, episode_step);
    resets on done with per-episode step counter from 0; commits in
    batches of 1000. Returns total rows written.
    """
    init_training_db(db_path)
    policy_fn: ActionPolicy = heuristic_action if policy is None else policy
    rng = np.random.RandomState()  # instance RNG — global seed untouched

    conn = _connect(db_path)
    written = 0
    episode_step = 0
    batch: list[tuple[bytes, bytes, bytes, float, int, int]] = []
    try:
        env.reset()
        while written < n_samples:
            state_vec = env.get_state_vector()
            action_vec = policy_fn(state_vec, rng)

            # env.step accepts our 12-vector directly (normalizes + DOF-gates)
            _, reward, done, _ = env.step(action_vec)
            next_state_vec = env.get_state_vector()

            batch.append(
                (
                    state_vec.astype(np.float32).tobytes(),
                    action_vec.astype(np.float32).tobytes(),
                    next_state_vec.astype(np.float32).tobytes(),
                    float(reward),
                    int(done),
                    episode_step,
                )
            )
            written += 1
            episode_step += 1

            if done:
                env.reset()
                episode_step = 0

            if len(batch) >= 1000 or written == n_samples:
                conn.executemany(_INSERT, batch)
                conn.commit()
                batch.clear()

            if written % 5000 == 0:
                print(f"[collect] {written}/{n_samples} samples", flush=True)
    finally:
        conn.close()

    print(f"Data collection complete: {written} samples", flush=True)
    return written
