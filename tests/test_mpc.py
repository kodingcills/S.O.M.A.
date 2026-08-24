"""Task 1.8 — MPC planning agent named tests.

Spec: docs/specs/WORLD_MODEL.md "MPC PLANNING AGENT" +
ARCHITECTURE.md EVENT LOOP NON-BLOCKING INVARIANT (the 0.05s yield).

Covers:
  - candidate generation contract (8 x (12,) f32, one-hot modes)
  - value function ordering + hard-constraint dominance
  - real-env smoke run of the async loop against a stub WorldModel
    (WorldModel is Task 1.5, built in parallel — code against contract)
"""

import asyncio
import time
from types import SimpleNamespace

import numpy as np
import pytest

from backend.event_log import get_events_since, init_db
from backend.mpc_agent import MPCAgent
from backend.prediction_net import PredictedState, PredictionNetwork
from backend.simulation import SimConfig, SurROLTissueEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state_ee_target(
    ee_xy: tuple[float, float] = (0.3, 0.5),
    tgt: tuple[float, float] = (0.7, 0.5),
) -> np.ndarray:
    """806-vector with only the fields _compute_value reads populated."""
    s = np.zeros(806, dtype=np.float32)
    s[768:770] = ee_xy          # normalized EE xy
    s[776:778] = tgt            # normalized target col,row
    return s


def _predicted(
    pred_ee_xy: tuple[float, float],
    damage_prob: float = 0.1,
    tissue_mean: float = 0.9,
) -> PredictedState:
    """Fabricated PredictedState with controlled fields."""
    return PredictedState(
        tissue=np.full((16, 16), tissue_mean, dtype=np.float32),
        damage_prob=damage_prob,
        vessel_risk=np.zeros(5, dtype=np.float32),
        pred_ee_norm=np.array(
            [pred_ee_xy[0], pred_ee_xy[1], 0.5, 0.0], dtype=np.float32
        ),
    )


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def test_generates_exactly_8_candidates():
    # Given: a zero state and an agent whose pure functions need no deps
    agent = MPCAgent(world_model=None, env=None)  # type: ignore[arg-type]
    # When: candidates are generated
    candidates = agent._generate_candidates(np.zeros(806, dtype=np.float32))
    # Then: exactly 8, each (12,) float32 with exactly one one-hot mode slot
    assert len(candidates) == 8
    for action in candidates:
        assert action.shape == (12,)
        assert action.dtype == np.float32
        mode = action[7:11]
        assert float(mode.sum()) == pytest.approx(1.0), f"not one-hot: {mode}"
        assert set(np.unique(mode).tolist()) <= {0.0, 1.0}


def test_candidate_modes_diverse():
    # Given: zero state (no bleeding -> cauterize slot becomes dz probe)
    agent = MPCAgent(world_model=None, env=None)  # type: ignore[arg-type]
    # When: collecting the argmax mode of each candidate's one-hot block
    modes = {
        int(np.argmax(a[7:11]))
        for a in agent._generate_candidates(np.zeros(806, dtype=np.float32))
    }
    # Then: at least MOVE(0) and WAIT(3) are represented
    assert {0, 3} <= modes


# ---------------------------------------------------------------------------
# Value function
# ---------------------------------------------------------------------------

def test_value_function_prefers_progress():
    # Given: EE at (0.3,0.5), target at (0.7,0.5); identical otherwise
    agent = MPCAgent(world_model=None, env=None)  # type: ignore[arg-type]
    state = _state_ee_target()
    toward = _predicted((0.4, 0.5))   # predicted EE closer to target
    away = _predicted((0.2, 0.5))     # predicted EE farther from target
    # When: scoring both predictions
    v_toward = agent._compute_value(state, toward)
    v_away = agent._compute_value(state, away)
    # Then: progress dominates the ordering
    assert v_toward > v_away


def test_value_hard_constraint_dominates():
    # Given: a prediction with PERFECT progress but 90% damage probability,
    # and a safe prediction with zero progress
    agent = MPCAgent(world_model=None, env=None)  # type: ignore[arg-type]
    state = _state_ee_target()
    deadly_on_target = _predicted((0.7, 0.5), damage_prob=0.9)
    safe_stationary = _predicted((0.3, 0.5), damage_prob=0.1)
    # When: scoring both
    v_deadly = agent._compute_value(state, deadly_on_target)
    v_safe = agent._compute_value(state, safe_stationary)
    # Then: the -100 hard penalty crushes any progress term (< -90 always)
    assert v_deadly < -90.0
    assert v_deadly < v_safe


# ---------------------------------------------------------------------------
# Run-loop smoke (real env + untrained net + stub WorldModel)
# ---------------------------------------------------------------------------

@pytest.fixture
def real_env():
    env = SurROLTissueEnv(SimConfig(seed=42))
    env.reset()
    yield env
    env.close()


async def test_run_loop_smoke(real_env, tmp_path):
    # Given: tmp event DB + stub WorldModel honoring the Task 1.5 contract
    init_db(db_path=tmp_path / "events.db")
    world_model = SimpleNamespace(
        network=PredictionNetwork(),
        _predict_lock=asyncio.Lock(),
        belief=SimpleNamespace(active_dof=1),
        update=lambda state, action, next_state: None,
    )
    agent = MPCAgent(world_model, real_env)

    # When: run() is driven for ~1.5s then cancelled mid-flight
    task = asyncio.create_task(agent.run())
    await asyncio.sleep(1.5)
    task.cancel()
    results = await asyncio.gather(task, return_exceptions=True)

    # Then: no exception surfaced except CancelledError itself...
    unexpected = [r for r in results if not isinstance(r, asyncio.CancelledError)]
    assert unexpected == []
    # ...and at least one primary simulation_step reached the tmp DB.
    steps = [
        e
        for e in get_events_since(0.0)
        if e["event_type"] == "simulation_step" and e["sim_id"] == "primary"
    ]
    assert len(steps) >= 1

    # Latency probe: time one full 8-candidate planning cycle on this
    # machine. Target <50ms/candidate (QUICK_REFERENCE.md); logged, not
    # asserted hard — only a pathological-hang guard below.
    state = real_env.get_state_vector()
    async with world_model._predict_lock:
        candidates = agent._generate_candidates(state)
        start = time.perf_counter()
        for action in candidates:
            world_model.network.predict(state, action)
        avg_ms = (time.perf_counter() - start) / len(candidates) * 1000
    print(f"\n[MPC latency] avg predict over 8 candidates: {avg_ms:.2f} ms")
    assert avg_ms < 500.0, "pathological regression: predict >500ms avg"
