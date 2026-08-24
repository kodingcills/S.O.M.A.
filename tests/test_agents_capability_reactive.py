"""SOMA Task 1.7b — CapabilityAgent + ReactiveAgent tests.

Named targets (ARCHITECTURE.md testing protocol):
  test_capability_unlock_event_order  — unlock precedes completed; DOF atomic
  test_capability_seed_formula        — dof env created with seed = 42 + target_dof
  test_reactive_isolation             — grep-enforced zero world-model imports
  test_reactive_select_action         — greedy policy branches (pure function)
  test_reactive_loop_smoke            — live loop writes simulation_step events
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import backend.agents.capability as capability_module
from backend.agents.capability import CapabilityAgent
from backend.agents.reactive import ReactiveAgent
from backend.belief_state import BeliefState
from backend.event_log import get_events_since
from backend.simulation import SimConfig, create_env

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _synthetic_samples(n: int = 20) -> list:
    """Minimal (state, action, next_state) triples — stub fine_tune ignores them."""
    return [
        (
            np.zeros(806, dtype=np.float32),
            np.zeros(12, dtype=np.float32),
            np.zeros(806, dtype=np.float32),
        )
        for _ in range(n)
    ]


def _stub_world_model(belief: BeliefState) -> SimpleNamespace:
    """WorldModel contract stub backed by a REAL BeliefState."""

    async def fake_fine_tune(samples, epochs: int = 5) -> float:
        return 0.01

    return SimpleNamespace(belief=belief, fine_tune=fake_fine_tune)


@pytest.fixture
def fast_timeout(monkeypatch):
    """AGENT_TIMEOUT_SECONDS read at call time from module global — patchable."""
    monkeypatch.setattr(capability_module, "AGENT_TIMEOUT_SECONDS", 60)


@pytest.fixture
def stubbed_collect(monkeypatch):
    """Replace the 1000-step collection with 20 synthetic samples."""

    async def fast_collect(self, env, n: int = 1000):
        return _synthetic_samples(20)

    monkeypatch.setattr(CapabilityAgent, "collect_dof_samples", fast_collect)


# ---------------------------------------------------------------------------
# CapabilityAgent
# ---------------------------------------------------------------------------

async def test_capability_unlock_event_order(fast_timeout, stubbed_collect):
    """capability_unlocked must be written BEFORE agent_completed; DOF update
    must land atomically on both sims + belief; run() reports success."""
    sim_a = await create_env(SimConfig(seed=42, dof=1))
    sim_b = await create_env(SimConfig(seed=42, dof=1))
    try:
        belief = BeliefState()
        belief.update_from_observation(sim_a.get_state_vector())
        world_model = _stub_world_model(belief)

        agent = CapabilityAgent("cap_2", world_model, sim_a, sim_b)
        output = await agent.run()

        # Event ordering invariant.
        events = get_events_since(0)
        types = [e["event_type"] for e in events]
        assert "capability_unlocked" in types
        assert "agent_completed" in types
        assert types.index("capability_unlocked") < types.index("agent_completed")

        unlocked = next(e for e in events if e["event_type"] == "capability_unlocked")
        assert unlocked["agent_id"] == "cap_2"
        assert unlocked["payload"]["new_dof"] == 2
        assert unlocked["payload"]["previous_dof"] == 1
        # Spec indexing: DOF_THRESHOLDS[target_dof - 1] = DOF_THRESHOLDS[1].
        assert unlocked["payload"]["threshold"] == pytest.approx(0.08)

        # Atomic DOF update across all three holders.
        assert sim_a.config.dof == 2
        assert sim_b.config.dof == 2
        assert belief.active_dof == 2

        # Output contract.
        assert output["success"] is True
        assert output["samples_collected"] == 20
        assert output["fine_tune_loss"] == pytest.approx(0.01)
        assert output["target_dof"] == 2
    finally:
        await asyncio.to_thread(sim_a.close)
        await asyncio.to_thread(sim_b.close)


async def test_capability_seed_formula(fast_timeout, stubbed_collect, monkeypatch):
    """The dedicated DOF env must be created with seed = 42 + target_dof."""
    sim_a = await create_env(SimConfig(seed=42, dof=1, n_vessels=2))
    sim_b = await create_env(SimConfig(seed=42, dof=1))
    try:
        captured_configs: list[SimConfig] = []
        real_create_env = capability_module.create_env

        async def capturing_create_env(config: SimConfig):
            captured_configs.append(config)
            return await real_create_env(config)

        monkeypatch.setattr(
            capability_module, "create_env", capturing_create_env
        )

        belief = BeliefState()
        belief.update_from_observation(sim_a.get_state_vector())
        world_model = _stub_world_model(belief)

        agent = CapabilityAgent("cap_2", world_model, sim_a, sim_b)
        output = await agent.run()
        assert output["success"] is True

        assert len(captured_configs) == 1
        cfg = captured_configs[0]
        assert cfg.seed == 42 + 2          # SEED FORMULA: 42 + target_dof
        assert cfg.dof == 2
        assert cfg.n_vessels == sim_a.config.n_vessels
    finally:
        await asyncio.to_thread(sim_a.close)
        await asyncio.to_thread(sim_b.close)


# ---------------------------------------------------------------------------
# ReactiveAgent
# ---------------------------------------------------------------------------

def test_reactive_isolation():
    """REACTIVE_AGENT_ISOLATION INVARIANT: zero imports from world_model,
    prediction_net, belief_state, orchestrator — and never anthropic."""
    result = subprocess.run(
        [
            "grep",
            "-E",
            r"from backend\.(world_model|prediction_net|belief_state|orchestrator)"
            r"|import anthropic",
            str(REPO_ROOT / "backend" / "agents" / "reactive.py"),
        ],
        capture_output=True,
    )
    assert result.stdout == b"", (
        f"ReactiveAgent imports forbidden modules:\n{result.stdout.decode()}"
    )


def _make_state_vec(
    *,
    ee_xy: tuple[float, float],
    tgt_xy: tuple[float, float],
    bleeding_cells: int = 0,
) -> np.ndarray:
    vec = np.zeros(806, dtype=np.float32)
    vec[768], vec[769] = ee_xy
    vec[776], vec[777] = tgt_xy
    if bleeding_cells > 0:
        vec[512 : 512 + bleeding_cells] = 1.0
    return vec


def test_reactive_select_action():
    """Greedy branches: bleed→CAUTERIZE; far→MOVE@0.1; near→MOVE@0.05."""
    # _select_action is pure — a bare object satisfies the constructor.
    agent = ReactiveAgent(SimpleNamespace(config=SimpleNamespace(dof=1)))

    # Bleeding active → CAUTERIZE mode one-hot, magnitude 1.0.
    action = agent._select_action(_make_state_vec(
        ee_xy=(0.5, 0.5), tgt_xy=(0.8, 0.8), bleeding_cells=4
    ))
    assert action.shape == (12,)
    assert action.dtype == np.float32
    assert action[9] == 1.0
    assert action[11] == 1.0

    # Far from target → MOVE toward it at speed 0.1, magnitude 0.7.
    action = agent._select_action(_make_state_vec(ee_xy=(0.2, 0.2), tgt_xy=(0.8, 0.8)))
    assert action[7] == 1.0
    assert action[11] == pytest.approx(0.7)
    unit = np.array([0.6, 0.6]) / np.linalg.norm([0.6, 0.6])
    np.testing.assert_allclose(action[0:2], unit * 0.1, atol=1e-6)

    # Near target → slowed approach, speed 0.05, magnitude 0.4.
    action = agent._select_action(_make_state_vec(ee_xy=(0.79, 0.79), tgt_xy=(0.8, 0.8)))
    assert action[7] == 1.0
    assert action[11] == pytest.approx(0.4)
    np.testing.assert_allclose(action[0:2], unit * 0.05, atol=1e-6)


async def test_reactive_loop_smoke():
    """Live loop writes simulation_step events for sim_id='comparison';
    cancellation is the only way it stops; no other exceptions escape."""
    env = await create_env(SimConfig(seed=42, dof=1))
    try:
        agent = ReactiveAgent(env)
        task = asyncio.create_task(agent.run())
        await asyncio.sleep(1.2)
        task.cancel()
        results = await asyncio.gather(task, return_exceptions=True)

        # Only CancelledError may surface from cancellation.
        for r in results:
            if isinstance(r, BaseException):
                assert isinstance(r, asyncio.CancelledError), f"unexpected: {r!r}"

        steps = [
            e for e in get_events_since(0)
            if e["event_type"] == "simulation_step" and e["sim_id"] == "comparison"
        ]
        assert len(steps) >= 1
        first = steps[0]["payload"]
        assert "reward" in first and "tissue_mean" in first
        assert first["active_dof"] == 1
        assert len(first["ee_pos"]) == 3
    finally:
        await asyncio.to_thread(env.close)
