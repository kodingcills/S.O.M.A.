"""Tests for backend/orchestrator.py — SOMA Task 1.6.

Named tests per ORCHESTRATION.md TESTING PROTOCOL + ARCHITECTURE.md invariants:
  test_graph_compiles                     — test:orchestrator:graph_compiles
  test_route_returns_valid_strings        — test:orchestrator:route_valid_strings
  test_route_decision_unlock              — priority 1: DOF unlock routing
  test_route_decision_spawn               — priority 2: exploration routing
  test_circuit_breaker                    — open breaker forces "wait"
  test_max_concurrent_enforced            — 4 active agents → "wait"
  test_spawn_writes_event_first           — AGENT_SPAWNED_BEFORE_LOOP (C4) — MUST PASS
  test_runner_marks_failure_without_module— lazy-import fallback (pre-Task 1.7a)
  test_runner_completes_with_stub_module  — lazy-import success path
  test_graph_wait_cycle_bounded           — wiring smoke; documents recursion_limit
  test_graph_spawn_chain_writes_events    — Send fan-out through the real graph

The event DB is isolated per-test by the autouse conftest fixture (SOMA_DB_PATH).
"""

import sys
import time
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from langgraph.errors import GraphRecursionError

from backend import orchestrator
from backend.event_log import get_events_since, init_db
from backend.orchestrator import (
    CIRCUIT_BREAKER_THRESHOLD,
    DOF_THRESHOLDS,
    EXPLORE_THRESHOLD,
    MAX_CONCURRENT_AGENTS,
    build_soma_graph,
    exploration_agent_runner,
    initial_state,
    route_orchestrator,
    spawn_exploration_node,
)

REGIONS = (
    "upper_left",
    "upper_right",
    "lower_left",
    "lower_right",
    "tool_tissue_boundary",
    "surgical_target_vicinity",
)


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------

def make_errors(**overrides: float) -> dict[str, float]:
    """Regional errors dict with the exact 6 keys BeliefState produces."""
    errors = {region: 0.05 for region in REGIONS}
    errors.update(overrides)
    return errors


def make_state(**overrides) -> dict:
    state = {
        "regional_errors": make_errors(),
        "global_mean_error": 0.5,
        "active_dof": 1,
        "active_agents": [],
        "cycle_count": 0,
        "completed_agents": [],
        "failed_agents": [],
        "unlock_requested": False,
        "target_dof": 1,
        "region_failure_counts": {},
        "region_backoff_until": {},
        "agent_region": {},
    }
    state.update(overrides)
    return state


def make_mock_world_model(regional: float = 0.5, active_dof: int = 1):
    """SimpleNamespace matching the WorldModel surface orchestrator_node reads.

    global_mean_error is derived from prediction_error_map.mean(), so the map
    is filled with the same value the regional errors carry.
    """
    return SimpleNamespace(
        belief=SimpleNamespace(
            get_regional_errors=lambda: {r: regional for r in REGIONS},
            prediction_error_map=np.full((16, 16), regional, dtype=np.float32),
            active_dof=active_dof,
        )
    )


def make_mock_driver(n_achievements: int = 0):
    achievements = np.zeros(67, dtype=bool)
    achievements[:n_achievements] = True
    return SimpleNamespace(
        env=SimpleNamespace(achievements=lambda: achievements.copy())
    )


# ---------------------------------------------------------------------------
# test:orchestrator:graph_compiles
# ---------------------------------------------------------------------------

def test_graph_compiles():
    graph = build_soma_graph(make_mock_world_model(), make_mock_driver())
    assert graph is not None


# ---------------------------------------------------------------------------
# test:orchestrator:route_valid_strings
# ---------------------------------------------------------------------------

def test_route_returns_valid_strings():
    low = make_state(global_mean_error=0.01)
    high = make_state(regional_errors=make_errors(upper_left=0.8))
    mixed = make_state(
        regional_errors=make_errors(lower_right=0.2),
        global_mean_error=0.07,
        active_dof=2,
    )
    for state in (low, high, mixed):
        result = route_orchestrator(state)
        assert result in ("spawn_explore", "unlock_dof", "wait")


# ---------------------------------------------------------------------------
# Routing priorities
# ---------------------------------------------------------------------------

def test_route_decision_unlock():
    state = make_state(unlock_requested=True, target_dof=2)
    assert route_orchestrator(state) == "unlock_dof"


async def test_unlock_fires_on_achievement_bucket_increase(monkeypatch):
    # Given: Craftax has crossed the four-achievement boundary while belief
    # still exposes DOF 1, and JSD error is too high for the legacy threshold.
    monkeypatch.setattr(orchestrator, "ORCHESTRATOR_SLEEP_S", 0.0)
    world_model = make_mock_world_model(regional=0.10, active_dof=1)
    achievements = np.zeros(67, dtype=bool)
    achievements[:4] = True
    driver = SimpleNamespace(
        env=SimpleNamespace(achievements=lambda: achievements.copy())
    )
    graph = build_soma_graph(world_model, driver)
    config = {
        "recursion_limit": 12,
        "configurable": {"thread_id": "t-achievement-unlock"},
    }

    # When: one bounded orchestration burst observes the real achievement set.
    with pytest.raises(GraphRecursionError):
        await graph.ainvoke(initial_state(), config=config)

    # Then: the bucket increase unlocks DOF 2 using the real JSD mean.
    unlock = next(
        event
        for event in get_events_since(0)
        if event["event_type"] == "capability_unlocked"
    )
    assert unlock["payload"]["previous_dof"] == 1
    assert unlock["payload"]["new_dof"] == 2
    assert unlock["payload"]["trigger_error"] == pytest.approx(0.10)
    assert unlock["payload"]["threshold"] == DOF_THRESHOLDS[1]
    assert world_model.belief.active_dof == 2


def test_route_decision_spawn():
    # upper_left error 0.8 >= EXPLORE_THRESHOLD=0.15; unlock not eligible.
    state = make_state(regional_errors=make_errors(upper_left=0.8))
    assert route_orchestrator(state) == "spawn_explore"
    assert EXPLORE_THRESHOLD == 0.15  # guard the constant the test depends on


def test_circuit_breaker():
    # Region at failure threshold with future backoff must NOT spawn,
    # despite its error being far above EXPLORE_THRESHOLD.
    state = make_state(
        regional_errors=make_errors(upper_left=0.8),
        region_failure_counts={"upper_left": CIRCUIT_BREAKER_THRESHOLD},
        region_backoff_until={"upper_left": time.time() + 60.0},
    )
    assert route_orchestrator(state) == "wait"


def test_max_concurrent_enforced():
    agents = [f"exp_{i:06d}" for i in range(MAX_CONCURRENT_AGENTS)]
    state = make_state(
        active_agents=agents,
        regional_errors=make_errors(upper_left=0.8, lower_right=0.9),
    )
    assert route_orchestrator(state) == "wait"


# ---------------------------------------------------------------------------
# test:agents:spawned_before_loop — AGENT_SPAWNED_BEFORE_LOOP (C4) — MUST PASS
# ---------------------------------------------------------------------------

async def test_spawn_writes_event_first():
    init_db()  # idempotent; conftest already pointed SOMA_DB_PATH at tmp
    state = make_state(regional_errors=make_errors(upper_left=0.8))

    sends = await spawn_exploration_node(state)

    events = get_events_since(0)
    assert events, "spawn node wrote no events"
    first = events[0]
    assert first["event_type"] == "agent_spawned"
    assert first["agent_id"].startswith("exp_")
    assert first["parent_id"] == "orchestrator"
    assert first["region"] == "upper_left"
    assert first["error_before"] == pytest.approx(0.8)

    assert len(sends) == 1
    assert sends[0].node == "exploration_agent_runner"
    assert sends[0].arg["agent_id"] == first["agent_id"]
    assert sends[0].arg["region"] == "upper_left"


# ---------------------------------------------------------------------------
# Lazy runner: fallback + success paths (exploration.py lands in Task 1.7a)
# ---------------------------------------------------------------------------

async def test_runner_marks_failure_without_module(monkeypatch):
    # sys.modules entry None forces `from ... import` to raise ImportError,
    # so this stays valid after Task 1.7a adds the real module.
    monkeypatch.setitem(sys.modules, "backend.agents.exploration", None)

    out = await exploration_agent_runner(
        {"agent_id": "exp_dead01", "region": "upper_left", "error_before": 0.8}
    )

    assert out == {"failed_agents": ["exp_dead01"]}
    failures = [
        e for e in get_events_since(0) if e["event_type"] == "agent_failed"
    ]
    assert failures and failures[0]["agent_id"] == "exp_dead01"


async def test_runner_completes_with_stub_module(monkeypatch):
    class StubAgent:
        def __init__(
            self, agent_id: str, world_model, driver, region: str, error_before: float
        ) -> None:
            self.agent_id = agent_id

        async def run(self) -> dict:
            return {"success": True}

    stub = ModuleType("backend.agents.exploration")
    stub.ExplorationAgent = StubAgent
    monkeypatch.setitem(sys.modules, "backend.agents.exploration", stub)
    build_soma_graph(make_mock_world_model(), make_mock_driver())

    out = await exploration_agent_runner(
        {"agent_id": "exp_live01", "region": "lower_left", "error_before": 0.4}
    )

    assert out == {"completed_agents": ["exp_live01"]}


# ---------------------------------------------------------------------------
# Wiring smokes — validate the compiled graph end to end and document that
# the orchestrator loop never reaches END by design (see module docstring of
# backend/orchestrator.py): ainvoke must be bounded via recursion_limit.
# ---------------------------------------------------------------------------

async def _run_bounded(world_model, limit: int, thread_id: str):
    graph = build_soma_graph(world_model, make_mock_driver())
    config = {"recursion_limit": limit, "configurable": {"thread_id": thread_id}}
    with pytest.raises(GraphRecursionError):
        await graph.ainvoke(initial_state(), config=config)


async def test_graph_wait_cycle_bounded(monkeypatch):
    # All regions below EXPLORE_THRESHOLD, global above unlock threshold:
    # pure wait loop — orchestrator ↔ collect_status forever until bounded.
    monkeypatch.setattr(orchestrator, "ORCHESTRATOR_SLEEP_S", 0.0)
    await _run_bounded(make_mock_world_model(regional=0.10), limit=12, thread_id="t-wait")


async def test_graph_spawn_chain_writes_events(monkeypatch):
    # High errors everywhere: real Send fan-out → lazy runners fail (no
    # exploration.py yet) → agent_spawned AND agent_failed both recorded.
    monkeypatch.setattr(orchestrator, "ORCHESTRATOR_SLEEP_S", 0.0)
    await _run_bounded(make_mock_world_model(regional=0.5), limit=40, thread_id="t-spawn")

    event_types = {e["event_type"] for e in get_events_since(0)}
    assert "agent_spawned" in event_types
    assert "agent_failed" in event_types
