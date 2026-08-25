from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend import api as api_module
from backend import event_log
from backend.api import (
    STATE,
    ConnectionManager,
    app,
    set_env_registry,
    set_world_model,
)
from backend.belief_state import BeliefState
from backend.event_log import init_db, write_event


class DeadSocketError(RuntimeError):
    pass


class FakeCraftaxEnv:
    def __init__(self) -> None:
        self.done = True
        self.last_reward = 1.25
        self.reset_calls = 0

    def observation(self) -> dict[str, np.ndarray]:
        return {
            "token_2d": np.arange(396, dtype=np.int32).reshape(99, 4),
            "vector": np.arange(47, dtype=np.float32),
            "token": np.array([3], dtype=np.int32),
        }

    def raw_observation(self) -> np.ndarray:
        return np.arange(8268, dtype=np.float32)

    def achievements(self) -> np.ndarray:
        values = np.zeros(67, dtype=bool)
        values[[0, 2, 5]] = True
        return values

    def game_stats(self) -> dict[str, float]:
        return {
            "health": 7.0,
            "drink": 6.0,
            "food": 5.0,
            "energy": 4.0,
            "light_level": 3.0,
            "is_sleeping": 0.0,
            "is_resting": 1.0,
        }


class FakeCraftaxDriver:
    def __init__(self, step_count: int = 17) -> None:
        self.env = FakeCraftaxEnv()
        self.step_count = step_count

    def reset(self) -> None:
        self.env.reset_calls += 1
        self.step_count = 0
        self.env.done = False
        self.env.last_reward = 0.0


@pytest.fixture
def registered_runtime() -> tuple[FakeCraftaxDriver, FakeCraftaxDriver, BeliefState]:
    primary = FakeCraftaxDriver()
    comparison = FakeCraftaxDriver(step_count=9)
    belief = BeliefState(active_dof=4)
    belief.set_jsd_error_map(np.full((16, 16), 0.2, dtype=np.float32))
    set_env_registry({"primary": primary, "comparison": comparison})
    set_world_model(SimpleNamespace(belief=belief))
    yield primary, comparison, belief
    api_module._env_registry.clear()
    api_module._world_model_ref.clear()


def test_ws_replay_connect() -> None:
    init_db()
    for index in range(6):
        write_event("system_ready", seq=index)

    with TestClient(app) as client, client.websocket_connect("/ws/events") as ws:
        messages = [ws.receive_json() for _ in range(6)]

    assert [message["data"]["payload"]["seq"] for message in messages] == list(range(6))
    assert all(message["type"] == "event" for message in messages)


async def test_connection_manager_broadcast_removes_dead_clients() -> None:
    sent: list[dict[str, str]] = []

    class FakeWebSocket:
        def __init__(self, sink: list[dict[str, str]] | None) -> None:
            self._sink = sink

        async def send_json(self, message: dict[str, str]) -> None:
            if self._sink is None:
                raise DeadSocketError
            self._sink.append(message)

    manager = ConnectionManager()
    healthy = FakeWebSocket(sent)
    dead = FakeWebSocket(None)
    manager.active_connections = [healthy, dead]

    await manager.broadcast({"type": "ping"})

    assert sent == [{"type": "ping"}]
    assert manager.active_connections == [healthy]


def test_health_schema_is_byte_compatible() -> None:
    original = dict(STATE)
    STATE.update(
        status="starting",
        world_model_version=0,
        global_error=0.0,
        active_agents=0,
        primary_sim_step=0,
        training_progress=None,
    )
    try:
        with TestClient(app) as client:
            response = client.get("/health")
    finally:
        STATE.clear()
        STATE.update(original)

    assert response.status_code == 200
    assert response.content == (
        b'{"status":"starting","world_model_version":0,"global_error":0.0,'
        b'"active_agents":0,"primary_sim_step":0,"training_progress":null}'
    )


def test_health_reflects_state_mutation() -> None:
    original = dict(STATE)
    try:
        STATE["active_agents"] = 2
        with TestClient(app) as client:
            assert client.get("/health").json()["active_agents"] == 2
    finally:
        STATE.clear()
        STATE.update(original)


def test_events_endpoint_since_filter() -> None:
    init_db()
    write_event("system_ready", marker="first")
    time.sleep(0.050)
    write_event("system_ready", marker="second")
    time.sleep(0.005)
    write_event("system_ready", marker="third")
    all_events = event_log.get_events_since(0.0)
    midpoint = (all_events[0]["timestamp"] + all_events[2]["timestamp"]) / 2

    with TestClient(app) as client:
        response = client.get("/events", params={"since": midpoint})

    assert [event["payload"]["marker"] for event in response.json()] == ["second", "third"]


def test_events_agent_endpoint_subtree() -> None:
    init_db()
    write_event("agent_spawned", agent_id="exp_x", parent_id="orchestrator")
    write_event("agent_step", agent_id="exp_y", parent_id="exp_x", step=1)

    with TestClient(app) as client:
        response = client.get("/events/exp_x")

    assert {event["agent_id"] for event in response.json()} == {"exp_x", "exp_y"}


@pytest.mark.parametrize("sim_id", ["primary", "comparison"])
def test_detail_exposes_only_observed_craftax_values(
    registered_runtime: tuple[FakeCraftaxDriver, FakeCraftaxDriver, BeliefState],
    sim_id: str,
) -> None:
    _, _, belief = registered_runtime

    with TestClient(app) as client:
        response = client.get(f"/detail/{sim_id}")

    assert response.status_code == 200
    body = response.json()
    simulation = body["simulation_state"]
    assert set(simulation) == {
        "simulation_id", "token_2d", "vector", "direction", "raw_observation",
        "achievements_count", "stats", "step", "done", "reward", "active_dof",
    }
    assert len(simulation["token_2d"]) == 99
    assert len(simulation["token_2d"][0]) == 4
    assert len(simulation["vector"]) == 47
    assert len(simulation["raw_observation"]) == 8268
    assert simulation["direction"] == [3]
    assert simulation["achievements_count"] == 3
    assert simulation["active_dof"] == belief.active_dof
    assert set(simulation["stats"]) == {
        "health", "drink", "food", "energy", "light", "is_sleeping", "is_resting",
    }
    assert body["belief_state"] == belief.snapshot()
    assert not {"tissue_integrity", "tissue_vascularity", "bleeding_mask", "vessels"} & set(simulation)


def test_detail_unknown_sim_returns_404() -> None:
    with TestClient(app) as client:
        assert client.get("/detail/unknown").status_code == 404


def test_reset_operates_on_registered_driver(
    registered_runtime: tuple[FakeCraftaxDriver, FakeCraftaxDriver, BeliefState],
) -> None:
    primary, _, _ = registered_runtime

    with TestClient(app) as client:
        response = client.post("/simulation/primary/reset")

    assert response.json() == {"ok": True}
    assert primary.env.reset_calls == 1
    assert primary.step_count == 0


def test_debug_endpoints_override_real_jsd_map(
    registered_runtime: tuple[FakeCraftaxDriver, FakeCraftaxDriver, BeliefState],
) -> None:
    _, _, belief = registered_runtime
    belief.regional_override = {name: 0.1 for name in belief.get_regional_errors()}

    with TestClient(app) as client:
        assert client.get("/debug/reset_belief").status_code == 200
        assert belief.prediction_error_map.mean() == pytest.approx(0.5)
        assert belief.regional_override is None
        assert client.get("/debug/trigger_agent").status_code == 200

    assert belief.prediction_error_map[0:8, 0:8].mean() == pytest.approx(0.8)


def test_craftax_stream_is_documented_404(
    registered_runtime: tuple[FakeCraftaxDriver, FakeCraftaxDriver, BeliefState],
) -> None:
    with TestClient(app) as client:
        response = client.get("/sim/primary/stream")

    assert response.status_code == 404
    assert response.json()["detail"] == "Craftax MJPEG stream is unavailable"
