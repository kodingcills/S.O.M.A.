"""Tests for backend/api.py — SOMA Task 1.1.

Named tests per ARCHITECTURE.md TESTING PROTOCOL:
  test_ws_replay_connect           — WEBSOCKET REPLAY INVARIANT (C6) — MUST PASS
  test_health_schema               — /health response contract
  test_events_endpoint_since_filter — REST fallback polling contract
  test_events_agent_endpoint_subtree — node-click drill-down contract
  test_mjpeg_stream_stub_404       — intentional stub until Task 2.x
"""

import asyncio
import time

from backend import event_log
from backend.api import STATE, ConnectionManager, app
from backend.event_log import init_db, write_event
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# test:api:ws_replay — WEBSOCKET REPLAY INVARIANT (C6)
# ---------------------------------------------------------------------------

def test_ws_replay_connect():
    # Seed history BEFORE connecting: a late-joining client must receive the
    # full replay burst (type=event dicts) before any live traffic.
    init_db()
    for i in range(6):
        write_event("system_ready", seq=i)

    with TestClient(app) as client:
        with client.websocket_connect("/ws/events") as ws:
            messages = [ws.receive_json() for _ in range(6)]

    assert all(m["type"] == "event" for m in messages)
    assert [m["data"]["payload"]["seq"] for m in messages] == [0, 1, 2, 3, 4, 5]
    assert all(m["data"]["event_type"] == "system_ready" for m in messages)


async def test_connection_manager_broadcast_removes_dead_clients():
    # Live delivery requires write_event to run inside the app's event loop
    # (true for every runtime component; a sync test thread has no loop, so
    # an end-to-end live test can't drive it). Test the manager directly.
    manager = ConnectionManager()
    sent: list[dict] = []

    class FakeWebSocket:
        def __init__(self, sink: list[dict] | None):
            self._sink = sink

        async def send_json(self, message: dict) -> None:
            if self._sink is None:
                raise RuntimeError("socket is dead")
            self._sink.append(message)

    healthy = FakeWebSocket(sent)
    dead = FakeWebSocket(None)
    manager.active_connections = [healthy, dead]

    await manager.broadcast({"type": "ping"})

    assert sent == [{"type": "ping"}]
    assert manager.active_connections == [healthy]


# ---------------------------------------------------------------------------
# test:api:health_schema
# ---------------------------------------------------------------------------

def test_health_schema():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {
        "status",
        "world_model_version",
        "global_error",
        "active_agents",
        "primary_sim_step",
        "training_progress",
    }
    assert body["status"] == "starting"
    assert body["world_model_version"] == 0
    assert body["global_error"] == 0.0
    assert body["active_agents"] == 0
    assert body["primary_sim_step"] == 0
    assert body["training_progress"] is None


def test_health_reflects_state_mutation():
    # Later tasks mutate the module-level STATE dict; /health must follow.
    original = dict(STATE)
    try:
        STATE["active_agents"] = 2
        with TestClient(app) as client:
            body = client.get("/health").json()
        assert body["active_agents"] == 2
    finally:
        STATE.clear()
        STATE.update(original)


# ---------------------------------------------------------------------------
# test:api:events_since_filter
# ---------------------------------------------------------------------------

def test_events_endpoint_since_filter():
    init_db()
    # Asymmetric gaps keep the midpoint far from BOTH neighbors: the filter
    # is strict `>`, so `second` must land in the LATER half (first gap
    # longer). Equal 2ms gaps made that a coin flip on scheduler jitter
    # (observed ~1/5 failures in isolation).
    write_event("system_ready", marker="first")
    time.sleep(0.050)
    write_event("system_ready", marker="second")
    time.sleep(0.005)
    write_event("system_ready", marker="third")

    all_events = event_log.get_events_since(0.0)
    assert len(all_events) == 3
    midpoint = (all_events[0]["timestamp"] + all_events[2]["timestamp"]) / 2

    with TestClient(app) as client:
        response = client.get("/events", params={"since": midpoint})

    assert response.status_code == 200
    markers = [e["payload"]["marker"] for e in response.json()]
    assert markers == ["second", "third"]


def test_events_agent_endpoint_subtree():
    init_db()
    write_event("agent_spawned", agent_id="exp_x", parent_id="orchestrator")
    write_event("agent_step", agent_id="exp_y", parent_id="exp_x", step=1)

    with TestClient(app) as client:
        response = client.get("/events/exp_x")

    assert response.status_code == 200
    agent_ids = {e["agent_id"] for e in response.json()}
    assert agent_ids == {"exp_x", "exp_y"}


# ---------------------------------------------------------------------------
# Stubs: real implementations arrive in later tasks
# ---------------------------------------------------------------------------

def test_stub_endpoints():
    with TestClient(app) as client:
        assert client.post("/simulation/primary/reset").json() == {"ok": True}


def test_detail_unknown_sim_returns_404():
    with TestClient(app) as client:
        assert client.get("/detail/primary").status_code == 404


def test_detail_atomic_shape_with_registered_sim():
    import asyncio
    from types import SimpleNamespace

    from backend.api import set_env_registry, set_world_model
    from backend.belief_state import BeliefState
    from backend.simulation import SimConfig, create_env

    async def _setup():
        env = await create_env(SimConfig(seed=42, dof=1, n_vessels=2))
        env.reset()
        return env

    env = asyncio.run(_setup())
    belief = BeliefState()
    belief.update_from_observation(env.get_state_vector())
    set_env_registry({"primary": env})
    set_world_model(SimpleNamespace(belief=belief))
    try:
        with TestClient(app) as client:
            r = client.get("/detail/primary")
            assert r.status_code == 200
            body = r.json()
            sim = body["simulation_state"]
            bel = body["belief_state"]
            assert len(sim["tissue_integrity"]) == 16
            assert len(sim["tissue_integrity"][0]) == 16
            assert isinstance(sim["bleeding_mask"][0][0], bool)
            assert len(sim["vessels"]) == len(env.vessels)
            assert sim["surgical_target"]["position"] is not None
            assert set(bel.keys()) >= {
                "global_mean_error", "regional_errors",
                "world_model_version", "error_map",
            }
    finally:
        from backend import api as api_module

        api_module._env_registry.clear()
        api_module._world_model_ref.clear()
        asyncio.run(asyncio.to_thread(env.close))


def test_debug_endpoints_require_world_model():
    from types import SimpleNamespace

    import pytest

    from backend.api import set_world_model
    from backend.belief_state import BeliefState

    with TestClient(app) as client:
        r = client.get("/debug/reset_belief")
        assert r.status_code == 503

        wm = SimpleNamespace(belief=BeliefState())
        set_world_model(wm)
        try:
            r = client.get("/debug/reset_belief")
            assert r.status_code == 200
            assert wm.belief.prediction_error_map.mean() == pytest.approx(0.5)

            r = client.get("/debug/trigger_agent")
            assert r.status_code == 200
            assert wm.belief.prediction_error_map[0:8, 0:8].mean() == pytest.approx(0.8)
        finally:
            from backend import api as api_module

            api_module._world_model_ref.clear()


def test_mjpeg_stream_stub_returns_404():
    with TestClient(app) as client:
        assert client.get("/sim/primary/stream").status_code == 404


# ---------------------------------------------------------------------------
# test:api:mjpeg_stream — real PyBullet MJPEG endpoint (Task 2.3)
# ---------------------------------------------------------------------------

async def test_mjpeg_generator_real_env_frames():
    """Direct generator probe: real PyBullet camera produces JPEG frames."""
    from backend.api import pybullet_frame_generator
    from backend.simulation import SimConfig, create_env

    env = await create_env(SimConfig(seed=42, dof=1, n_vessels=2))
    try:
        frames = []
        async for chunk in pybullet_frame_generator(env, max_frames=2):
            frames.append(chunk)
        assert len(frames) == 2
        for chunk in frames:
            assert chunk.startswith(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
            payload = chunk.split(b"\r\n\r\n", 1)[1]
            assert payload[:2] == b"\xff\xd8", "JPEG SOI marker missing"
    finally:
        await asyncio.to_thread(env.close)


def test_mjpeg_stream_endpoint_wiring(monkeypatch):
    """HTTP contract via canned generator — routing/headers/termination."""
    from backend import api as api_module
    from backend.api import set_env_registry

    async def canned(env, max_frames: int = 0):
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n\xff\xd8stub\r\n"

    monkeypatch.setattr(api_module, "pybullet_frame_generator", canned)
    set_env_registry({"primary": object()})
    try:
        with TestClient(app) as client:
            with client.stream("GET", "/sim/primary/stream") as response:
                assert response.status_code == 200
                assert response.headers["content-type"].startswith(
                    "multipart/x-mixed-replace"
                )
                body = b"".join(response.iter_bytes())
                assert b"--frame" in body and b"\xff\xd8" in body
    finally:
        api_module._env_registry.clear()
