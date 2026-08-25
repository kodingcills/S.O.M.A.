"""FastAPI layer — SOMA Task 1.1.

REST endpoints + WebSocket event stream with full-history replay.

Invariants enforced here:
  - WEBSOCKET REPLAY INVARIANT (C6): on connect, replay ALL events from
    get_events_since(0) BEFORE adding the client to the broadcast set.
    Known race (RISKS D-7): events written between replay completion and
    broadcast-set append are missed by that client — accepted per spec.
  - /health schema is the contract consumed by the frontend top bar.
  - api.py never imports world_model — dependencies arrive via params in
    later tasks (FILE OWNERSHIP MAP).

Dual-import shim: this module must run BOTH as `uvicorn api:app`
(cwd=backend) and as `backend.api` (repo root). Same shim as main.py.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from contextlib import asynccontextmanager
from importlib import import_module
from typing import Any, Protocol, TypedDict

import anyio
import numpy as np

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.websockets import WebSocket, WebSocketDisconnect

_event_log = import_module("backend.event_log" if __package__ else "event_log")
get_events_for_agent = _event_log.get_events_for_agent
get_events_since = _event_log.get_events_since
init_db = _event_log.init_db
set_broadcast_callback = _event_log.set_broadcast_callback

PING_INTERVAL_S = 5.0
PONG_TIMEOUT_S = 15.0


class CraftaxStats(TypedDict):
    health: float
    drink: float
    food: float
    energy: float
    light: float
    is_sleeping: float
    is_resting: float


class CraftaxEnvironment(Protocol):
    done: bool
    last_reward: float

    def observation(self) -> dict[str, np.ndarray]: ...

    def raw_observation(self) -> np.ndarray: ...

    def achievements(self) -> np.ndarray: ...

    def game_stats(self) -> dict[str, float]: ...


class CraftaxDriverProvider(Protocol):
    env: CraftaxEnvironment
    step_count: int

    def reset(self, seed: int | None = None) -> None: ...


class BeliefProvider(Protocol):
    active_dof: int
    prediction_error_map: np.ndarray
    confidence_map: np.ndarray
    regional_override: dict[str, float] | None

    def snapshot(self) -> dict[str, Any]: ...


class WorldModelProvider(Protocol):
    belief: BeliefProvider

# Module-level runtime state. Later tasks (orchestrator, world model)
# mutate this dict in place; /health always reads the live values.
STATE: dict[str, Any] = {
    "status": "starting",
    "world_model_version": 0,
    "global_error": 0.0,
    "active_agents": 0,
    "primary_sim_step": 0,
    "training_progress": None,
}


class ConnectionManager:
    """WebSocket registry with replay-first connect and silent disconnect."""

    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []
        # Per-connection send lock: ping task + broadcasts must never
        # interleave frames on the same socket.
        self._send_locks: dict[WebSocket, anyio.Lock] = {}

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        # C6: replay ALL history BEFORE joining the broadcast set.
        for event in get_events_since(0.0):
            await self._send(websocket, {"type": "event", "data": event})
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        # Silent removal — a dying client must never raise upstream.
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        self._send_locks.pop(websocket, None)

    async def _send(self, websocket: WebSocket, message: dict[str, Any]) -> None:
        lock = self._send_locks.setdefault(websocket, anyio.Lock())
        async with lock:
            await websocket.send_json(message)

    async def broadcast(self, message: dict[str, Any]) -> None:
        for websocket in list(self.active_connections):
            try:
                await self._send(websocket, message)
            except (RuntimeError, WebSocketDisconnect):
                self.disconnect(websocket)


manager = ConnectionManager()


async def _ping_loop(websocket: WebSocket) -> None:
    try:
        while True:
            await anyio.sleep(PING_INTERVAL_S)
            await manager._send(websocket, {"type": "ping"})
    except (RuntimeError, WebSocketDisconnect):
        return  # socket died; endpoint's finally handles cleanup


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    set_broadcast_callback(manager.broadcast)
    yield
    set_broadcast_callback(None)


app = FastAPI(title="SOMA", lifespan=lifespan)

# Vite dev origin must reach REST directly (WS is exempt from CORS;
# without this, useHealth/useDetail polling silently fails cross-origin).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, Any]:
    return dict(STATE)


@app.get("/events")
async def events_since(since: float = Query(default=0.0)) -> list[dict[str, Any]]:
    return get_events_since(since)


@app.get("/events/{agent_id}")
async def events_for_agent(agent_id: str) -> list[dict[str, Any]]:
    return get_events_for_agent(agent_id)


@app.get("/detail/{sim_id}")
def detail(sim_id: str) -> dict[str, Any]:
    driver = _env_registry.get(sim_id)
    wm = _world_model_ref[0] if _world_model_ref else None
    if driver is None or wm is None:
        raise HTTPException(status_code=404, detail=f"unknown sim_id '{sim_id}'")

    # ATOMICITY (ARCHITECTURE.md): both reads are in-memory; NO await between
    # them so the panel's heatmap and sim view reflect one logical timestep.
    observation = driver.env.observation()
    raw_observation = driver.env.raw_observation()
    achievements = driver.env.achievements()
    game_stats = driver.env.game_stats()
    belief_snapshot = wm.belief.snapshot()
    stats: CraftaxStats = {
        "health": float(game_stats["health"]),
        "drink": float(game_stats["drink"]),
        "food": float(game_stats["food"]),
        "energy": float(game_stats["energy"]),
        "light": float(game_stats["light_level"]),
        "is_sleeping": float(game_stats["is_sleeping"]),
        "is_resting": float(game_stats["is_resting"]),
    }
    simulation_state = {
        "simulation_id": sim_id,
        "token_2d": [[int(value) for value in row] for row in observation["token_2d"]],
        "vector": [float(value) for value in observation["vector"].reshape(-1)],
        "direction": [int(value) for value in observation["token"].reshape(-1)],
        "raw_observation": [float(value) for value in raw_observation.reshape(-1)],
        "achievements_count": int(np.count_nonzero(achievements)),
        "stats": stats,
        "step": int(driver.step_count),
        "done": bool(driver.env.done),
        "reward": float(driver.env.last_reward),
        "active_dof": int(wm.belief.active_dof),
    }
    return {"simulation_state": simulation_state, "belief_state": belief_snapshot}


@app.post("/simulation/{sim_id}/reset")
def reset_simulation(sim_id: str) -> dict[str, bool]:
    driver = _env_registry.get(sim_id)
    if driver is None:
        raise HTTPException(status_code=404, detail=f"unknown sim_id '{sim_id}'")
    driver.reset()
    return {"ok": True}


_world_model_ref: list[WorldModelProvider] = []


def set_world_model(world_model: WorldModelProvider) -> None:
    """main.py registers the singleton here for debug endpoints."""
    _world_model_ref.clear()
    _world_model_ref.append(world_model)


@app.get("/debug/reset_belief")
async def debug_reset_belief() -> dict[str, Any]:
    # P-1 (RISKS.md): converged system shows no agent activity during demo.
    # Reset the error map to 0.5 everywhere -> orchestrator re-spawns within
    # one 2s cycle. Demo-prep only.
    wm = _world_model_ref[0] if _world_model_ref else None
    if wm is None:
        raise HTTPException(status_code=503, detail="world model not ready")
    wm.belief.prediction_error_map[:] = 0.5
    wm.belief.confidence_map[:] = 0.5
    wm.belief.regional_override = None
    return {"ok": True, "reset": "prediction_error_map=0.5"}


@app.get("/debug/trigger_agent")
async def debug_trigger_agent() -> dict[str, Any]:
    wm = _world_model_ref[0] if _world_model_ref else None
    if wm is None:
        raise HTTPException(status_code=503, detail="world model not ready")
    wm.belief.prediction_error_map[0:8, 0:8] = 0.8  # upper_left region
    wm.belief.confidence_map[0:8, 0:8] = 0.2
    wm.belief.regional_override = None
    return {"ok": True, "boosted": "upper_left=0.8"}


_env_registry: dict[str, CraftaxDriverProvider] = {}


def set_env_registry(envs: Mapping[str, CraftaxDriverProvider]) -> None:
    _env_registry.clear()
    _env_registry.update(envs)


@app.get("/sim/{sim_id}/stream")
async def sim_stream(sim_id: str) -> None:
    del sim_id
    raise HTTPException(
        status_code=404,
        detail="Craftax MJPEG stream is unavailable",
    )


@app.websocket("/ws/events")
async def ws_events(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    last_pong = time.monotonic()
    async with anyio.create_task_group() as task_group:
        task_group.start_soon(_ping_loop, websocket)
        try:
            while True:
                remaining = max(0.0, PONG_TIMEOUT_S - (time.monotonic() - last_pong))
                with anyio.fail_after(remaining):
                    message = await websocket.receive_text()
                if message == "pong":
                    last_pong = time.monotonic()
        except (TimeoutError, WebSocketDisconnect):
            return
        finally:
            task_group.cancel_scope.cancel()
            manager.disconnect(websocket)
