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

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.websockets import WebSocket, WebSocketDisconnect

try:
    from backend.event_log import (
        get_events_for_agent,
        get_events_since,
        init_db,
        set_broadcast_callback,
    )
except ImportError:  # cwd=backend direct-run mode (uvicorn api:app)
    from event_log import (  # type: ignore[no-redef]
        get_events_for_agent,
        get_events_since,
        init_db,
        set_broadcast_callback,
    )

PING_INTERVAL_S = 5.0
PONG_TIMEOUT_S = 15.0

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
        self._send_locks: dict[WebSocket, asyncio.Lock] = {}

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
        lock = self._send_locks.setdefault(websocket, asyncio.Lock())
        async with lock:
            await websocket.send_json(message)

    async def broadcast(self, message: dict[str, Any]) -> None:
        for websocket in list(self.active_connections):
            try:
                await self._send(websocket, message)
            except Exception:
                self.disconnect(websocket)


manager = ConnectionManager()


async def _ping_loop(websocket: WebSocket) -> None:
    try:
        while True:
            await asyncio.sleep(PING_INTERVAL_S)
            await manager._send(websocket, {"type": "ping"})
    except Exception:
        return  # socket died; endpoint's finally handles cleanup


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    set_broadcast_callback(manager.broadcast)
    yield
    set_broadcast_callback(None)


app = FastAPI(title="SOMA", lifespan=lifespan)


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
async def detail(sim_id: str) -> dict[str, Any]:
    # Placeholder until SimulationState + BeliefState exist (Task 1.9+).
    # Real impl reads both back-to-back with NO await between (atomicity).
    return {"simulation_state": None, "belief_state": None}


@app.post("/simulation/{sim_id}/reset")
async def reset_simulation(sim_id: str) -> dict[str, Any]:
    return {"ok": True}


_world_model_ref: list[Any] = []


def set_world_model(world_model: Any) -> None:
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
    return {"ok": True, "reset": "prediction_error_map=0.5"}


@app.get("/debug/trigger_agent")
async def debug_trigger_agent() -> dict[str, Any]:
    wm = _world_model_ref[0] if _world_model_ref else None
    if wm is None:
        raise HTTPException(status_code=503, detail="world model not ready")
    wm.belief.prediction_error_map[0:8, 0:8] = 0.8  # upper_left region
    return {"ok": True, "boosted": "upper_left=0.8"}


_env_registry: dict[str, Any] = {}


def set_env_registry(envs: dict[str, Any]) -> None:
    """main.py registers live sims here so MJPEG can find them by id."""
    _env_registry.clear()
    _env_registry.update(envs)


async def pybullet_frame_generator(env: Any, max_frames: int = 0):
    # max_frames=0 -> unlimited (production); N>0 -> bounded (tests).
    # RISKS S-4: p.getCameraImage blocks 20-100ms — must run in to_thread and
    # yield between frames or WS heartbeats die during streaming.
    import io

    import numpy as np
    import pybullet as p
    from PIL import Image

    cid = getattr(env, "cid", 0)
    view = getattr(env, "_view_matrix", None)
    proj = getattr(env, "_proj_matrix", None)

    frames_sent = 0
    while max_frames == 0 or frames_sent < max_frames:
        width, height, rgba, _depth, _seg = await asyncio.to_thread(
            p.getCameraImage,
            320,
            240,
            viewMatrix=view,
            projectionMatrix=proj,
            physicsClientId=cid,
        )
        frame = np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)[..., :3]
        buf = io.BytesIO()
        Image.fromarray(frame).save(buf, format="JPEG", quality=85)
        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
            + buf.getvalue()
            + b"\r\n"
        )
        frames_sent += 1
        await asyncio.sleep(1 / 15)


@app.get("/sim/{sim_id}/stream")
async def sim_stream(sim_id: str):
    env = _env_registry.get(sim_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"unknown sim_id '{sim_id}'")
    return StreamingResponse(
        pybullet_frame_generator(env),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.websocket("/ws/events")
async def ws_events(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    ping_task = asyncio.create_task(_ping_loop(websocket))
    last_pong = time.monotonic()
    try:
        while True:
            try:
                message = await asyncio.wait_for(
                    websocket.receive_text(), timeout=PONG_TIMEOUT_S
                )
                if message == "pong":
                    last_pong = time.monotonic()
            except asyncio.TimeoutError:
                if time.monotonic() - last_pong >= PONG_TIMEOUT_S:
                    break  # no pong within 15s — drop client
    except WebSocketDisconnect:
        pass
    finally:
        ping_task.cancel()
        manager.disconnect(websocket)
