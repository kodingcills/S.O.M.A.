"""SOMA backend entrypoint — Task 1.9 full system wiring.

`uvicorn backend.main:app` (repo root) or `uvicorn main:app` (cwd=backend)
boots the ENTIRE system per ARCHITECTURE.md "STARTUP SEQUENCE" (order is
load-bearing, steps 1-9):

  1. init_db + broadcast registration   (idempotent — dedupes api.py's own)
  2. primary_sim + comparison_sim       (seed=42 pair, SEED REPRODUCIBILITY)
  3. background data collection         (if training.db < 50k samples)
  4. initial training                   (if model checkpoint missing; blocking
                                        under to_thread so WS stays alive)
  5. WorldModel singleton               (THE only instantiation site)
  6. MPCAgent + ReactiveAgent
  7. LangGraph orchestrator loop        (bounded bursts, GraphRecursionError
                                        → continue; see orchestrator.py docstring)
  8. STATE bridge task                  (api.STATE mutation every 2s)
  9. belief_snapshot_loop + system_ready

Composition strategy: this module builds a FRESH FastAPI shell that MOUNTS
backend.api.app at "/". The shared api.app instance keeps its minimal
lifespan untouched — tests/test_api.py boots it directly and must never
trigger real sim startup. A mounted sub-app's lifespan does not run, so
this module's lifespan owns init_db + set_broadcast_callback itself (both
idempotent). All routes (/health, /events, /ws/events, ...) are preserved
through the mount.

DEVIATIONS from ARCHITECTURE.md literal text (for MASTER_STATE.md):
  1. Background collection uses a DEDICATED SimConfig(seed=0) env instead
     of primary_sim: the spec's non-blocking requirement means MPC steps
     primary concurrently — two writers on one PyBullet env is a race.
     Seed 0 matches the data-collection convention (QUICK_REFERENCE.md).
  2. training_started/training_completed are ALSO written by train.py
     internally on its success path; main.py writes them around the call
     anyway per spec (duplicate SYSTEM events are canvas-inert), except
     training_completed is written only when training actually produced a
     summary — claiming completion after a skip would be false telemetry.
  3. Orchestrator receives a SimpleNamespace(primary=..., comparison=...)
     shim as sim_manager — unlock_dof_node duck-types .primary/.comparison.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI

try:
    from backend.agents.reactive import ReactiveAgent
    from backend.api import (
        STATE,
        app as api_app,
        manager,
        set_env_registry,
        set_world_model,
    )
    from backend.data_collector import collect_training_data
    from backend.event_log import init_db, set_broadcast_callback, write_event
    from backend.mpc_agent import MPCAgent
    from backend.orchestrator import build_soma_graph, initial_state
    from backend.simulation import SimConfig, SurROLTissueEnv, create_env
    from backend.train import train_prediction_network
    from backend.world_model import WorldModel
except ImportError:  # cwd=backend direct-run mode (uvicorn main:app)
    from agents.reactive import ReactiveAgent  # type: ignore[no-redef]
    from api import (  # type: ignore[no-redef]
        STATE,
        app as api_app,
        manager,
        set_env_registry,
        set_world_model,
    )
    from data_collector import collect_training_data  # type: ignore[no-redef]
    from event_log import init_db, set_broadcast_callback, write_event  # type: ignore[no-redef]
    from mpc_agent import MPCAgent  # type: ignore[no-redef]
    from orchestrator import build_soma_graph, initial_state  # type: ignore[no-redef]
    from simulation import SimConfig, SurROLTissueEnv, create_env  # type: ignore[no-redef]
    from train import train_prediction_network  # type: ignore[no-redef]
    from world_model import WorldModel  # type: ignore[no-redef]

# CWD-proof anchors (same philosophy as WorldModel's default model_path).
_BACKEND_DIR = Path(__file__).resolve().parent
TRAINING_DB_PATH = _BACKEND_DIR / "data" / "training.db"
MODEL_PATH = _BACKEND_DIR / "models" / "prediction_network.pt"

# Constants (ARCHITECTURE.md startup sequence + QUICK_REFERENCE.md).
TRAINING_SAMPLES_TARGET = 50_000
COLLECT_BATCH_SAMPLES = 50_000
STATE_BRIDGE_INTERVAL_S = 2.0
BELIEF_SNAPSHOT_INTERVAL_S = 10.0
ORCH_RECURSION_LIMIT = 100
ORCH_THREAD_ID = "soma"

_ORCH_CONFIG = {"recursion_limit": ORCH_RECURSION_LIMIT,
                "configurable": {"thread_id": ORCH_THREAD_ID}}


def _count_training_samples() -> int:
    """Rows in training.db; 0 on any error (missing db / schema / lock)."""
    try:
        conn = sqlite3.connect(TRAINING_DB_PATH)
        try:
            (n,) = conn.execute("SELECT COUNT(*) FROM samples").fetchone()
        finally:
            conn.close()
        return int(n)
    except Exception:
        return 0


async def _collection_worker() -> None:
    """Background Task-1.3 collector on a dedicated seed-0 env (deviation 1)."""
    try:
        env = await create_env(SimConfig(seed=0))
    except Exception as exc:
        print(f"[main] collection env failed: {exc}", flush=True)
        return
    try:
        n = await asyncio.to_thread(
            collect_training_data, env, COLLECT_BATCH_SAMPLES
        )
        print(f"[main] background collection wrote {n} samples", flush=True)
    except Exception as exc:
        print(f"[main] background collection failed: {exc}", flush=True)
    finally:
        with contextlib.suppress(Exception):
            await asyncio.to_thread(env.close)


async def _state_bridge_task(world_model: WorldModel, mpc: MPCAgent,
                             graph: Any, ready: dict[str, bool]) -> None:
    """Step 8: mirror live system state into api.STATE every 2s."""
    while True:
        belief = world_model.belief
        active_agents = 0
        with contextlib.suppress(Exception):
            snapshot = await graph.aget_state(_ORCH_CONFIG)
            active_agents = len(snapshot.values.get("active_agents", []))
        STATE.update({
            "status": "ok" if ready.get("system") else "starting",
            "world_model_version": int(belief.world_model_version),
            "global_error": float(belief.prediction_error_map.mean()),
            "active_agents": active_agents,
            "primary_sim_step": getattr(mpc, "_step_count", 0),
            "training_progress": None,
        })
        await asyncio.sleep(STATE_BRIDGE_INTERVAL_S)


async def _belief_snapshot_task(world_model: WorldModel) -> None:
    """Step 9: write belief_snapshot immediately, then every 10s."""
    while True:
        belief = world_model.belief
        write_event(
            "belief_snapshot",
            global_mean_error=float(belief.prediction_error_map.mean()),
            regional_errors=dict(belief.get_regional_errors()),
            world_model_version=int(belief.world_model_version),
            error_map=belief.prediction_error_map.tolist(),
        )
        await asyncio.sleep(BELIEF_SNAPSHOT_INTERVAL_S)


async def _orchestrator_loop(graph: Any) -> None:
    """Step 7: drive the END-less graph in bounded bursts (orchestrator.py
    docstring pattern). collect_status sleeps one cycle per pass, so each
    burst always makes progress before hitting the recursion limit."""
    while True:
        try:
            await graph.ainvoke(initial_state(), config=_ORCH_CONFIG)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # GraphRecursionError + node boundary guards
            print(f"[main] orchestrator burst ended: {exc}", flush=True)
            await asyncio.sleep(0.1)


@contextlib.asynccontextmanager
async def _soma_lifespan(_: FastAPI):
    # ── Step 1: databases + broadcast (idempotent dedupe of api.py) ────────
    init_db()
    set_broadcast_callback(manager.broadcast)

    tasks: list[asyncio.Task[Any]] = []
    sims: list[SurROLTissueEnv] = []
    ready = {"system": False}
    try:
        # ── Step 2: primary + comparison simulations (seed=42 pair) ────────
        primary_sim = await create_env(SimConfig(seed=42, dof=1))
        comparison_sim = await create_env(SimConfig(seed=42, dof=1))
        sims += [primary_sim, comparison_sim]

        # ── Step 3: conditional background data collection ─────────────────
        if _count_training_samples() < TRAINING_SAMPLES_TARGET:
            write_event("training_started", reason="data_collection")
            tasks.append(asyncio.create_task(_collection_worker()))

        # ── Step 4: conditional initial training (blocking, in to_thread) ──
        # SOMA_SKIP_INITIAL_TRAIN=1 boots on untrained weights (fresh-system
        # state) — used to validate the self-improvement loop end-to-end.
        if not MODEL_PATH.exists() and not os.environ.get("SOMA_SKIP_INITIAL_TRAIN"):
            STATE["status"] = "training"
            write_event("training_started", reason="initial_training")
            summary = await asyncio.to_thread(
                train_prediction_network,
                str(TRAINING_DB_PATH), str(MODEL_PATH.parent),
            )
            if summary is not None:
                write_event("training_completed", payload_summary=bool(summary))
            else:
                print("[main] initial training skipped — no usable "
                      "training.db; running on untrained weights", flush=True)

        # ── Step 5: THE WorldModel singleton ───────────────────────────────
        world_model = WorldModel()
        set_world_model(world_model)
        set_env_registry({
            "primary": primary_sim,
            "comparison": comparison_sim,
        })

        # ── Step 6: MPC + Reactive agents ──────────────────────────────────
        mpc = MPCAgent(world_model, primary_sim)
        reactive = ReactiveAgent(comparison_sim)

        # ── Step 7: LangGraph orchestrator loop ────────────────────────────
        graph = build_soma_graph(
            world_model,
            SimpleNamespace(primary=primary_sim, comparison=comparison_sim),
        )

        # ── Step 9 (tasks): everything runs until shutdown cancels it ──────
        tasks += [
            asyncio.create_task(mpc.run()),
            asyncio.create_task(reactive.run()),
            asyncio.create_task(_orchestrator_loop(graph)),
            asyncio.create_task(
                _state_bridge_task(world_model, mpc, graph, ready)
            ),
            asyncio.create_task(_belief_snapshot_task(world_model)),
        ]

        ready["system"] = True
        try:
            import wandb

            if wandb.run is None:
                wandb.init(
                    project="soma-surgical",
                    name=f"demo-{int(time.time())}",
                    mode=os.environ.get("WANDB_MODE", "offline"),
                )
            print(f"[main] W&B dashboard: {wandb.run.url}", flush=True)
        except Exception as exc:
            print(f"[main] wandb unavailable ({exc}) — continuing without", flush=True)
        write_event("system_ready")
        print("[main] SOMA system ready "
              f"(wandb optional — {'absent' if _wandb_missing() else 'available'})",
              flush=True)
        yield
    finally:
        ready["system"] = False
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for env in sims:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(env.close)
        set_broadcast_callback(None)
        STATE.clear()
        STATE.update({
            "status": "starting", "world_model_version": 0,
            "global_error": 0.0, "active_agents": 0,
            "primary_sim_step": 0, "training_progress": None,
        })


def _wandb_missing() -> bool:
    with contextlib.suppress(Exception):
        import wandb  # noqa: F401 — presence probe only
        return False
    return True


# Fresh shell app mounting the untouched api.app — see module docstring.
app = FastAPI(title="SOMA", lifespan=_soma_lifespan,
              docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/", api_app)

__all__ = ["app"]
