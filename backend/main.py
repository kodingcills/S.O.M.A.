"""SOMA application composition for the released Craftax/Simulus substrate."""

from __future__ import annotations

import asyncio
import contextlib
from types import SimpleNamespace

from fastapi import FastAPI

from backend.api import (
    STATE,
    app as api_app,
    manager,
    set_env_registry,
    set_world_model,
)
from backend.craftax_driver import CraftaxDriver
from backend.event_log import init_db, set_broadcast_callback, write_event
from backend.orchestrator import build_soma_graph
from backend.rerun_logger import init as init_rerun
from backend.simulus.runtime import load_simulus_agent
from backend.soma.craftax_runner import run_craftax_loop
from backend.soma.state_machine import SomaStateMachine
from backend.world_model import WorldModel


CRAFTAX_SEED = 42
BELIEF_SNAPSHOT_INTERVAL_S = 10.0


async def _belief_snapshot_task(world_model: WorldModel) -> None:
    while True:
        belief = world_model.belief
        write_event(
            "belief_snapshot",
            global_mean_error=float(belief.prediction_error_map.mean()),
            regional_errors=dict(belief.get_regional_errors()),
            world_model_version=int(belief.world_model_version),
            active_dof=int(belief.active_dof),
            error_map=belief.prediction_error_map.tolist(),
        )
        await asyncio.sleep(BELIEF_SNAPSHOT_INTERVAL_S)


@contextlib.asynccontextmanager
async def _soma_lifespan(_: FastAPI):
    init_db()
    set_broadcast_callback(manager.broadcast)
    init_rerun()

    runner_task: asyncio.Task[None] | None = None
    belief_task: asyncio.Task[None] | None = None
    try:
        agent, _config, _device = await asyncio.to_thread(load_simulus_agent)
        primary_driver = await asyncio.to_thread(CraftaxDriver, agent, CRAFTAX_SEED)
        comparison_driver = await asyncio.to_thread(
            CraftaxDriver, agent, CRAFTAX_SEED
        )

        # WORLD MODEL SINGLETON: this is the process's only construction site.
        world_model = WorldModel(primary_driver)
        drivers = SimpleNamespace(
            primary=primary_driver,
            comparison=comparison_driver,
        )
        set_world_model(world_model)
        set_env_registry(
            {
                "primary": primary_driver,
                "comparison": comparison_driver,
            }
        )
        build_soma_graph(world_model, drivers)

        belief = world_model.belief
        STATE.update(
            {
                "status": "ok",
                "world_model_version": int(belief.world_model_version),
                "global_error": float(belief.prediction_error_map.mean()),
                "active_agents": 0,
                "primary_sim_step": int(primary_driver.step_count),
                "training_progress": None,
            }
        )
        belief_task = asyncio.create_task(_belief_snapshot_task(world_model))
        write_event("system_ready")
        # CRAFTAX RUNNER: the only place the continuous episode loop starts.
        runner_task = asyncio.create_task(
            run_craftax_loop(
                primary_driver,
                primary_driver.instrumented,
                SomaStateMachine(),
            )
        )
        yield
    finally:
        for task in (runner_task, belief_task):
            if task is not None:
                task.cancel()
        await asyncio.gather(runner_task, belief_task, return_exceptions=True)
        set_broadcast_callback(None)
        STATE.update(
            {
                "status": "starting",
                "world_model_version": 0,
                "global_error": 0.0,
                "active_agents": 0,
                "primary_sim_step": 0,
                "training_progress": None,
            }
        )


app = FastAPI(
    title="SOMA",
    lifespan=_soma_lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
app.mount("/", api_app)

__all__ = ["app"]
