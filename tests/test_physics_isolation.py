import asyncio  # noqa: ANYIO_OK

import numpy as np
import pytest

import backend.simulation as simulation
from backend.simulation import SimConfig, create_env


@pytest.mark.asyncio(scope="module")
async def test_four_concurrent_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(simulation, "_pybullet_init_lock", asyncio.Lock())
    envs = await asyncio.gather(
        *(create_env(SimConfig(seed=seed, dof=1)) for seed in range(42, 46))
    )
    try:
        for env in envs:
            state = env.get_state_vector()
            assert state.shape == (806,)
            assert state.dtype == np.float32
            assert np.isfinite(state).all()
    finally:
        await asyncio.gather(
            *(asyncio.to_thread(env.close) for env in envs)
        )


@pytest.mark.asyncio(scope="module")
async def test_close_does_not_break_sibling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(simulation, "_pybullet_init_lock", asyncio.Lock())
    env_a, env_b = await asyncio.gather(
        create_env(SimConfig(seed=42, dof=1)),
        create_env(SimConfig(seed=43, dof=1)),
    )
    try:
        await asyncio.to_thread(env_a.close)
        state = env_b.get_state_vector()
        assert state.shape == (806,)
        assert state.dtype == np.float32
        assert np.isfinite(state).all()
    finally:
        await asyncio.to_thread(env_b.close)
