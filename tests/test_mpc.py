from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

import backend.mpc_agent as mpc_module
from backend.craftax_driver import DriverStep, PreActionPrediction
from backend.event_log import get_events_since, init_db
from backend.mpc_agent import MPCAgent
from backend.simulus.instrumentation import InstrumentedActionOutput


class FakeDriver:
    def __init__(self) -> None:
        achievements = np.zeros(67, dtype=bool)
        achievements[:4] = True
        self.env = SimpleNamespace(achievements=lambda: achievements.copy())
        self.evaluate_calls = 0
        self.actions: list[int] = []
        self.reset_calls = 0

    def evaluate_stamped_candidates(self) -> list[PreActionPrediction]:
        self.evaluate_calls += 1
        candidates: list[PreActionPrediction] = []
        for index in range(43):
            reward = 0.0
            jsd = 0.0
            if index == 7:
                reward, jsd = 1.0, float(np.log(2.0))
            if index == 8:
                reward, jsd = 0.75, 0.0
            candidates.append(
                PreActionPrediction(
                    1,
                    0,
                    InstrumentedActionOutput(
                        action_idx=index,
                        J_ua=jsd,
                        reward_expectation=reward,
                    ),
                )
            )
        return candidates

    def step_with_action(self, action: int) -> DriverStep:
        self.actions.append(action)
        return DriverStep(
            step=1,
            action=action,
            reward=3.0,
            done=False,
            n_achievements=4,
            stats={"health": 0.9, "energy": 0.8},
        )

    def reset(self, seed: int | None = None) -> None:
        del seed
        self.reset_calls += 1


@pytest.fixture(autouse=True)
def event_db(tmp_path) -> None:
    init_db(db_path=tmp_path / "events.db")


def test_score_is_reward_minus_half_normalized_jsd() -> None:
    output = InstrumentedActionOutput(
        action_idx=1,
        J_ua=float(np.log(2.0)),
        reward_expectation=2.0,
    )
    assert MPCAgent._score(output) == pytest.approx(1.5)


async def test_cycle_batches_once_and_steps_argmax() -> None:
    # Given
    driver = FakeDriver()
    agent = MPCAgent(SimpleNamespace(), driver)

    # When
    await agent.run_cycle()

    # Then
    assert driver.evaluate_calls == 1
    assert driver.actions == [8]
    event = next(
        event for event in get_events_since(0)
        if event["event_type"] == "simulation_step"
    )
    assert event["sim_id"] == "primary"
    assert event["payload"]["reward"] == 3.0
    assert event["payload"]["stats"] == {"health": 0.9, "energy": 0.8}
    assert event["payload"]["n_achievements"] == 4


async def test_cycle_emits_optional_rerun_telemetry(monkeypatch) -> None:
    driver = FakeDriver()
    calls: list[tuple] = []
    monkeypatch.setattr(
        mpc_module.rerun_logger,
        "log_craftax_step",
        lambda *args: calls.append(args),
        raising=False,
    )

    await MPCAgent(SimpleNamespace(), driver).run_cycle()

    assert len(calls) == 1
    assert calls[0][0] == "primary"


async def test_run_sleeps_point_zero_five(monkeypatch) -> None:
    driver = FakeDriver()
    delays: list[float] = []

    async def stop(delay: float) -> None:
        delays.append(delay)
        raise asyncio.CancelledError

    monkeypatch.setattr(mpc_module.asyncio, "sleep", stop)
    with pytest.raises(asyncio.CancelledError):
        await MPCAgent(SimpleNamespace(), driver).run()
    assert delays == [0.05]
