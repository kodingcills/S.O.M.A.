from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from backend.agents.capability import CapabilityAgent
from backend.agents.reactive import ReactiveAgent
from backend.craftax_driver import DriverStep
from backend.event_log import get_events_since, init_db, write_event


class FakeDriver:
    def __init__(self, achievements: int = 4) -> None:
        unlocked = np.zeros(67, dtype=bool)
        unlocked[:achievements] = True
        self.observation = {
            "token_2d": np.zeros((99, 4), dtype=np.int32),
            "vector": np.zeros(47, dtype=np.float32),
            "token": np.zeros(1, dtype=np.int32),
        }
        self.env = SimpleNamespace(
            achievements=lambda: unlocked.copy(),
            game_stats=lambda: {
                "health": 1.0,
                "drink": 1.0,
                "food": 1.0,
                "energy": 1.0,
            },
            observation=lambda: self.observation,
        )
        self.actions: list[int] = []
        self.seed = 42

    def current_obs_tokens(self) -> dict[str, np.ndarray]:
        return self.observation

    def step_with_action(self, action: int) -> DriverStep:
        self.actions.append(action)
        return DriverStep(1, action, 2.5, False, 4, self.env.game_stats())

    def reset(self, seed: int | None = None) -> None:
        assert seed in (None, 42)


@pytest.fixture(autouse=True)
def event_db(tmp_path) -> None:
    init_db(db_path=tmp_path / "events.db")


async def test_capability_unlocks_from_real_achievement_bucket() -> None:
    # Given
    driver = FakeDriver(achievements=4)
    belief = SimpleNamespace(
        active_dof=1,
        regional_override={"upper_left": 0.2, "upper_right": 0.4},
        prediction_error_map=np.zeros((16, 16), dtype=np.float32),
    )
    write_event("agent_spawned", agent_id="cap_2")
    agent = CapabilityAgent("cap_2", SimpleNamespace(belief=belief), driver)

    # When
    output = await agent.run()

    # Then
    assert belief.active_dof == 2
    assert output == {
        "agent_id": "cap_2",
        "target_dof": 2,
        "samples_collected": 0,
        "fine_tune_loss": 0.0,
        "success": True,
    }
    events = get_events_since(0)
    types = [event["event_type"] for event in events]
    assert types.index("agent_spawned") < types.index("capability_unlocked")
    assert types.index("capability_unlocked") < types.index("agent_completed")
    unlocked = next(event for event in events if event["event_type"] == "capability_unlocked")
    assert unlocked["payload"]["trigger_error"] == pytest.approx(0.3)


async def test_capability_completes_when_orchestrator_already_applied_unlock() -> None:
    driver = FakeDriver(achievements=4)
    belief = SimpleNamespace(
        active_dof=2,
        regional_override={"upper_left": 0.2},
        prediction_error_map=np.zeros((16, 16), dtype=np.float32),
    )

    output = await CapabilityAgent(
        "cap_2", SimpleNamespace(belief=belief), driver
    ).run()

    assert output["success"] is True
    assert output["target_dof"] == 2


def _observation(block: int, row: int, col: int) -> dict[str, np.ndarray]:
    token_2d = np.zeros((99, 4), dtype=np.int32)
    token_2d[row * 11 + col, 0] = block
    vector = np.zeros(47, dtype=np.float32)
    return {"token_2d": token_2d, "vector": vector, "token": np.zeros(1)}


def test_reactive_contract_priorities() -> None:
    agent = ReactiveAgent(FakeDriver())
    base = {"health": 1.0, "drink": 1.0, "food": 1.0, "energy": 1.0}

    assert agent._select_action(_observation(0, 0, 0), {**base, "energy": 0.1}) == 6
    assert agent._select_action(_observation(3, 4, 6), {**base, "drink": 0.1}) == 5
    assert agent._select_action(_observation(16, 4, 7), {**base, "food": 0.1}) == 2

    craft = _observation(11, 4, 6)
    craft["vector"][0] = 2.0
    assert agent._select_action(craft, base) == 11
    assert agent._select_action(_observation(5, 4, 2), base) == 1
    assert agent._select_action(_observation(0, 0, 0), base) == 0


async def test_reactive_step_writes_real_comparison_event() -> None:
    driver = FakeDriver()
    agent = ReactiveAgent(driver)

    await agent.run_step()

    event = next(
        event for event in get_events_since(0)
        if event["event_type"] == "simulation_step"
    )
    assert event["sim_id"] == "comparison"
    assert event["payload"]["action"] == driver.actions[0]
    assert event["payload"]["stats"]["health"] == 1.0
    assert event["payload"]["n_achievements"] == 4


def test_reactive_isolation() -> None:
    source = Path("backend/agents/reactive.py").read_text(encoding="utf-8")
    forbidden = ("world_model", "prediction_net", "belief_state", "anthropic", "orchestrator")
    assert all(name not in source for name in forbidden)


async def test_reactive_run_yields(monkeypatch) -> None:
    driver = FakeDriver()
    yielded = asyncio.Event()

    async def stop_after_yield(_delay: float) -> None:
        yielded.set()
        raise asyncio.CancelledError

    monkeypatch.setattr("backend.agents.reactive.asyncio.sleep", stop_after_yield)
    with pytest.raises(asyncio.CancelledError):
        await ReactiveAgent(driver).run()
    assert yielded.is_set()
