from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

import backend.agents.exploration as exploration
from backend.agents.exploration import AgentAPIError, ExplorationAgent
from backend.craftax_driver import DriverStep, PreActionPrediction
from backend.event_log import get_events_since, init_db, write_event
from backend.simulus.instrumentation import InstrumentedActionOutput


class FakeDriver:
    def __init__(self) -> None:
        self.env = SimpleNamespace(
            raw_observation=lambda: np.full(8, self.steps, dtype=np.float32),
            achievements=lambda: np.zeros(67, dtype=bool),
        )
        self.steps = 0
        self.actions: list[int] = []
        self.controller_steps = 0
        self.close_calls = 0

    def evaluate_stamped_candidates(self) -> list[PreActionPrediction]:
        return [
            PreActionPrediction(
                1,
                self.steps,
                InstrumentedActionOutput(
                    action_idx=index,
                    J_ua=float(index) / 100.0,
                    reward_expectation=float(index),
                ),
            )
            for index in range(43)
        ]

    def step_with_action(self, action: int) -> DriverStep:
        self.actions.append(action)
        self.steps += 1
        return DriverStep(self.steps, action, 1.0, False, 0, {"health": 1.0})

    def step_controller(self) -> tuple[DriverStep, np.ndarray]:
        self.controller_steps += 1
        return self.step_with_action(0), np.full(43, 1 / 43)

    def reset(self, seed: int | None = None) -> None:
        del seed
        self.steps = 0

    def close(self) -> None:
        self.close_calls += 1


class FakeWorldModel:
    def __init__(self) -> None:
        self.belief = SimpleNamespace(
            regional_override={"upper_left": 0.8},
            active_dof=1,
            prediction_error_map=np.full((16, 16), 0.8, dtype=np.float32),
        )
        self.calls: list[str] = []

    async def predict(self) -> str:
        self.calls.append("predict")
        return "prediction"

    async def update(self, prediction: str) -> str:
        assert prediction == "prediction"
        self.calls.append("update")
        return "transition"

    async def ingest(self, transition: str) -> None:
        assert transition == "transition"
        self.calls.append("ingest")
        self.belief.regional_override["upper_left"] = 0.6

    async def fine_tune(self, *_args, **_kwargs) -> None:
        raise AssertionError("read-only world model must never train")


@pytest.fixture(autouse=True)
def event_db(tmp_path) -> None:
    init_db(db_path=tmp_path / "events.db")


async def test_episode_uses_claude_rank_only_on_advisory_steps(monkeypatch) -> None:
    # Given
    driver = FakeDriver()
    model = FakeWorldModel()
    agent = ExplorationAgent("exp_ranked", model, driver, "upper_left", 0.8)
    agent._client = SimpleNamespace()

    async def select_second(*_args, **_kwargs) -> tuple[dict[str, int | str], int]:
        return {"candidate_rank": 2, "reasoning": "rank two"}, 7

    monkeypatch.setattr(exploration, "_call_claude", select_second)

    # When
    samples, tokens = await agent.run_episode(n_steps=6)

    # Then
    assert [sample[1] for sample in samples] == [41, 0, 0, 0, 0, 41]
    assert all(np.array_equal(sample[2], sample[0] + 1) for sample in samples)
    assert driver.controller_steps == 4
    assert tokens == 14


async def test_api_failure_falls_back_to_real_controller(monkeypatch) -> None:
    # Given
    driver = FakeDriver()
    agent = ExplorationAgent(
        "exp_fallback", FakeWorldModel(), driver, "upper_left", 0.8
    )
    agent._client = SimpleNamespace()

    async def fail(*_args, **_kwargs) -> tuple[dict[str, int | str], int]:
        raise AgentAPIError("offline")

    monkeypatch.setattr(exploration, "_call_claude", fail)

    # When
    samples, tokens = await agent.run_episode(n_steps=1)

    # Then
    assert samples[0][1] == 0
    assert driver.controller_steps == 1
    assert tokens == 0


async def test_completion_follows_ingest_and_never_trains() -> None:
    # Given
    write_event("agent_spawned", agent_id="exp_ingest", region="upper_left")
    driver = FakeDriver()
    model = FakeWorldModel()
    agent = ExplorationAgent("exp_ingest", model, driver, "upper_left", 0.8)
    agent._client = None

    # When
    output = await agent.run(n_steps=2)

    # Then
    assert model.calls == ["predict", "update", "ingest"]
    assert output["error_before"] == pytest.approx(0.8)
    assert output["error_after"] == pytest.approx(0.6)
    assert output["samples_collected"] == 2
    assert output["fine_tune_loss"] == 0.0
    assert driver.close_calls == 1
    events = get_events_since(0)
    types = [event["event_type"] for event in events]
    assert types.index("agent_spawned") < types.index("agent_completed")
    assert "world_model_updated" not in types


async def test_timeout_releases_driver(monkeypatch) -> None:
    # Given
    driver = FakeDriver()
    agent = ExplorationAgent(
        "exp_timeout", FakeWorldModel(), driver, "upper_left", 0.8
    )

    async def stall(*_args, **_kwargs) -> tuple[list, int]:
        await asyncio.sleep(1)
        return [], 0

    monkeypatch.setattr(agent, "run_episode", stall)
    monkeypatch.setattr(exploration, "AGENT_TIMEOUT_SECONDS", 0.01)

    # When
    output = await agent.run()

    # Then
    assert output["success"] is False
    assert driver.close_calls == 1
    assert any(event["event_type"] == "agent_failed" for event in get_events_since(0))


def test_tool_schema_accepts_candidate_rank_only() -> None:
    schema = exploration.EXPLORE_TOOL["input_schema"]
    assert schema["required"] == ["candidate_rank", "reasoning"]
    assert schema["properties"]["candidate_rank"]["maximum"] == 5
