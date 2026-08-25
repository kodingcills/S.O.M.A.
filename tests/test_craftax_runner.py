from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

import backend.soma.craftax_runner as runner
from backend.craftax_driver import CraftaxDriver, DriverStep, PreActionPrediction
from backend.mpc_agent import MPCAgent
from backend.simulus.instrumentation import InstrumentedActionOutput, SimulusInstrumented
from backend.soma.state_machine import SomaStateMachine


class LoopFinished(Exception):
    pass

class FakeInstrumented:
    pass

class FakeDriver:
    def __init__(
        self,
        scripted: list[tuple[list[PreActionPrediction], DriverStep]],
        cancel_on_evaluate: bool = False,
    ) -> None:
        self._scripted = list(scripted)
        self._current: tuple[list[PreActionPrediction], DriverStep] | None = None
        self._generation = 0
        self._cancel_on_evaluate = cancel_on_evaluate
        self.action_calls: list[int] = []
        self.reset_seeds: list[int | None] = []
        achievements = np.array([True, False, True, True], dtype=bool)
        self.env = SimpleNamespace(achievements=lambda: achievements.copy())

    @property
    def episode_generation(self) -> int:
        return self._generation

    def evaluate_stamped_candidates(self) -> list[PreActionPrediction]:
        if self._cancel_on_evaluate:
            raise asyncio.CancelledError
        if not self._scripted:
            raise LoopFinished
        self._current = self._scripted.pop(0)
        return self._current[0]

    def step_with_action(self, action: int) -> DriverStep:
        assert self._current is not None
        candidates, step = self._current
        expected = max(candidates, key=lambda item: MPCAgent._score(item.output))
        assert action == int(expected.output.action_idx)
        self.action_calls.append(action)
        self._current = None
        return step

    def reset(self, seed: int | None = None) -> None:
        self.reset_seeds.append(seed)
        self._generation += 1


class RecordingStateMachine:
    def __init__(self) -> None:
        self.inner = SomaStateMachine()
        self.records: list[tuple[float, float]] = []
        self.transitions = 0

    @property
    def n_z_u(self) -> int:
        return self.inner.n_z_u

    @property
    def n_harmful(self) -> int:
        return self.inner.n_harmful

    def record(self, predicted: float, reward: float) -> bool:
        self.records.append((predicted, reward))
        return self.inner.record(predicted, reward)

    def should_transition_to_gap(self) -> bool:
        return self.inner.should_transition_to_gap()

    def advance_phase(self):
        self.transitions += 1
        return self.inner.advance_phase()


@pytest.fixture
def written_events(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, str | float | int | bool]]]:
    written: list[tuple[str, dict[str, str | float | int | bool]]] = []

    def record_event(
        event_type: str, **kwargs: str | float | int | bool
    ) -> int:
        written.append((event_type, kwargs))
        return len(written)

    monkeypatch.setattr(runner, "write_event", record_event)
    return written


def _candidates(step: int, jsd_base: float, generation: int = 1) -> list[PreActionPrediction]:
    candidates: list[PreActionPrediction] = []
    for action_idx in range(43):
        reward = 2.0 if action_idx == 17 else 0.0
        output = InstrumentedActionOutput(
            action_idx=action_idx,
            J_ua=jsd_base + action_idx / 1000.0,
            reward_expectation=reward,
        )
        candidates.append(PreActionPrediction(generation, step - 1, output))
    return candidates


def _script(
    steps: int,
    *,
    reward: float = 0.0,
    done_at: int | None = None,
    jsd_start: float = 0.0,
) -> list[tuple[list[PreActionPrediction], DriverStep]]:
    return [
        (
            _candidates(step, jsd_start + step / 100.0),
            DriverStep(step, 17, reward, step == done_at, 0, {}),
        )
        for step in range(1, steps + 1)
    ]


async def _run_until_script_finishes(
    driver: FakeDriver,
    state_machine: SomaStateMachine | RecordingStateMachine,
    n_audit_every: int = 10,
) -> None:
    with pytest.raises(LoopFinished):
        await runner.run_craftax_loop(
            driver, FakeInstrumented(), state_machine, n_audit_every
        )


def test_signature_matches_contract() -> None:
    async def expected(
        driver: CraftaxDriver,
        instrumented: SimulusInstrumented,
        state_machine: SomaStateMachine,
        n_audit_every: int = 10,
    ) -> None:
        del driver, instrumented, state_machine, n_audit_every

    assert inspect.signature(runner.run_craftax_loop) == inspect.signature(expected)


async def test_writes_craftax_step_each_iteration_with_required_payload_keys(
    written_events,
) -> None:
    driver = FakeDriver(_script(2, reward=1.25))

    await _run_until_script_finishes(driver, SomaStateMachine())

    step_events = [payload for event_type, payload in written_events if event_type == "craftax_step"]
    assert len(step_events) == 2
    assert set(step_events[0]) == {
        "episode", "step", "jsd_mean", "score", "action", "reward", "done"
    }


async def test_jsd_mean_is_mean_of_candidate_J_ua(written_events) -> None:
    candidates = _candidates(1, 0.2)
    driver = FakeDriver([(candidates, DriverStep(1, 17, 0.0, False, 0, {}))])

    await _run_until_script_finishes(driver, SomaStateMachine())

    payload = next(payload for kind, payload in written_events if kind == "craftax_step")
    assert payload["jsd_mean"] == pytest.approx(
        np.mean([candidate.output.J_ua for candidate in candidates])
    )


async def test_executes_exactly_one_env_step_per_iteration() -> None:
    driver = FakeDriver(_script(3))

    await _run_until_script_finishes(driver, SomaStateMachine())

    assert len(driver.action_calls) == 3


async def test_audits_decision_point_every_n_steps() -> None:
    driver = FakeDriver(_script(5, reward=0.0))
    state_machine = RecordingStateMachine()

    await _run_until_script_finishes(driver, state_machine, n_audit_every=2)

    assert len(state_machine.records) == 2


async def test_gap_transition_emits_graph_node_added_once_then_resumes(
    written_events,
) -> None:
    driver = FakeDriver(_script(4, reward=1.0))
    state_machine = RecordingStateMachine()

    await _run_until_script_finishes(driver, state_machine, n_audit_every=1)

    graph_events = [payload for kind, payload in written_events if kind == "graph_node_added"]
    assert len(graph_events) == 1
    assert len([kind for kind, _ in written_events if kind == "craftax_step"]) == 4
    assert state_machine.transitions == 1


async def test_done_writes_craftax_episode_then_resets_and_generation_advances(
    written_events,
) -> None:
    driver = FakeDriver(_script(2, reward=2.0, done_at=2))

    await _run_until_script_finishes(driver, SomaStateMachine())

    kinds = [kind for kind, _ in written_events]
    assert kinds[:3] == ["craftax_step", "craftax_step", "craftax_episode"]
    episode = written_events[2][1]
    assert episode["episode"] == 1
    assert driver.reset_seeds == [42, 42]
    assert driver.episode_generation == 2


async def test_only_allowlisted_events_written(written_events) -> None:
    driver = FakeDriver(_script(2, reward=1.0, done_at=2))

    await _run_until_script_finishes(driver, SomaStateMachine(), n_audit_every=1)

    allowed = {"craftax_step", "craftax_episode", "graph_node_added"}
    assert {kind for kind, _ in written_events} <= allowed
    assert "world_model_updated" not in {kind for kind, _ in written_events}
    assert "simulation_step" not in {kind for kind, _ in written_events}
    assert "training_completed" not in {kind for kind, _ in written_events}


async def test_jsd_mean_varies_across_scripted_candidates(written_events) -> None:
    driver = FakeDriver(_script(4, jsd_start=0.1))

    await _run_until_script_finishes(driver, SomaStateMachine())

    means = [payload["jsd_mean"] for kind, payload in written_events if kind == "craftax_step"]
    assert len(set(means)) >= 4


async def test_cancellederror_propagates_out_of_loop() -> None:
    driver = FakeDriver([], cancel_on_evaluate=True)

    with pytest.raises(asyncio.CancelledError):
        await runner.run_craftax_loop(
            driver, FakeInstrumented(), SomaStateMachine()
        )
