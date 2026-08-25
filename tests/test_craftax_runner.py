from __future__ import annotations

import asyncio
import inspect

import numpy as np
import pytest

import backend.soma.craftax_runner as runner
from backend.craftax_driver import CraftaxDriver, DriverStep, PreActionPrediction
from backend.mpc_agent import MPCAgent
from backend.simulus.instrumentation import InstrumentedActionOutput, SimulusInstrumented
from backend.soma.state_machine import SomaStateMachine
from tests.craftax_runner_fakes import (
    AuditRecorder,
    FakeDriver,
    FakeInstrumented,
    LoopFinished,
    RecordingStateMachine,
)


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


@pytest.fixture(autouse=True)
def audit_recorder(monkeypatch: pytest.MonkeyPatch) -> AuditRecorder:
    recorder = AuditRecorder()
    monkeypatch.setattr(runner, "audit_decision_point", recorder)
    return recorder


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

    assert len(state_machine.harm_calls) == 2


async def test_runner_snapshots_before_step_with_action(audit_recorder) -> None:
    # Given
    driver = FakeDriver(_script(1))

    # When
    await _run_until_script_finishes(driver, RecordingStateMachine(), n_audit_every=1)

    # Then
    assert driver.call_order.index("snapshot") < driver.call_order.index("step_with_action")
    audit_call = audit_recorder.calls[0]
    assert audit_call.craftax_state is driver.env.state_marker
    assert audit_call.key is driver.env.key_marker


async def test_event_gains_regret_harmful_only_on_audit_steps(
    written_events,
) -> None:
    # Given
    driver = FakeDriver(_script(3))

    # When
    await _run_until_script_finishes(driver, RecordingStateMachine(), n_audit_every=2)

    # Then
    steps = [payload for kind, payload in written_events if kind == "craftax_step"]
    assert "regret" not in steps[0] and "harmful" not in steps[0]
    assert isinstance(steps[1]["regret"], float)
    assert isinstance(steps[1]["harmful"], bool)
    assert "regret" not in steps[2] and "harmful" not in steps[2]


async def test_runner_calls_record_harm_with_record_harmful(audit_recorder) -> None:
    # Given
    driver = FakeDriver(_script(2))
    state_machine = RecordingStateMachine()

    # When
    await _run_until_script_finishes(driver, state_machine, n_audit_every=1)

    # Then
    assert state_machine.harm_calls == [call.record.harmful for call in audit_recorder.calls]
    assert "state_machine.record(" not in inspect.getsource(runner.run_craftax_loop)


async def test_runner_passes_jsd_mean_kwarg(audit_recorder) -> None:
    # Given
    candidates = _candidates(1, 0.2)
    driver = FakeDriver([(candidates, DriverStep(1, 17, 0.0, False, 0, {}))])

    # When
    await _run_until_script_finishes(driver, RecordingStateMachine(), n_audit_every=1)

    # Then
    assert audit_recorder.calls[0].jsd_mean == pytest.approx(
        np.mean([candidate.output.J_ua for candidate in candidates])
    )


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
