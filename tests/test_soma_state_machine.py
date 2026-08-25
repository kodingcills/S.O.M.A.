import inspect

import pytest

from backend.soma.state_machine import (
    HARM_DELTA_THRESHOLD,
    MIN_AUDITS_FOR_GAP,
    PhaseNodeInfo,
    SomaStateMachine,
)


def test_record_flags_harm_when_delta_exceeds_threshold() -> None:
    machine = SomaStateMachine()

    assert machine.record(1.0, 1.6) is True
    assert machine.n_z_u == 1
    assert machine.n_harmful == 1


def test_record_no_harm_within_threshold() -> None:
    machine = SomaStateMachine()

    assert machine.record(1.0, 1.5) is False
    assert machine.n_z_u == 1
    assert machine.n_harmful == 0


def test_should_transition_false_below_min_audits_even_with_harm() -> None:
    machine = SomaStateMachine()
    machine.record(0.0, HARM_DELTA_THRESHOLD + 0.1)

    assert machine.should_transition_to_gap() is False


def test_should_transition_false_with_min_audits_but_zero_harmful() -> None:
    machine = SomaStateMachine()
    for _ in range(MIN_AUDITS_FOR_GAP):
        machine.record(1.0, 1.0)

    assert machine.should_transition_to_gap() is False


def test_should_transition_true_at_thresholds() -> None:
    machine = SomaStateMachine()
    machine.record(0.0, HARM_DELTA_THRESHOLD + 0.1)
    for _ in range(MIN_AUDITS_FOR_GAP - 1):
        machine.record(1.0, 1.0)

    assert machine.should_transition_to_gap() is True


def test_advance_phase_returns_immutable_info_resets_counters_increments_phase() -> None:
    machine = SomaStateMachine()
    machine.record(0.0, 1.0)
    machine.record(1.0, 1.0)

    info = machine.advance_phase()

    assert info == PhaseNodeInfo("gap-1", "gap", 2, 1)
    assert machine.n_z_u == 0
    assert machine.n_harmful == 0
    with pytest.raises((AttributeError, TypeError)):
        setattr(info, "phase", "detect")


def test_module_imports_no_event_log() -> None:
    source = inspect.getsource(__import__("backend.soma.state_machine", fromlist=["SomaStateMachine"]))

    assert "event_log" not in source
