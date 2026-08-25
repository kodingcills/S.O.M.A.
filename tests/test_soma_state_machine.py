import inspect

import pytest

from backend.soma.state_machine import MIN_AUDITS_FOR_GAP, PhaseNodeInfo, SomaStateMachine


def test_record_harm_increments_n_z_u_always() -> None:
    machine = SomaStateMachine()

    machine.record_harm(False)
    machine.record_harm(True)

    assert machine.n_z_u == 2


def test_record_harm_counts_n_harmful_only_when_true() -> None:
    machine = SomaStateMachine()

    machine.record_harm(False)
    assert machine.n_harmful == 0

    machine.record_harm(True)
    assert machine.n_harmful == 1


def test_record_harm_returns_harmful_flag() -> None:
    machine = SomaStateMachine()

    assert machine.record_harm(False) is False
    assert machine.record_harm(True) is True


def test_gap_requires_min_audits_and_one_harmful() -> None:
    machine = SomaStateMachine()
    machine.record_harm(True)

    assert machine.should_transition_to_gap() is False

    for _ in range(MIN_AUDITS_FOR_GAP - 1):
        machine.record_harm(False)
    assert machine.should_transition_to_gap() is True


def test_advance_phase_resets_counters() -> None:
    machine = SomaStateMachine()
    machine.record_harm(True)
    machine.record_harm(False)

    info = machine.advance_phase()

    assert info == PhaseNodeInfo("gap-1", "gap", 2, 1)
    assert machine.n_z_u == 0
    assert machine.n_harmful == 0
    with pytest.raises((AttributeError, TypeError)):
        setattr(info, "phase", "detect")


def test_no_delta_threshold_constant() -> None:
    source = inspect.getsource(__import__("backend.soma.state_machine", fromlist=["SomaStateMachine"]))

    assert "HARM_DELTA_THRESHOLD" not in source
