import pytest

from backend.craftax_driver import DriverStep, PreActionPrediction
from backend.simulus.instrumentation import InstrumentedActionOutput
from backend.soma.audit import (
    AuditStepMismatchError,
    AuditRecord,
    audit_decision_point,
)


def _prediction(step: int, action: int, reward: float, jsd: float) -> PreActionPrediction:
    return PreActionPrediction(
        episode_generation=3,
        prediction_step=step,
        output=InstrumentedActionOutput(
            action_idx=action, J_ua=jsd, reward_expectation=reward
        ),
    )


def _realized(step: int, action: int, reward: float = 1.25) -> DriverStep:
    return DriverStep(step, action, reward, False, 0, {})


def test_audit_rejects_step_mismatch_with_typed_error() -> None:
    chosen = _prediction(7, 2, 0.8, 0.1)
    with pytest.raises(AuditStepMismatchError) as caught:
        audit_decision_point([chosen], chosen, _realized(9, 2))
    assert caught.value.expected == 8
    assert caught.value.actual == 9


def test_audit_pairs_chosen_t_with_realized_t_plus_1() -> None:
    chosen = _prediction(7, 2, 0.8, 0.1)
    record = audit_decision_point([chosen], chosen, _realized(8, 2))
    assert record.prediction_step == 7
    assert record.realized_reward == 1.25


def test_audit_computes_jsd_mean_across_all_candidates() -> None:
    candidates = [_prediction(7, 0, 0.5, 0.0), _prediction(7, 1, 0.7, 0.2)]
    record = audit_decision_point(candidates, candidates[1], _realized(8, 1))
    assert record.jsd_mean == pytest.approx(0.1)
    assert record.chosen_score == pytest.approx(0.7 - 0.5 * (0.2 / 0.6931471805599453))


def test_audit_record_is_frozen() -> None:
    chosen = _prediction(7, 2, 0.8, 0.1)
    record = audit_decision_point([chosen], chosen, _realized(8, 2))
    with pytest.raises(AttributeError):
        record.realized_reward = 2.0
