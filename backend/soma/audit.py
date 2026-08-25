from dataclasses import dataclass
from statistics import fmean
from typing import Sequence

from backend.craftax_driver import DriverStep, PreActionPrediction
from backend.mpc_agent import MPCAgent


class AuditStepMismatchError(Exception):
    """Raised when a realized transition does not match its prediction."""

    def __init__(self, expected: int, actual: int, field: str = "step") -> None:
        self.expected = expected
        self.actual = actual
        self.field = field
        super().__init__(f"{field} mismatch: expected {expected}, got {actual}")


@dataclass(frozen=True, slots=True)
class AuditRecord:
    episode_generation: int
    prediction_step: int
    chosen_action_idx: int
    predicted_reward_expectation: float
    chosen_score: float
    jsd_mean: float
    realized_reward: float


def audit_decision_point(
    stamped_candidates: Sequence[PreActionPrediction],
    chosen: PreActionPrediction,
    realized: DriverStep,
) -> AuditRecord:
    """Pair one stamped prediction with its exact realized transition."""
    expected_step = chosen.prediction_step + 1
    if realized.step != expected_step:
        raise AuditStepMismatchError(expected_step, realized.step)
    if chosen.output.action_idx != realized.action:
        raise AuditStepMismatchError(chosen.output.action_idx, realized.action, "action")
    output = chosen.output
    return AuditRecord(
        episode_generation=chosen.episode_generation,
        prediction_step=chosen.prediction_step,
        chosen_action_idx=output.action_idx,
        predicted_reward_expectation=float(output.reward_expectation),
        chosen_score=MPCAgent._score(output),
        jsd_mean=fmean(float(candidate.output.J_ua) for candidate in stamped_candidates),
        realized_reward=float(realized.reward),
    )
