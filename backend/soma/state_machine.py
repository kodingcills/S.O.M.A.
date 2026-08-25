"""Pure phase-transition state machine for audited SOMA decisions."""

from dataclasses import dataclass
from typing import Final

HARM_DELTA_THRESHOLD: Final = 0.5
MIN_AUDITS_FOR_GAP: Final = 3
PHASES: Final = ("detect", "gap")


@dataclass(frozen=True, slots=True)
class PhaseNodeInfo:
    """Immutable information about a completed phase transition."""

    node_id: str
    phase: str
    n_z_u: int
    n_harmful: int


class SomaStateMachine:
    """Count audits and provisional harm signals across phase transitions.

    The harm label is a provisional real-signal proxy pending the §3 Q_P
    oracle pipeline.
    """

    def __init__(self, phase_index: int = 0) -> None:
        self._phase_index = phase_index % len(PHASES)
        self._transition_count = 0
        self._n_z_u = 0
        self._n_harmful = 0

    @property
    def n_z_u(self) -> int:
        return self._n_z_u

    @property
    def n_harmful(self) -> int:
        return self._n_harmful

    def record(self, predicted_reward_expectation: float, realized_reward: float) -> bool:
        """Record one audit and return whether its harm proxy fired."""
        harm = abs(realized_reward - predicted_reward_expectation) > HARM_DELTA_THRESHOLD
        self._n_z_u += 1
        if harm:
            self._n_harmful += 1
        return harm

    def should_transition_to_gap(self) -> bool:
        """Return whether enough audits include at least one harmful result."""
        return self._n_z_u >= MIN_AUDITS_FOR_GAP and self._n_harmful >= 1

    def advance_phase(self) -> PhaseNodeInfo:
        """Advance cyclically, return immutable node info, and reset audits."""
        self._phase_index = (self._phase_index + 1) % len(PHASES)
        self._transition_count += 1
        phase = PHASES[self._phase_index]
        info = PhaseNodeInfo(
            node_id=f"{phase}-{self._transition_count}",
            phase=phase,
            n_z_u=self._n_z_u,
            n_harmful=self._n_harmful,
        )
        self._n_z_u = 0
        self._n_harmful = 0
        return info
