from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

RegionName = Literal[
    "upper_left",
    "upper_right",
    "lower_left",
    "lower_right",
    "tool_tissue_boundary",
    "surgical_target_vicinity",
]
RegionError: TypeAlias = tuple[RegionName, float]
ErrorMap: TypeAlias = tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    model_revision: str
    checkpoint_sha256: str
    adaptation_generation: int


@dataclass(frozen=True, slots=True, order=True)
class DecisionPoint:
    episode_generation: int
    prediction_step: int


@dataclass(frozen=True, slots=True)
class SymbolicObservation:
    token_2d: tuple[tuple[int, ...], ...]
    vector: tuple[float, ...]
    token: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ActionPrediction:
    action_idx: int
    jsd: float
    reward_expectation: float
    termination_probability: float


@dataclass(frozen=True, slots=True)
class PredictionResult:
    decision_point: DecisionPoint
    observation: SymbolicObservation
    actions: tuple[ActionPrediction, ...]
    controller_logits: tuple[float, ...]
    controller_probabilities: tuple[float, ...]
    regional_errors: tuple[RegionError, ...]
    error_map: ErrorMap


@dataclass(frozen=True, slots=True)
class TransitionResult:
    prediction: PredictionResult
    selected_prediction: ActionPrediction
    action: int
    controller_probabilities: tuple[float, ...]
    reward: float
    done: bool
    reward_step: int
    achievements: tuple[bool, ...]
    stats: tuple[tuple[str, float], ...]
    next_observation: SymbolicObservation


@dataclass(frozen=True, slots=True)
class IngestResult:
    world_model_version: int
    global_error: float
    regional_errors: tuple[RegionError, ...]
    error_map: ErrorMap
    active_dof: int
    episode_count: int
    timestamp: float
    event_id: int


FineTuneSamples: TypeAlias = tuple[TransitionResult, ...]


@dataclass(frozen=True, slots=True)
class StalePredictionError(Exception):
    field: str
    expected: str
    actual: str

    def __str__(self) -> str:
        return f"stale {self.field}: expected {self.expected}, got {self.actual}"


@dataclass(frozen=True, slots=True)
class DuplicateIngestError(Exception):
    decision_point: DecisionPoint

    def __str__(self) -> str:
        return f"decision point {self.decision_point} was already ingested"


@dataclass(frozen=True, slots=True)
class OutOfOrderIngestError(Exception):
    previous: DecisionPoint
    received: DecisionPoint

    def __str__(self) -> str:
        return f"decision point {self.received} precedes {self.previous}"


@dataclass(frozen=True, slots=True)
class InvalidInstrumentationError(Exception):
    field: str
    detail: str

    def __str__(self) -> str:
        return f"invalid instrumentation {self.field}: {self.detail}"


@dataclass(frozen=True, slots=True)
class ReadOnlyWorldModelError(Exception):
    operation: str = "fine_tune"

    def __str__(self) -> str:
        return f"released Simulus WorldModel is read-only; {self.operation} is rejected"
