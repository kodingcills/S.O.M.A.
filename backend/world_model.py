from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from typing import Final

import numpy as np
import torch

from backend.belief_state import BeliefState
from backend.craftax_driver import CraftaxDriver, DriverStep, PreActionPrediction
from backend.event_log import write_event
from backend.simulus import CHECKPOINT_SHA256, MODEL_REVISION
from backend.simulus.soma_bridge import (
    achievements_to_dof,
    jsd_to_error_map,
    jsd_to_regional_errors,
)
from backend.world_model_types import (
    ActionPrediction,
    DecisionPoint,
    DuplicateIngestError,
    FineTuneSamples,
    IngestResult,
    InvalidInstrumentationError,
    ModelIdentity,
    OutOfOrderIngestError,
    PredictionResult,
    ReadOnlyWorldModelError,
    RegionError,
    RegionName,
    StalePredictionError,
    SymbolicObservation,
    TransitionResult,
)

_ACTION_COUNT: Final = 43
_REGION_ORDER: Final[tuple[RegionName, ...]] = (
    "upper_left",
    "upper_right",
    "lower_left",
    "lower_right",
    "tool_tissue_boundary",
    "surgical_target_vicinity",
)


@dataclass(frozen=True, slots=True)
class _CapturedStep:
    step: DriverStep
    probabilities: np.ndarray
    observation: dict[str, np.ndarray]
    achievements: np.ndarray


def _symbolic(observation: dict[str, np.ndarray]) -> SymbolicObservation:
    token_2d = np.asarray(observation["token_2d"])
    vector = np.asarray(observation["vector"])
    token = np.asarray(observation["token"])
    return SymbolicObservation(
        token_2d=tuple(tuple(int(value) for value in row) for row in token_2d),
        vector=tuple(float(value) for value in vector.reshape(-1)),
        token=tuple(int(value) for value in token.reshape(-1)),
    )


def _finite_vector(
    value: np.ndarray | torch.Tensor | None, field: str
) -> tuple[float, ...]:
    if value is None:
        raise InvalidInstrumentationError(field, "missing")
    result = tuple(float(item) for item in np.asarray(value).reshape(-1))
    if len(result) != _ACTION_COUNT or not all(math.isfinite(item) for item in result):
        raise InvalidInstrumentationError(field, "expected 43 finite values")
    return result


def _validate_probabilities(
    value: np.ndarray | torch.Tensor | None, field: str
) -> tuple[float, ...]:
    result = _finite_vector(value, field)
    if any(item < 0.0 for item in result) or not math.isclose(
        sum(result), 1.0, rel_tol=1e-5, abs_tol=1e-6
    ):
        raise InvalidInstrumentationError(field, "expected a normalized distribution")
    return result


def _finite_scalar(value: float | None, field: str) -> float:
    if value is None or not math.isfinite(value):
        raise InvalidInstrumentationError(field, "expected a finite value")
    return value


def _capture_step(driver: CraftaxDriver) -> _CapturedStep:
    step, probabilities = driver.step_controller()
    return _CapturedStep(
        step=step,
        probabilities=np.asarray(probabilities),
        observation=driver.current_obs_tokens(),
        achievements=driver.env.achievements(),
    )


class WorldModel:
    def __init__(self, driver: CraftaxDriver, belief: BeliefState | None = None) -> None:
        self._driver = driver
        self.belief = belief if belief is not None else BeliefState()
        if self.belief.world_model_version != 0:
            raise InvalidInstrumentationError(
                "belief.world_model_version", "released checkpoint generation must be zero"
            )
        self._driver_lock = asyncio.Lock()
        self._ingest_lock = asyncio.Lock()
        self._last_ingested: DecisionPoint | None = None

    @property
    def identity(self) -> ModelIdentity:
        return ModelIdentity(
            model_revision=MODEL_REVISION,
            checkpoint_sha256=CHECKPOINT_SHA256,
            adaptation_generation=self.belief.world_model_version,
        )

    async def predict(self) -> PredictionResult:
        async with self._driver_lock:
            stamped = await asyncio.to_thread(self._driver.evaluate_stamped_candidates)
            observation = _symbolic(self._driver.current_obs_tokens())
        return self._parse_prediction(stamped, observation)

    def _parse_prediction(
        self,
        stamped: list[PreActionPrediction],
        observation: SymbolicObservation,
    ) -> PredictionResult:
        if len(stamped) != _ACTION_COUNT:
            raise InvalidInstrumentationError("actions", "expected indices 0..42")
        indices = tuple(item.output.action_idx for item in stamped)
        if indices != tuple(range(_ACTION_COUNT)):
            raise InvalidInstrumentationError("action_idx", "expected ordered indices 0..42")
        provenance = {(item.episode_generation, item.prediction_step) for item in stamped}
        if len(provenance) != 1:
            raise InvalidInstrumentationError("provenance", "candidates disagree")

        outputs = [item.output for item in stamped]
        first = outputs[0]
        logits = _finite_vector(first.controller_logits, "controller_logits")
        probabilities = _validate_probabilities(
            first.controller_probs, "controller_probabilities"
        )
        actions: list[ActionPrediction] = []
        for output in outputs:
            jsd = _finite_scalar(output.J_ua, "jsd")
            reward = _finite_scalar(output.reward_expectation, "reward_expectation")
            termination = _finite_scalar(
                output.termination_prob, "termination_probability"
            )
            if not 0.0 <= termination <= 1.0:
                raise InvalidInstrumentationError("termination_probability", "outside [0, 1]")
            if _finite_vector(output.controller_logits, "controller_logits") != logits:
                raise InvalidInstrumentationError("controller_logits", "candidates disagree")
            if _validate_probabilities(output.controller_probs, "controller_probabilities") != probabilities:
                raise InvalidInstrumentationError("controller_probabilities", "candidates disagree")
            actions.append(
                ActionPrediction(
                    action_idx=output.action_idx,
                    jsd=jsd,
                    reward_expectation=reward,
                    termination_probability=termination,
                )
            )
        episode_generation, prediction_step = next(iter(provenance))
        regions = jsd_to_regional_errors(outputs)
        error_map = jsd_to_error_map(outputs)
        return PredictionResult(
            decision_point=DecisionPoint(episode_generation, prediction_step),
            observation=observation,
            actions=tuple(actions),
            controller_logits=logits,
            controller_probabilities=probabilities,
            regional_errors=tuple((region, regions[region]) for region in _REGION_ORDER),
            error_map=tuple(tuple(float(value) for value in row) for row in error_map),
        )

    async def update(self, prediction: PredictionResult) -> TransitionResult:
        async with self._driver_lock:
            self._reject_stale(prediction)
            captured = await asyncio.to_thread(_capture_step, self._driver)
        probabilities = _validate_probabilities(
            captured.probabilities, "actual_controller_probabilities"
        )
        action = captured.step.action
        if action not in range(_ACTION_COUNT):
            raise InvalidInstrumentationError("selected_action", str(action))
        return TransitionResult(
            prediction=prediction,
            selected_prediction=prediction.actions[action],
            action=action,
            controller_probabilities=probabilities,
            reward=captured.step.reward,
            done=captured.step.done,
            reward_step=captured.step.step,
            achievements=tuple(bool(value) for value in captured.achievements.reshape(-1)),
            stats=tuple(sorted((name, float(value)) for name, value in captured.step.stats.items())),
            next_observation=_symbolic(captured.observation),
        )

    def _reject_stale(self, prediction: PredictionResult) -> None:
        point = prediction.decision_point
        checks = (
            ("episode_generation", self._driver.episode_generation, point.episode_generation),
            ("prediction_step", self._driver.step_count, point.prediction_step),
            ("symbolic_observation", _symbolic(self._driver.current_obs_tokens()), prediction.observation),
        )
        for field, expected, actual in checks:
            if actual != expected:
                raise StalePredictionError(field, str(expected), str(actual))

    async def ingest(self, transition: TransitionResult) -> IngestResult:
        async with self._ingest_lock:
            point = transition.prediction.decision_point
            if point == self._last_ingested:
                raise DuplicateIngestError(point)
            if self._last_ingested is not None and point < self._last_ingested:
                raise OutOfOrderIngestError(self._last_ingested, point)
            incoming_map = np.asarray(transition.prediction.error_map, dtype=np.float32)
            self.belief.set_jsd_error_map(incoming_map)
            self.belief.regional_override = dict(transition.prediction.regional_errors)
            self.belief.active_dof = achievements_to_dof(np.asarray(transition.achievements))
            self.belief.episode_count += 1
            self.belief.last_updated = time.time()
            updated_map = tuple(
                tuple(float(value) for value in row)
                for row in self.belief.prediction_error_map
            )
            regions: tuple[RegionError, ...] = transition.prediction.regional_errors
            event_id = write_event(
                "belief_snapshot",
                global_mean_error=float(self.belief.prediction_error_map.mean()),
                regional_errors=dict(regions),
                world_model_version=self.belief.world_model_version,
                active_dof=self.belief.active_dof,
                episode_count=self.belief.episode_count,
                error_map=updated_map,
                decision_episode=point.episode_generation,
                decision_step=point.prediction_step,
                action=transition.action,
                reward=transition.reward,
            )
            self._last_ingested = point
            return IngestResult(
                world_model_version=self.belief.world_model_version,
                global_error=float(self.belief.prediction_error_map.mean()),
                regional_errors=regions,
                error_map=updated_map,
                active_dof=self.belief.active_dof,
                episode_count=self.belief.episode_count,
                timestamp=self.belief.last_updated,
                event_id=event_id,
            )

    async def fine_tune(
        self, new_samples: FineTuneSamples, epochs: int | None = None
    ) -> None:
        del new_samples, epochs
        raise ReadOnlyWorldModelError()
