"""Live Craftax episode driver under the released Simulus controller.

State-visitation protocol (contract §2): primary trajectories come from the
controller's sampled policy (temperature 1.0); candidate-action JSD queries
run on cloned world-model state and NEVER replace the live sampled action.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
import torch

from backend.simulus import runtime
from backend.simulus.craftax_env import SomaCraftaxEnv
from backend.simulus.instrumentation import (
    InstrumentedActionOutput,
    SimulusInstrumented,
    _clone_recurrent_state,
    build_block,
    embed_action_flat,
    embed_obs_block,
)


@dataclass(frozen=True, slots=True)
class DriverStep:
    step: int
    action: int
    reward: float
    done: bool
    n_achievements: int
    stats: dict[str, float]


@dataclass(frozen=True, slots=True)
class TransitionAlignmentError(Exception):
    field: str
    expected: int
    actual: int

    def __str__(self) -> str:
        return f"{self.field} must be {self.expected}, got {self.actual}"


@dataclass(frozen=True, slots=True)
class PreActionPrediction:
    episode_generation: int
    prediction_step: int
    output: InstrumentedActionOutput


@dataclass(frozen=True, slots=True)
class EvaluatedTransition:
    prediction_step: int
    reward_step: int
    observation_step: int
    next_observation_step: int
    prediction: InstrumentedActionOutput
    observation: Mapping[str, np.ndarray]
    next_observation: Mapping[str, np.ndarray]
    reward: float
    step: DriverStep

    def __post_init__(self) -> None:
        self.validate_indices()
        if self.prediction.action_idx != self.step.action:
            raise TransitionAlignmentError(
                "prediction.action_idx", self.step.action, self.prediction.action_idx
            )

    def validate_indices(
        self,
        *,
        reward_step: int | None = None,
        next_observation_step: int | None = None,
    ) -> None:
        expected_next = self.prediction_step + 1
        actual_reward = self.reward_step if reward_step is None else reward_step
        actual_next_obs = (
            self.next_observation_step
            if next_observation_step is None
            else next_observation_step
        )
        indices = (
            ("observation_step", self.prediction_step, self.observation_step),
            ("reward_step", expected_next, actual_reward),
            ("next_observation_step", expected_next, actual_next_obs),
        )
        for field, expected, actual in indices:
            if actual != expected:
                raise TransitionAlignmentError(field, expected, actual)


def _freeze_symbolic_observation(
    observation: dict[str, np.ndarray],
) -> Mapping[str, np.ndarray]:
    frozen: dict[str, np.ndarray] = {}
    for name, value in observation.items():
        retained = value.copy()
        retained.flags.writeable = False
        frozen[name] = retained
    return MappingProxyType(frozen)


class CraftaxDriver:
    """One live episode stream: env + WM recurrent state + prior context."""

    def __init__(self, agent, seed: int = 42) -> None:
        self._agent = agent
        self._wm = agent.world_model
        self._device = next(self._wm.parameters()).device
        self._instrumented = SimulusInstrumented(agent)
        self.env = SomaCraftaxEnv(seed=seed)
        self.seed = seed
        self._rs = self._wm.get_empty_state()
        self._prior_context: torch.Tensor
        self.step_count = 0
        self._episode_generation = 0
        self.reset(seed)

    def reset(self, seed: int | None = None) -> None:
        self._episode_generation += 1
        if seed is not None:
            self.seed = int(seed)
        self.env.reset(self.seed)
        obs_model = self.env.to_model_obs(self._device)
        self._agent.reset_actor_critic(
            n=1, burnin_observations=None, mask_padding=None
        )
        obs_emb_flat = embed_obs_block(
            self._wm, self._agent.tokenizer, obs_model, self._device
        )
        noop_emb = embed_action_flat(self._wm, 0, self._device)
        # RetNet needs ≥context_length full blocks before prediction latents exist.
        ctx_blocks = max(1, int(getattr(self._wm, "context_length", 1)))
        with torch.no_grad():
            for _ in range(ctx_blocks):
                block = build_block(self._wm, obs_emb_flat, noop_emb)
                outs = self._wm.forward_inference(
                    block.unsqueeze(1), recurrent_state=self._rs
                )
                del outs
        self._prior_context = obs_emb_flat
        self.step_count = 0

    @property
    def instrumented(self) -> SimulusInstrumented:
        return self._instrumented

    @property
    def episode_generation(self) -> int:
        return self._episode_generation

    def current_obs_tokens(self):
        return self.env.observation()

    def evaluate_candidates(self) -> list[InstrumentedActionOutput]:
        return self._instrumented.evaluate_all_actions_batched(
            self.env.to_model_obs(self._device),
            self._prior_context,
            self._rs,
        )

    def evaluate_stamped_candidates(self) -> list[PreActionPrediction]:
        """Evaluate candidates stamped with their exact driver decision point."""
        return [
            PreActionPrediction(self._episode_generation, self.step_count, output)
            for output in self.evaluate_candidates()
        ]

    def step_with_action(self, action: int) -> DriverStep:
        """Pinned world_model_env semantics: the live state consumes exactly
        ONE (obs_t ‖ a_t) block per step; obs_{t+1} joins the next block."""
        step, _next_observation = self._advance_action(action)
        return step

    def _advance_action(
        self, action: int
    ) -> tuple[DriverStep, dict[str, np.ndarray]]:
        act_emb = embed_action_flat(self._wm, int(action), self._device)
        block = build_block(self._wm, self._prior_context, act_emb)
        with torch.no_grad():
            outs = self._wm.forward_inference(
                block.unsqueeze(1), recurrent_state=self._rs
            )
            del outs
        next_observation, reward, done = self.env.step(int(action))
        new_obs_model = self.env.to_model_obs(self._device)
        self._prior_context = embed_obs_block(
            self._wm, self._agent.tokenizer, new_obs_model, self._device
        )
        self.step_count += 1
        step = DriverStep(
            step=self.step_count,
            action=int(action),
            reward=float(reward),
            done=bool(done),
            n_achievements=self.env.n_achievements(),
            stats=self.env.game_stats(),
        )
        return step, next_observation

    def step_evaluated(
        self,
        action: int,
        prediction: PreActionPrediction | None = None,
    ) -> EvaluatedTransition:
        """Pair a pre-action prediction with its exact real transition target."""
        prediction_step = self.step_count
        if prediction is None:
            measured_prediction = self._instrumented.evaluate_action(
                self.env.to_model_obs(self._device),
                self._prior_context,
                self._rs,
                int(action),
            )
            if measured_prediction.action_idx != int(action):
                raise TransitionAlignmentError(
                    "prediction.action_idx", int(action), measured_prediction.action_idx
                )
        else:
            provenance = (
                (
                    "prediction.episode_generation",
                    self._episode_generation,
                    prediction.episode_generation,
                ),
                (
                    "prediction.prediction_step",
                    prediction_step,
                    prediction.prediction_step,
                ),
                ("prediction.action_idx", int(action), prediction.output.action_idx),
            )
            for field, expected, actual in provenance:
                if actual != expected:
                    raise TransitionAlignmentError(field, expected, actual)
            measured_prediction = prediction.output
        observation = _freeze_symbolic_observation(self.env.observation())
        step, next_observation = self._advance_action(action)
        return EvaluatedTransition(
            prediction_step=prediction_step,
            reward_step=step.step,
            observation_step=prediction_step,
            next_observation_step=step.step,
            prediction=measured_prediction,
            observation=observation,
            next_observation=_freeze_symbolic_observation(next_observation),
            reward=step.reward,
            step=step,
        )

    def step_controller(
        self, temperature: float = 1.0
    ) -> tuple[DriverStep, np.ndarray]:
        """Sample from the released controller and advance one env step."""
        action, probs = runtime.act_with_probs(
            self._agent,
            self.env.to_model_obs(self._device),
            temperature,
        )
        return self.step_with_action(action), probs

    def controller_probs(self, temperature: float = 1.0):
        _logits, probs = self._instrumented.controller_distribution(
            self.env.to_model_obs(self._device), temperature
        )
        p = np.asarray(probs, dtype=np.float64)
        p = np.clip(p, 0.0, None)
        return p / p.sum()

    def snapshot_recurrent_state(self):
        return _clone_recurrent_state(self._rs)

    @property
    def prior_context(self):
        return self._prior_context
