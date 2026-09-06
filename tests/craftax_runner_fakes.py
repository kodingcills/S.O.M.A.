from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from backend.craftax_driver import DriverStep, PreActionPrediction
from backend.mpc_agent import MPCAgent
from backend.soma.audit import AuditRecord
from backend.soma.state_machine import SomaStateMachine


class LoopFinished(Exception):
    pass


class FakeInstrumented:
    pass


class FakeEnv:
    def __init__(self, call_order: list[str]) -> None:
        self.call_order = call_order
        self.state_marker = SimpleNamespace(name="pre-state")
        self.key_marker = SimpleNamespace(name="pre-key")
        self._achievements = np.array([True, False, True, True], dtype=bool)

    def achievements(self) -> np.ndarray:
        return self._achievements.copy()

    def snapshot(self) -> tuple[SimpleNamespace, SimpleNamespace]:
        self.call_order.append("snapshot")
        return self.state_marker, self.key_marker

    def fork_step_rewards(
        self,
        state: SimpleNamespace,
        key: SimpleNamespace,
        actions: list[int],
    ) -> dict[int, float]:
        assert state is self.state_marker
        assert key is self.key_marker
        self.call_order.append("fork_step_rewards")
        return {action: float(action) for action in actions}


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
        self.call_order: list[str] = []
        self.env = FakeEnv(self.call_order)

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
        self.call_order.append("step_with_action")
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
        self.harm_calls: list[bool] = []
        self.legacy_calls: list[tuple[float, float]] = []
        self.transitions = 0

    @property
    def n_z_u(self) -> int:
        return self.inner.n_z_u

    @property
    def n_harmful(self) -> int:
        return self.inner.n_harmful

    def record(self, predicted: float, reward: float) -> bool:
        self.legacy_calls.append((predicted, reward))
        return self.inner.record_harm(predicted > reward)

    def record_harm(self, harmful: bool) -> bool:
        self.harm_calls.append(harmful)
        return self.inner.record_harm(harmful)

    def should_transition_to_gap(self) -> bool:
        return self.inner.should_transition_to_gap()

    def advance_phase(self):
        self.transitions += 1
        return self.inner.advance_phase()


@dataclass(frozen=True, slots=True)
class AuditCall:
    env: FakeEnv | None
    craftax_state: SimpleNamespace | None
    key: SimpleNamespace | None
    jsd_mean: float | None
    record: AuditRecord


class AuditRecorder:
    def __init__(self) -> None:
        self.calls: list[AuditCall] = []

    def __call__(
        self,
        stamped_candidates: list[PreActionPrediction],
        chosen: PreActionPrediction,
        realized: DriverStep,
        env: FakeEnv | None = None,
        craftax_state: SimpleNamespace | None = None,
        key: SimpleNamespace | None = None,
        *,
        jsd_mean: float | None = None,
    ) -> AuditRecord:
        del stamped_candidates
        harmful = realized.reward < chosen.output.reward_expectation
        record = AuditRecord(
            episode_generation=chosen.episode_generation,
            prediction_step=chosen.prediction_step,
            chosen_action_idx=chosen.output.action_idx,
            predicted_reward_expectation=chosen.output.reward_expectation,
            chosen_score=MPCAgent._score(chosen.output),
            jsd_mean=-1.0 if jsd_mean is None else jsd_mean,
            realized_reward=realized.reward,
            q_p_values={chosen.output.action_idx: realized.reward},
            regret=0.125,
            harmful=harmful,
            harm_provisional=False,
        )
        self.calls.append(AuditCall(env, craftax_state, key, jsd_mean, record))
        return record
