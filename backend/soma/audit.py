from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Protocol, TypeVar

import numpy as np
from torch import Tensor

from backend.craftax_driver import DriverStep, PreActionPrediction
from backend.mpc_agent import MPCAgent

REGRET_THRESHOLD: Final = 0.05
TOP_K_DIRECT: Final = 6
N_RANDOM_FILLS: Final = 2
K_CANDIDATES: Final = 8

StateT = TypeVar("StateT", contravariant=True)
KeyT = TypeVar("KeyT", contravariant=True)


class ForkRewardEnv(Protocol[StateT, KeyT]):
    def fork_step_rewards(
        self, state: StateT, key: KeyT, actions: Sequence[int]
    ) -> dict[int, float]: ...


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
    q_p_values: dict[int, float]
    regret: float
    harmful: bool
    harm_provisional: bool = False


def select_candidate_pool(
    controller_probs: Tensor,
    chosen_action_idx: int,
    episode_generation: int,
    prediction_step: int,
) -> list[int]:
    probabilities = np.asarray(controller_probs, dtype=float)
    direct = np.argsort(-probabilities, kind="stable")[:TOP_K_DIRECT]
    pool = [int(action) for action in direct]

    rng = np.random.default_rng(
        np.random.SeedSequence(
            [int(episode_generation), int(prediction_step)]
        )
    )
    random_fills: list[int] = []
    while len(random_fills) < N_RANDOM_FILLS:
        candidate = int(rng.integers(0, 43))
        if candidate not in pool and candidate not in random_fills:
            random_fills.append(candidate)
    pool.extend(random_fills)

    if chosen_action_idx not in pool:
        pool[-1] = int(chosen_action_idx)
    pool = list(dict.fromkeys(pool))
    assert len(pool) == K_CANDIDATES
    return pool


def audit_decision_point(
    stamped_candidates: Sequence[PreActionPrediction],
    chosen: PreActionPrediction,
    realized: DriverStep,
    env: ForkRewardEnv[StateT, KeyT],
    craftax_state: StateT,
    key: KeyT,
    *,
    jsd_mean: float,
) -> AuditRecord:
    """Label §3 harm from extrinsic-only Q_P model-free CRN forks.

    The strict threshold compares the chosen one-step return with the best
    counterfactual rollout in the deterministic candidate pool.
    """
    del stamped_candidates
    expected_step = chosen.prediction_step + 1
    if realized.step != expected_step:
        raise AuditStepMismatchError(expected_step, realized.step)
    if chosen.output.action_idx != realized.action:
        raise AuditStepMismatchError(chosen.output.action_idx, realized.action, "action")

    output = chosen.output
    assert output.controller_probs is not None
    candidate_pool = select_candidate_pool(
        output.controller_probs,
        output.action_idx,
        chosen.episode_generation,
        chosen.prediction_step,
    )
    q_p_values = env.fork_step_rewards(craftax_state, key, candidate_pool)
    chosen_action_idx = int(output.action_idx)
    regret = float(
        max(q_p_values.values()) - q_p_values[chosen_action_idx]
    )
    harmful = bool(regret > REGRET_THRESHOLD)

    return AuditRecord(
        episode_generation=chosen.episode_generation,
        prediction_step=chosen.prediction_step,
        chosen_action_idx=chosen_action_idx,
        predicted_reward_expectation=float(output.reward_expectation),
        chosen_score=MPCAgent._score(output),
        jsd_mean=float(jsd_mean),
        realized_reward=float(realized.reward),
        q_p_values=q_p_values,
        regret=regret,
        harmful=harmful,
    )
