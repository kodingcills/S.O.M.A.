from collections.abc import Sequence
from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path
import subprocess

import pytest
import torch

from backend.craftax_driver import DriverStep, PreActionPrediction
from backend.simulus.instrumentation import InstrumentedActionOutput
from backend.soma import audit


@dataclass(frozen=True, slots=True)
class _FakeEnv:
    scripted_rewards: dict[int, float]

    def fork_step_rewards(
        self, state: int, key: int, actions: Sequence[int]
    ) -> dict[int, float]:
        del state, key
        return {
            int(action): self.scripted_rewards.get(int(action), 0.0)
            for action in actions
        }


def _controller_probs() -> torch.Tensor:
    values = torch.rand(43, generator=torch.Generator().manual_seed(71))
    return values / values.sum()


def _ranked_probs() -> torch.Tensor:
    values = torch.full((43,), 0.01)
    values[11] = 0.9
    values[4] = 0.8
    values[9] = 0.8
    values[2] = 0.7
    values[22] = 0.6
    values[1] = 0.5
    return values / values.sum()


def _prediction(
    step: int,
    action: int,
    reward: float = 0.8,
    probs: torch.Tensor | None = None,
) -> PreActionPrediction:
    return PreActionPrediction(
        episode_generation=3,
        prediction_step=step,
        output=InstrumentedActionOutput(
            action_idx=action,
            J_ua=0.1,
            controller_probs=_controller_probs() if probs is None else probs,
            reward_expectation=reward,
        ),
    )


def _realized(step: int, action: int, reward: float = 1.25) -> DriverStep:
    return DriverStep(step, action, reward, False, 0, {})


def _audit_record(
    chosen: PreActionPrediction,
    env: _FakeEnv | None = None,
    jsd_mean: float = 0.2,
) -> audit.AuditRecord:
    selected_env = _FakeEnv({}) if env is None else env
    return audit.audit_decision_point(
        [chosen],
        chosen,
        _realized(chosen.prediction_step + 1, chosen.output.action_idx),
        selected_env,
        101,
        202,
        jsd_mean=jsd_mean,
    )


def test_mismatch_error_kept_step_pairing() -> None:
    chosen = _prediction(7, 2)

    with pytest.raises(audit.AuditStepMismatchError) as caught:
        audit.audit_decision_point(
            [chosen], chosen, _realized(9, 3), _FakeEnv({}), 101, 202, jsd_mean=0.2
        )

    assert caught.value.field == "step"
    assert caught.value.expected == 8
    assert caught.value.actual == 9


def test_mismatch_error_kept_action_pairing() -> None:
    chosen = _prediction(7, 2)

    with pytest.raises(audit.AuditStepMismatchError) as caught:
        audit.audit_decision_point(
            [chosen], chosen, _realized(8, 3), _FakeEnv({}), 101, 202, jsd_mean=0.2
        )

    assert caught.value.field == "action"
    assert caught.value.expected == 2
    assert caught.value.actual == 3


def test_record_frozen_slots() -> None:
    record = audit.AuditRecord(
        episode_generation=3,
        prediction_step=7,
        chosen_action_idx=2,
        predicted_reward_expectation=0.8,
        chosen_score=0.7,
        jsd_mean=0.2,
        realized_reward=1.25,
        q_p_values={2: 1.0},
        regret=0.0,
        harmful=False,
    )

    with pytest.raises(FrozenInstanceError):
        record.realized_reward = 2.0

    assert not hasattr(record, "__dict__")


def test_pool_size_is_k8_unique() -> None:
    pool = audit.select_candidate_pool(_controller_probs(), 17, 3, 7)

    assert len(pool) == audit.K_CANDIDATES == 8
    assert len(set(pool)) == 8


def test_pool_top6_by_probs_desc_stable() -> None:
    pool = audit.select_candidate_pool(_ranked_probs(), 11, 3, 7)

    assert pool[:6] == [11, 4, 9, 2, 22, 1]


def test_pool_deterministic_same_inputs() -> None:
    probs = _controller_probs()

    first = audit.select_candidate_pool(probs, 17, 3, 7)
    second = audit.select_candidate_pool(probs, 17, 3, 7)

    assert first == second


def test_pool_includes_chosen_action_forced() -> None:
    probs = torch.arange(43, 0, -1, dtype=torch.float32)
    probs /= probs.sum()

    pool = audit.select_candidate_pool(probs, 42, 3, 7)

    assert pool[-1] == 42
    assert len(pool) == len(set(pool)) == 8


def test_pool_random_fills_seedsequence_per_decision_point() -> None:
    probs = torch.arange(43, 0, -1, dtype=torch.float32)
    probs /= probs.sum()

    first = audit.select_candidate_pool(probs, 0, 3, 7)
    repeated = audit.select_candidate_pool(probs, 0, 3, 7)
    other = audit.select_candidate_pool(probs, 0, 9, 11)

    assert first == repeated
    assert first[:6] == other[:6]
    assert first[6:] != other[6:]


def test_regret_is_max_minus_chosen() -> None:
    probs = torch.arange(43, 0, -1, dtype=torch.float32)
    probs /= probs.sum()
    chosen = _prediction(7, 0, probs=probs)

    record = _audit_record(chosen, _FakeEnv({0: 1.25, 1: 2.0}))

    assert record.regret == pytest.approx(0.75)
    assert record.q_p_values[0] == pytest.approx(1.25)
    assert record.q_p_values[1] == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("alternative_reward", "expected"),
    [(0.05, False), (0.050001, True)],
)
def test_harmful_strict_gt_boundary(
    alternative_reward: float, expected: bool
) -> None:
    probs = torch.arange(43, 0, -1, dtype=torch.float32)
    probs /= probs.sum()
    chosen = _prediction(7, 0, probs=probs)

    record = _audit_record(chosen, _FakeEnv({0: 0.0, 1: alternative_reward}))

    assert record.harmful is expected


def test_harm_provisional_always_false() -> None:
    record = _audit_record(_prediction(7, 2))

    assert record.harm_provisional is False


def test_jsd_mean_caller_supplied_kwarg() -> None:
    chosen = _prediction(7, 2)

    record = _audit_record(chosen, jsd_mean=0.314)

    assert record.jsd_mean == pytest.approx(0.314)
    with pytest.raises(TypeError):
        audit.audit_decision_point(
            [chosen], chosen, _realized(8, 2), _FakeEnv({}), 101, 202
        )


def test_audit_file_zero_forbidden_tokens() -> None:
    path = Path("backend/soma/audit.py")

    result = subprocess.run(
        ["grep", "-n", "intrinsic\\|curiosity\\|JSD\\|J_ua", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.stdout == ""
    assert result.returncode == 1
