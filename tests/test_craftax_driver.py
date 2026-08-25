"""tests/test_craftax_driver.py — live loop semantics (slow: loads ckpt)."""
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _state_fingerprint(state) -> bytes:
    parts = []
    stack = [state]
    while stack:
        value = stack.pop()
        if value is None:
            parts.append(b"n")
        elif isinstance(value, torch.Tensor):
            parts.append(value.detach().cpu().numpy().tobytes())
        elif isinstance(value, tuple):
            stack.extend(reversed(value))
        else:
            stack.append(getattr(value, "state", None))
    return b"".join(parts)


def _driver_fingerprint(driver) -> tuple[int, bytes, bytes]:
    import jax

    env_parts = [
        driver.env.raw_observation().tobytes(),
        np.asarray(driver.env._key).tobytes(),
    ]
    env_parts.extend(
        np.asarray(leaf).tobytes()
        for leaf in jax.tree_util.tree_leaves(driver.env._state)
    )
    recurrent = _state_fingerprint(driver.snapshot_recurrent_state())
    return driver.step_count, b"".join(env_parts), recurrent


@pytest.fixture(scope="module")
def driver():
    from backend.craftax_driver import CraftaxDriver
    from backend.simulus.runtime import load_simulus_agent

    agent, _cfg, _dev = load_simulus_agent("cpu")
    d = CraftaxDriver(agent, seed=42)
    return d


def test_reset_seeds_and_state_populated(driver):
    driver.reset(seed=42)
    assert driver.step_count == 0
    assert driver._rs.state is not None or driver._rs.n > 0


def test_step_advances_and_returns_real_values(driver):
    driver.reset(seed=42)
    step0, probs0 = None, None
    step, probs = driver.step_controller()
    assert step.step == 1
    assert 0 <= step.action < 43
    assert isinstance(step.reward, float)
    assert abs(float(probs.sum()) - 1.0) < 1e-3
    assert step.stats["health"] >= 0.0


def test_step_controller_delegates_to_pinned_act_with_probs(driver, monkeypatch):
    from backend.simulus import runtime

    driver.reset(seed=42)
    expected_action = 17
    expected_probs = np.arange(1, 44, dtype=np.float32)
    expected_probs /= expected_probs.sum()
    calls = []

    def pinned_act_with_probs(agent, model_obs, temperature=1.0):
        calls.append((agent, model_obs, temperature))
        return expected_action, expected_probs

    monkeypatch.setattr(runtime, "act_with_probs", pinned_act_with_probs)

    step, probs = driver.step_controller(temperature=0.75)

    assert len(calls) == 1
    assert calls[0][0] is driver._agent
    assert calls[0][2] == 0.75
    assert step.action == expected_action
    assert probs is expected_probs
    assert probs.shape == (43,)


def test_step_controller_matches_released_actor_state_advancement(driver):
    from backend.simulus.runtime import act_with_probs

    driver.reset(seed=42)
    torch.manual_seed(2026)
    expected_action, expected_probs = act_with_probs(
        driver._agent,
        driver.env.to_model_obs(driver._device),
        temperature=1.0,
    )
    expected_actor_state = _state_fingerprint(driver._agent.actor_critic.actor_state)

    driver.reset(seed=42)
    torch.manual_seed(2026)
    step, probs = driver.step_controller()
    actual_actor_state = _state_fingerprint(driver._agent.actor_critic.actor_state)

    assert step.action == expected_action
    assert np.array_equal(probs, expected_probs)
    assert actual_actor_state == expected_actor_state


def test_evaluate_candidates_uses_one_batched_pass(driver, monkeypatch):
    from backend.simulus.instrumentation import InstrumentedActionOutput

    driver.reset(seed=42)
    expected = [
        InstrumentedActionOutput(action_idx=action, J_ua=float(action))
        for action in range(43)
    ]
    calls = []

    def evaluate_batched(model_obs, prior_context, recurrent_state):
        calls.append((model_obs, prior_context, recurrent_state))
        return expected

    def evaluate_sequential(*args, **kwargs):
        pytest.fail("sequential candidate evaluation must not run")

    monkeypatch.setattr(
        driver.instrumented, "evaluate_all_actions_batched", evaluate_batched
    )
    monkeypatch.setattr(
        driver.instrumented, "evaluate_all_actions", evaluate_sequential
    )

    result = driver.evaluate_candidates()

    assert len(calls) == 1
    assert calls[0][1] is driver.prior_context
    assert result == expected
    assert len(result) == 43


def test_candidate_eval_between_steps_is_finite_and_varied(driver):
    driver.reset(seed=42)
    for _ in range(3):
        outs = driver.evaluate_candidates()
        jvals = np.array([o.J_ua for o in outs])
        assert jvals.shape == (43,)
        assert np.all(np.isfinite(jvals))
        assert float(jvals.min()) >= -1e-6
        driver.step_controller()


def test_live_state_not_mutated_by_candidates(driver):
    driver.reset(seed=256)
    before = _state_fingerprint(driver.snapshot_recurrent_state())
    driver.evaluate_candidates()
    after = _state_fingerprint(driver.snapshot_recurrent_state())
    assert before == after


def test_reset_reproducibility(driver):
    driver.reset(seed=42)
    torch.manual_seed(2026)
    s1, p1 = driver.step_controller()
    driver.reset(seed=42)
    torch.manual_seed(2026)
    s2, p2 = driver.step_controller()
    assert s1.action == s2.action
    assert np.allclose(p1, p2)


def test_t7_prediction_at_t_pairs_with_exact_reward_at_t_plus_one(driver):
    from backend.craftax_driver import TransitionAlignmentError

    # Given: a real pre-action candidate output at transition index t.
    driver.reset(seed=42)
    prediction_t = driver.evaluate_stamped_candidates()[5]

    # When: that precomputed prediction and action advance the real environment once.
    record = driver.step_evaluated(5, prediction_t)

    # Then: the exact returned reward is explicitly indexed as R_(t+1).
    assert record.prediction is prediction_t.output
    assert record.prediction_step == 0
    assert record.reward_step == 1
    assert record.reward == record.step.reward == driver.env.last_reward
    assert record.prediction.action_idx == record.step.action == 5

    # And: an off-by-one record fails by index, even when adjacent rewards are equal.
    with pytest.raises(TransitionAlignmentError):
        record.validate_indices(reward_step=2)


def test_stale_prediction_from_prior_step_is_rejected_before_mutation(driver):
    from backend.craftax_driver import TransitionAlignmentError

    # Given: a stamped prediction from t=0 after the driver has advanced to t=1.
    driver.reset(seed=256)
    stale = driver.evaluate_stamped_candidates()[5]
    driver.step_with_action(0)
    before = _driver_fingerprint(driver)

    # When: the stale candidate is offered for the current transition.
    with pytest.raises(TransitionAlignmentError):
        driver.step_evaluated(5, stale)

    # Then: rejection precedes every driver, environment, and recurrent mutation.
    assert _driver_fingerprint(driver) == before


def test_stale_prediction_from_prior_reset_is_rejected_before_mutation(driver):
    from backend.craftax_driver import TransitionAlignmentError

    # Given: a t=0 prediction from the prior episode generation at the same seed.
    driver.reset(seed=512)
    stale = driver.evaluate_stamped_candidates()[5]
    driver.reset(seed=512)
    before = _driver_fingerprint(driver)

    # When: the old-generation candidate is offered at the same numeric step.
    with pytest.raises(TransitionAlignmentError):
        driver.step_evaluated(5, stale)

    # Then: rejection precedes every driver, environment, and recurrent mutation.
    assert _driver_fingerprint(driver) == before


def test_t8_prediction_at_t_pairs_with_exact_symbolic_next_observation(driver):
    # Given: the exact symbolic observation at t before a real action.
    driver.reset(seed=137)
    observation_t = {
        name: value.copy() for name, value in driver.current_obs_tokens().items()
    }

    # When: the driver records the instrumented prediction and same real transition.
    record = driver.step_evaluated(0)

    # Then: the NLL target is explicitly indexed at t+1 and is the returned next obs.
    assert record.observation_step == record.prediction_step == 0
    assert record.next_observation_step == record.reward_step == 1
    for name, value in observation_t.items():
        assert np.array_equal(record.observation[name], value)
    for name, value in driver.current_obs_tokens().items():
        assert np.array_equal(record.next_observation[name], value)
    record.validate_indices(next_observation_step=1)
