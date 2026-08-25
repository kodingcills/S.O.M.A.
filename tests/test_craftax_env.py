"""tests/test_craftax_env.py — SomaCraftaxEnv behavior (jax on cpu)."""
import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(scope="module")
def env():
    from backend.simulus.craftax_env import SomaCraftaxEnv

    e = SomaCraftaxEnv(seed=42)
    yield e
    del e


def test_reset_shapes_and_dtypes(env):
    obs = env.reset(seed=42)
    assert obs["token_2d"].shape == (99, 4)
    assert np.issubdtype(obs["token_2d"].dtype, np.integer)
    assert obs["vector"].shape == (47,)
    assert obs["vector"].dtype == np.float32
    assert obs["token"].shape == (1,)
    raw = env.raw_observation()
    assert raw.shape == (8268,) and raw.dtype == np.float32


def test_seed_determinism():
    from backend.simulus.craftax_env import SomaCraftaxEnv

    a, b = SomaCraftaxEnv(seed=42), SomaCraftaxEnv(seed=42)
    oa, ob = a.reset(seed=42), b.reset(seed=42)
    for key in oa:
        assert np.array_equal(oa[key], ob[key]), f"obs mismatch: {key}"
    rng = np.random.default_rng(7)
    for act in rng.integers(0, 43, size=10):
        sa, _, _ = a.step(int(act))
        sb, _, _ = b.step(int(act))
        for key in sa:
            assert np.array_equal(sa[key], sb[key])


def test_step_contract_and_no_autoreset(env):
    env.reset(seed=137)
    steps = 0
    while True:
        obs, reward, done = env.step(steps % 43)
        steps += 1
        assert isinstance(reward, float)
        if done or steps > 500:
            break
    if done:
        with pytest.raises(RuntimeError):
            env.step(0)


def test_achievements_captured(env):
    env.reset(seed=42)
    ach = env.achievements()
    assert ach.shape == (67,)
    assert ach.dtype == bool
    assert env.n_achievements() >= 0


def test_game_stats_indices_sane(env):
    env.reset(seed=42)
    stats = env.game_stats()
    expected_keys = {
        "health", "drink", "food", "energy",
        "light_level", "is_sleeping", "is_resting",
    }
    assert set(stats) == expected_keys
    for value in stats.values():
        assert np.isfinite(value)
        assert -0.01 <= value <= 2.1


def test_to_model_obs_preprocessing(env):
    import torch

    model_obs = env.to_model_obs(torch.device("cpu"))
    assert set(model_obs.keys()) == {"token_2d", "vector", "token"} or all(
        hasattr(k, "name") for k in model_obs
    )
    for tensor in model_obs.values():
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape[0] == 1  # batch dim added by processor pipeline


def test_t11_each_integer_matches_raw_no_autoreset_craftax_semantics(env):
    import jax
    from craftax.craftax.constants import Action

    # Given: every semantic action in installed Craftax 1.4.5.
    assert [int(action.value) for action in Action] == list(range(43))

    for action in Action:
        env.reset(seed=1000 + int(action.value))
        next_key, transition_key = jax.random.split(env._key)
        expected_obs, expected_state, expected_reward, expected_done, _ = (
            env._env.step(
                transition_key,
                env._state,
                int(action.value),
                env._params,
            )
        )

        # When: the explicit-key SOMA adapter forwards that same integer once.
        _symbolic_obs, reward, done = env.step(int(action.value))

        # Then: raw observation, reward, done, key, and state match that transition.
        assert np.array_equal(env.raw_observation(), np.asarray(expected_obs))
        assert reward == float(expected_reward)
        assert done is bool(expected_done)
        assert np.array_equal(np.asarray(env._key), np.asarray(next_key))
        actual_leaves = jax.tree_util.tree_leaves(env._state)
        expected_leaves = jax.tree_util.tree_leaves(expected_state)
        assert len(actual_leaves) == len(expected_leaves)
        for actual, expected in zip(actual_leaves, expected_leaves, strict=True):
            assert np.array_equal(np.asarray(actual), np.asarray(expected))
