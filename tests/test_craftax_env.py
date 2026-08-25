"""tests/test_craftax_env.py — SomaCraftaxEnv behavior (jax on cpu)."""
import jax
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


def test_snapshot_returns_live_state_and_key(env):
    # Given: a live Craftax episode.
    env.reset(seed=42)

    # When: the functional state/key pair is snapshotted.
    state, key = env.snapshot()

    # Then: no copies obscure the exact live references.
    assert state is env._state
    assert key is env._key


def test_snapshot_survives_live_step_rebind(env):
    # Given: a snapshot and byte copies of its pre-step leaves.
    env.reset(seed=42)
    state, key = env.snapshot()
    state_structure = jax.tree_util.tree_structure(state)
    state_leaves = [
        np.asarray(leaf).copy() for leaf in jax.tree_util.tree_leaves(state)
    ]
    key_before = np.asarray(key).copy()

    # When: the live environment advances and rebinds its state and key.
    env.step(0)

    # Then: the snapshot remains the unchanged pre-step pair.
    assert jax.tree_util.tree_structure(state) == state_structure
    for actual, expected in zip(
        jax.tree_util.tree_leaves(state), state_leaves, strict=True
    ):
        assert np.array_equal(np.asarray(actual), expected)
    assert np.array_equal(np.asarray(key), key_before)


def test_fork_purity_live_state_key_unchanged(env):
    # Given: copies of every live state/key leaf.
    env.reset(seed=42)
    state, key = env._state, env._key
    before_leaves, before_structure = jax.tree_util.tree_flatten((state, key))
    before_copies = [np.asarray(leaf).copy() for leaf in before_leaves]

    # When: counterfactual rewards are evaluated.
    env.fork_step_rewards(state, key, [3, 17])

    # Then: every live state/key leaf is byte-identical.
    after_leaves, after_structure = jax.tree_util.tree_flatten(
        (env._state, env._key)
    )
    assert after_structure == before_structure
    for actual, expected in zip(after_leaves, before_copies, strict=True):
        assert np.array_equal(np.asarray(actual), expected)


def test_common_random_numbers_single_split(env):
    # Given: a live snapshot and independently derived shared transition key.
    env.reset(seed=42)
    state, key = env._state, env._key
    _, transition_key = jax.random.split(key)
    actions = [3, 17]
    expected = {
        action: float(
            env._env.step(transition_key, state, action, env._params)[2]
        )
        for action in actions
    }

    # When: both counterfactual actions are forked.
    rewards = env.fork_step_rewards(state, key, actions)

    # Then: both use exactly the one independently split transition key.
    assert rewards == expected


def test_fork_rewards_are_float_extrinsic(env):
    # Given: two actions from a live functional state/key pair.
    env.reset(seed=42)
    state, key = env._state, env._key

    # When: their one-step raw environment rewards are evaluated.
    rewards = env.fork_step_rewards(state, key, [3, 17])

    # Then: the API exposes only action-indexed Python floats.
    assert set(rewards) == {3, 17}
    assert all(type(action) is int for action in rewards)
    assert all(type(reward) is float for reward in rewards.values())


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
