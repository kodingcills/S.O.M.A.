"""Craftax adapter with pinned Simulus preprocessing.

Dynamics: raw craftax v1.4.5 stepped via the functional gymnax API with
explicit PRNG keys (no-auto-reset; caller decides episode boundaries).
Observations: transformed by the VENDORED CraftaxWrapper class imported
verbatim from backend/simulus_runtime/src — byte-identical preprocessing
to the pinned playback path (verified by tests/test_craftax_env.py).

Extras over the pinned chain (recorded in the reproduction manifest):
  - explicit seed control via jax PRNGKey
  - achievements captured from the raw jax state each step
"""
from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import backend.simulus as pins


def _shim_env(observation_space: Any, action_space: Any):
    """CraftaxWrapper asserts gymnasium.Env — satisfy it with real spaces."""
    import gymnasium as gym

    class _Env(gym.Env):
        metadata: dict = {}

        def __init__(self) -> None:
            self.observation_space = observation_space
            self.action_space = action_space

    return _Env()


class SomaCraftaxEnv:
    """One Craftax episode stream. Not a gymnasium.Env on purpose: SOMA owns
    lifecycle explicitly (reset/step/achievements) instead of wrapper magic.
    """

    NUM_ACTIONS = 43

    def __init__(self, seed: int = 42) -> None:
        import gymnasium as gym

        import backend.simulus as pins

        pins.bootstrap()
        from craftax.craftax_env import make_craftax_env_from_name
        # Vendored pinned preprocessing (verbatim class).
        from envs.wrappers.craftax import CraftaxWrapper

        self._env = make_craftax_env_from_name(
            "Craftax-Symbolic-v1", auto_reset=False
        )
        self._params = self._env.default_params
        self._key = jax.random.PRNGKey(int(seed))
        self._state = None
        self._obs_raw: np.ndarray | None = None
        self._done = False
        self.episode_step = 0

        raw_obs_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(8268,), dtype=np.float32
        )
        raw_act_space = gym.spaces.Discrete(self.NUM_ACTIONS)
        self._preprocess = CraftaxWrapper(
            _shim_env(raw_obs_space, raw_act_space)
        )
        self.last_reward: float = 0.0

    # -- lifecycle -----------------------------------------------------------

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        if seed is not None:
            self._key = jax.random.PRNGKey(int(seed))
        k1, k2 = jax.random.split(self._key)
        self._key = k1
        obs, self._state = self._env.reset(k2, self._params)
        self._obs_raw = np.asarray(obs, dtype=np.float32)
        self._done = False
        self.episode_step = 0
        self.last_reward = 0.0
        return self.observation()

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool]:
        if self._done or self._state is None:
            raise RuntimeError("step() called on finished episode — reset() first")
        k1, k2 = jax.random.split(self._key)
        self._key = k1
        obs, self._state, _reward_raw, done, _info = self._env.step(
            k2, self._state, int(action), self._params
        )
        self._obs_raw = np.asarray(obs, dtype=np.float32)
        self._done = bool(done)
        self.last_reward = float(_reward_raw)
        self.episode_step += 1
        return self.observation(), self.last_reward, self._done

    @property
    def done(self) -> bool:
        return self._done

    # -- observations ----------------------------------------------------------

    def observation(self) -> dict[str, np.ndarray]:
        assert self._obs_raw is not None, "reset() first"
        transformed = self._preprocess.observation(self._obs_raw)
        return {
            "token_2d": np.asarray(transformed["map"], dtype=np.int32),
            "vector": np.asarray(transformed["stats"], dtype=np.float32),
            "token": np.asarray(transformed["direction"], dtype=np.int32),
        }

    def raw_observation(self) -> np.ndarray:
        """Flat (8268,) float32 — the exact simulator observation vector."""
        assert self._obs_raw is not None
        return self._obs_raw.copy()

    def to_model_obs(self, device) -> dict:
        """Enum-keyed preprocessed tensors exactly as the pinned collector
        builds them: token/2d long, vector sym_log(float32), batch dim [1,…].
        """
        import torch

        from utils.preprocessing import get_obs_processor  # vendored
        from utils.types import ObsModality  # vendored

        obs = self.observation()
        mapping = {
            "token_2d": ObsModality.token_2d,
            "vector": ObsModality.vector,
            "token": ObsModality.token,
        }
        model_obs: dict = {}
        for key, modality in mapping.items():
            processor = get_obs_processor(modality)
            tensor = processor.to_torch(
                np.asarray(obs[key])[None, ...], device=device
            )
            model_obs[modality] = processor(tensor)
        assert all(isinstance(v, torch.Tensor) for v in model_obs.values())
        return model_obs

    # -- privileged reads (evaluator-side; never model inputs) -----------------

    def achievements(self) -> np.ndarray:
        """bool (67,) copy of state.achievements."""
        return np.asarray(jax.device_get(self._state.achievements)).astype(bool)

    def n_achievements(self) -> int:
        return int(self.achievements().sum())

    def game_stats(self) -> dict[str, float]:
        """Named scalar stats from the stats vector (indices verified in tests).

        Raw inventory layout: inv(16) potions(6) intrinsics(9) dir(4)
        armour(4) ench(4) specials(8). stats drops dir(4) → indices ≥31 shift −4.
        """
        v = self.observation()["vector"]
        return {
            "health": float(v[22]),
            "drink": float(v[23]),
            "food": float(v[24]),
            "energy": float(v[25]),
            "light_level": float(v[39]),
            "is_sleeping": float(v[40]),
            "is_resting": float(v[41]),
        }
