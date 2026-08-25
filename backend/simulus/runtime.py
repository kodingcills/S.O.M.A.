"""Loads the pinned Simulus agent for the Craftax benchmark.

Mirrors play_craftax() from the vendored pinned commit exactly:
SingleProcessEnv(make_craftax) → build_agent → agent.load(Craftax.pt,
load_tokenizer=False, load_world_model=True, load_actor_critic=True).

Device resolution: SOMA_SIMULUS_DEVICE env override > mps (Apple Silicon)
> cpu. bf16 autocast stays off (cfg.common.use_bf16_autocast default).
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import backend.simulus as pins

_LOCK = threading.Lock()
_CACHED: dict[str, Any] = {}


def checkpoint_path() -> Path:
    p = Path(__file__).resolve().parent.parent / "models" / "Craftax.pt"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing — download leorc/Simulus Craftax.pt @ {pins.MODEL_REVISION}"
        )
    return p


def resolve_device() -> str:
    override = os.environ.get("SOMA_SIMULUS_DEVICE")
    if override:
        return override
    # NOT mps: vendored Agent.act() opens torch.autocast unconditionally and
    # torch 2.2 rejects device_type='mps'. SOMA_SIMULUS_DEVICE overrides.
    return "cpu"


def _compose_cfg(device: str):
    """Hydra-compose the craftax benchmark config, mirroring play.py."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    pins.bootstrap()  # vendored modules must be importable for _target_ resolution
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(pins.runtime_dir() / "config"),
        version_base=None,
        job_name="soma",
    ):
        cfg = compose(
            config_name="base",
            overrides=[
                "benchmark=craftax",
                f"common.device={device}",
                "hydra.run.dir=.",
                "hydra.output_subdir=null",
            ],
        )
    return cfg


class _EnvSpecShim:
    """Duck-types SingleProcessEnv just enough for build_agent.

    build_agent reads: action_space, observation_space, modalities,
    num_actions, and (for vector tokenizer) observation_space[vector].shape.
    Constructing the real make_craftax chain once gives identical values
    without a second jax env instance in the stepping path.
    """

    def __init__(self) -> None:
        from envs.wrappers.craftax import make_craftax

        env = make_craftax()
        self.modalities = set(env.modalities)
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.num_actions = int(env.action_space.n)

    @property
    def unwrapped(self):  # pragma: no cover - parity with gym API
        return self


def install_meta_model_shims(agent) -> int:
    """Declare curiosity-head meta_model slots the checkpoint expects.

    The released Craftax.pt was trained before commit 492329c removed the
    vmap scaffolding: EnsembleObsHead then carried `meta_model`
    (`self._build_model().to('meta')`) alongside the live `ensemble`.
    meta_model participates in NO forward path — estimate_uncertainty and
    forward_all read self.ensemble exclusively (verified by
    test_instrumentation.py::test_jsd_matches_native_estimate_uncertainty).
    Shims are built with the head's own builder so keys/shapes match
    exactly, consume their checkpoint tensors during strict load, and are
    parked back on the meta device afterwards. Returns count installed.
    """
    installed = 0
    wm = agent.world_model
    if getattr(wm, "curiosity_head", None):
        for head in wm.curiosity_head.values():
            if not hasattr(head, "meta_model"):
                head.meta_model = head._build_model()
                installed += 1
    return installed


def park_meta_models(agent) -> None:
    for head in getattr(agent.world_model, "curiosity_head", {}).values():
        if hasattr(head, "meta_model"):
            head.meta_model = head.meta_model.to("meta")


def act_with_probs(agent, model_obs, temperature: float = 1.0) -> tuple[int, Any]:
    """Vendored Agent.act mirrored line-for-line, additionally returning the
    controller categorical probabilities at the final position (SOMA logs
    them per decision point). State advancement and process_action side
    effects are identical to agent.act.
    """
    import torch

    with torch.autocast(
        agent.device.type, dtype=torch.bfloat16, enabled=agent.use_bf16_autocast
    ):
        input_ac = agent._embed_obs(model_obs)
        actions_dist = agent.actor_critic(inputs=input_ac)[
            0
        ].get_actions_distributions(temperature)
        action = actions_dist.sample()[:, -1]
        probs = actions_dist.probs[:, -1]
        if agent.actor_critic.include_action_inputs:
            agent.actor_critic.process_action(action)
    return int(action.reshape(-1)[0].item()), probs.detach().float().cpu().numpy()


def load_agent_strict(agent, checkpoint: Path, device) -> dict[str, int]:
    """Vendored Agent.load equivalent (play_craftax flags: no tokenizer)
    plus meta-key accounting. The training-time .to('meta') scaffolding
    serialized its params WITHOUT storage, so torch.load reconstructs them
    as meta tensors. Those slots receive zero-filled stand-ins (identical
    to their trained content: nothing), strict validation still checks
    every key name and shape, and park_meta_models re-parks them.
    """
    import torch

    import sys

    src_dir = str(pins.runtime_dir() / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from utils import extract_state_dict  # vendored helper

    stats = {"zero_filled_meta_keys": 0}
    sd = torch.load(checkpoint, map_location=device, weights_only=True)

    wm_sd: dict[str, torch.Tensor] = {}
    for k, v in extract_state_dict(sd, "world_model").items():
        if isinstance(v, torch.Tensor) and v.is_meta:
            assert ".meta_model." in k, (
                f"unexpected dataless tensor outside meta_model scaffolding: {k}"
            )
            wm_sd[k] = torch.zeros(v.shape, dtype=v.dtype, device=device)
            stats["zero_filled_meta_keys"] += 1
        else:
            wm_sd[k] = v
    agent.world_model.load_state_dict(wm_sd)

    agent.actor_critic.load_state_dict(extract_state_dict(sd, "actor_critic"))
    return stats


def load_simulus_agent(device: str | None = None) -> tuple[Any, Any, Any]:
    """Returns (agent, cfg, torch_device). Process-wide singleton.

    Strict §0 load: torch weights_only=True plus default strict
    load_state_dict — every checkpoint key lands in a declared parameter
    slot (see install_meta_model_shims for the meta_model accounting).
    """
    with _LOCK:
        dev = device or resolve_device()
        key = f"agent:{dev}"
        if key in _CACHED:
            return _CACHED[key]

        import torch

        pins.bootstrap()
        cfg = _compose_cfg(dev)
        torch_device = torch.device(dev)

        # Import AFTER bootstrap — these are the vendored top-level modules.
        import main as simulus_main  # noqa: PLC0415  (vendored src/main.py)

        spec_env = _EnvSpecShim()
        agent = simulus_main.build_agent(spec_env, cfg, torch_device)
        n_shims = install_meta_model_shims(agent)
        stats = load_agent_strict(agent, checkpoint_path(), torch_device)
        park_meta_models(agent)
        agent.eval()

        _CACHED[key] = (agent, cfg, torch_device)
        from envs import SingleProcessEnv  # noqa: PLC0415
        from envs.wrappers.craftax import make_craftax  # noqa: PLC0415

        _CACHED.setdefault("make_craftax", make_craftax)
        _CACHED.setdefault("SingleProcessEnv", SingleProcessEnv)
        _CACHED["meta_shims_installed"] = n_shims
        _CACHED["load_stats"] = stats
        return _CACHED[key]
