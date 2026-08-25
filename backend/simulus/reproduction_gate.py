"""§0 reproduction gate — nothing downstream begins before this passes.

Usage: python -m backend.simulus.reproduction_gate

Hard-pass criteria (contract §0):
  - strict checkpoint load succeeds (no ignored missing/unexpected params)
  - observation/action schemas constant over the smoke test
  - controller probs finite, nonnegative, sum≈1 at every probed decision point
  - ≥2 distinct argmax actions and ≥2 prob vectors with L1 distance > tol
  - pins/seeds/schemas recorded exactly in the reproduction manifest

The smoke trajectory runs the released controller (sampled, temperature 1.0)
on SomaCraftaxEnv — explicit-key functional craftax v1.4.5 with the vendored
pinned preprocessing. The manifest records that the checkpoint's original
training Craftax version could not be proven.
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

import backend.simulus as pins

MANIFEST_PATH = Path(__file__).resolve().parent.parent / "models" / "reproduction_manifest.json"
L1_TOLERANCE = 1e-6


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    import torch

    import jax

    from backend.simulus.craftax_env import SomaCraftaxEnv
    from backend.simulus.runtime import act_with_probs, load_simulus_agent

    manifest: dict = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_revision": pins.MODEL_REVISION,
        "code_commit": pins.VENDORED_COMMIT,
        "checkpoint_path": None,
        "checkpoint_sha256": None,
        "container_digest": "none-local-venv",
        "craftax_version": pins.CRAFTAX_VERSION,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "jax_version": jax.__version__,
        "hardware": platform.machine(),
        "device": None,
        "smoke_seeds": list(pins.SMOKE_SEEDS),
        "n_smoke": pins.N_SMOKE,
        "l1_tolerance": L1_TOLERANCE,
        "training_version_proven": False,
        "training_version_statement": (
            "The original training version could not be proven. The experiment "
            "uses the earliest official code/environment combination "
            "demonstrated to load and execute the released checkpoint."
        ),
        "wrapper_stack": [
            "raw craftax 1.4.5 (gymnax functional, explicit PRNGKey)",
            "CraftaxWrapper (vendored pinned class, verbatim)",
            "modality dict {map→token_2d(99,4), stats→vector(47), direction→token(1)}",
            "(SOMA SomaCraftaxEnv — obs equality vs pinned make_craftax chain "
             "verified by tests/test_craftax_env.py::test_obs_matches_pinned_chain)",
        ],
        "action_mapping": "craftax.craftax.constants.Action, 43 discrete, index=enum value",
        "observation_schema": {
            "map/token_2d": [99, 4],
            "stats/vector": [47],
            "direction/token": [1],
        },
        "gates": {},
    }

    # Gate 1 — checksum
    ckpt = Path(__file__).resolve().parent.parent / "models" / "Craftax.pt"
    sha = _sha256(ckpt)
    manifest["checkpoint_path"] = str(ckpt)
    manifest["checkpoint_sha256"] = sha
    g1 = sha == pins.CHECKPOINT_SHA256
    manifest["gates"]["checksum"] = g1
    print(f"Gate 1 checksum: {'PASS' if g1 else 'FAIL'} ({sha[:16]}…)")
    if not g1:
        return _fail(manifest)

    # Gate 2 — strict load (+ device)
    agent, cfg, device = load_simulus_agent()
    manifest["device"] = str(device)
    manifest["gates"]["strict_load"] = True
    print(f"Gate 2 strict_load: PASS (device={device})")

    # Gates 3-5 — seeded smoke trajectory under the sampled controller
    env = SomaCraftaxEnv(seed=int(pins.SMOKE_SEEDS[0]))
    all_actions: list[int] = []
    all_probs: list[np.ndarray] = []
    bad_points = 0
    schema_ok = True

    for seed in pins.SMOKE_SEEDS:
        obs = env.reset(seed=seed)
        agent.reset_actor_critic(n=1, burnin_observations=None, mask_padding=None)
        for _ in range(pins.N_SMOKE):
            if not (
                obs["token_2d"].shape == (99, 4)
                and obs["vector"].shape == (47,)
                and obs["token"].shape == (1,)
            ):
                schema_ok = False
            model_obs = env.to_model_obs(agent.device)
            a, probs = act_with_probs(agent, model_obs, temperature=1.0)
            if not 0 <= a < env.NUM_ACTIONS or not np.all(np.isfinite(probs)):
                bad_points += 1
                continue
            obs, _reward, done = env.step(a)
            all_actions.append(a)
            all_probs.append(probs.reshape(-1))
            if done:
                obs = env.reset(seed=seed + 1000)
                agent.reset_actor_critic(
                    n=1, burnin_observations=None, mask_padding=None
                )

    probs_arr = np.stack(all_probs) if all_probs else np.zeros((0, env.NUM_ACTIONS))
    finite_norm_bad = int(np.sum(
        ~np.isfinite(probs_arr).all(axis=1)
        | (probs_arr < 0).any(axis=1)
        | (np.abs(probs_arr.sum(axis=1) - 1.0) > 1e-3)
    )) if len(probs_arr) else 1

    distinct_argmax = len({int(p.argmax()) for p in all_probs})
    l1_max = max(
        (
            float(np.abs(probs_arr[i] - probs_arr[j]).sum())
            for i in range(len(probs_arr))
            for j in range(i + 1, len(probs_arr))
        ),
        default=0.0,
    )

    manifest["gates"]["schema_constant"] = schema_ok
    manifest["gates"]["valid_actions_no_nans_no_hangs"] = bad_points == 0
    manifest["gates"]["probs_finite_normalized_nonneg"] = finite_norm_bad == 0
    manifest["gates"]["diverse_actions_ge2_argmax"] = len(set(all_actions)) >= 2
    manifest["gates"]["diverse_probs_l1_gt_tol"] = bool(
        distinct_argmax >= 2 and l1_max > L1_TOLERANCE
    )
    manifest["smoke_decision_points"] = len(all_actions)
    manifest["smoke_distinct_actions"] = sorted(set(all_actions))
    manifest["probe_distinct_argmax"] = distinct_argmax
    manifest["probe_l1_max"] = l1_max

    ok = all(manifest["gates"].values())
    manifest["verdict"] = "PASS" if ok else "FAIL"
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["gates"], indent=2))
    print(f"REPRODUCTION GATE: {manifest['verdict']} → {MANIFEST_PATH}")
    return 0 if ok else 1


def _fail(manifest: dict) -> int:
    manifest["verdict"] = "FAIL"
    manifest["gates"].setdefault("checksum", False)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"REPRODUCTION GATE: FAIL → {MANIFEST_PATH}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
