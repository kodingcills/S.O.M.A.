"""SOMA ↔ Simulus bridge package.

Pinned artifacts (soma-research-contract v0.6.1):
  model revision : 6676919c643ec9e058dd7d00c6c0cea323c80f26 (HF leorc/Simulus)
  checkpoint SHA : fd1800f8b47cf1a2f7f22eebb60f604ad9b9fcb0d9e379f967060742ea5b73bc
  code commit    : c1e75e9c98699dfde47190b03420e12a519d7245 (vendored in
                   backend/simulus_runtime/, see VENDORED_COMMIT below)
  craftax        : 1.4.5

Import isolation: the vendored runtime uses top-level module names (utils,
models, envs, dataset, game) that must never collide with backend.*. The
bootstrap inserts the vendored src dir at sys.path FRONT only once; no
backend.* import may happen from inside those modules.
"""
from __future__ import annotations

import sys
from pathlib import Path

MODEL_REVISION = "6676919c643ec9e058dd7d00c6c0cea323c80f26"
CHECKPOINT_SHA256 = (
    "fd1800f8b47cf1a2f7f22eebb60f604ad9b9fcb0d9e379f967060742ea5b73bc"
)
VENDORED_COMMIT = "c1e75e9c98699dfde47190b03420e12a519d7245"
CRAFTAX_VERSION = "1.4.5"
SMOKE_SEEDS = [42, 137, 256, 512, 1024]
N_SMOKE = 100

_RUNTIME_DIR = Path(__file__).resolve().parent.parent / "simulus_runtime"
_SRC_DIR = _RUNTIME_DIR / "src"

_BOOTSTRAPPED = False


def bootstrap() -> Path:
    """Put the vendored Simulus source on sys.path (idempotent).

    Returns the vendored src directory. Raises if the vendored tree is
    missing so callers fail loudly instead of importing a stale copy.
    """
    global _BOOTSTRAPPED
    if not _SRC_DIR.is_dir():
        raise RuntimeError(
            f"vendored Simulus runtime missing at {_SRC_DIR} — "
            "backend/simulus_runtime/src must be committed alongside this file"
        )
    if not _BOOTSTRAPPED:
        # JAX_PLATFORMS=cpu mirrors the pinned wrapper's own side effect
        # (envs/wrappers/craftax.py sets it at import); set it first so the
        # very first jax import already respects it.
        import os

        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        sys.path.insert(0, str(_RUNTIME_DIR))  # for config/ relative lookup
        sys.path.insert(0, str(_SRC_DIR))
        _BOOTSTRAPPED = True
    return _SRC_DIR


def runtime_dir() -> Path:
    return _RUNTIME_DIR
