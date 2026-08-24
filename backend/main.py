"""SOMA backend entrypoint — Task 1.1 minimal re-export.

Makes `app` importable two ways:
  - repo root:   uvicorn backend.main:app   → `backend.api` resolves
  - cwd=backend: uvicorn main:app           → top-level `api` resolves

Full lifespan (sims, world model, orchestrator) arrives in Task 1.9.
"""

try:
    from backend.api import app
except ImportError:  # cwd=backend direct-run mode
    from api import app  # type: ignore[no-redef]

__all__ = ["app"]
