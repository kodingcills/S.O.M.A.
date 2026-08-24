#!/usr/bin/env python
"""Task 1.3b CLI — launch training-data collection.

Usage: uv run python scripts/collect_data.py [--n 50000] [--seed 0]

DATA COLLECTION SEED=0 (deliberately different from operational seed 42 —
see docs/specs/SIMULATION.md "DATA COLLECTION").
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.data_collector import collect_training_data
from backend.simulation import SimConfig, SurROLTissueEnv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(
        f"Collecting {args.n} samples "
        f"(SimConfig(seed={args.seed}, dof=1, n_vessels=2))",
        flush=True,
    )
    env = SurROLTissueEnv(SimConfig(seed=args.seed, dof=1, n_vessels=2))
    try:
        total = collect_training_data(env, n_samples=args.n)
    finally:
        env.close()
    print(f"COLLECTION DONE: {total} samples", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
