#!/usr/bin/env python
"""Task 1.3b — polars quality gate over backend/data/training.db.

Samples ≤2000 random rows, computes per-row metrics with numpy, summarizes
with polars, and enforces the RISKS D-8 gates:

  bad_rate < 0.01          (shape / NaN / range / one-hot violations)
  mean_episode_step > 15   (no early-episode dominance)
  max tissue_mean > 0.5    (healthy-tissue states present)

Exit 0 + "DATA QUALITY: PASS" on success; exit 1 with reasons on fail.
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_DB = Path("backend/data/training.db")
SAMPLE_ROWS = 2000


def _row_metrics(blob: bytes, episode_step: int) -> dict[str, object]:
    s = np.frombuffer(blob, dtype=np.float32)
    if s.size != 806:
        return {
            "bad_shape": True,
            "has_nan": False,
            "bad_range": False,
            "bad_onehot": False,
            "episode_step": episode_step,
            "tissue_mean": 0.0,
        }
    tissue = s[0:768]
    return {
        "bad_shape": False,
        "has_nan": bool(np.isnan(s).any()),
        "bad_range": bool(((tissue < -0.1) | (tissue > 1.1)).any()),
        "bad_onehot": bool(abs(float(s[799:805].sum()) - 1.0) > 0.01),
        "episode_step": int(episode_step),
        "tissue_mean": float(s[0:256].mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--rows", type=int, default=SAMPLE_ROWS)
    args = parser.parse_args()

    if not args.db.exists():
        print(f"DATA QUALITY: FAIL — db not found: {args.db}")
        return 1

    conn = sqlite3.connect(args.db)
    try:
        rows = conn.execute(
            "SELECT state, episode_step FROM samples "
            "ORDER BY RANDOM() LIMIT ?",
            (args.rows,),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        print("DATA QUALITY: FAIL — no samples in db")
        return 1

    df = pl.DataFrame(
        [_row_metrics(blob, step) for blob, step in rows],
        schema_overrides={"episode_step": pl.Int64},
    )

    df = df.with_columns(
        bad=pl.any_horizontal(
            "bad_shape", "has_nan", "bad_range", "bad_onehot"
        )
    )
    print(df.describe())
    print(df.group_by("bad").len())

    bad_rate = float(df["bad"].mean())
    mean_step = float(df["episode_step"].mean())
    max_tissue = float(df["tissue_mean"].max())
    n = len(df)
    print(
        f"\nGate inputs over {n} rows: bad_rate={bad_rate:.4f} "
        f"mean_episode_step={mean_step:.1f} max_tissue_mean={max_tissue:.3f}"
    )

    failures: list[str] = []
    if bad_rate >= 0.01:
        failures.append(f"bad_rate {bad_rate:.4f} >= 0.01")
    if mean_step <= 15:
        failures.append(f"mean_episode_step {mean_step:.1f} <= 15 (D-8 trap)")
    if max_tissue <= 0.5:
        failures.append(f"max tissue_mean {max_tissue:.3f} <= 0.5")

    if failures:
        print("DATA QUALITY: FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("DATA QUALITY: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
