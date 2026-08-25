#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["polars"]
# ///

# ─── How to run ───
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run directly (no venv, no pip install needed):
#      uv run scripts/verify_qp_labeling.py --db backend/data/events.db
# 3. Or make executable and run:
#      chmod +x scripts/verify_qp_labeling.py && ./scripts/verify_qp_labeling.py
# ──────────────────

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import polars as pl


def main() -> int:
    """Verify that craftax Q_P regret values are varying and observable."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=Path("backend/data/events.db"))
    args = parser.parse_args()
    try:
        with sqlite3.connect(args.db) as conn:
            rows = conn.execute(
                "SELECT payload FROM events WHERE event_type='craftax_step' "
                "AND json_extract(payload,'$.regret') IS NOT NULL "
                "ORDER BY id DESC LIMIT 20"
            ).fetchall()
        vals = [float(json.loads(row[0])["regret"]) for row in rows]
    except (sqlite3.Error, OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        print(f"FAIL: could not read regret values: {error}")
        return 1

    if len(vals) < 3:
        print(f"FAIL: insufficient regret rows ({len(vals)}; need at least 3)")
        return 1
    s = pl.Series("regret", vals)
    print(s.describe())
    harmful = s > 0.05
    print(f"harmful rate: {harmful.sum()}/{len(vals)} ({harmful.mean():.3f})")
    if s.n_unique() < 3:
        print("FAIL: regret not varying")
        return 1
    print("PASS: real Q_P regret flowing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
