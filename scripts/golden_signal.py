#!/usr/bin/env python
"""Task 1.9 — golden signal gate over backend/data/events.db.

The demo's money metric: exploration agents must REDUCE regional prediction
error. Reads the last 10 agent_completed events and reports the fraction
where error_after < error_before (delta > 0).

Exit 0 + "GOLDEN SIGNAL: PASS" when at least half the agents improved,
or when no agents have completed yet AND --allow-empty is given.
Exit 1 otherwise.
"""

import argparse
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_DB = Path("backend/data/events.db")
GOLDEN_QUERY = """
SELECT agent_id,
       printf('%.3f', error_before)                AS before_str,
       printf('%.3f', error_after)                 AS after_str,
       printf('%.3f', error_before - error_after)  AS delta_str,
       error_before,
       error_after
FROM events
WHERE event_type = 'agent_completed' AND error_before IS NOT NULL
ORDER BY id DESC
LIMIT 10
"""


def _fetch_rows(db_path: Path) -> list[tuple]:
    # Read-only URI: never create/lock the live db from a reporting script.
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return conn.execute(GOLDEN_QUERY).fetchall()
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--wait", type=int, default=0,
                        help="seconds to poll for agent_completed rows")
    parser.add_argument("--allow-empty", action="store_true",
                        help="PASS when no agents have completed yet")
    args = parser.parse_args()

    deadline = time.monotonic() + args.wait
    rows: list[tuple] = []
    while True:
        if args.db.exists():
            try:
                rows = _fetch_rows(args.db)
            except sqlite3.Error as exc:
                print(f"GOLDEN SIGNAL: FAIL — cannot read {args.db}: {exc}")
                return 1
        if rows or time.monotonic() >= deadline:
            break
        time.sleep(1.0)

    if not rows:
        if args.allow_empty:
            print("GOLDEN SIGNAL: PASS — no agents completed yet "
                  "(--allow-empty)")
            return 0
        print(f"GOLDEN SIGNAL: FAIL — no agent_completed events in {args.db}")
        return 1

    print(f"{'agent_id':<12} {'before':>8} {'after':>8} {'delta':>8}")
    positives = 0
    for agent_id, before_str, after_str, delta_str, _eb, ea in rows:
        improved = ea is not None and float(delta_str) > 0.0
        positives += improved
        print(f"{agent_id:<12} {before_str:>8} {after_str:>8} {delta_str:>8}")

    fraction = positives / len(rows)
    print(f"\nImproved: {positives}/{len(rows)} ({fraction:.0%})")
    if fraction >= 0.5:
        print("GOLDEN SIGNAL: PASS")
        return 0
    print("GOLDEN SIGNAL: FAIL — fewer than half of agents reduced error")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
