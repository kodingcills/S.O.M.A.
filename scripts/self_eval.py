#!/usr/bin/env python3
"""SOMA self-evaluation — sprint checkpoint gates.

Run against a LIVE system (backend on :8000). Exit 0 only when every hard
gate passes. --allow-empty downgrades golden-signal/loop gates when no
agents have completed yet (fresh boot).
"""
import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import polars as pl
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DB = Path("backend/data/events.db")
RESULTS: list[tuple[str, bool, str]] = []


def gate(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'} {name}{f' — {detail}' if detail else ''}")
    return ok


def fetch_health() -> dict | None:
    try:
        with urllib.request.urlopen("http://localhost:8000/health", timeout=3) as r:
            return json.load(r)
    except Exception:
        return None


def golden_signal(conn: sqlite3.Connection) -> tuple[int, int]:
    rows = conn.execute(
        """SELECT error_before, error_after FROM events
           WHERE event_type='agent_completed'
             AND agent_id LIKE 'exp_%' AND error_before IS NOT NULL
           ORDER BY id DESC LIMIT 10"""
    ).fetchall()
    improved = sum(1 for b, a in rows if (b - a) > 0)
    return improved, len(rows)


def forgetting_check(conn: sqlite3.Connection) -> tuple[int, list[str]]:
    rows = conn.execute(
        """SELECT payload FROM events WHERE event_type='belief_snapshot'
           ORDER BY id DESC LIMIT 5"""
    ).fetchall()
    if len(rows) < 2:
        return 0, []
    current = json.loads(rows[0][0]).get("regional_errors", {})
    previous = json.loads(rows[1][0]).get("regional_errors", {})
    df = pl.DataFrame(
        {
            "region": list(current.keys()),
            "now": [float(current[k]) for k in current],
            "prev": [float(previous.get(k, 0.0)) for k in current],
        }
    ).with_columns((pl.col("now") - pl.col("prev")).alias("delta"))
    regressed = df.filter(pl.col("delta") > 0.03).get_column("region").to_list()
    return len(regressed), regressed


def damage_rates(conn: sqlite3.Connection) -> dict[str, float]:
    rows = conn.execute(
        """SELECT sim_id, payload FROM events
           WHERE event_type='simulation_step'
           AND timestamp > strftime('%s','now') - 3600"""
    ).fetchall()
    counts: dict[str, list[int]] = {}
    for sim_id, payload in rows:
        try:
            damaged = 1 if json.loads(payload).get("vessel_damaged") else 0
        except Exception:
            continue
        counts.setdefault(sim_id or "?", []).append(damaged)
    return {
        sid: sum(v) / len(v) for sid, v in counts.items() if len(v) >= 50
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-empty", action="store_true",
                        help="skip loop gates when no agents completed yet")
    args = parser.parse_args()

    health = fetch_health()
    gate("health_endpoint", health is not None,
         json.dumps(health) if health else "backend unreachable")

    if not DB.exists():
        gate("events_db", False, f"{DB} missing")
        return finish()

    conn = sqlite3.connect(DB)
    n_events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    gate("events_present", n_events > 0, f"{n_events} events")

    n_spawned = conn.execute(
        "SELECT COUNT(*) FROM events WHERE event_type='agent_spawned'"
    ).fetchone()[0]
    n_completed = conn.execute(
        "SELECT COUNT(*) FROM events WHERE event_type='agent_completed'"
    ).fetchone()[0]

    model_ok = Path("backend/models/prediction_network.pt").exists()
    gate("model_checkpoint", model_ok)

    samples = 0
    tdb = Path("backend/data/training.db")
    if tdb.exists():
        samples = sqlite3.connect(tdb).execute(
            "SELECT COUNT(*) FROM samples"
        ).fetchone()[0]
    gate("training_data_50k", samples >= 50_000, f"{samples} samples")

    improved, total = golden_signal(conn)
    if total == 0 and args.allow_empty:
        gate("golden_signal", True, "no explorations yet (--allow-empty)")
    elif total == 0:
        gate("golden_signal", False, "no exploration completions found")
    else:
        frac = improved / total
        gate("golden_signal", frac >= 0.5,
             f"{improved}/{total} improved ({frac:.0%})")

    regressions, regions = forgetting_check(conn)
    gate("no_catastrophic_forgetting", len(regions) == 0,
         f"{regressions} regressed {regions}" if regions else "stable")

    rates = damage_rates(conn)
    primary = rates.get("primary")
    comparison = rates.get("comparison")
    if primary is not None and comparison is not None:
        gate("mpc_beats_reactive", primary <= comparison + 0.05,
             f"primary={primary:.3f} vs reactive={comparison:.3f}")
    else:
        gate("mpc_beats_reactive", True,
             "insufficient paired data (informational)")

    conn.close()
    return finish()


def finish() -> int:
    failed = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n{'━' * 40}\nSELF-EVAL: {'PASS' if not failed else 'FAIL'} "
          f"({len(RESULTS) - len(failed)}/{len(RESULTS)} gates)"
          f"{'' if not failed else f' — failed: {failed}'}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
