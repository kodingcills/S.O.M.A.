"""Tests for backend/belief_state.py — SOMA Task 1.3a.

Named tests per WORLD_MODEL.md TESTING PROTOCOL:
  test_circular_input         — CIRCULAR INPUT INVARIANT (ARCHITECTURE.md) — MUST PASS
  test_to_input_vector_shape  — (806,) float32 + observation round-trip
  test_ema_update             — EMA alpha=0.3 accumulation; perfect prediction stays ~0
  test_regional_errors_keys   — exact 6-key contract consumed by orchestrator
  test_snapshot_json          — JSON-serializable belief_snapshot payload
  test_region_to_slice        — exported region resolver for exploration agents
"""

import json

import numpy as np
import pytest

from backend.belief_state import BeliefState, VesselBelief, region_to_slice


def make_state_vec(seed: int = 0) -> np.ndarray:
    """Synthetic (806,) float32 vector matching the SIMULATION.md index table."""
    rng = np.random.RandomState(seed)
    v = np.zeros(806, dtype=np.float32)
    v[0:256] = rng.uniform(0.3, 1.0, 256).astype(np.float32)            # integrity
    v[256:512] = rng.uniform(0.0, 1.0, 256).astype(np.float32)          # vascularity
    v[512:768] = (rng.uniform(0, 1, 256) > 0.8).astype(np.float32)      # bleeding
    v[768:771] = rng.uniform(0.05, 0.95, 3).astype(np.float32)          # EE x,y,z
    v[771:775] = rng.uniform(-1.0, 1.0, 4).astype(np.float32)           # quaternion
    v[775] = 0.25                                                        # gripper
    v[776] = 0.4                                                         # target col/15
    v[777] = 0.6                                                         # target row/15
    v[778] = 0.0                                                         # not reached
    v[779:783] = np.array([10 / 15, 5 / 15, 2 / 15, 0.0], dtype=np.float32)
    v[783:787] = np.array([3 / 15, 12 / 15, 1 / 15, 1.0], dtype=np.float32)
    # slots 2-4 stay zero (empty vessels)
    v[799] = 1.0                                                         # DOF 1 active
    return v


def make_belief() -> BeliefState:
    b = BeliefState()
    b.update_from_observation(make_state_vec())
    return b


# ---------------------------------------------------------------------------
# test:belief:circular_input — THE killer test (ARCHITECTURE.md invariant)
# ---------------------------------------------------------------------------

def test_circular_input():
    belief = make_belief()
    vec_before = belief.to_input_vector().copy()
    belief.update_prediction_error(
        predicted=np.ones((16, 16), dtype=np.float32),
        actual=np.zeros((16, 16), dtype=np.float32),
    )
    assert np.array_equal(vec_before, belief.to_input_vector()), (
        "to_input_vector() changed after update_prediction_error() — "
        "CIRCULAR INPUT VIOLATED"
    )


# ---------------------------------------------------------------------------
# test:belief:to_input_vector_shape
# ---------------------------------------------------------------------------

def test_to_input_vector_shape():
    # Bare construction must work and produce (806,) float32.
    belief = BeliefState()
    vec = belief.to_input_vector()
    assert vec.shape == (806,)
    assert vec.dtype == np.float32

    # Round-trip: unpack a synthetic observation, repack — observation fields
    # must be recovered within float32 equality (vessel grid-unit round-trip
    # introduces <=1e-6 wobble; everything else is bit-exact).
    original = make_state_vec(seed=7)
    belief = BeliefState()
    belief.update_from_observation(original)
    repacked = belief.to_input_vector()
    np.testing.assert_allclose(repacked, original, rtol=0.0, atol=1e-6)

    # Structured fields landed where the spec says.
    assert belief.integrity.shape == (16, 16)
    assert belief.bleeding.dtype == np.bool_
    assert belief.active_dof == 1
    assert len(belief.vessels) == 5
    assert belief.vessels[0].col == pytest.approx(10.0, abs=1e-4)
    assert belief.vessels[1].damaged is True
    assert belief.target_reached is False


# ---------------------------------------------------------------------------
# test:belief:ema_update
# ---------------------------------------------------------------------------

def test_ema_update():
    ones = np.ones((16, 16), dtype=np.float32)
    zeros = np.zeros((16, 16), dtype=np.float32)

    # One call on a fresh belief: map == alpha * pointwise == 0.3 exactly.
    belief = BeliefState()
    belief.update_prediction_error(ones, zeros)
    np.testing.assert_allclose(belief.prediction_error_map, 0.3 * ones, atol=1e-7)

    # Ten sustained-error calls: EMA converges toward 1 (1 - 0.7^10 ~ 0.97).
    for _ in range(9):
        belief.update_prediction_error(ones, zeros)
    assert belief.prediction_error_map.mean() > 0.5

    # Perfect prediction (pred == actual) on a fresh belief: error stays ~0.
    fresh = BeliefState()
    fresh.update_prediction_error(zeros, zeros)
    assert fresh.prediction_error_map.mean() < 1e-6

    # Confidence map always mirrors the error map.
    assert np.allclose(fresh.confidence_map, 1.0 - fresh.prediction_error_map)


# ---------------------------------------------------------------------------
# test:belief:regional_errors_keys
# ---------------------------------------------------------------------------

def test_regional_errors_keys():
    belief = make_belief()
    rng = np.random.RandomState(3)
    pred = rng.uniform(0, 1, (16, 16)).astype(np.float32)
    actual = rng.uniform(0, 1, (16, 16)).astype(np.float32)
    belief.update_prediction_error(pred, actual)

    errors = belief.get_regional_errors()
    assert set(errors.keys()) == {
        "upper_left",
        "upper_right",
        "lower_left",
        "lower_right",
        "tool_tissue_boundary",
        "surgical_target_vicinity",
    }
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in errors.values())


# ---------------------------------------------------------------------------
# test:belief:snapshot_json
# ---------------------------------------------------------------------------

def test_snapshot_json():
    belief = make_belief()
    belief.world_model_version = 4
    snap = belief.snapshot()

    json.dumps(snap)  # must not raise — payload goes straight into SQLite/WS

    assert set(snap.keys()) == {
        "global_mean_error",
        "regional_errors",
        "world_model_version",
        "active_dof",
        "episode_count",
        "error_map",
        "timestamp",
    }
    error_map = snap["error_map"]
    assert isinstance(error_map, list) and len(error_map) == 16
    assert all(isinstance(row, list) and len(row) == 16 for row in error_map)
    assert snap["world_model_version"] == 4
    assert 0.0 <= snap["global_mean_error"] <= 1.0


# ---------------------------------------------------------------------------
# Exported helper used later by the exploration agent
# ---------------------------------------------------------------------------

def test_region_to_slice():
    belief = make_belief()

    # Static quadrants come straight from REGION_SLICE_MAP.
    rs, cs = region_to_slice("upper_left", belief)
    assert (rs, cs) == (slice(0, 8), slice(0, 8))

    # Dynamic regions resolve to a 3x3 window fully inside the 16x16 grid.
    for name in ("tool_tissue_boundary", "surgical_target_vicinity"):
        rs, cs = region_to_slice(name, belief)
        assert rs.stop - rs.start == 3 and cs.stop - cs.start == 3
        assert rs.start >= 0 and rs.stop <= 16
        assert cs.start >= 0 and cs.stop <= 16

    with pytest.raises(ValueError):
        region_to_slice("nonexistent", belief)


def test_vessel_belief_dataclass():
    v = VesselBelief(col=10.0, row=5.0, radius=2.0, damaged=False)
    assert (v.col, v.row, v.radius, v.damaged) == (10.0, 5.0, 2.0, False)
