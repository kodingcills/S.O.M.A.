"""tests/test_belief_state_simulus.py — JSD ingestion path (fast, numpy only)."""
import numpy as np
import pytest

from backend.belief_state import BeliefState

_REGIONS = (
    "upper_left",
    "upper_right",
    "lower_left",
    "lower_right",
    "tool_tissue_boundary",
    "surgical_target_vicinity",
)


def test_set_jsd_error_map_moves_toward_real_map():
    b = BeliefState()
    real = np.full((16, 16), 0.5, dtype=np.float32)
    b.set_jsd_error_map(real)
    assert float(b.prediction_error_map.mean()) == pytest.approx(0.3 * 0.5)
    assert float(b.confidence_map.mean()) == pytest.approx(1 - 0.3 * 0.5)


def test_set_jsd_error_map_clips_to_unit_range():
    b = BeliefState()
    wild = np.full((16, 16), 50.0, dtype=np.float32)
    b.set_jsd_error_map(wild)
    assert float(b.prediction_error_map.max()) == pytest.approx(0.3)


def test_regional_override_consulted():
    b = BeliefState()
    override = {name: float(i + 1) / 10 for i, name in enumerate(_REGIONS)}
    b.regional_override = dict(override)
    got = b.get_regional_errors()
    assert set(got) == set(_REGIONS)
    assert got["upper_left"] == pytest.approx(0.1)
    assert got["surgical_target_vicinity"] == pytest.approx(0.6)


def test_regional_errors_without_override_uses_geometry():
    b = BeliefState()
    b.regional_override = None
    got = b.get_regional_errors()
    assert set(got) == set(_REGIONS)
    assert all(v == pytest.approx(0.0) for v in got.values())


def test_circular_input_still_clean_after_jsd_updates():
    """CIRCULAR INPUT INVARIANT: error map changes must not alter inputs."""
    b = BeliefState()
    before = b.to_input_vector().copy()
    real = np.ones((16, 16), dtype=np.float32)
    b.set_jsd_error_map(real)
    b.regional_override = {name: 1.0 for name in _REGIONS}
    after = b.to_input_vector()
    assert np.array_equal(before, after)
