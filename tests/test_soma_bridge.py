"""tests/test_soma_bridge.py — pure mapping tests (fast, no torch/jax)."""
import numpy as np
import pytest

from backend.simulus.instrumentation import InstrumentedActionOutput
from backend.simulus.soma_bridge import (
    ACTION_REGIONS,
    achievements_to_dof,
    action_region,
    jsd_to_error_map,
    jsd_to_regional_errors,
)


def _outputs(jsd_values: list[float]) -> list[InstrumentedActionOutput]:
    return [
        InstrumentedActionOutput(action_idx=i, J_ua=float(v))
        for i, v in enumerate(jsd_values)
    ]


def test_partition_covers_exactly_43_disjoint():
    all_idx = sorted(i for idxs in ACTION_REGIONS.values() for i in idxs)
    assert all_idx == list(range(43))
    assert len(ACTION_REGIONS) == 6


def test_action_region_roundtrip():
    assert action_region(0) == "upper_left"
    assert action_region(5) == "upper_right"
    assert action_region(11) == "lower_left"
    assert action_region(24) == "lower_right"
    assert action_region(29) == "tool_tissue_boundary"
    assert action_region(18) == "surgical_target_vicinity"
    with pytest.raises(ValueError):
        action_region(43)


def test_regional_errors_are_means_of_group_jsd():
    jsd = [0.0] * 43
    jsd[0] = np.log(2)  # upper_left group gets 1.0 for one of seven members
    out = _outputs(jsd)
    regional = jsd_to_regional_errors(out)
    assert regional["upper_left"] == pytest.approx(np.log(2) / 7 / np.log(2))
    assert regional["lower_right"] == 0.0


def test_error_map_shape_range_and_quadrants():
    jsd = [np.log(2)] * 43  # everything maximally uncertain → map ≈ 1.0
    grid = jsd_to_error_map(_outputs(jsd))
    assert grid.shape == (16, 16)
    assert grid.dtype == np.float32
    assert float(grid.max()) <= 1.0
    assert float(grid.mean()) == pytest.approx(1.0, abs=1e-5)

    zero_map = jsd_to_error_map(_outputs([0.0] * 43))
    assert float(zero_map.max()) == 0.0

    # overlay regions paint peaks into the quadrant interior
    jsd2 = [0.0] * 43
    for i in ACTION_REGIONS["surgical_target_vicinity"]:
        jsd2[i] = np.log(2)
    g2 = jsd_to_error_map(_outputs(jsd2))
    assert g2[7, 7] == pytest.approx(1.0)          # inside overlay band
    assert g2[0, 0] == pytest.approx(0.0)          # outside it


def test_dof_mapping_bounds_and_buckets():
    z = np.zeros(67, dtype=bool)
    assert achievements_to_dof(z) == 1
    o = np.ones(67, dtype=bool)
    assert achievements_to_dof(o) == 6
    half = z.copy()
    half[:8] = True
    assert achievements_to_dof(half) == 3
