"""Maps Simulus per-action JSD onto SOMA's error-map / regional / DOF shapes.

Region names are FROZEN (orchestrator keys + frontend REGION_LABELS). The 43
Craftax actions are partitioned disjointly across the six regions; each
region's error = mean normalized JSD of its actions; the 16×16 map paints
the same values into quadrant blocks so WorldModelPanel renders real signal.
"""
from __future__ import annotations

import numpy as np

from backend.simulus.instrumentation import InstrumentedActionOutput

ACTION_REGIONS: dict[str, list[int]] = {
    "upper_left": [0, 1, 2, 3, 4, 6, 17],           # idle + movement + rest
    "upper_right": [5, 7, 8, 9, 10, 28],            # DO + placement
    "lower_left": [11, 12, 13, 14, 15, 16, 20, 21, 22, 23, 25, 38],  # crafting
    "lower_right": [24, 26, 27],                    # combat
    "tool_tissue_boundary": [29, 30, 31, 32, 33, 34],         # potions
    "surgical_target_vicinity": [18, 19, 35, 36, 37, 39, 40, 41, 42],
}

REGION_QUADRANT: dict[str, tuple[slice, slice]] = {
    "upper_left": (slice(0, 8), slice(0, 8)),
    "upper_right": (slice(0, 8), slice(8, 16)),
    "lower_left": (slice(8, 16), slice(0, 8)),
    "lower_right": (slice(8, 16), slice(8, 16)),
}

# Overlaid semantic bands painted after quadrants (np.maximum keeps peaks).
_REGION_OVERLAY: dict[str, tuple[slice, slice]] = {
    "tool_tissue_boundary": (slice(4, 12), slice(4, 12)),
    "surgical_target_vicinity": (slice(6, 10), slice(6, 10)),
}

_JSD_MAX = float(np.log(2.0))


def _normalized_jsd(outputs: list[InstrumentedActionOutput]) -> np.ndarray:
    j = np.array([o.J_ua for o in outputs], dtype=np.float32)
    return np.clip(j / (_JSD_MAX + 1e-8), 0.0, 1.0)


def jsd_to_regional_errors(
    outputs: list[InstrumentedActionOutput],
) -> dict[str, float]:
    j = _normalized_jsd(outputs)
    return {
        region: float(j[indices].mean())
        for region, indices in ACTION_REGIONS.items()
    }


def jsd_to_error_map(outputs: list[InstrumentedActionOutput]) -> np.ndarray:
    regional = jsd_to_regional_errors(outputs)
    grid = np.zeros((16, 16), dtype=np.float32)
    for region, (rows, cols) in REGION_QUADRANT.items():
        grid[rows, cols] = regional[region]
    for region, (rows, cols) in _REGION_OVERLAY.items():
        grid[rows, cols] = np.maximum(grid[rows, cols], regional[region])
    return grid


def achievements_to_dof(achievements: np.ndarray) -> int:
    n = int(np.asarray(achievements, dtype=bool).sum())
    return int(min(6, max(1, n // 4 + 1)))


def action_region(action_idx: int) -> str:
    for region, indices in ACTION_REGIONS.items():
        if action_idx in indices:
            return region
    raise ValueError(f"action {action_idx} outside partition")
