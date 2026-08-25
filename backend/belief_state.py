"""BeliefState — SOMA Task 1.3a.

Persistent structured representation of the observed simulation state,
plus the computed prediction error / confidence maps.

Spec: docs/specs/WORLD_MODEL.md "BELIEF STATE".
Invariants:
  - CIRCULAR INPUT INVARIANT (ARCHITECTURE.md): to_input_vector() must
    never include prediction_error_map or confidence_map.
  - FILE OWNERSHIP LAW (QUICK_REFERENCE.md): this module imports numpy,
    dataclasses, time only — zero backend imports.
"""

import time
from dataclasses import dataclass, field

import numpy as np

# Static quadrant regions: {name: (row_slice, col_slice)} over the 16x16 grid.
# Dynamic regions (tool_tissue_boundary, surgical_target_vicinity) are
# resolved from live belief positions — see region_to_slice().
REGION_SLICE_MAP: dict[str, tuple[slice, slice]] = {
    "upper_left": (slice(0, 8), slice(0, 8)),
    "upper_right": (slice(0, 8), slice(8, 16)),
    "lower_left": (slice(8, 16), slice(0, 8)),
    "lower_right": (slice(8, 16), slice(8, 16)),
}

_REGION_ORDER = (
    "upper_left",
    "upper_right",
    "lower_left",
    "lower_right",
    "tool_tissue_boundary",
    "surgical_target_vicinity",
)

_EMA_ALPHA = 0.3


@dataclass
class VesselBelief:
    """One vessel slot in grid units (col/row/radius in [0, 15])."""

    col: float
    row: float
    radius: float
    damaged: bool


@dataclass
class BeliefState:
    # Observation fields — copied from env.get_state_vector() each step.
    integrity: np.ndarray = field(
        default_factory=lambda: np.zeros((16, 16), dtype=np.float32)
    )
    vascularity: np.ndarray = field(
        default_factory=lambda: np.zeros((16, 16), dtype=np.float32)
    )
    bleeding: np.ndarray = field(
        default_factory=lambda: np.zeros((16, 16), dtype=bool)
    )
    ee_position: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    ee_quaternion: np.ndarray = field(
        default_factory=lambda: np.zeros(4, dtype=np.float32)
    )
    gripper_state: float = 0.0
    target_pos: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float32)  # (col, row)/15
    )
    target_reached: bool = False
    vessels: list[VesselBelief] = field(default_factory=list)
    active_dof: int = 1

    # Computed fields — NOT observation, NEVER network input.
    prediction_error_map: np.ndarray = field(
        default_factory=lambda: np.zeros((16, 16), dtype=np.float32)
    )
    confidence_map: np.ndarray = field(
        default_factory=lambda: np.ones((16, 16), dtype=np.float32)
    )
    # Real per-region JSD (Simulus) overrides geometric region_to_slice,
    # which needs SurRoL EE/target positions this substrate doesn't have.
    regional_override: dict[str, float] | None = None

    # Metadata.
    world_model_version: int = 0
    episode_count: int = 0
    last_updated: float = 0.0

    def update_from_observation(self, state_vec: np.ndarray) -> None:
        """Unpack env.get_state_vector() (806 floats) into structured fields.

        Index ranges per SIMULATION.md state vector table. Never touches
        prediction_error_map, confidence_map, or world_model_version.
        """
        assert state_vec.shape == (806,), (
            f"expected state_vec shape (806,), got {state_vec.shape}"
        )
        self.integrity = state_vec[0:256].reshape(16, 16).copy()
        self.vascularity = state_vec[256:512].reshape(16, 16).copy()
        self.bleeding = state_vec[512:768].reshape(16, 16).astype(bool)
        self.ee_position = state_vec[768:771].copy()
        self.ee_quaternion = state_vec[771:775].copy()
        self.gripper_state = float(state_vec[775])
        self.target_pos = state_vec[776:778].copy()  # normalized (col, row)
        self.target_reached = bool(state_vec[778] > 0.5)
        self._unpack_vessels(state_vec[779:799])
        self.active_dof = int(np.argmax(state_vec[799:805])) + 1
        # state_vec[805]: padding, ignored.
        self.last_updated = time.time()
        self.episode_count += 1

    def to_input_vector(self) -> np.ndarray:
        """Re-pack observation fields into the exact (806,) get_state_vector() order.

        CIRCULAR INPUT INVARIANT: must not include prediction_error_map or
        confidence_map; output is identical regardless of their values.
        """
        parts = [
            self.integrity.flatten(),                      # 256
            self.vascularity.flatten(),                    # 256
            self.bleeding.flatten().astype(np.float32),    # 256
            self.ee_position,                              # 3
            self.ee_quaternion,                            # 4
            np.array([self.gripper_state], dtype=np.float32),   # 1
            self.target_pos,                               # 2
            np.array([float(self.target_reached)], dtype=np.float32),  # 1
            self._pack_vessels(),                          # 20 (5 slots x 4)
            self._dof_onehot(),                            # 6
            np.zeros(1, dtype=np.float32),                 # 1 padding
        ]
        vec = np.concatenate(parts).astype(np.float32)
        assert vec.shape == (806,)
        return vec

    def update_prediction_error(
        self, predicted: np.ndarray, actual: np.ndarray
    ) -> None:
        """EMA update of the error map from predicted vs actual tissue integrity.

        Both inputs are (16, 16) float32. alpha=0.3: ~10 steps to respond to a
        sustained change (see WORLD_MODEL.md). Confidence is always 1 - error.
        """
        pointwise = (predicted - actual) ** 2
        self.prediction_error_map = (
            _EMA_ALPHA * pointwise + (1 - _EMA_ALPHA) * self.prediction_error_map
        ).astype(np.float32)
        self.confidence_map = 1.0 - self.prediction_error_map

    def set_jsd_error_map(self, jsd_map: np.ndarray) -> None:
        """Simulus substrate: EMA toward a real per-cell JSD map.

        Same alpha/decay semantics as update_prediction_error; the incoming
        map is already an error magnitude (not a squared residual), so it is
        blended directly instead of being squared.
        """
        assert jsd_map.shape == (16, 16), (
            f"expected (16, 16), got {jsd_map.shape}"
        )
        clipped = np.clip(jsd_map.astype(np.float32), 0.0, 1.0)
        self.prediction_error_map = (
            _EMA_ALPHA * clipped + (1 - _EMA_ALPHA) * self.prediction_error_map
        ).astype(np.float32)
        self.confidence_map = 1.0 - self.prediction_error_map

    def get_regional_errors(self) -> dict[str, float]:
        """Mean error per region — exactly the 6 keys the orchestrator expects."""
        if self.regional_override is not None:
            return {name: float(self.regional_override[name]) for name in _REGION_ORDER}
        m = self.prediction_error_map
        return {
            name: float(m[rs, cs].mean())
            for name in _REGION_ORDER
            for rs, cs in (region_to_slice(name, self),)
        }

    def snapshot(self) -> dict:
        """JSON-serializable dict for belief_snapshot events and /detail API."""
        return {
            "global_mean_error": float(self.prediction_error_map.mean()),
            "regional_errors": self.get_regional_errors(),
            "world_model_version": self.world_model_version,
            "active_dof": self.active_dof,
            "episode_count": self.episode_count,
            "error_map": self.prediction_error_map.tolist(),  # [[float]]x16x16
            "timestamp": self.last_updated,
        }

    # -- internals ----------------------------------------------------------

    def _dof_onehot(self) -> np.ndarray:
        v = np.zeros(6, dtype=np.float32)
        v[self.active_dof - 1] = 1.0  # active_dof in [1, 6] -> index [0, 5]
        return v

    def _unpack_vessels(self, raw: np.ndarray) -> None:
        """raw: 20 floats, 5 slots x (col/15, row/15, radius/15, damaged).

        Stored in grid units (x15) so _pack_vessels divides back out.
        Empty slots decode to VesselBelief(0, 0, 0, False) and repack to zeros.
        """
        self.vessels = [
            VesselBelief(
                col=float(raw[i * 4]) * 15.0,
                row=float(raw[i * 4 + 1]) * 15.0,
                radius=float(raw[i * 4 + 2]) * 15.0,
                damaged=bool(raw[i * 4 + 3] > 0.5),
            )
            for i in range(5)
        ]

    def _pack_vessels(self) -> np.ndarray:
        v = np.zeros(20, dtype=np.float32)
        for i, vessel in enumerate(self.vessels[:5]):
            base = i * 4
            v[base] = vessel.col / 15.0
            v[base + 1] = vessel.row / 15.0
            v[base + 2] = vessel.radius / 15.0
            v[base + 3] = float(vessel.damaged)
        return v


def region_to_slice(region: str, belief: "BeliefState") -> tuple[slice, slice]:
    """Resolve a region name to (row_slice, col_slice) on the 16x16 grid.

    Static quadrants come from REGION_SLICE_MAP; dynamic regions resolve to a
    3x3 window centered on the EE / target grid position, clipped to [1, 14]
    so the window always stays inside the grid.
    """
    if region in REGION_SLICE_MAP:
        return REGION_SLICE_MAP[region]
    if region == "tool_tissue_boundary":
        r = int(np.clip(belief.ee_position[1] * 15, 1, 14))
        c = int(np.clip(belief.ee_position[0] * 15, 1, 14))
    elif region == "surgical_target_vicinity":
        r = int(np.clip(belief.target_pos[1] * 15, 1, 14))
        c = int(np.clip(belief.target_pos[0] * 15, 1, 14))
    else:
        raise ValueError(f"unknown region: {region!r}")
    return slice(r - 1, r + 2), slice(c - 1, c + 2)
