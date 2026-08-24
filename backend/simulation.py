"""SOMA Task 1.2 — SurROLTissueEnv surgical simulation environment.

Produces the locked 806-float32 state vector (STATE VECTOR DIMENSION
INVARIANT, ARCHITECTURE.md). Subclasses SurRoL's PsmEnv — the vendor
package is never modified; all adaptation happens via hook overrides.

Coordinate frames (SIMULATION.md "COORDINATE SYSTEM"):
  - Grid:      col, row in [0, 15] — tissue array indexing.
  - Canonical: x, y in [-0.1, 0.1] m, z in [0, 0.3] m — the frame every
    spec formula (state vector normalization, grid conversion, vessel
    proximity) operates on.
  - World:     PyBullet world frame. The stock PSM workspace lives at
    x in [0.50, 0.60], y in [-0.05, 0.05], z in [0.675, 0.745]
    (PsmEnv.WORKSPACE_LIMITS1). We map world <-> canonical affinely so
    the full grid stays reachable by construction (S-3 mitigation:
    _set_action clips tip targets into these exact limits).

Deviations from SIMULATION.md (documented for MASTER_STATE.md):
  1. TaskBase does not exist upstream; we subclass PsmEnv (SurRoLGoalEnv
     -> SurRoLEnv). Hook mapping table in module docstring bottom.
  2. action_to_surrol() is a pure module function taking `dof` explicitly
     (matches RISKS.md I-6 diagnostic usage), returning the 7-vector
     [dxyz(3), drpy(3), gripper(1)]; _set_action consumes it with
     ACTION_MODE='yaw' (only drpy[2] actuates rotation).
  3. step() accepts both the SOMA 12-vector (mode-aware) and a raw
     7-vector (mode inferred: all-zero -> WAIT, else MOVE).
  4. No events are written from this module (dependency-graph cleanliness;
     event emission belongs to agents/mpc per ARCHITECTURE.md).

Hook mapping (upstream PsmEnv -> this class):
  __init__            -> config + RandomState stored before super()
  _env_setup          -> tissue arrays, vessels, target, PyBullet bodies
  _sample_goal        -> target workspace position (goal-env contract)
  _get_obs            -> parent dict obs (observation/achieved/desired)
  _set_action         -> 7-vector PSM control (yaw mode)
  compute_reward/_step_callback inherited no-ops; reward computed in step()
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import pkgutil
import types
from dataclasses import dataclass

import numpy as np
import pybullet as p
from surrol.tasks.psm_env import PsmEnv
from surrol.utils.pybullet_utils import get_link_pose, wrap_angle
from surrol.utils.robotics import get_euler_from_matrix, get_matrix_from_euler

# ---------------------------------------------------------------------------
# Locked constants (QUICK_REFERENCE.md — change one, update that file)
# ---------------------------------------------------------------------------

GRID_SIZE = 16
GRID_MAX_IDX = 15.0
CANONICAL_XY_HALF = 0.1          # canonical workspace x,y in [-0.1, 0.1]
CANONICAL_Z_RANGE = 0.3          # canonical z in [0, 0.3]
CONTACT_SCALE_FACTOR = 50.0      # empirical Newtons -> [0, 1]
BLEED_DAMAGE_PER_STEP = 0.01
CAUTERIZE_INTEGRITY_GAIN = 0.1
MAX_GRID_DIST = 21.213           # sqrt(15^2 + 15^2), hardcoded per spec

REWARD_W_PROGRESS = 1.0
REWARD_W_DAMAGE = 2.0
REWARD_W_BLEEDING = 5.0
REWARD_W_STEP = 0.1
REWARD_W_COMPLETION = 10.0
REWARD_W_FAILURE = 50.0

SAFE_ZONE = (2.0, 14.0)          # vessel col/row sampling range
MIN_VESSEL_SEPARATION = 3.0      # cells, Euclidean
VESSEL_ATTEMPTS = 100
TARGET_ZONE = (3.0, 13.0)        # target col/row sampling range
MIN_TARGET_VESSEL_DIST = 3.0     # relaxed to 2.0 after first 200 attempts
MIN_TARGET_START_DIST = 5.0      # cells from instrument start
TARGET_ATTEMPTS = 200
GRID_START_POS = (1.0, 8.0)      # approx grid coords of EE start
TARGET_RADIUS = 1.5              # cells; reach threshold

MAX_VESSEL_SLOTS = 5             # fixed slot count in state vector
TIP_COLLISION_MARGIN_M = 0.002   # +2mm tip radius for vessel damage check
JAW_OPEN_ANGLE_RAD = np.deg2rad(40.0)

# Action mode indices into the 12-vector one-hot block [7:11]
MODE_MOVE = 0
MODE_GRASP = 1
MODE_CAUTERIZE = 2
MODE_WAIT = 3

# S-2: PyBullet keeps a global physics-server registry; concurrent env
# construction corrupts world IDs. All construction must go through
# create_env(), which serializes on this lock.
_pybullet_init_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Config + placement dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SimConfig:
    seed: int = 42       # SEED REPRODUCIBILITY INVARIANT — RandomState only
    dof: int = 1         # active DOF level, 1-6
    max_steps: int = 200
    n_vessels: int = 2   # major vessels to place, 0-5
    difficulty: int = 2  # >=4 enables autonomous bleeding spread


@dataclass
class Vessel:
    """Major vessel placed on the tissue grid.

    col/row are float grid units; radius_workspace is metres in the
    canonical workspace frame.
    """

    col: float
    row: float
    radius_workspace: float
    damaged: bool = False

    @property
    def workspace_pos(self) -> np.ndarray:
        """Canonical-frame position on the tissue plane (z=0)."""
        return np.array([
            (self.col / GRID_MAX_IDX) * 0.2 - CANONICAL_XY_HALF,
            (self.row / GRID_MAX_IDX) * 0.2 - CANONICAL_XY_HALF,
            0.0,
        ])


@dataclass
class Target:
    """Surgical target on the tissue grid."""

    grid_pos: np.ndarray        # (2,) float32 — (col, row)
    reached: bool = False
    radius: float = TARGET_RADIUS


# ---------------------------------------------------------------------------
# Pure action translation (12-float SOMA vector -> 7-float SurRoL vector)
# ---------------------------------------------------------------------------

def action_to_surrol(action_vec: np.ndarray, dof: int) -> np.ndarray:
    """Translate a SOMA 12-action vector into SurRoL's 7-action vector.

    Layout out: [dx, dy, dz, droll, dpitch, dyaw, gripper], all scaled by
    magnitude[11]; DOF gating zeroes inactive components; WAIT returns
    zeros(7); CAUTERIZE zeroes xyz+rpy and keeps the gripper channel.
    """
    vec = np.asarray(action_vec, dtype=np.float32).ravel()
    if vec.size != 12:
        raise ValueError(f"action_to_surrol expects 12 floats, got {vec.size}")

    magnitude = float(np.clip(vec[11], 0.0, 1.0))
    delta_xyz = vec[0:3].copy() * magnitude
    delta_rpy = vec[3:6].copy() * magnitude
    gripper = vec[6:7].copy()
    mode = int(np.argmax(vec[7:11]))

    if mode == MODE_WAIT:
        return np.zeros(7, dtype=np.float32)
    if mode == MODE_CAUTERIZE:
        delta_xyz[:] = 0.0
        delta_rpy[:] = 0.0

    if dof < 2:
        delta_xyz[0] = 0.0   # no x movement below DOF 2
    if dof < 3:
        delta_rpy[:] = 0.0   # no rotation below DOF 3
    if dof < 4:
        gripper[:] = 0.0     # no gripper below DOF 4

    return np.concatenate([delta_xyz, delta_rpy, gripper]).astype(np.float32)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SurROLTissueEnv(PsmEnv):
    """PSM surgical env with a 16x16 numpy tissue overlay.

    The tissue grid exists purely in numpy; a thin static PyBullet box
    under the PSM workspace provides real collision contact. Vessel and
    goal markers are visual-only bodies (no collision) so they never
    interfere with contact detection.
    """

    def __init__(self, config: SimConfig):
        self.config = config
        # SEED REPRODUCIBILITY INVARIANT: instance RNG, never np.random.seed
        self._rng = np.random.RandomState(config.seed)

        self.integrity = np.ones((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        self.vascularity = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        self.bleeding = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        self.elasticity = np.ones((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        self.vessels: list[Vessel] = []
        self.target = Target(grid_pos=np.zeros(2, dtype=np.float32))

        self.tissue_plane_id: int | None = None
        self._vessel_body_ids: list[int] = []
        self._step_count = 0
        self._last_reward = 0.0
        self._last_action_mode = MODE_MOVE
        self._last_contact_force = 0.0
        self._completion_rewarded = False
        self._failure_rewarded = False
        self._task_failed = False

        # PyBullet allows exactly ONE in-process GUI connection; SOMA runs
        # 6-8 concurrent envs, so DIRECT is the only viable default. Override
        # with SOMA_RENDER_MODE=human for single-instance visual debugging.
        render_mode = os.environ.get("SOMA_RENDER_MODE", "direct")
        super().__init__(render_mode=render_mode)

    # -- lifecycle ---------------------------------------------------------

    def reset(self):
        # S-5: _env_setup rebuilds tissue on every reset, so the RNG must be
        # re-seeded here or episode 2 would get different anatomy.
        self._rng = np.random.RandomState(self.config.seed)
        return super().reset()

    def close(self) -> None:
        try:
            super().close()
        except p.error:
            pass  # already disconnected — idempotent

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001 — interpreter shutdown guard
            pass

    # -- world construction hooks ------------------------------------------

    def _env_setup(self):
        super()._env_setup()  # camera, PSM1, table, goal sphere (vendor)
        self._init_tissue()
        self._place_vessels()
        self._place_target()
        self._build_tissue_bodies()
        self._step_count = 0
        self._last_reward = 0.0
        self._last_action_mode = MODE_MOVE
        self._last_contact_force = 0.0
        self._completion_rewarded = False
        self._failure_rewarded = False
        self._task_failed = False

    def _init_tissue(self):
        """Exact RNG consumption order (SIMULATION.md): 6 + 256 floats."""
        rng = self._rng

        # Step 1 — vascularity: 3 Gaussians, A=0.8, sigma=2.0 (6 floats)
        centers = [(rng.uniform(2.5, 13.5), rng.uniform(2.5, 13.5))
                   for _ in range(3)]
        rows, cols = np.mgrid[0:GRID_SIZE, 0:GRID_SIZE]
        vascularity = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        for cx, cy in centers:
            vascularity += 0.8 * np.exp(
                -((rows - cy) ** 2 + (cols - cx) ** 2) / 8.0)
        self.vascularity = np.clip(vascularity, 0.0, 1.0).astype(np.float32)

        # Step 2 — elasticity: 256 floats, row-major
        noise = rng.normal(0.0, 0.1, size=(GRID_SIZE, GRID_SIZE)).astype(np.float32)
        self.elasticity = np.clip(1.0 + noise, 0.5, 1.5).astype(np.float32)

        # Mutable episode state
        self.integrity = np.ones((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        self.bleeding = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)

    def _place_vessels(self):
        """Rejection sampling; consumes 2 floats per attempt + 1 per accept."""
        placed: list[Vessel] = []
        for _ in range(self.config.n_vessels):
            for _attempt in range(VESSEL_ATTEMPTS):
                col = self._rng.uniform(*SAFE_ZONE)
                row = self._rng.uniform(*SAFE_ZONE)
                if all(np.hypot(col - v.col, row - v.row) >= MIN_VESSEL_SEPARATION
                       for v in placed):
                    radius = self._rng.uniform(0.003, 0.007)
                    placed.append(Vessel(col=float(col), row=float(row),
                                         radius_workspace=float(radius)))
                    break
            else:
                break  # crowded — ship fewer vessels rather than loop forever
        self.vessels = placed

    def _place_target(self):
        """Target rejection sampling; relaxes vessel distance once (spec)."""
        col = row = None
        for min_vessel_dist in (MIN_TARGET_VESSEL_DIST, MIN_TARGET_VESSEL_DIST - 1.0):
            for _attempt in range(TARGET_ATTEMPTS):
                col = self._rng.uniform(*TARGET_ZONE)
                row = self._rng.uniform(*TARGET_ZONE)
                vessel_ok = all(
                    np.hypot(col - v.col, row - v.row) >= min_vessel_dist
                    for v in self.vessels)
                start_ok = np.hypot(col - GRID_START_POS[0],
                                    row - GRID_START_POS[1]) >= MIN_TARGET_START_DIST
                if vessel_ok and start_ok:
                    self.target = Target(
                        grid_pos=np.array([col, row], dtype=np.float32))
                    return
        # Deterministic last-sample fallback after both passes fail.
        self.target = Target(grid_pos=np.array([col, row], dtype=np.float32))

    def _build_tissue_bodies(self):
        """Static tissue plane (collidable) + visual-only vessel markers."""
        lim = self.workspace_limits1  # (3, 2) world-frame min/max per axis
        center_x = float(lim[0].mean())
        center_y = float(lim[1].mean())
        half_x = float(lim[0, 1] - lim[0, 0]) / 2 + 0.04
        half_y = float(lim[1, 1] - lim[1, 0]) / 2 + 0.04
        plane_top_z = float(lim[2, 0]) + 0.010  # just above lowest reachable tip

        plane_shape = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[half_x, half_y, 0.001])
        self.tissue_plane_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=plane_shape,
            basePosition=[center_x, center_y, plane_top_z - 0.001])
        p.changeVisualShape(self.tissue_plane_id, -1,
                            rgbaColor=[0.85, 0.70, 0.55, 0.55])

        self._vessel_body_ids = []
        for vessel in self.vessels:
            wx, wy = self._grid_to_world(vessel.col, vessel.row)
            visual = p.createVisualShape(
                p.GEOM_SPHERE, radius=vessel.radius_workspace,
                rgbaColor=[0.75, 0.10, 0.10, 0.9])
            body = p.createMultiBody(
                baseMass=0.0,
                baseVisualShapeIndex=visual,
                baseCollisionShapeIndex=-1,
                basePosition=[wx, wy, plane_top_z + vessel.radius_workspace])
            self._vessel_body_ids.append(body)

    # -- coordinate mapping --------------------------------------------------

    def _grid_to_world(self, col: float, row: float) -> tuple[float, float]:
        lim = self.workspace_limits1
        wx = lim[0, 0] + (col / GRID_MAX_IDX) * (lim[0, 1] - lim[0, 0])
        wy = lim[1, 0] + (row / GRID_MAX_IDX) * (lim[1, 1] - lim[1, 0])
        return float(wx), float(wy)

    def _world_to_canonical(self, pos_world: np.ndarray) -> np.ndarray:
        """Affine world -> canonical mapping (see module docstring)."""
        lim = self.workspace_limits1
        cx = ((pos_world[0] - lim[0, 0]) / (lim[0, 1] - lim[0, 0])) \
            * 0.2 - CANONICAL_XY_HALF
        cy = ((pos_world[1] - lim[1, 0]) / (lim[1, 1] - lim[1, 0])) \
            * 0.2 - CANONICAL_XY_HALF
        cz = np.clip(
            (pos_world[2] - lim[2, 0]) / (lim[2, 1] - lim[2, 0]),
            0.0, 1.0) * CANONICAL_Z_RANGE
        return np.array([cx, cy, cz], dtype=np.float64)

    def _get_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Tip pose in world frame (same link parent _get_obs uses)."""
        pos, orn = get_link_pose(self.psm1.body, self.psm1.TIP_LINK_INDEX)
        return np.array(pos), np.array(orn)

    def _get_ee_workspace_pos(self) -> np.ndarray:
        """Tip position in the canonical spec frame."""
        pos, _ = self._get_ee_pose()
        return self._world_to_canonical(pos)

    def _ee_grid_pos(self) -> tuple[float, float]:
        """Continuous (col_f, row_f) grid coordinates of the tip."""
        ws = self._get_ee_workspace_pos()
        col_f = (ws[0] + CANONICAL_XY_HALF) / 0.2 * GRID_MAX_IDX
        row_f = (ws[1] + CANONICAL_XY_HALF) / 0.2 * GRID_MAX_IDX
        return float(col_f), float(row_f)

    # -- goal-env contract ---------------------------------------------------

    def _sample_goal(self) -> np.ndarray:
        """Goal-env contract: target position in WORLD frame (the parent
        moves the visual goal sphere here and compares against the tip)."""
        wx, wy = self._grid_to_world(self.target.grid_pos[0],
                                     self.target.grid_pos[1])
        plane_top_z = float(self.workspace_limits1[2, 0]) + 0.010
        return np.array([wx, wy, plane_top_z], dtype=np.float64)

    def _get_obs(self) -> dict:
        return super()._get_obs()

    @property
    def action_size(self) -> int:
        return 7  # [dx, dy, dz, droll, dpitch, dyaw, gripper]

    def _set_action(self, action: np.ndarray):
        """Apply our 7-vector via the PSM (yaw mode, parent kinematics)."""
        action = np.asarray(action, dtype=np.float64).flatten().copy()
        if action.size != 7:
            raise ValueError(f"_set_action expects 7 floats, got {action.size}")

        # Parent drives yaw only (ACTION_MODE='yaw'); gripper semantics:
        # ours 1=closed -> parent negative command closes.
        action5 = np.array([action[0], action[1], action[2],
                            action[5], 1.0 - 2.0 * float(action[6])])

        action5[:3] *= 0.01 * self.SCALING  # position, max 1cm per step
        pose_world = self.psm1.pose_rcm2world(self.psm1.get_current_position())
        limits = self.workspace_limits1
        pose_world[:3, 3] = np.clip(
            pose_world[:3, 3] + action5[:3],
            limits[:, 0] - [0.02, 0.02, 0.0],
            limits[:, 1] + [0.02, 0.02, 0.08])

        rot = get_euler_from_matrix(pose_world[:3, :3])
        action5[3] *= np.deg2rad(30)  # yaw, max 30deg per step
        rot = (self.psm1_eul[0], self.psm1_eul[1],
               wrap_angle(rot[2] + action5[3]))
        pose_world[:3, :3] = get_matrix_from_euler(rot)

        self.psm1.move(self.psm1.pose_world2rcm(pose_world))
        if action5[4] < 0:
            self.psm1.close_jaw()
        else:
            self.psm1.move_jaw(JAW_OPEN_ANGLE_RAD)

    # -- gym loop -------------------------------------------------------------

    def step(self, action: np.ndarray):
        """Advance one episode step.

        Accepts the SOMA 12-vector (mode-aware, converted with the active
        DOF) or a raw 7-vector (all-zero -> WAIT, else MOVE). Returns the
        parent-format (obs, reward, done, info); reward is OUR dense reward
        recomputed after tissue effects (parent's sparse goal reward is
        stale by then and discarded).
        """
        action7, mode = self._normalize_action(action)
        self._last_action_mode = mode

        obs, _, _, info = super().step(action7)  # robot move + physics
        self._step_count += 1

        self.apply_tissue_effects(mode)
        self._check_vessel_damage()
        self._update_target_reached()
        self._propagate_bleeding()

        reward = self._compute_reward()
        self._last_reward = reward

        success = self._check_success()
        failure = self._check_failure()
        if failure:
            self._task_failed = True

        info = dict(info)
        info.update(
            task_success=bool(success),
            task_failed=bool(failure),
            step=self._step_count,
            contact_force=self._last_contact_force,
        )
        return obs, float(reward), bool(success or failure), info

    def _normalize_action(self, action: np.ndarray) -> tuple[np.ndarray, int]:
        vec = np.asarray(action, dtype=np.float32).ravel()
        if vec.size == 12:
            mode = int(np.argmax(vec[7:11]))
            return action_to_surrol(vec, self.config.dof), mode
        if vec.size == 7:
            mode = MODE_WAIT if not np.any(vec) else MODE_MOVE
            return vec.copy(), mode
        raise ValueError(f"step() expects a 12- or 7-vector, got {vec.size}")

    # -- tissue dynamics -------------------------------------------------------

    def apply_tissue_effects(self, mode: int) -> None:
        """Contact damage / cauterization at the current EE cell."""
        if mode == MODE_WAIT:
            self._last_contact_force = 0.0
            return

        row_i, col_j = self._ee_grid_cell()

        if mode == MODE_CAUTERIZE:
            # Contact not required — thermally precise (SIMULATION.md).
            self.bleeding[row_i, col_j] = False
            self.integrity[row_i, col_j] = min(
                1.0, self.integrity[row_i, col_j] + CAUTERIZE_INTEGRITY_GAIN)
            self._last_contact_force = 0.0
            return

        contacts = p.getContactPoints(self.psm1.body, self.tissue_plane_id)
        force = 0.0
        if contacts:
            total_newtons = sum(point[9] for point in contacts)
            force = min(1.0, total_newtons / CONTACT_SCALE_FACTOR)
        self._last_contact_force = force
        if force <= 0.0:
            return

        # Primary cell
        self.integrity[row_i, col_j] = max(
            0.0, self.integrity[row_i, col_j] - force * 0.3)
        if self.vascularity[row_i, col_j] > 0.6:
            self.bleeding[row_i, col_j] = True

        # 8 neighbours, decay by primary-cell elasticity
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni, nj = row_i + di, col_j + dj
                if 0 <= ni < GRID_SIZE and 0 <= nj < GRID_SIZE:
                    dist = np.sqrt(di * di + dj * dj)
                    decay = np.exp(-dist * self.elasticity[row_i, col_j])
                    self.integrity[ni, nj] = max(
                        0.0, self.integrity[ni, nj] - force * 0.1 * decay)

    def _ee_grid_cell(self) -> tuple[int, int]:
        col_f, row_f = self._ee_grid_pos()
        row_i = int(np.clip(row_f, 0, GRID_MAX_IDX))
        col_j = int(np.clip(col_f, 0, GRID_MAX_IDX))
        return row_i, col_j

    def _check_vessel_damage(self) -> None:
        ee_ws = self._get_ee_workspace_pos()
        for vessel in self.vessels:
            if vessel.damaged:
                continue
            dist = np.linalg.norm(ee_ws[:2] - vessel.workspace_pos[:2])
            if dist < vessel.radius_workspace + TIP_COLLISION_MARGIN_M:
                vessel.damaged = True

    def _update_target_reached(self) -> None:
        col_f, row_f = self._ee_grid_pos()
        dist = np.hypot(col_f - self.target.grid_pos[0],
                        row_f - self.target.grid_pos[1])
        if dist <= self.target.radius:
            self.target.reached = True

    def _propagate_bleeding(self):
        # BLEEDING DOUBLE-BUFFER INVARIANT: read self.bleeding, write next,
        # atomic swap. A view here makes results iteration-order dependent.
        next_bleeding = self.bleeding.copy()
        assert next_bleeding.base is None, "double-buffer copy required"

        if self.config.difficulty >= 4:  # autonomous spread at high difficulty
            for i in range(GRID_SIZE):
                for j in range(GRID_SIZE):
                    if not self.bleeding[i, j]:
                        continue
                    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < GRID_SIZE and 0 <= nj < GRID_SIZE \
                                and self.vascularity[ni, nj] > 0.3:
                            next_bleeding[ni, nj] = True
        self.bleeding = next_bleeding

        self.integrity[self.bleeding] = np.maximum(
            0.0, self.integrity[self.bleeding] - BLEED_DAMAGE_PER_STEP)

    # -- reward + termination ----------------------------------------------------

    def _compute_reward(self) -> float:
        col_f, row_f = self._ee_grid_pos()
        progress = 1.0 - np.hypot(col_f - self.target.grid_pos[0],
                                  row_f - self.target.grid_pos[1]) / MAX_GRID_DIST
        damage = 1.0 - float(self.integrity.mean())
        bleeding_frac = float(self.bleeding.sum()) / (
            GRID_SIZE * GRID_SIZE)

        completion = 0.0
        if self.target.reached and not self._completion_rewarded:
            completion = 1.0
            self._completion_rewarded = True

        # One-shot failure penalty: fires on the first step where any vessel
        # is damaged (reset() rebuilds vessels, so no cross-episode leakage).
        failure = 0.0
        if any(v.damaged for v in self.vessels) and not self._failure_rewarded:
            failure = 1.0
            self._failure_rewarded = True

        return float(
            REWARD_W_PROGRESS * progress
            - REWARD_W_DAMAGE * damage
            - REWARD_W_BLEEDING * bleeding_frac
            - REWARD_W_STEP
            + REWARD_W_COMPLETION * completion
            - REWARD_W_FAILURE * failure
        )

    def _check_success(self) -> bool:
        return bool(self.target.reached
                    and not any(v.damaged for v in self.vessels))

    def _check_failure(self) -> bool:
        return bool(any(v.damaged for v in self.vessels)
                    or self._step_count >= self.config.max_steps)

    # -- observation ----------------------------------------------------------

    def get_state_vector(self) -> np.ndarray:
        """Locked 806-float32 state vector (QUICK_REFERENCE index table)."""
        vec = np.zeros(806, dtype=np.float32)

        vec[0:256] = self.integrity.ravel()
        vec[256:512] = self.vascularity.ravel()
        vec[512:768] = self.bleeding.astype(np.float32).ravel()

        pos, orn = self._get_ee_pose()
        ws = self._world_to_canonical(pos)
        vec[768] = (ws[0] + CANONICAL_XY_HALF) / 0.2
        vec[769] = (ws[1] + CANONICAL_XY_HALF) / 0.2
        vec[770] = ws[2] / CANONICAL_Z_RANGE
        vec[771:775] = orn  # unit quaternion, raw [-1, 1]

        jaw = float(self.psm1.get_current_jaw_position())  # 0 closed .. open
        vec[775] = np.clip(1.0 - jaw / JAW_OPEN_ANGLE_RAD, 0.0, 1.0)

        vec[776] = self.target.grid_pos[0] / GRID_MAX_IDX
        vec[777] = self.target.grid_pos[1] / GRID_MAX_IDX
        vec[778] = 1.0 if self.target.reached else 0.0

        for slot, vessel in enumerate(self.vessels[:MAX_VESSEL_SLOTS]):
            base = 779 + slot * 4
            vec[base] = vessel.col / GRID_MAX_IDX
            vec[base + 1] = vessel.row / GRID_MAX_IDX
            vec[base + 2] = vessel.radius_workspace / GRID_MAX_IDX
            vec[base + 3] = 1.0 if vessel.damaged else 0.0

        vec[799 + self.config.dof - 1] = 1.0  # DOF one-hot
        vec[805] = 0.0                         # padding
        return vec


# ---------------------------------------------------------------------------
# Async factory (S-2 CONCURRENT INIT RISK)
# ---------------------------------------------------------------------------

async def create_env(config: SimConfig) -> SurROLTissueEnv:
    """Construct + reset an env under the PyBullet init lock.

    Always await this from async context — never construct directly.
    """
    async with _pybullet_init_lock:
        env = await asyncio.to_thread(SurROLTissueEnv, config)
        await asyncio.to_thread(env.reset)
        return env
