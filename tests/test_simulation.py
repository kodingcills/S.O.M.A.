"""Task 1.2 — SurROLTissueEnv named tests.

Covers the ARCHITECTURE.md invariants owned by simulation.py:
  - test:simulation:state_vector_dim      (STATE VECTOR DIMENSION INVARIANT)
  - test:simulation:seed_reproducibility  (SEED REPRODUCIBILITY INVARIANT)
  - test:simulation:bleeding_double_buf   (BLEEDING DOUBLE-BUFFER INVARIANT / I-6)
  - test:simulation:dof_gating
  - test:simulation:reward_range
  - test:simulation:contact_detection

Run: uv run pytest tests/test_simulation.py -v   (from repo root)
"""

import numpy as np
import pytest

from backend.simulation import (
    SimConfig,
    SurROLTissueEnv,
    action_to_surrol,
)


def _move_action(dx: float, dy: float, dz: float, magnitude: float = 1.0) -> np.ndarray:
    """12-float SOMA action vector: MOVE mode with the given delta."""
    vec = np.zeros(12, dtype=np.float32)
    vec[0:3] = (dx, dy, dz)
    vec[7] = 1.0  # MOVE
    vec[11] = magnitude
    return vec


def _wait_action() -> np.ndarray:
    vec = np.zeros(12, dtype=np.float32)
    vec[10] = 1.0  # WAIT
    return vec


@pytest.fixture(scope="module")
def seed_pair():
    """Two independent seed-42 environments for reproducibility checks."""
    env_a = SurROLTissueEnv(SimConfig(seed=42))
    env_b = SurROLTissueEnv(SimConfig(seed=42))
    env_a.reset()
    env_b.reset()
    yield env_a, env_b
    env_a.close()
    env_b.close()


# ---------------------------------------------------------------------------
# test:simulation:state_vector_dim — STATE VECTOR DIMENSION INVARIANT
# ---------------------------------------------------------------------------

def test_state_vector_dim(seed_pair):
    env_a, _ = seed_pair
    v = env_a.get_state_vector()

    assert v.shape == (806,), f"Expected (806,), got {v.shape}"
    assert v.dtype == np.float32, f"Expected float32, got {v.dtype}"
    assert not np.any(np.isnan(v)), "NaN in state vector"
    assert not np.any(np.isinf(v)), "Inf in state vector"

    # DOF one-hot block [799:805] must sum to exactly 1.0
    assert v[799:805].sum() == pytest.approx(1.0), \
        f"DOF one-hot sums to {v[799:805].sum()}"

    # Padding [805] must be exactly zero
    assert v[805] == 0.0, f"Padding cell is {v[805]}"

    # Integrity + vascularity blocks are bounded [0, 1]
    assert v[0:512].min() >= 0.0 and v[0:512].max() <= 1.0


# ---------------------------------------------------------------------------
# test:simulation:seed_reproducibility — SEED REPRODUCIBILITY INVARIANT
# ---------------------------------------------------------------------------

def test_seed_reproducibility(seed_pair):
    env_a, env_b = seed_pair

    assert np.array_equal(env_a.vascularity, env_b.vascularity), \
        "Vascular layouts diverged for identical seeds"
    assert len(env_a.vessels) == len(env_b.vessels), "Vessel count diverged"
    for va, vb in zip(env_a.vessels, env_b.vessels):
        assert va.col == pytest.approx(vb.col)
        assert va.row == pytest.approx(vb.row)
        assert va.radius_workspace == pytest.approx(vb.radius_workspace)
    assert np.array_equal(env_a.target.grid_pos, env_b.target.grid_pos), \
        "Target placement diverged"
    assert np.array_equal(env_a.elasticity, env_b.elasticity), \
        "Elasticity maps diverged"


# ---------------------------------------------------------------------------
# test:simulation:bleeding_double_buf — BLEEDING DOUBLE-BUFFER INVARIANT (I-6)
# ---------------------------------------------------------------------------

def test_bleeding_double_buffer(seed_pair):
    env_a, env_b = seed_pair

    # Re-reset (fixture may be shared with earlier tests) and inject an
    # identical bleeding seed cell so propagation actually has work to do.
    env_a.reset()
    env_b.reset()
    env_a.bleeding[8, 8] = True
    env_b.bleeding[8, 8] = True

    rng = np.random.RandomState(0)
    for _ in range(30):
        act12 = rng.randn(12).clip(-1, 1).astype(np.float32)
        act12[7] = 1.0  # force MOVE mode so argmax is deterministic
        act12[11] = abs(float(act12[11]))
        surrol_act = action_to_surrol(act12, 1)
        env_a.step(surrol_act)
        env_b.step(surrol_act)

    assert np.array_equal(env_a.bleeding, env_b.bleeding), \
        "Bleeding states diverged — double-buffer violated"
    assert np.array_equal(env_a.integrity, env_b.integrity), \
        "Integrity states diverged — tissue update not deterministic"


# ---------------------------------------------------------------------------
# test:simulation:dof_gating — pure function, no env needed
# ---------------------------------------------------------------------------

class TestDofGating:
    move = _move_action(1.0, -1.0, 0.5)

    def test_dof1_zeroes_x_rpy_gripper(self):
        out = action_to_surrol(self.move, dof=1)
        assert out.shape == (7,)
        assert out[0] == 0.0, "x delta must be gated at DOF 1"
        assert out[1] != 0.0 and out[2] != 0.0, "y/z deltas must survive DOF 1"
        assert np.all(out[3:6] == 0.0), "rpy must be gated below DOF 3"
        assert out[6] == 0.0, "gripper must be gated below DOF 4"

    def test_dof2_keeps_x(self):
        out = action_to_surrol(self.move, dof=2)
        assert out[0] != 0.0, "x delta must be active at DOF 2"
        assert np.all(out[3:6] == 0.0)

    def test_dof3_keeps_rpy(self):
        rpy_move = self.move.copy()
        rpy_move[3:6] = (0.2, -0.4, 0.6)
        out = action_to_surrol(rpy_move, dof=3)
        assert np.all(out[3:6] != 0.0), "rpy must be active at DOF 3"
        assert out[6] == 0.0

    def test_dof4_keeps_gripper(self):
        grip_move = self.move.copy()
        grip_move[6] = 1.0
        out = action_to_surrol(grip_move, dof=4)
        assert out[6] != 0.0, "gripper must be active at DOF 4"

    def test_wait_returns_zeros(self):
        wait = np.zeros(12, dtype=np.float32)
        wait[10] = 1.0  # WAIT one-hot
        out = action_to_surrol(wait, dof=6)
        assert np.array_equal(out, np.zeros(7, dtype=np.float32))

    def test_cauterize_zeroes_motion_keeps_gripper(self):
        caut = np.zeros(12, dtype=np.float32)
        caut[9] = 1.0  # CAUTERIZE
        caut[6] = 1.0
        caut[11] = 0.5
        out = action_to_surrol(caut, dof=6)
        assert np.all(out[0:6] == 0.0), "CAUTERIZE must zero xyz+rpy"
        assert out[6] != 0.0, "CAUTERIZE keeps gripper channel"

    def test_magnitude_scales_deltas(self):
        full = action_to_surrol(_move_action(1.0, 0.0, 0.0, magnitude=1.0), dof=6)
        half = action_to_surrol(_move_action(1.0, 0.0, 0.0, magnitude=0.5), dof=6)
        assert half[1] == pytest.approx(full[1] * 0.5)


# ---------------------------------------------------------------------------
# test:simulation:reward_range
# ---------------------------------------------------------------------------

def test_reward_range():
    env = SurROLTissueEnv(SimConfig(seed=42))
    try:
        env.reset()

        # Start: EE far from target, pristine tissue → small reward.
        _, reward_start, done, _ = env.step(action_to_surrol(_wait_action(), 1))
        assert -1.0 <= reward_start <= 2.0, f"Start reward {reward_start} out of [-1, 2]"
        assert not done

        # Craft near-vessel damage: flag a vessel damaged, next step fires
        # the one-time failure penalty (~-50).
        env.vessels[0].damaged = True
        _, reward_fail, done, _ = env.step(action_to_surrol(_wait_action(), 1))
        assert reward_fail < -40.0, \
            f"Vessel-damage step reward {reward_fail} should be large negative"
        assert done, "Vessel damage must terminate the episode"

        # One-time: second step must NOT fire the -50 again.
        _, reward_after, _, _ = env.step(action_to_surrol(_wait_action(), 1))
        assert reward_after > -10.0, \
            f"Failure penalty refired: {reward_after}"
    finally:
        env.close()


# ---------------------------------------------------------------------------
# test:simulation:contact_detection
# ---------------------------------------------------------------------------

def test_contact_detection():
    env = SurROLTissueEnv(SimConfig(seed=42))
    try:
        env.reset()
        push = _move_action(0.0, 0.0, -1.0, magnitude=1.0)
        contact_seen = False
        for _ in range(40):
            env.step(action_to_surrol(push, 1))
            if env._last_contact_force > 0.0:
                contact_seen = True
                break

        assert contact_seen, "EE pushed into tissue plane but PyBullet reported no contact"

        pos, _ = env._get_ee_pose()
        ws = env._world_to_canonical(pos)
        col = int(np.clip((ws[0] + 0.1) / 0.2 * 15, 0, 15))
        row = int(np.clip((ws[1] + 0.1) / 0.2 * 15, 0, 15))

        assert env.integrity[row, col] < 0.999, \
            f"Contact detected at ({row},{col}) but integrity unchanged: " \
            f"{env.integrity[row, col]}"
    finally:
        env.close()
