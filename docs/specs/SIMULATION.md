# SIMULATION.md
# SOMA — Simulation Specification
# Authoritative for: SurRoL task implementation, tissue model, state
# vector construction, action effects, reward function, DOF progression,
# placement algorithms, instance management, data collection.
#
# Invariants referenced here are DEFINED in ARCHITECTURE.md.
# Constants are duplicated in QUICK_REFERENCE.md for fast lookup.
# Read this before building: simulation.py, tissue.py (if split),
# data_collector.py, train.py.

---

## SURROL INTEGRATION

### What SurRoL Provides

SurRoL (github.com/med-air/SurRoL) provides:
- da Vinci PSM robot model with accurate kinematics (PyBullet-loaded URDF)
- PyBullet physics server management
- Gymnasium-compatible `TaskBase` with `reset()`, `step()`, `render()`
- 3D rendering from PyBullet's OpenGL renderer

We provide:
- `SurROLTissueEnv(TaskBase)` — custom task subclass
- 16×16 tissue overlay grid (numpy arrays, not PyBullet bodies)
- `get_state_vector()` — our custom observation (806 floats)
- Reward function, success/failure conditions

### Installation

```bash
cd soma
git clone https://github.com/med-air/SurRoL.git
cd SurRoL && uv pip install -e .
cd .. && uv add pybullet==3.2.6
```

Verify: `python -c "import surrol; from surrol.tasks.task_base import TaskBase; print('ok')"`

If `pybullet` version conflict: `uv add pybullet==3.2.6 --force-reinstall`.

### Class Structure

```python
from surrol.tasks.task_base import TaskBase
import pybullet as p
import numpy as np
from dataclasses import dataclass, field
from backend.event_log import write_event

class SurROLTissueEnv(TaskBase):
    def __init__(self, config: SimConfig):
        self.config = config
        self._rng = np.random.RandomState(config.seed)  # SEED REPRO INVARIANT
        super().__init__()  # initializes PyBullet, loads PSM

    # Override these four — touch nothing else in TaskBase:
    def _build_world(self):   ...  # place tissue grid, vessels, target
    def _get_obs(self):       ...  # returns our custom obs dict
    def _compute_reward(self, obs, action, next_obs): ...
    def _check_success(self): ...  # target reached without vessel damage
    def _check_failure(self): ...  # vessel damaged OR step >= max_steps
```

**DO NOT** override PSM kinematics, joint limits, physics parameters,
or `_load_robot()`. These are correct for the da Vinci PSM and produce
the 3D visualization. Modifying them breaks both physics and rendering.

### PyBullet Instance Isolation

Each `SurROLTissueEnv` creates its own PyBullet physics server.
Creating two instances simultaneously from concurrent asyncio tasks
corrupts physics world IDs. See ARCHITECTURE.md `_pybullet_init_lock`.
Always use `await create_env(config)` — never construct directly.

---

## SIMCONFIG

```python
@dataclass
class SimConfig:
    seed:        int = 42    # RNG seed — see SEED REPRODUCIBILITY INVARIANT
    dof:         int = 1     # active DOF level, 1-6
    max_steps:   int = 200
    n_vessels:   int = 2     # major vessels to place, 0-5
    difficulty:  int = 2     # see Difficulty Levels section
```

---

## COORDINATE SYSTEM

Two coordinate spaces. Always be explicit about which one you're in.

| Space | Range | Used for |
|---|---|---|
| **Grid** | col, row ∈ [0, 15] integer | Tissue array indexing |
| **Workspace** | x,y ∈ [−0.1, 0.1]m, z ∈ [0, 0.3]m | PyBullet / PSM positions |

Conversion (both directions needed for contact detection):
```python
# Grid → Workspace
workspace_x = (grid_col / 15.0) * 0.2 - 0.1
workspace_y = (grid_row / 15.0) * 0.2 - 0.1

# Workspace → Grid (for instrument contact)
grid_col = int(np.clip((workspace_x + 0.1) / 0.2 * 15, 0, 15))
grid_row = int(np.clip((workspace_y + 0.1) / 0.2 * 15, 0, 15))
```

EE z-axis normalization for state vector: `norm_z = ee_z / 0.3` → [0, 1].

**Instrument start position (workspace):** x=−0.09, y=0.0, z=0.05.
Maps to grid approximately (col=1, row=8) — near left edge, vertically
centered. The surgical target is always placed in the central-right
region, forcing the robot to plan a real path.

---

## TISSUE MODEL

The tissue grid exists purely in numpy — it is not a PyBullet body.
It is overlaid on the PyBullet workspace and updated each step based
on PSM end-effector position.

### Four Arrays Per Episode

| Array | Dtype | Shape | Initial | Mutable |
|---|---|---|---|---|
| `integrity` | float32 | (16,16) | 1.0 everywhere | Yes |
| `vascularity` | float32 | (16,16) | 3 Gaussians | **No** |
| `bleeding` | bool | (16,16) | False everywhere | Yes |
| `elasticity` | float32 | (16,16) | 1.0 + N(0,0.1) | **No** |

`vascularity` and `elasticity` are anatomical properties — they are
fixed at episode start and never change. Mutating them is a bug.

### Initialization (exact RNG consumption order — do not deviate)

```
SEED REPRODUCIBILITY INVARIANT applies here. See ARCHITECTURE.md.
rng = np.random.RandomState(config.seed)  — one instance, used for all
```

**Step 1 — vascularity (3 Gaussians, A=0.8, σ=2.0):**
```python
# Consume 6 floats: cx1,cy1, cx2,cy2, cx3,cy3
# Each center: cx ~ Uniform(2.5, 13.5), cy ~ Uniform(2.5, 13.5)
centers = [(rng.uniform(2.5,13.5), rng.uniform(2.5,13.5)) for _ in range(3)]

rows, cols = np.mgrid[0:16, 0:16]
vascularity = np.zeros((16,16), dtype=np.float32)
for cx, cy in centers:
    vascularity += 0.8 * np.exp(-((rows-cy)**2 + (cols-cx)**2) / (2*2.0**2))
vascularity = np.clip(vascularity, 0.0, 1.0).astype(np.float32)
```

**Step 2 — elasticity (256 floats, row-major order):**
```python
# Consume 256 floats: one per cell, row 0 left→right, then row 1, ...
noise = rng.normal(0.0, 0.1, size=(16,16)).astype(np.float32)
elasticity = np.clip(1.0 + noise, 0.5, 1.5).astype(np.float32)
```

**Step 3 — vessel placement (2-10 floats, see Placement section)**
**Step 4 — target placement (2+ floats, see Placement section)**

Total RNG calls before vessels: 6 (Gaussian centers) + 256 (elasticity) = 262.
Any deviation in this count shifts vessel and target placement even
for the same seed, breaking the comparison demo.

### Contact Detection

On each step, check if PSM end-effector is contacting the tissue:
```python
ee_pos = self._get_ee_workspace_pos()  # from SurRoL's PSM link state
grid_col = int(np.clip((ee_pos[0] + 0.1) / 0.2 * 15, 0, 15))
grid_row = int(np.clip((ee_pos[1] + 0.1) / 0.2 * 15, 0, 15))
contact_force = p.getContactPoints(self.robot_id, self.tissue_plane_id)
# contact_force is non-empty when EE is within PyBullet contact threshold
```

Apply tissue action effects only when `contact_force` is non-empty.
This uses PyBullet's actual collision detection rather than a distance
heuristic — more accurate and consistent with the 3D rendering.

---

## ACTION SPACE

### SurRoL Native Action

SurRoL's PSM expects a 7-element continuous action:
`[Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]` all in [−1, 1].

We translate our 12-element action vector to this format in
`action_to_surrol(action_vec: np.ndarray) → np.ndarray`:

```python
def action_to_surrol(action_vec: np.ndarray) -> np.ndarray:
    # action_vec layout: see Action Vector section below
    delta_xyz   = action_vec[0:3] * action_vec[11]  # scaled by magnitude
    delta_rpy   = action_vec[3:6] * action_vec[11]
    gripper     = action_vec[6:7]
    action_mode = np.argmax(action_vec[7:11])  # MOVE,GRASP,CAUTERIZE,WAIT

    if action_mode == 2:   # CAUTERIZE — no EE movement
        delta_xyz[:] = 0.0
        delta_rpy[:] = 0.0
    elif action_mode == 3:  # WAIT
        return np.zeros(7)

    # DOF gating: zero out components not yet active
    if self.config.dof < 2: delta_xyz[0] = 0.0  # no x movement at DOF 1
    if self.config.dof < 3: delta_rpy[:] = 0.0  # no rotation at DOF 1-2
    if self.config.dof < 4: gripper[:] = 0.0    # no gripper at DOF 1-3

    return np.concatenate([delta_xyz, delta_rpy, gripper])
```

### Our 12-Element Action Vector

Fixed size regardless of active DOF. Inactive DOF slots are zeroed
by `action_to_surrol()`, not by the caller.

| Indices | Component | Range | Notes |
|---|---|---|---|
| [0:3] | delta EE x,y,z | [−1, 1] | Scaled by magnitude[11] |
| [3:6] | delta EE roll,pitch,yaw | [−1, 1] | Zeroed below DOF 3 |
| [6:7] | gripper command | [0, 1] | 0=open, 1=close; zeroed below DOF 4 |
| [7:11] | action mode one-hot | {0,1} | MOVE=7, GRASP=8, CAUTERIZE=9, WAIT=10 |
| [11:12] | magnitude scale | [0, 1] | Applied to delta_xyz and delta_rpy |

**Action mode effects on tissue:**
- `MOVE (7)`: movement only, no direct tissue effect
- `GRASP (8)`: movement + gripper close; tissue effect via contact
- `CAUTERIZE (9)`: no EE movement; applies CAUTERIZE effect at current position
- `WAIT (10)`: no movement, no tissue effect, step advances

### Tissue Action Effects

Applied each step when PyBullet detects contact AND action mode is not WAIT.

**CAUTERIZE** (action mode = 9, contact not required):
```python
i, j = grid_row, grid_col  # current EE grid position
bleeding[i, j] = False
integrity[i, j] = min(1.0, integrity[i, j] + 0.1)  # scar tissue gain
# No neighbor effects — cautery is thermally precise
```

**Contact damage** (all modes except WAIT, when contact detected):
```python
# PRIMARY CELL
primary_damage = contact_force_magnitude * 0.3
integrity[i, j] = max(0.0, integrity[i, j] - primary_damage)

# Bleeding trigger
if vascularity[i, j] > 0.6:
    bleeding[i, j] = True  # write to NEXT buffer — see DOUBLE-BUFFER INVARIANT

# 8 NEIGHBORS (including diagonals)
for di, dj in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
    ni, nj = i+di, j+dj
    if 0 <= ni < 16 and 0 <= nj < 16:
        d = np.sqrt(di**2 + dj**2)           # 1.0 or √2
        decay = np.exp(-d * elasticity[i, j]) # elasticity of PRIMARY cell
        neighbor_damage = contact_force_magnitude * 0.1 * decay
        integrity[ni, nj] = max(0.0, integrity[ni, nj] - neighbor_damage)
```

Why `contact_force_magnitude` not fixed: PyBullet returns actual contact
force from the physics simulation. This makes harder contacts cause more
damage, which the world model must learn. Clamped to [0, 1] via
`min(1.0, raw_contact_force / CONTACT_SCALE_FACTOR)` where
`CONTACT_SCALE_FACTOR = 50.0` (empirical, based on PSM contact forces).

**Vessel damage check** (after contact damage):
```python
ee_ws = self._get_ee_workspace_pos()
for vessel in self.vessels:
    dist = np.linalg.norm(ee_ws[:2] - vessel.workspace_pos[:2])
    if dist < (vessel.radius_workspace + 0.002):  # +2mm tip radius
        vessel.damaged = True
```

### Bleeding Propagation

Called every step. See BLEEDING DOUBLE-BUFFER INVARIANT in ARCHITECTURE.md.

```python
def _propagate_bleeding(self):
    next_bleeding = self.bleeding.copy()  # MUST be a copy, not a view
    for i in range(16):
        for j in range(16):
            if not self.bleeding[i, j]:
                continue
            # 4-connected only — bleeding spreads along tissue planes, not corners
            for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                ni, nj = i+di, j+dj
                if 0 <= ni < 16 and 0 <= nj < 16:
                    if self.vascularity[ni, nj] > 0.3:
                        next_bleeding[ni, nj] = True
    self.bleeding = next_bleeding  # atomic replacement

    # Bleeding degrades tissue slowly
    BLEED_DAMAGE_PER_STEP = 0.01
    self.integrity[self.bleeding] = np.maximum(
        0.0, self.integrity[self.bleeding] - BLEED_DAMAGE_PER_STEP
    )
```

Bleeding propagation runs every step at difficulty ≥ 4. At difficulty
1-3, `self.bleeding` can only change via direct contact — it does not
spread. Static bleeding is still visible on the heatmap and penalized
by reward, but does not grow autonomously.

---

## STATE VECTOR

**`get_state_vector() → np.ndarray` — shape `(806,)`, dtype `float32`.**

See STATE VECTOR DIMENSION INVARIANT in ARCHITECTURE.md.
Test: `test:simulation:state_vector_dim`

Construction — exact order, exact normalization:

| Index range | Component | Source | Normalization |
|---|---|---|---|
| [0:256] | tissue_integrity, row-major | `self.integrity` | already [0,1] |
| [256:512] | tissue_vascularity, row-major | `self.vascularity` | already [0,1] |
| [512:768] | bleeding_mask as float, row-major | `self.bleeding.astype(float32)` | False→0.0, True→1.0 |
| [768:771] | EE workspace position x,y,z | SurRoL PSM link state | x: (x+0.1)/0.2, y: (y+0.1)/0.2, z: z/0.3 |
| [771:775] | EE quaternion w,x,y,z | SurRoL PSM link state | already unit quat, range [−1,1] |
| [775:776] | gripper state | SurRoL gripper joint | 0.0=open, 1.0=closed |
| [776:778] | target col, row | `self.target.grid_pos` | col/15.0, row/15.0 |
| [778:779] | target reached flag | `self.target.reached` | False→0.0, True→1.0 |
| [779:799] | vessel data, 5 slots × 4 | `self.vessels` (padded) | col/15, row/15, radius/15, damaged→{0,1} |
| [799:805] | active_dof one-hot, 6 values | `self.config.dof` | index (dof−1) = 1.0, rest 0.0 |
| [805:806] | padding | — | always 0.0 |

**Total: 256+256+256+3+4+1+2+1+20+6+1 = 806 ✓**

**Vessel slot padding (indices [779:799]):**
Always exactly 5 vessel slots. Unused slots: all four values = 0.0.
Network learns to ignore zero-padded slots — but the slot count must
be fixed at 5 for the network's input layer to remain constant.

**EE quaternion range:** Unit quaternions have components in [−1, 1].
Do NOT normalize to [0, 1] — this would break the quaternion
representation. The network handles [−1, 1] values fine.

**Normalization consistency invariant:** The normalization applied
during data collection (in `data_collector.py`) must be identical to
what `get_state_vector()` produces at runtime. Any difference causes
the network to receive a distribution at inference time that differs
from what it was trained on — predictions become meaningless. These
are the same function call, so this cannot drift if you don't add
any separate normalization step in training.

### Action Vector

**`action.to_vector() → np.ndarray` — shape `(12,)`, dtype `float32`.**

See Action Space section above for layout. Repeated here for
co-location with state vector:

| Index range | Component | Range |
|---|---|---|
| [0:3] | delta EE x,y,z | [−1, 1] |
| [3:6] | delta EE roll,pitch,yaw | [−1, 1] |
| [6:7] | gripper command | [0, 1] |
| [7:11] | action mode one-hot | {0, 1} |
| [11:12] | magnitude | [0, 1] |

**Network input = concatenate([state_806, action_12]) → shape (818,).**

---

## PLACEMENT ALGORITHMS

### Random State

```python
self._rng = np.random.RandomState(config.seed)  # initialized in __init__
# By the time vessels are placed, 262 values have been consumed (see Initialization).
# Do not reset _rng between tissue init and placement.
```

### Vessel Placement

Placed in `_build_world()` after tissue initialization.

```python
MIN_VESSEL_SEPARATION = 3.0   # cells, Euclidean
SAFE_ZONE = (2.0, 14.0)       # col and row range

placed_vessels = []
for k in range(config.n_vessels):
    for attempt in range(100):
        col = self._rng.uniform(*SAFE_ZONE)   # consume 1 float
        row = self._rng.uniform(*SAFE_ZONE)   # consume 1 float
        if all(np.hypot(col-v.col, row-v.row) >= MIN_VESSEL_SEPARATION
               for v in placed_vessels):
            radius_workspace = self._rng.uniform(0.003, 0.007)  # consume 1 float
            placed_vessels.append(Vessel(col=col, row=row,
                                         radius_workspace=radius_workspace))
            break
    else:
        # 100 attempts failed — reduce vessel count, log warning
        break
```

Each vessel also gets a PyBullet sphere body placed at its workspace
position for 3D rendering. Radius = `radius_workspace`.

### Target Placement

Placed after all vessels.

```python
MIN_TARGET_VESSEL_DIST = 3.0   # 2.0 at difficulty=3
MIN_TARGET_START_DIST  = 5.0   # grid cells from instrument start
GRID_START_POS = (1.0, 8.0)   # approximate grid coords of EE start

for attempt in range(200):
    col = self._rng.uniform(3.0, 13.0)   # consume 1 float
    row = self._rng.uniform(3.0, 13.0)   # consume 1 float
    vessel_ok = all(np.hypot(col-v.col, row-v.row) >= min_dist
                    for v in placed_vessels)
    start_ok  = np.hypot(col-GRID_START_POS[0], row-GRID_START_POS[1]) >= MIN_TARGET_START_DIST
    if vessel_ok and start_ok:
        self.target = Target(col=col, row=row, radius=1.5)
        break
else:
    # Relax vessel distance to 2.0 and retry 200 more times
    ...
```

Target rendered as a PyBullet sphere (green, semi-transparent) in
workspace coordinates for 3D visualization.

---

## REWARD FUNCTION

Called in `_compute_reward()`. Returns scalar float. Used for:
1. Writing to `simulation_step` event payload
2. MPC candidate value estimation (indirectly via `_compute_value()`)

```
r = w1×progress − w2×damage − w3×bleeding − w4×step + w5×completion − w6×failure
```

| Term | Formula | Weight | Range | Notes |
|---|---|---|---|---|
| `progress` | `1.0 − dist(EE, target) / 21.21` | 1.0 | [0,1] | Dense signal every step |
| `damage` | `1.0 − mean(integrity)` | 2.0 | [0,1] | Global tissue health |
| `bleeding` | `sum(bleeding) / 256.0` | 5.0 | [0,1] | Bleeding count fraction |
| `step` | 1.0 (constant) | 0.1 | 1.0 | Time pressure |
| `completion` | 1.0 if target reached this step | 10.0 | {0,1} | Fires once |
| `failure` | 1.0 if vessel damaged this step | 50.0 | {0,1} | Fires once |

`max_dist = √(15² + 15²) = 21.213...` — hardcode `21.213`, not `math.sqrt(450)` every call.

**Reward range:** [−57.1, 10.9]. Typical per-step range: [−7.1, 1.9].
The Tanh×60 output head in the prediction network is scaled to cover
this range without clipping.

**One-time bonuses:** `completion` fires on the first step where
`target.reached` becomes True. `failure` fires on the first step
where any `vessel.damaged` becomes True. They do not fire repeatedly
on subsequent steps of the same episode.

---

## DOF PROGRESSION

DOF controls which components of `action_to_surrol()` are active.

| DOF | Capability | Unlock threshold | New action components |
|---|---|---|---|
| 1 | z-axis only (depth reach) | starting state | delta_z |
| 2 | full xy movement | global_error < 0.08 | + delta_x |
| 3 | wrist rotation | global_error < 0.06 | + delta_rpy |
| 4 | gripper control | global_error < 0.05 | + gripper |
| 5 | full PSM control | global_error < 0.04 | + delta_pitch precision |
| 6 | complete kinematics | global_error < 0.03 | all components |

**Unlock procedure (in CapabilityAgent, not here):**
1. Write `agent_spawned` event
2. Update `primary_sim.config.dof = new_dof`
3. Update `comparison_sim.config.dof = new_dof`
4. All active exploration instances retain their current DOF
5. Write `capability_unlocked` event
6. Write `agent_completed` event

DOF unlock does not reset the simulation. The episode continues from
the current state with expanded capabilities.

---

## SIMULATION INSTANCES

| Name | Seed | DOF | Owner | Lifecycle |
|---|---|---|---|---|
| `primary_sim` | 42 | current | MPCAgent | startup → shutdown |
| `comparison_sim` | 42 | current | ReactiveAgent | startup → shutdown |
| `explore_{agent_id}` | 42 | current | ExplorationAgent | agent start → agent end |
| `dof_train_{agent_id}` | 42+dof | new_dof | CapabilityAgent | agent start → agent end |

**Max concurrent instances: 8.** Beyond 8, PyBullet physics server
degrades. The orchestrator must not spawn a new exploration agent if
`len(active_exploration_agents) >= 4` (primary + comparison + 4 = 6 ≤ 8,
leaving 2 buffer for capability agents).

**Why all operational instances use seed 42:**
Exploration agents collect data from the same tissue topology as
primary. If seeds differ, fine-tuning data teaches the model about
tissue that doesn't exist in primary — the model improves on the
wrong environment. Comparison is only valid with identical anatomy.

**Why dof_training uses seed 42 + dof:**
Variety in DOF training data without diverging from known topology.
Adding `dof` (1-6) to the base seed produces 6 distinct but
deterministic environments.

---

## DATA COLLECTION

**File:** `backend/data_collector.py`
**Produces:** `backend/data/training.db`

### Training DB Schema

```sql
CREATE TABLE IF NOT EXISTS samples (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    state      BLOB NOT NULL,    -- 806 float32 values, little-endian
    action     BLOB NOT NULL,    -- 12 float32 values, little-endian
    next_state BLOB NOT NULL,    -- 806 float32 values, little-endian
    reward     REAL NOT NULL,
    done       INTEGER NOT NULL  -- 0 or 1
);
```

Serialization: `array.astype(np.float32).tobytes()` — consistent
endianness guaranteed by numpy. Deserialization:
`np.frombuffer(blob, dtype=np.float32)`.

### Collection Procedure

```python
def collect_training_data(env: SurROLTissueEnv, n_samples: int = 50_000):
    # Uses data_collection sim: SimConfig(seed=0, dof=1, n_vessels=2)
    # WHY SEED 0: different from operational seed 42 — training data
    # from a different anatomy prevents the network from overfitting
    # to one specific tissue layout while still covering same dynamics.
    
    collected = 0
    obs = env.reset()
    
    while collected < n_samples:
        state_vec = env.get_state_vector()
        action_vec = random_action_vector()  # see below
        surrol_action = action_to_surrol(action_vec)
        
        _, reward, done, _ = env.step(surrol_action)
        next_state_vec = env.get_state_vector()
        
        insert_sample(state_vec, action_vec, next_state_vec, reward, done)
        collected += 1
        
        if done:
            obs = env.reset()
        
        if collected % 5_000 == 0:
            print(f"Collected {collected}/{n_samples} samples")
    
    print(f"Data collection complete: {collected} samples")
```

### Random Action Sampling

```python
def random_action_vector() -> np.ndarray:
    vec = np.zeros(12, dtype=np.float32)
    mode = np.random.randint(0, 4)  # MOVE, GRASP, CAUTERIZE, WAIT
    vec[7 + mode] = 1.0             # one-hot into [7:11]
    vec[0:3] = np.random.uniform(-1, 1, 3)   # delta xyz
    vec[3:6] = np.random.uniform(-1, 1, 3)   # delta rpy
    vec[6]   = np.random.uniform(0, 1)        # gripper
    vec[11]  = np.random.uniform(0, 1)        # magnitude
    return vec
```

Why uniform random: maximum state space coverage. A smart policy
never visits destroyed-tissue states or maximum-bleeding states —
but the world model needs to predict those transitions too.

**Target: 50,000 samples.** Estimated time on M4 Pro: 15-20 minutes.
Start this by hour 2 or the entire training schedule slips. See
BUILD_PLAN.md clock rule.

---

## PITFALLS — SYMPTOM FIRST

---

**SYMPTOM: Two envs with same seed produce different tissue layouts.**
Match: `np.array_equal(env_a.vascularity, env_b.vascularity)` is False.

CAUSE 1 (most likely): Using `np.random.seed()` (global state) instead
of `np.random.RandomState(seed)` (instance state). Any numpy random
call — including from SurRoL's own initialization — corrupts global
state between the seed call and tissue initialization.
FIX: Verify `self._rng = np.random.RandomState(config.seed)` in
`__init__`. Every random call in the tissue model uses `self._rng.xxx()`,
never `np.random.xxx()`.

CAUSE 2: RNG consumption order deviated from the 262-value sequence.
CONFIRM: Add `print(self._rng.randint(0, 1000))` after tissue init
in both envs. If the values differ, something in the sequence changed.
FIX: Restore exact order: 6 Gaussian center floats, then 256 elasticity
floats, then vessel floats, then target floats.

Test: `test:simulation:seed_reproducibility` — see ARCHITECTURE.md.

---

**SYMPTOM: `RuntimeError: mat1 and mat2 shapes cannot be multiplied`
on first `world_model.predict()` call.**

CAUSE: `get_state_vector()` returns wrong number of values — not 806.
The concatenation `[state_vec, action_12]` produces wrong total for
the network's `Linear(818, 512)` first layer.
CONFIRM: `print(env.get_state_vector().shape)` — must be `(806,)`.
FIX: Recount components using the index table above. Off-by-one usually
in vessel padding (wrong slot count) or quaternion (3 values instead of 4).
Test: `test:simulation:state_vector_dim`

---

**SYMPTOM: Bleeding spreads differently in two envs with same seed and
same action sequence.**

CAUSE: Bleeding double-buffer violated. `next_bleeding` is a view
of `self.bleeding` (from `np.view()` or slice) rather than a copy.
Writing to `next_bleeding` modifies `self.bleeding` mid-loop.
CONFIRM: Add `assert next_bleeding.base is None` at the top of
`_propagate_bleeding`. If assertion fails: not a copy.
FIX: Use `next_bleeding = self.bleeding.copy()` — not
`next_bleeding = self.bleeding` or `np.asarray(self.bleeding)`.
Test: `test:simulation:bleeding_reproducibility`

---

**SYMPTOM: PyBullet raises `Invalid world ID` or `Segfault` when
creating a second SurROLTissueEnv.**

CAUSE: Two envs constructed concurrently without `_pybullet_init_lock`.
PyBullet's global physics server registry is corrupted when two
`TaskBase.__init__()` calls run simultaneously.
FIX: Use `await create_env(config)` in `simulation.py` which acquires
`_pybullet_init_lock` before calling `SurROLTissueEnv(config)`.
Never call the constructor directly from async context.
Test: `test:simulation:concurrent_creation`

---

**SYMPTOM: Contact damage applies every step even when robot is
not near tissue.**

CAUSE: Contact detection using distance threshold instead of PyBullet
contact points. Distance-based detection fires when EE happens to be
at the right grid coordinate regardless of actual physical contact.
FIX: Use `p.getContactPoints(robot_id, tissue_plane_id)` — returns
empty list when no contact. Apply tissue effects only when non-empty.
Test: `test:simulation:contact_detection`

---

## TESTING PROTOCOL

Named test targets for this component. Run only these after Task 1.2.
Do not run the full suite until sprint checkpoint.

```bash
pytest tests/test_simulation.py -v                                  # all simulation tests
pytest tests/test_simulation.py::test_state_vector_dim -v           # test:simulation:state_vector_dim
pytest tests/test_simulation.py::test_seed_reproducibility -v       # test:simulation:seed_reproducibility
pytest tests/test_simulation.py::test_bleeding_reproducibility -v   # test:simulation:bleeding_double_buf
pytest tests/test_simulation.py::test_concurrent_creation -v        # test:simulation:concurrent_creation
pytest tests/test_simulation.py::test_contact_detection -v          # test:simulation:contact_detection
pytest tests/test_simulation.py::test_reward_range -v               # test:simulation:reward_range
pytest tests/test_simulation.py::test_vessel_placement -v           # test:simulation:vessel_placement
pytest tests/test_simulation.py::test_dof_gating -v                 # test:simulation:dof_gating
```

### Minimal Test Set for Task 1.2 Completion

These three must pass before marking Task 1.2 complete:

```python
# test:simulation:state_vector_dim
env = SurROLTissueEnv(SimConfig(seed=42)); env.reset()
v = env.get_state_vector()
assert v.shape == (806,) and v.dtype == np.float32
assert not np.any(np.isnan(v)) and not np.any(np.isinf(v))

# test:simulation:seed_reproducibility
env_a = SurROLTissueEnv(SimConfig(seed=42)); env_a.reset()
env_b = SurROLTissueEnv(SimConfig(seed=42)); env_b.reset()
assert np.array_equal(env_a.vascularity, env_b.vascularity)
assert len(env_a.vessels) == len(env_b.vessels)
assert all(np.isclose(a.col, b.col) for a,b in zip(env_a.vessels, env_b.vessels))

# test:simulation:bleeding_reproducibility
env_a = SurROLTissueEnv(SimConfig(seed=42)); env_a.reset()
env_b = SurROLTissueEnv(SimConfig(seed=42)); env_b.reset()
rng = np.random.RandomState(0)
for _ in range(30):
    act = np.random.RandomState(0).randn(12).clip(-1,1).astype(np.float32)
    env_a.step(action_to_surrol(act)); env_b.step(action_to_surrol(act))
assert np.array_equal(env_a.bleeding, env_b.bleeding)
```

---

## VERIFICATION CHECKLIST

Run manually before Task 1.2 is marked complete. Automated tests above
cover most items — this checklist catches visual and integration issues
that tests cannot.

```
TISSUE MESH:
  □ PyBullet window opens and shows PSM robot arm
  □ env.reset() with seed=42 produces non-uniform vascularity
    (visually: some bright regions, some dark)
  □ env.vascularity.min() > 0.0 (Gaussians produce positive values)
  □ env.elasticity range: min >= 0.5, max <= 1.5
  □ After CUT action at high-vascularity cell: env.bleeding shows True
  □ After CAUTERIZE at bleeding cell: bleeding[i,j] becomes False

ROBOT AND MOVEMENT:
  □ At DOF 1: MOVE_TO with delta_x nonzero → EE x does not change
  □ At DOF 2: MOVE_TO with delta_x nonzero → EE x changes
  □ Vessel markers visible in PyBullet render (spheres)
  □ Target marker visible (green sphere)

STATE VECTOR:
  □ get_state_vector().shape == (806,)
  □ get_state_vector().dtype == float32
  □ Indices [0:256] all in [0.0, 1.0]
  □ Indices [799:805] sum to exactly 1.0 (one-hot)
  □ Index [805] == 0.0 (padding)

REWARD:
  □ reward at start (EE far from target, no damage): approximately [0.1, 0.4]
  □ reward on vessel damage: approximately -50 (large negative)
  □ reward on target reach: approximately +10 (large positive)

DATA COLLECTION:
  □ training.db created after collect_training_data()
  □ SELECT count(*) FROM samples returns >= 50000
  □ BLOB deserialization: np.frombuffer(row[0], dtype=np.float32).shape == (806,)
```
