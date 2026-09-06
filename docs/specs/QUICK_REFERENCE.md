# QUICK_REFERENCE.md
# SOMA — Quick Reference
# SCAN THIS FILE, DON'T READ IT.
# When you need a specific value: Ctrl+F the constant name.
# When something is broken: go to GOTCHA TABLE (bottom).
# When you need to know if you're on track: go to DEAD RECKONING.
#
# Status markers:
#   (LOCKED) — from spec, never changes without breaking downstream
#   [PENDING] — fill in as tasks complete
#   [MEASURED: X] — confirmed runtime value on this machine

---

## CRITICAL NUMBERS

These 12 values appear across multiple components. One wrong value
breaks an integration silently. Memorize or verify before using.

```
State vector:        806  float32  (LOCKED — do not change)
Action vector:        12  float32  (LOCKED)
Network input:       818  float32  (= 806 + 12, LOCKED)
Tissue grid:      16×16  = 256 cells per array
Gaussians:             3  A=0.8, σ=2.0
Primary seed:         42  (primary, comparison, exploration)
Data collect seed:     0  (separate from operational seed)
DOF training seed:  42+N  (42+target_dof)
Max sim instances:     8  (PyBullet stability limit)
Max agent concurrency: 4  exploration agents
Orchestrator cycle:  2.0  seconds
MPC loop sleep:     0.05  seconds (20 steps/sec max)
```

---

## STATE VECTOR — INDICES (LOCKED)

`get_state_vector() → np.ndarray shape=(806,) dtype=float32`

| Indices | Component | Source | Normalization |
|---|---|---|---|
| [0:256] | tissue_integrity, row-major | `self.integrity` | already [0,1] |
| [256:512] | tissue_vascularity, row-major | `self.vascularity` | already [0,1] |
| [512:768] | bleeding_mask float, row-major | `bleeding.astype(f32)` | False→0, True→1 |
| [768:771] | EE position x,y,z | PSM link state | `(x+0.1)/0.2`, `(y+0.1)/0.2`, `z/0.3` |
| [771:775] | EE quaternion w,x,y,z | PSM link state | unit quat, range [−1,1] |
| [775:776] | gripper state | SurRoL gripper | 0=open, 1=closed |
| [776:778] | target col, row | `self.target.grid_pos` | col/15, row/15 |
| [778:779] | target reached | `self.target.reached` | False→0, True→1 |
| [779:799] | vessel data, 5 slots × 4 | `self.vessels` padded | col/15, row/15, radius/15, damaged→{0,1} |
| [799:805] | active_dof one-hot (6) | `self.config.dof` | index (dof−1)=1.0, rest 0.0 |
| [805:806] | padding | — | always 0.0 |

Unused vessel slots → all four values = 0.0.
EE quaternion range [−1,1] — do NOT normalize to [0,1], Sigmoid kills gradients at boundary.

---

## ACTION VECTOR — INDICES (LOCKED)

| Indices | Component | Range |
|---|---|---|
| [0:3] | delta EE x,y,z | [−1,1] |
| [3:6] | delta EE roll,pitch,yaw | [−1,1] |
| [6:7] | gripper command | [0,1] |
| [7:11] | action mode one-hot | {0,1}: MOVE=7, GRASP=8, CAUTERIZE=9, WAIT=10 |
| [11:12] | magnitude scale | [0,1] |

Network input: `np.concatenate([state_806, action_12])` → shape (818,)

---

## NETWORK ARCHITECTURE (LOCKED)

```
Input (818)
 → Linear(818,512) → LayerNorm(512) → ReLU → Dropout(0.1)
 → Linear(512,256) → LayerNorm(256) → ReLU → Dropout(0.1)
 → Linear(256,128) → LayerNorm(128) → ReLU
 → 128-dim shared representation → 4 heads:

tissue:      Linear(128,256) → Sigmoid    output shape: (B,256)
damage_prob: Linear(128,1)   → Sigmoid    output shape: (B,1)
vessel_risk: Linear(128,5)   → Sigmoid    output shape: (B,5)
instrument:  Linear(128,4)   → (none)     output shape: (B,4)
```

Expected parameters: ~615K
Measured parameters: [PENDING]
Measured inference latency (MPS): [PENDING] ms
Measured inference latency (CPU): [PENDING] ms

---

## THRESHOLDS (all LOCKED)

| Constant | Value | Used in | Breaks if wrong |
|---|---|---|---|
| `EXPLORE_THRESHOLD` | 0.15 | `orchestrator.py:route_orchestrator` | Agents never spawn (too high) or constantly spawn (too low) |
| `CIRCUIT_BREAKER_THRESHOLD` | 3 | `orchestrator.py:collect_status_node` | High-error region depletes API credits |
| `CIRCUIT_BREAKER_BACKOFF_S` | 300 | same | Backoff too short: same failures repeat |
| `MAX_CONCURRENT_AGENTS` | 4 | `orchestrator.py` | >6 total instances → PyBullet instability |
| `AGENT_TIMEOUT_S` | 600 | `agents/exploration.py` | Zombie agents leak sim instances |
| `CLAUDE_CALL_EVERY_N` | 5 | `agents/exploration.py` | API rate issues if < 3 |
| `EXPLORE_N_STEPS` | 100 | `agents/exploration.py` | Too few: insufficient training data |
| `FINE_TUNE_EPOCHS` | 5 | `world_model.py:_fine_tune_sync` | Too many: catastrophic forgetting |
| `FINE_TUNE_LR` | 0.0001 | same | Too large: overwrites prior knowledge |
| `REPLAY_RATIO_OLD` | 0.80 | same (80% old, 20% new) | Pure new = catastrophic forgetting |
| `EMA_ALPHA` | 0.3 | `belief_state.py:update_prediction_error` | Too high: noisy heatmap; too low: slow response |
| `DOF_THRESHOLDS` | {1:0.08, 2:0.06, 3:0.05, 4:0.04, 5:0.03} | `orchestrator.py:route_orchestrator` | DOF never unlocks or unlocks too early |
| `CONTACT_SCALE_FACTOR` | 50.0 (empirical) | `simulation.py:contact_damage` | Needs measurement after Task 1.2 |

---

## REWARD FUNCTION (LOCKED)

```
r = 1.0×progress − 2.0×damage − 5.0×bleeding − 0.1×step + 10.0×completion − 50.0×failure

progress   = 1.0 − dist(EE,target) / 21.213   (max_dist = √(15²+15²))
damage     = 1.0 − mean(integrity[0:256])
bleeding   = sum(bleeding[512:768]) / 256.0
step       = 1.0 per step (constant)
completion = 1.0 on first step target_reached=True
failure    = 1.0 on first step vessel_damaged=True

Reward range: [−57.1, +10.9]
Tanh×60 output head covers this range without clipping.
Hardcode max_dist = 21.213 (not math.sqrt(450) per call).
```

---

## TISSUE MODEL (LOCKED)

```
Vascularity init: 3 Gaussians, A=0.8, σ=2.0
  v[r,c] = clip(Σ 0.8 × exp(−((r−cr)²+(c−cc)²)/8.0), 0, 1)
  Note: 2σ² = 8.0

Elasticity init: clip(1.0 + N(0,0.1), 0.5, 1.5)

Bleeding trigger:     vascularity[r,c] > 0.6
Bleeding propagation: neighbor vascularity > 0.3
Bleeding damage/step: 0.01 subtracted from integrity

Contact damage primary:   integrity[r,c] -= force × 0.3
Contact damage neighbors: -= force × 0.1 × exp(−d × elasticity[center])
  where d = Euclidean distance to neighbor

CAUTERIZE:    bleeding[r,c] = False; integrity[r,c] = min(1.0, integrity+0.1)
```

RNG consumption order (must match exactly for seed reproducibility):
1. Gaussian centers: 6 floats (cx,cy × 3)
2. Elasticity noise: 256 floats (row-major)
3. Vessel positions: 2-3 floats per vessel (rejection sampling)
4. Target position: 2+ floats (rejection sampling)

---

## API ENDPOINTS

| Method | Path | Returns | Poll / When |
|---|---|---|---|
| GET | `/health` | `HealthResponse` | 1000ms (top bar) |
| GET | `/events?since={ts}` | `EventLogEntry[]` | WS fallback only |
| GET | `/events/{agent_id}` | `EventLogEntry[]` | On node click |
| GET | `/detail/{sim_id}` | `{simulation_state, belief_state}` | 300ms (detail view) |
| GET | `/sim/{sim_id}/stream` | MJPEG stream | `<img src>` (no poll) |
| GET | `/simulation/list` | sim id list | 5000ms |
| POST | `/simulation/{sim_id}/reset` | — | On button |
| GET | `/debug/trigger_agent` | — | Demo prep |
| GET | `/debug/reset_belief` | — | Demo prep (if converged) |
| WS | `/ws/events` | `{type,data}` stream | Persistent |

WebSocket replay: server sends ALL events via `get_events_since(0)` before adding client to broadcast set. Without this: canvas empty on late connect.

---

## EVENT TYPES — EXACT STRINGS

Typos cause silent frontend failures (event written to DB but canvas ignores it).

```
SYSTEM:      system_ready  training_started  training_completed
SIMULATION:  simulation_started  simulation_step
WORLD MODEL: world_model_updated  belief_snapshot
AGENTS:      agent_spawned  agent_step  agent_completed  agent_failed
CAPABILITY:  capability_unlocked
```

Canvas mapping (agent_id prefix → node type):
```
exp_*  →  ExplorationNode     (COLOR_EXPLORATION  #007AFF)
cap_*  →  CapabilityNode      (COLOR_CAPABILITY   #AF52DE)
wm_*   →  WorldModelNode      (COLOR_WORLDMODEL   #34C759)
root   →  RootNode            (COLOR_ROOT         #FFFFFF)
```

---

## ANIMATION TIMINGS (LOCKED)

| Animation | Duration | Keyframes |
|---|---|---|
| Node spawn pop | 300ms | scale(0)→scale(1.08)→scale(1.0) |
| Spawning→active | 500ms after event.timestamp | CSS class swap (not Date.now()) |
| Node glow pulse | 2000ms infinite | box-shadow 0→8px→0 |
| Completion flash | 800ms total | border: color→white (400ms)→green |
| Particle burst | 400ms | 8 divs radiate 30px, fade |
| Edge draw | 500ms | stroke-dashoffset path_len→0 |
| Error reduction flash | 100ms | cell→white |
| Error reduction yellow | 200ms | white→yellow |
| Error reduction settle | 300ms | yellow→final color |
| **Total error reduction** | **600ms** | **The demo's money moment** |
| Camera pan | 800ms | setCenter() with duration |
| Camera debounce | 2000ms | only pan once per 2s |
| DOF badge pulse | 1500ms (3×500ms) | green pulse ×3 |
| Vessel damage flash | continuous | opacity 0.6+0.2×sin(t/125) |

---

## FILE OWNERSHIP MAP

| File | Owns | Must NOT import from |
|---|---|---|
| `event_log.py` | SQLite schema, write/read | anything in backend/ |
| `simulation.py` | SurROLTissueEnv, SimConfig, `get_state_vector()` | world_model, agents |
| `belief_state.py` | BeliefState, `to_input_vector()`, `get_regional_errors()` | simulation, world_model |
| `prediction_net.py` | PredictionNetwork, PredictedState | anything in backend/ |
| `world_model.py` | WorldModel singleton, locks, replay buffer | agents, orchestrator, api |
| `orchestrator.py` | LangGraph graph, route logic, spawn nodes | agents (builds them) |
| `agents/exploration.py` | ExplorationAgent, Claude API tool use | orchestrator |
| `agents/capability.py` | CapabilityAgent | orchestrator |
| `agents/reactive.py` | ReactiveAgent | world_model, anthropic, orchestrator |
| `mpc_agent.py` | MPCAgent, candidate generation, value function | orchestrator, agents |
| `api.py` | FastAPI app, ConnectionManager, MJPEG stream | world_model (pass as param) |
| `main.py` | WorldModel singleton creation, startup sequence | (imports everything) |
| `types/index.ts` | all TypeScript interfaces | (nothing) |
| `utils/colors.ts` | all color constants, `errorToColor()` | (nothing) |
| `utils/buildCanvasState.ts` | pure canvas function | hooks, backend |

**Safe import chain (never import in reverse):**
```
event_log → simulation → belief_state → prediction_net
         → world_model → orchestrator → agents/* → api → main
```

---

## COMMANDS

### Setup (run once)
```bash
cd soma/backend
uv venv && source .venv/bin/activate
uv sync                          # installs from uv.lock

# Install SurRoL (if not done)
cd ../SurRoL && uv pip install -e . && cd ../backend
uv add pybullet==3.2.6

# Frontend
cd ../frontend && npm install
```

### Run system
```bash
# Terminal 1 — backend
cd soma/backend && source .venv/bin/activate
uvicorn main:app --reload --port 8000

# Terminal 2 — frontend
cd soma/frontend && npm run dev

# Terminal 3 — data collection (Task 1.3 only)
cd soma/backend && python -c "
from simulation import SurROLTissueEnv, SimConfig
from data_collector import collect_training_data
env = SurROLTissueEnv(SimConfig(seed=0))
env.reset()
collect_training_data(env, n_samples=50_000)"

# Terminal 4 — training (after data collection)
cd soma/backend && python -m train
```

### Add dependency
```bash
# Python (updates pyproject.toml + uv.lock)
cd soma/backend && uv add <package>

# TypeScript (updates package.json + package-lock.json)
cd soma/frontend && npm install <package>
```

### Verify components
```bash
# State vector shape
python -c "
from simulation import SurROLTissueEnv, SimConfig
import numpy as np
env = SurROLTissueEnv(SimConfig(seed=42)); env.reset()
v = env.get_state_vector()
print(f'shape={v.shape} dtype={v.dtype} nan={np.any(np.isnan(v))}')"
# Expected: shape=(806,) dtype=float32 nan=False

# Training data count
sqlite3 backend/data/training.db "SELECT COUNT(*) FROM samples;"
# Expected: 50000+

# Event log
sqlite3 backend/data/events.db "
SELECT event_type, COUNT(*) FROM events GROUP BY event_type;"

# Model exists
ls -lh backend/models/prediction_network.pt

# TypeScript clean
cd frontend && npx tsc --noEmit

# React Flow version
cat frontend/package.json | python3 -c "
import json,sys; d=json.load(sys.stdin)
print(d['dependencies'].get('@xyflow/react','NOT FOUND'))"
# Expected: ^12.x.x

# CSS import count (must be exactly 1)
grep -c "xyflow/react/dist/style.css" frontend/src/App.tsx
```

---

## DEBUG ONE-LINERS (organized by symptom)

**Canvas blank, no nodes, no console errors:**
```bash
grep -c "xyflow/react/dist/style.css" frontend/src/App.tsx  # must be 1
cat frontend/package.json | grep -E "xyflow|reactflow"      # no reactflow
```

**Canvas re-layouts every event (nodes jump around):**
```bash
grep -n "nodeTypes" frontend/src/components/Canvas/index.tsx | head -5
# nodeTypes definition must NOT be inside a function/component body
```

**agent_completed events exist but error_before == error_after:**
```bash
sqlite3 backend/data/events.db "
SELECT agent_id, error_before, error_after, payload
FROM events WHERE event_type='agent_completed' LIMIT 5;"
# If error_before IS NULL: fix write_event() call in exploration.py
# If both are same value: _measure_region_error() called before fine_tune()
```

**Heatmap mid-range appears brown not yellow:**
```javascript
// In browser console:
import { errorToColor } from './utils/colors'
errorToColor(0.5)  // must return exactly: hsl(60, 85%, 45%)
// If hsl(30,...) or hex brown: RGB interpolation used instead of HSL
```

**WebSocket connects but canvas stays empty after running for hours:**
```bash
# Backend must replay ALL events on connect
sqlite3 backend/data/events.db "SELECT COUNT(*) FROM events;"
# If >0 but canvas empty: WebSocket handler missing get_events_since(0) on connect
```

**WorldModel.fine_tune() causes WebSocket disconnect:**
```bash
grep -n "to_thread\|asyncio.to_thread" backend/world_model.py
# _fine_tune_sync call MUST be wrapped in asyncio.to_thread()
# Missing to_thread: event loop blocks for minutes → WS heartbeat timeout
```

**SQLite 'database is locked' errors:**
```bash
sqlite3 backend/data/events.db "PRAGMA journal_mode;"
# Must return: wal
# If 'delete' or 'memory': WAL mode not set on connection
grep -n "journal_mode=WAL\|journal_mode" backend/event_log.py
```

**Agents spawn but world model error never decreases:**
```python
# Test circular input invariant:
from backend.belief_state import BeliefState
import numpy as np
b = BeliefState()
v1 = b.to_input_vector().copy()
b.update_prediction_error(np.ones((16,16),dtype=np.float32),
                          np.zeros((16,16),dtype=np.float32))
assert np.array_equal(v1, b.to_input_vector()), "CIRCULAR INPUT VIOLATED"
```

**Two simulations with same seed produce different tissue:**
```bash
grep -n "np.random.seed\|random.seed" backend/simulation.py
# Must return empty — only RandomState(seed) is allowed
grep -n "RandomState" backend/simulation.py
# Must appear in __init__ as: self._rng = np.random.RandomState(seed)
```

**MPC loop starving other coroutines (slow WS, frozen heatmap):**
```bash
grep -n "asyncio.sleep" backend/mpc_agent.py
# MUST have: await asyncio.sleep(0.05) at end of loop body
# Without it: event loop cooperative yield never happens
```

---

## OH MY OPENCODE QUICK PATTERNS

### Invoke an agent
```
/agent oracle          # architecture, debugging, LangGraph, async
/agent sisyphus        # standard implementation tasks
/agent librarian       # library docs lookup
/agent prometheus      # planning before complex tasks
/agent explore         # fast codebase grep
```

### Context7 library lookups (via Librarian)
```
/context7 @surrol        # SurRoL TaskBase API, available methods
/context7 @langgraph     # LangGraph StateGraph, Send, Command patterns
/context7 @xyflow/react  # React Flow v12 hooks, node API
/context7 @plotly        # react-plotly.js data/layout format
/context7 @anthropic     # Claude API tool_use, streaming
/context7 @torch         # PyTorch MPS, LayerNorm, optimizer API
```

### MASTER_STATE.md injection
Every prompt must include:
```
=== CURRENT BUILD STATE ===
{paste full MASTER_STATE.md here}
===========================
```
Agents have no memory between sessions. A session without MASTER_STATE.md
will re-implement completed tasks and break verified interfaces.

### Session recovery after crash
If a session crashes mid-task:
1. Open `MASTER_STATE.md` — check what was completed vs what was in progress
2. Check git log: `git log --oneline -10` — what was committed?
3. Run the task's success condition — does it pass? If yes: mark complete, move on
4. If no: generate a new prompt for ONLY the failing part (not the whole task)
5. Never re-run a complete previous task unless its success condition fails

### Handling spec ambiguity mid-session
When an agent encounters spec ambiguity:
1. Make the most conservative choice
2. Add comment: `# SPEC AMBIGUITY: {description}. Chose {option} because {reason}.`
3. Write to MASTER_STATE.md under KNOWN DEVIATIONS
4. Continue — do not pause waiting for resolution

---

## CONTEXT BUDGET MAP

What to inject per agent type to avoid wasting context on irrelevant docs.

| Agent | Always inject | Inject for task | Never inject |
|---|---|---|---|
| Sisyphus | MASTER_STATE.md | 1 spec section, QUICK_REFERENCE.md | All spec files |
| Oracle | MASTER_STATE.md | ARCHITECTURE.md, task spec section | RISKS.md, INTERFACE.md (unless frontend) |
| Librarian | MASTER_STATE.md, task spec section | Context7 query | Other spec files |
| Prometheus | BUILD_PLAN.md, MASTER_STATE.md | Task spec section | Implementation details |

**Context budget rule:** 1 full spec doc + MASTER_STATE.md + task prompt ≈ 40-60K tokens.
Most agent context windows are 100-200K. Leave room for the agent's implementation.
If you inject all 8 spec docs: context fills before the agent writes code.

---

## DEAD RECKONING TABLE

Single-glance schedule check. At each hour, everything above the line must be done.

```
Hour  │ Must be complete                    │ If not: action
──────┼──────────────────────────────────────┼────────────────────────────
  1   │ Task 1.1 (event log + API)          │ Drop MJPEG stub, just finish core
  3   │ Task 1.2 (SurRoL env), data running │ ⚑ STOP — start data collection NOW
  5   │ Task 1.4 (network built)            │ Drop Dropout, just build encoder+heads
  5   │ Training started (Task 1.4b)        │ Start with whatever data exists
  7   │ Tasks 1.5 + 1.6 (WM + LangGraph)   │ Simplify: use asyncio tasks, not LangGraph
  9   │ Task 1.7 (all agents built)         │ Drop Claude API → random actions
 10   │ Task 1.8 (MPC agent)                │ Simplify: greedy selection (1 candidate)
 12   │ Sprint 1 checkpoint passes          │ Fix checkpoint before any Sprint 2
 14   │ Task 2.1 (frontend scaffold)        │ Use CRA instead of Vite to save time
 17   │ Task 2.2 (canvas + nodes)           │ Cut node animations, keep tree growth
 20   │ Task 2.3 (sim panel + MJPEG)        │ Drop MJPEG → HTML Canvas grid
 23   │ Task 2.4 (heatmap + feed)           │ Cut Claude reasoning feed, keep heatmap
 26   │ Task 2.5 (comparison)               │ Cut comparison view entirely
 28   │ Sprint 2 checkpoint passes          │ Fix error reduction animation FIRST
 30   │ Demo video recorded                 │ Record immediately, even if broken
 33   │ Devpost submitted                   │ Submit what exists — late = disqualified
 35.5 │ Everything stops                    │ Pitch rehearsal only
```

**Yellow zone** (hours 3, 28, 30, 33): non-negotiable interventions.
Missing hour 3 data collection cascades to miss hour 7 training → miss hour 12 checkpoint.

---

## GOTCHA TABLE

Most common wrong values and their exact symptoms. Match symptom before reading spec.

| Symptom | Wrong value | Correct value | File to fix |
|---|---|---|---|
| `RuntimeError: mat1 and mat2 shapes cannot be multiplied` | state_vec.shape = (802,) | (806,) | `simulation.py:get_state_vector` |
| Network forward pass wrong output | input is (1,806) not (1,818) | concatenate `[state_806, action_12]` | network caller |
| Two seeds produce different tissue | `np.random.seed(42)` used | `np.random.RandomState(42)` | `simulation.py:__init__` |
| Bleeding different in two identical-seed sims | `next_bleeding = self.bleeding` (view) | `next_bleeding = self.bleeding.copy()` | `simulation.py:_propagate_bleeding` |
| Canvas blank, no errors | CSS import missing | `import '@xyflow/react/dist/style.css'` in App.tsx | `frontend/src/App.tsx` |
| Canvas re-layouts every event | `const NODE_TYPES = {...}` inside component | Move outside component to module scope | `Canvas/index.tsx` |
| `agent_completed` error fields null | kwargs used instead of positional args | `write_event('agent_completed', ..., error_before=X, error_after=Y)` | `agents/exploration.py` |
| Error map never decreases (good training loss) | `to_input_vector()` includes error_map | Remove all error_map refs from that method | `belief_state.py` |
| WS disconnects during agent completion | `fine_tune()` not using `to_thread` | `await asyncio.to_thread(self._fine_tune_sync, ...)` | `world_model.py` |
| SQLite deadlock, backend hangs | WAL not set | `conn.execute("PRAGMA journal_mode=WAL")` on every connection | `event_log.py` |
| Agent canvas nodes never appear | `agent_spawned` written after episode | Move write_event to orchestrator's spawn_exploration_node | `orchestrator.py` |
| Error reduction animation never fires | `error_before == error_after` in event | `_measure_region_error()` must run AFTER `fine_tune()` returns | `agents/exploration.py` |
| MPS not used (training takes 3× longer) | `torch.device('cpu')` used | Check `torch.backends.mps.is_available()`, use `'mps'` | `prediction_net.py:__init__` |
| `GraphRecursionError` in orchestrator | `collect_status_node` not draining `completed_agents` | Ensure node returns updated state with completed agents removed from `active_agents` | `orchestrator.py` |
| heatmap(0.5) appears brown | RGB interpolation in Plotly colorscale | Use `hsl(60,85%,45%)` in colorscale, not hex | `WorldModelPanel.tsx` |

---

## MEASURED PERFORMANCE LOG

Fill in as tasks complete. Used for scheduling decisions.

```
System:
  Python version:                     [PENDING]
  PyTorch version:                    [PENDING]
  MPS available:                      [PENDING]
  Node version:                       [PENDING]

Simulation:
  env.step() latency:                 [PENDING] ms
  get_state_vector() latency:         [PENDING] ms
  Seed reproducibility confirmed:     [PENDING] yes/no
  CONTACT_SCALE_FACTOR (measured):    [PENDING] — test with PSM contact

Training:
  Data collection time (50K samples): [PENDING] min
  Samples per second:                 [PENDING]
  Training time (50 epochs, MPS):     [PENDING] min
  Training time (50 epochs, CPU):     [PENDING] min
  Final val_tissue_mse:               [PENDING]
  Final val_damage_acc:               [PENDING]
  Model parameter count:              [PENDING]

World model:
  predict() latency:                  [PENDING] ms
  fine_tune() duration (5 epochs):    [PENDING] sec
  Error reduction per agent cycle:    [PENDING] mean delta

Agents:
  Exploration episode duration:       [PENDING] min
  Claude API call latency (p50):      [PENDING] ms
  Claude API call latency (p99):      [PENDING] ms

Frontend:
  WS replay burst (events on connect):[PENDING]
  WS replay time:                     [PENDING] ms
  Canvas render time (50 nodes):      [PENDING] ms
  Canvas render time (200 nodes):     [PENDING] ms
  errorToColor(0.5) verified yellow:  [PENDING] yes/no
  Error reduction animation tested:   [PENDING] N times
```

---

## GIT CONVENTIONS

```bash
# Task completion commit
git add -A && git commit -m "task: 1.1 event-log-and-api
- init_db, write_event, get_events_since implemented
- WebSocket replay on connect
- MJPEG stream stub
deviations: none"

# Sprint checkpoint commit
git add -A && git commit -m "sprint: 1-checkpoint
- full loop running
- all Sprint 1 checkpoint items pass"

# Mid-task save (agent session ending before task complete)
git add -A && git commit -m "wip: 1.7 exploration-agent
- episode loop complete
- Claude API tool use implemented
- fine_tune integration TODO"

# Fix commit
git add -A && git commit -m "fix: world-model-to-thread
- fine_tune was blocking event loop
- wrapped _fine_tune_sync in asyncio.to_thread"
```
