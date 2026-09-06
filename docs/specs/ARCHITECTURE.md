# ARCHITECTURE.md
# SOMA — System Architecture
# Authoritative for: component communication law, event log schema,
# API contracts, concurrency model, startup sequence, integration
# contracts, named invariants, testing protocol.
#
# Read this before building any backend component.
# Cross-references: SIMULATION.md (state vector), WORLD_MODEL.md
# (network + belief), AGENTS.md (LangGraph graph), QUICK_REFERENCE.md
# (all constants).

---

## SYSTEM DIAGRAM

```
┌─────────────────────────────────────────────────────────────────┐
│                         main.py                                 │
│   creates all singletons, starts all asyncio tasks              │
└──────────────────────┬──────────────────────────────────────────┘
                       │ owns
        ┌──────────────┼──────────────────────┐
        │              │                      │
   WorldModel     SimulationManager    LangGraph Orchestrator
   (singleton)    (singleton)          (singleton)
        │              │                      │
    ┌───┴───┐      ┌───┴────────┐        ┌────┴──────────────┐
    │ Pred  │      │ primary_sim│        │ ExplorationAgent  │
    │ Net   │      │ (seed=42)  │        │ CapabilityAgent   │
    │       │      │comparison  │        │ ReactiveAgent     │
    │Belief │      │ (seed=42)  │        └──────────────────┬┘
    │ State │      │ explore_*  │                           │
    └───────┘      │ (seed=42)  │                           │
        │          └────────────┘                           │
        │                                                   │
        └────────────────────┬──────────────────────────────┘
                             │  all write via write_event()
                             ▼
                    ┌─────────────────┐
                    │  events.db      │  ← append-only SQLite
                    │  (WAL mode)     │
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │   FastAPI       │
                    │  REST + WS      │
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │   Frontend      │
                    │ React + Plotly  │
                    │ @xyflow/react   │
                    └─────────────────┘
```

**Three direct-call relationships (all others are event-log mediated):**

```
MPCAgent ──────────────────► WorldModel.predict()  [every sim step, ~50ms budget]
MPCAgent ──────────────────► WorldModel.update()   [every sim step]
LangGraph Orchestrator ─────► WorldModel.belief.get_regional_errors()  [every 2s]
Agents ─────────────────────► SurROLTissueEnv.step()  [tight data collection loop]
```

---

## THE COMMUNICATION LAW

**Every component communicates through `write_event()` and
`get_events_since()`. No direct component-to-component calls
except the three above.**

Why this law exists — and why violating it breaks the demo specifically:

The frontend's infinite canvas reconstructs the full system state by
replaying the event log from timestamp 0. If two components exchange
information directly without logging it, the canvas cannot reflect that
interaction. An exploration agent that calls `world_model.update()`
without writing an `agent_completed` event will fine-tune the model
invisibly — the heatmap changes but no node on the canvas explains why.
Judges see the heatmap but not the cause. The demo's entire story
depends on this causal chain being visible.

The law also allows Claude Code sessions to build components
independently. A component that doesn't exist yet is simply one that
isn't producing events. The rest of the system degrades gracefully
rather than crashing.

**Before making any direct call, ask: "Is this call one of the three?"**
If no: use `write_event()` instead. This is not a performance tradeoff
to evaluate — it is a hard rule with a demo-visible consequence for
violation.

---

## NAMED INVARIANTS

Every invariant has a name, a requirement, a cause-effect reasoning
chain, and a test. Reference these by name in code comments and in
MASTER_STATE.md entries.

---

### WAL MODE INVARIANT

**Requirement:** Execute `PRAGMA journal_mode=WAL` on every SQLite
connection immediately after opening it. Not once globally — on every
`sqlite3.connect()` call.

**Why:** SQLite's default journal mode acquires exclusive file locks
during writes. Multiple asyncio tasks calling `write_event()`
concurrently cannot acquire the lock simultaneously and deadlock.
The backend appears frozen with no error in the main process — the
deadlock is in a thread pool worker. WAL mode allows one active writer
and unlimited concurrent readers, eliminating the deadlock.

**Why per-connection:** Each `sqlite3.connect()` creates a connection
that starts in the default journal mode regardless of other connections.
Setting WAL mode once does not persist across new connections.

**Test:** `test:event_log:wal`
```python
conn = sqlite3.connect("data/events.db")
row = conn.execute("PRAGMA journal_mode").fetchone()
assert row[0] == "wal", f"WAL mode not set, got: {row[0]}"
```

---

### STATE VECTOR DIMENSION INVARIANT

**Requirement:** `SurROLTissueEnv.get_state_vector()` returns exactly
806 `float32` values. This size is fixed regardless of active DOF level.
Inactive DOF slots are padded with 0.0.

**Why:** The prediction network's first `nn.Linear` layer has
`in_features=818` (806 state + 12 action). PyTorch raises
`RuntimeError: mat1 and mat2 shapes cannot be multiplied` if the
concatenated input is any other size. This error appears only on the
first forward pass, not during construction — if data collection
happens before the network exists, a wrong vector size won't be caught
until training starts.

**Why not to "fix" this by making the input layer dynamic:**
Dynamic input size means the saved checkpoint is incompatible with
any future fine-tuning call that uses a different size. The network
becomes unreloadable.

**Test:** `test:simulation:state_vector_dim`
```python
env = SurROLTissueEnv(SimConfig(seed=42))
env.reset()
v = env.get_state_vector()
assert v.shape == (806,), f"Expected (806,), got {v.shape}"
assert v.dtype == np.float32, f"Expected float32, got {v.dtype}"
assert not np.any(np.isnan(v)), "NaN in state vector"
assert not np.any(np.isinf(v)), "Inf in state vector"
```

---

### CIRCULAR INPUT INVARIANT

**Requirement:** `BeliefState.to_input_vector()` must never include
`prediction_error_map` or `confidence_map`. These fields exist on
`BeliefState` but must be excluded from the method's output.

**Why:** `prediction_error_map` is computed from the network's own
output (the difference between predicted and actual next states). If it
appears as network input, the network's input at step T depends on the
network's output at step T-1. The network learns to exploit this
circularity: it adjusts its predictions based on its own error history
rather than tissue dynamics. Training loss decreases — the model appears
to converge — but predictions have no relationship to actual tissue
behavior. The heatmap error never decreases regardless of agent activity.
This failure mode is undetectable without explicitly testing the invariant.

**Test:** `test:belief:circular_input`
```python
belief = BeliefState()
vec_before = belief.to_input_vector().copy()
belief.update_prediction_error(
    predicted=np.ones((16,16), dtype=np.float32),
    actual=np.zeros((16,16), dtype=np.float32)
)
vec_after = belief.to_input_vector()
assert np.array_equal(vec_before, vec_after), \
    "to_input_vector() changed after update_prediction_error() — CIRCULAR INPUT"
```

---

### BLEEDING DOUBLE-BUFFER INVARIANT

**Requirement:** Bleeding propagation must allocate a separate
`next_bleeding` array before the propagation loop. Read exclusively
from `bleeding`. Write exclusively to `next_bleeding`. Replace
`bleeding` with `next_bleeding` atomically after the loop.

**Why:** Writing propagation results back into the array being read
makes propagation order-dependent. A cell at (r, c) that bleeds into
(r, c+1) in iteration step N changes (r, c+1)'s value. If (r, c+1)
is processed later in the same loop iteration, it incorrectly propagates
the already-propagated blood. The same seed produces different bleeding
states depending on numpy's memory layout. The comparison demo requires
`primary_sim.bleeding == comparison_sim.bleeding` at every step — this
invariant is what ensures it.

**Test:** `test:simulation:bleeding_reproducibility`
```python
env_a = SurROLTissueEnv(SimConfig(seed=42))
env_b = SurROLTissueEnv(SimConfig(seed=42))
env_a.reset(); env_b.reset()
for _ in range(20):
    action = env_a.action_space.sample()
    env_a.step(action)
    env_b.step(action)
assert np.array_equal(env_a.tissue.bleeding, env_b.tissue.bleeding), \
    "Bleeding states diverged — double-buffer violated"
```

---

### SEED REPRODUCIBILITY INVARIANT

**Requirement:** `SurROLTissueEnv(SimConfig(seed=42)).reset()` must
always produce identical tissue vascularity, vessel positions, and
target position regardless of how many other environments have been
created before or after. Use `numpy.random.RandomState(seed)` —
never `numpy.random.seed()`.

**Why:** `numpy.random.seed()` sets global RNG state. Any numpy random
call anywhere in the process — including inside SurRoL's own library
code, PyBullet's initialization, or a library that happens to call
numpy — corrupts the global state between the seed call and the point
where tissue initialization consumes random numbers. `RandomState` is
an instance with its own state, immune to global corruption.

Exploration agents use the same seed (42) as the primary simulation.
If seeds diverge, agents collect data from a tissue that doesn't exist
in the primary simulation. Fine-tuning on that data teaches the model
the wrong anatomy. The world model improves on the training metric but
worsens on the primary simulation — the heatmap shows no improvement
even as agents complete successfully.

**Test:** `test:simulation:seed_reproducibility`
```python
env_a = SurROLTissueEnv(SimConfig(seed=42))
env_b = SurROLTissueEnv(SimConfig(seed=42))
env_a.reset(); env_b.reset()
assert np.array_equal(env_a.tissue.vascularity, env_b.tissue.vascularity)
assert all(np.allclose(va.position, vb.position)
           for va, vb in zip(env_a.vessels, env_b.vessels))
assert np.allclose(env_a.target_position, env_b.target_position)
```

---

### WORLD MODEL SINGLETON INVARIANT

**Requirement:** Exactly one `WorldModel` instance exists for the
entire process lifetime. It is created in `main.py` and passed as
a parameter. Never instantiate `WorldModel` inside an agent,
orchestrator node, or API handler.

**Why:** `WorldModel` holds the prediction network weights and the
belief state (prediction error map). A second instance starts with
uninitialized weights and a blank error map. Any component that
constructs `WorldModel` locally uses a model that has never been
trained and a belief state that reflects no experience. MPC planning
produces random actions. Orchestrator spawning decisions are based
on zero error everywhere — no agents ever spawn. The system runs
without crashing and produces no useful behavior.

**Test:** `test:architecture:singleton`
```bash
grep -r "WorldModel(" backend/ | grep -v "main.py" | grep -v "test_" | grep -v "#"
# Must return empty. Any hit is a violation.
```

---

### LAYERNORM INVARIANT

**Requirement:** Use `nn.LayerNorm` after each linear layer in the
prediction network encoder. Never `nn.BatchNorm1d`.

**Why:** `BatchNorm` computes normalization statistics differently in
`model.train()` vs `model.eval()` mode. During training it uses batch
statistics; during inference it uses running means accumulated during
training. As training progresses, the gap between batch and running
statistics grows. The practical consequence: predictions during the
demo differ from what the model learned to produce during training.
The heatmap shows values that fluctuate erratically rather than
converging steadily. `LayerNorm` normalizes over feature dimensions —
its behavior is identical in both modes, so predictions are consistent
between training and demo.

**Test:** `test:world_model:layernorm`
```python
from backend.prediction_net import PredictionNetwork
net = PredictionNetwork()
has_bn = any(isinstance(m, nn.BatchNorm1d) for m in net.modules())
assert not has_bn, "BatchNorm found in PredictionNetwork — use LayerNorm"
```

---

### EVENT LOOP NON-BLOCKING INVARIANT

**Requirement:** Any CPU-bound operation inside an `async` function
must run via `asyncio.to_thread()`. Operations that are already
`async def` must NOT be wrapped in `to_thread()`.

**Applies to:** `WorldModel.fine_tune()` (runs for 2-5 minutes),
numpy operations on the full 16×16 tissue grid called in a loop,
SQLite queries scanning more than a few rows, initial training.

**Does NOT apply to:** Functions already defined as `async def`,
`asyncio.sleep()`, awaiting other coroutines.

**Why:** FastAPI runs on a single asyncio event loop. WebSocket
heartbeats, REST responses, and all other coroutines share this loop.
A synchronous call blocking for more than ~10ms prevents the loop from
processing other coroutines. `fine_tune()` blocks for minutes. During
that time, no WebSocket messages are sent. The frontend receives no
events, detects a heartbeat timeout after 15 seconds, and shows
"disconnected." The canvas appears frozen. This is visible to judges
as a broken demo, not a slow one.

**Why NOT double-wrapping:** `to_thread()` submits its argument to a
thread pool. If the argument is an async coroutine, the thread pool
worker cannot run it (no event loop in the worker thread). This raises
`RuntimeError: no running event loop` inside the thread — a confusing
error that points at `to_thread()` rather than the actual cause.

**Test:** `test:world_model:nonblocking`
```python
import asyncio, time

async def test():
    start = time.monotonic()
    ws_ticks = 0
    
    async def count_ticks():
        nonlocal ws_ticks
        for _ in range(10):
            await asyncio.sleep(0.1)
            ws_ticks += 1
    
    tick_task = asyncio.create_task(count_ticks())
    world_model = get_singleton_world_model()
    await world_model.fine_tune(sample_batch)
    await tick_task
    
    assert ws_ticks >= 8, \
        f"Event loop blocked during fine_tune: only {ws_ticks}/10 ticks received"

asyncio.run(test())
```

---

### PURE CANVAS STATE INVARIANT

**Requirement:** `buildCanvasState(events)` is a pure function. Same
`events` array → identical `nodes` and `edges` output every call. No
`Date.now()`, `Math.random()`, external state reads, or side effects.

**Why:** React may call this function multiple times for the same
events array (during reconciliation, StrictMode double-invocation,
memo comparison). If positions differ between calls, React Flow detects
the change, triggers a re-layout animation, and every node on the
canvas jumps to a new position on every incoming event. This is
indistinguishable from a broken demo to an observer.

Node positions are computed from `parent_id` and sibling count — both
derivable from the events array alone. No randomness, no timestamps.

**Test:** `test:frontend:pure_canvas` (run in browser console or Vitest)
```typescript
const events = getTestEvents() // reproducible fixture
const result1 = buildCanvasState(events)
const result2 = buildCanvasState(events)
console.assert(JSON.stringify(result1) === JSON.stringify(result2),
    "buildCanvasState is not pure")
```

---

### REACT FLOW VERSION INVARIANT

**Requirement:** All canvas imports use `@xyflow/react`. Never
`reactflow`. The CSS import `import '@xyflow/react/dist/style.css'`
appears exactly once in `App.tsx`.

**Why:** `reactflow` is the v11 package. `@xyflow/react` is v12. They
have different hook APIs, different component props, and different CSS
paths. The CSS import omission produces a blank white canvas with no
error — nodes are positioned in React Flow state but rendered without
the library's positioning CSS. This is the most common React Flow
setup failure and produces no console error.

**Test:** `test:frontend:react_flow_version`
```bash
cat frontend/package.json | python3 -c "
import json,sys
pkg = json.load(sys.stdin)
deps = pkg.get('dependencies', {})
assert '@xyflow/react' in deps, 'Missing @xyflow/react'
assert 'reactflow' not in deps, 'reactflow (v11) installed alongside v12'
print('PASS')
"
grep -c "xyflow/react/dist/style.css" frontend/src/App.tsx
# Must return 1
```

---

### WEBSOCKET REPLAY INVARIANT

**Requirement:** On WebSocket connect, the server sends all events
from `get_events_since(0)` before adding the client to the live
broadcast set. No events may be sent live until replay is complete.

**Why:** The frontend canvas builds its state by processing events
in order from the beginning. A client that connects after the system
has been running receives no historical agents, no completed nodes,
no error reduction history. The canvas shows only the root node
regardless of how long the system has been running. During a demo
where the system was started hours before judging, this means judges
see an empty canvas.

Replay before broadcast ensures the client processes events in the
correct order: historical events first (as a burst), then live events.
No events can arrive out of order because live broadcasting is blocked
until replay completes.

**Test:** `test:api:ws_replay`
```python
# Write 5 events before connecting
for i in range(5):
    write_event("system_ready", payload_data=f"test_{i}")

# Connect WebSocket
received = []
async with websockets.connect("ws://localhost:8000/ws/events") as ws:
    for _ in range(5):
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=3))
        if msg["type"] == "event":
            received.append(msg["data"])

assert len(received) >= 5, f"Replay incomplete: got {len(received)} events"
```

---

## EVENT LOG

**File:** `backend/event_log.py`
**Storage:** `backend/data/events.db` (SQLite)
**Imports:** Only stdlib — `sqlite3`, `json`, `time`, `pathlib`.
Zero imports from other backend modules. This is enforced because
circular imports through `event_log.py` break everything that imports
it, which is every component.

### Schema

```sql
CREATE TABLE IF NOT EXISTS events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp    REAL    NOT NULL,
    event_type   TEXT    NOT NULL,
    agent_id     TEXT,
    parent_id    TEXT,
    sim_id       TEXT,
    region       TEXT,
    error_before REAL,
    error_after  REAL,
    payload      TEXT    NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_timestamp  ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_agent_id   ON events(agent_id);
CREATE INDEX IF NOT EXISTS idx_event_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_sim_id     ON events(sim_id);
```

### API

| Function | Signature | Returns | Notes |
|---|---|---|---|
| `init_db` | `() → None` | — | Creates schema + sets WAL mode. Safe to call multiple times. Call before anything else. |
| `write_event` | `(event_type: str, **kwargs) → int` | new row id | Accepted kwargs: `agent_id`, `parent_id`, `sim_id`, `region`, `error_before`, `error_after`. All other kwargs go into `payload` JSON. Sets `timestamp` automatically. |
| `get_events_since` | `(timestamp: float = 0.0, limit: int = 1000) → list[dict]` | event dicts | Ordered by timestamp ASC. `payload` deserialized to dict — never raw string. |
| `get_events_for_agent` | `(agent_id: str) → list[dict]` | event dicts | Returns events where `agent_id` OR `parent_id` matches. Returns full subtree. |
| `set_broadcast_callback` | `(fn: Callable[[dict], Coroutine]) → None` | — | Called once during startup by `api.py`. `write_event()` dispatches to this after SQLite write via `asyncio.create_task()`. Fire-and-forget. |

**`write_event()` broadcast behavior:** After writing to SQLite,
constructs the complete event dict (all columns + deserialized payload)
and calls `asyncio.create_task(_broadcast_callback(event_dict))` if a
callback is registered. `write_event()` is synchronous — it never
awaits anything. The broadcast is fire-and-forget. If no callback is
registered, the SQLite write still happens.

### Event Type Registry

Typos in event_type cause silent failures: the event writes to SQLite
but the frontend mapping function doesn't recognize it and ignores it.
No error is raised.

```
SYSTEM:
  system_ready            — emitted once at startup completion
  training_started        — data collection begins
  training_completed      — initial training done, model saved

SIMULATION:
  simulation_started      — new SurROL instance created
  simulation_step         — one step completed in any instance

WORLD MODEL:
  world_model_updated     — after fine_tune() completes
  belief_snapshot         — periodic belief state snapshot

AGENTS:
  agent_spawned           — BEFORE agent episode loop starts
  agent_step              — every 10 steps during exploration
  agent_completed         — AFTER fine_tune() completes
  agent_failed            — agent errored out

CAPABILITY:
  capability_unlocked     — DOF level increased
```

### Payload Schemas (required fields per event type)

```python
# simulation_step
{"step": int, "reward": float, "tissue_mean": float,
 "vessel_damaged": bool, "target_reached": bool,
 "task_failed": bool, "active_dof": int,
 "ee_pos": [float, float, float]}

# agent_spawned
{"reason": str, "target_region": str,
 "error_level": float, "planned_steps": int}

# agent_completed
{"samples_collected": int, "fine_tune_loss": float}
# note: error_before and error_after go in top-level columns, not payload

# capability_unlocked
{"previous_dof": int, "new_dof": int,
 "trigger_error": float, "threshold": float}

# belief_snapshot
{"global_mean_error": float, "regional_errors": dict,
 "world_model_version": int, "active_agents": int,
 "error_map": list}  # list of lists, 16×16
```

### Frontend Event → Canvas Mapping

Changes to `event_type` strings or required payload fields break
the frontend canvas. Do not modify after Task 1.1 ships.

| Event | Canvas Action |
|---|---|
| `agent_spawned` (agent_id starts `exp_`) | Create ExplorationNode + edge from parent_id |
| `agent_spawned` (agent_id starts `cap_`) | Create CapabilityNode + edge from parent_id |
| `agent_completed` | Update node status → `completed`, set `error_after` |
| `agent_failed` | Update node status → `failed` |
| `capability_unlocked` | Update CapabilityNode → `completed`, pulse DOF badge |
| `belief_snapshot` | Update heatmap with `payload.error_map` |
| All others | No canvas change |

---

## API LAYER

**File:** `backend/api.py`
**Framework:** FastAPI 0.109
**Port:** 8000

### REST Endpoints

| Method | Path | Description | Poll interval |
|---|---|---|---|
| GET | `/health` | Status, version, global error, agent count | 1000ms (top bar) |
| GET | `/events?since={ts}` | All events since timestamp (float) | WebSocket fallback only |
| GET | `/events/{agent_id}` | All events for agent subtree | On node click |
| GET | `/detail/{sim_id}` | Atomic SimulationState + BeliefStateSnapshot | 300ms (detail view) |
| GET | `/simulation/list` | Active sim IDs + metadata | 5000ms |
| POST | `/simulation/{sim_id}/reset` | Reset episode | On button press |
| GET | `/debug/reset_belief` | Reset belief state to 0.5 everywhere | Demo prep only |

**`/detail/{sim_id}` atomicity requirement:**
Read `simulation_manager.get_state(sim_id)` and
`world_model.belief.snapshot()` back-to-back in the same request
handler without any `await` between reads. Both reads use in-memory
data (no I/O), so no `await` is needed. The two states must reflect
the same logical timestep. Making them two separate requests or
awaiting between reads produces a visible state inconsistency: the
heatmap shows a different step than the simulation panel.

**`/health` response schema:**
```json
{
  "status": "starting | training | ok",
  "world_model_version": 0,
  "global_error": 0.0,
  "active_agents": 0,
  "primary_sim_step": 0,
  "training_progress": null
}
```

### WebSocket

**Endpoint:** `ws://localhost:8000/ws/events`

**Protocol:**
```
Client connects
  → Server: replay all events via get_events_since(0)
      each as: {"type": "event", "data": EventLogEntry}
  → Server adds client to broadcast set
  → Server: live events as they arrive
      same format as replay messages

Server → Client every 5s: {"type": "ping"}
Client → Server: {"type": "pong"}

If pong not received within 15s: remove client, no error
On client disconnect: remove from broadcast set, no error
```

**ConnectionManager responsibilities:**
- Maintain `active_connections: list[WebSocket]`
- On `broadcast(message)`: send to all, silently remove any that fail
- On `connect(ws)`: replay history, then add to list
- Register itself with `set_broadcast_callback()` during startup

---

## CONCURRENCY MODEL

All backend components run as asyncio tasks on the single event loop
that Uvicorn owns. No `threading.Thread`. No `multiprocessing.Process`.

### Task Registry

Started in `main.py` lifespan, all as `asyncio.create_task()`:

| Task | Sleep interval | CPU bound? | to_thread required? |
|---|---|---|---|
| `mpc_agent.run()` | 0.05s per step | Yes (inference) | Yes, per predict call |
| `orchestrator.run()` | 2.0s per cycle | No | No |
| `belief_snapshot_loop()` | 10.0s | No | No |
| `exploration_agent.run()` | 0 (tight loop) | Yes | Yes, per step |
| `reactive_agent.run()` | 0.05s | No | No |

Every task must yield the event loop at least once per iteration.
Tasks without a natural `await` must include `await asyncio.sleep(0)`.
Without explicit yielding, a tight loop monopolizes the event loop
between the coroutine scheduler's forced preemption points (which do
not exist in asyncio — it is cooperative, not preemptive).

### Locks

```python
# In WorldModel.__init__():
self._predict_lock = asyncio.Lock()   # held for ~3ms per predict call
self._finetune_lock = asyncio.Lock()  # held for entire fine_tune() duration

# Usage:
async def predict(self, state, action):
    async with self._predict_lock:
        return self._predict_sync(state, action)

async def fine_tune(self, samples):
    async with self._finetune_lock:
        async with self._predict_lock:  # block predictions during weight swap
            await asyncio.to_thread(self._train_sync, samples)
        # predict_lock released, fine_tune_lock still held
        self._increment_version()
    # fine_tune_lock released
```

**PyBullet global state lock:**
PyBullet maintains a global physics server registry. Creating two
`SurROLTissueEnv` instances simultaneously in concurrent asyncio tasks
corrupts their physics world IDs. Protect all SurRoL environment
construction with a module-level `asyncio.Lock`:

```python
# In simulation.py:
_pybullet_init_lock = asyncio.Lock()

async def create_env(config: SimConfig) -> SurROLTissueEnv:
    async with _pybullet_init_lock:
        env = SurROLTissueEnv(config)
        await asyncio.to_thread(env.reset)
    return env
```

---

## STARTUP SEQUENCE

Order is load-bearing. Steps 1-9 are blocking. Step 10 yields to Uvicorn.

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Step 1: Databases ──────────────────────────────────────
    init_db()          # events.db schema + WAL mode
    init_training_db() # training.db schema

    # ── Step 2: Primary + Comparison Simulations ───────────────
    primary_sim    = await create_env(SimConfig(seed=42, dof=1))
    comparison_sim = await create_env(SimConfig(seed=42, dof=1))

    # ── Step 3: Data Collection (conditional) ──────────────────
    if training_sample_count() < 50_000:
        write_event("training_started")
        await asyncio.to_thread(collect_training_data, primary_sim)
        # ⚑ CLOCK RULE: if this hasn't started by hour 2, stop
        # everything else and run it. Training cannot begin without it.

    # ── Step 4: Initial Training (conditional) ─────────────────
    model_path = Path("models/prediction_network.pt")
    if not model_path.exists():
        await asyncio.to_thread(train_prediction_network)
        write_event("training_completed")

    # ── Step 5: WorldModel (singleton) ─────────────────────────
    world_model = WorldModel(model_path=str(model_path))
    # Pass world_model to every component that needs it. Never
    # instantiate WorldModel anywhere else. See SINGLETON INVARIANT.

    # ── Step 6: MPC + Reactive Agents ──────────────────────────
    mpc_agent      = MPCAgent(world_model, primary_sim)
    reactive_agent = ReactiveAgent(comparison_sim)

    # ── Step 7: LangGraph Orchestrator ─────────────────────────
    orchestrator = LangGraphOrchestrator(world_model, primary_sim)

    # ── Step 8: WebSocket ConnectionManager ────────────────────
    manager = ConnectionManager()
    set_broadcast_callback(manager.broadcast)

    # ── Step 9: Background Tasks ───────────────────────────────
    asyncio.create_task(mpc_agent.run())
    asyncio.create_task(reactive_agent.run())
    asyncio.create_task(orchestrator.run())
    asyncio.create_task(belief_snapshot_loop(world_model))

    write_event("system_ready")

    yield  # ← Uvicorn starts accepting requests here

    # Shutdown: cancel all tasks gracefully
```

Steps 1-8 must complete before step 9. A component started before
its dependencies exist will crash with an `AttributeError` or
`ImportError` that looks like a configuration error, not a sequencing
error.

---

## INTEGRATION CONTRACTS

Each contract states the guarantee, what downstream component depends
on it, and exactly what breaks if it's violated.

| # | Producer | Guarantee | Consumer | Breaks if violated |
|---|---|---|---|---|
| C1 | `SurROLTissueEnv.get_state_vector()` | Returns `(806,) float32`, no NaN/Inf | `WorldModel.predict()`, `WorldModel.update()`, training | `RuntimeError` on first forward pass |
| C2 | `BeliefState.to_input_vector()` | Returns `(806,) float32`, no `error_map` or `confidence_map` | `PredictionNetwork.forward()` | Silent: model trains on circular data, never improves |
| C3 | `WorldModel.fine_tune()` | Awaitable, runs in `to_thread()`, holds `_finetune_lock` | FastAPI event loop, WebSocket clients | Frontend disconnects during fine-tune |
| C4 | `write_event('agent_spawned', ...)` | Written BEFORE episode loop starts | Frontend canvas | Node never appears; tree has invisible branches |
| C5 | `write_event('agent_completed', error_before=X, error_after=Y)` | Both `error_before` and `error_after` are non-null floats | Error reduction animation | Animation never fires — condition `error_after < error_before` is False |
| C6 | `get_events_since(0)` on WS connect | Replays ALL history before live broadcast | Frontend canvas on load | Canvas empty after system has been running |
| C7 | `payload` field in `get_events_since()` result | Always a Python dict, never a raw JSON string | Frontend `payload.key` access | `undefined` on all payload field accesses, silent |
| C8 | `SurROLTissueEnv(SimConfig(seed=42)).reset()` | Same vascularity, vessels, target every call | Comparison demo, exploration agent validity | Comparison shows different anatomy; training data teaches wrong tissue |

---

## TESTING PROTOCOL

**The rule: run only the test for the component you just changed.**

Full suite reruns after every small change wastes 3-5 minutes and
generates noise that obscures which specific invariant broke. Each
component has a named test target. Run that target. If it passes, move
on. Run the full suite only at sprint checkpoints (hours 10, 26, 36).

### Named Test Targets

```bash
# Run one component's tests:
pytest tests/test_event_log.py      -v  # test:event_log
pytest tests/test_simulation.py     -v  # test:simulation
pytest tests/test_belief_state.py   -v  # test:belief
pytest tests/test_prediction_net.py -v  # test:prediction_net
pytest tests/test_world_model.py    -v  # test:world_model
pytest tests/test_orchestrator.py   -v  # test:orchestrator
pytest tests/test_agents.py         -v  # test:agents
pytest tests/test_api.py            -v  # test:api

# Run named invariant test only:
pytest tests/test_event_log.py::test_wal_mode              -v
pytest tests/test_simulation.py::test_state_vector_dim     -v
pytest tests/test_belief_state.py::test_circular_input     -v
pytest tests/test_simulation.py::test_bleeding_double_buf  -v
pytest tests/test_simulation.py::test_seed_reproducibility -v
pytest tests/test_world_model.py::test_singleton           -v
pytest tests/test_prediction_net.py::test_layernorm        -v
pytest tests/test_world_model.py::test_nonblocking         -v
pytest tests/test_api.py::test_ws_replay                   -v

# Sprint checkpoint full suite:
pytest tests/ -v --tb=short
```

### Test File → Component Mapping

Each test file tests exactly one component. Tests in the wrong file
create misleading failure messages.

| File | Tests | Uses real I/O? |
|---|---|---|
| `test_event_log.py` | WAL mode, write/read round-trip, payload serialization, broadcast callback | Yes (temp SQLite) |
| `test_simulation.py` | State vector shape/dtype, seed reproducibility, bleeding double-buffer, DOF progression | Yes (SurRoL + PyBullet) |
| `test_belief_state.py` | Circular input invariant, EMA update formula, regional error dict keys | No (pure numpy) |
| `test_prediction_net.py` | LayerNorm invariant, input/output shapes, forward pass with batch | No (PyTorch) |
| `test_world_model.py` | Singleton test (grep), non-blocking fine_tune, replay ratio, fine_tune updates version | Partial (mock sim) |
| `test_orchestrator.py` | Graph compiles, route_decision logic, should_unlock thresholds | No (mocked belief) |
| `test_agents.py` | agent_spawned before loop, agent_completed after fine_tune, same seed as primary | Yes (SurRoL) |
| `test_api.py` | WebSocket replay, /detail atomicity, /health schema | Yes (test client) |

### What Tests Check vs What They Don't

Tests check **contracts and invariants**. They do not check:
- Internal implementation details (private method names, class structure)
- That specific log messages appear
- That performance meets a target (measure separately, see QUICK_REFERENCE.md)
- Whether the Plotly heatmap looks correct (visual, check manually)
- Whether agents spawn within a given real-time window (non-deterministic)

A test that checks implementation details breaks when you refactor
without changing behavior. Write tests against the interface contracts
in this document, not against the implementation.

### Incremental Test Strategy Per Task

When you complete a task, run only its named test target. The test
for Task 1.1 (`test:event_log`) does not rerun when you finish
Task 1.2. If Task 1.2's tests fail, the bug is in Task 1.2, not
Task 1.1. This is the property that makes incremental testing fast.

Sprint checkpoint tests catch cross-component integration failures
that component-level tests cannot catch. Run them at the checkpoint,
not continuously.

---

## MODULE IMPORT RULES

Import violations cause circular import errors at startup that
are difficult to diagnose. Every import chain eventually goes through
`event_log.py`, which must import nothing from the backend.

```
event_log.py      imports: stdlib only
simulation.py     imports: event_log, stdlib, numpy, surrol, pybullet
belief_state.py   imports: stdlib, numpy
prediction_net.py imports: stdlib, numpy, torch
world_model.py    imports: event_log, belief_state, prediction_net, stdlib, torch, wandb
agents/*.py       imports: event_log, world_model, simulation, stdlib, anthropic
orchestrator.py   imports: event_log, world_model, simulation, agents/*, stdlib, langgraph
api.py            imports: event_log, world_model, simulation, stdlib, fastapi
main.py           imports: all of the above
```

If a file needs to import something not in its allowed list: that is a
design error. Surface it in MASTER_STATE.md under SPEC ISSUES FOUND
rather than adding a cross-cutting import.

---

## FILE CHECKLIST

```
backend/
├── event_log.py          Task 1.1  ← built first, never modified after
├── api.py                Task 1.1  ← WebSocket + REST
├── simulation.py         Task 1.2  ← SurROLTissueEnv + SimConfig
├── belief_state.py       Task 1.3
├── data_collector.py     Task 1.3  ← populates training.db
├── prediction_net.py     Task 1.4
├── world_model.py        Task 1.5
├── mpc_agent.py          Task 1.8  ← uses WorldModel directly
├── orchestrator.py       Task 1.6
├── agents/
│   ├── exploration.py    Task 1.7  ← Claude API every 5 steps
│   ├── capability.py     Task 1.7
│   └── reactive.py       Task 1.7  ← zero WorldModel imports
├── train.py              Task 1.3  ← initial training script
├── main.py               Task 1.9
├── data/
│   ├── events.db                   ← git-ignored
│   └── training.db                 ← git-ignored
└── models/
    └── prediction_network.pt       ← git-ignored

tests/
├── test_event_log.py     Task 1.1
├── test_simulation.py    Task 1.2
├── test_belief_state.py  Task 1.3
├── test_prediction_net.py Task 1.4
├── test_world_model.py   Task 1.5
├── test_orchestrator.py  Task 1.6
├── test_agents.py        Task 1.7
└── test_api.py           Task 1.9
```
