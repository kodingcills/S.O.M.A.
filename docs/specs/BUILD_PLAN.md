# BUILD_PLAN.md
# SOMA — Build Plan
# Authoritative for: sprint structure, task order with time estimates,
# Oh My OpenCode agent routing, training timing, checkpoint gates,
# scope reduction order, fallback hierarchy, submission checklist.
#
# Read this at the start of every sprint.
# Update MASTER_STATE.md after every task.
# The clock rules in this document are not suggestions.

---

## SPRINT OVERVIEW

```
Sprint 1  Hours 0–12    Backend loop end-to-end, training running
Sprint 2  Hours 12–28   Frontend live, all three views functional
Sprint 3  Hours 28–36   Polish, video, submission, pitch rehearsal
```

**Hard deadlines — non-negotiable:**

| Hour | Deadline | Consequence if missed |
|---|---|---|
| 3 | Data collection running in background | Training delayed → model not ready until hour 8+ |
| 12 | Sprint 1 checkpoint passes | Sprint 2 starts with broken foundation |
| 28 | Sprint 2 checkpoint passes | Sprint 3 starts with broken demo |
| 30 | Demo video recorded | No fallback exists |
| 33 | Devpost submitted | Late = disqualified |
| 35.5 | Everything stops | Pitch rehearsal only |

---

## OH MY OPENCODE AGENT ROUTING

Each task maps to a specific Oh My OpenCode agent. Wrong agent choice
wastes time on model switching and context re-establishment.

| Task | Agent | Reason |
|---|---|---|
| 1.1 Event log + API | Sisyphus | Straightforward implementation |
| 1.2 SurRoL environment | Sisyphus + Librarian | Librarian pulls SurRoL live docs via Context7 |
| 1.3 BeliefState | Sisyphus | Clear spec, pure numpy |
| 1.4 PredictionNetwork | Sisyphus | PyTorch boilerplate, well-specified |
| 1.5 WorldModel | **Oracle** | Singleton, two-lock model, async complexity |
| 1.6 LangGraph orchestrator | **Oracle** | Graph architecture, Command routing, TypedDict |
| 1.7 Exploration + Capability agents | **Oracle** | Claude API tool use, streaming, circuit breaker |
| 1.8 MPC agent | Sisyphus | Math-heavy but fully specified |
| 1.9 main.py integration | Sisyphus | Wiring, startup sequence |
| 2.1 Frontend scaffold | Sisyphus | Setup, types, hooks |
| 2.2 Canvas + nodes | **Oracle** | React Flow v12, pure function invariant |
| 2.3 Simulation panel | Sisyphus | HTML Canvas drawing, MJPEG stream |
| 2.4 Heatmap + AgentFeed | Sisyphus + Librarian | Librarian for Plotly API |
| 2.5 Comparison view | Sisyphus | Straightforward composition |
| 2.6 W&B + debug endpoints | Sisyphus | Integration, not architecture |
| Any integration failure | **Oracle** | Architecture/debugging specialist |
| Planning Task 1.6 graph topology | **Prometheus** | Run before Oracle starts 1.6 |

**Before Task 1.6:** Run Prometheus to plan the LangGraph graph topology.
Changing graph structure after agents are implemented requires rebuilding
the orchestrator. One planning session saves two hours of rework.

---

## OPENCODE TASK PROMPT TEMPLATE

Use this structure for every task. Fill in all sections. Paste current
MASTER_STATE.md — do not assume agents remember previous sessions.

```
/agent {sisyphus|oracle|librarian}   ← specify the agent

SOMA BUILD — Task {ID}: {Name}

=== CURRENT BUILD STATE ===
{paste full MASTER_STATE.md here}
===========================

TASK (one sentence)
{What to build, exactly.}

WHAT EXISTS (import these, don't reimplment)
{exact filenames and key interfaces available}

WHAT TO BUILD
Files to create or modify:
  {list}
Key classes and methods:
  {list with brief behavioral description}
Spec reference (read before writing code):
  docs/specs/{FILE}.md sections: {SECTION LIST}

KNOWN PITFALLS FOR THIS COMPONENT
1. {symptom} — {cause} — {fix}
2. {symptom} — {cause} — {fix}

SUCCESS CONDITIONS (run these, paste output)
{exact commands and expected output}

OUTPUT FORMAT
When complete, provide:
1. Summary of files created/modified
2. Exact test command output (copy-paste)
3. Deviations from spec: {none} or {description + reason}
4. MASTER_STATE.md update block (exact format below)

MASTER_STATE.md UPDATE BLOCK:
## COMPLETED: Task {ID} — {Name}
  Files: {list}
  Key interfaces:
    {function_name}({params}) → {return_type}
  Deviations: {none | description + downstream impact}
  Tests passed: {test:component:invariant_name}
  Verified contracts: {C1, C2, ...} (from ARCHITECTURE.md)

DO NOT BUILD (explicitly out of scope)
{list of adjacent things to NOT implement}
```

---

## MASTER_STATE.MD FORMAT

Keep this file at project root. Every OpenCode session reads it first
and writes an update block at the end. Never skip the update.

```markdown
# SOMA MASTER STATE
Updated: {timestamp}
Sprint: {1|2|3}
Build health: {GREEN|YELLOW|RED}
Hours elapsed: {N}

## COMMITTED STACK
Simulation:    SurRoL + SurROLTissueEnv(TaskBase)
World model:   Custom PredictionNetwork (818→512→256→128, 4 heads)
Orchestration: LangGraph (MemorySaver checkpointer)
Agent AI:      Claude API claude-sonnet-4-6 (tool_choice="any")
3D render:     PyBullet via SurRoL + MJPEG stream
Dashboard:     Weights & Biases
Heatmap:       react-plotly.js
Canvas:        @xyflow/react v12
State vector:  806 float32 values
Action vector: 12 float32 values
Network input: 818 float32 values

## TRAINING STATUS
Samples collected: {N} / 50,000
Training status:   {not_started|running|complete}
val_tissue_mse:    {value|pending}
val_damage_acc:    {value|pending}
Model checkpoint:  {none|exists}

## COMPLETED TASKS
{task entries from each session}

## IN PROGRESS
{current task if any}

## KNOWN DEVIATIONS
{deviations logged here — name, reason, downstream impact}

## INTEGRATION CONTRACTS VERIFIED
[ ] C1: SurROLTissueEnv.get_state_vector() → (806,) float32
[ ] C2: BeliefState.to_input_vector() → (806,) float32, no error_map
[ ] C3: WorldModel.fine_tune() awaitable, runs in to_thread()
[ ] C4: agent_spawned written BEFORE episode loop
[ ] C5: agent_completed written AFTER fine_tune() with non-null error fields
[ ] C6: WebSocket replays all history on connect
[ ] C7: payload deserialized to dict, never raw string
[ ] C8: SurROLTissueEnv(seed=42).reset() reproducible

## NEXT TASK
Task: {ID} — {Name}
Agent: {sisyphus|oracle|librarian}
Prerequisites confirmed: {yes|no — list what's missing}
Spec sections to read: {list}

## MEASURED PERFORMANCE LOG
SurRoL env.step() latency:          [PENDING] ms
WorldModel.predict() latency:        [PENDING] ms
Data collection time (50K samples):  [PENDING] min
Initial training time (MPS):         [PENDING] min
Fine-tune time per agent (5 epochs): [PENDING] sec
Canvas render time (50 nodes):       [PENDING] ms
WebSocket replay time (500 events):  [PENDING] ms
CONTACT_SCALE_FACTOR measured:       [PENDING] — see SIMULATION.md note
```

---

## SPRINT 1 — BACKEND (Hours 0–12)

### Sprint 1 Goal

Complete, running, end-to-end loop:
simulation → world model prediction → error computed → orchestrator decides
→ exploration agent spawns → fine-tuning runs → error decreases → events log
visible via curl. Frontend not required. Verify with sqlite3 and curl.

### Timing Strategy

```
Hour 0.00  Task 1.1  Event log + API + MJPEG skeleton    (45 min)
Hour 0.75  Task 1.2  SurRoL environment                  (2.5 hr)
Hour 3.25  Task 1.3  BeliefState code THEN start data    (30 min code)
Hour 3.75  ─── DATA COLLECTION RUNNING IN BACKGROUND ───
Hour 3.75  Task 1.4  PredictionNetwork + train.py        (1.5 hr, parallel)
Hour 5.25  ─── START TRAINING (once data hits 50K) ──────
Hour 5.25  Task 1.5  WorldModel class                    (1 hr, parallel)
Hour 6.25  Task 1.6  LangGraph orchestrator              (1.5 hr, parallel)
Hour 7.75  Task 1.7  Exploration + Capability + Reactive (2.5 hr)
Hour 10.25 Task 1.8  MPC agent                           (1.5 hr)
Hour 11.75 Task 1.9  main.py integration + full loop     (30 min)
Hour 12.00 ─── SPRINT 1 CHECKPOINT ─────────────────────
```

⚑ **CLOCK RULE — DATA COLLECTION:** If data collection has not started
by hour 3, stop everything and run it immediately. This is the only
interrupt that overrides task dependency order. Training cannot begin
without it, and training takes 60-90 minutes on MPS.

---

### Task 1.1 — Event Log + API Skeleton
**Time:** 45 min | **Agent:** Sisyphus | **Reads:** ARCHITECTURE.md

**Build:**
- `backend/event_log.py`: `init_db()`, `write_event()`, `get_events_since()`, `get_events_for_agent()`, `set_broadcast_callback()`. WAL mode on every connection. Payload serialized/deserialized as JSON dict, never raw string.
- `backend/api.py`: FastAPI app, `ConnectionManager`, all REST endpoints from ARCHITECTURE.md, WebSocket `/ws/events` with history replay on connect, ping/pong heartbeat, MJPEG stream endpoint `/sim/{sim_id}/stream` (stub that returns 404 until sim exists).
- `backend/main.py`: minimal skeleton — `init_db()`, `uvicorn.run()`.
- `backend/data/` and `backend/models/` directories.

**Do not build:** simulation, ML, agents, browser tests.

**Success condition:**
```bash
# Terminal 1
uvicorn backend.main:app --reload

# Terminal 2
curl http://localhost:8000/health
# Expected: {"status":"starting","world_model_version":0,...}

python -c "
from backend.event_log import init_db, write_event, get_events_since
init_db()
eid = write_event('system_ready', test_value=True)
events = get_events_since(0)
assert len(events) == 1
assert events[0]['event_type'] == 'system_ready'
assert isinstance(events[0]['payload'], dict)   # NOT a string
assert events[0]['payload']['test_value'] == True
print(f'PASS: event id={eid}')
"
```

---

### Task 1.2 — SurRoL Environment
**Time:** 2.5 hr | **Agent:** Sisyphus + Librarian | **Reads:** SIMULATION.md (all)

**SurRoL installation (run first, before writing any code):**
```bash
cd soma
git clone https://github.com/med-air/SurRoL.git
cd SurRoL && uv pip install -e .
cd ..
uv add pybullet==3.2.6
python -c "import surrol; from surrol.tasks.task_base import TaskBase; print('SurRoL OK')"
```

If install fails within 45 minutes: skip SurRoL and implement custom PyBullet
simulation (see Scope Cut 3). Do not spend more than 45 minutes on SurRoL
installation. The rest of the system doesn't care which sim it uses.

**Build:**
- `backend/simulation.py`: `SurROLTissueEnv(TaskBase)`, `SimConfig`, `create_env()` async factory (with `_pybullet_init_lock`). Override `_build_world()`, `_get_obs()`, `_compute_reward()`, `_check_success()`, `_check_failure()`. Tissue grid (4 numpy arrays), vessel placement, target placement, `get_state_vector()` → (806,) float32, `action_to_surrol()` translator.

**Do not build:** BeliefState, WorldModel, any ML code.

**Success condition:**
```python
# Run: pytest tests/test_simulation.py -v
# Minimum passing:
import asyncio, numpy as np
from backend.simulation import SurROLTissueEnv, SimConfig, action_to_surrol

async def test():
    env_a = SurROLTissueEnv(SimConfig(seed=42)); env_a.reset()
    env_b = SurROLTissueEnv(SimConfig(seed=42)); env_b.reset()

    v = env_a.get_state_vector()
    assert v.shape == (806,), f"Expected (806,), got {v.shape}"
    assert v.dtype == np.float32
    assert not np.any(np.isnan(v))

    assert np.array_equal(env_a.vascularity, env_b.vascularity), "Seed not reproducible"
    print("Task 1.2 PASS")

asyncio.run(test())
# Also: PyBullet window must open showing PSM robot arm
```

---

### Task 1.3 — BeliefState + Data Collection
**Time:** 30 min code + 15-20 min runtime | **Agent:** Sisyphus | **Reads:** WORLD_MODEL.md (BeliefState section), SIMULATION.md (Data Collection section)

**Build:**
- `backend/belief_state.py`: `BeliefState` dataclass, `update_from_observation()`, `update_prediction_error()` (EMA α=0.3, tissue only), `to_input_vector()` → (806,) float32 no error_map, `get_regional_errors()` → exactly 6 keys, `snapshot()` → JSON-serializable dict.
- `backend/data_collector.py`: `collect_training_data(env, n_samples=50_000)`. Uses `SimConfig(seed=0, dof=1, n_vessels=2)`. Stores blobs in `training.db`. Logs every 5,000 samples. Returns when complete.
- `backend/data/training.db` schema (create in `init_training_db()`).

**Immediately after writing code — start data collection:**
```bash
# Run in a dedicated terminal — leave running
python -c "
import asyncio
from backend.simulation import SurROLTissueEnv, SimConfig
from backend.data_collector import collect_training_data
env = SurROLTissueEnv(SimConfig(seed=0, dof=1, n_vessels=2))
env.reset()
collect_training_data(env, n_samples=50_000)
"
# Move on to Task 1.4 while this runs
```

**Success condition:**
```python
# test:belief:circular_input — MUST PASS
from backend.belief_state import BeliefState
import numpy as np
b = BeliefState()
v1 = b.to_input_vector().copy()
b.update_prediction_error(np.ones((16,16), dtype=np.float32),
                          np.zeros((16,16), dtype=np.float32))
v2 = b.to_input_vector()
assert np.array_equal(v1, v2), "CIRCULAR INPUT INVARIANT VIOLATED"
assert b.to_input_vector().shape == (806,)
errors = b.get_regional_errors()
assert set(errors.keys()) == {
    'upper_left','upper_right','lower_left','lower_right',
    'tool_tissue_boundary','surgical_target_vicinity'
}
print("Task 1.3 PASS")
```

---

### Task 1.4 — PredictionNetwork + train.py
**Time:** 1.5 hr (build while data collects) | **Agent:** Sisyphus | **Reads:** WORLD_MODEL.md (Prediction Network + Initial Training sections)

**Build:**
- `backend/prediction_net.py`: `PredictionNetwork(nn.Module)`. Encoder: `Linear(818,512)→LayerNorm→ReLU→Dropout(0.1)`, `Linear(512,256)→LayerNorm→ReLU→Dropout(0.1)`, `Linear(256,128)→LayerNorm→ReLU`. Four heads: tissue(128→256, Sigmoid), damage(128→1, Sigmoid), vessel\_risk(128→5, Sigmoid), instrument(128→4, no activation). `PredictedState` dataclass. Device selection: MPS → CUDA → CPU.
- `backend/train.py`: `train_prediction_network()`. Adam lr=0.001, CosineAnnealingLR, early stopping patience=10 on val\_tissue\_mse. Loss = 0.6×tissue\_mse + 0.3×damage\_bce + 0.1×vessel\_bce. W&B logging every epoch. Saves checkpoint to `models/prediction_network.pt`.

**Do not run training yet** — wait for 50,000 samples in training.db.

**Success condition:**
```python
import torch, numpy as np
from backend.prediction_net import PredictionNetwork

net = PredictionNetwork()
assert not any(isinstance(m, torch.nn.BatchNorm1d) for m in net.modules()), "BatchNorm found"

x = torch.randn(4, 818)
t, d, v, inst = net(x)
assert t.shape == (4, 256); assert d.shape == (4, 1)
assert v.shape == (4, 5);   assert inst.shape == (4, 4)
assert t.min() >= 0 and t.max() <= 1      # Sigmoid
assert d.min() >= 0 and d.max() <= 1      # Sigmoid

net.eval()
with torch.no_grad():
    import time; start = time.time()
    for _ in range(100): net(torch.randn(1,818))
    latency_ms = (time.time()-start)/100*1000
assert latency_ms < 10, f"Inference {latency_ms:.1f}ms — too slow"
print(f"Task 1.4 PASS — inference {latency_ms:.1f}ms")
```

---

### Task 1.4b — Run Initial Training
**Time:** 60-90 min background | **Agent:** (you run this, not OpenCode)

**Prerequisite check:**
```bash
python -c "
import sqlite3
c = sqlite3.connect('backend/data/training.db').execute('SELECT COUNT(*) FROM samples')
n = c.fetchone()[0]
assert n >= 50_000, f'Only {n} samples — wait for collection to finish'
print(f'{n} samples ready')
"
```

**Start training (dedicated terminal):**
```bash
python -m backend.train
# Expected output per epoch:
# Epoch 1/50 | train_loss: 0.421 | val_tissue_mse: 0.089 | damage_acc: 73.2%
# Training target: val_tissue_mse < 0.05 AND damage_acc > 85%
# If not met at epoch 50: script automatically continues to epoch 100 at lr=0.0001
```

**While training runs:** Build Tasks 1.5 and 1.6 in parallel.
Training uses MPS on Apple Silicon. Expected time: 30-45 min on MPS, 60-90 min on CPU.

---

### Task 1.5 — WorldModel Class
**Time:** 1 hr | **Agent:** Oracle | **Reads:** WORLD_MODEL.md (WorldModel Class section), ARCHITECTURE.md (WORLD MODEL SINGLETON INVARIANT, EVENT LOOP NON-BLOCKING INVARIANT)

**Build:**
- `backend/world_model.py`: `WorldModel` class. Two locks: `_predict_lock`, `_finetune_lock`. `predict(state_806, action_12) → PredictedState` (sync, acquires predict\_lock). `update(state, action, next_state)` (sync, updates belief, adds to replay). `async fine_tune(samples)` (acquires both locks, runs `_fine_tune_sync` in `asyncio.to_thread()`, increments version, W&B log, saves checkpoint). `_replay_buffer` as deque(maxlen=100\_000). `load(path)` with graceful handling when no checkpoint exists. Fine-tune queue: `asyncio.Queue(maxsize=10)` + background worker.

**Do not instantiate WorldModel in this file.** It is instantiated in `main.py` and passed everywhere.

**Success condition:**
```bash
# test:world_model:singleton
grep -r "WorldModel(" backend/ | grep -v "main.py" | grep -v "test_" | grep -v "#"
# Must return empty

# test:world_model:nonblocking (from ARCHITECTURE.md)
pytest tests/test_world_model.py::test_nonblocking -v
# Must pass: fine_tune() does not block WebSocket ticks
```

---

### Task 1.6 — LangGraph Orchestrator
**Time:** 1.5 hr | **Agent:** Oracle (after Prometheus planning) | **Reads:** AGENTS.md (LangGraph Graph Definition section, Orchestrator section)

**Before starting:** Run Prometheus to plan the graph topology. Ask Prometheus:
`"Plan the LangGraph StateGraph for SOMA's orchestrator. State needs regional_errors,
active_agents, circuit breaker state. Nodes: orchestrator, spawn_explore, unlock_dof,
collect_status. Use Send for fan-out. Show the full graph.compile() call."`

**Build:**
- `backend/orchestrator.py`: `SomaOrchestratorState(TypedDict)` with `Annotated` list fields. `build_soma_graph()` function. `orchestrator_node`, `route_orchestrator`, `spawn_exploration_node` (using `Send`), `unlock_dof_node`, `collect_status_node`. `MemorySaver` checkpointer. Circuit breaker state in graph state. Constants: `EXPLORE_THRESHOLD=0.15`, `MAX_CONCURRENT_AGENTS=4`, `ORCHESTRATOR_SLEEP_S=2.0`, `CIRCUIT_BREAKER_THRESHOLD=3`.

**Do not build:** agent implementations (Task 1.7), MPC agent (Task 1.8).

**Success condition:**
```python
from backend.orchestrator import build_soma_graph, SomaOrchestratorState

# Graph must compile without error
graph = build_soma_graph(mock_world_model, mock_sim_manager)
assert graph is not None

# Route function must return valid strings
from backend.orchestrator import route_orchestrator
high_error = SomaOrchestratorState(
    regional_errors={'upper_left': 0.8, ...all 6 keys...},
    global_mean_error=0.6, active_dof=1, active_agents=[],
    completed_agents=[], failed_agents=[], cycle_count=0,
    unlock_requested=False, target_dof=0,
    region_failure_counts={}, region_backoff_until={},
    consecutive_fine_tune_failures=0
)
result = route_orchestrator(high_error)
assert result in ('spawn_explore', 'unlock_dof', 'wait')
print(f"Task 1.6 PASS — route returned '{result}'")
```

---

### Task 1.7 — Exploration + Capability + Reactive Agents
**Time:** 2.5 hr | **Agent:** Oracle | **Reads:** AGENTS.md (all agent sections, Claude API tool use, production engineering)

**Build:**
- `backend/agents/exploration.py`: `ExplorationAgent`. Claude API with `tool_choice={"type": "any"}`, streaming via `client.messages.stream()`, retry with exponential backoff, `asyncio.Semaphore(10)` rate limiter, `AgentFeedBuffer`. Call Claude every 5 steps. `asyncio.timeout(600)`. Event sequence: orchestrator writes `agent_spawned` before `Send`, agent writes `agent_step` every 10 steps, agent writes `agent_completed` after `fine_tune()`.
- `backend/agents/capability.py`: `CapabilityAgent`. DOF unlock on primary + comparison atomically. 1000-step data collection. `fine_tune()` with `epochs=10`. `capability_unlocked` event before `agent_completed`.
- `backend/agents/reactive.py`: `ReactiveAgent`. Zero imports from `world_model`, `anthropic`, `orchestrator`. Greedy policy. `asyncio.sleep(0.05)` in loop.

**Do not build:** MPC agent (Task 1.8), curriculum agent (cut from scope).

**Success condition:**
```bash
# test:agents:reactive_isolation — MUST PASS
grep -E "from backend\.(world_model|prediction_net|belief_state|orchestrator)" \
    backend/agents/reactive.py
# Must return empty

# test:agents:tool_choice — Claude API must return tool call
pytest tests/test_agents.py::test_tool_choice_forces_structured_output -v

# Manual: start system, watch for agent_spawned event
python -c "
import asyncio
from backend.event_log import init_db, get_events_since

async def watch():
    init_db()
    import time; start = time.time()
    # Manually trigger an exploration run
    from backend.agents.exploration import ExplorationAgent
    from unittest.mock import MagicMock
    agent = ExplorationAgent('exp_test01', 'root', 'upper_left', 0.8)
    # ... inject mocks ...
    await agent.run()
    events = get_events_since(start)
    types = [e['event_type'] for e in events]
    assert 'agent_step' in types
    assert 'agent_completed' in types
    completed = next(e for e in events if e['event_type']=='agent_completed')
    assert completed['error_before'] is not None
    assert completed['error_after'] is not None
    print('Task 1.7 PASS')

asyncio.run(watch())
"
```

---

### Task 1.8 — MPC Agent
**Time:** 1.5 hr | **Agent:** Sisyphus | **Reads:** WORLD_MODEL.md (MPC Planning section)

**Build:**
- `backend/mpc_agent.py`: `MPCAgent`. `_generate_candidates(state_vec) → list[np.ndarray]` — exactly 8 action vectors. `_compute_value(current_state_vec, predicted) → float`. `async run()` loop: acquire `_predict_lock`, run 8 predictions in `asyncio.to_thread()`, execute best action via `asyncio.to_thread(env.step, ...)`, call `world_model.update()`, `write_event('simulation_step', ...)`, `await asyncio.sleep(0.05)`.

**Do not build:** anything not in `mpc_agent.py`.

**Success condition:**
```python
# Must generate exactly 8 candidates
from backend.mpc_agent import MPCAgent
import numpy as np
mpc = MPCAgent(mock_world_model, mock_env)
state = np.zeros(806, dtype=np.float32)
candidates = mpc._generate_candidates(state)
assert len(candidates) == 8
assert all(c.shape == (12,) for c in candidates)
assert all(c.dtype == np.float32 for c in candidates)
print("Task 1.8 PASS")
```

---

### Task 1.9 — main.py Integration
**Time:** 30 min | **Agent:** Sisyphus | **Reads:** ARCHITECTURE.md (Startup Sequence section)

**Build:**
- `backend/main.py` (final version): full lifespan context manager in exact startup order from ARCHITECTURE.md. Conditional training start. WorldModel singleton creation. All asyncio tasks started. `set_broadcast_callback` registered. `system_ready` event written.

**Success condition (Sprint 1 Checkpoint — see below).**

---

## SPRINT 1 CHECKPOINT (Hour 12)

Run every item. Do not start Sprint 2 if any fails.

```bash
# Start system
uvicorn backend.main:app

# In another terminal — run all checks:

# 1. Health endpoint
curl http://localhost:8000/health
# Expected: status "ok" or "training"

# 2. Events exist
curl "http://localhost:8000/events?since=0" | python -c "
import json,sys; events=json.load(sys.stdin)
types = {e['event_type'] for e in events}
print('Event types:', types)
assert 'system_ready' in types
assert 'simulation_step' in types
print('EVENTS OK')
"

# 3. Full loop running (wait 60s after start)
sqlite3 backend/data/events.db "
SELECT event_type, COUNT(*) as n
FROM events
GROUP BY event_type
ORDER BY n DESC
LIMIT 10;"
# Must see: simulation_step, world_model_updated or similar

# 4. WebSocket works
# Install wscat: npm install -g wscat
wscat -c ws://localhost:8000/ws/events
# Must see events streaming. Ctrl+C to exit.

# 5. Agent activity (may need to wait 2-5 min for first spawn)
sqlite3 backend/data/events.db "
SELECT event_type, agent_id, error_before, error_after
FROM events
WHERE event_type IN ('agent_spawned','agent_completed','agent_failed')
ORDER BY id DESC LIMIT 5;"
```

**Checkpoint gate — ALL must be true:**
```
□ /health returns status 'ok' (not 'starting' or 'training')
□ simulation_step events visible in DB
□ world_model_updated events visible
□ agent_spawned event appeared
□ agent_completed event appeared with non-null error_before and error_after
□ WebSocket streams events live
□ pytest tests/ -v passes all must-pass tests
□ No crashed asyncio tasks in logs
```

**If behind at hour 12:** Do not start Sprint 2. Use Scope Reduction in order below.

---

## SPRINT 2 — FRONTEND (Hours 12–28)

### Sprint 2 Goal

All three views functional with live data. Error reduction animation tested manually
5× and confirmed working. MJPEG stream shows PyBullet 3D render.

---

### Task 2.1 — Frontend Scaffold
**Time:** 1.5 hr | **Agent:** Sisyphus | **Reads:** INTERFACE.md (Tech Stack, TypeScript Interfaces, Data Architecture sections)

**Build:**
- `frontend/` Vite React TypeScript project with exact deps from INTERFACE.md.
- `frontend/src/types/index.ts` — all interfaces, complete before any component.
- `frontend/src/utils/colors.ts` — all constants, `errorToColor()`, `agentColor()`, `REGION_LABELS`.
- `frontend/src/hooks/useEventLog.ts` — WebSocket hook with 100ms batching, exponential backoff reconnect, REST fallback. `useReducer` for events array, cap at 10,000.
- `frontend/src/hooks/useHealth.ts`, `useDetail.ts`.
- `frontend/src/App.tsx` — shell with view switching, `display: none` pattern (not unmount).

**Success condition:**
```bash
npm run dev
# Browser opens localhost:5173
# Browser console shows events arriving from WebSocket
# No TypeScript errors: npm run build
```

---

### Task 2.2 — Canvas View + Node Components
**Time:** 3 hr | **Agent:** Oracle | **Reads:** INTERFACE.md (Canvas View, Node Components sections), ARCHITECTURE.md (PURE CANVAS STATE INVARIANT, REACT FLOW VERSION INVARIANT)

**Build:**
- `frontend/src/utils/buildCanvasState.ts` — pure function, `useMemo`'d at call site.
- `frontend/src/components/Canvas/index.tsx` — `ReactFlow` with `@xyflow/react` v12 hooks (`useNodesState`, `useEdgesState`). `NODE_TYPES` defined outside component. `nodesDraggable={false}`, `nodesFocusable={false}`. Camera follow debounced 2000ms.
- Node components: `RootNode`, `ExplorationNode`, `CapabilityNode`, `WorldModelNode`, `SimulationNode`. Spawn pop animation, glow pulse, completion particles. Spawning→active timer keyed on `event.timestamp`.

**Critical imports (wrong version = blank canvas):**
```typescript
import { ReactFlow, useNodesState, useEdgesState, useReactFlow } from '@xyflow/react'
import '@xyflow/react/dist/style.css'   // in App.tsx, exactly once
```

**Success condition:**
```bash
# Manual tests (automated tests cannot catch visual failures):
# 1. Root node visible at (0,0) immediately on load
# 2. Trigger agent_spawned event manually via:
curl -X POST http://localhost:8000/debug/trigger_agent
# 3. New node appears with pop animation within 1 second
# 4. buildCanvasState called twice with same events → deep equal output
# 5. Canvas does NOT re-layout on every new WebSocket event
```

---

### Task 2.3 — Simulation Panel + MJPEG Stream
**Time:** 2 hr | **Agent:** Sisyphus | **Reads:** INTERFACE.md (SimulationPanel section, MJPEG stream section)

**Build:**
- Backend: `pybullet_frame_generator()` async generator in `api.py`. `GET /sim/{sim_id}/stream` endpoint returning MJPEG `StreamingResponse`. `p.getCameraImage()` wrapped in `asyncio.to_thread()`. 15fps frame rate.
- Frontend: `frontend/src/components/DetailView/SimulationPanel.tsx`. MJPEG stream as `<img src="/sim/{simId}/stream">`. HTML Canvas 2D overlay for vessels, target, prediction error cells. `requestAnimationFrame` loop for vessel damage flash + target pulse.

**Success condition:**
```bash
# MJPEG stream works in browser:
# Open: http://localhost:8000/sim/primary/stream
# Must see: live PyBullet 3D render of PSM robot arm, refreshing ~15fps
# No audio, no JS required

# Frontend:
# Click root node → detail view opens
# SimulationPanel shows live MJPEG stream
# Canvas overlay shows vessel circles and target ring
```

---

### Task 2.4 — WorldModelPanel + AgentFeed
**Time:** 2 hr | **Agent:** Sisyphus + Librarian | **Reads:** INTERFACE.md (WorldModelPanel, AgentFeed sections, animation specs)

**Build:**
- `frontend/src/components/DetailView/WorldModelPanel.tsx`: `react-plotly.js` heatmap with HSL colorscale, `useMemo`'d data + layout. CSS overlay div for error reduction flash animation. Error reduction: watch `agent_completed` events with `error_before - error_after > 0.05`, trigger `@keyframes heatmapFlash` 600ms animation. `recharts` BarChart for regional confidence bars. 6-circle DOF progress display.
- `frontend/src/components/DetailView/AgentFeed.tsx`: reverse chronological, exact entry format from INTERFACE.md, Claude reasoning rendered from `payload.reasoning`. Virtual list (react-window) when >500 entries.

**The error reduction animation is the demo's primary moment.
Test it manually 5× before marking this task complete.**

**Success condition:**
```bash
# 1. Heatmap renders with correct colors:
#    errorToColor(0.0) → browser shows green (not brown, not blue)
#    errorToColor(0.5) → browser shows yellow (critical — if brown, RGB interpolation)
#    errorToColor(1.0) → browser shows red

# 2. Error reduction animation fires:
# Manually trigger: force a world_model.belief.prediction_error_map update
# via the /debug/reset_belief endpoint (implement this in Task 2.6)
# Then trigger agent_completed with error_before=0.5, error_after=0.2
# Watch: heatmap cell flashes white → yellow → settles to new color over 600ms
# This must be visible and satisfying. Retest 5× before proceeding.
```

---

### Task 2.5 — Comparison View
**Time:** 1.5 hr | **Agent:** Sisyphus | **Reads:** INTERFACE.md (Comparison View section)

**Build:**
- `frontend/src/components/CompareView/index.tsx`. Two `<img>` MJPEG streams side by side (`/sim/primary/stream` and `/sim/comparison/stream`). Poll `/detail/primary` and `/detail/comparison` every 300ms. Stats bar with better-value bolded green. Vessel damage toast notification.

**Success condition:**
```bash
# Click COMPARE button → both streams visible
# Stats bar shows steps, tissue integrity, vessels damaged, task status
# Manually damage a vessel in comparison_sim
# Vessel damage toast appears and fades after 3 seconds
```

---

### Task 2.6 — W&B + Debug Endpoints
**Time:** 30 min | **Agent:** Sisyphus

**Build in `backend/api.py`:**
```python
GET /debug/trigger_agent  # artificially set one region's error to 0.8
GET /debug/reset_belief   # set all regional errors to 0.5 (demo prep)
```

**W&B** already wired in `backend/train.py` and `backend/world_model.py`. Log the W&B dashboard URL to console at startup:
```python
import wandb
run = wandb.init(project="soma-surgical", name=f"demo-{int(time.time())}")
print(f"\n🔗 W&B Dashboard: {run.url}\n")
```

Open this URL in a browser tab during the demo. It provides independent
evidence of learning to judges who ask "how do you know it's actually improving?"

---

## SPRINT 2 CHECKPOINT (Hour 28)

```bash
# Start full system: uvicorn backend.main:app + npm run dev

# Canvas view
□ Root node visible on load
□ Nodes appear within 2s of agent_spawned event
□ Node pop animation fires (slight overshoot)
□ Completed nodes turn green with particles
□ Canvas does NOT flicker on every event
□ Camera follows rightmost active node

# Detail view
□ Click any node → SimulationPanel shows MJPEG stream
□ Canvas overlay: vessels visible as circles
□ Canvas overlay: target visible with pulse
□ WorldModelPanel: heatmap renders with HSL colors (yellow at 0.5)
□ Error reduction animation fires (tested 5× manually)
□ Regional confidence bars update
□ DOF circles show correct state
□ AgentFeed shows events with correct format

# Comparison view
□ Both MJPEG streams load simultaneously
□ Stats bar updates every 300ms
□ Vessel damage toast fires

# If any item fails: fix before Sprint 3. Sprint 3 = polish only.
```

---

## SPRINT 3 — POLISH + SUBMISSION (Hours 28–36)

### Sprint 3 Goal

Demo rehearsed 3×. Video recorded. Devpost submitted. Fallbacks ready.

---

### Task 3.1 — Visual Polish
**Time:** 2 hr | Fix in priority order, stop when time runs out

```
Priority 1: Error reduction animation — test 10×, must be reliable
Priority 2: Node spawn timing — spawning→active exactly 500ms
Priority 3: Camera follow — smooth, not jarring
Priority 4: Heatmap color accuracy — 0.5 must be yellow, not brown
Priority 5: Comparison stats — better value visually obvious
Priority 6: Top bar stats — update every second
Priority 7: 1920×1080 layout — no overflow, no broken flexbox
Priority 8: JetBrains Mono loading — agent feed font correct
Priority 9: Loading overlay — shows during training, dismisses on 'ok'
```

**Do not add features.** Fix only what exists.

---

### Task 3.2 — Demo Video (HARD DEADLINE: Hour 30)
**Time:** 30 min | **Do this even if the system is not fully polished**

Record a 2-minute screen recording. If the system breaks during the live demo,
this video plays. It must exist by hour 30 regardless of system state.

```
[0:00-0:15] Canvas view with nodes growing
  "This is SOMA. Every node is an autonomous decision."

[0:15-0:45] Show exploration agent spawning + heatmap region going red
  "The world model knows where it's wrong. It sends an agent."

[0:45-1:00] Show error reduction animation firing
  "The agent comes back. The region turns green. The model improved."

[1:00-1:30] Switch to comparison view
  "Same anatomy. No world model on the right. Watch the vessel."
  [Show reactive agent damaging a vessel]
  "SOMA avoided it. It predicted the vessel was there."

[1:30-1:50] Show DOF progression (if unlocked)
  "We seeded it with one joint. It earned the rest."

[1:50-2:00] Zoom out to full canvas tree
  "Every branch: an insight. Every green node: understanding."

Save as: demo_fallback.mp4
```

---

### Task 3.3 — Devpost Submission (HARD DEADLINE: Hour 33)

**Repository prep:**
```bash
# Clean repo
echo "backend/data/*.db\nbackend/models/*.pt\nbackend/.venv\nfrontend/node_modules" >> .gitignore
git add -A && git commit -m "final: submission state"
git push origin main
```

**Prize categories to submit (verify against current Bitcamp rules):**
- Best Machine Learning Track Hack (track prize)
- Best Bitcamp Hack (grand prize — primary target)
- Best Moonshot Hack (secondary Bitcamp)
- Peraton Data Visualization (sponsor — Plotly alignment explicit in pitch)
- Check: Bitcamp Unwrapped slot rules

**Devpost content:**
- Title: `SOMA — Self-Organizing Model Architecture for Surgical Robotics`
- Tagline: `A world model that knows what it doesn't know — and acts on that.`
- Description: dead reckoning hook, three-component architecture, honest autonomy framing, adjacent domain applications. See project narrative documents.
- Tech stack tags: Python, PyTorch, FastAPI, React, TypeScript, @xyflow/react, Plotly, LangGraph, Claude API, PyBullet, SurRoL, WebSocket, MPS
- 4+ screenshots: canvas with tree, detail with heatmap, error reduction animation captured, comparison with vessel damage notification
- Demo video: `demo_fallback.mp4`

**Submit by 9:00 AM.** 30-minute buffer is non-negotiable.

---

### Task 3.4 — Demo Prep (Hours 35.5–36)

**Setup checklist (complete before judges arrive):**
```
□ Backend running: uvicorn backend.main:app
□ Frontend running: npm run dev (or built and served)
□ Browser: Chrome, full screen, zoom 125%
□ Canvas view open with nodes visible
□ Notifications disabled (macOS: Do Not Disturb ON)
□ Screen saver disabled
□ Power plugged in (M-series under load drains fast)
□ W&B dashboard tab open
□ demo_fallback.mp4 queued and tested full-screen playback
□ /debug/reset_belief endpoint bookmarked (if world model converged)
```

**Rehearsal protocol:** Run pitch 3×. Time each. Target 3:30 (30s buffer for Q&A).
After each run:
- What broke visually?
- What was hard to explain?
- What got the strongest reaction?
- Under 4 minutes?

**Demo recovery:** If the system has been running for hours and global error
is near zero (no agents spawning), use `/debug/reset_belief` before the demo.
This sets all regional errors to 0.5, triggering fresh agent spawns within 2 cycles.

---

## SCOPE REDUCTION ORDER

When behind schedule, cut in this exact order. Pre-evaluated — no deliberation needed.

| Cut | Time saved | Demo impact | Canvas still grows? |
|---|---|---|---|
| **1. Claude API → random actions** | 45 min | Agent feed shows no reasoning | Yes |
| **2. Comparison view** | 1.5 hr | No side-by-side demo | Yes |
| **3. Plotly → HTML Canvas heatmap** | 1 hr | Less polished, still functional | Yes |
| **4. W&B dashboard** | 30 min | No public live dashboard URL | Yes |
| **5. MJPEG stream → PNG snapshot** | 1 hr | Sim panel refreshes at 1fps, not 15fps | Yes |
| **6. Camera follow → manual pan** | 30 min | Judges must pan manually | Yes |
| **7. Node particles → border flash** | 1 hr | Less satisfying completion animation | Yes |

**Never cut:** event log, canvas tree growth, heatmap, error reduction animation.
These are the demo. Everything else is supporting cast.

---

## FALLBACK HIERARCHY

In order of desirability. Have each level ready before you need it.

```
Level 1: FULL LIVE DEMO
  Canvas + detail + comparison + MJPEG streams + W&B dashboard
  This is the target.

Level 2: CANVAS + DETAIL (no comparison)
  Cut 2 applied. Still shows self-improvement story fully.

Level 3: CANVAS ONLY (no detail panels)
  Event log drives the tree. Still visually compelling.
  Describe world model behavior verbally.

Level 4: DEMO VIDEO
  demo_fallback.mp4, narrated live over the recording.
  Judges understand the system from the video.
  Must exist by hour 30.

Level 5: BACKEND TERMINAL ONLY
  Show events streaming in a watch loop:
  watch -n0.5 "sqlite3 backend/data/events.db
    'SELECT event_type, agent_id, error_before, error_after
     FROM events ORDER BY id DESC LIMIT 8'"
  Show agents spawning and completing. Explain architecture verbally.

Level 6: DOCUMENTATION
  Show docs/specs/ directory. Walk through the architecture.
  Technical difficulty is demonstrable from the spec quality.
  This is the floor, not the ceiling.
```

---

## THE ONE QUESTION

Before every judge interaction:

> If a judge walked up right now and asked for a 4-minute demo, could you do it?

If yes: you're ready.
If no: fix the single most blocking thing before doing anything else.
