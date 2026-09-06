# RISKS.md
# SOMA — Known Risks and Mitigations
#
# HOW TO USE THIS DOCUMENT:
# Match your symptom before reading the cause.
# Diagnose causes in the listed order — most likely first.
# Check before you hypothesize. Run the cheapest test first.
# If you're debugging for > 5 min without identifying root cause:
#   come here, match the symptom, follow the diagnostic chain.
#
# Sections:
#   1. INVISIBLE FAILURES — system runs, produces wrong results silently
#   2. ENDLESS LOOP TRAPS — debugging patterns that go nowhere
#   3. TIME-CRITICAL CASCADES — miss one, miss the checkpoint
#   4. SURROL + PYBULLET RISKS
#   5. LANGGRAPH RISKS
#   6. CLAUDE API RISKS
#   7. FRONTEND RISKS
#   8. DEMO RISKS
#   9. SENIOR ENGINEERING PROTOCOL

---

## SECTION 1: INVISIBLE FAILURES

These are the most dangerous failures. The system runs. No exception is raised.
The output looks plausible. The error is invisible until you measure the outcome
against ground truth — often hours after the bug was introduced.

For each: the observable symptom is NOT an error message, it's a wrong value
or a non-result. Test these invariants explicitly after every relevant task.

---

### I-1: CIRCULAR INPUT (world model never learns)

**Probability: Medium. Consequence: Critical — demo is entirely broken.**

**How it manifests:**
Training loss decreases normally. Fine-tuning runs. W&B shows converging loss.
Agents complete. But `error_after ≈ error_before` in every `agent_completed`
event. The heatmap never turns green regardless of how many agents run.

**Why it's invisible:** The network learns to reproduce its own previous error
signal rather than tissue dynamics. This is self-consistent behavior — the
predictions *are* accurate in the sense that they correctly predict the previous
error map — but they have zero relationship to actual tissue deformation.
Loss curves look completely normal.

**Root cause:** `BeliefState.to_input_vector()` includes `prediction_error_map`
or `confidence_map`. These fields are *computed* from network output — including
them as input creates a loop: output → error_map → input → output.

**Diagnostic test (30 seconds):**
```python
from backend.belief_state import BeliefState
import numpy as np
b = BeliefState()
v1 = b.to_input_vector().copy()
b.update_prediction_error(np.ones((16,16), dtype=np.float32),
                          np.zeros((16,16), dtype=np.float32))
assert np.array_equal(v1, b.to_input_vector()), "CIRCULAR INPUT CONFIRMED"
print("Input vector unchanged after error update — invariant holds")
```

**Fix:** Remove every reference to `prediction_error_map` and `confidence_map`
from `to_input_vector()`. Retrain from scratch — the weights are corrupted.

**When to suspect this:** Any time agents complete successfully but the heatmap
doesn't change. Check this FIRST before all other hypotheses.

---

### I-2: WRONG EXPLORATION SEED (training on wrong anatomy)

**Probability: Medium. Consequence: High — fine-tuning produces no improvement.**

**How it manifests:**
Exploration agents complete. `fine_tune()` runs with no errors. World model
version increments. But regional errors don't decrease, and agents keep spawning
in the same regions indefinitely.

**Why it's invisible:** The exploration agent is collecting valid training data —
but from a different tissue layout than the primary simulation. The world model
improves at predicting that layout, which is never seen in the primary sim.
Both the agent episode and fine-tuning succeed; the data is just wrong.

**Root cause:** `SurROLTissueEnv` in exploration agent constructed with
`SimConfig(seed=N)` where N ≠ 42. Different seed → different vascularity
Gaussian centers → different tissue anatomy → different bleeding patterns →
fine-tuning teaches model about wrong tissue.

**Diagnostic test (20 seconds):**
```python
from backend.simulation import SurROLTissueEnv, SimConfig
import numpy as np
env_primary = SurROLTissueEnv(SimConfig(seed=42)); env_primary.reset()
env_explore  = SurROLTissueEnv(SimConfig(seed=???))  # whatever seed agent uses
env_explore.reset()
same = np.array_equal(env_primary.vascularity, env_explore.vascularity)
print(f"Same anatomy: {same}")  # must be True
```

**Fix:** Hardcode `seed=42` in `ExplorationAgent.run()` when constructing its env.
Do not generate a random seed. Do not use the agent_id as a seed.

---

### I-3: BATCHNORM TRAIN/EVAL DIVERGENCE (predictions wrong during demo)

**Probability: Low (if spec is followed). Consequence: Critical — demo broken.**

**How it manifests:**
Training converges normally. Unit tests pass. But when the system runs live,
MPC agent predictions are erratic — the agent makes poor decisions that look
random. The heatmap updates but with inconsistent values. System was fine in
isolation but breaks during integration.

**Why it's invisible:** `BatchNorm` stores running statistics that accumulate
during training. During inference (`eval()` mode) it uses those running stats
instead of batch statistics. If the distribution of live inference inputs differs
from training (which it will, because live states are sequential not i.i.d.),
the running stats are wrong. Predictions are biased in ways that look like
random noise. No exception is raised.

**Root cause:** `nn.BatchNorm1d` used in the encoder instead of `nn.LayerNorm`.

**Diagnostic test (10 seconds):**
```python
from backend.prediction_net import PredictionNetwork
import torch
net = PredictionNetwork()
has_bn = any(isinstance(m, torch.nn.BatchNorm1d) for m in net.modules())
print(f"BatchNorm found: {has_bn}")  # must be False
```

**Fix:** Replace all `nn.BatchNorm1d` with `nn.LayerNorm`. Retrain.
`LayerNorm` normalizes per sample, not per batch — behavior is identical
in train and eval modes.

---

### I-4: CATASTROPHIC FORGETTING (fine-tuning degrades past knowledge)

**Probability: Medium. Consequence: High — world model gets progressively worse.**

**How it manifests:**
First exploration agent completes → error in target region decreases (good).
Second exploration agent completes → error in first region increases.
Third agent completes → second region improves but first two degrade.
Net effect: global_mean_error stays constant or increases despite agents running.

**Why it's invisible:** Each individual fine-tuning run succeeds. Each agent
improves its specific region. But the optimizer finds new weights for the new
region by *moving away* from the weights that encoded the previous regions.
This is textbook catastrophic forgetting from the continual learning literature.

**Root cause:** Fine-tuning exclusively on new exploration samples, no replay
of previous experience. The 80/20 old/new replay ratio is missing.

**The physics:** Neural network weights encode a local minimum of the loss
landscape. Fine-tuning on new data shifts the loss landscape. Without old
samples anchoring the previous minimum, gradient descent moves away from it.
The 20% new / 80% old ratio is a form of experience replay from DQN — the old
samples serve as "anchors" that constrain the optimizer to stay near the
previous minimum while adapting to the new data.

**Diagnostic test:**
```python
# After each agent completes, log regional errors:
errors = world_model.belief.get_regional_errors()
print(f"After agent {n}: {errors}")
# Plot across agents — if any region's error INCREASES after an agent run:
# catastrophic forgetting is occurring
```

**Fix:** In `_fine_tune_sync()`, ensure `n_old = len(new_samples) * 4` samples
are drawn from the replay buffer. Verify `self._replay_buffer` has samples in
it (at least from initial training).

---

### I-5: WORLDMODEL MULTIPLE INSTANCES (fine-tuning updates the wrong model)

**Probability: Low-Medium. Consequence: High — improvements vanish.**

**How it manifests:**
Fine-tuning runs and logs show the version incrementing. But predictions from
the MPC agent don't improve. Heatmap error values don't change. It looks like
fine-tuning isn't working, but actually it's updating a model nobody reads.

**Why it's invisible:** Two `WorldModel` instances. `OrchestrationAgent.fine_tune()`
updates instance A. `MPCAgent.predict()` reads from instance B (the one at the
`world_model` variable in api.py, which was never updated). No exception. Both
instances have valid weights. They just diverge silently.

**Diagnostic test (5 seconds):**
```bash
grep -r "WorldModel(" backend/ | grep -v "main.py" | grep -v "test_" | grep -v "#"
# Must return empty. Any hit = violation.
```

**Fix:** Every `WorldModel` consumer receives the instance as a parameter from
`main.py`. No module-level `WorldModel()` instantiation anywhere except `main.py`.

---

### I-6: BLEEDING PROPAGATION ORDER DEPENDENCY (comparison demo shows different anatomy)

**Probability: Medium. Consequence: Medium — comparison demo broken.**

**How it manifests:**
Both simulations initialized with seed=42. Initial tissue vascularity is
identical (as expected). After 20 steps with the same actions, `bleeding_mask`
differs between primary and comparison simulation.

**Why it's invisible:** The code looks correct. The propagation loop iterates
over all cells. But it's reading from and writing to the same array — so
a cell at (r, c) that propagates bleeding to (r, c+1) in iteration step N
changes the source value for (r, c+1) → (r, c+2) in step N+1 of the SAME
loop iteration. The final bleeding state depends on whether iteration is
row-major or column-major, which is an implementation detail.

**The physics:** This is identical to the core constraint in Conway's Game of
Life — all cell updates in a generation must use only the previous generation's
state. Reading and writing to the same buffer violates this by creating implicit
time ordering within a single step. Two implementations that both look "correct"
(both iterate all cells) can produce different results if one uses a copy and
one doesn't.

**Diagnostic test:**
```python
env_a = SurROLTissueEnv(SimConfig(seed=42)); env_a.reset()
env_b = SurROLTissueEnv(SimConfig(seed=42)); env_b.reset()
for _ in range(20):
    a = np.zeros(12, dtype=np.float32); a[7]=1.0; a[0]=0.1; a[11]=0.7
    env_a.step(action_to_surrol(a, 1))
    env_b.step(action_to_surrol(a, 1))
match = np.array_equal(env_a.bleeding, env_b.bleeding)
print(f"Bleeding identical: {match}")  # must be True
```

**Fix:** In `_propagate_bleeding()`, the first line must be
`next_bleeding = self.bleeding.copy()` (a new array, not a view).
Read exclusively from `self.bleeding`. Write exclusively to `next_bleeding`.
Replace after the loop: `self.bleeding = next_bleeding`.

---

## SECTION 2: ENDLESS LOOP TRAPS

These are specific debugging patterns that waste hours. Each one is
a sequence where every individual check looks fine but the bug is
elsewhere. Recognizing the pattern saves hours.

---

### D-1: THE CANVAS BLANK TRAP

**Pattern:** Canvas blank → check CSS → CSS present → check WS → WS events
arriving → check buildCanvasState → returns nodes → check React Flow render
→ looks correct → restart → still blank → spend 2 hours checking everything
except the one thing that's wrong.

**The trap:** The CSS file IS imported. But it's imported in the wrong component.
`@xyflow/react/dist/style.css` must be imported in the component that renders
`<ReactFlow>`, or in a parent that is always mounted when `<ReactFlow>` renders.
If the CSS import is in a component that conditionally unmounts (e.g., only in
the detail view but `<ReactFlow>` is in the canvas view), the CSS is absent
when the canvas renders.

**Exit the trap:**
```bash
# Check 1: CSS imported at all
grep -rn "xyflow/react/dist/style.css" frontend/src/
# Check 2: where exactly
grep -n "xyflow/react/dist/style.css" frontend/src/App.tsx
# If not in App.tsx: move it there. App.tsx is always mounted.
```

---

### D-2: THE NAN LOSS TRAP

**Pattern:** Training shows NaN loss → reduce learning rate → still NaN →
change optimizer → still NaN → increase batch size → still NaN → spend
1 hour tuning hyperparameters.

**The trap:** NaN from epoch 1, batch 1 is never a hyperparameter problem.
Hyperparameter issues appear as loss that starts finite and then diverges
(usually after 5-20 batches). Immediate NaN is a data problem or a loss
function numerical problem. No amount of LR tuning fixes bad data.

**Exit the trap:**
```python
# Check data before touching any hyperparameter
import sqlite3, numpy as np
conn = sqlite3.connect('backend/data/training.db')
rows = conn.execute("SELECT state FROM samples LIMIT 100").fetchall()
for row in rows:
    v = np.frombuffer(row[0], dtype=np.float32)
    if np.any(np.isnan(v)) or np.any(np.isinf(v)):
        print(f"BAD SAMPLE: nan={np.sum(np.isnan(v))} inf={np.sum(np.isinf(v))}")
        break
else:
    print("Data clean — check loss function numerics")
    # Add epsilon clamp: loss = F.binary_cross_entropy(pred.clip(1e-7, 1-1e-7), target)
```

---

### D-3: THE MPS SILENT FALLBACK TRAP

**Pattern:** Added `.to(device)` to model → training still slow → check device →
`model.device` shows MPS → check training speed → still slow → spend 30 min
investigating MPS compatibility.

**The trap:** Moving the model to MPS is necessary but not sufficient. The INPUT
TENSORS must also be moved to MPS. If you call `.to(device)` on the model but
not on `x` and `y_target`, PyTorch silently moves the data back to CPU for
the forward pass. You see the model on MPS but computation on CPU.

**Symptom difference:** MPS training: GPU utilization visible in Activity Monitor.
CPU training: all CPU cores at 100%, GPU at 0%.

**Exit the trap:**
```python
# Every forward pass must have:
x = torch.from_numpy(data).float().to(self.device)  # ← tensor on device
y = torch.from_numpy(target).float().to(self.device)  # ← target on device
pred = self.network(x)                                # ← model on device
loss = F.mse_loss(pred, y)                            # ← loss on device
```

---

### D-4: THE SURROL INSTALL LOOP TRAP

**Pattern:** `pip install surrol` fails → fix dependency → still fails → different
error → fix that → yet another error → 2 hours later: wrong approach entirely.

**The trap:** SurRoL is not on PyPI. It must be installed from source.
`pip install surrol` will never work.

**Exit the trap immediately (if you see any SurRoL import error):**
```bash
# Check this first — takes 5 seconds
pip show surrol 2>&1 | head -3
# If "Package not found": run the source install, not pip
cd soma && git clone https://github.com/med-air/SurRoL.git
cd SurRoL && uv pip install -e . && cd ..

# If installed but env.reset() crashes: PyBullet version mismatch
python -c "import pybullet; print(pybullet.__version__)"  # must be 3.2.x
uv add pybullet==3.2.6 --force-reinstall
```

**45-minute rule:** If SurRoL is not successfully imported within 45 minutes
of starting Task 1.2, switch to custom PyBullet simulation (see Scope Cut 3
in BUILD_PLAN.md). Do not spend more time on SurRoL installation.

---

### D-5: THE AGENT IMPROVEMENT TRAP

**Pattern:** Agents spawn → agents complete → check fine_tune → fine_tune runs →
check error measurement → measurement works → check LangGraph state → state
updates → check orchestrator → orchestrator reads errors → everything looks fine
but error doesn't decrease → 2 hours of confusion.

**The trap:** The orchestrator reads errors BEFORE fine-tuning updates the
belief state. The timing chain is: agent completes → `write_event(agent_completed)`
→ `await world_model.fine_tune(samples)` → belief state updated → orchestrator
reads updated errors on NEXT 2-second cycle. But if the orchestrator reads
immediately after `write_event` and before `fine_tune()` returns, it sees the
old error values. The improvement is real but the orchestrator doesn't see it
until the cycle AFTER fine-tuning.

**This is not a bug** — it's the async timing of the 2-second orchestrator cycle.
The improvement IS happening. Just wait for 2-4 orchestrator cycles after agent
completion to see the updated error values.

**To verify improvement IS happening:**
```bash
sqlite3 backend/data/events.db "
SELECT agent_id, error_before, error_after,
       (error_before - error_after) as delta
FROM events WHERE event_type='agent_completed'
ORDER BY id DESC LIMIT 10;"
# If delta > 0: improvement is happening, orchestrator will see it next cycle
# If delta ≈ 0: real improvement failure — check I-2 (wrong seed) or I-1 (circular)
```

---

### D-6: THE LANGGRAPH RECURSION TRAP

**Pattern:** LangGraph runs for 30+ cycles → `GraphRecursionError` →
search for infinite loop in code → check conditions → conditions look correct →
add print statements → still recursing → spend 1 hour looking for the loop.

**The trap:** `GraphRecursionError` doesn't mean there's an infinite loop.
LangGraph has a default recursion limit of 25 (number of node invocations
per `invoke()` call). A graph with 5 nodes and 5 agents spawning simultaneously
can hit 25 invocations legitimately. The error is not about code logic,
it's about LangGraph's execution budget.

**Exit the trap:**
```python
# Increase the limit (check LangGraph docs for current syntax)
from langgraph.graph import StateGraph
graph = builder.compile(
    checkpointer=checkpointer,
    recursion_limit=100  # or set in config
)
# OR: set per-invocation
result = graph.invoke(state, config={"recursion_limit": 100})
```

**The secondary trap:** After increasing the limit, the REAL infinite loop
(if one exists) will now run longer before failing. Use the limit increase
as a diagnostic: if it eventually stops after N cycles, it's legitimately
working. If it runs until a timeout, there's an actual loop in
`collect_status_node` not draining `completed_agents` from `active_agents`.

---

### D-7: THE WEBSOCKET REPLAY RACE TRAP

**Pattern:** Canvas works perfectly on the development machine → demoed on a
different machine (or incognito browser) → canvas appears empty despite
backend running for hours → WebSocket connected (events are arriving live) →
but no historical nodes → can't reproduce on dev machine.

**The trap:** The WebSocket replay (sending history on connect) is
implemented correctly, but there's a race condition: the backend adds the
client to the broadcast set BEFORE sending the historical replay, so live
events interleave with historical ones. The frontend receives them out of
order and `buildCanvasState` produces inconsistent node positions.

**The correct order:**
```python
# WRONG (race condition):
active_connections.append(ws)  # start receiving live events
for event in get_events_since(0):
    await ws.send_json({"type":"event","data":event})  # then replay

# CORRECT (no race):
for event in get_events_since(0):
    await ws.send_json({"type":"event","data":event})  # replay first
active_connections.append(ws)  # only then start receiving live
```

---

### D-8: THE TRAINING DATA DISTRIBUTION TRAP

**Pattern:** Training completes with val_mse = 0.03 (better than target) →
system runs → world model predictions look plausible → MPC agent runs →
but agent collides with vessels repeatedly despite "good" predictions.

**The trap:** The model trained to 0.03 MSE but on a skewed dataset.
If 90% of training episodes end within 15 steps (robot immediately fails
near vessels), 90% of training data is early-episode states. The model
is excellent at predicting early-episode tissue (which barely changes) and
useless at predicting mid-episode tissue (where vessels are actually relevant).
Val MSE is low because early-episode predictions are easy.

**The physics:** This is the "easy data dominance" problem in supervised learning.
When a distribution is imbalanced, a model that perfectly memorizes the majority
class will have low aggregate loss even if it has zero accuracy on the minority.
The minority (mid-episode states near vessels) is exactly what matters for demo.

**Diagnostic test:**
```python
# Check episode length distribution in training data
import sqlite3, numpy as np
conn = sqlite3.connect('backend/data/training.db')
steps = conn.execute(
    "SELECT episode_step FROM samples ORDER BY RANDOM() LIMIT 1000"
).fetchall()
steps = [s[0] for s in steps]
print(f"Mean step: {np.mean(steps):.1f}, Max step: {max(steps)}")
# If mean step < 20: data dominated by early-episode states
# If max step < 50: robot never survives past ~50 steps — add collision avoidance
```

**Fix:** Add a heuristic collision-avoidance wrapper to the data collection
policy (move toward target while avoiding grid cells near known vessels).
This extends episode length and diversifies training data.

---

## SECTION 3: TIME-CRITICAL CASCADES

Miss any of these and the cascade makes it worse each hour.

| Risk | Trigger | Cascade |
|---|---|---|
| **Data collection not running by hour 3** | Any delay in Task 1.2 | Training can't start until hour 5+, model not ready until hour 8+, Sprint 1 checkpoint at hour 14+ |
| **Training NaN discovered at hour 6** | Bad data or loss numerics | Costs 1hr to fix + re-collect (30min) + retrain (60min) = total +2.5hrs |
| **Sprint 1 checkpoint fails at hour 12** | Any unresolved backend bug | Sprint 2 starts on broken foundation; frontend bugs masked by backend bugs; double debugging |
| **Error reduction animation broken at hour 28** | Any frontend issue in Task 2.4 | Demo's primary moment missing; animation is the pitch's money shot; no time to fix in Sprint 3 |
| **Demo video not recorded by hour 30** | Any delay in polish | If demo breaks live at judging, no fallback exists |

**Protocol for cascade prevention:**
When ANY task runs 30+ minutes over estimate:
1. Stop. Check the dead reckoning table.
2. If you're behind on data collection: STOP and start data collection NOW.
3. If you're behind on everything else: apply the scope cut for this sprint.
4. Do not debug "a little longer." The cascade compounds.

---

## SECTION 4: SURROL + PYBULLET RISKS

---

### S-1: SURROL INSTALLATION FAILURE

**Probability: Medium-High. Time budget: 45 minutes maximum.**

**Symptoms (in likely order):**
- `ModuleNotFoundError: No module named 'surrol'` → not installed from source
- `ImportError: cannot import name TaskBase` → wrong installation method
- `env.reset()` raises physics server error → PyBullet version mismatch
- `env.reset()` raises AttributeError on PSM → SurRoL version incompatibility

**Diagnosis chain:**
```bash
# Check 1 (10s): is it installed at all?
python -c "import surrol; print(surrol.__file__)"

# Check 2 (10s): PyBullet version
python -c "import pybullet; print(pybullet.__version__)"  # need 3.2.x

# Check 3 (10s): can we instantiate?
python -c "from surrol.tasks.task_base import TaskBase; print('TaskBase OK')"
```

**Fix sequence:**
```bash
git clone https://github.com/med-air/SurRoL.git
cd SurRoL && uv pip install -e . && cd ..
uv add pybullet==3.2.6 --force-reinstall
```

**45-minute rule → custom PyBullet fallback:**
If SurRoL is not running within 45 minutes, implement a custom PyBullet
environment with a simple robot arm (no PSM URDF, just a sphere end-effector).
The world model, agents, and frontend are entirely unaffected.
What you lose: da Vinci PSM visual credibility. What you keep: everything else.

---

### S-2: PYBULLET CONCURRENT INITIALIZATION

**Probability: High (without lock). Consequence: Intermittent crashes.**

**Symptom:** Second `SurROLTissueEnv` creation crashes with `Invalid world ID`,
`Segfault`, or physics body confusion between instances. Crashes are intermittent
and don't reproduce reliably. Environment objects appear created but behave
strangely.

**Root cause:** PyBullet maintains a global physics server registry. Two
`TaskBase.__init__()` calls running concurrently in asyncio tasks corrupt
the internal server ID assignment.

**Fix:** `_pybullet_init_lock = asyncio.Lock()` at module level in `simulation.py`.
All environment construction goes through `await create_env(config)` which
acquires this lock before calling the constructor.

**The deeper principle:** Any library that uses global process state (shared
memory, file handles, GPU context) must be protected against concurrent
initialization. PyBullet, OpenGL, SQLite in WAL mode, and multiprocessing
pools all fall into this category. The pattern is always the same: module-level
lock, factory function, lock acquisition before construction.

---

### S-3: PSM JOINT LIMITS PREVENTING GRID COVERAGE

**Probability: Medium. Consequence: Training data has coverage gaps.**

**Symptom:** After training, prediction errors in certain grid regions
(typically the corners) are consistently higher than others, even after multiple
exploration agents target those regions.

**Root cause:** The PSM has joint angle limits that prevent the end-effector
from reaching grid cells near the corners of the workspace. `action_to_surrol()`
translates our action vectors to PSM joint commands, but some corner positions
are kinematically unreachable with the default URDF limits.

**Physics:** This is the workspace/joint-space mapping problem in robot
kinematics. The Cartesian workspace of a robot with joint limits is a
non-convex subset of the full Cartesian space. Cells outside the reachable
workspace are kinematically infeasible — the robot can never contact them.

**Diagnostic test:**
```python
# Command end-effector to all 4 corners, measure achieved position
for (col, row) in [(0,0), (0,15), (15,0), (15,15)]:
    wx = (col/15.0)*0.2 - 0.1
    wy = (row/15.0)*0.2 - 0.1
    # Send move command targeting (wx, wy)
    state_vec = env.get_state_vector()
    achieved_pos = state_vec[768:771]
    target_pos = np.array([(col/15.0), (row/15.0)])
    print(f"Target ({col},{row}): achieved EE at {achieved_pos[:2]}")
```

**Fix (if corner cells unreachable):**
Reduce the workspace bounds in `SimConfig` so that all `n_vessels` and the
surgical target are placed only within the reachable workspace. Adjust the
tissue grid coordinate-to-workspace mapping to cover only the reachable area.
This reduces the grid to ~14×14 effective cells but ensures full coverage.

---

### S-4: GETCAMERAIMAGE BLOCKING EVENT LOOP

**Probability: Medium. Consequence: MJPEG stream freezes, WS disconnects.**

**Symptom:** MJPEG stream works initially but freezes periodically. Browser
shows stale frame. WS clients disconnect during the freeze window.

**Root cause:** `p.getCameraImage()` is a synchronous PyBullet call that
can take 20-100ms depending on scene complexity. Called in an `async def`
without `asyncio.to_thread()`, it blocks the event loop.

**Fix:**
```python
# WRONG:
async def pybullet_frame_generator(env):
    while True:
        rgba = p.getCameraImage(...)  # blocks

# CORRECT:
async def pybullet_frame_generator(env):
    while True:
        rgba = await asyncio.to_thread(
            p.getCameraImage, 320, 240, viewMatrix, projMatrix,
            physicsClientId=env.physics_client_id
        )
```

---

### S-5: SURROL RESET PARTIAL STATE ISSUE

**Probability: Low-Medium. Consequence: Seed reproducibility broken.**

**Symptom:** First episode after `env.reset()` has correct tissue layout.
Second episode after `env.reset()` has different tissue (vascularity changed).

**Root cause:** SurRoL's `TaskBase.reset()` may not call our custom
`_build_world()` fully — it depends on which methods we override and whether
`super().reset()` re-invokes the world-building hooks.

**Diagnostic test:**
```python
env = SurROLTissueEnv(SimConfig(seed=42))
env.reset()
v1 = env.vascularity.copy()
env.reset()
v2 = env.vascularity.copy()
print(f"Vascularity stable across resets: {np.array_equal(v1, v2)}")
```

**Fix:** Override `reset()` explicitly and call `self._build_world()` with
the original `SimConfig` seed after `super().reset()`. Reinitialize `self._rng`
from the seed at the start of each reset.

---

## SECTION 5: LANGGRAPH RISKS

---

### L-1: GRAPH COMPILATION FAILURE

**Symptom:** `graph.compile()` raises `ValueError` about undefined nodes
or unreachable nodes.

**Diagnostic:** LangGraph requires every node referenced in an edge to be
added via `add_node()` before `add_edge()`. Check that:
1. `add_node()` called for every string that appears in `add_edge()`
2. Every conditional edge returns only strings that are valid node names
3. There's a path from `START` to `END` (or the graph won't compile if required)

**The Send pattern gotcha:** `Send("node_name", state_subset)` requires
`"node_name"` to be a registered node. If you return `Send("spawn_explore", ...)`
from a conditional edge function but the node is named `"spawn_exploration"`:
silent wrong behavior (the Send is ignored, no error).

---

### L-2: STATE ANNOTATION MISSING FOR PARALLEL WRITES

**Symptom:** Running multiple exploration agents in parallel → only one agent's
result appears in `completed_agents` after the fan-out. Other results are lost.

**Root cause:** List fields in `TypedDict` state that are written by parallel
branches need `Annotated[list, operator.add]` to tell LangGraph to merge
the lists rather than overwrite.

```python
# WRONG: last writer wins
class State(TypedDict):
    completed_agents: list[str]

# CORRECT: parallel writes merged
class State(TypedDict):
    completed_agents: Annotated[list[str], operator.add]
```

---

### L-3: MEMORYSERVER NOT PERSISTING ACROSS RESTARTS

**Symptom:** After restarting the backend, the orchestrator starts from
cycle 0. All in-progress agents are lost. Agents that were running are
never marked complete.

**Root cause:** `MemorySaver` stores state in-process memory. Process restart
= state lost. This is documented behavior.

**Why this is acceptable for SOMA:** The exploration agents write their
`agent_spawned` and `agent_completed` events to the SQLite event log before
the orchestrator's graph state is updated. The SQLite log persists. On restart:
- Historical events replay correctly to the frontend
- The world model weights are saved to disk (`models/prediction_network.pt`)
- Only the orchestrator's in-flight state is lost
- Agents that were in-flight are abandoned (their sim instances are closed)

**Protocol on restart:** After restart, wait 1-2 orchestrator cycles. The
orchestrator reads current belief state (from the persisted world model) and
re-spawns agents for still-high-error regions. The system self-heals.

---

## SECTION 6: CLAUDE API RISKS

---

### A-1: TOOL USE FORMAT FAILURE

**Probability: Low with `tool_choice={"type":"any"}`. High without it.**

**Symptom:** `StopIteration` in `_call_claude_with_retry()` when extracting
tool block. OR Claude returns a text response explaining the action instead
of calling the tool.

**Root cause:** Without `tool_choice={"type":"any"}`, Claude decides whether
to use a tool or respond in prose. It often chooses prose when the prompt is
phrased conversationally. With `tool_choice={"type":"any"}`, Claude MUST
use one of the provided tools on every response.

**Fix:**
```python
response = client.messages.create(
    model="claude-sonnet-4-6",
    tools=TOOLS,
    tool_choice={"type": "any"},  # REQUIRED — forces tool use
    messages=[...]
)
```

---

### A-2: STREAMING BUFFER INCOMPLETE

**Symptom:** `json.JSONDecodeError` when parsing tool input from stream.
Tool input appears cut off or malformed.

**Root cause:** Calling `json.loads()` on streaming text before the stream
completes. The text arrives in chunks; parsing a chunk produces invalid JSON.

**Fix:** Always use `stream.get_final_message()` before accessing tool content.
Never parse `stream.text_stream` tokens as JSON — that stream is for display only.

```python
async with client.messages.stream(...) as stream:
    async for text in stream.text_stream:
        callback(text)  # display tokens as they arrive
    final = await stream.get_final_message()  # wait for complete response
    tool_block = next(b for b in final.content if b.type == "tool_use")
    tool_input = tool_block.input  # already parsed dict — no json.loads needed
```

---

### A-3: RATE LIMITING UNDER CONCURRENT AGENTS

**Symptom:** `anthropic.RateLimitError` during periods when multiple exploration
agents are active simultaneously.

**Diagnosis:** With 4 concurrent agents, each calling Claude every 5 steps at
~10 steps/second, peak rate = 4 × (10/5) = 8 requests/second. Burst peaks
during agent initialization can spike to 20+/second.

**Fix:** Module-level semaphore already in spec:
```python
_claude_api_semaphore = asyncio.Semaphore(10)
```

If rate limit errors still occur: reduce semaphore to 5 AND ensure retry
uses exponential backoff (1s, 2s, 4s). The `max_retries=3` in
`_call_claude_with_retry()` handles this already.

**Fallback if Claude API is completely unavailable:**
Replace `get_next_action()` with `_random_focused_action()` (random actions
biased toward the target region). The agent episode continues, data is
collected, fine-tuning runs. The agent feed shows no reasoning, but the
self-improvement loop continues. Canvas tree still grows.

---

## SECTION 7: FRONTEND RISKS

---

### F-1: CANVAS BLANK (CSS IMPORT)

**Probability: High — this is the most common React Flow setup failure.**

Covered in QUICK_REFERENCE.md GOTCHA TABLE. Short version:
```bash
grep -n "xyflow/react/dist/style.css" frontend/src/App.tsx
# Must return exactly 1 result
# If 0: add the import to App.tsx (not to Canvas/index.tsx)
```

---

### F-2: CANVAS RE-LAYOUT ON EVERY EVENT

**Probability: Medium.**

`NODE_TYPES` defined inside the component function → new object on every
render → React Flow detects "new" types → resets all node positions.

```typescript
// WRONG: inside component
function CanvasView() {
  const NODE_TYPES = { root: RootNode, ... }  // new object every render!
  return <ReactFlow nodeTypes={NODE_TYPES} .../>
}

// CORRECT: outside component at module scope
const NODE_TYPES = { root: RootNode, ... }  // created once
function CanvasView() {
  return <ReactFlow nodeTypes={NODE_TYPES} .../>
}
```

---

### F-3: ERROR REDUCTION ANIMATION NOT FIRING

**This is the highest-priority frontend fix. If this breaks, the demo's
primary moment doesn't happen.**

**Cause 1: `error_before` or `error_after` is null in event**
```bash
sqlite3 backend/data/events.db "
SELECT agent_id, error_before, error_after
FROM events WHERE event_type='agent_completed' LIMIT 5;"
# If NULL: fix write_event() call in agents/exploration.py
# error_before and error_after must be passed as top-level kwargs
```

**Cause 2: Animation trigger condition wrong**
```typescript
// WRONG: event.payload.error_before (it's a top-level field!)
// CORRECT: event.error_before
const shouldFlash = event.error_before !== null
  && event.error_after !== null
  && (event.error_before - event.error_after) > 0.05  // threshold
```

**Cause 3: useEffect dependency array wrong**
```typescript
// WRONG: missing events dependency
useEffect(() => {
  // ... check for qualifying events
}, [])  // never re-fires

// CORRECT
useEffect(() => {
  // ... check for qualifying events
}, [events])  // fires whenever events array changes
```

**Test this 5× manually before Sprint 3. If it fails any of the 5 times, fix.**

---

### F-4: PLOTLY HEATMAP BROWN AT MIDPOINT

**Probability: Medium. Consequence: Looks broken to any judge who sees it.**

**Symptom:** `errorToColor(0.5)` should produce yellow. If it produces brown
or olive, RGB interpolation is being used instead of HSL.

```typescript
// WRONG: Plotly colorscale with hex codes
colorscale: [[0, '#2db82d'], [0.5, '#b8972d'], [1, '#b82d2d']]

// CORRECT: Plotly colorscale with HSL
colorscale: [
  [0,   'hsl(120, 85%, 45%)'],  // green
  [0.5, 'hsl(60, 85%, 45%)'],   // yellow (not brown!)
  [1,   'hsl(0, 85%, 45%)'],    // red
]
```

**Verify in browser:**
```javascript
// Open browser console, check:
document.querySelector('.js-plotly-plot').__data[0].colorscale
// Must show hsl() strings, not hex codes
```

---

### F-5: MJPEG STREAM FREEZING

**Probability: Low-Medium in Chrome.**

**Symptoms:**
- Stream loads initially but freezes after ~30 seconds
- Stream works in Firefox but not Chrome
- Stream shows correct first frame then goes black

**Causes in priority order:**
1. `getCameraImage()` blocking event loop — fix: wrap in `asyncio.to_thread()`
2. Missing `Cache-Control: no-cache` header — Chrome may cache the stream URL
3. Missing `await asyncio.sleep(1/15)` in generator — generator runs tight loop
   blocking the event loop between frames

```python
# Required headers in StreamingResponse:
return StreamingResponse(
    generator,
    media_type="multipart/x-mixed-replace; boundary=frame",
    headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
)
```

**Fallback:** If MJPEG freezes and can't be fixed in 30 minutes, replace with
PNG snapshot polling:
```typescript
// Replace <img src="/sim/.../stream"> with:
const [frame, setFrame] = useState('')
useEffect(() => {
  const id = setInterval(async () => {
    const r = await fetch(`/sim/${simId}/frame`)  // returns base64 PNG
    setFrame(await r.text())
  }, 333)  // 3fps is enough for demo
  return () => clearInterval(id)
}, [simId])
<img src={`data:image/png;base64,${frame}`} />
```

---

## SECTION 8: DEMO RISKS

---

### P-1: SYSTEM CONVERGED BEFORE DEMO (no agents spawning)

**Probability: Medium if system runs for >4 hours before judging.**

**Symptom:** World model has low error everywhere. Orchestrator's
`route_orchestrator()` returns `"wait"` every cycle. No agents spawn.
Canvas is static. The "self-organizing" story has no visible evidence.

**Root cause:** The system worked. The world model successfully learned the
surgical simulation. No bug — just a demo timing problem.

**Protocol:**
```bash
# 30 minutes before judges arrive:
curl http://localhost:8000/debug/reset_belief
# Sets all regional errors to 0.5
# Orchestrator will spawn agents within 4 seconds
```

**Alternatively:** Restart the backend with a modified `SimConfig` (different
`difficulty` level). The world model hasn't seen this configuration, so errors
will be high everywhere.

---

### P-2: PYBULLET WINDOW CRASHES

**Probability: Low. Consequence: MJPEG stream dies.**

**Symptom:** PyBullet window closes unexpectedly. MJPEG stream returns errors.
Backend may not crash (FastAPI continues) but simulation panel goes blank.

**Root cause:** Memory exhaustion from too many sim instances. PyBullet's
OpenGL renderer consumes ~200MB per physics server instance. With 6 concurrent
instances: ~1.2GB RAM just for PyBullet.

**Prevention:** Enforce max 6 concurrent instances (primary + comparison + 4
exploration agents max). `CapabilityAgent` must close its temp env before
returning. `ExplorationAgent` must close its env in the `finally:` block.

**Recovery during demo:**
1. Backend will restart the primary sim on next step command
2. MJPEG stream will reconnect automatically
3. Say: "The simulation is resetting — watch what happens when it restarts"
4. This can look intentional if you stay calm

---

### P-3: TRAINING STILL RUNNING AT DEMO TIME

**Probability: Low if data collection started by hour 3.**

**Symptom:** `/health` returns `status: "training"`. Training overlay covers
the frontend. System is not in demo-ready state.

**Protocol if training is running when judges arrive:**
1. Do NOT wait for training to complete
2. Kill training: Ctrl+C on the training terminal
3. Load whatever checkpoint exists:
   ```python
   # In train.py, save checkpoint every 10 epochs
   if epoch % 10 == 0:
       torch.save(checkpoint, f'models/checkpoint_e{epoch}.pt')
   ```
4. Copy best checkpoint to `prediction_network.pt`
5. Restart backend — world model loads with partial training
6. System runs with higher initial errors (more agent activity = better demo)

---

## SECTION 9: SENIOR ENGINEERING PROTOCOL

This section contains debugging and decision-making frameworks.
Use these when standard approaches aren't working.

---

### 9.1: BINARY SEARCH DEBUGGING

When something is broken and you don't know where, don't check everything.
Halve the problem space at each step.

**SOMA application:**
- Agent not improving error? Check: (1) does the network forward pass work?
  If yes → problem is in fine-tuning. If no → problem is in prediction_net.
- Fine-tuning → (2) does the loss decrease during fine-tune? If yes →
  problem is in how we measure error after fine-tune. If no → check replay.
- Check replay → (3) is n_old > 0? If no → buffer empty. If yes →
  check that old samples are actually from the right distribution.

Binary search isolates the problem in O(log N) checks, not O(N).

**The rule:** Each check you run should halve the hypothesis space.
If a check doesn't rule something out, it's not a useful check.

---

### 9.2: EVIDENCE FIRST, HYPOTHESIS SECOND

The most common debugging mistake: forming a hypothesis before observing.
This leads to confirmation bias — you look for evidence that supports
your hypothesis and miss evidence that contradicts it.

**Protocol:**
1. OBSERVE: what exact value is wrong? Not "the heatmap looks bad" but
   "regional errors: {'upper_left': 0.73, ...} — no change between cycles"
2. CHEAPEST TEST: what's the fastest thing I can check to narrow it down?
   (Usually a sqlite query or print statement, not a code change)
3. FORM HYPOTHESIS ONLY AFTER: the test result constrains the possibilities
4. TEST THE HYPOTHESIS: run the specific test that would falsify it
5. FIX the confirmed cause

**Anti-pattern to avoid:** "I think it might be the replay buffer, let me
add some logging to fine_tune()..." without first checking whether
`error_after < error_before` in the event log (which would immediately
tell you if fine-tuning is helping at all).

---

### 9.3: PHYSICS AND ML KNOWLEDGE APPLICATION

These are the moments where academic knowledge shortcuts empirical debugging.
Apply them when the behavior seems physically impossible.

**QUATERNION RANGE:** Unit quaternions have components in [−1, 1] due to
the constraint w²+x²+y²+z²=1. If you apply Sigmoid to quaternion output,
you constrain it to (0,1) — a physically impossible range for a unit quaternion.
The network will produce outputs that look "bounded" but are actually wrong
representations. Symptom: MPC agent makes decisions that don't correspond
to actual end-effector positions. Fix: no activation on the instrument head.

**EMA TIME CONSTANT:** The EMA update `e = α×new + (1−α)×old` with α=0.3
has a time constant τ = 1/α ≈ 3 steps (time to reach 63% of a step change).
After ~10 steps, the EMA has fully responded to a sustained new input.
Application: if the orchestrator checks regional errors 2 seconds after
fine-tuning completes, and MPC runs at 20 Hz = 40 steps per 2 seconds,
the error map has had 40 × 0.3 ≈ fully converged to the new prediction
error values. But if fine-tuning is slow and only 5 steps have occurred
since the last fine-tune: the error map still reflects old values.

**GAUSSIAN OVERFLOW:** Three Gaussians with amplitude 0.8 can sum to > 1.0
near overlapping centers. If not clipped to [0,1], vascularity values > 1
cause bleeding to trigger on cells with vascularity 1.1 that shouldn't bleed
(threshold is > 0.6, so 1.1 > 0.6 is correct... but the bleeding propagation
also triggers on cells with vascularity 0.35 that border a vascularity-1.1 cell,
creating wider-than-expected bleeding spread). Symptom: bleeding spreads
much faster than expected even at low contact force. Fix: clip vascularity
to [0,1] immediately after Gaussian summation.

**CONTACT FORCE NORMALIZATION:** PyBullet returns contact forces in Newtons.
The PSM surgical robot in SurRoL has a maximum contact force of ~20N (real
da Vinci PSM specs). The CONTACT_SCALE_FACTOR must normalize this to [0,1].
If CONTACT_SCALE_FACTOR=50 and maximum real force is 20N, the maximum
normalized force is 20/50=0.4 — contact damage never exceeds 40% of maximum.
If you want fuller utilization, CONTACT_SCALE_FACTOR should be closer to 20.
Measure actual PSM contact forces during Task 1.2 and update the constant.

**CATASTROPHIC FORGETTING MECHANISM:** The network weights represent the
minimum of a loss landscape over all training data. Fine-tuning on new data
moves the optimizer to a new minimum that fits the new data but is far from
the old minimum. The replay buffer's 80/20 ratio works because 80% of the
loss gradient still comes from old data, so the optimizer cannot move far
from the previous minimum in a single fine-tune step. Without replay,
the optimizer is free to move arbitrarily far in one fine-tune step.

**MPC HORIZON CHOICE:** 1-step lookahead with 8 candidates is appropriate
for surgical robotics because: (1) surgical movements are slow (~1mm/s for
PSM), so the state doesn't change dramatically in one step; (2) tissue
dynamics are nearly linear for small deformations (Hookean regime); (3) the
action space has 12 dimensions but only 3-6 are active at any DOF level,
making 8 candidates reasonable coverage. A longer horizon would require
exponentially more network evaluations (8^N for N steps) or a value function
trained separately. For a hackathon: 1-step is the correct scope choice.

---

### 9.4: WHEN TO STOP DEBUGGING AND CUT SCOPE

The hardest decision in a hackathon is stopping a debugging session.
These are the conditions that make scope cutting mandatory:

| Condition | Action |
|---|---|
| Debug session has been running > 30 minutes with no root cause identified | Stop. Cut scope. The debug time already consumed more than the cut saves. |
| Root cause identified but fix requires > 45 minutes | Cut scope. The remaining build tasks don't get done. |
| A known fallback exists that produces a working demo | Take the fallback immediately. Polish later if time allows. |
| The bug would only be noticed by a technically sophisticated judge | Move on. Most bugs aren't noticed. Polished demo beats correct internals. |
| The sprint checkpoint is < 2 hours away | Stop fixing bugs. Verify the checkpoint items that WORK. |

**The decision test:** "If I spend 30 more minutes on this, and it doesn't work,
am I in a better or worse position than if I spend 30 minutes polishing
what already works?"

Almost always: polish what works. Debugging has uncertain payoff.
Polishing has certain payoff.

---

### 9.5: THE GOLDEN SIGNAL

For SOMA, there is one observable that tells you definitively whether
the core claim is working:

```bash
sqlite3 backend/data/events.db "
SELECT agent_id,
       printf('%.3f', error_before) as before,
       printf('%.3f', error_after)  as after,
       printf('%.3f', error_before - error_after) as delta
FROM events
WHERE event_type = 'agent_completed'
ORDER BY id DESC LIMIT 10;"
```

If `delta > 0` for most agents: the core claim is working. World model
improves. Self-improvement is real. Everything else is polish.

If `delta ≈ 0` for all agents: something in the self-improvement loop is
broken. Check in this order: I-1 (circular input), I-2 (wrong seed),
I-4 (no replay), I-5 (multiple WorldModel instances).

**Don't look at the heatmap colors to verify this.** The heatmap is an
animated representation with smoothing and EMA. The event log is ground truth.

---

## RISK PRIORITY MATRIX

**Fix before any demo attempt:**
- M-1 (SQLite WAL) — prevents backend from running stably
- M-2 (event loop blocking) — WS disconnects during demo
- F-1 (canvas blank) — nothing to show
- F-3 (WS replay missing) — canvas empty on late connect
- I-1 (circular input) — world model never learns

**Fix before Sprint 3 begins:**
- F-4 (error reduction animation) — demo's primary moment
- I-2 (wrong exploration seed) — agents don't improve anything
- I-4 (catastrophic forgetting) — progressive degradation
- S-2 (PyBullet concurrent init) — intermittent crashes

**Fix if time allows:**
- F-2 (canvas flickering) — distracting but not blocking
- F-4 (Plotly brown midpoint) — visually wrong but demo works
- I-6 (bleeding double-buffer) — comparison demo broken, but comparison is cuttable
- D-8 (training data distribution) — weaker model, more agent activity

**Managed with fallback (don't debug, just activate fallback):**
- S-1 (SurRoL install) → custom PyBullet in 45min
- A-3 (Claude API down) → random actions
- P-1 (system converged) → `/debug/reset_belief`
- F-5 (MJPEG freeze) → PNG snapshot polling
