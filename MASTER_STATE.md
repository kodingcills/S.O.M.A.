# SOMA — Master State
#
# Read this first in every OpenCode session.
# Update this last in every OpenCode session.
# This file is the build's working memory.
# Do not start a task if prerequisites below are not marked complete.

---

## BUILD STATUS

```
Updated:        2026-08-25 ~04:13 ET
Sprint:         3 (Sprints 1+2 checkpoints PASSED)
Hours elapsed:  ~8 / 36
Build health:   GREEN
```

---

## COMMITTED STACK

```
Simulation:     SurRoL + SurROLTissueEnv(TaskBase), custom tissue task
World model:    PredictionNetwork (818 → 512 → 256 → 128, 4 heads)
Orchestration:  LangGraph (StateGraph + MemorySaver)
Agent AI:       Claude API claude-sonnet-4-6, tool_choice="any", streaming
3D rendering:   PyBullet via SurRoL + MJPEG stream endpoint
Interface:      React graph-first scientific/research workspace
Observability:  Weights & Biases (optional)
Heatmap:        native 16×16 CSS grid, sequential uncertainty scale
Canvas:         @xyflow/react v12 (NOT reactflow)
State vector:   806 float32 values
Action vector:  12 float32 values
Network input:  818 float32 values (806 + 12)
Python:         3.11 (pinned in .python-version)
Package mgr:    uv (pyproject.toml + uv.lock)
```

---

## TRAINING STATUS

```
Samples collected:   50,000 / 50,000 (gate PASS bad_rate=0 mean_step=98.4)
Training status:     COMPLETE; golden signal PROVEN: exp_fded40 tool_tissue_boundary 0.3002->0.0567 (delta +0.244, 1/1 improved PASS)
val_tissue_mse:      0.000929 (< 0.05 target)
val_damage_acc:      100% (> 85% target)
Checkpoint:          backend/models/prediction_network.pt (physics probes 2/3)
```

---

## COMPLETED TASKS

Task 0 — bootstrap | uv py3.11 torch-MPS; pybullet-3.2.5 patched install; SurRoL main-branch vendor
Task 1.1 — event-log-and-api | event_log.py(WAL,C6,C7) api.py(main skeleton) | tests 39-suite GREEN
Task 1.2 — surrol-tissue-env | simulation.py SurROLTissueEnv(PsmEnv) 806-f32 create_env-lock | 12/12 sim tests GREEN
Task 1.3a — belief-state | belief_state.py EMA .3 circular-input-safe region_to_slice | GREEN
Task 1.4 — prediction-net+train.py | LayerNorm 4-head ~620K params; train.py auto-extend | 5/5 GREEN
Task 1.4b — TRAINED: val_mse=0.0009 dam_acc=100% epoch-1 early-stop; physics probes 2/3 PASS
Task 2.1 — frontend-scaffold | vite+ts xyflow-v12 types/colors/hooks/App shell | build+tsc clean
Task UI-R1 — research-interface-redesign | graph-first shell, inspector trace/evidence, runs/models/compare, simulation workspace | 22 frontend tests + build GREEN

---

## IN PROGRESS

None.

---

## BLOCKED

None.

## KNOWN ISSUES (Sprint 1 close)
1. RESOLVED — PYBULLET CID SHIM (task 2.x): vendored un-scoped p.* calls routed to client 0 regardless of env. simulation.py now patches pybullet module fns to inject ContextVar-bound physicsClientId; public env methods bind via @_cid_bound decorator. Repro confirmed: sibling close no longer severs other envs. Note: shim patches the pybullet module globally at import of backend.simulation.
2. ORCHESTRATOR RECURSION BURSTS: main loop's graph.ainvoke bursts end at recursion_limit=100 ('burst ended' logs) then re-invoke — functional but noisy; tune burst length or move sleep out of collect_status.

---

## KNOWN DEVIATIONS FROM SPEC

Task 0 (bootstrap) | File: backend/tools/install_pybullet.sh
Deviation:  pybullet 3.2.6/3.2.7/latest all fail to compile on macOS SDK 15 (bundled zlib zutil.h fdopen macro collides with SDK _stdio.h:322). Installed patched pybullet==3.2.5 via scripts.
Reason:     upstream sdist bug; guard `#ifndef fdopen` lacks `!defined(__APPLE__)`.
Impact:     none — API version identical (202010061); reproducible via backend/tools/install_pybullet.sh.

Task 0 (bootstrap) | File: vendor/SurRoL
Deviation:  SurRoL installed from med-air/SurRoL **main** branch (default branch SR-VPPV was restructured upstream, no package). Spec's `surrol.tasks.task_base.TaskBase` does not exist in reality — actual base classes: `surrol.gym.surrol_env.SurRoLEnv(gym.Env)` → `SurRoLGoalEnv` → `surrol.tasks.psm_env.PsmEnv`.
Reason:     spec was written against idealized API; upstream repo reorganized.
Impact:     Task 1.2 subclasses PsmEnv instead of TaskBase; hook mapping: `_build_world`→`_env_setup`+`_sample_goal`, `_get_obs`→`_get_obs`, `_check_success`→`_is_success`. All 806-vector/reward/seed invariants unchanged. gym pinned 0.25.2 (old step API).

Task UI-R1 | Files: frontend/src/**, docs/specs/INTERFACE.md
Deviation:  Legacy dark dashboard visuals and simulation-first detail view were replaced by a light graph-first research interface. Plotly is no longer used by the model panel at runtime.
Reason:     Product redesign makes learning lineage, event evidence, and before→after model changes the primary mental model.
Impact:     Backend event/health/detail/MJPEG contracts are unchanged. Recorded playback, trajectory files, and seekable trace-to-frame correlation remain unavailable because the backend exposes only a live MJPEG stream and event timestamps.

Format for new entries:
```
Task <id> | File: <filename>
Deviation: <what changed>
Reason:    <why>
Impact:    <downstream effect, or "none">
```

---

## INTEGRATION CONTRACTS

Check off each contract after both sides are verified working end-to-end.
A contract is verified when the integration test in the task prompt passes,
not just when the component's unit tests pass.

```
[ ] C1: SurROLTissueEnv.get_state_vector() → shape (806,) float32, no NaN
[ ] C2: BeliefState.to_input_vector() → shape (806,) float32, no error_map
[ ] C3: WorldModel.fine_tune() awaitable, runs in to_thread(), holds _finetune_lock
[ ] C4: agent_spawned written BEFORE episode loop starts
[ ] C5: agent_completed written AFTER fine_tune() with non-null error_before/after
[ ] C6: WebSocket replays all history on connect before live broadcast
[ ] C7: event payload always a dict, never raw JSON string
[ ] C8: SurROLTissueEnv(SimConfig(seed=42)).reset() produces same tissue every call
```

---

## INTERFACE REGISTRY

Populated as tasks complete. Format:

```
<function_name>(<params>) → <return_type>
  Defined in: <file>
  Used by:    <downstream components>
  Status:     VERIFIED | BUILT | STUB
```

---

## NEXT TASK

```
Task:                Optional evidence-recording API
Agent:               Sisyphus + Oracle for event/frame correlation contract
Prerequisites:       UI-R1 complete; existing live MJPEG and event log remain functional
Spec sections:       INTERFACE.md current direction; simulation evidence workflow
Estimated time:      TBD
Key requirements:    recorded frames, trajectory artifacts, shared run IDs, seek timestamp mapping
```

---

## MEASURED PERFORMANCE LOG

Fill in as tasks complete. Used for timing decisions.

```
env.step() latency:                 [PENDING] ms
get_state_vector() latency:         [PENDING] ms
predict() latency:                  [PENDING] ms
fine_tune() duration (5 epochs):    [PENDING] sec
Data collection (50K samples):      [PENDING] min
Initial training (MPS, 50 epochs):  [PENDING] min
Initial training (CPU, 50 epochs):  [PENDING] min
WS replay time (500 events):        [PENDING] ms
Canvas render time (50 nodes):      [PENDING] ms
Claude API p50 latency:             [PENDING] ms
CONTACT_SCALE_FACTOR (measured):    [PENDING]
```

---

## SPEC ISSUES FOUND

Errors discovered in spec documents during implementation.
Format: `<file> §<section>: <description of error>`

None yet.
