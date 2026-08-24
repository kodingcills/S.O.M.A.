# SOMA — Self-Organizing Model Architecture

**A surgical-robot world model that knows what it doesn't know — and acts on that.**

SOMA trains a neural dynamics model `f_θ(s_t, a_t) → ŝ_{t+1}` of a simulated
da-Vinci-style surgical scene (SurRoL/PyBullet), tracks **per-cell prediction
error** on a 16×16 tissue grid, and autonomously dispatches LLM-driven
exploration agents into the regions where it is most wrong. Every agent
episode becomes a fine-tuning set; every improvement lights up the heatmap.

## The loop

```
simulate → predict → measure error → orchestrator decides → spawn agent
   ↑                                                          │
   └────── fine-tune world model ← collect episodes ──────────┘
```

- **World model** — 818→512→256→128 LayerNorm MLP with 4 heads (tissue,
  damage, vessel-risk, instrument). Trained on 50k simulation samples;
  continually fine-tuned (80/20 replay) by exploration agents.
- **Orchestrator** — LangGraph StateGraph; fans out agents via `Send`,
  enforces circuit breakers, and unlocks degrees-of-freedom (DOF 1→6) as
  global error crosses thresholds. The robot *earns* its kinematics.
- **Exploration agents** — Claude tool-use streaming (`tool_choice=any`)
  picks data-collection actions; reasoning streams live to the dashboard.
- **Comparison baseline** — a reactive no-world-model policy runs the same
  anatomy side-by-side; SOMA damages measurably fewer vessels.

## Demo numbers (this repo, this machine)

| Gate | Result |
|---|---|
| Exploration golden signal | error 0.300 → 0.057 (Δ +0.244) |
| MPC vs reactive vessel-damage rate | **0.016 vs 0.037** |
| Training | val_mse 0.0009 · damage_acc 100% |
| Test suite | 86 pytest · 20 vitest — all green |

## Run it

```bash
# backend (Python 3.11, uv)
uv sync && bash backend/tools/install_all.sh   # pybullet+SurRoL stack
uv run uvicorn backend.main:app --port 8000    # first boot auto-trains

# frontend
cd frontend && npm install && npm run dev      # http://localhost:5173
```

Environment: `.env` with `ANTHROPIC_API_KEY` enables live agent reasoning
(system degrades to random-focused actions without it), optional
`HF_TOKEN`/`KAGGLE_TOKEN`.

## Architecture

```
backend/
  event_log.py      append-only SQLite event store (WAL) — THE communication law
  simulation.py     SurROLTissueEnv: 806-float state vector, tissue physics
  belief_state.py   per-cell error map (EMA), regional errors, snapshots
  prediction_net.py the learned dynamics model
  world_model.py    singleton: locks, replay buffer, async fine-tuning
  orchestrator.py   LangGraph graph: route → Send fan-out → circuit breaker
  mpc_agent.py      8-candidate model-predictive control on the primary sim
  agents/           exploration (Claude) · capability (DOF unlock) · reactive
frontend/
  Canvas            infinite agent tree (@xyflow/react v12), pure reducer
  DetailView        live sim stream + Plotly error heatmap + agent feed
  CompareView       SOMA vs reactive, dual MJPEG streams
```

Nine named invariants (WAL-per-connection, circular-input exclusion, seed
reproducibility, event-loop non-blocking, …) are enforced by named tests in
`tests/`. Spec documents live in `docs/specs/`.

## Screenshots

`logs/screenshots/` — canvas tree, detail view (live PyBullet render +
heatmap + confidence bars), comparison view.

