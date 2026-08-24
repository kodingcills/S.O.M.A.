# Devpost Submission Draft — SOMA

## Title
SOMA — Self-Organizing Model Architecture for Surgical Robotics

## Tagline
A world model that knows what it doesn't know — and acts on that.

## Description (draft)

### Inspiration
Surgical robots execute policies but never *understand* their workspace.
We asked: what if a surgical robot could measure its own ignorance — per
square millimeter of tissue — and spend its effort learning exactly where
it's wrong?

### What it does
SOMA trains a world model of a simulated surgical scene and visualizes its
own prediction error as a living heatmap over 16×16 tissue cells. When a
region's error crosses a threshold, an autonomous exploration agent spawns,
streams its reasoning while collecting targeted experience, fine-tunes the
model, and — visibly — turns its region of the heatmap green. As the world
model masters the scene, SOMA unlocks new degrees of freedom for its robot
arm (DOF 1→6): capability is earned by understanding, not configured.

A second robot runs the same anatomy with NO world model (reactive greedy
baseline). Side-by-side, SOMA damages measurably fewer vessels
(0.016 vs 0.037 damage rate in our validation soak).

### How we built it
- **Simulation**: SurRoL da-Vinci PSM + PyBullet, custom tissue physics
  (vascularity Gaussians, contact damage, bleeding propagation), locked
  806-float state vector.
- **World model**: PyTorch MLP (818→512→256→128, LayerNorm, 4 heads),
  trained on 50k rollouts; continual fine-tuning with 80/20 experience
  replay against catastrophic forgetting.
- **Orchestration**: LangGraph StateGraph — Send-based agent fan-out,
  circuit breakers, DOF-unlock thresholds; append-only SQLite event log as
  the single communication backbone (replayed to the browser over
  WebSocket).
- **Agents**: Claude tool-use streaming with forced structured output;
  graceful degradation to random-focused collection when the API is down.
- **Frontend**: React + @xyflow/react v12 infinite canvas (pure event
  reducer), Plotly HSL heatmap with a 600 ms error-reduction flash, live
  MJPEG PyBullet renders, real-time agent reasoning feed.

### Challenges we ran into
- PyBullet routes un-scoped calls to client 0 — closing one simulation
  severed every other one. We patched the binding with a ContextVar-routed
  client shim so N sims coexist safely.
- SurRoL's upstream repo was restructured; we revived the original package
  from the `main` branch and patched an EGL crash for Apple Silicon.
- pybullet 3.2.6+ doesn't compile on modern macOS SDKs (bundled-zlib fdopen
  collision) — we ship a reproducible patch-and-build script.
- The classic ML trap: our first "converged" model looked great but only
  predicted near-constant tissue. Physics probes caught it.

### Accomplishments we're proud of
- A closed self-improvement loop verified end-to-end: error 0.300 → 0.057
  after one autonomous exploration episode.
- Every architectural invariant has a named failing-test-first suite:
  86 pytest + 20 vitest, all green.
- The whole system runs autonomously for hours — the demo IS the loop.

### What we learned
Error maps are honesty machines: they don't just show progress, they show
*where the model lies to itself*. Replay ratios matter more than learning
rates in continual learning. And "it compiles" says nothing about whether
your event ordering will draw the tree the judges are supposed to see.

### What's next for SOMA
- Multi-step MPC rollout (current: 1-step lookahead, 8 candidates)
- Real tendon-driven tissue FEM instead of grid decay models
- Sim-to-real transfer study on cloth/deformable benchmarks

## Tech stack tags
python · pytorch · fastapi · langgraph · claude-api · pybullet · surrol ·
react · typescript · xyflow-react · plotly · websocket · sqlite · uv · mps

## Assets checklist
- [x] logs/screenshots/canvas.png — agent tree
- [x] logs/screenshots/detail.png — live render + heatmap + bars
- [x] logs/screenshots/compare.png — dual streams + stats bar
- [ ] demo_fallback.mp4 (record per DEMO_SCRIPT.md)
