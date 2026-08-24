# Demo Script — 2:00 (rehearse ×3, target 3:30 with Q&A buffer)

## Pre-flight (30 min before judges)
```
□ backend: uv run uvicorn backend.main:app --port 8000   → /health "ok"
□ frontend: cd frontend && npm run dev                    → :5173 loads
□ browser: Chrome fullscreen, zoom 125%, canvas view
□ Do Not Disturb ON · screensaver OFF · power plugged in
□ W&B tab open (or note offline mode)  □ demo_fallback.mp4 queued
□ if heatmap all-green & no agents: curl localhost:8000/debug/reset_belief
```

## Beats

**[0:00–0:15] Canvas — "every node is a decision"**
Point at SEED root. *"This is SOMA. Everything that grows from this node is
a decision the system made about its own understanding."*

**[0:15–0:45] Spawn + reasoning feed**
Click DETAIL on an active exploration node (trigger one:
`curl localhost:8000/debug/trigger_agent` if quiet). Show the agent feed
streaming Claude's reasoning token-by-token while the left panel renders the
live PyBullet scene. *"It's deciding where to poke the tissue — and telling
us why."*

**[0:45–1:00] THE MONEY MOMENT — error-reduction flash**
Wait for completion (or replay exp_fire02 story). The heatmap cells flash
white→yellow and settle darker/green. *"It came back, it learned, and the
map of its own ignorance just shrank."*

**[1:00–1:30] Compare view — "watch the vessel"**
Switch to COMPARE. Same anatomy, two robots: left has the world model,
right is greedy. *"Watch the vessels."* When reactive clips one, the red
toast fires. Quote the numbers: damage rate **0.016 vs 0.037**.

**[1:30–1:50] DOF progression**
Back on canvas: capability nodes climbing 1→6. *"We seeded it one joint.
It earned the other five by getting good enough to deserve them."*

**[1:50–2:00] Zoom out**
Full tree. *"Every branch is an insight. Every green node is understanding.
That's SOMA."*

## If live breaks
1. Don't debug on stage — switch to demo_fallback.mp4 (narrate over it).
2. Floor mode: `watch -n0.5 'sqlite3 backend/data/events.db "SELECT
   event_type,agent_id,error_before,error_after FROM events ORDER BY id
   DESC LIMIT 8"'` — the event log tells the same story.

## Known quirks (own them calmly)
- First boot auto-trains (~20 s) — overlay shows progress; subsequent boots <2 min.
- Agent feed is empty until you select a node — click one.
- Heatmap can converge to all-green after long runs: that's SUCCESS; hit
  reset_belief to re-seed the story.
