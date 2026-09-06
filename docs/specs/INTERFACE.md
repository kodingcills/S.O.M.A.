# INTERFACE.md
# SOMA — Interface Specification
# Authoritative for: tech stack, TypeScript types, data fetching,
# WebSocket batching, color system, canvas view, node components,
# simulation panel (HTML Canvas), Plotly heatmap, comparison view,
# animation timings, performance requirements, MJPEG stream.
#
# Read this before building: any frontend component.
# Load the frontend-design skill before writing CSS or layouts.
# Invariants referenced here are DEFINED in ARCHITECTURE.md.

## CURRENT INTERFACE DIRECTION — 2026-08-25

The production frontend now uses a graph-first, light scientific interface.
This section supersedes the legacy visual examples later in this document
(dark palette, glowing nodes, particle effects, full-screen detail view, and
`CANVAS / DETAIL / COMPARE` tabs). The data contracts, React Flow version,
pure canvas-state invariant, WebSocket batching, detail polling, and simulation
stream contracts remain authoritative.

Current component/state map:

- `App`: stable top bar, `Graph / Runs / Compare / Models` navigation, bottom
  session status, and view preservation via `display: none`.
- `CanvasView`: the dominant surface; deterministic left-to-right event
  generations, persistent selection, search/filtering, ancestry/descendant
  emphasis, manual live-follow, and a conditional minimap.
- `DetailView`: persistent node inspector with trigger/action/observation/update
  summary, before→after deltas, expandable execution trace, failures, and
  evidence links.
- `SimulationWorkspace`: focused live evidence view. Playback/seeking controls
  stay explicitly unavailable until the backend records frames.
- `RunsView`, `ModelsView`, `CompareView`: aggregate analysis derived only from
  the event log, health state, and atomic detail endpoints.

Current visual principles: light mode, sans-serif hierarchy, selective mono,
warm neutral canvas, white information objects, muted semantic accents, thin
causal edges, no glow or particles, no forced camera motion, and no rainbow
uncertainty map.

---

## TECH STACK

```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "@xyflow/react": "^12.0.0",
    "react-plotly.js": "^2.6.0",
    "plotly.js": "^2.27.0",
    "recharts": "^2.10.0",
    "tailwindcss": "^3.4.0"
  },
  "devDependencies": {
    "typescript": "^5.3.0",
    "vite": "^5.0.0",
    "@vitejs/plugin-react": "^4.2.0",
    "@types/react": "^18.2.0",
    "@types/react-plotly.js": "^2.6.0"
  }
}
```

**What renders what:**
| Component | Library | Why |
|---|---|---|
| Agent tree (infinite canvas) | `@xyflow/react` v12 | Spec. See REACT FLOW VERSION INVARIANT. |
| Prediction error heatmap | Plotly (`react-plotly.js`) | Peraton prize alignment. Built-in colorscale. |
| Regional confidence bars | `recharts` BarChart | Simple, lightweight. Plotly is overkill here. |
| Surgical simulation panel | HTML Canvas 2D API | 16×16 grid + custom overlays at 30fps. SVG can't do this. |
| PyBullet 3D render | MJPEG stream `<img>` tag | Native browser MJPEG decoding. No JS needed. |

See REACT FLOW VERSION INVARIANT in ARCHITECTURE.md before installing.
Run `npm ls @xyflow/react` after install to confirm version 12.x.

### Vite Config

```typescript
// vite.config.ts
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api':  { target: 'http://localhost:8000', changeOrigin: true },
      '/ws':   { target: 'ws://localhost:8000', ws: true },
      '/sim':  { target: 'http://localhost:8000', changeOrigin: true },
    }
  }
})
```

### Environment Variables

```bash
# frontend/.env
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000/ws/events
```

---

## TYPESCRIPT INTERFACES

**File:** `frontend/src/types/index.ts`

Canonical. Field names must exactly match Python JSON serialization.
`T | null` not `T | undefined` — this matters for JSON deserialization.
Do not add fields without updating the backend serializer.

```typescript
export type EventType =
  | 'system_ready' | 'training_started' | 'training_completed'
  | 'simulation_started' | 'simulation_step'
  | 'world_model_updated' | 'belief_snapshot'
  | 'prediction_error_high'
  | 'agent_spawned' | 'agent_step' | 'agent_completed' | 'agent_failed'
  | 'capability_unlocked'

export interface EventLogEntry {
  id:           number
  timestamp:    number             // unix float
  event_type:   EventType
  agent_id:     string | null
  parent_id:    string | null
  sim_id:       string | null
  region:       string | null
  error_before: number | null
  error_after:  number | null
  payload:      Record<string, unknown>
}

export type NodeStatus = 'spawning' | 'active' | 'completed' | 'failed'

export type NodeKind =
  | 'root' | 'exploration' | 'capability' | 'worldmodel' | 'simulation'

export interface SomaNodeData {
  label:       string
  status:      NodeStatus
  agent_id:    string
  kind:        NodeKind
  region:      string | null
  error_before: number | null
  error_after:  number | null
  dof_level:   number | null
  timestamp:   number
  sim_id:      string | null
  // Mutable during lifetime — set by useEffect timer:
  activeMs:    number              // ms since spawned (drives spawning→active)
}

export interface SomaEdgeData {
  label:  string
  status: 'active' | 'completed' | 'failed'
}

export interface BeliefStateSnapshot {
  version:            number
  prediction_error_map: number[][]   // [16][16]
  global_mean_error:  number
  regional_errors:    Record<string, number>  // exactly 6 keys
  active_dof:         number
  episode_count:      number
  timestamp:          number
  error_map:          number[][]    // alias for prediction_error_map
}

export interface SimulationState {
  step_id:           number
  simulation_id:     string
  tissue_integrity:  number[][]     // [16][16]
  tissue_vascularity: number[][]    // [16][16]
  bleeding_mask:     boolean[][]    // [16][16]
  vessels:           VesselState[]
  ee_position:       [number, number, number]  // normalized
  gripper_state:     number
  surgical_target:   { position: [number, number]; reached: boolean }
  active_dof:        number
  task_complete:     boolean
  task_failed:       boolean
  reward:            number
}

export interface VesselState {
  col:     number
  row:     number
  radius:  number
  damaged: boolean
}

export interface DetailResponse {
  simulation_state: SimulationState
  belief_state:     BeliefStateSnapshot
}

export interface HealthResponse {
  status:              'starting' | 'training' | 'ok'
  world_model_version: number
  global_error:        number
  active_agents:       number
  primary_sim_step:    number
  training_progress:   number | null
}
```

---

## COLOR SYSTEM

**File:** `frontend/src/utils/colors.ts`

Single source of truth. Never hardcode hex in components.

```typescript
// Background
export const CANVAS_BG       = '#0A0E1A'
export const SURFACE_BG      = '#0F1629'
export const PANEL_BORDER    = '#1E2D4A'
export const SURFACE_HOVER   = '#162035'

// Text
export const TEXT_PRIMARY    = '#E8EFF8'
export const TEXT_SECONDARY  = '#7A9CC4'
export const TEXT_MONO       = '#5AC8FA'
export const TEXT_ERROR      = '#FF3B30'

// Agent type colors
export const COLOR_EXPLORATION = '#007AFF'  // blue
export const COLOR_CAPABILITY  = '#AF52DE'  // purple
export const COLOR_WORLDMODEL  = '#34C759'  // green
export const COLOR_SIMULATION  = '#5AC8FA'  // light blue
export const COLOR_ROOT        = '#FFFFFF'

// Status colors
export const STATUS_SPAWNING  = '#FFD60A'   // yellow pulse
export const STATUS_COMPLETED = '#34C759'   // green
export const STATUS_FAILED    = '#FF3B30'   // red

// HSL HEATMAP FORMULA — do not change
// errorToColor(0.0) → hsl(120, 85%, 45%) green
// errorToColor(0.5) → hsl(60,  85%, 45%) yellow
// errorToColor(1.0) → hsl(0,   85%, 45%) red
// If 0.5 appears brown: RGB interpolation is being used instead of HSL.
export function errorToColor(error: number): string {
  const e   = Math.max(0, Math.min(1, error))
  const hue = 120 * (1 - e)
  return `hsl(${hue}, 85%, 45%)`
}

// Brighter on hover
export function errorToColorHover(error: number): string {
  const e   = Math.max(0, Math.min(1, error))
  const hue = 120 * (1 - e)
  return `hsl(${hue}, 85%, 65%)`
}

// Map agent_id prefix to type color
export function agentColor(agentId: string): string {
  if (agentId.startsWith('exp_')) return COLOR_EXPLORATION
  if (agentId.startsWith('cap_')) return COLOR_CAPABILITY
  if (agentId === 'root')         return COLOR_ROOT
  if (agentId.startsWith('wm_'))  return COLOR_WORLDMODEL
  return TEXT_SECONDARY
}

// Human-readable region names
export const REGION_LABELS: Record<string, string> = {
  upper_left:              'Upper Left',
  upper_right:             'Upper Right',
  lower_left:              'Lower Left',
  lower_right:             'Lower Right',
  tool_tissue_boundary:    'Tool Zone',
  surgical_target_vicinity:'Target Zone',
}
```

---

## DATA ARCHITECTURE

### WebSocket Hook

**File:** `frontend/src/hooks/useEventLog.ts`

```typescript
export function useEventLog(): EventLogEntry[] {
  const [events, dispatch] = useReducer(eventsReducer, [])
  const bufferRef  = useRef<EventLogEntry[]>([])
  const timerRef   = useRef<ReturnType<typeof setTimeout> | null>(null)
  const lastTsRef  = useRef<number>(0)
  const retryRef   = useRef<number>(0)
  const wsRef      = useRef<WebSocket | null>(null)

  const flush = useCallback(() => {
    if (bufferRef.current.length === 0) return
    dispatch({ type: 'APPEND', events: bufferRef.current })
    bufferRef.current = []
    timerRef.current  = null
  }, [])

  const scheduleFlush = useCallback(() => {
    // BATCH UPDATE: collect events for 100ms, then flush once.
    // Without batching, the connect-time replay burst (potentially
    // 1000+ events in <1s) triggers 1000 re-renders and freezes
    // the browser for several seconds.
    if (timerRef.current) return
    timerRef.current = setTimeout(flush, 100)
  }, [flush])

  useEffect(() => {
    function connect() {
      const ws = new WebSocket(import.meta.env.VITE_WS_URL)
      wsRef.current = ws

      ws.onmessage = (e) => {
        const msg = JSON.parse(e.data)
        if (msg.type === 'event') {
          bufferRef.current.push(msg.data as EventLogEntry)
          lastTsRef.current = Math.max(lastTsRef.current, msg.data.timestamp)
          scheduleFlush()
        }
        if (msg.type === 'ping') {
          ws.send(JSON.stringify({ type: 'pong' }))
        }
      }

      ws.onclose = () => {
        if (retryRef.current < 5) {
          // Exponential backoff: 1s, 2s, 4s, 8s, 16s
          const delay = Math.min(1000 * 2 ** retryRef.current, 16000)
          retryRef.current++
          setTimeout(connect, delay)
        } else {
          startRestFallback()
        }
      }

      ws.onopen = () => { retryRef.current = 0 }
    }

    function startRestFallback() {
      const id = setInterval(async () => {
        const r  = await fetch(`${import.meta.env.VITE_API_URL}/events?since=${lastTsRef.current}`)
        const es = await r.json() as EventLogEntry[]
        if (es.length > 0) {
          dispatch({ type: 'APPEND', events: es })
          lastTsRef.current = es[es.length - 1].timestamp
        }
      }, 500)
      return () => clearInterval(id)
    }

    connect()
    return () => { wsRef.current?.close(); if (timerRef.current) clearTimeout(timerRef.current) }
  }, [scheduleFlush])

  return events
}

function eventsReducer(
  state: EventLogEntry[],
  action: { type: 'APPEND'; events: EventLogEntry[] }
): EventLogEntry[] {
  if (action.type !== 'APPEND') return state
  const next = [...state, ...action.events]
  // Cap at 10,000 to prevent memory growth over long sessions
  return next.length > 10_000 ? next.slice(-10_000) : next
}
```

### Detail View Polling

```typescript
// frontend/src/hooks/useDetail.ts
export function useDetail(simId: string | null): DetailResponse | null {
  const [detail, setDetail] = useState<DetailResponse | null>(null)

  useEffect(() => {
    if (!simId) return
    const id = setInterval(async () => {
      try {
        const r = await fetch(`${import.meta.env.VITE_API_URL}/detail/${simId}`)
        setDetail(await r.json())
      } catch { /* backend starting or offline — keep previous value */ }
    }, 300)
    return () => clearInterval(id)
  }, [simId])

  return detail
}
```

Always clear the interval on unmount. A leaked interval polls even after
the detail view closes — 3 requests/second × every open node ever clicked.

### Health Polling

```typescript
// frontend/src/hooks/useHealth.ts
export function useHealth(): HealthResponse | null {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  useEffect(() => {
    const id = setInterval(async () => {
      try {
        const r = await fetch(`${import.meta.env.VITE_API_URL}/health`)
        setHealth(await r.json())
      } catch { /* ignore — top bar shows stale values */ }
    }, 1000)
    return () => clearInterval(id)
  }, [])
  return health
}
```

---

## APP SHELL

**File:** `frontend/src/App.tsx`

```typescript
type View = 'canvas' | 'detail' | 'compare'

export default function App() {
  const events  = useEventLog()
  const health  = useHealth()
  const [view, setView]         = useState<View>('canvas')
  const [activeNode, setActiveNode] = useState<string | null>(null)

  // Derive active sim_id from active node
  const activeSimId = useMemo(() => {
    if (!activeNode || view !== 'detail') return null
    return activeNode === 'root' ? 'primary' : activeNode
  }, [activeNode, view])

  const detail = useDetail(activeSimId)

  const handleNodeClick = useCallback((nodeId: string) => {
    setActiveNode(nodeId)
    setView('detail')
  }, [])

  return (
    <div style={{ width: '100vw', height: '100vh', background: CANVAS_BG,
                  display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <TopBar
        health={health}
        view={view}
        onViewChange={setView}
        activeNode={activeNode}
        onBack={() => { setView('canvas'); setActiveNode(null) }}
      />
      <div style={{ flex: 1, overflow: 'hidden', position: 'relative' }}>
        {view === 'canvas'  && <CanvasView events={events} onNodeClick={handleNodeClick} />}
        {view === 'detail'  && <DetailView events={events} detail={detail} activeNode={activeNode!} />}
        {view === 'compare' && <CompareView />}
      </div>
      {health?.status === 'training' && <TrainingOverlay health={health} />}
    </div>
  )
}
```

Do not unmount views when switching — use CSS `display: none` instead.
Unmounting the canvas view destroys React Flow's internal position state
and the canvas re-lays-out on every view switch.

```typescript
// Pattern for conditional visibility without unmounting:
<div style={{ display: view === 'canvas' ? 'block' : 'none', ... }}>
  <CanvasView ... />
</div>
```

### Top Bar

```
┌──────────────────────────────────────────────────────────────────┐
│ SOMA / Self-Organizing Model Architecture  │  WM v3 │ ERR 0.34 │ AGENTS 2 │  [CANVAS] [DETAIL] [COMPARE]  │
└──────────────────────────────────────────────────────────────────┘
```

- Height: 48px, `position: sticky top-0`, `z-index: 100`
- Background: `${SURFACE_BG}CC` (80% opacity) + `backdrop-filter: blur(8px)`
- Font: 11px JetBrains Mono throughout

**Stat chip colors:**
- WM VERSION: `COLOR_WORLDMODEL`
- GLOBAL ERROR: `errorToColor(health.global_error)` — color matches heatmap
- ACTIVE AGENTS: `COLOR_EXPLORATION`

When `health.status === 'training'`: replace stats with a training indicator
that pulses amber. The system is not yet ready for demo.

**Training Overlay** (shown over the full screen during `status === 'training'`):

```
┌────────────────────────────────────────────────┐
│              SOMA initializing...              │
│                                                │
│  [████████████████░░░░░░░░░░░░░░░░░░]  44%    │
│  Training prediction network                   │
│  ~60 min remaining on first run.               │
│  Subsequent runs load in <2 minutes.           │
└────────────────────────────────────────────────┘
```

Progress bar fills from `health.training_progress` (0.0→1.0).
Semi-transparent overlay (`rgba(10,14,26,0.92)`) — does not block events.

---

## CANVAS VIEW

**File:** `frontend/src/components/Canvas/index.tsx`

### buildCanvasState — PURE FUNCTION INVARIANT

See PURE CANVAS STATE INVARIANT in ARCHITECTURE.md. This function must
be pure and must be memoized with `useMemo`.

```typescript
// frontend/src/utils/buildCanvasState.ts
import type { Node, Edge } from '@xyflow/react'

export interface CanvasState {
  nodes: Node<SomaNodeData>[]
  edges: Edge<SomaEdgeData>[]
}

export function buildCanvasState(events: EventLogEntry[]): CanvasState {
  // ROOT NODE: always present, not from events
  const nodes: Node<SomaNodeData>[] = [{
    id:       'root',
    type:     'root',
    position: { x: 0, y: 0 },
    data:     { label: 'SEED', status: 'completed', agent_id: 'root',
                kind: 'root', region: null, error_before: null,
                error_after: null, dof_level: 1, timestamp: 0,
                sim_id: 'primary', activeMs: Infinity }
  }]
  const edges: Edge<SomaEdgeData>[] = []
  // Map: agent_id → sibling count per parent (for positioning)
  const siblingCounts = new Map<string, number>()

  for (const event of events) {
    switch (event.event_type) {
      case 'agent_spawned': {
        const kind = agentIdToKind(event.agent_id!)
        const parentId = event.parent_id ?? 'root'
        const key = `${parentId}:${kind}`
        const siblings = siblingCounts.get(key) ?? 0
        siblingCounts.set(key, siblings + 1)

        nodes.push({
          id:   event.agent_id!,
          type: kind,
          position: computePosition(parentId, kind, siblings, nodes),
          data: {
            label:       kindToLabel(kind),
            status:      'spawning',
            agent_id:    event.agent_id!,
            kind,
            region:      event.region,
            error_before: event.error_before,
            error_after:  null,
            dof_level:   (event.payload?.target_dof as number) ?? null,
            timestamp:   event.timestamp,
            sim_id:      event.sim_id,
            activeMs:    0,
          }
        })
        edges.push({
          id:     `${parentId}-${event.agent_id}`,
          source: parentId,
          target: event.agent_id!,
          type:   'smoothstep',
          animated: true,
          label:  edgeLabel(event),
          data:   { label: edgeLabel(event), status: 'active' },
          style:  { stroke: agentColor(event.agent_id!), strokeWidth: 2 },
          labelStyle: { fill: TEXT_SECONDARY, fontSize: 9,
                        fontFamily: 'JetBrains Mono' },
          labelBgStyle: { fill: CANVAS_BG }
        })
        break
      }

      case 'agent_completed': {
        const node = nodes.find(n => n.id === event.agent_id)
        if (node) {
          node.data = { ...node.data, status: 'completed',
                        error_after: event.error_after }
        }
        const edge = edges.find(e => e.target === event.agent_id)
        if (edge) {
          edge.animated = false
          edge.data!.status = 'completed'
          edge.style = { ...edge.style, stroke: STATUS_COMPLETED,
                         strokeWidth: 2, opacity: 0.6 }
        }
        break
      }

      case 'agent_failed': {
        const node = nodes.find(n => n.id === event.agent_id)
        if (node) node.data = { ...node.data, status: 'failed' }
        const edge = edges.find(e => e.target === event.agent_id)
        if (edge) {
          edge.animated = false
          edge.data!.status = 'failed'
          edge.style = { ...edge.style, stroke: STATUS_FAILED, opacity: 0.4 }
        }
        break
      }

      case 'capability_unlocked': {
        const node = nodes.find(n => n.id === event.agent_id)
        if (node) node.data = { ...node.data, status: 'completed',
                                dof_level: event.payload?.new_dof as number }
        break
      }
    }
  }

  return { nodes, edges }
}
```

### Node Positioning

```typescript
const H_SPACING = 250  // px between parent and child
const V_SPACING = 120  // px between siblings

const V_OFFSETS: Record<NodeKind, number> = {
  root:        0,
  exploration: 0,
  capability:  -V_SPACING * 2,   // capability unlocks go upward
  worldmodel:  V_SPACING,
  simulation:  -V_SPACING,
}

function computePosition(
  parentId: string,
  kind: NodeKind,
  siblingIndex: number,
  existingNodes: Node<SomaNodeData>[]
): { x: number; y: number } {
  const parent = existingNodes.find(n => n.id === parentId)
  const px = parent?.position.x ?? 0
  const py = parent?.position.y ?? 0
  const candidate = {
    x: px + H_SPACING,
    y: py + V_OFFSETS[kind] + siblingIndex * V_SPACING
  }
  // Simple collision avoidance: bump y if within 80px of any existing node
  for (let attempt = 0; attempt < 5; attempt++) {
    const collision = existingNodes.some(n =>
      Math.abs(n.position.x - candidate.x) < 80 &&
      Math.abs(n.position.y - candidate.y) < 80
    )
    if (!collision) break
    candidate.y += V_SPACING
  }
  return candidate
}
```

### Canvas Component

```typescript
import { ReactFlow, Background, MiniMap, Controls,
         useNodesState, useEdgesState, useReactFlow } from '@xyflow/react'
import '@xyflow/react/dist/style.css'  // ← MUST be here in App.tsx or Canvas/index.tsx
                                        // exactly once. Without it: blank canvas.

// nodeTypes must be defined OUTSIDE the component render function.
// Defining it inline causes React to think it's a new object every render
// and React Flow re-registers all node types, resetting positions.
const NODE_TYPES = {
  root:        RootNode,
  exploration: ExplorationNode,
  capability:  CapabilityNode,
  worldmodel:  WorldModelNode,
  simulation:  SimulationNode,
}

export function CanvasView({ events, onNodeClick }: Props) {
  // Memoize buildCanvasState — expensive pure function, same events = same output
  const { nodes: builtNodes, edges: builtEdges } = useMemo(
    () => buildCanvasState(events),
    [events]  // ← correct dep: events array reference changes only on new data
  )

  const [nodes, setNodes, onNodesChange] = useNodesState(builtNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(builtEdges)

  // Sync built state into RF state when events change
  useEffect(() => { setNodes(builtNodes) }, [builtNodes, setNodes])
  useEffect(() => { setEdges(builtEdges) }, [builtEdges, setEdges])

  // Camera: follow rightmost active node, debounced 2000ms
  const { setCenter } = useReactFlow()
  useEffect(() => {
    const active = nodes.filter(n => n.data.status === 'active')
    if (active.length === 0) return
    const rightmost = active.reduce((a, b) =>
      a.position.x > b.position.x ? a : b
    )
    const id = setTimeout(() => {
      setCenter(rightmost.position.x + 70, rightmost.position.y + 45,
                { zoom: 0.8, duration: 800 })
    }, 2000)
    return () => clearTimeout(id)
  }, [nodes, setCenter])

  return (
    <ReactFlow
      nodes={nodes} edges={edges}
      onNodesChange={onNodesChange} onEdgesChange={onEdgesChange}
      onNodeClick={(_, node) => onNodeClick(node.id)}
      nodeTypes={NODE_TYPES}
      nodesDraggable={false}     // agents manage their own positions
      nodesConnectable={false}   // no manual edge creation
      nodesFocusable={false}     // prevents tab-key canvas navigation
      fitView={false}
      minZoom={0.1} maxZoom={2.0}
      defaultViewport={{ x: 100, y: 200, zoom: 0.8 }}
      proOptions={{ hideAttribution: true }}
      style={{ background: CANVAS_BG }}
    >
      <Background variant="dots" gap={24} size={1} color={PANEL_BORDER} />
      <MiniMap
        position="bottom-right"
        nodeColor={(n) => agentColor(n.data?.agent_id as string ?? '')}
        maskColor="rgba(10,14,26,0.8)"
        style={{ background: SURFACE_BG, border: `1px solid ${PANEL_BORDER}`,
                 borderRadius: 8, width: 160, height: 100 }}
      />
      <Controls position="bottom-left"
        style={{ background: SURFACE_BG, border: `1px solid ${PANEL_BORDER}`,
                 borderRadius: 8 }} />
    </ReactFlow>
  )
}
```

**Why `nodeTypes` outside component:** React creates a new object literal
on every render. React Flow compares nodeTypes by reference. New reference
= re-register all types = reset all node positions to zero. This causes
the canvas to collapse to `(0,0)` on every WebSocket event.

**Why `nodesDraggable={false}`:** Dragging changes node positions, breaking
the PURE CANVAS STATE INVARIANT (dragged positions are not derivable from
events). The canvas would diverge from the event log and re-synchronize
on the next event, causing nodes to snap back.

---

## NODE COMPONENTS

All nodes: `width=140px, height=90px`. Set via `style` prop on root div
AND via React Flow's `measured.width`/`measured.height` if using v12 auto-sizing.
Fixed dimensions prevent layout recalculation on every render.

### Spawning → Active Transition

All nodes start with `status: 'spawning'`. After 500ms they become 'active'.
Drive this with a `useEffect` timer keyed on the event timestamp, not `Date.now()`.

```typescript
// Inside any node component:
useEffect(() => {
  if (data.status !== 'spawning') return
  const elapsedMs = Date.now() - data.timestamp * 1000
  const remaining = Math.max(0, 500 - elapsedMs)
  // Use remaining time so late-rendered nodes (from replay) transition immediately
  const id = setTimeout(() => {
    setLocalStatus('active')
  }, remaining)
  return () => clearTimeout(id)
}, [data.status, data.timestamp])
```

Why keyed on `data.timestamp * 1000`: if the frontend connects after the
system has been running, events replay in a burst. An agent that spawned
10 minutes ago has `remaining ≈ 0` and transitions to 'active' immediately.
Using `Date.now()` for the 500ms window would make all replayed agents
flash 'spawning' for 500ms before going active — visually wrong.

### Spawn Pop Animation

```css
/* In node component's root div */
@keyframes nodeSpawn {
  0%   { transform: scale(0);    opacity: 0; }
  70%  { transform: scale(1.08); opacity: 1; }
  100% { transform: scale(1.0);  opacity: 1; }
}
.node-spawn { animation: nodeSpawn 300ms cubic-bezier(0.34, 1.56, 0.64, 1) forwards; }
```

Apply this class when the node mounts.

### Active Glow Pulse

```css
@keyframes nodePulse {
  0%, 100% { box-shadow: none; }
  50%      { box-shadow: 0 0 8px var(--node-color); }
}
.node-active { animation: nodePulse 2000ms ease-in-out infinite; }
```

Stop on completion by removing the class.

### Node Components (per type)

**Shared structure — every node:**
```
┌──────────────────────────────┐  ← 2px border, color = type color
│  LABEL                    ●  │  ← header 20px, type-colored bg, status dot
│                              │
│  content varies by type      │  ← 70px content area
└──────────────────────────────┘
```

**RootNode:** `SEED` label. Content: `DOF {dof_level}` + status text.
Click → `onNodeClick('root')` → detail view for `primary_sim`.

**ExplorationNode:** `EXPLORE` label, `COLOR_EXPLORATION` border.
Content shows region name formatted (`errorToColor` the status dot):
```
Tool Zone              ●  (active: blue pulse)
0.43 → 0.21    ✓      (completed: green arrow)
0.43 → ...     ···    (active: pulsing dots)
```

**CapabilityNode:** `CAPABILITY` label, `COLOR_CAPABILITY` border.
Content: `DOF {n} Unlock` + 6 tiny circles showing DOF progress.

**WorldModelNode:** `WM v{version}` label, `COLOR_WORLDMODEL` border.
Content: 32×32 mini-heatmap thumbnail + error value colored by `errorToColor`.
Fetch thumbnail from `/detail/primary` once on mount and cache it.

**SimulationNode:** `SIM #{id_short}` label, `COLOR_SIMULATION` border.
Content: 32×32 tissue integrity thumbnail snapshot.

### Completion Particle Burst

On status → 'completed': create 8 particles in a `useEffect`:
```typescript
useEffect(() => {
  if (status !== 'completed') return
  const particles = Array.from({ length: 8 }, (_, i) => ({
    id: i,
    dx: Math.cos(i * Math.PI / 4) * 30,
    dy: Math.sin(i * Math.PI / 4) * 30,
  }))
  setParticles(particles)
  const id = setTimeout(() => setParticles([]), 400)
  return () => clearTimeout(id)
}, [status])
```

Each particle: 4px circle, `STATUS_COMPLETED` color, absolute positioned
relative to node, animates from center to (dx, dy) over 400ms, fades out.

---

## DETAIL VIEW

**File:** `frontend/src/components/DetailView/index.tsx`

```
┌─────────────────────────────────────────────────────────┐
│ TOP BAR: [← CANVAS] SOMA / exploration / exp_a3f7   ... │
├─────────────────────────────┬───────────────────────────┤
│                             │                           │
│      SimulationPanel        │    WorldModelPanel        │
│         55% width           │       45% width           │
│                             │                           │
├─────────────────────────────┴───────────────────────────┤
│ AgentFeed                          22% viewport height  │
└─────────────────────────────────────────────────────────┘
```

Polling: `/detail/{sim_id}` every 300ms — both panels read from this single
response to ensure time-coherence. See Contract C7 in ARCHITECTURE.md.

### PyBullet 3D Render

The SimulationPanel shows the actual PyBullet 3D scene via MJPEG stream.

**Backend endpoint (add to api.py):**
```python
from fastapi.responses import StreamingResponse
import pybullet as p
import io, time

async def pybullet_frame_generator(env):
    while True:
        width, height, rgba, depth, seg = p.getCameraImage(
            320, 240,
            viewMatrix=p.computeViewMatrix([0.0, -0.3, 0.2], [0,0,0], [0,0,1]),
            projectionMatrix=p.computeProjectionMatrixFOV(60, 320/240, 0.01, 10),
            physicsClientId=env.physics_client_id,
        )
        import numpy as np
        from PIL import Image
        img = Image.fromarray(np.array(rgba, dtype=np.uint8).reshape(height, width, 4)[:,:,:3])
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=85)
        frame = buf.getvalue()
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
        await asyncio.sleep(1/15)   # 15fps — sufficient for demo

@app.get("/sim/{sim_id}/stream")
async def stream_sim(sim_id: str):
    env = sim_manager.get_env(sim_id)
    return StreamingResponse(
        pybullet_frame_generator(env),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
```

**Frontend usage:**
```typescript
// Inside SimulationPanel, shows live 3D render:
<img
  src={`${import.meta.env.VITE_API_URL}/sim/${simId}/stream`}
  style={{ width: '100%', aspectRatio: '4/3', objectFit: 'contain',
           imageRendering: 'pixelated' }}
  alt="Live simulation"
/>
```

No polling. No state. The browser natively handles MJPEG streams.
The browser sends one HTTP request; the server streams frames indefinitely.

### SimulationPanel — HTML Canvas Overlay

On top of the MJPEG stream, draw a 2D canvas overlay for interactive elements
(vessel markers, target ring, prediction error overlay, action history).
Position the canvas absolutely over the MJPEG `<img>` with `pointer-events: none`.

```typescript
useEffect(() => {
  const canvas = canvasRef.current
  if (!canvas || !sim) return
  const ctx  = canvas.getContext('2d')!
  const W    = canvas.width
  const H    = canvas.height
  const CELL = W / 16
  ctx.clearRect(0, 0, W, H)

  // LAYER: subtle grid
  ctx.strokeStyle = 'rgba(255,255,255,0.05)'
  for (let i = 0; i <= 16; i++) {
    ctx.beginPath(); ctx.moveTo(i*CELL, 0); ctx.lineTo(i*CELL, H); ctx.stroke()
    ctx.beginPath(); ctx.moveTo(0, i*CELL); ctx.lineTo(W, i*CELL); ctx.stroke()
  }

  // LAYER: vessels
  for (const v of sim.vessels) {
    ctx.beginPath()
    ctx.arc(v.col*CELL, v.row*CELL, v.radius*CELL, 0, Math.PI*2)
    if (v.damaged) {
      ctx.fillStyle = `rgba(255,59,48,${0.6 + 0.2*Math.sin(Date.now()/125)})`
    } else {
      ctx.fillStyle = 'rgba(90,200,250,0.3)'
      ctx.strokeStyle = 'rgba(90,200,250,0.6)'
      ctx.lineWidth = 1
      ctx.stroke()
    }
    ctx.fill()
  }

  // LAYER: surgical target
  const [tc, tr] = sim.surgical_target.position
  const pulse = 0.15 + 0.15*Math.sin(Date.now()/500)
  ctx.beginPath(); ctx.arc(tc*CELL, tr*CELL, 18, 0, Math.PI*2)
  ctx.fillStyle = `rgba(52,199,89,${sim.surgical_target.reached ? 0.4 : pulse})`
  ctx.fill()
  ctx.beginPath(); ctx.arc(tc*CELL, tr*CELL, 8, 0, Math.PI*2)
  ctx.fillStyle = `rgba(52,199,89,${sim.surgical_target.reached ? 0.9 : 0.7})`
  ctx.fill()

  // LAYER: prediction error overlay (toggleable)
  if (showPredictionOverlay && detail?.belief_state) {
    const em = detail.belief_state.prediction_error_map
    for (let r = 0; r < 16; r++) {
      for (let c = 0; c < 16; c++) {
        if (em[r][c] > 0.3) {
          ctx.strokeStyle = `rgba(255,214,0,${em[r][c] * 0.8})`
          ctx.lineWidth = 1
          ctx.strokeRect(c*CELL, r*CELL, CELL, CELL)
        }
      }
    }
  }

  // requestAnimationFrame loop for vessel flash + target pulse
  const raf = requestAnimationFrame(() => { /* re-trigger */ })
  return () => cancelAnimationFrame(raf)
}, [sim, detail, showPredictionOverlay])
```

### WorldModelPanel — Plotly Heatmap

```typescript
import Plot from 'react-plotly.js'

// Inside WorldModelPanel:
const plotlyData: Plotly.Data[] = useMemo(() => [{
  z:    belief?.prediction_error_map ?? Array(16).fill(Array(16).fill(0)),
  type: 'heatmap' as const,
  colorscale: [
    [0,   'hsl(120,85%,45%)'],   // green at 0
    [0.5, 'hsl(60,85%,45%)'],    // yellow at 0.5
    [1,   'hsl(0,85%,45%)'],     // red at 1
  ],
  zmin: 0, zmax: 1,
  showscale: true,
  colorbar: {
    title: { text: 'Error', font: { color: TEXT_SECONDARY, size: 10 } },
    thickness: 12,
    tickfont: { color: TEXT_SECONDARY, size: 9 },
  },
  hovertemplate: 'Row %{y}, Col %{x}<br>Error: %{z:.4f}<extra></extra>',
}], [belief?.prediction_error_map])

const layout: Partial<Plotly.Layout> = useMemo(() => ({
  paper_bgcolor: SURFACE_BG,
  plot_bgcolor:  SURFACE_BG,
  font:  { color: TEXT_SECONDARY, family: 'JetBrains Mono', size: 9 },
  margin: { t: 24, r: 16, b: 24, l: 16 },
  xaxis: { title: { text: 'Column' }, gridcolor: PANEL_BORDER, zeroline: false },
  yaxis: { title: { text: 'Row' },    gridcolor: PANEL_BORDER, zeroline: false,
           autorange: 'reversed' },   // row 0 at top
  annotations: [{
    text: `Global: ${(belief?.global_mean_error ?? 0).toFixed(3)}`,
    xref: 'paper', yref: 'paper', x: 1, y: 1.05, xanchor: 'right',
    showarrow: false,
    font: { color: errorToColor(belief?.global_mean_error ?? 0),
            size: 11, family: 'JetBrains Mono' },
  }]
}), [belief])

<Plot
  data={plotlyData}
  layout={layout}
  config={{ displayModeBar: false, responsive: true }}
  style={{ width: '100%', height: '45%' }}
  useResizeHandler
/>
```

### Error Reduction Animation

Triggered by `agent_completed` events where `error_after < error_before - 0.05`.
The animation is a CSS overlay div — Plotly's own animation API is too slow.

```typescript
// Track animation state per cell
interface CellAnimation {
  row: number; col: number
  startMs: number
}
const [flashCells, setFlashCells] = useState<CellAnimation[]>([])

// Watch for qualifying agent_completed events
useEffect(() => {
  const qualifying = events.filter(e =>
    e.event_type === 'agent_completed' &&
    e.error_before != null && e.error_after != null &&
    (e.error_before - e.error_after) > 0.05
  )
  if (qualifying.length === 0) return
  // Trigger flash on all cells in the affected region
  const region = qualifying[qualifying.length - 1].region
  if (!region) return
  const cells = regionToCells(region)  // returns [{row,col}] for the region
  setFlashCells(cells.map(c => ({ ...c, startMs: Date.now() })))
  const id = setTimeout(() => setFlashCells([]), 600)
  return () => clearTimeout(id)
}, [events])
```

The flash: a positioned `<div>` absolutely overlaid on the Plotly container,
using CSS `keyframes` to go white → yellow → transparent over 600ms.
`pointer-events: none` on the overlay.

```css
@keyframes heatmapFlash {
  0%   { background: rgba(255,255,255,0.8); }  /* white flash */
  20%  { background: rgba(255,214,0,0.6); }    /* yellow */
  100% { background: rgba(255,214,0,0); }      /* fade out */
}
.heatmap-flash { animation: heatmapFlash 600ms ease-out forwards; }
```

This is the most important animation in the entire demo. It must fire
reliably on every `agent_completed` event with sufficient error reduction.
Test it manually 5× before Sprint 3.

### Regional Confidence Bars (recharts)

```typescript
import { BarChart, Bar, XAxis, YAxis, Cell } from 'recharts'

const barData = useMemo(() =>
  Object.entries(belief?.regional_errors ?? {}).map(([key, err]) => ({
    name:       REGION_LABELS[key] ?? key,
    confidence: Math.round((1 - err) * 100),
    error:      err,
  })),
  [belief?.regional_errors]
)

<BarChart data={barData} layout="vertical" width={panelWidth - 32} height={150}>
  <XAxis type="number" domain={[0,100]} hide />
  <YAxis type="category" dataKey="name" width={72}
         tick={{ fill: TEXT_SECONDARY, fontSize: 10 }} />
  <Bar dataKey="confidence" radius={3}>
    {barData.map((d, i) => (
      <Cell key={i} fill={errorToColor(d.error)} />
    ))}
  </Bar>
</BarChart>
```

### DOF Progress Display

6 circles in a row. Left-to-right = DOF 1→6. Filled = unlocked.
The circle being unlocked (CapabilityAgent active) pulses between
`COLOR_WORLDMODEL` and `COLOR_CAPABILITY`.

### AgentFeed

**File:** `frontend/src/components/DetailView/AgentFeed.tsx`

```typescript
// Filter events to the selected node's subtree
const feedEvents = useMemo(() =>
  events.filter(e =>
    e.agent_id === activeNode ||
    e.parent_id === activeNode ||
    // Include world model updates triggered by this agent
    (e.event_type === 'world_model_updated' && isRecentlyAfter(e, lastCompleted))
  ).reverse(),   // newest first
  [events, activeNode]
)
```

Each entry (one line, fixed height 20px, virtual list if >500 entries):
```
14:32:01  SPAWN  exp_a3f7  region:Tool Zone | err:0.43
14:32:08  STEP   exp_a3f7  step:10/100
14:35:12  RESULT exp_a3f7  0.43 → 0.21 ✓
```

**Claude API reasoning (from streaming):**
When `event_type === 'agent_step'` and `payload.reasoning` is present,
render it as a sub-row indented below the step entry:
```
14:32:09         ···      "Moving toward upper-left quadrant where vascularity
                            density suggests unexplored tissue dynamics."
```
Color: `TEXT_MONO` (light blue monospace). This is the "agents thinking"
visible to judges. It appears token-by-token if the WebSocket receives
intermediate events during streaming.

---

## COMPARISON VIEW

**File:** `frontend/src/components/CompareView/index.tsx`

```
┌──────────────────────────────┬──────────────────────────────┐
│  SOMA (MPC + World Model)    │  Reactive Agent (No WM)      │
│  ┌────────────────────────┐  │  ┌────────────────────────┐  │
│  │   PyBullet 3D stream   │  │  │   PyBullet 3D stream   │  │
│  │   /sim/primary/stream  │  │  │   /sim/comparison/     │  │
│  └────────────────────────┘  │  └────────────────────────┘  │
├──────────────────────────────┴──────────────────────────────┤
│  STEPS: 143  vs  139  │  TISSUE: 94.2%  vs  71.3%  │  VESSELS: 0 ✓  vs  1 ✗  │
└──────────────────────────────────────────────────────────────┘
```

Poll `GET /detail/primary` and `GET /detail/comparison` simultaneously.
The stats bar bolts the better value green:
- Steps: lower = better (efficiency)
- Tissue integrity: higher = better
- Vessels damaged: lower = better, show `0 ✓` in green / `1 ✗` in red

**Vessel damage notification:**
When `comparison_sim.vessels` shows a newly damaged vessel:
```typescript
// Detect transition: previously undamaged → now damaged
if (prevState && sim.vessels.some((v,i) => v.damaged && !prevState.vessels[i].damaged)) {
  setNotification({ text: 'Reactive agent: vessel damaged', type: 'danger' })
  setTimeout(() => setNotification(null), 3000)
}
```

Toast notification: `position: absolute`, bottom-center of the reactive panel,
`background: rgba(255,59,48,0.1)`, `border: 1px solid ${TEXT_ERROR}`, fades out.

---

## ANIMATION TIMINGS

All exact. Do not approximate.

| Animation | Trigger | Duration | Keyframes |
|---|---|---|---|
| Node spawn | On mount | 300ms | 0%: scale(0) opacity(0) → 70%: scale(1.08) → 100%: scale(1) |
| Spawning→active | 500ms after `event.timestamp` | Instant | status class swap |
| Node glow pulse | While `status === 'active'` | 2000ms infinite | box-shadow 0→8px→0 |
| Completion border | `status → 'completed'` | 800ms | border: white→green (400ms) |
| Particle burst | `status → 'completed'` | 400ms | 8 divs radiate 30px outward, fade |
| Edge draw | On edge mount | 500ms | stroke-dashoffset path_length→0 |
| Heatmap cell | Continuous | Per frame | Exponential approach: `e += (target-e) × 0.15` |
| Error reduction flash | `agent_completed` with `Δerror > 0.05` | 600ms | white→yellow→transparent |
| DOF badge unlock | `capability_unlocked` | 1500ms | 3 green pulses (500ms each) |
| Camera pan | New rightmost node | 800ms | `setCenter` with `duration: 800` |
| Vessel damage flash | `vessel.damaged` becomes true | Continuous | `opacity: 0.6 + 0.2×sin(t/125)` at 8Hz |

---

## PERFORMANCE

| Concern | Requirement | Implementation |
|---|---|---|
| Canvas FPS target | 60fps to 200 nodes | React Flow virtualizes off-screen nodes — no manual virtualization |
| Heatmap FPS | 30fps smooth interpolation | `requestAnimationFrame` loop runs always, interpolates `current_error` toward `target_error` |
| Sim canvas FPS | 30fps | `requestAnimationFrame` keyed on sim poll timestamp |
| Events array cap | 10,000 max | Slice oldest in `eventsReducer` |
| Agent feed display cap | 500 visible entries | `react-window` `FixedSizeList` for virtual scrolling |
| `buildCanvasState` | Never re-run without new events | `useMemo([events])` — events ref stable until new data arrives |
| `nodeTypes` | Never re-created per render | Defined outside component at module scope |
| WebSocket batch | 100ms debounce on state flush | Prevents 1000+ renders during connect-time replay burst |
| Polling cleanup | All intervals cleared on unmount | Return `() => clearInterval(id)` from every `useEffect` |
| MJPEG memory | Browser handles | Native MJPEG decoding — no JS buffer |

---

## PITFALLS — SYMPTOM FIRST

---

**SYMPTOM: Canvas is a white rectangle. No nodes visible. No console errors.**

CAUSE 1 (most likely): `@xyflow/react/dist/style.css` not imported.
Nodes are positioned in React Flow state but rendered without the library's
CSS. No error is thrown — nodes simply have no visible style.
CONFIRM: `grep -r "xyflow/react/dist/style.css" frontend/src` — must return one hit.
FIX: Add `import '@xyflow/react/dist/style.css'` to `App.tsx`. Exactly once.
Test: `test:frontend:react_flow_css`

CAUSE 2: `reactflow` (v11) installed instead of `@xyflow/react` (v12).
CONFIRM: `cat frontend/package.json | grep -E "xyflow|reactflow"`.
FIX: `npm uninstall reactflow && npm install @xyflow/react`.

CAUSE 3: Parent container has no explicit height. ReactFlow uses `100%` height
of its container — if the container has `height: 0`, the canvas is invisible.
CONFIRM: Inspect parent element height in browser DevTools.
FIX: Set `height: 100vh` or `flex: 1` on the container.

---

**SYMPTOM: Canvas re-animates the full layout every time a new event arrives.
All nodes jump to new positions on each WebSocket message.**

CAUSE: `nodeTypes` defined inside the component render function.
React Flow detects a new object reference and re-registers all node types,
resetting positions to (0,0).
CONFIRM: Move `const NODE_TYPES = {...}` outside the component. If the
flickering stops: this was the cause.
FIX: See PURE CANVAS STATE INVARIANT. Also verify `buildCanvasState` is
called through `useMemo` and not called directly on every render.
Test: `test:frontend:pure_canvas`

---

**SYMPTOM: Error reduction animation never fires even though `agent_completed`
events arrive with different `error_before` and `error_after` values.**

CAUSE 1: `error_before` and `error_after` are `null` on the event.
CONFIRM: `console.log(events.filter(e => e.event_type === 'agent_completed'))`.
If `error_before: null`: the backend wrote `agent_completed` without passing
these fields. FIX: Fix in `backend/agents/exploration.py` — pass `error_before`
and `error_after` to `write_event` as top-level kwargs, not inside payload.

CAUSE 2: The `useEffect` watching for qualifying events has wrong dependency
array and doesn't re-fire when new events arrive.
CONFIRM: Add `console.log('Checking events for flash')` inside the `useEffect`.
If it doesn't log on `agent_completed` arrival: stale closure on `events`.
FIX: Ensure `events` is in the dependency array of the animation `useEffect`.

---

**SYMPTOM: Plotly heatmap renders but has wrong colors. Mid-range values
appear brown or orange instead of yellow.**

CAUSE: RGB interpolation used in the colorscale instead of HSL.
Plotly uses the colorscale CSS format but may interpret HSL differently
in some versions.
CONFIRM: `errorToColor(0.5)` in browser console must return `hsl(60, 85%, 45%)`.
If the heatmap mid-value looks orange: the colorscale is using hex/RGB.
FIX: Use the exact colorscale format:
```
[[0, 'hsl(120,85%,45%)'], [0.5, 'hsl(60,85%,45%)'], [1, 'hsl(0,85%,45%)']]
```
If Plotly doesn't support HSL in colorscale: convert to hex and test
that `#2db82d` (green), `#b89e2d` (yellow), `#b82d2d` (red) look correct.

---

**SYMPTOM: MJPEG stream shows in Safari but not in Chrome, or shows in
Chrome but freezes after ~30 seconds.**

CAUSE: Chrome requires correct MIME type for MJPEG.
FIX: Confirm `media_type="multipart/x-mixed-replace; boundary=frame"` in
`StreamingResponse`. Add `Cache-Control: no-cache, no-store` header.

CAUSE (freeze): Backend not yielding the event loop between frames — the
`await asyncio.sleep(1/15)` is missing or blocked.
FIX: Verify `pybullet_frame_generator` is an `async def` with `await asyncio.sleep`.
`p.getCameraImage()` is synchronous — wrap in `asyncio.to_thread`.

---

## TESTING PROTOCOL

Run only these after Tasks 2.1–2.7. Sprint checkpoint runs full suite.

```bash
# Setup (Task 2.1)
npx tsc --noEmit                                    # TypeScript zero errors
grep -c "xyflow/react/dist/style.css" frontend/src/App.tsx  # must return 1

# Canvas (Task 2.2)
# Manual: start backend, load frontend. Canvas shows root node.
# Manual: trigger agent_spawned event. New node appears animated.
# Manual: buildCanvasState(events) === buildCanvasState(events) [deep equal]

# Heatmap animation (Task 2.4) — critical demo moment
# Manual: force an agent_completed event with error_before=0.5, error_after=0.2
# Verify: flash animation fires (white→yellow→fade over 600ms)
# Test at least 5x before Sprint 3

# Comparison view (Task 2.5)
# Manual: both sim streams load. Stats update. Vessel damage toast fires.
```

---

## VERIFICATION CHECKLIST

```
SETUP:
  □ @xyflow/react ^12.0.0 installed, not reactflow
  □ CSS import in App.tsx, exactly once
  □ TypeScript compiles with zero errors
  □ nodeTypes defined outside component

COLOR SYSTEM:
  □ errorToColor(0.0) → visible green in browser
  □ errorToColor(0.5) → visible yellow (not brown)
  □ errorToColor(1.0) → visible red

DATA FLOW:
  □ WebSocket connects on load
  □ 100ms batch flush prevents burst re-renders
  □ REST fallback activates after 5 failed reconnects
  □ Events array capped at 10,000 entries
  □ All polling intervals cleared on component unmount

CANVAS:
  □ Root node visible at (0,0)
  □ New node appears on agent_spawned (animated pop)
  □ Node status transitions spawning→active after 500ms
  □ Active node has glow pulse
  □ Completed node has border flash + particles
  □ Canvas does NOT re-layout on every event
  □ Camera follows rightmost active node (debounced 2s)

DETAIL VIEW:
  □ MJPEG stream loads in SimulationPanel
  □ Canvas overlay shows vessels, target, prediction errors
  □ Plotly heatmap updates every 300ms
  □ Error reduction animation fires on qualifying agent_completed
  □ Regional confidence bars update smoothly
  □ DOF circles show correct state
  □ AgentFeed shows Claude reasoning in monospace

COMPARISON VIEW:
  □ Both MJPEG streams load simultaneously
  □ Stats bar bolts better value green
  □ Vessel damage notification fires and fades
```
