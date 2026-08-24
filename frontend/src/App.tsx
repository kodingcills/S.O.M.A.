// SOMA App shell — per INTERFACE.md "APP SHELL".
// Views are NEVER unmounted on switch: display:none pattern preserves
// React Flow internal position state (canvas would re-layout otherwise).
import { useCallback, useMemo, useState } from 'react'
import '@xyflow/react/dist/style.css' // REACT FLOW VERSION INVARIANT — exactly once in App.tsx
import { useEventLog } from './hooks/useEventLog'
import { useHealth } from './hooks/useHealth'
import { useDetail } from './hooks/useDetail'
import type { HealthResponse } from './types'
import {
  CANVAS_BG, SURFACE_BG, PANEL_BORDER,
  TEXT_PRIMARY, TEXT_SECONDARY,
  COLOR_WORLDMODEL, COLOR_EXPLORATION, STATUS_SPAWNING,
  errorToColor,
} from './utils/colors'
import { CanvasView } from './components/Canvas'
import { DetailView } from './components/DetailView'
import { CompareView } from './components/CompareView'

type View = 'canvas' | 'detail' | 'compare'

const MONO = "'JetBrains Mono', ui-monospace, Menlo, monospace"

export default function App() {
  const events  = useEventLog()
  const health  = useHealth()
  const [view, setView]         = useState<View>(() => {
    // QA hook: ?view=detail|compare deep-links a view for screenshot runs.
    const v = new URLSearchParams(window.location.search).get('view')
    return v === 'detail' || v === 'compare' ? v : 'canvas'
  })
  const [activeNode, setActiveNode] = useState<string | null>(null)

  // Derive active sim_id from active node.
  // Agent nodes (exp_/cap_/wm_/sim_) execute on primary-anatomy envs; the
  // backend registry only exposes 'primary'/'comparison', so polling
  // /detail/{exp_x} would 404 forever — map them to 'primary'.
  const activeSimId = useMemo(() => {
    if (view !== 'detail') return null
    // Deep-linked detail (?view=detail) has no node yet — stream primary.
    if (!activeNode) return 'primary'
    if (activeNode === 'root' || /^(exp_|cap_|wm_|sim_)/.test(activeNode)) {
      return 'primary'
    }
    return activeNode
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
        {/* display:none visibility pattern — do NOT unmount views */}
        <div style={{ position: 'absolute', inset: 0,
                      display: view === 'canvas' ? 'block' : 'none' }}>
          <CanvasView events={events} onNodeClick={handleNodeClick} />
        </div>
        <div style={{ position: 'absolute', inset: 0,
                      display: view === 'detail' ? 'block' : 'none' }}>
          <DetailView events={events} detail={detail} activeNode={activeNode ?? ''} />
        </div>
        <div style={{ position: 'absolute', inset: 0,
                      display: view === 'compare' ? 'block' : 'none' }}>
          <CompareView />
        </div>
      </div>
      {health?.status === 'training' && <TrainingOverlay health={health} />}
    </div>
  )
}

// ── Top Bar ─────────────────────────────────────────────────────────────
// 48px, sticky top-0, z-100, SURFACE_BG+CC backdrop-blur, JetBrains Mono 11px.
interface TopBarProps {
  health:      HealthResponse | null
  view:        View
  onViewChange: (v: View) => void
  activeNode:  string | null
  onBack:      () => void
}

function TopBar({ health, view, onViewChange, activeNode, onBack }: TopBarProps) {
  const training = health?.status === 'training'

  return (
    <div style={{
      height: 48, minHeight: 48,
      display: 'flex', alignItems: 'center',
      padding: '0 16px', gap: 16,
      background: `${SURFACE_BG}CC`,
      backdropFilter: 'blur(8px)',
      WebkitBackdropFilter: 'blur(8px)',
      borderBottom: `1px solid ${PANEL_BORDER}`,
      fontFamily: MONO, fontSize: 11,
      color: TEXT_PRIMARY,
      position: 'sticky', top: 0, zIndex: 100,
    }}>
      {/* Title */}
      <span style={{ letterSpacing: '0.08em', whiteSpace: 'nowrap' }}>
        SOMA <span style={{ color: TEXT_SECONDARY }}>/ Self-Organizing Model Architecture</span>
      </span>

      {/* Stat chips — replaced by pulsing amber indicator while training */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 14, flex: 1 }}>
        {training ? (
          <span style={{ color: STATUS_SPAWNING, animation: 'pulseAmber 2000ms ease-in-out infinite' }}>
            ● TRAINING…
          </span>
        ) : (
          <>
            <Chip label="WM VERSION" value={`v${health?.world_model_version ?? 0}`}
                  color={COLOR_WORLDMODEL} />
            <Chip label="GLOBAL ERROR"
                  value={(health?.global_error ?? 0).toFixed(3)}
                  color={errorToColor(health?.global_error ?? 0)} />
            <Chip label="ACTIVE AGENTS" value={String(health?.active_agents ?? 0)}
                  color={COLOR_EXPLORATION} />
          </>
        )}
      </div>

      {/* Back button (detail view) */}
      {view === 'detail' && (
        <>
          <span style={{ color: TEXT_SECONDARY, whiteSpace: 'nowrap' }}>
            NODE {activeNode}
          </span>
          <button onClick={onBack} style={btnStyle}>
            ← CANVAS
          </button>
        </>
      )}

      {/* View switch buttons */}
      <div style={{ display: 'flex', gap: 6 }}>
        {(['canvas', 'detail', 'compare'] as View[]).map(v => (
          <button key={v}
                  onClick={() => onViewChange(v)}
                  style={{
                    ...btnStyle,
                    ...(view === v
                      ? { borderColor: TEXT_PRIMARY, color: TEXT_PRIMARY }
                      : {}),
                  }}>
            {v.toUpperCase()}
          </button>
        ))}
      </div>
    </div>
  )
}

function Chip({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <span style={{ display: 'inline-flex', alignItems: 'baseline', gap: 6, whiteSpace: 'nowrap' }}>
      <span style={{ color: TEXT_SECONDARY }}>{label}</span>
      <span style={{ color }}>{value}</span>
    </span>
  )
}

const btnStyle: React.CSSProperties = {
  fontFamily: MONO, fontSize: 10,
  background: 'transparent',
  color: TEXT_SECONDARY,
  border: `1px solid ${PANEL_BORDER}`,
  borderRadius: 4,
  padding: '4px 8px',
  cursor: 'pointer',
  whiteSpace: 'nowrap',
}

// ── Training Overlay ────────────────────────────────────────────────────
// Full-screen during status==='training'. Semi-transparent — does not block events.
function TrainingOverlay({ health }: { health: HealthResponse }) {
  const progress = health.training_progress ?? 0
  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 200,
      background: 'rgba(10,14,26,0.92)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      pointerEvents: 'none',
      fontFamily: MONO,
    }}>
      <div style={{
        width: 420, border: `1px solid ${PANEL_BORDER}`, borderRadius: 8,
        background: SURFACE_BG, padding: '24px 28px',
      }}>
        <div style={{ color: TEXT_PRIMARY, fontSize: 13, marginBottom: 18, textAlign: 'center' }}>
          SOMA initializing...
        </div>
        <div style={{
          height: 10, width: '100%', borderRadius: 5,
          border: `1px solid ${PANEL_BORDER}`, overflow: 'hidden',
        }}>
          <div style={{
            height: '100%', width: `${Math.round(progress * 100)}%`,
            background: COLOR_WORLDMODEL, transition: 'width 300ms ease-out',
          }} />
        </div>
        <div style={{ marginTop: 12, fontSize: 11, color: TEXT_SECONDARY, textAlign: 'center' }}>
          Training prediction network — {Math.round(progress * 100)}%
          <br />
          ~60 min remaining on first run. Subsequent runs load in &lt;2 minutes.
        </div>
      </div>
    </div>
  )
}
