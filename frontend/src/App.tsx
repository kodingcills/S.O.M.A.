import { useCallback, useState } from 'react'
import '@xyflow/react/dist/style.css'
import { CanvasView } from './components/Canvas'
import { CompareView } from './components/CompareView'
import { DetailView } from './components/DetailView'
import { ModelsView } from './components/ModelsView'
import { RunsView } from './components/RunsView'
import { SimulationWorkspace } from './components/SimulationWorkspace'
import { useDetail } from './hooks/useDetail'
import { useEventLog } from './hooks/useEventLog'
import { useHealth } from './hooks/useHealth'
import type { EventLogEntry, HealthResponse } from './types'

export type View = 'graph' | 'runs' | 'compare' | 'models'

const NAV: { id: View; label: string; icon: string }[] = [
  { id: 'graph', label: 'Graph', icon: '⌘' },
  { id: 'runs', label: 'Runs', icon: '▤' },
  { id: 'compare', label: 'Compare', icon: '⇄' },
  { id: 'models', label: 'Models', icon: '◇' },
]

function initialView(): View {
  const requested = new URLSearchParams(window.location.search).get('view')
  if (requested === 'runs' || requested === 'compare' || requested === 'models') return requested
  return 'graph'
}

export default function App() {
  const events = useEventLog()
  const health = useHealth()
  const [view, setView] = useState<View>(initialView)
  const [activeNode, setActiveNode] = useState<string | null>(null)
  const [simulationOpen, setSimulationOpen] = useState(false)

  const detail = useDetail(activeNode !== null || view === 'models' ? 'primary' : null)

  const selectNode = useCallback((nodeId: string) => setActiveNode(nodeId), [])
  const jumpToNode = useCallback((nodeId: string) => {
    setActiveNode(nodeId)
    setView('graph')
  }, [])

  const generation = 1 + events.filter(e => e.event_type === 'capability_unlocked').length
  const firstSnapshot = events.find(e => e.event_type === 'belief_snapshot')
  const firstError = typeof firstSnapshot?.payload.global_mean_error === 'number'
    ? firstSnapshot.payload.global_mean_error
    : null
  const improvement = firstError && health
    ? ((firstError - health.global_error) / firstError) * 100
    : null

  return (
    <div className="app-shell">
      <TopBar health={health} />

      <div className="app-body">
        <Sidebar view={view} onChange={(next) => { setView(next); setSimulationOpen(false) }} />

        <main className="workspace" aria-label="SOMA workspace">
          <section className="view-layer" style={{ display: view === 'graph' ? 'block' : 'none' }} aria-hidden={view !== 'graph'}>
            <div className={`graph-layout ${activeNode ? 'has-selection' : ''}`}>
              <div className="graph-surface">
                <CanvasView events={events} selectedNodeId={activeNode} onNodeClick={selectNode} />
                {health?.status === 'training' && <TrainingNotice health={health} />}
              </div>
              <aside className="inspector-shell" aria-label="Node inspector">
                <DetailView
                  events={events}
                  detail={detail}
                  activeNode={activeNode}
                  onClose={() => setActiveNode(null)}
                  onOpenSimulation={() => setSimulationOpen(true)}
                  onOpenModels={() => setView('models')}
                />
              </aside>
            </div>
          </section>

          <section className="view-layer" style={{ display: view === 'runs' ? 'block' : 'none' }} aria-hidden={view !== 'runs'}>
            <RunsView events={events} health={health} onSelectNode={jumpToNode} />
          </section>

          <section className="view-layer" style={{ display: view === 'compare' ? 'block' : 'none' }} aria-hidden={view !== 'compare'}>
            <CompareView events={events} health={health} active={view === 'compare'} />
          </section>

          <section className="view-layer" style={{ display: view === 'models' ? 'block' : 'none' }} aria-hidden={view !== 'models'}>
            <ModelsView events={events} detail={detail} onSelectNode={jumpToNode} />
          </section>

          {simulationOpen && activeNode && (
            <SimulationWorkspace nodeId={activeNode} events={events} detail={detail} onClose={() => setSimulationOpen(false)} />
          )}
        </main>
      </div>

      <footer className="statusbar" aria-label="Session status">
        <span>generation <span className="mono">{generation}</span></span>
        <span><span className="mono">{events.length}</span> events</span>
        <span><span className="mono">{health?.active_agents ?? 0}</span> active</span>
        <span className="right">
          model error {improvement !== null && Number.isFinite(improvement)
            ? <><span aria-hidden="true">↓</span> <span className="mono">{Math.max(0, improvement).toFixed(1)}%</span></>
            : <span className="mono">—</span>}
        </span>
      </footer>
    </div>
  )
}

function TopBar({ health }: { health: HealthResponse | null }) {
  const status = health?.status ?? 'starting'
  const statusText = status === 'ok' ? 'Running' : status === 'training' ? 'Training' : 'Connecting'
  return (
    <header className="topbar">
      <div className="brand">
        <span className="brand-mark">SOMA</span>
        <span className="experiment">/ current runtime session</span>
      </div>
      <div className="topbar-meta" aria-label="Current model state">
        <span>World model <strong className="mono">{String(health?.world_model_version ?? '—').padStart(2, '0')}</strong></span>
        <span>Error <strong className="mono">{health ? health.global_error.toFixed(3) : '—'}</strong></span>
        <span><strong className="mono">{health?.active_agents ?? '—'}</strong> agents active</span>
        <span className="status-inline"><i className={`status-dot ${status === 'ok' ? 'running' : status}`} aria-hidden="true" />{statusText}</span>
      </div>
    </header>
  )
}

function Sidebar({ view, onChange }: { view: View; onChange: (view: View) => void }) {
  return (
    <nav className="sidebar" aria-label="Primary navigation">
      <div className="nav-label">Workspace</div>
      {NAV.map(item => (
        <button
          key={item.id}
          className={`nav-item ${view === item.id ? 'selected' : ''}`}
          onClick={() => onChange(item.id)}
          aria-current={view === item.id ? 'page' : undefined}
          title={item.label}
        >
          <span className="nav-icon" aria-hidden="true">{item.icon}</span>
          <span className="nav-text">{item.label}</span>
        </button>
      ))}
      <div className="sidebar-context">
        Live event log<br />
        <span className="mono">primary / comparison</span>
      </div>
    </nav>
  )
}

function TrainingNotice({ health }: { health: HealthResponse }) {
  const progress = Math.round((health.training_progress ?? 0) * 100)
  return (
    <div className="training-notice" role="status">
      Initial model training in progress · <span className="mono">{progress}%</span>
      <div className="training-progress" aria-hidden="true"><span style={{ width: `${progress}%` }} /></div>
    </div>
  )
}

export function nodeEvents(events: EventLogEntry[], nodeId: string): EventLogEntry[] {
  return events.filter(event => event.agent_id === nodeId || event.parent_id === nodeId)
}
