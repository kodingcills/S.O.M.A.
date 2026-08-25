import { useState } from 'react'
import type { DetailResponse, EventLogEntry } from '../../types'
import { AgentTrace } from '../DetailView/AgentFeed'
import { SimulationPanel } from '../DetailView/SimulationPanel'

interface SimulationWorkspaceProps {
  nodeId: string
  events: EventLogEntry[]
  detail: DetailResponse | null
  onClose: () => void
}

export function SimulationWorkspace({ nodeId, events, detail, onClose }: SimulationWorkspaceProps) {
  const [showErrors, setShowErrors] = useState(false)
  const [resetState, setResetState] = useState<'idle' | 'pending' | 'done' | 'failed'>('idle')
  const sim = detail?.simulation_state ?? null
  const belief = detail?.belief_state ?? null

  const resetSimulation = async () => {
    setResetState('pending')
    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL}/simulation/primary/reset`, { method: 'POST' })
      setResetState(response.ok ? 'done' : 'failed')
    } catch {
      setResetState('failed')
    }
  }

  return (
    <section className="simulation-workspace" aria-label={`Simulation evidence for ${nodeId}`}>
      <header className="simulation-workspace-head">
        <h2>Simulation evidence</h2>
        <span className="workspace-meta mono">{nodeId}</span>
        <span className="workspace-meta">Live primary environment</span>
        <button onClick={onClose}>Close</button>
      </header>
      <div className="simulation-workspace-body">
        <div className="simulation-main">
          <SimulationPanel simId="primary" sim={sim} showErrors={showErrors} errorMap={belief?.error_map} />
          <div className="simulation-controls" aria-label="Simulation controls">
            <span className="status-inline"><i className="status-dot running" aria-hidden="true" />Live</span>
            <button disabled title="The backend exposes a live MJPEG stream, not recorded playback.">Play / pause</button>
            <button disabled title="Recorded simulation frames are not available in the current API.">Timeline scrubber</button>
            <button onClick={resetSimulation} disabled={resetState === 'pending'}>{resetState === 'pending' ? 'Resetting…' : 'Reset'}</button>
            <label><input type="checkbox" checked={showErrors} onChange={event => setShowErrors(event.target.checked)} /> Prediction-error overlay</label>
            <span style={{ marginLeft: 'auto' }}>{resetState === 'done' ? 'Reset requested' : resetState === 'failed' ? 'Reset failed' : 'Replay unavailable'}</span>
          </div>
        </div>
        <aside className="simulation-aside">
          <h3 className="section-title">Event-aligned trace</h3>
          <p style={{ color: 'var(--muted)', fontSize: 10, lineHeight: 1.45, margin: '0 0 16px' }}>
            Trace events are timestamp-aligned with the live session. Frame seeking requires recorded simulation frames, which the current API does not provide.
          </p>
          <AgentTrace events={events} activeNode={nodeId} />
        </aside>
      </div>
    </section>
  )
}
