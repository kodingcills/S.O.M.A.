import { useDetail } from '../../hooks/useDetail'
import type { EventLogEntry, HealthResponse, SimulationState } from '../../types'
import { CraftaxMap } from '../DetailView/CraftaxMap'

interface CompareViewProps {
  events: EventLogEntry[]
  health: HealthResponse | null
  active: boolean
}

function StreamSide({ sim, title, mechanism }: { readonly sim: SimulationState | null; readonly title: string; readonly mechanism: string }) {
  return (
    <div className="stream-side">
      <div className="stream-header"><span className="status-dot running" aria-hidden="true" />{title}<span>{mechanism}</span></div>
      <div className="stream-frame">
        {sim ? <CraftaxMap tokenGrid={sim.token_2d} direction={sim.direction} /> : <span>Craftax observation unavailable.</span>}
      </div>
      {sim && <div className="stream-stats mono"><span>HP {sim.stats.health.toFixed(1)}</span><span>ACH {sim.achievements_count}</span><span>STEP {sim.step}</span></div>}
    </div>
  )
}

function signed(value: number, digits = 0): string {
  const rendered = value.toFixed(digits)
  return value > 0 ? `+${rendered}` : rendered
}

export function CompareView({ events, health, active }: CompareViewProps) {
  const primary = useDetail(active ? 'primary' : null)
  const comparison = useDetail(active ? 'comparison' : null)
  const pSim = primary?.simulation_state ?? null
  const cSim = comparison?.simulation_state ?? null

  const measurements = [
    { label: 'Achievements', soma: pSim?.achievements_count ?? '—', baseline: cSim?.achievements_count ?? '—', delta: pSim && cSim ? signed(pSim.achievements_count - cSim.achievements_count) : '—' },
    { label: 'Health', soma: pSim ? pSim.stats.health.toFixed(1) : '—', baseline: cSim ? cSim.stats.health.toFixed(1) : '—', delta: pSim && cSim ? signed(pSim.stats.health - cSim.stats.health, 1) : '—' },
    { label: 'Reward', soma: pSim ? pSim.reward.toFixed(3) : '—', baseline: cSim ? cSim.reward.toFixed(3) : '—', delta: pSim && cSim ? signed(pSim.reward - cSim.reward, 3) : '—' },
    { label: 'Steps', soma: pSim?.step ?? '—', baseline: cSim?.step ?? '—', delta: pSim && cSim ? signed(pSim.step - cSim.step) : '—' },
    { label: 'Model error', soma: health ? health.global_error.toFixed(3) : '—', baseline: 'not applicable', delta: '—' },
  ]

  const timelineEvents = events.filter(event => event.event_type === 'simulation_step').slice(-400)
  const timelineMarkers = timelineEvents.filter((_, index) => index === 0 || index === timelineEvents.length - 1 || index % 10 === 0)
  const timelineStart = timelineEvents[0]?.timestamp ?? 0
  const timelineEnd = timelineEvents[timelineEvents.length - 1]?.timestamp ?? timelineStart
  const timelineSpan = Math.max(0.001, timelineEnd - timelineStart)

  return (
    <div className="compare-layout">
      <div className="page-heading"><h1>Compare runs</h1><p>Same Craftax world; different decision mechanisms.</p></div>

      <table className="compare-metrics">
        <thead><tr><th>Measurement</th><th>SOMA / primary</th><th>Reactive / comparison</th><th>Observed Δ</th></tr></thead>
        <tbody>{measurements.map(row => (
          <tr key={row.label}>
            <td style={{ fontFamily: 'inherit', color: 'var(--text)' }}>{row.label}</td>
            <td>{row.soma}</td><td>{row.baseline}</td>
            <td className={row.delta !== '—' && !String(row.delta).startsWith('-') ? 'delta-positive' : ''}>{row.delta}</td>
          </tr>
        ))}</tbody>
      </table>

      <div className="analysis-panel-header" style={{ maxWidth: 880, border: '1px solid var(--border)', borderBottom: 0, borderRadius: '8px 8px 0 0', background: 'var(--panel)' }}>
        <h2>Aligned live observation</h2><span>primary and comparison environments</span>
      </div>
      <div className="streams" style={{ maxWidth: 1180 }}>
        <StreamSide sim={pSim} title="SOMA" mechanism="MPC + world model" />
        <StreamSide sim={cSim} title="Reactive baseline" mechanism="greedy · no world model" />
      </div>

      <section className="timeline" style={{ maxWidth: 1180 }} aria-label="Shared run timeline">
        <div className="analysis-panel-header" style={{ padding: 0, border: 0 }}><h2>Shared event timeline</h2><span>{timelineEvents.length} recent simulation steps</span></div>
        <div className="timeline-track">
          {timelineMarkers.map(event => (
            <i
              key={event.id}
              className="timeline-marker"
              style={{ left: `${((event.timestamp - timelineStart) / timelineSpan) * 100}%` }}
              title={`${event.sim_id ?? 'simulation'} step at ${new Date(event.timestamp * 1000).toLocaleTimeString()}`}
            />
          ))}
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--tertiary)', fontSize: 9 }}>
          <span className="mono">{timelineEvents.length ? new Date(timelineStart * 1000).toLocaleTimeString() : 'No timeline events'}</span>
          <span>{timelineEvents.length ? `${timelineEvents.length} recorded Craftax steps` : 'No recorded steps in this window'}</span>
          <span className="mono">{timelineEvents.length ? new Date(timelineEnd * 1000).toLocaleTimeString() : '—'}</span>
        </div>
      </section>
    </div>
  )
}
