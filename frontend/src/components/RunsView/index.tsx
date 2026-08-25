import { useMemo } from 'react'
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { EventLogEntry, HealthResponse } from '../../types'
import { COLOR_EXPLORATION, PANEL_BORDER, REGION_LABELS, TEXT_TERTIARY } from '../../utils/colors'

interface RunsViewProps {
  events: EventLogEntry[]
  health: HealthResponse | null
  onSelectNode: (nodeId: string) => void
}

function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return '—'
  const minutes = Math.floor(seconds / 60)
  const remainder = Math.floor(seconds % 60)
  return `${minutes}m ${String(remainder).padStart(2, '0')}s`
}

export function RunsView({ events, health, onSelectNode }: RunsViewProps) {
  const agents = useMemo(() => events.filter(event =>
    event.event_type === 'agent_spawned' || event.event_type === 'agent_completed' || event.event_type === 'agent_failed' || event.event_type === 'capability_unlocked'), [events])
  const completions = events.filter(event => event.event_type === 'agent_completed')
  const failures = events.filter(event => event.event_type === 'agent_failed')
  const unlocks = events.filter(event => event.event_type === 'capability_unlocked')
  const duration = events.length > 1 ? events[events.length - 1].timestamp - events[0].timestamp : 0
  const errorHistory = completions
    .filter(event => event.error_after !== null)
    .map((event, index) => ({ index: index + 1, error: event.error_after, node: event.agent_id }))

  return (
    <div className="panel-page">
      <div className="page-heading"><h1>Runs</h1><p>Analysis of the current runtime session from the persistent event log.</p></div>
      <div className="summary-list">
        <div className="summary-cell"><div className="summary-label">Status</div><div className="summary-value">{health?.status ?? 'connecting'}</div></div>
        <div className="summary-cell"><div className="summary-label">World model</div><div className="summary-value mono">{health?.world_model_version ?? '—'}</div></div>
        <div className="summary-cell"><div className="summary-label">Duration observed</div><div className="summary-value mono">{formatDuration(duration)}</div></div>
        <div className="summary-cell"><div className="summary-label">Current error</div><div className="summary-value mono">{health ? health.global_error.toFixed(3) : '—'}</div></div>
      </div>

      <div className="analysis-grid">
        <section className="analysis-panel">
          <div className="analysis-panel-header"><h2>Measured regional error after exploration</h2><span>{errorHistory.length} measurements</span></div>
          {errorHistory.length > 0 ? (
            <div style={{ width: '100%', height: 260, padding: '14px 12px 6px' }}>
              <ResponsiveContainer>
                <LineChart data={errorHistory} margin={{ top: 8, right: 15, bottom: 4, left: -14 }}>
                  <CartesianGrid stroke={PANEL_BORDER} vertical={false} />
                  <XAxis dataKey="index" tick={{ fill: TEXT_TERTIARY, fontSize: 9 }} tickLine={false} axisLine={{ stroke: PANEL_BORDER }} />
                  <YAxis domain={[0, 'auto']} tick={{ fill: TEXT_TERTIARY, fontSize: 9 }} tickLine={false} axisLine={false} width={42} />
                  <Tooltip contentStyle={{ border: `1px solid ${PANEL_BORDER}`, borderRadius: 6, boxShadow: 'none', fontSize: 10 }} formatter={(value) => [typeof value === 'number' ? value.toFixed(4) : value, 'error']} />
                  <Line type="monotone" dataKey="error" stroke={COLOR_EXPLORATION} strokeWidth={1.5} dot={{ r: 2.5, fill: COLOR_EXPLORATION }} activeDot={{ r: 4 }} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : <div className="empty-state"><div><strong>No post-exploration measurements yet.</strong>Error history appears after an agent completes evaluation.</div></div>}
        </section>

        <section className="analysis-panel">
          <div className="analysis-panel-header"><h2>Session outcomes</h2><span>actual recorded events</span></div>
          <table className="data-table"><tbody>
            <tr><td>Completed agents</td><td className="mono">{completions.length}</td></tr>
            <tr><td>Failed agents</td><td className="mono">{failures.length}</td></tr>
            <tr><td>Capabilities unlocked</td><td className="mono">{unlocks.length}</td></tr>
            <tr><td>Total events</td><td className="mono">{events.length}</td></tr>
          </tbody></table>
        </section>
      </div>

      <section className="analysis-panel" style={{ maxWidth: 1180, marginTop: 16 }}>
        <div className="analysis-panel-header"><h2>Agent events</h2><span>select an event to return to the graph</span></div>
        {agents.length === 0 ? <div className="empty-state"><div><strong>No agent events recorded.</strong>The runtime is waiting for prediction error to initiate exploration.</div></div> : (
          <table className="data-table">
            <thead><tr><th>Time</th><th>Node</th><th>Event</th><th>Region</th><th>Error</th></tr></thead>
            <tbody>{agents.slice(-30).reverse().map(event => (
              <tr key={event.id}>
                <td className="mono">{new Date(event.timestamp * 1000).toLocaleTimeString()}</td>
                <td>{event.agent_id ? <button onClick={() => onSelectNode(event.agent_id!)}>{event.agent_id}</button> : '—'}</td>
                <td>{event.event_type.replace(/_/g, ' ')}</td>
                <td>{event.region ? REGION_LABELS[event.region] ?? event.region : '—'}</td>
                <td className="mono">{event.error_after?.toFixed(3) ?? event.error_before?.toFixed(3) ?? '—'}</td>
              </tr>
            ))}</tbody>
          </table>
        )}
      </section>
    </div>
  )
}
