import type { DetailResponse, EventLogEntry } from '../../types'
import { REGION_LABELS } from '../../utils/colors'
import { WorldModelPanel } from '../DetailView/WorldModelPanel'

interface ModelsViewProps {
  events: EventLogEntry[]
  detail: DetailResponse | null
  onSelectNode: (nodeId: string) => void
}

export function ModelsView({ events, detail, onSelectNode }: ModelsViewProps) {
  const belief = detail?.belief_state ?? null
  const updates = events.filter(event => event.event_type === 'world_model_updated')
  const evidence = events.filter(event => event.event_type === 'agent_completed' && event.error_after !== null)

  return (
    <div className="panel-page">
      <div className="page-heading"><h1>Models</h1><p>Current belief state, uncertainty, and recorded update evidence.</p></div>
      <div className="summary-list">
        <div className="summary-cell"><div className="summary-label">World-model version</div><div className="summary-value mono">{belief?.world_model_version ?? belief?.version ?? '—'}</div></div>
        <div className="summary-cell"><div className="summary-label">Global error</div><div className="summary-value mono">{belief?.global_mean_error.toFixed(3) ?? '—'}</div></div>
        <div className="summary-cell"><div className="summary-label">Active DOF</div><div className="summary-value mono">{belief ? `${belief.active_dof} / 6` : '—'}</div></div>
        <div className="summary-cell"><div className="summary-label">Recorded updates</div><div className="summary-value mono">{updates.length}</div></div>
      </div>

      <div className="analysis-grid">
        <section className="analysis-panel">
          <WorldModelPanel
            belief={belief}
            events={events}
          />
        </section>
        <section className="analysis-panel">
          <div className="analysis-panel-header"><h2>Model update log</h2><span>backend update events</span></div>
          {updates.length === 0 ? <div className="empty-state"><div><strong>No update events recorded.</strong>The current model state remains inspectable above.</div></div> : (
            <table className="data-table"><thead><tr><th>Version</th><th>Loss</th><th>Time</th></tr></thead><tbody>
              {updates.slice(-20).reverse().map(update => (
                <tr key={update.id}>
                  <td className="mono">{typeof update.payload.payload_version === 'number' ? update.payload.payload_version : '—'}</td>
                  <td className="mono">{typeof update.payload.payload_loss === 'number' ? update.payload.payload_loss.toFixed(4) : '—'}</td>
                  <td className="mono">{new Date(update.timestamp * 1000).toLocaleTimeString()}</td>
                </tr>
              ))}
            </tbody></table>
          )}
        </section>
      </div>

      <section className="analysis-panel" style={{ maxWidth: 1180, marginTop: 16 }}>
        <div className="analysis-panel-header"><h2>Learning evidence</h2><span>measured before → after changes</span></div>
        {evidence.length === 0 ? <div className="empty-state">No measured model deltas are available yet.</div> : (
          <table className="data-table"><thead><tr><th>Node</th><th>Region</th><th>Before</th><th>After</th><th>Change</th></tr></thead><tbody>
            {evidence.slice(-20).reverse().map(event => {
              const change = event.error_before !== null && event.error_after !== null && event.error_before > 0
                ? ((event.error_before - event.error_after) / event.error_before) * 100
                : null
              return <tr key={event.id}>
                <td><button onClick={() => event.agent_id && onSelectNode(event.agent_id)}>{event.agent_id ?? '—'}</button></td>
                <td>{event.region ? REGION_LABELS[event.region] ?? event.region : '—'}</td>
                <td className="mono">{event.error_before?.toFixed(3) ?? '—'}</td>
                <td className="mono">{event.error_after?.toFixed(3) ?? '—'}</td>
                <td className="mono">{change !== null ? `↓ ${change.toFixed(1)}%` : '—'}</td>
              </tr>
            })}
          </tbody></table>
        )}
      </section>
    </div>
  )
}
