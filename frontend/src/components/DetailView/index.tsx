import { useMemo } from 'react'
import type { DetailResponse, EventLogEntry, NodeKind, NodeStatus } from '../../types'
import { buildCanvasState } from '../../utils/buildCanvasState'
import { REGION_LABELS } from '../../utils/colors'
import { AgentTrace } from './AgentFeed'
import { CraftaxMap } from './CraftaxMap'

export interface DetailViewProps {
  events: EventLogEntry[]
  detail: DetailResponse | null
  activeNode: string | null
  onClose: () => void
  onOpenSimulation: () => void
  onOpenModels: () => void
}

function kindSymbol(kind: NodeKind): string {
  if (kind === 'exploration') return '●'
  if (kind === 'capability') return '◆'
  if (kind === 'worldmodel') return '◇'
  if (kind === 'simulation') return '□'
  return '○'
}

function statusLabel(status: NodeStatus): string {
  if (status === 'failed') return 'Failed'
  if (status === 'completed') return 'Completed'
  return 'Running'
}

function payloadNumber(event: EventLogEntry | undefined, key: string): number | null {
  const value = event?.payload?.[key]
  return typeof value === 'number' ? value : null
}

function modelVersionAt(events: EventLogEntry[], timestamp: number): number | null {
  const updates = events.filter(event => event.event_type === 'world_model_updated' && event.timestamp <= timestamp)
  const version = updates[updates.length - 1]?.payload.payload_version
  return typeof version === 'number' ? version : null
}

export function DetailView({ events, detail, activeNode, onClose, onOpenSimulation, onOpenModels }: DetailViewProps) {
  const node = useMemo(() => activeNode ? buildCanvasState(events).nodes.find(item => item.id === activeNode) : null, [events, activeNode])

  if (!activeNode || !node) {
    return (
      <div className="inspector-empty">
        <div>
          <div className="empty-symbol" aria-hidden="true">↗</div>
          <strong>Select a learning event</strong><br />
          Inspect its trigger, execution trace, model delta, and supporting evidence.
        </div>
      </div>
    )
  }

  const direct = events.filter(event => event.agent_id === activeNode)
  const spawned = direct.find(event => event.event_type === 'agent_spawned')
  const completed = direct.find(event => event.event_type === 'agent_completed')
  const failed = direct.find(event => event.event_type === 'agent_failed')
  const unlocked = direct.find(event => event.event_type === 'capability_unlocked')
  const before = node.data.error_before
  const after = node.data.error_after
  const change = before !== null && after !== null && before > 0 ? ((before - after) / before) * 100 : null
  const beforeVersion = spawned ? modelVersionAt(events, spawned.timestamp) : null
  const afterVersion = completed ? modelVersionAt(events, completed.timestamp) : beforeVersion
  const samples = payloadNumber(completed, 'samples_collected')
  const loss = payloadNumber(completed, 'fine_tune_loss')
  const region = node.data.region ? REGION_LABELS[node.data.region] ?? node.data.region : null
  const isRoot = activeNode === 'root'

  const summary = isRoot ? {
    trigger: 'SOMA runtime initialized with the primary Craftax world and seed capability state.',
    action: 'No exploration action is associated with the seed state.',
    observation: `${events.length} events are currently present in the persistent session history.`,
    update: 'Downstream nodes record subsequent model and capability changes.',
  } : node.data.kind === 'capability' ? {
    trigger: unlocked
      ? `Global error ${payloadNumber(unlocked, 'trigger_error')?.toFixed(3) ?? '—'} crossed the ${payloadNumber(unlocked, 'threshold')?.toFixed(3) ?? '—'} unlock threshold.`
      : 'A capability unlock was requested by the orchestrator.',
    action: samples !== null ? `Collected ${samples} focused samples for the new degree of freedom.` : 'DOF-focused adaptation is in progress.',
    observation: unlocked ? `Active capability changed from DOF ${String(unlocked.payload.previous_dof ?? '—')} to ${String(unlocked.payload.new_dof ?? '—')}.` : 'No capability observation has been recorded yet.',
    update: loss !== null ? `World-model fine-tuning completed with loss ${loss.toFixed(4)}.` : 'No completed model update is attached yet.',
  } : {
    trigger: before !== null ? `Prediction error ${before.toFixed(3)} was observed in ${region ?? 'the selected region'}.` : 'The orchestrator initiated focused exploration.',
    action: samples !== null ? `The agent collected ${samples} state–action–next-state samples in a dedicated simulation.` : 'Focused data collection is in progress.',
    observation: after !== null ? `Post-update evaluation measured regional error ${after.toFixed(3)}.` : failed ? `Execution stopped: ${String(failed.payload.error ?? 'unknown failure')}.` : 'Awaiting the post-update observation.',
    update: change !== null ? `Regional prediction error changed by ${change.toFixed(1)}%.` : 'No measured model delta has been recorded yet.',
  }

  return (
    <div className="inspector">
      <button className="inspector-close" onClick={onClose} aria-label="Close inspector">×</button>
      <div className="inspector-kicker">
        <span className="node-symbol" aria-hidden="true">{kindSymbol(node.data.kind)}</span>
        <span>{node.data.label}</span>
        <span className="mono">#{activeNode === 'root' ? '000' : activeNode.replace(/^\w+_/, '')}</span>
      </div>
      <h2>{node.data.title}</h2>
      <div className="inspector-subtitle">
        <span className={`status-dot ${node.data.status === 'completed' ? 'running' : node.data.status === 'failed' ? 'failed' : 'training'}`} aria-hidden="true" />
        {statusLabel(node.data.status)}
        {spawned && <span>· {new Date(spawned.timestamp * 1000).toLocaleString()}</span>}
      </div>

      {before !== null && (
        <div className="metric-delta" aria-label={`Prediction error ${before} to ${after ?? 'pending'}`}>
          <div><div className="summary-label">Before</div><span className="value">{before.toFixed(3)}</span></div>
          <span className="arrow" aria-hidden="true">→</span>
          <div><div className="summary-label">After</div><span className="value">{after?.toFixed(3) ?? '…'}</span></div>
          {change !== null && <span className="delta-label">Prediction error ↓ {change.toFixed(1)}%</span>}
        </div>
      )}

      <div className="metric-grid">
        <div><span>World model</span><strong>{beforeVersion ?? '—'} → {afterVersion ?? '—'}</strong></div>
        <div><span>DOF</span><strong>{node.data.dof_level ?? detail?.belief_state.active_dof ?? '—'} / 6</strong></div>
        <div><span>Samples</span><strong>{samples ?? '—'}</strong></div>
        <div><span>Fine-tune loss</span><strong>{loss?.toFixed(4) ?? '—'}</strong></div>
      </div>

      {failed && (
        <section className="section failure-section">
          <h3 className="section-title">Failure</h3>
          <div className="failure-message">{String(failed.payload.error ?? 'Unknown failure')}</div>
          <div className="failure-meta">Stage: execution · Retry information was not recorded by the backend.</div>
        </section>
      )}

      <section className="section">
        <h3 className="section-title">What happened</h3>
        <dl className="event-summary">
          <dt>Trigger</dt><dd>{summary.trigger}</dd>
          <dt>Action</dt><dd>{summary.action}</dd>
          <dt>Observation</dt><dd>{summary.observation}</dd>
          <dt>Model update</dt><dd>{summary.update}</dd>
        </dl>
      </section>

      <section className="section">
        <h3 className="section-title">Trace</h3>
        <AgentTrace events={events} activeNode={activeNode} />
      </section>

      <section className="section">
        <h3 className="section-title">Evidence</h3>
        <div className="evidence-preview">
          {detail ? <CraftaxMap tokenGrid={detail.simulation_state.token_2d} direction={detail.simulation_state.direction} compact /> : null}
          <span className="evidence-preview-label">Craftax observation</span>
        </div>
        <button className="evidence-link" onClick={onOpenSimulation}><span>Simulation</span><span aria-hidden="true">→</span></button>
        <button className="evidence-link" onClick={onOpenModels}><span>World-model state</span><span aria-hidden="true">→</span></button>
        <div className="evidence-link" aria-disabled="true"><span>Recorded trajectory</span><span className="mono">unavailable</span></div>
      </section>
    </div>
  )
}
