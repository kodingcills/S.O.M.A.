import { useMemo, useState } from 'react'
import type { EventLogEntry } from '../../types'
import { REGION_LABELS } from '../../utils/colors'

export interface AgentFeedProps {
  events: EventLogEntry[]
  activeNode: string
}

function eventTitle(event: EventLogEntry): string {
  switch (event.event_type) {
    case 'agent_spawned': return 'Agent spawned'
    case 'agent_step': return 'Exploration step'
    case 'agent_completed': return 'Evaluation complete'
    case 'agent_failed': return 'Execution failed'
    case 'capability_unlocked': return 'Capability unlocked'
    default: return event.event_type.replace(/_/g, ' ')
  }
}

function eventSummary(event: EventLogEntry): string {
  const region = event.region ? REGION_LABELS[event.region] ?? event.region : null
  switch (event.event_type) {
    case 'agent_spawned':
      return event.error_before !== null
        ? `Evaluating ${region ?? 'the selected region'} at prediction error ${event.error_before.toFixed(3)}.`
        : `Execution initialized${region ? ` for ${region}` : ''}.`
    case 'agent_step': {
      const reasoning = event.payload.reasoning
      if (typeof reasoning === 'string' && reasoning.trim()) return reasoning.trim()
      const step = event.payload.step
      return `Simulation step ${typeof step === 'number' ? step : '—'} completed.`
    }
    case 'agent_completed': {
      const samples = typeof event.payload.samples_collected === 'number' ? event.payload.samples_collected : null
      if (event.error_before !== null && event.error_after !== null) {
        return `Prediction error changed ${event.error_before.toFixed(3)} → ${event.error_after.toFixed(3)}${samples !== null ? ` after ${samples} samples` : ''}.`
      }
      return samples !== null ? `Adaptation completed with ${samples} samples.` : 'Execution completed.'
    }
    case 'agent_failed':
      return String(event.payload.error ?? 'No failure message was recorded.')
    case 'capability_unlocked':
      return `DOF ${String(event.payload.previous_dof ?? '—')} → ${String(event.payload.new_dof ?? '—')}.`
    default:
      return 'Event recorded.'
  }
}

function markerClass(event: EventLogEntry): string {
  if (event.event_type === 'agent_failed') return 'failed'
  if (event.event_type === 'agent_completed' || event.event_type === 'capability_unlocked') return 'completed'
  if (event.event_type === 'agent_spawned' || event.event_type === 'agent_step') return 'active'
  return ''
}

export function AgentTrace({ events, activeNode }: AgentFeedProps) {
  const [expanded, setExpanded] = useState<number | null>(null)
  const relevant = useMemo(() => events
    .filter(event => event.agent_id === activeNode || event.parent_id === activeNode)
    .slice(-120), [events, activeNode])

  if (relevant.length === 0) {
    return <div className="empty-state" style={{ minHeight: 90 }}><span>No trace events recorded for this node.</span></div>
  }

  return (
    <div className="trace">
      {relevant.map((event, index) => {
        const previous = relevant[index - 1]
        const elapsedMs = previous ? Math.max(0, (event.timestamp - previous.timestamp) * 1000) : 0
        const duration = elapsedMs >= 1000 ? `+${(elapsedMs / 1000).toFixed(2)} s` : elapsedMs > 0 ? `+${Math.round(elapsedMs)} ms` : ''
        const isExpanded = expanded === event.id
        return (
          <div className="trace-row" key={event.id}>
            <i className={`trace-marker ${markerClass(event)}`} aria-hidden="true" />
            <button className="trace-button" onClick={() => setExpanded(isExpanded ? null : event.id)} aria-expanded={isExpanded}>
              <time className="trace-time">{new Date(event.timestamp * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</time>
              <span className="trace-title">{eventTitle(event)}</span>
              <span className="trace-meta">{duration}</span>
            </button>
            <p className="trace-summary">{eventSummary(event)}</p>
            {isExpanded && (
              <div className="trace-detail">
                <strong>Recorded event</strong>
                <pre>{JSON.stringify({
                  event_type: event.event_type,
                  agent_id: event.agent_id,
                  region: event.region,
                  error_before: event.error_before,
                  error_after: event.error_after,
                  payload: event.payload,
                }, null, 2)}</pre>
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

// Compatibility export for existing imports.
export const AgentFeed = AgentTrace
