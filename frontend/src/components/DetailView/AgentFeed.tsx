import type { EventLogEntry } from '../../types'
import {
  STATUS_COMPLETED,
  STATUS_FAILED,
  STATUS_SPAWNING,
  TEXT_ERROR,
  TEXT_SECONDARY,
} from '../../utils/colors'

export interface AgentFeedProps {
  events: EventLogEntry[]
  activeNode: string
}

const MONO = "'JetBrains Mono', ui-monospace, Menlo, monospace"
const MAX_ENTRIES = 500

function timeOf(ts: number): string {
  return new Date(ts * 1000).toLocaleTimeString('en-GB', { hour12: false })
}

interface Entry {
  id: number
  time: string
  label: string
  color: string
  body: string
  reasoning?: string
}

function buildEntry(e: EventLogEntry): Entry | null {
  const time = timeOf(e.timestamp)
  const who = e.agent_id ?? ''
  const delta =
    e.error_before != null && e.error_after != null
      ? e.error_before - e.error_after
      : null

  switch (e.event_type) {
    case 'agent_spawned':
      return {
        id: e.id, time, label: 'SPAWN', color: STATUS_SPAWNING,
        body: `${who} region:${REGION_NAME(e.region)} | err:${(e.error_before ?? 0).toFixed(2)}`,
      }
    case 'agent_completed':
      return {
        id: e.id, time, label: 'RESULT',
        color: delta != null && delta > 0 ? STATUS_COMPLETED : STATUS_FAILED,
        body: `${who} ${(e.error_before ?? 0).toFixed(2)} → ${(e.error_after ?? 0).toFixed(2)} ${delta != null && delta > 0 ? '✓' : '·'}`,
      }
    case 'agent_failed':
      return {
        id: e.id, time, label: 'FAILED', color: TEXT_ERROR,
        body: `${who} ${String(e.payload?.error ?? 'unknown')}`,
      }
    case 'agent_step': {
      const step = typeof e.payload?.step === 'number' ? e.payload.step : '?'
      const reasoning =
        typeof e.payload?.reasoning === 'string' && e.payload.reasoning.trim()
          ? e.payload.reasoning.trim()
          : undefined
      return {
        id: e.id, time, label: 'STEP', color: TEXT_SECONDARY,
        body: `${who} step:${step}`, reasoning,
      }
    }
    case 'capability_unlocked':
      return {
        id: e.id, time, label: 'UNLOCK', color: '#AF52DE',
        body: `${who} DOF ${String(e.payload?.new_dof ?? '?')}`,
      }
    default:
      return null
  }
}

function REGION_NAME(region: string | null): string {
  const known: Record<string, string> = {
    upper_left: 'Upper Left', upper_right: 'Upper Right',
    lower_left: 'Lower Left', lower_right: 'Lower Right',
    tool_tissue_boundary: 'Tool Zone', surgical_target_vicinity: 'Target Zone',
  }
  return (region && known[region]) || region || '—'
}

export function AgentFeed({ events, activeNode }: AgentFeedProps) {
  const relevant = events.filter(
    (e) => e.agent_id === activeNode || e.parent_id === activeNode,
  )
  const entries = relevant
    .slice(-MAX_ENTRIES)
    .reverse()
    .map(buildEntry)
    .filter((e): e is Entry => e !== null)

  return (
    <div style={{
      height: '100%', overflowY: 'auto', padding: '8px 10px',
      boxSizing: 'border-box', fontFamily: MONO, fontSize: 10,
    }}>
      <div style={{ color: TEXT_SECONDARY, letterSpacing: '0.1em', marginBottom: 4 }}>
        AGENT FEED · {activeNode || '—'}
      </div>
      {entries.length === 0 && (
        <div style={{ color: TEXT_SECONDARY }}>no events for this node yet…</div>
      )}
      {entries.map((entry) => (
        <div key={entry.id}>
          <div style={{ display: 'flex', gap: 10, lineHeight: '20px', whiteSpace: 'nowrap' }}>
            <span style={{ color: TEXT_SECONDARY }}>{entry.time}</span>
            <span style={{ color: entry.color, width: 52 }}>{entry.label}</span>
            <span style={{ color: TEXT_SECONDARY, overflow: 'hidden', textOverflow: 'ellipsis' }}>
              {entry.body}
            </span>
          </div>
          {entry.reasoning && (
            <div style={{
              marginLeft: 96, color: '#5AC8FA', fontStyle: 'italic',
              overflow: 'hidden', display: '-webkit-box',
              WebkitLineClamp: 2, WebkitBoxOrient: 'vertical',
              lineHeight: '16px', paddingBottom: 2,
            }}>
              “{entry.reasoning}”
            </div>
          )}
        </div>
      ))}
      {relevant.length > MAX_ENTRIES && (
        <div style={{ color: TEXT_SECONDARY, paddingTop: 4 }}>
          +{relevant.length - MAX_ENTRIES} earlier events hidden
        </div>
      )}
    </div>
  )
}
