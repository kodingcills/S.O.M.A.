import { useEffect, useMemo, useRef, useState } from 'react'
import type { BeliefStateSnapshot, EventLogEntry } from '../../types'
import { errorToColor, REGION_LABELS } from '../../utils/colors'
import { pickQualifyingCompletion, regionToCells } from '../../utils/regionToCells'

export interface WorldModelPanelProps {
  belief?: BeliefStateSnapshot | null
  events: EventLogEntry[]
  showErrors?: boolean
  eePos?: [number, number]
  targetPos?: [number, number]
}

const EMPTY_MAP = Array.from({ length: 16 }, () => Array<number>(16).fill(0))

export function WorldModelPanel({ belief, events, eePos, targetPos }: WorldModelPanelProps) {
  const errorMap = belief?.error_map ?? belief?.prediction_error_map ?? EMPTY_MAP
  const globalError = belief?.global_mean_error ?? 0
  const dof = belief?.active_dof ?? 1
  const [highlighted, setHighlighted] = useState<Set<string>>(new Set())
  const [annotation, setAnnotation] = useState(false)
  const lastCompletion = useRef<number | null>(null)

  useEffect(() => {
    const completion = pickQualifyingCompletion(events)
    if (!completion?.region || completion.id === lastCompletion.current) return
    lastCompletion.current = completion.id
    setHighlighted(new Set(regionToCells(completion.region, eePos, targetPos).map(cell => `${cell.row}:${cell.col}`)))
    setAnnotation(true)
    const id = window.setTimeout(() => {
      setHighlighted(new Set())
      setAnnotation(false)
    }, 750)
    return () => window.clearTimeout(id)
  }, [events, eePos, targetPos])

  const ranked = useMemo(() => Object.entries(belief?.regional_errors ?? {})
    .map(([key, error]) => ({ key, label: REGION_LABELS[key] ?? key, error }))
    .sort((a, b) => b.error - a.error), [belief?.regional_errors])
  const latestDelta = useMemo(() => [...events].reverse().find(event =>
    event.event_type === 'agent_completed' && event.error_before !== null && event.error_after !== null), [events])

  return (
    <div className="model-panel">
      <div className="model-panel-title">
        <strong>Spatial prediction error</strong>
        <span>WM <span className="mono">{belief?.world_model_version ?? belief?.version ?? '—'}</span> · error <span className="mono">{globalError.toFixed(3)}</span></span>
      </div>

      {latestDelta && (
        <div className="model-latest-delta">
          <span>Latest measured change</span>
          <strong className="mono">{latestDelta.error_before?.toFixed(3)} → {latestDelta.error_after?.toFixed(3)}</strong>
          <small>{latestDelta.region ? REGION_LABELS[latestDelta.region] ?? latestDelta.region : 'region unavailable'}</small>
        </div>
      )}

      <div style={{ position: 'relative' }}>
        <div className="uncertainty-map" role="img" aria-label="16 by 16 spatial prediction error map">
          {errorMap.flatMap((row, rowIndex) => row.map((error, colIndex) => (
            <span
              key={`${rowIndex}:${colIndex}`}
              className={`uncertainty-cell ${highlighted.has(`${rowIndex}:${colIndex}`) ? 'improved' : ''}`}
              style={{ background: errorToColor(error) }}
              title={`Row ${rowIndex}, column ${colIndex}: error ${error.toFixed(4)}`}
            />
          )))}
        </div>
        {annotation && <div className="model-update-annotation">Model updated · region improved</div>}
      </div>

      <div className="model-legend"><span>Lower uncertainty</span><i /><span>Higher uncertainty</span></div>

      <div className="region-list" aria-label="Regional prediction error ranking">
        {ranked.length === 0 ? (
          <div className="empty-state" style={{ minHeight: 80 }}>Regional errors unavailable.</div>
        ) : ranked.map(region => (
          <div className="region-rank" key={region.key}>
            <span>{region.label}</span>
            <span className="region-bar"><span style={{ width: `${Math.min(100, region.error * 100)}%` }} /></span>
            <span className="mono">{region.error.toFixed(3)}</span>
          </div>
        ))}
      </div>

      <div className="dof-row" aria-label={`Active degrees of freedom ${dof} of 6`}>
        <span>DOF</span>
        {[1, 2, 3, 4, 5, 6].map(index => <i key={index} className={`dof-mark ${index <= dof ? 'active' : ''}`} />)}
        <span className="mono">{dof}/6</span>
      </div>
    </div>
  )
}
