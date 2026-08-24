// ExplorationNode — INTERFACE.md "NODE COMPONENTS": region label,
// error line `before → after ✓` when completed else `before → ···`,
// header dot colored by errorToColor(error_before).
import { NodeProps } from '@xyflow/react'
import { NodeShell } from './NodeShell'
import { COLOR_EXPLORATION, TEXT_SECONDARY, STATUS_COMPLETED, REGION_LABELS, errorToColor } from '../../../utils/colors'
import type { SomaNode } from '../../../utils/buildCanvasState'

export function ExplorationNode({ data }: NodeProps<SomaNode>) {
  const regionLabel = data.region
    ? (REGION_LABELS[data.region] ?? data.region)
    : '—'
  const before = data.error_before?.toFixed(2) ?? '·'
  const after  = data.error_after?.toFixed(2)

  return (
    <NodeShell color={COLOR_EXPLORATION} label={data.label}
               status={data.status} timestamp={data.timestamp}
               dotColor={errorToColor(data.error_before ?? 0)}>
      <div style={{ fontSize: 9, letterSpacing: '0.06em',
                    color: TEXT_SECONDARY, whiteSpace: 'nowrap',
                    overflow: 'hidden', textOverflow: 'ellipsis' }}>
        {regionLabel}
      </div>
      <div style={{ whiteSpace: 'nowrap' }}>
        {before} → {after ?? '···'}
        {data.status === 'completed' && (
          <span style={{ color: STATUS_COMPLETED, marginLeft: 4 }}>✓</span>
        )}
      </div>
    </NodeShell>
  )
}
