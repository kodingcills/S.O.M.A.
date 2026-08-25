import type { NodeProps } from '@xyflow/react'
import type { SomaNode } from '../../../utils/buildCanvasState'
import { COLOR_EXPLORATION } from '../../../utils/colors'
import { NodeShell } from './NodeShell'

export function ExplorationNode({ data, selected }: NodeProps<SomaNode>) {
  const before = data.error_before
  const after = data.error_after
  const reduction = before != null && after != null && before > 0
    ? Math.round(((before - after) / before) * 100)
    : null

  return (
    <NodeShell color={COLOR_EXPLORATION} symbol="●" label="Exploration" agentId={data.agent_id} status={data.status} timestamp={data.timestamp} selected={selected}>
      <div className="node-title">{data.title}</div>
      {data.status === 'failed' ? (
        <div className="node-failure">{data.error_message ?? 'Execution failed'}</div>
      ) : (
        <div className="node-outcome">
          <span>error</span>
          <strong>{before?.toFixed(3) ?? '—'} <em>→</em> {after?.toFixed(3) ?? '…'}</strong>
          {reduction !== null && <small>↓{reduction}%</small>}
        </div>
      )}
      <div className="node-meta">
        <span>generation {data.generation}</span>
        <span>{data.samples_collected !== null ? `${data.samples_collected} samples` : 'collecting evidence'}</span>
      </div>
    </NodeShell>
  )
}
