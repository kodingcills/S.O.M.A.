import type { NodeProps } from '@xyflow/react'
import type { SomaNode } from '../../../utils/buildCanvasState'
import { COLOR_WORLDMODEL } from '../../../utils/colors'
import { NodeShell } from './NodeShell'

export function WorldModelNode({ data, selected }: NodeProps<SomaNode>) {
  return (
    <NodeShell color={COLOR_WORLDMODEL} symbol="◇" label="World model" agentId={data.agent_id} status={data.status} timestamp={data.timestamp} selected={selected}>
      <div className="node-title">{data.title}</div>
      <div className="node-outcome"><span>loss</span><strong>{data.fine_tune_loss?.toFixed(4) ?? '—'}</strong></div>
      <div className="node-meta"><span>generation {data.generation}</span></div>
    </NodeShell>
  )
}
