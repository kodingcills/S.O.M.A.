// WorldModelNode — minimal version for Task 2.2 (label + version text).
// 32×32 heatmap thumbnail + error value lands in a later task.
import { NodeProps } from '@xyflow/react'
import { NodeShell } from './NodeShell'
import { COLOR_WORLDMODEL, TEXT_SECONDARY } from '../../../utils/colors'
import type { SomaNode } from '../../../utils/buildCanvasState'

export function WorldModelNode({ data }: NodeProps<SomaNode>) {
  return (
    <NodeShell color={COLOR_WORLDMODEL} label={data.label}
               status={data.status} timestamp={data.timestamp}>
      <div style={{ fontSize: 9, color: TEXT_SECONDARY,
                    whiteSpace: 'nowrap', overflow: 'hidden',
                    textOverflow: 'ellipsis' }}>
        {data.agent_id}
      </div>
      <div style={{ fontSize: 9, color: TEXT_SECONDARY }}>thumbnail — later task</div>
    </NodeShell>
  )
}
