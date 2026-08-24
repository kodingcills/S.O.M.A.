// SimulationNode — minimal version for Task 2.2 (label + sim id text).
// 32×32 tissue integrity thumbnail lands in a later task.
import { NodeProps } from '@xyflow/react'
import { NodeShell } from './NodeShell'
import { COLOR_SIMULATION, TEXT_SECONDARY } from '../../../utils/colors'
import type { SomaNode } from '../../../utils/buildCanvasState'

export function SimulationNode({ data }: NodeProps<SomaNode>) {
  const idShort = (data.sim_id ?? data.agent_id).slice(0, 10)
  return (
    <NodeShell color={COLOR_SIMULATION} label={data.label}
               status={data.status} timestamp={data.timestamp}>
      <div style={{ fontSize: 9, color: TEXT_SECONDARY,
                    whiteSpace: 'nowrap', overflow: 'hidden',
                    textOverflow: 'ellipsis' }}>
        #{idShort}
      </div>
      <div style={{ fontSize: 9, color: TEXT_SECONDARY }}>thumbnail — later task</div>
    </NodeShell>
  )
}
