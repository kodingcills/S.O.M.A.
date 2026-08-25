import type { NodeProps } from '@xyflow/react'
import type { SomaNode } from '../../../utils/buildCanvasState'
import { COLOR_ROOT } from '../../../utils/colors'
import { NodeShell } from './NodeShell'

export function RootNode({ data, selected }: NodeProps<SomaNode>) {
  return (
    <NodeShell color={COLOR_ROOT} symbol="○" label="Seed" agentId="root" status={data.status} timestamp={data.timestamp} selected={selected}>
      <div className="node-title">Initial system state</div>
      <div className="node-outcome"><span>capability</span><strong>DOF {data.dof_level ?? 1}</strong></div>
      <div className="node-meta"><span>learning origin</span><span>primary anatomy</span></div>
    </NodeShell>
  )
}
