import type { NodeProps } from '@xyflow/react'
import type { SomaNode } from '../../../utils/buildCanvasState'
import { COLOR_CAPABILITY } from '../../../utils/colors'
import { NodeShell } from './NodeShell'

export function CapabilityNode({ data, selected }: NodeProps<SomaNode>) {
  const dof = data.dof_level
  return (
    <NodeShell color={COLOR_CAPABILITY} symbol="◆" label="Capability" agentId={data.agent_id} status={data.status} timestamp={data.timestamp} selected={selected}>
      <div className="node-title">{data.title}</div>
      <div className="node-outcome capability-outcome">
        <span>capability</span>
        <strong>{dof ? `DOF ${Math.max(1, dof - 1)} → ${dof}` : 'DOF —'}</strong>
      </div>
      <div className="node-meta"><span>generation {data.generation}</span><span>{data.samples_collected !== null ? `${data.samples_collected} samples` : 'adaptation pending'}</span></div>
    </NodeShell>
  )
}
