import type { NodeProps } from '@xyflow/react'
import type { SomaNode } from '../../../utils/buildCanvasState'
import { COLOR_SIMULATION } from '../../../utils/colors'
import { NodeShell } from './NodeShell'

export function SimulationNode({ data, selected }: NodeProps<SomaNode>) {
  return (
    <NodeShell color={COLOR_SIMULATION} symbol="□" label="Evaluation" agentId={data.agent_id} status={data.status} timestamp={data.timestamp} selected={selected}>
      <div className="node-title">{data.title}</div>
      <div className="node-outcome"><span>simulation</span><strong>{data.sim_id ?? 'primary'}</strong></div>
      <div className="node-meta"><span>generation {data.generation}</span></div>
    </NodeShell>
  )
}
