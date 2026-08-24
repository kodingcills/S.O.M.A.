// RootNode — INTERFACE.md "NODE COMPONENTS": SEED label, DOF level + status.
import { NodeProps } from '@xyflow/react'
import { NodeShell } from './NodeShell'
import { COLOR_ROOT, TEXT_SECONDARY } from '../../../utils/colors'
import type { SomaNode } from '../../../utils/buildCanvasState'

export function RootNode({ data }: NodeProps<SomaNode>) {
  const statusText =
    data.status === 'failed' ? 'FAILED' : 'RUNNING'
  return (
    <NodeShell color={COLOR_ROOT} label={data.label}
               status={data.status} timestamp={data.timestamp}>
      <div style={{ fontWeight: 700 }}>DOF {data.dof_level ?? 1}</div>
      <div style={{ fontSize: 9, letterSpacing: '0.08em', color: TEXT_SECONDARY }}>
        {statusText}
      </div>
    </NodeShell>
  )
}
