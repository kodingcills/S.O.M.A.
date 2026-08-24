// CapabilityNode — INTERFACE.md "NODE COMPONENTS": `DOF {n} Unlock`
// + 6 tiny circles showing DOF progress (filled ≤ dof_level).
import { NodeProps } from '@xyflow/react'
import { NodeShell } from './NodeShell'
import { COLOR_CAPABILITY, TEXT_SECONDARY, STATUS_COMPLETED } from '../../../utils/colors'
import type { SomaNode } from '../../../utils/buildCanvasState'

export function CapabilityNode({ data }: NodeProps<SomaNode>) {
  const level = data.dof_level ?? 0
  return (
    <NodeShell color={COLOR_CAPABILITY} label={data.label}
               status={data.status} timestamp={data.timestamp}>
      <div style={{ fontWeight: 700 }}>DOF {level || '—'} Unlock</div>
      <div style={{ display: 'flex', gap: 4, marginTop: 3 }}>
        {[1, 2, 3, 4, 5, 6].map(n => (
          <span key={n} style={{
            width: 8, height: 8, borderRadius: '50%', boxSizing: 'border-box',
            background: n <= level ? STATUS_COMPLETED : 'transparent',
            border: `1px solid ${n <= level ? STATUS_COMPLETED : TEXT_SECONDARY}`,
          }} />
        ))}
      </div>
    </NodeShell>
  )
}
