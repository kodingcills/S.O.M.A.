import { useEffect, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { Handle, Position } from '@xyflow/react'
import type { NodeStatus } from '../../../types'
import { STATUS_COMPLETED, STATUS_FAILED, STATUS_SPAWNING } from '../../../utils/colors'

export const NODE_W = 220
export const NODE_H = 124

interface NodeShellProps {
  color: string
  symbol: string
  label: string
  agentId: string
  status: NodeStatus
  timestamp: number
  selected?: boolean
  children: ReactNode
}

export function NodeShell({ color, symbol, label, agentId, status, timestamp, selected = false, children }: NodeShellProps) {
  const [localStatus, setLocalStatus] = useState<NodeStatus>(status)
  const [updated, setUpdated] = useState(false)
  const previous = useRef(status)

  useEffect(() => {
    if (status !== 'spawning') {
      setLocalStatus(status)
      return
    }
    const remaining = Math.max(0, 500 - (Date.now() - timestamp * 1000))
    const id = window.setTimeout(() => setLocalStatus('active'), remaining)
    return () => window.clearTimeout(id)
  }, [status, timestamp])

  useEffect(() => {
    if (previous.current === status || status !== 'completed') {
      previous.current = status
      return
    }
    previous.current = status
    setUpdated(true)
    const id = window.setTimeout(() => setUpdated(false), 700)
    return () => window.clearTimeout(id)
  }, [status])

  const statusText = localStatus === 'spawning' || localStatus === 'active'
    ? 'Running'
    : localStatus === 'failed' ? 'Failed' : 'Completed'
  const statusColor = localStatus === 'failed'
    ? STATUS_FAILED
    : localStatus === 'completed' ? STATUS_COMPLETED : STATUS_SPAWNING

  return (
    <article
      className={`learning-node ${selected ? 'selected' : ''} ${updated ? 'updated' : ''}`}
      style={{ ['--node-accent' as string]: color }}
      aria-label={`${label} ${agentId}, ${statusText}`}
    >
      <Handle type="target" position={Position.Left} className="node-handle" />
      <header className="node-header">
        <span className="node-type"><b style={{ color }} aria-hidden="true">{symbol}</b>{label}</span>
        <span className="node-id">{agentId === 'root' ? '#000' : `#${agentId.replace(/^\w+_/, '')}`}</span>
      </header>
      <div className="node-content">{children}</div>
      <footer className="node-footer">
        <span className={`node-state ${localStatus === 'active' ? 'running' : ''}`}>
          <i style={{ background: statusColor }} aria-hidden="true" />{statusText}
        </span>
      </footer>
      <Handle type="source" position={Position.Right} className="node-handle" />
    </article>
  )
}
