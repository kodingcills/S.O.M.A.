// Shared node shell — INTERFACE.md "NODE COMPONENTS" shared structure.
// 140×90 fixed, 2px border (type color), 20px type-colored header with
// status dot, 70px content area. Owns the three lifecycle behaviors:
// spawn pop on mount, spawning→active timer keyed on data.timestamp,
// completion flash + particle burst on status transition.
import { useEffect, useRef, useState } from 'react'
import { Handle, Position } from '@xyflow/react'
import type { ReactNode } from 'react'
import type { NodeStatus } from '../../../types'
import {
  STATUS_SPAWNING, STATUS_COMPLETED, STATUS_FAILED,
} from '../../../utils/colors'

export const NODE_W = 140
export const NODE_H = 90

interface NodeShellProps {
  color: string          // type color: border + header bg + glow (--node-color)
  label: string
  status: NodeStatus     // event-log status (spawning/completed/failed)
  timestamp: number      // unix seconds — keys the spawning→active timer
  dotColor?: string      // override header dot (e.g. errorToColor)
  children: ReactNode    // content area (70px)
}

const MONO = "'JetBrains Mono', ui-monospace, Menlo, monospace"

export function NodeShell({ color, label, status, timestamp, dotColor, children }: NodeShellProps) {
  // Spawning → active after 500ms, keyed on event timestamp so replayed
  // nodes (spawned long ago) transition immediately.
  const [localStatus, setLocalStatus] = useState<NodeStatus>(status)
  useEffect(() => {
    if (status !== 'spawning') { setLocalStatus(status); return }
    const elapsedMs = Date.now() - timestamp * 1000
    const remaining = Math.max(0, 500 - elapsedMs)
    const id = setTimeout(() => setLocalStatus('active'), remaining)
    return () => clearTimeout(id)
  }, [status, timestamp])

  // Completion flash + particle burst on transition into 'completed'.
  const prevStatus = useRef<NodeStatus>(status)
  const [flash, setFlash] = useState(false)
  const [particles, setParticles] = useState<{ id: number; dx: number; dy: number }[]>([])
  useEffect(() => {
    if (prevStatus.current === 'completed' || status !== 'completed') {
      prevStatus.current = status
      return
    }
    prevStatus.current = status
    setFlash(true)
    setParticles(Array.from({ length: 8 }, (_, i) => ({
      id: i,
      dx: Math.cos(i * Math.PI / 4) * 30,
      dy: Math.sin(i * Math.PI / 4) * 30,
    })))
    const t1 = setTimeout(() => setFlash(false), 400)
    const t2 = setTimeout(() => setParticles([]), 400)
    return () => { clearTimeout(t1); clearTimeout(t2) }
  }, [status])

  const borderColor =
    flash ? '#FFFFFF'
    : status === 'completed' ? STATUS_COMPLETED
    : status === 'failed' ? STATUS_FAILED
    : color

  const dotBg = dotColor
    ?? (localStatus === 'completed' ? STATUS_COMPLETED
      : localStatus === 'failed' ? STATUS_FAILED
      : localStatus === 'active' ? STATUS_COMPLETED
      : STATUS_SPAWNING)

  return (
    <div
      className={`node-spawn ${localStatus === 'active' ? 'node-active' : ''} ${flash ? 'node-completing' : ''}`}
      style={{
        width: NODE_W, height: NODE_H,
        border: `2px solid ${borderColor}`,
        borderRadius: 8,
        background: 'var(--node-bg, #0F1629)',
        // --node-color drives the glow keyframe in nodes.css
        ['--node-color' as string]: color,
        overflow: 'hidden',
        position: 'relative',
        fontFamily: MONO,
        boxSizing: 'border-box',
        transition: 'border-color 400ms ease-out',
      }}
    >
      <Handle type="target" position={Position.Left}
              style={{ width: 6, height: 6, background: color,
                       border: `1px solid #1E2D4A` }} />
      {/* Header: 20px, type-colored bg, label + status dot */}
      <div style={{
        height: 20, background: color,
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '0 6px', boxSizing: 'border-box',
      }}>
        <span style={{
          fontSize: 9, fontWeight: 700, letterSpacing: '0.08em',
          color: '#0A0E1A', whiteSpace: 'nowrap', overflow: 'hidden',
        }}>
          {label}
        </span>
        <span className={localStatus === 'spawning' || localStatus === 'active'
                         ? 'status-dot-pulsing' : ''}
              style={{
                width: 6, height: 6, borderRadius: '50%',
                background: dotBg, flexShrink: 0,
                border: '1px solid rgba(10,14,26,0.35)',
              }} />
      </div>
      {/* Content area */}
      <div style={{
        height: NODE_H - 20, padding: '5px 7px', boxSizing: 'border-box',
        fontSize: 10, lineHeight: 1.45, color: '#E8EFF8',
        display: 'flex', flexDirection: 'column', justifyContent: 'center',
        overflow: 'hidden',
      }}>
        {children}
      </div>
      <Handle type="source" position={Position.Right}
              style={{ width: 6, height: 6, background: color,
                       border: `1px solid #1E2D4A` }} />
      {particles.map(p => (
        <div key={p.id} className="particle"
             style={{ background: STATUS_COMPLETED,
                      ['--dx' as string]: `${p.dx}px`,
                      ['--dy' as string]: `${p.dy}px` }} />
      ))}
    </div>
  )
}
