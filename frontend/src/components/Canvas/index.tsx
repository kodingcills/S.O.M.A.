// PLACEHOLDER — Task 2.1 scaffold only.
// Real canvas (React Flow tree, buildCanvasState, node components) lands in Task 2.2.
// Props below are the contract App.tsx will pass; events is consumed then.
import type { EventLogEntry } from '../../types'
import { CANVAS_BG, TEXT_SECONDARY } from '../../utils/colors'

export interface CanvasViewProps {
  events:     EventLogEntry[]
  onNodeClick: (nodeId: string) => void
}

export function CanvasView({ events }: CanvasViewProps) {
  return (
    <div style={{
      width: '100%', height: '100%', background: CANVAS_BG,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      color: TEXT_SECONDARY,
      fontFamily: "'JetBrains Mono', ui-monospace, Menlo, monospace",
      fontSize: 12, letterSpacing: '0.15em',
    }}>
      CANVAS VIEW · {events.length} EVENTS
    </div>
  )
}
