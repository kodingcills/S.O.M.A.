// PLACEHOLDER — Task 2.1 scaffold only.
// Real detail view (SimulationPanel + WorldModelPanel + AgentFeed) lands in Tasks 2.3–2.4.
// Props below are the contract App.tsx will pass; events/detail are consumed then.
import type { DetailResponse, EventLogEntry } from '../../types'
import { CANVAS_BG, TEXT_SECONDARY } from '../../utils/colors'

export interface DetailViewProps {
  events:     EventLogEntry[]
  detail:     DetailResponse | null
  activeNode: string
}

export function DetailView({ activeNode }: DetailViewProps) {
  return (
    <div style={{
      width: '100%', height: '100%', background: CANVAS_BG,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      color: TEXT_SECONDARY,
      fontFamily: "'JetBrains Mono', ui-monospace, Menlo, monospace",
      fontSize: 12, letterSpacing: '0.15em',
    }}>
      DETAIL VIEW · {activeNode || 'NO NODE SELECTED'}
    </div>
  )
}
