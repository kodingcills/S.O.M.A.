import type { DetailResponse, EventLogEntry } from '../../types'
import { CANVAS_BG, PANEL_BORDER } from '../../utils/colors'
import { AgentFeed } from './AgentFeed'
import { SimulationPanel } from './SimulationPanel'
import { WorldModelPanel } from './WorldModelPanel'

export interface DetailViewProps {
  events: EventLogEntry[]
  detail: DetailResponse | null
  activeNode: string
}

export function DetailView({ events, detail, activeNode }: DetailViewProps) {
  const sim = detail?.simulation_state ?? null
  const belief = detail?.belief_state ?? null

  // Agent nodes execute on primary-anatomy envs; the registry only exposes
  // 'primary'/'comparison', so every detail-view selection streams primary.
  const simId = 'primary'

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        background: CANVAS_BG,
        overflow: 'hidden',
      }}
    >
      <div style={{ display: 'flex', flex: 1, minHeight: 0 }}>
        <div
          style={{
            flex: '0 0 55%',
            borderRight: `1px solid ${PANEL_BORDER}`,
            position: 'relative',
            minHeight: 0,
          }}
        >
          <SimulationPanel simId={simId} sim={sim} />
        </div>
        <div style={{ flex: 1, minWidth: 0, overflowY: 'auto' }}>
          <WorldModelPanel
            belief={belief}
            events={events}
            eePos={
              sim?.ee_position?.slice(0, 2) as [number, number] | undefined
            }
            targetPos={sim?.surgical_target?.position as
              | [number, number]
              | undefined}
          />
        </div>
      </div>
      <div style={{ height: '22vh', borderTop: `1px solid ${PANEL_BORDER}` }}>
        <AgentFeed events={events} activeNode={activeNode} />
      </div>
    </div>
  )
}
