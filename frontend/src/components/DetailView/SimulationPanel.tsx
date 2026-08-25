import type { SimulationState } from '../../types'
import { STATUS_COMPLETED, STATUS_SPAWNING } from '../../utils/colors'
import { CraftaxMap } from './CraftaxMap'

export interface SimulationPanelProps {
  simId: string | null
  sim: SimulationState | null
  showErrors?: boolean
  errorMap?: number[][]
}

export function SimulationPanel({ simId, sim, showErrors = false, errorMap }: SimulationPanelProps) {
  const statusColor = sim?.done ? STATUS_COMPLETED : STATUS_SPAWNING
  const statusText = sim?.done ? 'Complete' : 'Live'
  const activity = sim?.stats.is_sleeping ? 'SLEEPING' : sim?.stats.is_resting ? 'RESTING' : 'ACTIVE'

  return (
    <div className="simulation-panel">
      <div className="simulation-stage">
        <div className="simulation-visuals">
          {simId ? (
          <iframe
            src="http://localhost:9090/?url=rerun%2Bhttp://localhost:9876/proxy"
            title="Live Simulation — Rerun"
            allow="cross-origin-isolated"
          />
          ) : (
            <div className="empty-state"><span>Rerun workspace unavailable.</span></div>
          )}
          {sim ? <CraftaxMap tokenGrid={sim.token_2d} direction={sim.direction} showErrors={showErrors} errorMap={errorMap} /> : <div className="empty-state"><span>Craftax observation unavailable.</span></div>}
        </div>
        {sim && (
          <div className="craftax-stats" aria-label="Craftax survival statistics">
            <StatBar label="Health" value={sim.stats.health} max={9} />
            <StatBar label="Drink" value={sim.stats.drink} max={9} />
            <StatBar label="Food" value={sim.stats.food} max={9} />
            <StatBar label="Energy" value={sim.stats.energy} max={9} />
            <StatBar label="Light" value={sim.stats.light} max={1} />
          </div>
        )}
      </div>
      <div className="simulation-telemetry">
        <span>STEP {sim?.step ?? '—'}</span>
        <span>REWARD {sim ? sim.reward.toFixed(3) : '—'}</span>
        <span>Achievements {sim?.achievements_count ?? '—'}</span>
        <span>DOF {sim?.active_dof ?? '—'}</span>
        <span>{sim ? activity : '—'}</span>
        <span className="sim-status" style={{ color: statusColor }}>● {statusText}</span>
      </div>
    </div>
  )
}

function StatBar({ label, value, max }: { readonly label: string; readonly value: number; readonly max: number }) {
  return (
    <div className="craftax-stat">
      <span>{label}</span>
      <meter min={0} max={max} value={Math.max(0, Math.min(max, value))} />
      <strong>{value.toFixed(1)}</strong>
    </div>
  )
}
