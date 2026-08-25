import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import type { SimulationState } from '../../../types'
import { SimulationPanel } from '../SimulationPanel'

const craftaxState: SimulationState = {
  simulation_id: 'primary',
  token_2d: Array.from({ length: 99 }, (_, index) => [index % 12, 0, 0, 0]),
  vector: Array.from({ length: 47 }, () => 0),
  direction: [2],
  raw_observation: Array.from({ length: 8268 }, () => 0),
  achievements_count: 3,
  stats: {
    health: 8,
    drink: 7,
    food: 6,
    energy: 5,
    light: 0.75,
    is_sleeping: 0,
    is_resting: 1,
  },
  step: 42,
  done: false,
  reward: 1.25,
  active_dof: 2,
}

describe('SimulationPanel', () => {
  it('renders the Rerun iframe while preserving the simulation structure', () => {
    // Given
    const simId = 'primary'

    // When
    const markup = renderToStaticMarkup(<SimulationPanel simId={simId} sim={craftaxState} />)

    // Then
    expect(markup).toContain('<iframe')
    expect(markup).toContain('src="http://localhost:9090/?url=rerun%2Bhttp://localhost:9876/proxy"')
    expect(markup).toContain('title="Live Simulation — Rerun"')
    expect(markup).toContain('allow="cross-origin-isolated"')
    expect(markup).not.toContain('<img')
    expect(markup).toContain('<canvas')
    expect(markup).toContain('class="simulation-telemetry"')
  })

  it('preserves the unavailable state when no simulation is selected', () => {
    // Given
    const simId = null

    // When
    const markup = renderToStaticMarkup(<SimulationPanel simId={simId} sim={null} />)

    // Then
    expect(markup).not.toContain('<iframe')
    expect(markup).toContain('class="empty-state"')
    expect(markup).toContain('Craftax observation unavailable.')
    expect(markup).toContain('class="simulation-telemetry"')
  })

  it('renders real Craftax map and survival statistics', () => {
    // Given / When
    const markup = renderToStaticMarkup(<SimulationPanel simId="primary" sim={craftaxState} />)

    // Then
    expect(markup).toContain('aria-label="Craftax symbolic map, 11 columns by 9 rows"')
    expect(markup).toContain('Health')
    expect(markup).toContain('8.0')
    expect(markup).toContain('Achievements 3')
    expect(markup).toContain('STEP 42')
    expect(markup).toContain('RESTING')
  })
})
