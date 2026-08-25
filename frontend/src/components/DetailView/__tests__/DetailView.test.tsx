import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'
import type { DetailResponse } from '../../../types'
import { DetailView } from '..'

const detail: DetailResponse = {
  simulation_state: {
    simulation_id: 'primary',
    token_2d: Array.from({ length: 99 }, () => [1, 0, 0, 0]),
    vector: Array.from({ length: 47 }, () => 0),
    direction: [1],
    raw_observation: Array.from({ length: 8268 }, () => 0),
    achievements_count: 4,
    stats: { health: 9, drink: 8, food: 7, energy: 6, light: 1, is_sleeping: 0, is_resting: 0 },
    step: 12,
    done: false,
    reward: 0.5,
    active_dof: 1,
  },
  belief_state: {
    version: 2,
    world_model_version: 2,
    prediction_error_map: Array.from({ length: 16 }, () => Array.from({ length: 16 }, () => 0.1)),
    error_map: Array.from({ length: 16 }, () => Array.from({ length: 16 }, () => 0.1)),
    global_mean_error: 0.1,
    regional_errors: {},
    active_dof: 1,
    episode_count: 1,
    timestamp: 1,
  },
}

describe('DetailView evidence', () => {
  it('shows real Craftax evidence without a synthetic MJPEG image', () => {
    // Given / When
    const markup = renderToStaticMarkup(
      <DetailView
        events={[]}
        detail={detail}
        activeNode="root"
        onClose={vi.fn()}
        onOpenSimulation={vi.fn()}
        onOpenModels={vi.fn()}
      />,
    )

    // Then
    expect(markup).not.toContain('<img')
    expect(markup).toContain('aria-label="Craftax symbolic map, 11 columns by 9 rows"')
    expect(markup).toContain('Craftax observation')
    expect(markup).toContain('Simulation')
  })
})
