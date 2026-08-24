import { describe, it, expect } from 'vitest'
import type { EventLogEntry } from '../../types'
import {
  pickQualifyingCompletion,
  regionToCells,
} from '../regionToCells'

describe('regionToCells', () => {
  it('upper_left is exactly rows<8 ∧ cols<8 (64 cells)', () => {
    const cells = regionToCells('upper_left')
    expect(cells.length).toBe(64)
    expect(cells.every((c) => c.row < 8 && c.col < 8)).toBe(true)
  })

  it('lower_right is exactly rows≥8 ∧ cols≥8', () => {
    const cells = regionToCells('lower_right')
    expect(cells.length).toBe(64)
    expect(cells.every((c) => c.row >= 8 && c.col >= 8)).toBe(true)
  })

  it('dynamic tool zone resolves 3×3 window around ee position', () => {
    // gridCenter(0.5*15)=7 → window rows/cols 6..8, clipped inside grid
    const cells = regionToCells('tool_tissue_boundary', [0.5, 0.5])
    expect(cells.length).toBe(9)
    expect(cells.every((c) => c.row >= 5 && c.row <= 8 && c.col >= 5 && c.col <= 8))
      .toBe(true)
  })

  it('dynamic target zone clips at grid edges', () => {
    const cells = regionToCells('surgical_target_vicinity', undefined, [0.0, 1.0])
    // col=0 → clipped center 1 → cols 0..2; row=15 → clipped center 14 → rows 13..15
    expect(cells.length).toBe(9)
    expect(cells.every((c) => c.col >= 0 && c.col <= 2 && c.row >= 13)).toBe(true)
  })

  it('unknown region yields no cells', () => {
    expect(regionToCells('nonexistent')).toEqual([])
  })
})

function entry(partial: Partial<EventLogEntry>): EventLogEntry {
  return {
    id: 1,
    timestamp: 0,
    event_type: 'agent_completed',
    agent_id: 'exp_x',
    parent_id: null,
    sim_id: null,
    region: null,
    error_before: null,
    error_after: null,
    payload: {},
    ...partial,
  }
}

describe('pickQualifyingCompletion', () => {
  it('returns null on empty or non-qualifying events', () => {
    expect(pickQualifyingCompletion([])).toBeNull()
    expect(
      pickQualifyingCompletion([
        entry({ error_before: 0.4, error_after: 0.38 }), // delta 0.02 ≤ 0.05
      ]),
    ).toBeNull()
  })

  it('picks the LAST qualifying completion only', () => {
    const picked = pickQualifyingCompletion([
      entry({ id: 1, error_before: 0.5, error_after: 0.1 }),
      entry({ id: 2, error_before: 0.4, error_after: 0.39 }),
      entry({ id: 3, error_before: 0.6, error_after: 0.2 }),
    ])
    expect(picked?.id).toBe(3)
  })
})
