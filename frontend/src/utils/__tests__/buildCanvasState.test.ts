// Task 2.2 — PURE CANVAS STATE INVARIANT + event→canvas mapping tests.
// Fixture is deterministic (fixed timestamps, no Date.now/Math.random).
// buildCanvasState must be pure: same input → deep-equal output, always.
import { describe, it, expect } from 'vitest'
import { buildCanvasState } from '../buildCanvasState'
import { FIXTURE_EVENTS } from '../../dev/fixtures'
import { PANEL_BORDER, STATUS_FAILED } from '../colors'

describe('buildCanvasState', () => {
  // test:purity:same_input_deep_equal_output
  it('is pure: same input produces deep-equal output', () => {
    const a = JSON.stringify(buildCanvasState(FIXTURE_EVENTS))
    const b = JSON.stringify(buildCanvasState(FIXTURE_EVENTS))
    expect(a).toEqual(b)
  })

  // test:mapping:kinds_statuses_edges
  it('maps agent kinds, status transitions, and edges correctly', () => {
    const { nodes, edges } = buildCanvasState(FIXTURE_EVENTS)

    // Root node always first, not derived from events
    expect(nodes[0].id).toBe('root')
    expect(nodes[0].type).toBe('root')

    // One node per spawned agent, kind from agent_id prefix
    const byId = new Map(nodes.map(n => [n.id, n]))
    expect(byId.get('exp_a')?.type).toBe('exploration')
    expect(byId.get('exp_b')?.type).toBe('exploration')
    expect(byId.get('cap_1')?.type).toBe('capability')
    expect(nodes).toHaveLength(4) // root + 3 agents

    // Status transitions
    expect(byId.get('exp_a')?.data.status).toBe('completed')
    expect(byId.get('exp_a')?.data.error_after).toBe(0.21)
    expect(byId.get('exp_b')?.data.status).toBe('failed')
    expect(byId.get('cap_1')?.data.status).toBe('completed')

    // Edges: smoothstep, source=parent_id ?? 'root', target=agent_id
    expect(edges).toHaveLength(3)
    for (const e of edges) {
      expect(e.type).toBe('smoothstep')
      expect(e.source).toBe('root')
      expect(['exp_a', 'exp_b', 'cap_1']).toContain(e.target)
    }

    // Completed edge settles into the neutral lineage system.
    const edgeA = edges.find(e => e.target === 'exp_a')!
    expect(edgeA.animated).toBe(false)
    expect(edgeA.style?.stroke).toBe(PANEL_BORDER)
    expect(edgeA.style?.opacity).toBe(1)

    // Failed edge: STATUS_FAILED stroke, not animated
    const edgeB = edges.find(e => e.target === 'exp_b')!
    expect(edgeB.animated).toBe(false)
    expect(edgeB.style?.stroke).toBe(STATUS_FAILED)

    // Spawned-but-unresolved edges remain active without perpetual animation.
    const edgeC = edges.find(e => e.target === 'cap_1')!
    expect(edgeC.animated).toBe(false)
  })

  // test:capability:unlock_sets_dof
  it('capability_unlocked sets dof_level on the capability node', () => {
    const { nodes } = buildCanvasState(FIXTURE_EVENTS)
    const cap = nodes.find(n => n.id === 'cap_1')!
    expect(cap.data.dof_level).toBe(2)
    expect(cap.data.status).toBe('completed')
  })

  it('normalizes the runtime-only orchestrator parent to the visible seed', () => {
    const spawned = {
      ...FIXTURE_EVENTS[0], id: 99, agent_id: 'exp_orch', parent_id: 'orchestrator',
    }
    const { nodes, edges } = buildCanvasState([spawned])
    expect(edges[0].source).toBe('root')
    expect(nodes[1].position.x).toBeGreaterThan(nodes[0].position.x)
  })

  it('advances visual generations left-to-right as history grows', () => {
    const spawned = Array.from({ length: 5 }, (_, index) => ({
      ...FIXTURE_EVENTS[0], id: index + 20, timestamp: 1756000100 + index,
      agent_id: `exp_${index}`, parent_id: 'root',
    }))
    const { nodes } = buildCanvasState(spawned)
    expect(nodes[4].data.generation).toBe(1)
    expect(nodes[5].data.generation).toBe(2)
    expect(nodes[5].position.x).toBeGreaterThan(nodes[4].position.x)
  })
})
