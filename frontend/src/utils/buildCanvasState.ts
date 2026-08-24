// VERBATIM structure from docs/specs/INTERFACE.md "CANVAS VIEW — buildCanvasState".
// PURE CANVAS STATE INVARIANT (ARCHITECTURE.md): same events array → identical
// nodes+edges every call. No wall-clock reads, no RNG, no external state
// reads, no side effects. Positions derive only from parent_id + sibling count.
import type { Node, Edge } from '@xyflow/react'
import type { EventLogEntry, NodeKind, SomaNodeData, SomaEdgeData } from '../types'
import {
  STATUS_COMPLETED, STATUS_FAILED,
  TEXT_SECONDARY, CANVAS_BG,
  REGION_LABELS, agentColor,
} from './colors'

// @xyflow/react v12 constrains node/edge data to Record<string, unknown>;
// interfaces lack implicit index signatures, so widen via intersection.
export type SomaNode = Node<SomaNodeData & Record<string, unknown>>
export type SomaEdge = Edge<SomaEdgeData & Record<string, unknown>>

export interface CanvasState {
  nodes: SomaNode[]
  edges: SomaEdge[]
}

// ── Constants ────────────────────────────────────────────────────────────
const H_SPACING = 250  // px between parent and child
const V_SPACING = 120  // px between siblings

const V_OFFSETS: Record<NodeKind, number> = {
  root:        0,
  exploration: 0,
  capability:  -V_SPACING * 2,   // capability unlocks go upward
  worldmodel:  V_SPACING,
  simulation:  -V_SPACING,
}

// ── Helpers ──────────────────────────────────────────────────────────────
function agentIdToKind(agentId: string): NodeKind {
  if (agentId.startsWith('exp_')) return 'exploration'
  if (agentId.startsWith('cap_')) return 'capability'
  if (agentId.startsWith('wm_'))  return 'worldmodel'
  if (agentId.startsWith('sim_')) return 'simulation'
  return 'simulation'
}

function kindToLabel(kind: NodeKind): string {
  switch (kind) {
    case 'exploration': return 'EXPLORE'
    case 'capability':  return 'CAPABILITY'
    case 'worldmodel':  return 'WORLD MODEL'
    case 'simulation':  return 'SIMULATION'
    case 'root':        return 'SEED'
  }
}

function edgeLabel(event: EventLogEntry): string {
  if (event.region) return REGION_LABELS[event.region] ?? event.region
  const targetDof = event.payload?.target_dof
  if (typeof targetDof === 'number') return `DOF ${targetDof}`
  return (event.agent_id ?? '').slice(0, 8)
}

function computePosition(
  parentId: string,
  kind: NodeKind,
  siblingIndex: number,
  existingNodes: SomaNode[]
): { x: number; y: number } {
  const parent = existingNodes.find(n => n.id === parentId)
  const px = parent?.position.x ?? 0
  const py = parent?.position.y ?? 0
  const candidate = {
    x: px + H_SPACING,
    y: py + V_OFFSETS[kind] + siblingIndex * V_SPACING,
  }
  // Simple collision avoidance: bump y if within 80px of any existing node
  for (let attempt = 0; attempt < 5; attempt++) {
    const collision = existingNodes.some(n =>
      Math.abs(n.position.x - candidate.x) < 80 &&
      Math.abs(n.position.y - candidate.y) < 80
    )
    if (!collision) break
    candidate.y += V_SPACING
  }
  return candidate
}

// ── Pure builder ─────────────────────────────────────────────────────────
export function buildCanvasState(events: EventLogEntry[]): CanvasState {
  // ROOT NODE: always present, not from events
  const nodes: SomaNode[] = [{
    id:       'root',
    type:     'root',
    position: { x: 0, y: 0 },
    data:     { label: 'SEED', status: 'completed', agent_id: 'root',
                kind: 'root', region: null, error_before: null,
                error_after: null, dof_level: 1, timestamp: 0,
                sim_id: 'primary', activeMs: Infinity },
  }]
  const edges: SomaEdge[] = []
  // Map: parentId:kind → sibling count (for collision-free positioning)
  const siblingCounts = new Map<string, number>()

  for (const event of events) {
    switch (event.event_type) {
      case 'agent_spawned': {
        const agentId = event.agent_id ?? ''
        const kind = agentIdToKind(agentId)
        const parentId = event.parent_id ?? 'root'
        const key = `${parentId}:${kind}`
        const siblings = siblingCounts.get(key) ?? 0
        siblingCounts.set(key, siblings + 1)

        nodes.push({
          id:   agentId,
          type: kind,
          position: computePosition(parentId, kind, siblings, nodes),
          data: {
            label:       kindToLabel(kind),
            status:      'spawning',
            agent_id:    agentId,
            kind,
            region:      event.region,
            error_before: event.error_before,
            error_after:  null,
            dof_level:   (event.payload?.target_dof as number | undefined) ?? null,
            timestamp:   event.timestamp,
            sim_id:      event.sim_id,
            activeMs:    0,
          },
        })
        edges.push({
          id:     `${parentId}-${agentId}`,
          source: parentId,
          target: agentId,
          type:   'smoothstep',
          animated: true,
          label:  edgeLabel(event),
          data:   { label: edgeLabel(event), status: 'active' },
          style:  { stroke: agentColor(agentId), strokeWidth: 2 },
          labelStyle: { fill: TEXT_SECONDARY, fontSize: 9,
                        fontFamily: 'JetBrains Mono' },
          labelBgStyle: { fill: CANVAS_BG },
        })
        break
      }

      case 'agent_completed': {
        const node = nodes.find(n => n.id === event.agent_id)
        if (node) {
          node.data = { ...node.data, status: 'completed',
                        error_after: event.error_after }
        }
        const edge = edges.find(e => e.target === event.agent_id)
        if (edge && edge.data) {
          edge.animated = false
          edge.data.status = 'completed'
          edge.style = { ...edge.style, stroke: STATUS_COMPLETED,
                         strokeWidth: 2, opacity: 0.6 }
        }
        break
      }

      case 'agent_failed': {
        const node = nodes.find(n => n.id === event.agent_id)
        if (node) node.data = { ...node.data, status: 'failed' }
        const edge = edges.find(e => e.target === event.agent_id)
        if (edge && edge.data) {
          edge.animated = false
          edge.data.status = 'failed'
          edge.style = { ...edge.style, stroke: STATUS_FAILED, opacity: 0.4 }
        }
        break
      }

      case 'capability_unlocked': {
        const node = nodes.find(n => n.id === event.agent_id)
        if (node) {
          node.data = { ...node.data, status: 'completed',
                        dof_level: event.payload?.new_dof as number }
        }
        break
      }
    }
  }

  return { nodes, edges }
}
