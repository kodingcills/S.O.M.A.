import type { Edge, Node } from '@xyflow/react'
import type { EventLogEntry, NodeKind, SomaEdgeData, SomaNodeData } from '../types'
import {
  PANEL_BORDER,
  REGION_LABELS,
  STATUS_FAILED,
  TEXT_SECONDARY,
  TEXT_TERTIARY,
} from './colors'

export type SomaNode = Node<SomaNodeData & Record<string, unknown>>
export type SomaEdge = Edge<SomaEdgeData & Record<string, unknown>>

export interface CanvasState {
  nodes: SomaNode[]
  edges: SomaEdge[]
}

const H_SPACING = 292
const ROW_OFFSETS = [-232, -76, 80, 236]
const NODES_PER_GENERATION = ROW_OFFSETS.length

function agentIdToKind(agentId: string): NodeKind {
  if (agentId.startsWith('exp_')) return 'exploration'
  if (agentId.startsWith('cap_')) return 'capability'
  if (agentId.startsWith('wm_')) return 'worldmodel'
  if (agentId.startsWith('sim_')) return 'simulation'
  return 'simulation'
}

function kindToLabel(kind: NodeKind): string {
  switch (kind) {
    case 'exploration': return 'Exploration'
    case 'capability': return 'Capability'
    case 'worldmodel': return 'World model'
    case 'simulation': return 'Evaluation'
    case 'root': return 'Seed'
  }
}

function semanticTitle(kind: NodeKind, event: EventLogEntry, dof: number | null): string {
  if (kind === 'exploration') {
    return event.region ? `${REGION_LABELS[event.region] ?? event.region} exploration` : 'Focused exploration'
  }
  if (kind === 'capability') return dof ? `DOF ${dof} unlocked` : 'Capability expansion'
  if (kind === 'worldmodel') return 'Model update'
  if (kind === 'simulation') return 'Simulation evaluation'
  return 'Initial system state'
}

function payloadNumber(event: EventLogEntry, key: string): number | null {
  const value = event.payload?.[key]
  return typeof value === 'number' ? value : null
}

function dofFromEvent(event: EventLogEntry): number | null {
  const payloadDof = payloadNumber(event, 'target_dof') ?? payloadNumber(event, 'new_dof')
  if (payloadDof !== null) return payloadDof
  const match = event.agent_id?.match(/^cap_(\d+)$/)
  return match ? Number(match[1]) : null
}

function normalizeParent(parentId: string | null, knownIds: Set<string>): string {
  if (!parentId || parentId === 'orchestrator' || !knownIds.has(parentId)) return 'root'
  return parentId
}

function edgeLabel(event: EventLogEntry): string {
  if (event.region) return REGION_LABELS[event.region] ?? event.region
  const dof = dofFromEvent(event)
  return dof ? `DOF ${dof}` : 'spawned'
}

/**
 * Deterministic event-log projection. Time advances left-to-right in visual
 * generations of four events. Explicit agent parentage is preserved; the
 * runtime-only `orchestrator` parent is represented by the seed node because
 * it is not itself a persistent learning event.
 */
export function buildCanvasState(events: EventLogEntry[]): CanvasState {
  const nodes: SomaNode[] = [{
    id: 'root',
    type: 'root',
    position: { x: 0, y: 0 },
    data: {
      label: 'Seed', title: 'Initial system state', status: 'completed',
      agent_id: 'root', kind: 'root', region: null, error_before: null,
      error_after: null, dof_level: 1, timestamp: 0, sim_id: 'primary',
      event_id: 0, generation: 0, samples_collected: null,
      fine_tune_loss: null, api_tokens_used: null, error_message: null,
      activeMs: Infinity,
    },
  }]
  const edges: SomaEdge[] = []
  const knownIds = new Set<string>(['root'])
  let spawnIndex = 0

  for (const event of events) {
    if (event.event_type === 'agent_spawned') {
      const agentId = event.agent_id ?? ''
      if (!agentId || knownIds.has(agentId)) continue
      const kind = agentIdToKind(agentId)
      const parentId = normalizeParent(event.parent_id, knownIds)
      const visualGeneration = Math.floor(spawnIndex / NODES_PER_GENERATION) + 1
      const row = spawnIndex % NODES_PER_GENERATION
      const parent = nodes.find(node => node.id === parentId)
      const x = Math.max(visualGeneration * H_SPACING, (parent?.position.x ?? 0) + H_SPACING)
      const dof = dofFromEvent(event)

      nodes.push({
        id: agentId,
        type: kind,
        position: { x, y: ROW_OFFSETS[row] },
        data: {
          label: kindToLabel(kind),
          title: semanticTitle(kind, event, dof),
          status: 'spawning',
          agent_id: agentId,
          kind,
          region: event.region,
          error_before: event.error_before,
          error_after: null,
          dof_level: dof,
          timestamp: event.timestamp,
          sim_id: event.sim_id,
          event_id: event.id,
          generation: visualGeneration,
          samples_collected: null,
          fine_tune_loss: null,
          api_tokens_used: null,
          error_message: null,
          activeMs: 0,
        },
      })

      const label = edgeLabel(event)
      edges.push({
        id: `${parentId}-${agentId}`,
        source: parentId,
        target: agentId,
        type: 'smoothstep',
        animated: false,
        label,
        data: { label, status: 'active' },
        style: { stroke: TEXT_TERTIARY, strokeWidth: 1.25, opacity: 0.72 },
        labelStyle: { fill: TEXT_SECONDARY, fontSize: 9, fontFamily: 'Inter, sans-serif' },
        labelBgStyle: { fill: '#F8F8F6', fillOpacity: 0.9 },
      })
      knownIds.add(agentId)
      spawnIndex += 1
      continue
    }

    const node = nodes.find(candidate => candidate.id === event.agent_id)
    const edge = edges.find(candidate => candidate.target === event.agent_id)

    if (event.event_type === 'agent_completed' && node) {
      node.data = {
        ...node.data,
        status: 'completed',
        error_after: event.error_after,
        samples_collected: payloadNumber(event, 'samples_collected'),
        fine_tune_loss: payloadNumber(event, 'fine_tune_loss'),
        api_tokens_used: payloadNumber(event, 'api_tokens_used'),
      }
      if (edge?.data) {
        edge.data.status = 'completed'
        edge.style = { ...edge.style, stroke: PANEL_BORDER, opacity: 1 }
      }
    }

    if (event.event_type === 'agent_failed' && node) {
      const error = event.payload?.error
      node.data = {
        ...node.data,
        status: 'failed',
        error_message: typeof error === 'string' ? error : 'Unknown failure',
      }
      if (edge?.data) {
        edge.data.status = 'failed'
        edge.style = { ...edge.style, stroke: STATUS_FAILED, opacity: 0.7 }
      }
    }

    if (event.event_type === 'capability_unlocked' && node) {
      const dof = dofFromEvent(event)
      node.data = {
        ...node.data,
        status: 'completed',
        dof_level: dof,
        title: semanticTitle('capability', event, dof),
      }
    }
  }

  return { nodes, edges }
}
