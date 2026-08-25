import { useEffect, useMemo, useRef, useState } from 'react'
import {
  Background,
  BackgroundVariant,
  MiniMap,
  Panel,
  ReactFlow,
  ReactFlowProvider,
  useEdgesState,
  useNodesState,
  useReactFlow,
  useViewport,
} from '@xyflow/react'
import type { EventLogEntry, NodeKind } from '../../types'
import { CANVAS_BG, COLOR_SELECTION, PANEL_BORDER, SURFACE_BG, agentColor } from '../../utils/colors'
import { buildCanvasState } from '../../utils/buildCanvasState'
import { FIXTURE_EVENTS } from '../../dev/fixtures'
import { CapabilityNode } from './nodes/CapabilityNode'
import { ExplorationNode } from './nodes/ExplorationNode'
import { RootNode } from './nodes/RootNode'
import { SimulationNode } from './nodes/SimulationNode'
import { WorldModelNode } from './nodes/WorldModelNode'
import './nodes/nodes.css'

export interface CanvasViewProps {
  events: EventLogEntry[]
  selectedNodeId: string | null
  onNodeClick: (nodeId: string) => void
  useFixture?: boolean
}

type GraphFilter = 'all' | 'exploration' | 'capability' | 'failed' | 'active'

const NODE_TYPES = {
  root: RootNode,
  exploration: ExplorationNode,
  capability: CapabilityNode,
  worldmodel: WorldModelNode,
  simulation: SimulationNode,
}

function relatedIds(selected: string | null, edges: ReturnType<typeof buildCanvasState>['edges']): Set<string> {
  const related = new Set<string>()
  if (!selected) return related
  related.add(selected)

  const ancestors = [selected]
  while (ancestors.length) {
    const current = ancestors.pop()!
    for (const edge of edges) {
      if (edge.target === current && !related.has(edge.source)) {
        related.add(edge.source)
        ancestors.push(edge.source)
      }
    }
  }

  const descendants = [selected]
  while (descendants.length) {
    const current = descendants.pop()!
    for (const edge of edges) {
      if (edge.source === current && !related.has(edge.target)) {
        related.add(edge.target)
        descendants.push(edge.target)
      }
    }
  }
  return related
}

function matchesFilter(kind: NodeKind, status: string, filter: GraphFilter): boolean {
  if (kind === 'root') return true
  if (filter === 'all') return true
  if (filter === 'failed') return status === 'failed'
  if (filter === 'active') return status === 'active' || status === 'spawning'
  return kind === filter
}

function CanvasInner({ events, selectedNodeId, onNodeClick, useFixture = false }: CanvasViewProps) {
  const sourceEvents = useFixture ? FIXTURE_EVENTS : events
  const built = useMemo(() => buildCanvasState(sourceEvents), [sourceEvents])
  const [nodes, setNodes, onNodesChange] = useNodesState(built.nodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(built.edges)
  const [filter, setFilter] = useState<GraphFilter>('all')
  const [query, setQuery] = useState('')
  const [following, setFollowing] = useState(false)
  const [newEvents, setNewEvents] = useState(0)
  const previousEventCount = useRef(sourceEvents.length)
  const fittedOnce = useRef(false)
  const { fitView, zoomIn, zoomOut, setCenter } = useReactFlow()
  const viewport = useViewport()

  useEffect(() => setNodes(built.nodes), [built.nodes, setNodes])
  useEffect(() => setEdges(built.edges), [built.edges, setEdges])

  useEffect(() => {
    const difference = sourceEvents.length - previousEventCount.current
    previousEventCount.current = sourceEvents.length
    if (difference <= 0) return
    if (!fittedOnce.current && built.nodes.length > 1) {
      fittedOnce.current = true
      window.setTimeout(() => fitView({ padding: 0.22, duration: 320, maxZoom: 1 }), 0)
      return
    }
    if (following) {
      const frontier = built.nodes[built.nodes.length - 1]
      if (frontier) setCenter(frontier.position.x + 110, frontier.position.y + 62, { zoom: 0.9, duration: 320 })
    } else {
      setNewEvents(count => count + difference)
    }
  }, [sourceEvents.length, built.nodes, following, fitView, setCenter])

  const visibleNodeIds = useMemo(() => {
    const normalized = query.trim().toLowerCase()
    return new Set(nodes.filter(node => {
      if (node.data.kind === 'root') return true
      const filterMatch = matchesFilter(node.data.kind, node.data.status, filter)
      const queryMatch = !normalized || [node.id, node.data.title, node.data.region ?? '']
        .some(value => value.toLowerCase().includes(normalized))
      return filterMatch && queryMatch
    }).map(node => node.id))
  }, [nodes, filter, query])

  const relationshipIds = useMemo(() => relatedIds(selectedNodeId, edges), [selectedNodeId, edges])
  const displayNodes = useMemo(() => nodes
    .filter(node => visibleNodeIds.has(node.id))
    .map(node => ({
      ...node,
      selected: node.id === selectedNodeId,
      style: {
        ...node.style,
        opacity: selectedNodeId && !relationshipIds.has(node.id) ? 0.32 : 1,
        transition: 'opacity 140ms ease',
      },
    })), [nodes, visibleNodeIds, selectedNodeId, relationshipIds])

  const displayEdges = useMemo(() => edges
    .filter(edge => visibleNodeIds.has(edge.source) && visibleNodeIds.has(edge.target))
    .map(edge => {
      const emphasized = selectedNodeId && (edge.source === selectedNodeId || edge.target === selectedNodeId)
      const contextual = selectedNodeId && relationshipIds.has(edge.source) && relationshipIds.has(edge.target)
      return {
        ...edge,
        style: {
          ...edge.style,
          stroke: emphasized ? COLOR_SELECTION : edge.style?.stroke,
          strokeWidth: emphasized ? 2 : 1.25,
          opacity: selectedNodeId ? (emphasized ? 1 : contextual ? 0.45 : 0.08) : edge.style?.opacity,
          transition: 'opacity 140ms ease, stroke 140ms ease',
        },
      }
    }), [edges, visibleNodeIds, selectedNodeId, relationshipIds])

  const followFrontier = () => {
    const frontier = built.nodes[built.nodes.length - 1]
    if (frontier) setCenter(frontier.position.x + 110, frontier.position.y + 62, { zoom: 0.9, duration: 280 })
    setFollowing(true)
    setNewEvents(0)
  }

  return (
    <ReactFlow
      nodes={displayNodes}
      edges={displayEdges}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onNodeClick={(_, node) => onNodeClick(node.id)}
      onNodeDoubleClick={(_, node) => fitView({ nodes: [node], padding: 1.8, maxZoom: 1.15, duration: 260 })}
      onMoveStart={(event) => { if (event) setFollowing(false) }}
      nodeTypes={NODE_TYPES}
      nodesDraggable={false}
      nodesConnectable={false}
      nodesFocusable
      elementsSelectable
      fitView={false}
      minZoom={0.25}
      maxZoom={1.5}
      defaultViewport={{ x: 90, y: 300, zoom: 0.82 }}
      proOptions={{ hideAttribution: true }}
      style={{ background: CANVAS_BG }}
      onlyRenderVisibleElements
    >
      <Background variant={BackgroundVariant.Dots} gap={24} size={0.8} color={PANEL_BORDER} />

      <Panel position="top-left" className="graph-toolbar">
        <div className="zoom-tools" aria-label="Graph zoom controls">
          <button onClick={() => zoomOut({ duration: 120 })} aria-label="Zoom out">−</button>
          <span className="mono">{Math.round(viewport.zoom * 100)}%</span>
          <button onClick={() => zoomIn({ duration: 120 })} aria-label="Zoom in">+</button>
          <button className="fit-button" onClick={() => fitView({ padding: 0.2, duration: 260, maxZoom: 1 })}>Fit</button>
        </div>
        <div className="filter-tools" aria-label="Filter graph nodes">
          {(['all', 'exploration', 'capability', 'failed', 'active'] as GraphFilter[]).map(item => (
            <button key={item} className={filter === item ? 'selected' : ''} onClick={() => setFilter(item)}>
              {item === 'all' ? 'All nodes' : item === 'capability' ? 'Capabilities' : item[0].toUpperCase() + item.slice(1)}
            </button>
          ))}
        </div>
        <label className="graph-search">
          <span className="sr-only">Search nodes</span>
          <span aria-hidden="true">⌕</span>
          <input value={query} onChange={event => setQuery(event.target.value)} placeholder="Search nodes…" />
        </label>
      </Panel>

      <Panel position="top-right" className="graph-orientation">
        <span>Earlier</span><i aria-hidden="true">→</i><span>Newer</span>
      </Panel>

      {(newEvents > 0 || following) && (
        <Panel position="bottom-center" className="live-indicator">
          <span className="status-dot running" aria-hidden="true" />
          {following ? 'Following live frontier' : `${newEvents} new event${newEvents === 1 ? '' : 's'}`}
          {!following && <button onClick={followFrontier}>View →</button>}
        </Panel>
      )}

      {sourceEvents.length === 0 && (
        <Panel position="bottom-center" className="graph-empty">
          <strong>No learning events yet.</strong>
          The graph will grow when SOMA encounters prediction error and initiates exploration.
          <span>Waiting for first observation…</span>
        </Panel>
      )}

      {displayNodes.length > 8 && (
        <MiniMap
          position="bottom-right"
          nodeColor={node => node.id === selectedNodeId ? COLOR_SELECTION : agentColor(String(node.data?.agent_id ?? ''))}
          maskColor="rgba(248,248,246,.72)"
          pannable
          zoomable
          style={{ background: SURFACE_BG, border: `1px solid ${PANEL_BORDER}`, borderRadius: 6, width: 150, height: 92 }}
        />
      )}
    </ReactFlow>
  )
}

export function CanvasView(props: CanvasViewProps) {
  return <ReactFlowProvider><CanvasInner {...props} /></ReactFlowProvider>
}
