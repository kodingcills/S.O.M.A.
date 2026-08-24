// CanvasView — INTERFACE.md "CANVAS VIEW" verbatim structure.
// PURE CANVAS STATE INVARIANT: buildCanvasState memoized on [events];
// nodeTypes at MODULE SCOPE (inline object = re-registration = position reset);
// nodesDraggable/Connectable/Focusable false so positions stay event-derived.
//
// Task 2.2 addition: optional `useFixture` prop — when true, renders from
// src/dev/fixtures.ts instead of the live events prop. Default false, so
// App.tsx wiring is untouched. Fixture mode exists for visual checks with
// no backend running.
import { useEffect, useMemo } from 'react'
import {
  ReactFlow, Background, MiniMap, Controls,
  useNodesState, useEdgesState, useReactFlow, ReactFlowProvider,
} from '@xyflow/react'
import { BackgroundVariant } from '@xyflow/react'
import type { EventLogEntry } from '../../types'
import {
  CANVAS_BG, SURFACE_BG, PANEL_BORDER, agentColor,
} from '../../utils/colors'
import { buildCanvasState } from '../../utils/buildCanvasState'
import { FIXTURE_EVENTS } from '../../dev/fixtures'
import { RootNode } from './nodes/RootNode'
import { ExplorationNode } from './nodes/ExplorationNode'
import { CapabilityNode } from './nodes/CapabilityNode'
import { WorldModelNode } from './nodes/WorldModelNode'
import { SimulationNode } from './nodes/SimulationNode'
import './nodes/nodes.css'

export interface CanvasViewProps {
  events:     EventLogEntry[]
  onNodeClick: (nodeId: string) => void
  useFixture?: boolean   // default false — App wiring unchanged
}

// MODULE SCOPE — see "Why nodeTypes outside component" in INTERFACE.md.
const NODE_TYPES = {
  root:        RootNode,
  exploration: ExplorationNode,
  capability:  CapabilityNode,
  worldmodel:  WorldModelNode,
  simulation:  SimulationNode,
}

function CanvasInner({ events, onNodeClick, useFixture = false }: CanvasViewProps) {
  // Memoize buildCanvasState — expensive pure function, same events = same output
  const { nodes: builtNodes, edges: builtEdges } = useMemo(
    () => buildCanvasState(useFixture ? FIXTURE_EVENTS : events),
    [events, useFixture]  // events ref changes only on new data
  )

  const [nodes, setNodes, onNodesChange] = useNodesState(builtNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(builtEdges)

  // Sync built state into RF state when events change
  useEffect(() => { setNodes(builtNodes) }, [builtNodes, setNodes])
  useEffect(() => { setEdges(builtEdges) }, [builtEdges, setEdges])

  // Camera: follow rightmost active node, debounced 2000ms
  const { setCenter } = useReactFlow()
  useEffect(() => {
    const active = nodes.filter(n => n.data.status === 'active')
    if (active.length === 0) return
    const rightmost = active.reduce((a, b) =>
      a.position.x > b.position.x ? a : b
    )
    const id = setTimeout(() => {
      setCenter(rightmost.position.x + 70, rightmost.position.y + 45,
                { zoom: 0.8, duration: 800 })
    }, 2000)
    return () => clearTimeout(id)
  }, [nodes, setCenter])

  return (
    <ReactFlow
      nodes={nodes} edges={edges}
      onNodesChange={onNodesChange} onEdgesChange={onEdgesChange}
      onNodeClick={(_, node) => onNodeClick(node.id)}
      nodeTypes={NODE_TYPES}
      nodesDraggable={false}     // agents manage their own positions
      nodesConnectable={false}   // no manual edge creation
      nodesFocusable={false}     // prevents tab-key canvas navigation
      fitView={false}
      minZoom={0.1} maxZoom={2.0}
      defaultViewport={{ x: 100, y: 200, zoom: 0.8 }}
      proOptions={{ hideAttribution: true }}
      style={{ background: CANVAS_BG }}
    >
      <Background variant={BackgroundVariant.Dots} gap={24} size={1} color={PANEL_BORDER} />
      <MiniMap
        position="bottom-right"
        nodeColor={(n) => agentColor((n.data?.agent_id as string) ?? '')}
        maskColor="rgba(10,14,26,0.8)"
        style={{ background: SURFACE_BG, border: `1px solid ${PANEL_BORDER}`,
                 borderRadius: 8, width: 160, height: 100 }}
      />
      <Controls position="bottom-left"
        style={{ background: SURFACE_BG, border: `1px solid ${PANEL_BORDER}`,
                 borderRadius: 8 }} />
    </ReactFlow>
  )
}

// Provider wrapper: useReactFlow() requires context; App renders CanvasView
// bare, so the provider lives here (App wiring untouched).
export function CanvasView(props: CanvasViewProps) {
  return (
    <ReactFlowProvider>
      <CanvasInner {...props} />
    </ReactFlowProvider>
  )
}
