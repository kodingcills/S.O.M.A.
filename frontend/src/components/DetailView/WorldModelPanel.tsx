import { useEffect, useMemo, useState } from 'react'
import type { Data, Layout } from 'plotly.js'
import Plot from 'react-plotly.js'
import { BarChart, Bar, XAxis, YAxis, Cell } from 'recharts'
import type { BeliefStateSnapshot, EventLogEntry } from '../../types'
import {
  PANEL_BORDER,
  REGION_LABELS,
  SURFACE_BG,
  TEXT_SECONDARY,
  errorToColor,
} from '../../utils/colors'
import { pickQualifyingCompletion, regionToCells } from '../../utils/regionToCells'

export interface WorldModelPanelProps {
  belief?: BeliefStateSnapshot | null
  events: EventLogEntry[]
  showErrors?: boolean
  eePos?: [number, number]
  targetPos?: [number, number]
}

const MONO = "'JetBrains Mono', ui-monospace, Menlo, monospace"
const EMPTY_MAP: number[][] = Array.from({ length: 16 }, () => Array<number>(16).fill(0))

interface FlashState {
  cells: { row: number; col: number }[]
  seq: number
}

export function WorldModelPanel({
  belief,
  events,
  showErrors = false,
  eePos,
  targetPos,
}: WorldModelPanelProps) {
  const errorMap = belief?.error_map ?? EMPTY_MAP
  const globalError = belief?.global_mean_error ?? 0
  const dof = belief?.active_dof ?? 1

  const plotlyData: Data[] = useMemo(
    () => [
      {
        z: errorMap,
        type: 'heatmap' as const,
        colorscale: [
          [0, 'hsl(120,85%,45%)'],
          [0.5, 'hsl(60,85%,45%)'],
          [1, 'hsl(0,85%,45%)'],
        ],
        zmin: 0,
        zmax: 1,
        showscale: true,
        colorbar: {
          thickness: 12,
          tickfont: { color: TEXT_SECONDARY, size: 9 },
        },
        hovertemplate: 'Row %{y}, Col %{x}<br>Error: %{z:.4f}<extra></extra>',
      },
    ],
    [errorMap],
  )

  const layout: Partial<Layout> = useMemo(
    () => ({
      paper_bgcolor: SURFACE_BG,
      plot_bgcolor: SURFACE_BG,
      font: { color: TEXT_SECONDARY, family: MONO, size: 9 },
      margin: { t: 24, r: 12, b: 22, l: 12 },
      xaxis: { gridcolor: PANEL_BORDER, zeroline: false },
      yaxis: { gridcolor: PANEL_BORDER, zeroline: false, autorange: 'reversed' },
      annotations: [
        {
          text: `Global: ${globalError.toFixed(3)}`,
          xref: 'paper',
          yref: 'paper',
          x: 1,
          y: 1.08,
          xanchor: 'right',
          showarrow: false,
          font: {
            color: errorToColor(globalError),
            size: 11,
            family: MONO,
          },
        },
      ],
    }),
    [globalError],
  )

  // THE money moment: flash white→yellow→transparent over 600ms on the last
  // qualifying agent_completed (delta > 0.05). Cells resolve dynamic regions
  // against live EE/target positions when available.
  const [flash, setFlash] = useState<FlashState>({ cells: [], seq: 0 })

  useEffect(() => {
    const qualifying = pickQualifyingCompletion(events)
    if (!qualifying?.region) return
    const cells = regionToCells(qualifying.region, eePos, targetPos)
    if (cells.length === 0) return
    setFlash((prev) => ({ cells, seq: prev.seq + 1 }))
    const id = setTimeout(() => setFlash((prev) => ({ ...prev, cells: [] })), 650)
    return () => clearTimeout(id)
  }, [events, eePos, targetPos])

  const barData = useMemo(
    () =>
      Object.entries(belief?.regional_errors ?? {}).map(([key, err]) => ({
        name: REGION_LABELS[key] ?? key,
        confidence: Math.round((1 - err) * 100),
        error: err,
      })),
    [belief?.regional_errors],
  )

  void showErrors

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10, padding: 10, boxSizing: 'border-box', fontFamily: MONO }}>
      <div style={{ fontSize: 10, color: TEXT_SECONDARY, letterSpacing: '0.1em' }}>
        WORLD MODEL · v{belief?.world_model_version ?? belief?.version ?? 0}{' '}
        <span style={{ color: errorToColor(globalError), marginLeft: 8 }}>
          GLOBAL {globalError.toFixed(3)}
        </span>
      </div>

      <div style={{ position: 'relative', background: SURFACE_BG, border: `1px solid ${PANEL_BORDER}`, borderRadius: 8, padding: 4 }}>
        <Plot
          data={plotlyData}
          layout={layout}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: '100%', height: '260px' }}
          useResizeHandler
        />
        <div style={{ position: 'absolute', inset: 4, pointerEvents: 'none' }}>
          {flash.cells.map((c) => (
            <div
              key={`${flash.seq}-${c.row}-${c.col}`}
              className="heatmap-flash"
              style={{
                position: 'absolute',
                left: `${c.col * (100 / 16)}%`,
                top: `${c.row * (100 / 16)}%`,
                width: `${100 / 16}%`,
                height: `${100 / 16}%`,
              }}
            />
          ))}
        </div>
      </div>

      <div style={{ background: SURFACE_BG, border: `1px solid ${PANEL_BORDER}`, borderRadius: 8, padding: '6px 8px' }}>
        <BarChart data={barData} layout="vertical" width={340} height={150}>
          <XAxis type="number" domain={[0, 100]} hide />
          <YAxis
            type="category"
            dataKey="name"
            width={128}
            tick={{ fill: TEXT_SECONDARY, fontSize: 9 }}
          />
          <Bar dataKey="confidence" radius={3}>
            {barData.map((d) => (
              <Cell key={d.name} fill={errorToColor(d.error)} />
            ))}
          </Bar>
        </BarChart>
      </div>

      <div style={{ display: 'flex', gap: 6, alignItems: 'center', fontSize: 9, color: TEXT_SECONDARY }}>
        <span>DOF</span>
        {[1, 2, 3, 4, 5, 6].map((n) => (
          <span
            key={n}
            style={{
              width: 14, height: 14, borderRadius: '50%', display: 'inline-block',
              background: n <= dof ? '#34C759' : 'transparent',
              border: `1px solid ${n <= dof ? '#34C759' : PANEL_BORDER}`,
            }}
          />
        ))}
        <span>{dof}/6</span>
      </div>
    </div>
  )
}
