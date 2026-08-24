import type { EventLogEntry } from '../types'

export interface CellRef {
  row: number
  col: number
}

// Quadrant slices match backend BeliefState.REGION_SLICE_MAP exactly.
const QUADRANTS: Record<string, [number, number, number, number]> = {
  upper_left: [0, 8, 0, 8],
  upper_right: [0, 8, 8, 16],
  lower_left: [8, 16, 0, 8],
  lower_right: [8, 16, 8, 16],
}

function clip(v: number, lo: number, hi: number): number {
  return Math.min(Math.max(v, lo), hi)
}

// Mirrors backend int(np.clip(pos*15, 1, 14)) truncation semantics.
function gridCenter(v: number): number {
  return Math.floor(clip(v * 15, 1, 14))
}

export function regionToCells(
  region: string,
  eePos?: [number, number],
  targetPos?: [number, number],
): CellRef[] {
  const quad = QUADRANTS[region]
  if (quad) {
    const [r0, r1, c0, c1] = quad
    const cells: CellRef[] = []
    for (let r = r0; r < r1; r++) {
      for (let c = c0; c < c1; c++) cells.push({ row: r, col: c })
    }
    return cells
  }

  const pos =
    region === 'tool_tissue_boundary'
      ? eePos
      : region === 'surgical_target_vicinity'
        ? targetPos
        : undefined
  if (!pos) return []

  const row = gridCenter(pos[1])
  const col = gridCenter(pos[0])
  const cells: CellRef[] = []
  for (let dr = -1; dr <= 1; dr++) {
    for (let dc = -1; dc <= 1; dc++) cells.push({ row: row + dr, col: col + dc })
  }
  return cells
}

// Last qualifying completion wins — matches the demo's most-recent-flash rule.
export function pickQualifyingCompletion(
  events: EventLogEntry[],
): EventLogEntry | null {
  let found: EventLogEntry | null = null
  for (const e of events) {
    if (e.event_type !== 'agent_completed') continue
    if (e.error_before == null || e.error_after == null) continue
    if (e.error_before - e.error_after <= 0.05) continue
    found = e
  }
  return found
}
