export interface DrawMetrics {
  x: number
  y: number
  r: number
}

export function drawMetrics(
  col: number,
  row: number,
  radius: number,
  cellSize: number,
): DrawMetrics {
  return { x: col * cellSize, y: row * cellSize, r: radius * cellSize }
}

// Target pulse alpha per INTERFACE.md: 0.15 + 0.15*sin(t/500); reached pins high.
export function targetAlpha(nowMs: number, reached: boolean): number {
  if (reached) return 0.4
  return 0.15 + 0.15 * Math.sin(nowMs / 500)
}

export function vesselDamageAlpha(nowMs: number): number {
  return 0.6 + 0.2 * Math.sin(nowMs / 125)
}
