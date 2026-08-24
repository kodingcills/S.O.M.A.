export type CompareMode = 'min' | 'max'

export function betterOf(a: number, b: number, mode: CompareMode): 'a' | 'b' | 'tie' {
  if (a === b) return 'tie'
  if (mode === 'min') return a < b ? 'a' : 'b'
  return a > b ? 'a' : 'b'
}

export function integrityMean(grid: number[][]): number {
  let sum = 0
  let n = 0
  for (const row of grid) {
    for (const v of row) {
      sum += v
      n++
    }
  }
  if (n === 0) return 0
  return Math.round((sum / n) * 1000) / 10
}

export function detectNewDamage(prev: boolean[], next: boolean[]): boolean {
  return next.some((d, i) => d && !prev[i])
}
