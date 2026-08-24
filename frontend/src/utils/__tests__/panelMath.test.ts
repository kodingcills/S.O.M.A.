import { describe, it, expect } from 'vitest'
import { drawMetrics, targetAlpha, vesselDamageAlpha } from '../panelMath'

describe('drawMetrics', () => {
  it('maps grid units to pixel coordinates linearly', () => {
    const m = drawMetrics(4, 8, 2, 32)
    expect(m).toEqual({ x: 128, y: 256, r: 64 })
  })
})

describe('targetAlpha', () => {
  it('pulses within [0, 0.3] when not reached (spec: 0.15 + 0.15·sin)', () => {
    for (let t = 0; t < 5000; t += 37) {
      const a = targetAlpha(t, false)
      expect(a).toBeGreaterThanOrEqual(-1e-9)
      expect(a).toBeLessThanOrEqual(0.3 + 1e-9)
    }
  })

  it('pins to 0.4 when reached', () => {
    expect(targetAlpha(0, true)).toBe(0.4)
    expect(targetAlpha(999_999, true)).toBe(0.4)
  })
})

describe('vesselDamageAlpha', () => {
  it('oscillates within [0.4, 0.8]', () => {
    for (let t = 0; t < 2000; t += 13) {
      const a = vesselDamageAlpha(t)
      expect(a).toBeGreaterThanOrEqual(0.4 - 1e-9)
      expect(a).toBeLessThanOrEqual(0.8 + 1e-9)
    }
  })
})
