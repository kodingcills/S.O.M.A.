import { describe, it, expect } from 'vitest'
import { betterOf, detectNewDamage, integrityMean } from '../compareStats'

describe('betterOf', () => {
  it('respects min mode', () => {
    expect(betterOf(3, 7, 'min')).toBe('a')
    expect(betterOf(9, 2, 'min')).toBe('b')
  })

  it('respects max mode', () => {
    expect(betterOf(3, 7, 'max')).toBe('b')
    expect(betterOf(9, 2, 'max')).toBe('a')
  })

  it('ties are neutral', () => {
    expect(betterOf(5, 5, 'min')).toBe('tie')
    expect(betterOf(5, 5, 'max')).toBe('tie')
  })
})

describe('integrityMean', () => {
  it('uniform 0.5 grid → 50.0', () => {
    const grid = Array.from({ length: 16 }, () => Array<number>(16).fill(0.5))
    expect(integrityMean(grid)).toBeCloseTo(50.0, 5)
  })

  it('ones-with-hole averages correctly at 1 decimal', () => {
    const grid = Array.from({ length: 16 }, () => Array<number>(16).fill(1))
    grid[0][0] = 0 // 255 ones + one zero over 256 cells = 99.609…
    expect(integrityMean(grid)).toBeCloseTo(99.6, 1)
  })
})

describe('detectNewDamage', () => {
  it('fires only on false→true transitions', () => {
    expect(detectNewDamage([false, false], [false, true])).toBe(true)
    // already-damaged stays non-transitioning; unchanged flags stay quiet
    expect(detectNewDamage([false, true], [false, true])).toBe(false)
    expect(detectNewDamage([true, false], [true, false])).toBe(false)
    expect(detectNewDamage([true], [true])).toBe(false)
  })
})
