// VERBATIM from docs/specs/INTERFACE.md "Health Polling" — 1000ms interval.
import { useEffect, useState } from 'react'
import type { HealthResponse } from '../types'

export function useHealth(): HealthResponse | null {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  useEffect(() => {
    let cancelled = false
    const poll = async () => {
      try {
        const r = await fetch(`${import.meta.env.VITE_API_URL}/health`)
        if (!r.ok) return
        const next = await r.json() as HealthResponse
        if (!cancelled) setHealth(next)
      } catch { /* ignore — top bar shows stale values */ }
    }
    void poll()
    const id = setInterval(poll, 1000)
    return () => { cancelled = true; clearInterval(id) }
  }, [])
  return health
}
