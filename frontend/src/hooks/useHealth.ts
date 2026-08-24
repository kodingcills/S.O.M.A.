// VERBATIM from docs/specs/INTERFACE.md "Health Polling" — 1000ms interval.
import { useEffect, useState } from 'react'
import type { HealthResponse } from '../types'

export function useHealth(): HealthResponse | null {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  useEffect(() => {
    const id = setInterval(async () => {
      try {
        const r = await fetch(`${import.meta.env.VITE_API_URL}/health`)
        setHealth(await r.json())
      } catch { /* ignore — top bar shows stale values */ }
    }, 1000)
    return () => clearInterval(id)
  }, [])
  return health
}
