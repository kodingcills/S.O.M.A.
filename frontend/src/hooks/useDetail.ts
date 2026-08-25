// VERBATIM from docs/specs/INTERFACE.md "Detail View Polling" — 300ms interval.
// Always clear the interval on unmount. A leaked interval polls even after
// the detail view closes — 3 requests/second × every open node ever clicked.
import { useEffect, useState } from 'react'
import type { DetailResponse } from '../types'

export function useDetail(simId: string | null): DetailResponse | null {
  const [detail, setDetail] = useState<DetailResponse | null>(null)

  useEffect(() => {
    if (!simId) {
      setDetail(null)
      return
    }
    let cancelled = false
    const poll = async () => {
      try {
        const r = await fetch(`${import.meta.env.VITE_API_URL}/detail/${simId}`)
        if (!r.ok) return
        const next = await r.json() as DetailResponse
        if (!cancelled) setDetail(next)
      } catch { /* backend starting or offline — keep previous value */ }
    }
    void poll()
    const id = setInterval(poll, 300)
    return () => { cancelled = true; clearInterval(id) }
  }, [simId])

  return detail
}
