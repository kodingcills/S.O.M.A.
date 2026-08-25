// VERBATIM from docs/specs/INTERFACE.md "DATA ARCHITECTURE — WebSocket Hook".
import { useCallback, useEffect, useReducer, useRef } from 'react'
import type { EventLogEntry } from '../types'

export function useEventLog(): EventLogEntry[] {
  const [events, dispatch] = useReducer(eventsReducer, [])
  const bufferRef  = useRef<EventLogEntry[]>([])
  const timerRef   = useRef<ReturnType<typeof setTimeout> | null>(null)
  const lastTsRef  = useRef<number>(0)
  const retryRef   = useRef<number>(0)
  const wsRef      = useRef<WebSocket | null>(null)

  const flush = useCallback(() => {
    if (bufferRef.current.length === 0) return
    dispatch({ type: 'APPEND', events: bufferRef.current })
    bufferRef.current = []
    timerRef.current  = null
  }, [])

  const scheduleFlush = useCallback(() => {
    // BATCH UPDATE: collect events for 100ms, then flush once.
    // Without batching, the connect-time replay burst (potentially
    // 1000+ events in <1s) triggers 1000 re-renders and freezes
    // the browser for several seconds.
    if (timerRef.current) return
    timerRef.current = setTimeout(flush, 100)
  }, [flush])

  useEffect(() => {
    let cancelled = false
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null
    let restTimer: ReturnType<typeof setInterval> | null = null

    function connect() {
      if (cancelled) return
      const ws = new WebSocket(import.meta.env.VITE_WS_URL)
      wsRef.current = ws

      ws.onmessage = (e) => {
        let msg: { type?: string; data?: EventLogEntry }
        try { msg = JSON.parse(e.data) as { type?: string; data?: EventLogEntry } }
        catch { return }
        if (msg.type === 'event') {
          if (!msg.data) return
          bufferRef.current.push(msg.data)
          lastTsRef.current = Math.max(lastTsRef.current, msg.data.timestamp)
          scheduleFlush()
        }
        if (msg.type === 'ping') {
          // The backend heartbeat endpoint expects the literal text frame.
          ws.send('pong')
        }
      }

      ws.onclose = () => {
        if (cancelled) return
        if (retryRef.current < 5) {
          // Exponential backoff: 1s, 2s, 4s, 8s, 16s
          const delay = Math.min(1000 * 2 ** retryRef.current, 16000)
          retryRef.current++
          reconnectTimer = setTimeout(connect, delay)
        } else {
          startRestFallback()
        }
      }

      ws.onopen = () => { retryRef.current = 0 }
    }

    function startRestFallback() {
      if (restTimer) return
      restTimer = setInterval(async () => {
        try {
          const r = await fetch(`${import.meta.env.VITE_API_URL}/events?since=${lastTsRef.current}`)
          if (!r.ok) return
          const es = await r.json() as EventLogEntry[]
          if (es.length > 0) {
            dispatch({ type: 'APPEND', events: es })
            lastTsRef.current = es[es.length - 1].timestamp
          }
        } catch { /* backend still unavailable */ }
      }, 500)
    }

    connect()
    return () => {
      cancelled = true
      wsRef.current?.close()
      if (timerRef.current) clearTimeout(timerRef.current)
      if (reconnectTimer) clearTimeout(reconnectTimer)
      if (restTimer) clearInterval(restTimer)
    }
  }, [scheduleFlush])

  return events
}

function eventsReducer(
  state: EventLogEntry[],
  action: { type: 'APPEND'; events: EventLogEntry[] }
): EventLogEntry[] {
  if (action.type !== 'APPEND') return state
  // WebSocket reconnects replay full history. Preserve one copy per event id.
  const seen = new Set(state.map(event => event.id))
  const incoming = action.events.filter(event => !seen.has(event.id))
  const next = [...state, ...incoming]
  // Cap at 10,000 to prevent memory growth over long sessions
  return next.length > 10_000 ? next.slice(-10_000) : next
}
