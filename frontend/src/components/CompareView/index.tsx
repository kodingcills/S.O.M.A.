import { useEffect, useRef, useState } from 'react'
import type { SimulationState } from '../../types'
import { useDetail } from '../../hooks/useDetail'
import {
  PANEL_BORDER,
  STATUS_COMPLETED,
  SURFACE_BG,
  TEXT_ERROR,
  TEXT_SECONDARY,
} from '../../utils/colors'
import { betterOf, detectNewDamage, integrityMean } from '../../utils/compareStats'

const MONO = "'JetBrains Mono', ui-monospace, Menlo, monospace"
const API = import.meta.env.VITE_API_URL as string

interface SideProps {
  simId: string
  title: string
  accent: string
}

function StreamSide({ simId, title, accent }: SideProps) {
  const [broken, setBroken] = useState(false)
  return (
    <div style={{ display: 'flex', flexDirection: 'column', flex: 1, minWidth: 0 }}>
      <div style={{
        display: 'flex', alignItems: 'center', gap: 8, padding: '6px 10px',
        fontFamily: MONO, fontSize: 11, color: TEXT_SECONDARY,
        borderLeft: `3px solid ${accent}`, background: SURFACE_BG,
      }}>
        {title}
        <span style={{ marginLeft: 'auto' }}>{simId}</span>
      </div>
      <div style={{ position: 'relative', flex: 1, minHeight: 0 }}>
        {!broken ? (
          <img
            src={`${API}/sim/${simId}/stream`}
            alt={`${title} live stream`}
            onError={() => setBroken(true)}
            style={{
              width: '100%', height: '100%', objectFit: 'contain',
              imageRendering: 'pixelated', display: 'block',
            }}
          />
        ) : (
          <div style={{
            height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center',
            color: TEXT_SECONDARY, fontFamily: MONO, fontSize: 11,
          }}>
            stream offline
          </div>
        )}
      </div>
    </div>
  )
}

interface ToastState {
  text: string
  seq: number
}

export function CompareView() {
  const primary = useDetail('primary')
  const comparison = useDetail('comparison')
  const [toast, setToast] = useState<ToastState | null>(null)
  const prevVessels = useRef<Record<string, boolean[]>>({})

  useEffect(() => {
    for (const simId of ['primary', 'comparison'] as const) {
      const sim: SimulationState | null =
        simId === 'primary'
          ? (primary?.simulation_state ?? null)
          : (comparison?.simulation_state ?? null)
      if (!sim?.vessels) continue
      const nextFlags = sim.vessels.map((v) => v.damaged)
      const prev = prevVessels.current[simId]
      if (prev && detectNewDamage(prev, nextFlags)) {
        setToast({
          text:
            simId === 'primary'
              ? 'SOMA: vessel damaged'
              : 'Reactive agent: vessel damaged',
          seq: Date.now(),
        })
      }
      prevVessels.current[simId] = nextFlags
    }
  }, [primary, comparison])

  useEffect(() => {
    if (!toast) return
    const id = setTimeout(() => setToast(null), 3000)
    return () => clearTimeout(id)
  }, [toast])

  const pSim = primary?.simulation_state ?? null
  const cSim = comparison?.simulation_state ?? null
  const pTissue = pSim ? integrityMean(pSim.tissue_integrity) : 0
  const cTissue = cSim ? integrityMean(cSim.tissue_integrity) : 0
  const pDamage = pSim ? pSim.vessels.filter((v) => v.damaged).length : 0
  const cDamage = cSim ? cSim.vessels.filter((v) => v.damaged).length : 0

  interface StatSpec {
    label: string
    a: number
    b: number
    mode: 'min' | 'max'
    fmt: (v: number) => string
  }
  const stats: StatSpec[] = [
    {
      label: 'STEPS', a: pSim?.step_id ?? 0, b: cSim?.step_id ?? 0,
      mode: 'min', fmt: (v) => String(v),
    },
    {
      label: 'TISSUE', a: pTissue, b: cTissue, mode: 'max',
      fmt: (v) => `${v.toFixed(1)}%`,
    },
    {
      label: 'VESSELS', a: pDamage, b: cDamage, mode: 'min',
      fmt: (v) => (v === 0 ? `${v} ✓` : `${v} ✗`),
    },
  ]

  return (
    <div style={{
      display: 'flex', flexDirection: 'column', height: '100%',
      padding: 12, boxSizing: 'border-box', gap: 10, position: 'relative',
    }}>
      <div style={{ display: 'flex', gap: 12, flex: 1, minHeight: 0 }}>
        <StreamSide simId="primary" title="SOMA (MPC + WORLD MODEL)" accent="#34C759" />
        <StreamSide simId="comparison" title="REACTIVE AGENT (NO WM)" accent={TEXT_ERROR} />
      </div>

      <div style={{
        display: 'flex', justifyContent: 'center', gap: 28,
        fontFamily: MONO, fontSize: 11, padding: '8px 0',
        borderTop: `1px solid ${PANEL_BORDER}`,
      }}>
        {stats.map((s) => {
          const winner = betterOf(s.a, s.b, s.mode)
          return (
            <span key={s.label} style={{ color: TEXT_SECONDARY }}>
              {s.label}:{' '}
              <span style={{
                fontWeight: winner === 'a' && s.a !== s.b ? 700 : 400,
                color: winner === 'a' && s.a !== s.b ? STATUS_COMPLETED : TEXT_SECONDARY,
              }}>
                {s.fmt(s.a)}
              </span>
              {' vs '}
              <span style={{
                fontWeight: winner === 'b' && s.a !== s.b ? 700 : 400,
                color: winner === 'b' && s.a !== s.b ? STATUS_COMPLETED : TEXT_SECONDARY,
              }}>
                {s.fmt(s.b)}
              </span>
            </span>
          )
        })}
      </div>

      {toast && (
        <div style={{
          position: 'absolute', bottom: 64, left: '50%', transform: 'translateX(-50%)',
          background: 'rgba(255,59,48,0.1)', border: `1px solid ${TEXT_ERROR}`,
          color: TEXT_ERROR, fontFamily: MONO, fontSize: 11,
          padding: '6px 14px', borderRadius: 6,
        }} data-testid="damage-toast">
          {toast.text}
        </div>
      )}
    </div>
  )
}
