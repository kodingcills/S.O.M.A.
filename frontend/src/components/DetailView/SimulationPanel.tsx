import { useEffect, useRef, useState } from 'react'
import type { SimulationState } from '../../types'
import {
  STATUS_COMPLETED,
  STATUS_FAILED,
  STATUS_SPAWNING,
  SURFACE_BG,
  TEXT_SECONDARY,
} from '../../utils/colors'
import { drawMetrics, targetAlpha, vesselDamageAlpha } from '../../utils/panelMath'

export interface SimulationPanelProps {
  simId: string | null
  sim: SimulationState | null
  showErrors?: boolean
  errorMap?: number[][]
}

const MONO = "'JetBrains Mono', ui-monospace, Menlo, monospace"

export function SimulationPanel({ simId, sim, showErrors = false, errorMap }: SimulationPanelProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const simRef = useRef<SimulationState | null>(sim)
  const overlayRef = useRef({ showErrors, errorMap })
  const [streamBroken, setStreamBroken] = useState(false)
  simRef.current = sim
  overlayRef.current = { showErrors, errorMap }

  useEffect(() => {
    let raf = 0
    const draw = (nowMs: number) => {
      const canvas = canvasRef.current
      const ctx = canvas?.getContext('2d')
      if (canvas && ctx) {
        const W = canvas.width
        const H = canvas.height
        const CELL = W / 16
        ctx.clearRect(0, 0, W, H)

        ctx.strokeStyle = 'rgba(255,255,255,0.05)'
        for (let i = 0; i <= 16; i++) {
          ctx.beginPath()
          ctx.moveTo(i * CELL, 0)
          ctx.lineTo(i * CELL, H)
          ctx.stroke()
          ctx.beginPath()
          ctx.moveTo(0, i * CELL)
          ctx.lineTo(W, i * CELL)
          ctx.stroke()
        }

        const current = simRef.current
        if (current) {
          for (const v of current.vessels) {
            const { x, y, r } = drawMetrics(v.col, v.row, Math.max(v.radius, 0.4), CELL)
            ctx.beginPath()
            ctx.arc(x, y, r, 0, Math.PI * 2)
            if (v.damaged) {
              ctx.fillStyle = `rgba(255,59,48,${vesselDamageAlpha(nowMs)})`
              ctx.fill()
            } else {
              ctx.fillStyle = 'rgba(90,200,250,0.3)'
              ctx.strokeStyle = 'rgba(90,200,250,0.6)'
              ctx.lineWidth = 1
              ctx.fill()
              ctx.stroke()
            }
          }

          for (let r = 0; r < 16; r++) {
            for (let c = 0; c < 16; c++) {
              if (current.bleeding_mask?.[r]?.[c]) {
                ctx.fillStyle = 'rgba(255,59,48,0.25)'
                ctx.fillRect(c * CELL, r * CELL, CELL, CELL)
              }
            }
          }

          const [tc, tr] = current.surgical_target.position
          const alpha = targetAlpha(nowMs, current.surgical_target.reached)
          ctx.beginPath()
          ctx.arc(tc * CELL, tr * CELL, 18, 0, Math.PI * 2)
          ctx.fillStyle = `rgba(52,199,89,${alpha})`
          ctx.fill()
          ctx.beginPath()
          ctx.arc(tc * CELL, tr * CELL, 8, 0, Math.PI * 2)
          ctx.fillStyle = `rgba(52,199,89,${current.surgical_target.reached ? 0.9 : 0.7})`
          ctx.fill()

          const { showErrors: show, errorMap: em } = overlayRef.current
          if (show && em) {
            ctx.lineWidth = 1
            for (let r = 0; r < 16; r++) {
              for (let c = 0; c < 16; c++) {
                const err = em[r]?.[c] ?? 0
                if (err > 0.3) {
                  ctx.strokeStyle = `rgba(255,214,0,${err * 0.8})`
                  ctx.strokeRect(c * CELL, r * CELL, CELL, CELL)
                }
              }
            }
          }
        }
      }
      raf = requestAnimationFrame(draw)
    }
    raf = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(raf)
  }, [])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', padding: 10, boxSizing: 'border-box' }}>
      <div style={{ fontFamily: MONO, fontSize: 10, color: TEXT_SECONDARY, letterSpacing: '0.1em', marginBottom: 6 }}>
        SIMULATION · {simId ?? '—'}
      </div>
      <div style={{ position: 'relative', flex: 1, minHeight: 0, background: SURFACE_BG }}>
        {!streamBroken && simId ? (
          <img
            src={`${import.meta.env.VITE_API_URL}/sim/${simId}/stream`}
            alt="Live simulation"
            onError={() => setStreamBroken(true)}
            style={{
              width: '100%', height: '100%', objectFit: 'contain',
              imageRendering: 'pixelated', display: 'block',
            }}
          />
        ) : (
          <div style={{
            width: '100%', height: '100%', display: 'flex', alignItems: 'center',
            justifyContent: 'center', color: TEXT_SECONDARY, fontFamily: MONO, fontSize: 11,
          }}>
            stream offline
          </div>
        )}
        <canvas
          ref={canvasRef}
          width={640}
          height={480}
          style={{
            position: 'absolute', inset: 0, width: '100%', height: '100%',
            pointerEvents: 'none',
          }}
        />
      </div>
      <div style={{
        display: 'flex', gap: 14, alignItems: 'center', marginTop: 8,
        fontFamily: MONO, fontSize: 10, color: TEXT_SECONDARY,
      }}>
        <span>STEP {sim?.step_id ?? 0}</span>
        <span>REWARD {(sim?.reward ?? 0).toFixed(2)}</span>
        <span>DOF {sim?.active_dof ?? 1}</span>
        <span
          style={{
            marginLeft: 'auto',
            color:
              sim?.task_failed === true ? STATUS_FAILED
              : sim?.task_complete === true ? STATUS_COMPLETED
              : STATUS_SPAWNING,
          }}
        >
          {sim?.task_failed === true
            ? 'TASK FAILED'
            : sim?.task_complete === true
              ? 'TASK COMPLETE'
              : 'ACTIVE'}
        </span>
      </div>
    </div>
  )
}
