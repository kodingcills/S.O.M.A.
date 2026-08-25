import { useEffect, useRef } from 'react'

interface CraftaxMapProps {
  readonly tokenGrid: readonly (readonly number[])[]
  readonly direction: readonly number[]
  readonly showErrors?: boolean
  readonly errorMap?: readonly (readonly number[])[]
  readonly compact?: boolean
}

const MAP_COLUMNS = 11
const MAP_ROWS = 9

export function CraftaxMap({ tokenGrid, direction, showErrors = false, errorMap, compact = false }: CraftaxMapProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    const context = canvas?.getContext('2d')
    if (!canvas || !context) return

    const cellWidth = canvas.width / MAP_COLUMNS
    const cellHeight = canvas.height / MAP_ROWS
    context.clearRect(0, 0, canvas.width, canvas.height)

    for (let index = 0; index < MAP_COLUMNS * MAP_ROWS; index++) {
      const row = Math.floor(index / MAP_COLUMNS)
      const column = index % MAP_COLUMNS
      const token = tokenGrid[index]?.[0] ?? 0
      const hue = (token * 47 + 72) % 360
      const lightness = token === 0 ? 88 : 43 + (token % 4) * 7
      context.fillStyle = `hsl(${hue} 30% ${lightness}%)`
      context.fillRect(column * cellWidth, row * cellHeight, cellWidth, cellHeight)

      if (showErrors && errorMap) {
        const errorRow = Math.floor((row / MAP_ROWS) * errorMap.length)
        const sourceRow = errorMap[errorRow]
        const errorColumn = sourceRow ? Math.floor((column / MAP_COLUMNS) * sourceRow.length) : 0
        const error = sourceRow?.[errorColumn] ?? 0
        if (error > 0) {
          context.fillStyle = `rgba(83, 104, 165, ${Math.min(0.5, error * 0.55)})`
          context.fillRect(column * cellWidth, row * cellHeight, cellWidth, cellHeight)
        }
      }

      context.strokeStyle = 'rgba(255, 255, 255, 0.16)'
      context.strokeRect(column * cellWidth, row * cellHeight, cellWidth, cellHeight)
    }

    const facing = direction[0] ?? 0
    const arrows = ['↑', '→', '↓', '←'] as const
    context.fillStyle = 'rgba(255, 255, 255, 0.96)'
    context.font = `600 ${Math.max(16, cellHeight * 0.55)}px sans-serif`
    context.textAlign = 'center'
    context.textBaseline = 'middle'
    context.fillText(arrows[((facing % arrows.length) + arrows.length) % arrows.length], canvas.width / 2, canvas.height / 2)
  }, [direction, errorMap, showErrors, tokenGrid])

  return (
    <canvas
      ref={canvasRef}
      className={`craftax-map${compact ? ' compact' : ''}`}
      width={440}
      height={360}
      role="img"
      aria-label="Craftax symbolic map, 11 columns by 9 rows"
    />
  )
}
