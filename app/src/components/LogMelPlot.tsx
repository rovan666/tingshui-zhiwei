import { useEffect, useRef } from 'react'
import type { Alarm } from '@/types/demo'

const STOPS: Array<[number, [number, number, number]]> = [
  [0, [5, 8, 16]],
  [0.35, [12, 36, 58]],
  [0.65, [14, 116, 144]],
  [0.86, [34, 211, 238]],
  [1, [251, 191, 36]],
]

function colorAt(value: number) {
  const t = Math.max(0, Math.min(1, (value + 2) / 5))
  const index = Math.min(STOPS.length - 2, Math.floor(t * (STOPS.length - 1)))
  const [aT, a] = STOPS[index]
  const [bT, b] = STOPS[index + 1]
  const mix = (t - aT) / (bT - aT)
  return `rgb(${a.map((channel, i) => Math.round(channel + (b[i] - channel) * mix)).join(',')})`
}

export function LogMelPlot({ alarm }: { alarm: Alarm }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const context = canvas.getContext('2d')
    if (!context) return
    const { rows, cols, values } = alarm.logMel
    canvas.width = 760
    canvas.height = 240
    const cellW = canvas.width / cols
    const cellH = canvas.height / rows
    context.fillStyle = '#050810'
    context.fillRect(0, 0, canvas.width, canvas.height)
    for (let row = 0; row < rows; row += 1) {
      for (let col = 0; col < cols; col += 1) {
        context.fillStyle = colorAt(values[row * cols + col])
        context.fillRect(col * cellW, canvas.height - (row + 1) * cellH, Math.ceil(cellW), Math.ceil(cellH))
      }
    }
  }, [alarm])

  return <canvas ref={canvasRef} className="logmel-canvas" role="img" aria-label={`${alarm.id} 告警的 40 阶 log-Mel 时频特征图`} />
}
