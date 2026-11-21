import type { DemoData } from '@/types/demo'

type Props = {
  detection: DemoData['detection']
  duration: number
  currentTime: number
  onSeek: (time: number) => void
}

export function DetectionChart({ detection, duration, currentTime, onSeek }: Props) {
  const W = 1000
  const H = 220
  const left = 54
  const right = 18
  const top = 16
  const bottom = 34
  const innerW = W - left - right
  const innerH = H - top - bottom
  const minDb = -8
  const maxDb = 12
  const x = (time: number) => left + (time / duration) * innerW
  const y = (value: number) => top + (maxDb - Math.max(minDb, Math.min(maxDb, value))) / (maxDb - minDb) * innerH
  const path = detection.marginDb.map((value, index) =>
    `${index ? 'L' : 'M'}${x(detection.time[index]).toFixed(2)},${y(value).toFixed(2)}`,
  ).join(' ')
  const hitRanges: Array<[number, number]> = []
  let start: number | null = null
  detection.frameHit.forEach((hit, index) => {
    if (hit && start === null) start = index
    if ((!hit || index === detection.frameHit.length - 1) && start !== null) {
      hitRanges.push([detection.time[start], detection.time[index]])
      start = null
    }
  })

  const seek = (event: React.MouseEvent<SVGSVGElement>) => {
    const rect = event.currentTarget.getBoundingClientRect()
    const svgX = ((event.clientX - rect.left) / rect.width) * W
    onSeek(Math.max(0, Math.min(duration, ((svgX - left) / innerW) * duration)))
  }

  return (
    <svg className="chart-svg detection-chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="CFAR 归一化检测裕量，红色虚线为零分贝判决门限" onClick={seek}>
      <defs><clipPath id="detect-clip"><rect x={left} y={top} width={innerW} height={innerH} /></clipPath></defs>
      {hitRanges.map(([rangeStart, rangeEnd], index) => (
        <rect key={index} x={x(rangeStart)} y={top} width={Math.max(1, x(rangeEnd) - x(rangeStart))} height={innerH} className="hit-band" />
      ))}
      {[-6, 0, 6, 12].map((tick) => (
        <g key={tick}>
          <line x1={left} x2={W - right} y1={y(tick)} y2={y(tick)} className={tick === 0 ? 'threshold-line' : 'chart-grid'} />
          <text x={left - 12} y={y(tick) + 4} textAnchor="end" className={tick === 0 ? 'threshold-label' : 'chart-label'}>{tick > 0 ? `+${tick}` : tick}</text>
        </g>
      ))}
      {[0, 2, 4, 6, 8].map((tick) => <text key={tick} x={x(tick)} y={H - 10} textAnchor="middle" className="chart-label">{tick.toFixed(0)} s</text>)}
      <path d={path} className="detection-line" clipPath="url(#detect-clip)" vectorEffect="non-scaling-stroke" />
      <text x={W - right - 8} y={y(0) - 8} textAnchor="end" className="threshold-label">判决门限 0 dB</text>
      <line x1={x(currentTime)} x2={x(currentTime)} y1={top} y2={H - bottom} className="time-cursor" />
    </svg>
  )
}
