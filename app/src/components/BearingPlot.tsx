import type { Alarm } from '@/types/demo'

type Props = {
  alarm: Alarm
  angles: number[]
}

export function BearingPlot({ alarm, angles }: Props) {
  const size = 360
  const center = size / 2
  const radius = 126
  const point = (degrees: number, r: number) => {
    const radians = degrees * Math.PI / 180
    return [center + r * Math.cos(radians), center - r * Math.sin(radians)]
  }
  const spectrumPath = alarm.spectrumDb.map((db, index) => {
    const normalized = Math.max(0, Math.min(1, (db + 40) / 40))
    const [px, py] = point(angles[index], radius * (0.16 + normalized * 0.84))
    return `${index ? 'L' : 'M'}${px.toFixed(2)},${py.toFixed(2)}`
  }).join(' ') + ' Z'
  const estimate = point(alarm.bearingDeg, radius + 8)
  const truth = alarm.truthBearingDeg === null ? null : point(alarm.truthBearingDeg, radius + 8)

  return (
    <svg className="bearing-chart" viewBox={`0 0 ${size} ${size}`} role="img" aria-label={`CBF 方位谱，估计方位 ${alarm.bearingDeg.toFixed(1)} 度`}>
      <defs>
        <radialGradient id="bearing-fill"><stop offset="0%" stopColor="#22d3ee" stopOpacity="0.04" /><stop offset="100%" stopColor="#22d3ee" stopOpacity="0.28" /></radialGradient>
      </defs>
      <circle cx={center} cy={center} r={radius + 20} className="bearing-outer" />
      {[0.33, 0.66, 1].map((scale) => <circle key={scale} cx={center} cy={center} r={radius * scale} className="bearing-grid" />)}
      {Array.from({ length: 8 }, (_, index) => {
        const [px, py] = point(index * 45, radius)
        return <line key={index} x1={center} y1={center} x2={px} y2={py} className="bearing-grid" />
      })}
      <g className="radar-sweep">
        <line x1={center} y1={center} x2={center + radius} y2={center} className="sweep-line" />
      </g>
      <path d={spectrumPath} fill="url(#bearing-fill)" className="spectrum-shape" />
      {[0, 90, 180, 270].map((degree) => {
        const [px, py] = point(degree, radius + 32)
        return <text key={degree} x={px} y={py + 4} textAnchor="middle" className="bearing-label">{degree}°</text>
      })}
      <line x1={center} y1={center} x2={estimate[0]} y2={estimate[1]} className="estimate-line" />
      <circle cx={estimate[0]} cy={estimate[1]} r={4} className="estimate-dot" />
      {truth && <line x1={center} y1={center} x2={truth[0]} y2={truth[1]} className="truth-line" />}
      <circle cx={center} cy={center} r={4} className="array-center" />
      <text x={center} y={center - 12} textAnchor="middle" className="array-label">UCA–4</text>
    </svg>
  )
}
