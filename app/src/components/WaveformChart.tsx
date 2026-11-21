import type { Alarm, DemoData } from '@/types/demo'

const COLORS = ['#22d3ee', '#34d399', '#fbbf24', '#f87171']

type Props = {
  waveform: DemoData['waveform']
  alarms: Alarm[]
  duration: number
  currentTime: number
  activeChannels: string[]
  onSeek: (time: number) => void
}

export function WaveformChart({ waveform, alarms, duration, currentTime, activeChannels, onSeek }: Props) {
  const W = 1000
  const H = 324
  const left = 54
  const right = 18
  const top = 14
  const bottom = 34
  const innerW = W - left - right
  const innerH = H - top - bottom
  const laneH = innerH / waveform.channels.length
  const amplitude = Math.max(1, ...waveform.channels.flatMap((channel) =>
    [...channel.min, ...channel.max].map((value) => Math.abs(value)),
  )) * 1.04
  const x = (time: number) => left + (time / duration) * innerW

  const seek = (event: React.MouseEvent<SVGSVGElement>) => {
    const rect = event.currentTarget.getBoundingClientRect()
    const svgX = ((event.clientX - rect.left) / rect.width) * W
    onSeek(Math.max(0, Math.min(duration, ((svgX - left) / innerW) * duration)))
  }

  return (
    <svg className="chart-svg waveform-chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="四通道真实仿真波形包络，点击可定位时间" onClick={seek}>
      <defs>
        <clipPath id="wave-clip"><rect x={left} y={top} width={innerW} height={innerH} /></clipPath>
      </defs>
      {alarms.map((alarm) => (
        <rect key={alarm.id} x={x(alarm.tStart)} y={top} width={Math.max(1, x(alarm.tEnd) - x(alarm.tStart))} height={innerH} className={`alarm-band alarm-${alarm.label}`} />
      ))}
      {waveform.channels.map((channel, index) => {
        const centerY = top + laneH * (index + 0.5)
        const y = (value: number) => centerY - (value / amplitude) * laneH * 0.4
        const visible = activeChannels.includes(channel.id)
        const upper = channel.max.map((value, i) => `${i ? 'L' : 'M'}${x(waveform.time[i]).toFixed(2)},${y(value).toFixed(2)}`).join(' ')
        const lower = [...channel.min].reverse().map((value, reverseIndex) => {
          const i = channel.min.length - 1 - reverseIndex
          return `L${x(waveform.time[i]).toFixed(2)},${y(value).toFixed(2)}`
        }).join(' ')
        return (
          <g key={channel.id} opacity={visible ? 1 : 0.08}>
            <line x1={left} x2={W - right} y1={centerY} y2={centerY} className="chart-grid" />
            <text x={left - 12} y={centerY + 4} textAnchor="end" className="chart-label">{channel.id}</text>
            <path d={`${upper} ${lower} Z`} fill={COLORS[index]} fillOpacity={0.2} stroke={COLORS[index]} strokeWidth={0.75} clipPath="url(#wave-clip)" vectorEffect="non-scaling-stroke" />
          </g>
        )
      })}
      {[0, 2, 4, 6, 8].map((tick) => (
        <g key={tick}>
          <line x1={x(tick)} x2={x(tick)} y1={top} y2={H - bottom} className="chart-grid" />
          <text x={x(tick)} y={H - 10} textAnchor="middle" className="chart-label">{tick.toFixed(0)} s</text>
        </g>
      ))}
      <line x1={x(currentTime)} x2={x(currentTime)} y1={top} y2={H - bottom} className="time-cursor" />
      <circle cx={x(currentTime)} cy={top + 2} r={3} className="time-cursor-dot" />
    </svg>
  )
}
