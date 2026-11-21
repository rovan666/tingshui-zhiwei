import { useEffect, useMemo, useState } from 'react'
import { Activity, AlertTriangle, Download, Pause, Play, Radio, RotateCcw, Waves } from 'lucide-react'
import { BearingPlot } from '@/components/BearingPlot'
import { DetectionChart } from '@/components/DetectionChart'
import { LogMelPlot } from '@/components/LogMelPlot'
import { WaveformChart } from '@/components/WaveformChart'
import { useDemoData } from '@/hooks/useDemoData'
import type { Alarm, Label } from '@/types/demo'
import './App.css'

const LABEL_INFO: Record<Label, { zh: string; color: string }> = {
  ship: { zh: '船舶噪声', color: '#22d3ee' },
  human: { zh: '人员拍水', color: '#fbbf24' },
  fish: { zh: '鱼类活动', color: '#34d399' },
  noise: { zh: '背景噪声', color: '#94a3b8' },
}

function angleError(alarm: Alarm) {
  if (alarm.truthBearingDeg === null) return null
  const difference = Math.abs(alarm.bearingDeg - alarm.truthBearingDeg) % 360
  return Math.min(difference, 360 - difference)
}

export default function App() {
  const { data, error, retry } = useDemoData()
  const [currentTime, setCurrentTime] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [selectedId, setSelectedId] = useState('A-01')
  const [activeChannels, setActiveChannels] = useState(['CH1', 'CH2', 'CH3', 'CH4'])

  useEffect(() => {
    if (!playing || !data) return
    let animationFrame = 0
    let previous: number | null = null
    const tick = (now: number) => {
      if (previous !== null) {
        const elapsed = (now - previous) / 1000
        setCurrentTime((time) => {
          const next = time + elapsed * speed
          if (next >= data.meta.durationSec) {
            setPlaying(false)
            return data.meta.durationSec
          }
          return next
        })
      }
      previous = now
      animationFrame = requestAnimationFrame(tick)
    }
    animationFrame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(animationFrame)
  }, [playing, speed, data])

  const selectedAlarm = useMemo(() => {
    const playbackAlarm = playing
      ? data?.alarms.find((alarm) => currentTime >= alarm.tStart && currentTime <= alarm.tEnd)
      : undefined
    return playbackAlarm
      ?? data?.alarms.find((alarm) => alarm.id === selectedId)
      ?? data?.alarms[0]
  }, [data, selectedId, playing, currentTime])

  if (error) {
    return (
      <main className="app-state">
        <AlertTriangle size={28} />
        <h1>演示数据读取失败</h1>
        <p>{error}</p>
        <button className="primary-button" onClick={retry}>重新读取</button>
      </main>
    )
  }

  if (!data || !selectedAlarm) {
    return (
      <main className="app-state">
        <Radio size={30} className="loading-pulse" />
        <h1>正在装载仿真场景</h1>
        <p>读取检测、方位与分类的真实运行输出…</p>
      </main>
    )
  }

  const detectionIndex = Math.max(0, Math.min(
    data.detection.time.length - 1,
    Math.round(currentTime / data.meta.durationSec * (data.detection.time.length - 1)),
  ))
  const frameHit = data.detection.frameHit[detectionIndex]
  const currentMargin = data.detection.marginDb[detectionIndex]

  const seek = (time: number) => {
    setPlaying(false)
    setCurrentTime(time)
  }
  const selectAlarm = (alarm: Alarm) => {
    setPlaying(false)
    setSelectedId(alarm.id)
    setCurrentTime((alarm.tStart + alarm.tEnd) / 2)
  }
  const togglePlayback = () => {
    if (currentTime >= data.meta.durationSec) setCurrentTime(0)
    setPlaying((value) => !value)
  }
  const toggleChannel = (id: string) => {
    setActiveChannels((channels) => channels.includes(id)
      ? channels.filter((channel) => channel !== id)
      : [...channels, id])
  }

  return (
    <div className="workstation">
      <header className="topbar">
        <div className="brand-block">
          <div className="brand-mark" aria-hidden="true"><Waves size={25} /></div>
          <div>
            <div className="brand-title">听水<span>·</span>知微</div>
            <div className="brand-subtitle">UNDERWATER ACOUSTIC INTELLIGENCE</div>
          </div>
        </div>
        <div className="topbar-actions">
          <div className="system-status"><span className={playing ? 'status-dot live' : 'status-dot'} />{playing ? '链路回放中' : '链路已就绪'}</div>
          <span className="simulation-badge">SIMULATED · 仿真数据</span>
          <a className="export-button" href={`${import.meta.env.BASE_URL}demo-data.json`} download="tingshui-demo-data.json">
            <Download size={15} />导出数据
          </a>
        </div>
      </header>

      <section className="scenario-bar" aria-label="当前场景与系统参数">
        <div className="scenario-name"><span>当前场景</span>{data.meta.scenario}</div>
        <dl className="system-specs">
          <div><dt>阵列</dt><dd>四元均匀圆阵</dd></div>
          <div><dt>采样率</dt><dd>{data.meta.sampleRateHz / 1000} kHz</dd></div>
          <div><dt>检测</dt><dd>{data.meta.detector}</dd></div>
          <div><dt>定位</dt><dd>{data.meta.doa}</dd></div>
          <div><dt>分类</dt><dd>{data.meta.classifier}</dd></div>
        </dl>
      </section>

      <main className="dashboard-grid">
        <div className="primary-column">
          <section className="panel waveform-panel" aria-labelledby="waveform-title">
            <div className="panel-heading">
              <div><span className="panel-index">01 / SIGNAL</span><h2 id="waveform-title">四通道水听器波形</h2></div>
              <div className="channel-toggles" aria-label="通道显示开关">
                {data.waveform.channels.map((channel, index) => {
                  const active = activeChannels.includes(channel.id)
                  return <button key={channel.id} className={active ? 'channel-toggle active' : 'channel-toggle'} onClick={() => toggleChannel(channel.id)} disabled={active && activeChannels.length === 1} aria-pressed={active} title={active && activeChannels.length === 1 ? '至少保留一个通道' : `切换 ${channel.id}`}><span style={{ backgroundColor: ['#22d3ee', '#34d399', '#fbbf24', '#f87171'][index] }} />{channel.id}</button>
                })}
              </div>
            </div>
            <WaveformChart waveform={data.waveform} alarms={data.alarms} duration={data.meta.durationSec} currentTime={currentTime} activeChannels={activeChannels} onSeek={seek} />
            <div className="transport-bar">
              <button className="play-button" onClick={togglePlayback} aria-label={playing ? '暂停回放' : '开始回放'}>{playing ? <Pause size={17} fill="currentColor" /> : <Play size={17} fill="currentColor" />}{playing ? '暂停' : '播放'}</button>
              <button className="icon-button" onClick={() => seek(0)} aria-label="回到起点" title="回到起点"><RotateCcw size={16} /></button>
              <span className="time-readout">{currentTime.toFixed(2)} <span>/ {data.meta.durationSec.toFixed(2)} s</span></span>
              <input className="timeline" type="range" min={0} max={data.meta.durationSec} step={0.01} value={currentTime} onChange={(event) => seek(Number(event.target.value))} aria-label="回放时间" style={{ '--timeline-progress': `${currentTime / data.meta.durationSec * 100}%` } as React.CSSProperties} />
              <div className="speed-control" aria-label="回放倍速">{[1, 2, 4].map((value) => <button key={value} onClick={() => setSpeed(value)} className={speed === value ? 'active' : ''} aria-pressed={speed === value}>{value}×</button>)}</div>
            </div>
          </section>

          <section className="panel detection-panel" aria-labelledby="detection-title">
            <div className="panel-heading compact">
              <div><span className="panel-index">02 / DETECT</span><h2 id="detection-title">CFAR 自适应异常检测</h2></div>
              <div className={frameHit ? 'detector-state hit' : 'detector-state'}><span />{frameHit ? '原始帧命中' : '背景监听'}<strong>{currentMargin.toFixed(2)} dB</strong></div>
            </div>
            <DetectionChart detection={data.detection} duration={data.meta.durationSec} currentTime={currentTime} onSeek={seek} />
            <div className="chart-footnote"><span><i className="legend-line cyan" />归一化检测裕量</span><span><i className="legend-line red" />0 dB 判决门限</span><span><i className="legend-box" />原始帧命中区间</span><span className="footnote-right">设计帧级 Pfa = {data.detection.designPfa}</span></div>
          </section>

          <section className="panel alarms-panel" aria-labelledby="alarms-title">
            <div className="panel-heading compact">
              <div><span className="panel-index">05 / EVENTS</span><h2 id="alarms-title">事件告警记录</h2></div>
              <span className="alarm-count">{data.alarms.length} 段成形告警</span>
            </div>
            <div className="table-wrap">
              <table>
                <thead><tr><th>事件</th><th>告警时间</th><th>识别类型</th><th>估计 / 真实方位</th><th>检测裕量</th><th>模型得分</th><th>测向误差</th></tr></thead>
                <tbody>{data.alarms.map((alarm) => {
                  const selected = alarm.id === selectedAlarm.id
                  return <tr key={alarm.id} className={selected ? 'selected' : ''} onClick={() => selectAlarm(alarm)}>
                    <td><button className="event-select" onClick={() => selectAlarm(alarm)} aria-label={`查看 ${alarm.id} 告警`}><span className={selected ? 'event-dot selected' : 'event-dot'} />{alarm.id}</button></td>
                    <td className="mono">{alarm.tStart.toFixed(2)}–{alarm.tEnd.toFixed(2)} s</td>
                    <td><span className="class-label" style={{ color: LABEL_INFO[alarm.label].color }}>{LABEL_INFO[alarm.label].zh}</span><small>{alarm.label}</small></td>
                    <td className="mono">{alarm.bearingDeg.toFixed(1)}° <span>/ {alarm.truthBearingDeg?.toFixed(1) ?? '—'}°</span></td>
                    <td className="mono">{alarm.detectionStatDb.toFixed(2)} dB</td>
                    <td className="mono">{(Math.max(...Object.values(alarm.probabilities)) * 100).toFixed(1)}%</td>
                    <td className="mono">{angleError(alarm)?.toFixed(1) ?? '—'}°</td>
                  </tr>
                })}</tbody>
              </table>
            </div>
          </section>
        </div>

        <aside className="secondary-column">
          <section className="panel bearing-panel" aria-labelledby="bearing-title">
            <div className="panel-heading compact">
              <div><span className="panel-index">03 / LOCALIZE</span><h2 id="bearing-title">CBF 方位估计</h2></div>
              <span className="event-tag">{selectedAlarm.id}</span>
            </div>
            <BearingPlot alarm={selectedAlarm} angles={data.doaAnglesDeg} />
            <div className="bearing-readout">
              <div><span>估计方位</span><strong>{selectedAlarm.bearingDeg.toFixed(1)}<small>°</small></strong></div>
              <div><span>真实方位</span><strong>{selectedAlarm.truthBearingDeg?.toFixed(1) ?? '—'}<small>°</small></strong></div>
              <div><span>测向误差</span><strong>{angleError(selectedAlarm)?.toFixed(1) ?? '—'}<small>°</small></strong></div>
            </div>
            <p className="panel-note">分析频带 {(selectedAlarm.fBandHz[0] / 1000).toFixed(2)}–{(selectedAlarm.fBandHz[1] / 1000).toFixed(2)} kHz · 1 s 告警分析窗</p>
          </section>

          <section className="panel classification-panel" aria-labelledby="classification-title">
            <div className="panel-heading compact">
              <div><span className="panel-index">04 / CLASSIFY</span><h2 id="classification-title">log-Mel 智能分类</h2></div>
              <Activity size={17} className="heading-icon" />
            </div>
            <div className="classification-result">
              <div><span>识别类型</span><strong style={{ color: LABEL_INFO[selectedAlarm.label].color }}>{LABEL_INFO[selectedAlarm.label].zh}</strong></div>
              <span className="model-code">{selectedAlarm.label.toUpperCase()}</span>
            </div>
            <div className="logmel-wrap"><LogMelPlot alarm={selectedAlarm} /><div className="logmel-axis"><span>40 Mel</span><span>0</span></div><div className="logmel-time">1 s 分析窗 · 时间 →</div></div>
            <div className="probability-list">
              {(Object.keys(LABEL_INFO) as Label[]).map((label) => (
                <div className="probability-row" key={label}>
                  <span>{LABEL_INFO[label].zh}</span>
                  <div className="probability-track"><div style={{ width: `${selectedAlarm.probabilities[label] * 100}%`, backgroundColor: LABEL_INFO[label].color }} /></div>
                  <strong>{(selectedAlarm.probabilities[label] * 100).toFixed(1)}%</strong>
                </div>
              ))}
            </div>
            <p className="panel-note">未校准模型得分，不等同于现实置信度</p>
          </section>
        </aside>

      </main>

      <footer className="footer-note"><span><i className="status-dot" />固定随机种子 11 · 可复现实验</span><p>当前结果仅用于算法链路仿真验证，尚未接入真实水听器或公开水声数据集。</p><code>{data.meta.exportCommand}</code></footer>
    </div>
  )
}
