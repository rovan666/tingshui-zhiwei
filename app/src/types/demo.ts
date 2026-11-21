export type Label = 'ship' | 'human' | 'fish' | 'noise'

export interface WaveformChannel {
  id: string
  min: number[]
  max: number[]
}

export interface Alarm {
  id: string
  tStart: number
  tEnd: number
  detectionStatDb: number
  label: Label
  bearingDeg: number
  truthBearingDeg: number | null
  fBandHz: [number, number]
  windowStart: number
  windowEnd: number
  probabilities: Record<Label, number>
  spectrumDb: number[]
  logMel: {
    rows: number
    cols: number
    values: number[]
  }
}

export interface DemoData {
  meta: {
    project: string
    scenario: string
    dataMode: 'simulation'
    durationSec: number
    sampleRateHz: number
    channelCount: number
    arrayRadiusM: number
    soundSpeedMps: number
    detector: string
    doa: string
    classifier: string
    exportCommand: string
    scoreNote: string
  }
  waveform: {
    time: number[]
    channels: WaveformChannel[]
  }
  detection: {
    time: number[]
    marginDb: number[]
    frameHit: boolean[]
    thresholdDb: number
    designPfa: number
  }
  doaAnglesDeg: number[]
  alarms: Alarm[]
  truth: Array<{
    cls: Label
    t0: number
    t1: number
    bearing: number
    snr_db: number
  }>
}
