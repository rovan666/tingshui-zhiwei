# -*- coding: utf-8 -*-
"""把当前仿真流水线的真实输出导出为可视化工作台数据。

运行：python -m src.viz.export_demo
输出：app/public/demo-data.json
"""
import json
import pickle
from pathlib import Path

import numpy as np

from run_pipeline import EVENTS, FS, R, C, add_source, analysis_window, select_doa_band
from sim.sources import GENERATORS
from src.classify.features import logmel
from src.detect.cfar import cfar_detect
from src.doa.cbf import cbf_spectrum

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "app" / "public" / "demo-data.json"
LABELS = ["ship", "human", "fish", "noise"]


def rounded(values, digits=4):
    """压缩 JSON 体积，不改变图表所使用的数据顺序。"""
    return [round(float(v), digits) for v in values]


def waveform_envelope(x_all, n_bins=1600):
    """按时间桶导出真实采样的最小/最大包络，而不是重新生成波形。"""
    edges = np.linspace(0, x_all.shape[1], n_bins + 1, dtype=int)
    times = (edges[:-1] + edges[1:]) / (2.0 * FS)
    channels = []
    for index, x in enumerate(x_all):
        mins, maxs = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mins.append(float(np.min(x[lo:hi])))
            maxs.append(float(np.max(x[lo:hi])))
        channels.append({
            "id": f"CH{index + 1}",
            "min": rounded(mins),
            "max": rounded(maxs),
        })
    return {"time": rounded(times, 5), "channels": channels}


def main():
    rng = np.random.default_rng(11)
    n = int(8.0 * FS)
    x_all = 0.5 * rng.standard_normal((4, n))
    for event in EVENTS:
        sig = GENERATORS[event["cls"]](
            duration=event["t1"] - event["t0"], rng=rng
        )
        add_source(x_all, sig, event["bearing"], event["t0"], event["snr_db"], rng)

    with (ROOT / "models" / "mlp_classifier.pkl").open("rb") as f:
        clf = pickle.load(f)

    alarms, debug = cfar_detect(x_all[0], FS, pfa=1e-2)
    exported_alarms = []
    angles = None
    for index, alarm in enumerate(alarms):
        seg = analysis_window(x_all, alarm, FS)
        X = np.fft.rfft(seg, axis=1)
        freqs = np.fft.rfftfreq(seg.shape[1], 1 / FS)
        f_band = select_doa_band(X, freqs)
        angles, spectrum_db, estimate = cbf_spectrum(
            X, freqs, r=R, c=C, f_band=f_band
        )
        features = logmel(seg[0], FS)
        probabilities = clf.predict_proba(features.ravel()[None, :])[0]
        label = LABELS[int(np.argmax(probabilities))]
        truth = next((event for event in EVENTS if event["cls"] == label), None)
        center = (alarm["t_start"] + alarm["t_end"]) / 2.0
        window_start = min(max(0.0, center - 0.5), 7.0)
        exported_alarms.append({
            "id": f"A-{index + 1:02d}",
            "tStart": alarm["t_start"],
            "tEnd": alarm["t_end"],
            "detectionStatDb": alarm["detection_stat_db"],
            "label": label,
            "bearingDeg": estimate,
            "truthBearingDeg": truth["bearing"] if truth else None,
            "fBandHz": [float(f_band[0]), float(f_band[1])],
            "windowStart": window_start,
            "windowEnd": window_start + 1.0,
            "probabilities": {
                name: float(value) for name, value in zip(LABELS, probabilities)
            },
            "spectrumDb": rounded(spectrum_db, 3),
            "logMel": {
                "rows": int(features.shape[0]),
                "cols": int(features.shape[1]),
                "values": rounded(features.ravel(), 3),
            },
        })

    payload = {
        "meta": {
            "project": "听水·知微",
            "scenario": "8 秒三目标四通道仿真",
            "dataMode": "simulation",
            "durationSec": 8.0,
            "sampleRateHz": FS,
            "channelCount": 4,
            "arrayRadiusM": R,
            "soundSpeedMps": C,
            "detector": "双通道 CA-CFAR",
            "doa": "CBF 常规波束形成",
            "classifier": "log-Mel + MLP(128)",
            "exportCommand": "python -m src.viz.export_demo",
            "scoreNote": "分类数值为未校准的 softmax 模型得分，不是现实置信度。",
        },
        "waveform": waveform_envelope(x_all),
        "detection": {
            "time": rounded(debug["t"], 5),
            "marginDb": rounded(debug["stat"], 3),
            "frameHit": [bool(v) for v in debug["frame_hit"]],
            "thresholdDb": float(debug["th_db"]),
            "designPfa": 0.01,
        },
        "doaAnglesDeg": rounded(angles, 2),
        "alarms": exported_alarms,
        "truth": EVENTS,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"已导出 {len(exported_alarms)} 段真实仿真告警：{OUTPUT}")
    for alarm in exported_alarms:
        print(
            f"{alarm['id']} {alarm['tStart']:.2f}–{alarm['tEnd']:.2f}s | "
            f"{alarm['label']} | {alarm['bearingDeg']:.1f}° | "
            f"模型得分 {max(alarm['probabilities'].values()):.4f}"
        )


if __name__ == "__main__":
    main()
