# -*- coding: utf-8 -*-
"""全链路流水线 demo：仿真场景 → CFAR 检测 → CBF 定位 → 神经网络分类。

场景（8 秒，四通道圆阵）：
  1–3 s   船舶噪声，方位 60°
  4–5 s   人员落水拍水，方位 120°
  5.5–7 s 鱼类活动，方位 250°

运行：python run_pipeline.py
"""
import os
import pickle

import numpy as np

from sim.uca_sim import element_positions
from sim.sources import GENERATORS
from src.detect.cfar import cfar_detect
from src.doa.cbf import cbf_spectrum

FS = 96_000.0
R, C = 0.0375, 1500.0
OUT_DIR = os.path.join("figures", "day3")

EVENTS = [
    dict(cls="ship", t0=1.0, t1=3.0, bearing=60.0, snr_db=6.0),
    dict(cls="human", t0=4.0, t1=5.0, bearing=120.0, snr_db=3.0),
    dict(cls="fish", t0=5.5, t1=7.0, bearing=250.0, snr_db=0.0),
]


def add_source(x_all, sig, bearing_deg, t0, snr_db, rng):
    """把单通道源信号按方位角加入四通道阵列（频域分数时延）。"""
    n = x_all.shape[1]
    pos = element_positions(R)
    u = np.array([np.cos(np.deg2rad(bearing_deg)), np.sin(np.deg2rad(bearing_deg))])
    taus = pos @ u / C
    freqs = np.fft.rfftfreq(n, 1 / FS)
    S = np.fft.rfft(sig, n=n)
    start = int(t0 * FS)
    scale = np.sqrt(np.mean(sig ** 2) / (10 ** (snr_db / 10.0)))
    for i in range(4):
        delayed = np.fft.irfft(S * np.exp(-1j * 2 * np.pi * freqs * taus[i]), n=n)
        x_all[i, start:start + len(sig)] += scale * delayed[start:start + len(sig)]


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(11)
    n = int(8.0 * FS)
    x_all = 0.5 * rng.standard_normal((4, n))       # 各通道独立背景噪声
    for ev in EVENTS:
        sig = GENERATORS[ev["cls"]](duration=ev["t1"] - ev["t0"], rng=rng)
        add_source(x_all, sig, ev["bearing"], ev["t0"], ev["snr_db"], rng)

    with open(os.path.join("models", "mlp_classifier.pkl"), "rb") as f:
        clf = pickle.load(f)

    mono = x_all.mean(axis=0)
    alarms, dbg = cfar_detect(mono, FS, pfa=1e-2)
    print("=" * 66)
    print("听水·知微 | 全链路流水线（检测→定位→分类，仿真数据）")
    print("=" * 66)
    print(f"共 {len(alarms)} 段告警：")
    for a in alarms:
        i0, i1 = int(a["t_start"] * FS), int(a["t_end"] * FS)
        seg = x_all[:, i0:i1]
        X = np.fft.rfft(seg, axis=1)
        freqs = np.fft.rfftfreq(seg.shape[1], 1 / FS)
        # 自动选取能量最强的谱线所在频带做波束形成
        band_power = X.mean(axis=1)
        k = int(np.argmax(np.abs(band_power[(freqs > 500) & (freqs < 24000)])))
        f_peak = freqs[(freqs > 500) & (freqs < 24000)][k]
        f_lo, f_hi = max(500.0, f_peak - 1500), min(24000.0, f_peak + 1500)
        _, _, est = cbf_spectrum(X, freqs, r=R, c=C, f_band=(f_lo, f_hi))
        label, prob = __import__("src.classify.mlp", fromlist=["classify"]) \
            .classify(mono[i0:i1], clf)
        print(f"【告警】{a['t_start']:5.2f}s–{a['t_end']:5.2f}s | "
              f"方位 {est:6.1f}° | 类型 {label:6s} | 置信度 {prob:.2f}")
    print("\n真实事件对照：")
    for ev in EVENTS:
        print(f"  {ev['t0']}–{ev['t1']}s | {ev['cls']:6s} | 方位 {ev['bearing']}°")


if __name__ == "__main__":
    main()
