# -*- coding: utf-8 -*-
"""Day-1 demo：四元圆阵信号仿真 + CBF 方位估计，输出两张图到 figures/day1/。

运行：python run_demo_d1.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sim.uca_sim import simulate_uca
from src.doa.cbf import cbf_spectrum

# ---- 仿真参数（与硬件方案一致：10 kHz 频段、小孔径圆阵 r=λ/4≈3.75cm）----
F0 = 10_000.0
FS = 96_000.0
DURATION = 0.05
SNR_DB = 0.0
R = 0.0375
C = 1500.0
TRUE_ANGLE = 35.0

OUT_DIR = os.path.join("figures", "day1")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 56)
    print("听水·知微 | Day-1 demo：UCA 仿真 + CBF 方位估计（仿真数据）")
    print("=" * 56)
    print(f"阵元数 4 | 半径 {R*100:.2f} cm | 频率 {F0/1000:.1f} kHz | "
          f"SNR {SNR_DB:.0f} dB | 真实方位 {TRUE_ANGLE:.1f}°")

    t, x, X, freqs = simulate_uca(
        phi_deg=TRUE_ANGLE, f0=F0, fs=FS, duration=DURATION,
        snr_db=SNR_DB, r=R, c=C,
    )

    # 图 1：四通道波形（前 3 ms，可见阵元间时延差异）
    fig, axes = plt.subplots(4, 1, figsize=(9, 7), sharex=True)
    show = t <= 0.003
    for i, ax in enumerate(axes):
        ax.plot(t[show] * 1000.0, x[i][show], lw=0.9)
        ax.set_ylabel(f"CH{i+1}")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle("Simulated 4-element UCA waveforms "
                 f"(source at {TRUE_ANGLE:.0f} deg, SNR {SNR_DB:.0f} dB)")
    fig.tight_layout()
    p1 = os.path.join(OUT_DIR, "waveforms_4ch.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    # 图 2：CBF 方位谱（直角坐标 + 极坐标）
    angles_deg, P_db, est_deg = cbf_spectrum(X, freqs, r=R, c=C)
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(angles_deg, P_db, lw=1.2)
    ax1.axvline(TRUE_ANGLE, color="r", ls="--", alpha=0.7,
                label=f"True {TRUE_ANGLE:.0f} deg")
    ax1.axvline(est_deg, color="g", ls=":", alpha=0.9,
                label=f"Est {est_deg:.1f} deg")
    ax1.set(xlabel="Azimuth (deg)", ylabel="Normalized power (dB)",
            xlim=(0, 360), ylim=(-40, 0))
    ax1.grid(alpha=0.3)
    ax1.legend()
    ax1.set_title("CBF spatial spectrum")

    ax2 = fig.add_subplot(1, 2, 2, projection="polar")
    ax2.plot(np.deg2rad(angles_deg), P_db - P_db.max() + 40.0, lw=1.2)
    ax2.set_theta_zero_location("N")
    ax2.set_title("Polar view")
    fig.suptitle("Conventional Beamforming DOA (simulated data)")
    fig.tight_layout()
    p2 = os.path.join(OUT_DIR, "cbf_spectrum.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)

    err = abs(est_deg - TRUE_ANGLE)
    print(f"\n估计方位 {est_deg:.1f}° | 误差 {err:.1f}°")
    print(f"已保存：{p1}")
    print(f"已保存：{p2}")


if __name__ == "__main__":
    main()
