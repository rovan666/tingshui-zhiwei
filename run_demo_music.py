# -*- coding: utf-8 -*-
"""Day-2 demo：CBF vs MUSIC 方位谱对比（0 dB 与 -10 dB 两组信噪比）。

运行：python run_demo_music.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim.uca_sim import simulate_uca
from src.doa.cbf import cbf_spectrum
from src.doa.music import music_spectrum

F0, FS, DURATION = 10_000.0, 96_000.0, 0.05
R, C = 0.0375, 1500.0
TRUE_ANGLE = 35.0
OUT_DIR = os.path.join("figures", "day2")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 60)
    print("听水·知微 | CBF vs MUSIC 对比（仿真数据）")
    print("=" * 60)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for row, snr in enumerate([0.0, -10.0]):
        _, x, X, freqs = simulate_uca(
            phi_deg=TRUE_ANGLE, f0=F0, fs=FS, duration=DURATION, snr_db=snr, r=R, c=C
        )
        ang, P_cbf, est_cbf = cbf_spectrum(X, freqs, r=R, c=C)
        _, P_music, est_music = music_spectrum(x, FS, F0, r=R, c=C)

        ax = axes[row]
        ax.plot(ang, P_cbf, lw=1.2, label=f"CBF  est={est_cbf:.1f} deg")
        ax.plot(ang, P_music, lw=1.2, label=f"MUSIC  est={est_music:.1f} deg")
        ax.axvline(TRUE_ANGLE, color="r", ls="--", alpha=0.6, label="True 35 deg")
        ax.set_ylabel("Normalized power (dB)")
        ax.set_ylim(-45, 2)
        ax.set_title(f"SNR = {snr:.0f} dB")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right")
        print(f"SNR {snr:+.0f} dB | CBF 估计 {est_cbf:6.1f}° (误差 {abs(est_cbf-TRUE_ANGLE):.1f}°) "
              f"| MUSIC 估计 {est_music:6.1f}° (误差 {abs(est_music-TRUE_ANGLE):.1f}°)")

    axes[-1].set_xlabel("Azimuth (deg)")
    fig.suptitle("CBF vs MUSIC spatial spectrum (4-element small-aperture UCA, simulated)")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "cbf_vs_music.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n已保存：{out}")


if __name__ == "__main__":
    main()
