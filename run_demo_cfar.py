# -*- coding: utf-8 -*-
"""CFAR 检测 demo：场景检出 + 检测率/虚警率-SNR 蒙特卡洛实验。

运行：python run_demo_cfar.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.detect.cfar import cfar_detect

FS = 96_000.0
OUT_DIR = os.path.join("figures", "day3")


def make_scene(snr_db, with_target, rng):
    """4 秒场景：1.5–2.5 s 为目标出现窗口（10 kHz 窄带信号）。"""
    n = int(4.0 * FS)
    t = np.arange(n) / FS
    x = rng.standard_normal(n)                      # 背景噪声
    if with_target:
        mask = (t >= 1.5) & (t <= 2.5)
        sig = np.sin(2 * np.pi * 10_000.0 * t) * mask
        sig_pow = np.mean(sig[mask] ** 2)
        # 目标功率/噪声功率 = snr_lin → 幅度按 sqrt(snr_lin) 缩放
        x = x + np.sqrt(10 ** (snr_db / 10.0)) * sig / np.sqrt(sig_pow) \
            * np.sqrt(np.mean(x ** 2))
    return t, x


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(7)
    print("=" * 60)
    print("听水·知微 | CFAR 恒虚警率检测（仿真数据）")
    print("=" * 60)

    # 实验 1：单场景检出（SNR = -6 dB）
    t, x = make_scene(-6.0, True, rng)
    alarms, dbg = cfar_detect(x, FS, pfa=1e-2)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(dbg["t"], dbg["stat"], lw=0.8,
            label="Spectral CFAR normalized detection margin")
    ax.axhline(dbg["th_db"], color="r", lw=1.4, label="Decision threshold (0 dB)")
    ax.axvspan(1.5, 2.5, color="green", alpha=0.12, label="Target present (10 kHz, SNR -6 dB)")
    ax.set(xlabel="Time (s)", ylabel="Margin (dB)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.suptitle("Spectral CFAR detection margin (simulated)")
    fig.tight_layout()
    p1 = os.path.join(OUT_DIR, "cfar_scene.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"场景检出告警数：{len(alarms)}")
    for a in alarms:
        print(f"  告警 {a['t_start']:.2f}s–{a['t_end']:.2f}s  "
              f"频谱检测裕量 {a['detection_stat_db']:.1f} dB")

    # 实验 2：检测率与三种虚警口径；无目标场景与有目标场景分开统计。
    snrs = [-15, -12, -10, -8, -6, -3]
    pds, scene_pfas, frame_pfas = [], [], []
    trials = 40
    scene_duration = 4.0
    for snr in snrs:
        det, scene_fa = 0, 0
        false_frames, total_frames, false_alarms = 0, 0, 0
        for _ in range(trials):
            _, x1 = make_scene(snr, True, rng)
            a1, _ = cfar_detect(x1, FS, pfa=1e-2)
            det += int(any(a["t_start"] < 2.6 and a["t_end"] > 1.4 for a in a1))
            _, x0 = make_scene(snr, False, rng)
            a0, d0 = cfar_detect(x0, FS, pfa=1e-2)
            scene_fa += int(len(a0) > 0)
            false_alarms += len(a0)
            false_frames += int(np.count_nonzero(d0["frame_hit"]))
            total_frames += len(d0["frame_hit"])
        pds.append(det / trials)
        scene_pfas.append(scene_fa / trials)
        frame_pfas.append(false_frames / total_frames)
        alarms_per_hour = false_alarms / (trials * scene_duration) * 3600.0
        print(f"SNR {snr:+3d} dB | 检测率 {det/trials:.2f} | "
              f"场景级虚警概率 {scene_fa/trials:.2f} | "
              f"帧级虚警率 {false_frames/total_frames:.5f} | "
              f"虚警 {alarms_per_hour:.1f} 次/小时")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(snrs, pds, "o-", label="Probability of detection")
    axes[0].plot(snrs, scene_pfas, "s--", label="Scene false-alarm probability")
    axes[0].set(xlabel="SNR (dB)", ylabel="Probability", ylim=(-0.03, 1.05),
                title="Event and scene level")
    axes[1].plot(snrs, frame_pfas, "o-", color="tab:purple",
                 label="Measured frame false-alarm rate")
    axes[1].axhline(0.01, color="r", ls=":", alpha=0.6,
                    label="Design frame Pfa = 0.01")
    axes[1].set(xlabel="SNR (dB)", ylabel="Frame false-alarm rate",
                title="Frame level")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"CFAR performance, Monte Carlo x{trials} (simulated)")
    fig.tight_layout()
    p2 = os.path.join(OUT_DIR, "cfar_pd_pfa.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"\n已保存：{p1}\n已保存：{p2}")


if __name__ == "__main__":
    main()
