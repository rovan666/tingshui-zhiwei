# -*- coding: utf-8 -*-
"""四元均匀圆阵（UCA）远场声源信号仿真。

物理模型：
- 4 个水听器均匀分布在半径 r 的圆周上（45/135/225/315 度）
- 远场平面波从方位角 phi 入射，各阵元接收信号仅有传播时延差异
- 水中声速 c = 1500 m/s
- 加性高斯白噪声，按设定 SNR 控制噪声功率
"""
import numpy as np

ELEMENT_ANGLES_DEG = np.array([45.0, 135.0, 225.0, 315.0])


def element_positions(r: float) -> np.ndarray:
    """返回 (4, 2) 的阵元坐标 (x, y)，单位 m。"""
    ang = np.deg2rad(ELEMENT_ANGLES_DEG)
    return np.stack([r * np.cos(ang), r * np.sin(ang)], axis=1)


def simulate_uca(
    phi_deg: float,
    f0: float = 10_000.0,
    fs: float = 96_000.0,
    duration: float = 0.05,
    snr_db: float = 0.0,
    r: float = 0.0375,
    c: float = 1500.0,
    seed: int = 42,
):
    """仿真四元圆阵接收信号。

    返回:
        t:      时间轴 (N,)
        x:      时域信号 (4, N)
        X:      频域信号 rfft 结果 (4, F)
        freqs:  频率轴 (F,)
    """
    rng = np.random.default_rng(seed)
    n = int(fs * duration)
    t = np.arange(n) / fs

    # 源信号：10 kHz 单频 + 轻微幅度调制（模拟舰船/敲击类脉冲包络），加窗成形
    s = np.sin(2 * np.pi * f0 * t) * (1.0 + 0.3 * np.sin(2 * np.pi * 500.0 * t))
    s = s * np.hanning(n)

    # 各阵元相对阵列中心的传播时延 τ_i = p_i · u / c
    pos = element_positions(r)                                  # (4, 2)
    u = np.array([np.cos(np.deg2rad(phi_deg)),
                  np.sin(np.deg2rad(phi_deg))])
    taus = pos @ u / c                                          # (4,)

    # 频域加分数时延（比时域插值精确）
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    S = np.fft.rfft(s)
    X = S[None, :] * np.exp(-1j * 2 * np.pi * freqs[None, :] * taus[:, None])

    # 时域还原并加噪
    x = np.fft.irfft(X, n=n)
    sig_pow = float(np.mean(x ** 2))
    noise_pow = sig_pow / (10.0 ** (snr_db / 10.0))
    x = x + np.sqrt(noise_pow) * rng.standard_normal((4, n))
    X = np.fft.rfft(x, axis=1)

    return t, x, X, freqs
