# -*- coding: utf-8 -*-
"""MUSIC（多重信号分类）超分辨方位估计。

原理：对阵列接收信号的协方差矩阵做特征分解，把空间分成"信号子空间"和
"噪声子空间"。真实声源方向的导向矢量与噪声子空间正交，因此谱函数
P(θ) = 1 / |a(θ)^H En|² 在真实方向出现极尖锐的峰，分辨率远超 CBF。

注意：多途反射产生的相干信号会使协方差矩阵退秩、MUSIC 失效，
实际部署需配空间平滑或学习型方法（本阶段仿真不含多途）。
"""
import numpy as np

from sim.uca_sim import element_positions


def music_spectrum(
    x: np.ndarray,
    fs: float,
    f0: float,
    r: float = 0.0375,
    c: float = 1500.0,
    n_sources: int = 1,
    n_scan: int = 721,
    nperseg: int = 256,
    hop: int = 128,
):
    """计算 MUSIC 方位谱。

    参数:
        x:         时域信号 (4, N)
        fs:        采样率 Hz
        f0:        窄带目标频率 Hz
        n_sources: 信源个数（需 < 阵元数）
    返回:
        angles_deg: 扫描角 (n_scan,)
        P_db:       归一化 MUSIC 谱 dB（峰值 0 dB）
        est_deg:    方位估计值
    """
    n_el, n = x.shape
    win = np.hanning(nperseg)
    k0 = int(round(f0 / fs * nperseg))          # 最接近 f0 的频点
    f_bin = k0 * fs / nperseg                   # 实际窄带频率

    # 分帧 FFT，取 f0 频点作为窄带快拍
    snaps = []
    for s in range(0, n - nperseg + 1, hop):
        seg = x[:, s:s + nperseg] * win[None, :]
        F = np.fft.fft(seg, axis=1)
        snaps.append(F[:, k0])
    Xk = np.stack(snaps, axis=1)                # (4, K)

    # 协方差矩阵 + 特征分解（eigh 特征值升序，小特征值对应噪声子空间）
    R = Xk @ Xk.conj().T / Xk.shape[1]
    _, evecs = np.linalg.eigh(R)
    En = evecs[:, :n_el - n_sources]            # (4, 4-d)

    # 导向矢量与仿真模型一致：a(θ) = exp(-j 2π f τ(θ))
    pos = element_positions(r)
    angles = np.linspace(0.0, 2.0 * np.pi, n_scan, endpoint=False)
    U = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    taus = U @ pos.T / c                        # (n_scan, 4)
    A = np.exp(-1j * 2.0 * np.pi * f_bin * taus)

    # 谱函数 P = 1 / ||a(θ)^H En||²（注意是共轭转置 a^H，不是转置 a^T）
    proj = A.conj() @ En
    denom = np.sum(np.abs(proj) ** 2, axis=1)
    P = 1.0 / (denom + 1e-12)

    P_db = 10.0 * np.log10(P / P.max() + 1e-12)
    angles_deg = np.rad2deg(angles)
    est_deg = float(angles_deg[int(np.argmax(P))])
    return angles_deg, P_db, est_deg
