# -*- coding: utf-8 -*-
"""常规波束形成（CBF / delay-and-sum）方位估计。

对扫描角 θ：按平面波模型补偿各阵元时延，同相叠加后测输出功率，
功率最大的方向即方位估计值。输出归一化方位谱（dB）。
"""
import numpy as np

from sim.uca_sim import element_positions


def cbf_spectrum(
    X: np.ndarray,
    freqs: np.ndarray,
    r: float = 0.0375,
    c: float = 1500.0,
    f_band=(8_000.0, 12_000.0),
    n_scan: int = 721,
):
    """计算 CBF 方位谱。

    参数:
        X:      频域信号 (4, F)
        freqs:  频率轴 (F,)
        f_band: 参与波束形成的频带
    返回:
        angles_deg: 扫描角 (n_scan,)
        P_db:       归一化功率谱 dB（峰值 0 dB）
        est_deg:    方位估计值
    """
    mask = (freqs >= f_band[0]) & (freqs <= f_band[1])
    Xb = X[:, mask]                                   # (4, Fb)
    fb = freqs[mask]                                  # (Fb,)

    pos = element_positions(r)                        # (4, 2)
    angles = np.linspace(0.0, 2.0 * np.pi, n_scan, endpoint=False)
    U = np.stack([np.cos(angles), np.sin(angles)], axis=1)   # (n_scan, 2)
    taus = U @ pos.T / c                              # (n_scan, 4)

    # 导向矢量 E[f, k, i] = exp(+j 2π f τ_ki)，波束输出 Y[f, k] = Σ_i X[i, f]·E
    E = np.exp(1j * 2.0 * np.pi * fb[:, None, None] * taus[None, :, :])
    Y = np.einsum("if,fki->fk", Xb, E)                # (Fb, n_scan)
    P = np.mean(np.abs(Y) ** 2, axis=0)               # (n_scan,)

    P_db = 10.0 * np.log10(P / P.max() + 1e-12)
    angles_deg = np.rad2deg(angles)
    est_deg = float(angles_deg[int(np.argmax(P))])
    return angles_deg, P_db, est_deg
