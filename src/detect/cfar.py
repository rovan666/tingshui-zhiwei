# -*- coding: utf-8 -*-
"""频谱 CFAR（恒虚警率）检测——声呐标准的窄带谱线检测做法。

对每个时刻的功率谱，沿**频率**做 CA-CFAR：某频点的功率若显著高于
左右邻近频带的背景均值，则判为谱线（窄带目标）。任一频点超标即判该帧有目标。

为什么不沿时间做 CFAR：持续型目标（如持续的舰船线谱）会被时间参考窗
"吸收"进背景，导致检不出；沿频率做则不受目标持续时间影响。

虚警率分配：帧级虚警率 pfa，F 个参与检测的频点，
单频点虚警率 pfa_cell = pfa / F（独立近似）。
门限系数 α = N·(pfa_cell^(-1/N) - 1)，N 为参考频点数。
"""
import numpy as np


def stft_power(x_mono: np.ndarray, frame_len: int = 256, hop: int = 128,
               fs: float = 96_000.0):
    """返回 (功率谱 (F,T), 频率轴 (F,), 帧中心时间秒 (T,))。"""
    n = len(x_mono)
    win = np.hanning(frame_len)
    starts = np.arange(0, n - frame_len + 1, hop)
    segs = np.stack([x_mono[s:s + frame_len] * win for s in starts], axis=1)
    F = np.fft.rfft(segs, axis=0)
    P = np.abs(F) ** 2 / (frame_len ** 2)
    freqs = np.fft.rfftfreq(frame_len, 1 / fs)
    t = (starts + frame_len / 2.0) / fs
    return P, freqs, t


def cfar_detect(x_mono: np.ndarray, fs: float, frame_len: int = 256,
                hop: int = 128, pfa: float = 1e-2, ref: int = 8,
                guard: int = 2, min_frames: int = 5,
                merge_gap: float = 0.15,
                f_range=(500.0, 24_000.0)):
    """频谱 CFAR 检测主接口。

    返回 (alarms, debug)：
        alarms: list[dict]，含 t_start/t_end/snr_est（秒、dB）
        debug:  dict(t, stat, th_db)，stat 为每帧最大频点功率比（dB）
    """
    P, freqs, t = stft_power(x_mono, frame_len, hop, fs)
    band = (freqs >= f_range[0]) & (freqs <= f_range[1])
    Pb = P[band, :]                                  # (Fb, T)
    Fb, T = Pb.shape

    pfa_cell = pfa / Fb
    alpha = (2 * ref) * (pfa_cell ** (-1.0 / (2 * ref)) - 1.0)

    # 沿频率的滑窗背景估计（cumsum 向量化）：左 ref + 右 ref，跳过 guard
    C = np.concatenate([np.zeros((1, T)), np.cumsum(Pb, axis=0)], axis=0)
    idx = np.arange(Fb)
    l0, l1 = idx - guard - ref, idx - guard          # 左窗 [l0, l1)
    r0, r1 = idx + guard + 1, idx + guard + ref + 1  # 右窗 [r0, r1)
    valid = (l0 >= 0) & (r1 <= Fb)
    bg = np.full((Fb, T), np.nan)
    bg[valid] = (C[l1[valid]] - C[l0[valid]]
                 + C[r1[valid]] - C[r0[valid]]) / (2 * ref)

    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = Pb / bg
        ratio[~np.isfinite(ratio)] = 0.0
    hit = (ratio > alpha) & valid[:, None]
    stat = 10 * np.log10(np.where(valid[:, None], ratio, 0.0).max(axis=0) + 1e-20)

    frame_hit = hit.any(axis=0)
    alarms = []
    i = 0
    while i < T:
        if frame_hit[i]:
            j = i
            while j + 1 < T and frame_hit[j + 1]:
                j += 1
            if j - i + 1 >= min_frames:
                alarms.append(dict(
                    t_start=float(t[i]), t_end=float(t[j]),
                    snr_est=float(np.max(stat[i:j + 1])),
                ))
            i = j + 1
        else:
            i += 1
    # 合并间隔小于 merge_gap 的相邻告警（同一目标的断续检出）
    merged = []
    for a in alarms:
        if merged and a["t_start"] - merged[-1]["t_end"] <= merge_gap:
            m = merged[-1]
            m["t_end"] = a["t_end"]
            m["snr_est"] = max(m["snr_est"], a["snr_est"])
        else:
            merged.append(dict(a))
    return merged, dict(t=t, stat=stat, th_db=10 * np.log10(alpha))
