# -*- coding: utf-8 -*-
"""频谱 CFAR（恒虚警率）检测——声呐标准的窄带谱线检测做法。

对每个时刻的功率谱，沿**频率**做 CA-CFAR：某频点的功率若显著高于
左右邻近频带的背景均值，则判为谱线（窄带目标）。任一频点超标即判该帧有目标。

为什么不沿时间做 CFAR：持续型目标（如持续的舰船线谱）会被时间参考窗
"吸收"进背景，导致检不出；沿频率做则不受目标持续时间影响。

虚警率分配：频谱与能量通道各占帧级预算 pfa/2；频谱通道有 F 个频点，
单频点预算 pfa_cell = (pfa/2) / F，按并集上界分配，不假设频点相互独立。
门限系数 α = N·(pfa_cell^(-1/N) - 1)，N 为参考频点数。
该公式仍采用平方律单元服从指数分布的 CA-CFAR 近似，实际虚警率需实验校准。
"""
import numpy as np


def stft_power(x_mono: np.ndarray, frame_len: int = 256, hop: int = 128,
               fs: float = 96_000.0):
    """用矩形窗分帧，返回 (功率谱 (F,T), 频率轴 (F,), 帧中心时间秒 (T,))。

    白噪声经矩形窗正交 FFT 后各频点相互独立，符合当前 CA-CFAR 的
    指数分布参考单元假设；Hann 窗会引入相邻频点相关，使实测虚警率偏高。
    """
    n = len(x_mono)
    starts = np.arange(0, n - frame_len + 1, hop)
    segs = np.stack([x_mono[s:s + frame_len] for s in starts], axis=1)
    F = np.fft.rfft(segs, axis=0)
    P = np.abs(F) ** 2 / (frame_len ** 2)
    freqs = np.fft.rfftfreq(frame_len, 1 / fs)
    t = (starts + frame_len / 2.0) / fs
    return P, freqs, t


def _energy_cfar_hits(P: np.ndarray, pfa: float, ref: int = 16,
                      guard: int = 4):
    """宽带能量 CFAR（沿时间）：每帧总能量 vs 前后参考窗背景。

    负责宽带瞬态事件（落水拍水等）——这类事件在频谱上没有突出谱线，
    谱线 CFAR 抓不到，但帧总能量明显抬升。瞬态目标不会被时间参考窗
    完全吸收（事件时长 << 参考窗总长）。
    """
    e = P.sum(axis=0)                                # (T,)
    T = len(e)
    alpha = (2 * ref) * (pfa ** (-1.0 / (2 * ref)) - 1.0)
    hits = np.zeros(T, dtype=bool)
    stat = np.zeros(T)
    for i in range(T):
        lo, hi = max(0, i - guard - ref), max(0, i - guard)
        lo2, hi2 = min(T, i + guard + 1), min(T, i + guard + ref + 1)
        if (hi - lo) + (hi2 - lo2) < 4:
            continue
        bg = (e[lo:hi].sum() + e[lo2:hi2].sum()) / ((hi - lo) + (hi2 - lo2))
        stat[i] = e[i] / (bg + 1e-20)
        hits[i] = stat[i] > alpha
    return hits, stat


def cfar_detect(x_mono: np.ndarray, fs: float, frame_len: int = 1024,
                hop: int = 512, pfa: float = 1e-2, ref: int = 8,
                guard: int = 2, min_frames: int = 4,
                merge_gap: float = 0.15,
                f_range=(100.0, 24_000.0)):
    """双通道 CFAR 检测主接口（频谱谱线 + 宽带能量，命中任一即报警）。

    频段边缘使用已有的单侧参考窗，门限按实际参考单元数计算，避免低频
    谱线因另一侧窗口越界而被直接跳过。至少连续 min_frames 帧才形成告警。

    返回 (alarms, debug)：
        alarms: list[dict]，含 t_start/t_end/detection_stat_db（秒、dB）
        debug:  dict(t, stat, th_db, frame_hit)；stat 为归一化检测裕量，
                统一以 0 dB 为判决门限，frame_hit 为成形告警前的原始帧命中。
    """
    P, freqs, t = stft_power(x_mono, frame_len, hop, fs)
    band = (freqs >= f_range[0]) & (freqs <= f_range[1])
    Pb = P[band, :]                                  # (Fb, T)
    Fb, T = Pb.shape

    pfa_spectral = pfa / 2
    pfa_energy = pfa / 2
    pfa_cell = pfa_spectral / Fb

    # 沿频率估计背景；边缘频点保留可用的单侧参考窗，跳过 guard。
    C = np.concatenate([np.zeros((1, T)), np.cumsum(Pb, axis=0)], axis=0)
    idx = np.arange(Fb)
    l0 = np.maximum(0, idx - guard - ref)
    l1 = np.maximum(0, idx - guard)
    r0 = np.minimum(Fb, idx + guard + 1)
    r1 = np.minimum(Fb, idx + guard + ref + 1)
    n_ref = (l1 - l0) + (r1 - r0)
    valid = n_ref >= 4
    bg = (C[l1] - C[l0] + C[r1] - C[r0]) / np.maximum(n_ref, 1)[:, None]
    alpha = n_ref * (pfa_cell ** (-1.0 / np.maximum(n_ref, 1)) - 1.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = Pb / bg
        ratio[~np.isfinite(ratio)] = 0.0
    hit = (ratio > alpha[:, None]) & valid[:, None]
    margin = np.where(valid[:, None], ratio / alpha[:, None], 0.0)
    stat = 10 * np.log10(margin.max(axis=0) + 1e-20)

    frame_hit = hit.any(axis=0)
    # 两个通道各使用一半帧级虚警率预算，按并集上界控制总预算。
    e_hit, _ = _energy_cfar_hits(Pb, pfa_energy)
    frame_hit = frame_hit | e_hit
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
                    detection_stat_db=float(np.max(stat[i:j + 1])),
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
            m["detection_stat_db"] = max(m["detection_stat_db"], a["detection_stat_db"])
        else:
            merged.append(dict(a))
    return merged, dict(t=t, stat=stat, th_db=0.0, frame_hit=frame_hit.copy(),
                        pfa_spectral=pfa_spectral, pfa_energy=pfa_energy)
