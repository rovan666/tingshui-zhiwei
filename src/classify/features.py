# -*- coding: utf-8 -*-
"""log-Mel 时频特征提取（纯 numpy 实现）。

把一段波形转成"声音的照片"：横轴时间、纵轴按人耳感知加权的 Mel 频率、
颜色为对数能量。船舶的低频线谱、落水的宽带脉冲、鱼的高频咔哒在图上
呈现完全不同的纹理，供神经网络学习。
"""
import numpy as np


def _mel_filterbank(n_mels: int, n_fft: int, fs: float,
                    fmin: float = 50.0, fmax: float = 24_000.0):
    """标准 HTK Mel 三角滤波器组，返回 (n_mels, n_fft//2+1)。"""
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_pts = np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)
    hz_pts = mel2hz(mel_pts)
    freqs = np.fft.rfftfreq(n_fft, 1 / fs)
    fb = np.zeros((n_mels, len(freqs)))
    for m in range(1, n_mels + 1):
        f0, f1, f2 = hz_pts[m - 1], hz_pts[m], hz_pts[m + 1]
        up = (freqs - f0) / (f1 - f0 + 1e-12)
        down = (f2 - freqs) / (f2 - f1 + 1e-12)
        fb[m - 1] = np.maximum(0.0, np.minimum(up, down))
    return fb


def logmel(x_mono: np.ndarray, fs: float = 96_000.0, n_fft: int = 1024,
           hop: int = 512, n_mels: int = 40):
    """返回 log-Mel 特征 (n_mels, T)，已做均值方差归一化。"""
    n = len(x_mono)
    if n < n_fft:
        x_mono = np.pad(x_mono, (0, n_fft - n))
        n = len(x_mono)
    win = np.hanning(n_fft)
    starts = np.arange(0, n - n_fft + 1, hop)
    segs = np.stack([x_mono[s:s + n_fft] * win for s in starts], axis=1)
    P = np.abs(np.fft.rfft(segs, axis=0)) ** 2
    fb = _mel_filterbank(n_mels, n_fft, fs)
    S = fb @ P
    S = np.log(S + 1e-10)
    S = (S - S.mean()) / (S.std() + 1e-10)
    return S.astype(np.float32)
