# -*- coding: utf-8 -*-
"""方向引导延时求和聚焦与显示端中位数平滑。"""
from collections import deque
import numpy as np

from src.features.logmel import stft16


class BearingSmoother:
    def __init__(self, size: int = 3):
        self.values = deque(maxlen=size)

    def update(self, result: dict):
        if result.get("valid"):
            self.values.append(float(result["angle_deg"]))
        return float(np.median(self.values)) if self.values else None

    def reset(self):
        self.values.clear()


def focus_event(x: np.ndarray, fs: int, cfg, steering, point_index: int,
                selected, freqs: np.ndarray) -> np.ndarray:
    """有效测向频点使用聚焦频谱，其余频点保留 ch7，再重叠相加重建。"""
    if point_index is None:
        return np.asarray(x[cfg.array.reference_channel], dtype=np.float32)
    X, _, _ = stft16(x, fs, cfg.features)
    ref = cfg.array.reference_channel
    Y = X[ref].copy()
    lookup = {float(f): i for i, f in enumerate(steering.frequencies)}
    for indices in selected:
        for freq_index in indices:
            a = steering.values[lookup[float(freqs[freq_index])], point_index]
            Y[freq_index] = (a.conj() @ X[:, freq_index, :]) / (a.conj() @ a + 1e-20)

    cfg_feat = cfg.features
    frames = np.fft.irfft(Y, n=cfg_feat.n_fft, axis=0)
    window = np.hanning(cfg_feat.n_fft)
    frames *= window[:, None]
    padded_n = (frames.shape[1] - 1) * cfg_feat.hop + cfg_feat.n_fft
    output = np.zeros(padded_n)
    weight = np.zeros(padded_n)
    for frame_index in range(frames.shape[1]):
        start = frame_index * cfg_feat.hop
        output[start:start + cfg_feat.n_fft] += frames[:, frame_index]
        weight[start:start + cfg_feat.n_fft] += window ** 2
    np.divide(output, weight, out=output, where=weight > 1e-8)
    output[weight <= 1e-8] = 0.0
    return output[:x.shape[1]].astype(np.float32)
