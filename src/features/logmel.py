# -*- coding: utf-8 -*-
"""新方案统一 STFT 与 128×193 log-Mel 特征。"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import FeatureConfig


def _stft_mono(x: np.ndarray, fs: int, cfg: FeatureConfig):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or len(x) == 0:
        raise ValueError("STFT 输入必须是一维非空波形")
    if len(x) < cfg.n_fft:
        target = cfg.n_fft
    else:
        n_frames = int(np.ceil((len(x) - cfg.n_fft) / cfg.hop)) + 1
        target = (n_frames - 1) * cfg.hop + cfg.n_fft
    padded = np.pad(x, (0, target - len(x)))
    starts = np.arange(0, target - cfg.n_fft + 1, cfg.hop)
    frames = np.stack([padded[s:s + cfg.n_fft] for s in starts], axis=1)
    frames *= np.hanning(cfg.n_fft)[:, None]
    X = np.fft.rfft(frames, axis=0)
    freqs = np.fft.rfftfreq(cfg.n_fft, 1 / fs)
    times = (starts + cfg.n_fft / 2) / fs
    return X, freqs, times


def stft16(x: np.ndarray, fs: int, cfg: FeatureConfig):
    """返回 `(16,F,T)`；100000 点按规格右补零 352 点得到 193 帧。"""
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[0] != 16:
        raise ValueError("同步输入必须为 [16,N]")
    channels = [_stft_mono(ch, fs, cfg) for ch in x]
    return (np.stack([item[0] for item in channels]), channels[0][1], channels[0][2])


def mel_filterbank(fs: int, cfg: FeatureConfig) -> np.ndarray:
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)

    points = mel_to_hz(np.linspace(hz_to_mel(cfg.fmin), hz_to_mel(cfg.fmax), cfg.n_mels + 2))
    freqs = np.fft.rfftfreq(cfg.n_fft, 1 / fs)
    bank = np.zeros((cfg.n_mels, len(freqs)))
    for m in range(1, cfg.n_mels + 1):
        left, center, right = points[m - 1:m + 2]
        bank[m - 1] = np.maximum(0.0, np.minimum(
            (freqs - left) / (center - left + 1e-20),
            (right - freqs) / (right - center + 1e-20),
        ))
    return bank


class LogMelExtractor:
    """逐 Mel 频带归一化；只有 fit 修改统计量，推理时保持冻结。"""

    def __init__(self, fs: int = 100_000, cfg: FeatureConfig = None):
        self.fs = fs
        self.cfg = cfg or FeatureConfig()
        self.bank = mel_filterbank(fs, self.cfg)
        self.mean_ = None
        self.std_ = None

    def _raw(self, x: np.ndarray) -> np.ndarray:
        X, _, _ = _stft_mono(x, self.fs, self.cfg)
        power = np.abs(X) ** 2
        return np.log(self.bank @ power + 1e-10)

    def fit(self, x) -> "LogMelExtractor":
        waves = [x] if isinstance(x, np.ndarray) else list(x)
        if not waves:
            raise ValueError("归一化拟合需要训练波形")
        raw = np.concatenate([self._raw(w) for w in waves], axis=1)
        self.mean_ = raw.mean(axis=1, keepdims=True)
        self.std_ = np.maximum(raw.std(axis=1, keepdims=True), 1e-6)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("必须先用训练集 fit 归一化统计")
        feat = (self._raw(x) - self.mean_) / self.std_
        return feat[None].astype(np.float32)
