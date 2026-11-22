# -*- coding: utf-8 -*-
"""八种可控背景伪类，仅用于训练和校准，不代表真实海域覆盖。"""
import numpy as np

from sim.abc_sources import _bandlimit, _normalize

BACKGROUND_KINDS = (
    "white", "pink", "brown", "tonal",
    "modulated", "impulsive", "narrowband", "mixed",
)


def _colored_noise(rng, n, fs, exponent):
    raw = rng.standard_normal(n)
    X = np.fft.rfft(raw)
    f = np.fft.rfftfreq(n, 1 / fs)
    scale = np.ones_like(f)
    mask = f > 0
    scale[mask] = (f[mask] / 1000.0) ** (-exponent / 2)
    return np.fft.irfft(X * scale, n=n)


def generate_background(kind: str, rng: np.random.Generator,
                        duration: float, fs: int = 100_000) -> np.ndarray:
    if kind not in BACKGROUND_KINDS:
        raise ValueError(f"未知背景伪类: {kind}")
    n = int(round(duration * fs))
    t = np.arange(n) / fs
    if kind in {"white", "mixed"}:
        x = rng.standard_normal(n)
    elif kind == "pink":
        x = _colored_noise(rng, n, fs, 1.0)
    elif kind == "brown":
        x = _colored_noise(rng, n, fs, 2.0)
    elif kind == "tonal":
        x = sum(np.sin(2 * np.pi * rng.uniform(1000, 6000) * t + rng.uniform(0, 2 * np.pi))
                for _ in range(4))
    elif kind == "modulated":
        x = _colored_noise(rng, n, fs, 1.0)
        x *= 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(1, 8) * t) ** 2
    elif kind == "impulsive":
        x = 0.1 * rng.standard_normal(n)
        for _ in range(int(rng.integers(2, 8))):
            center = int(rng.integers(0, n))
            width = min(int(rng.uniform(0.005, 0.03) * fs), n - center)
            if width > 1:
                x[center:center + width] += rng.standard_normal(width) * np.hanning(width)
    else:
        center = rng.uniform(1200, 5500)
        x = _colored_noise(rng, n, fs, 1.0)
        X = np.fft.rfft(x)
        f = np.fft.rfftfreq(n, 1 / fs)
        X[np.abs(f - center) > 250] = 0
        x = np.fft.irfft(X, n=n)
    if kind == "mixed":
        x += 0.5 * _colored_noise(rng, n, fs, 1.0)
        x += 0.2 * np.sin(2 * np.pi * 1700 * t)
    return _normalize(_bandlimit(x, fs)).astype(np.float32)
