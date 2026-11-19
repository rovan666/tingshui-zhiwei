# -*- coding: utf-8 -*-
"""多类声源信号生成器（供分类模块训练/测试用）。

三类目标 + 背景噪声：
- ship  ：船舶噪声——低频线谱（固定音高谐波）+ 螺旋桨幅度调制
- human ：人员落水拍水——宽带脉冲串（1–8 kHz 突发噪声，衰减包络）
- fish  ：鱼类活动——高频弱短促咔哒声（12–20 kHz）
"""
import numpy as np

FS = 96_000.0


def _noise(n, rng):
    return rng.standard_normal(n)


def _bandpass_noise(n, rng, f_lo, f_hi, fs=FS):
    """用频域滤波生成带限噪声。"""
    x = rng.standard_normal(n)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / fs)
    X[(f < f_lo) | (f > f_hi)] = 0.0
    y = np.fft.irfft(X, n)
    return y / (np.std(y) + 1e-12)


def ship(duration=1.0, rng=None):
    rng = rng or np.random.default_rng()
    n = int(duration * FS)
    t = np.arange(n) / FS
    x = np.zeros(n)
    for k, f0 in enumerate([150.0, 300.0, 450.0]):
        x += (0.8 ** k) * np.sin(2 * np.pi * f0 * t + rng.uniform(0, 2 * np.pi))
    x *= 1.0 + 0.4 * np.sin(2 * np.pi * 8.0 * t)   # 螺旋桨调制
    x += 0.15 * _bandpass_noise(n, rng, 100, 3000)
    return x / (np.std(x) + 1e-12)


def human(duration=1.0, rng=None):
    rng = rng or np.random.default_rng()
    n = int(duration * FS)
    t = np.arange(n) / FS
    x = np.zeros(n)
    for _ in range(rng.integers(3, 6)):            # 3–5 次拍水
        t0 = rng.uniform(0.05, duration - 0.15)
        tau = rng.uniform(0.02, 0.05)
        env = np.exp(-np.maximum(t - t0, 0) / tau) * (t >= t0)
        x += env * _bandpass_noise(n, rng, 1000, 8000) * rng.uniform(0.8, 1.2)
    return x / (np.std(x) + 1e-12)


def fish(duration=1.0, rng=None):
    rng = rng or np.random.default_rng()
    n = int(duration * FS)
    t = np.arange(n) / FS
    x = np.zeros(n)
    for _ in range(rng.integers(10, 18)):           # 一串高频咔哒
        t0 = rng.uniform(0.02, duration - 0.03)
        f0 = rng.uniform(12_000, 20_000)
        env = np.exp(-np.maximum(t - t0, 0) / 0.003) * (t >= t0)
        x += 0.6 * env * np.sin(2 * np.pi * f0 * t)
    x += 0.05 * _noise(n, rng)
    return x / (np.std(x) + 1e-12)


def background(duration=1.0, rng=None):
    rng = rng or np.random.default_rng()
    n = int(duration * FS)
    x = _bandpass_noise(n, rng, 50, 24_000)
    return x / (np.std(x) + 1e-12)


GENERATORS = dict(ship=ship, human=human, fish=fish, noise=background)
