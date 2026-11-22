# -*- coding: utf-8 -*-
"""A/B/C 三类人工合成母素材，全部限制在 1–6 kHz。"""
import numpy as np


def _bandlimit(x, fs, fmin=1000.0, fmax=6000.0):
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    X[(freqs < fmin) | (freqs > fmax)] = 0.0
    return np.fft.irfft(X, n=len(x))


def _normalize(x):
    return x / (np.sqrt(np.mean(x ** 2)) + 1e-20)


def _pulse_train(rng, n, fs):
    x = np.zeros(n)
    count = int(rng.integers(3, 6))
    cursor = int(rng.uniform(0.01, 0.06) * fs)
    for _ in range(count):
        width = int(rng.uniform(0.020, 0.050) * fs)
        if cursor + width >= n:
            break
        noise = _bandlimit(rng.standard_normal(width), fs)
        pulse = noise * np.hanning(width)
        x[cursor:cursor + width] += pulse
        gap = int(rng.uniform(0.060, 0.110) * fs)
        cursor += width + gap
    return x


def _chirp(rng, n, fs):
    duration = n / fs
    f0 = float(rng.uniform(1000, 5000))
    direction = rng.choice([-1.0, 1.0])
    f1 = f0 + direction * float(rng.uniform(1000, 3000))
    f1 = float(np.clip(f1, 1000, 6000))
    if abs(f1 - f0) < 1000:
        f1 = 6000 if f0 <= 3000 else 1000
    t = np.arange(n) / fs
    k = (f1 - f0) / duration
    phase = 2 * np.pi * (f0 * t + 0.5 * k * t ** 2) + rng.uniform(0, 2 * np.pi)
    x = np.sin(phase)
    fade = min(int(0.010 * fs), n // 2)
    ramp = np.sin(np.linspace(0, np.pi / 2, fade)) ** 2
    x[:fade] *= ramp
    x[-fade:] *= ramp[::-1]
    return x


def _am_harmonic(rng, n, fs):
    t = np.arange(n) / fs
    f0 = float(rng.uniform(1000, 2000))
    phase = rng.uniform(0, 2 * np.pi)
    x = np.zeros(n)
    harmonic = 1
    while harmonic * f0 < 6000:
        x += np.sin(2 * np.pi * harmonic * f0 * t + harmonic * phase) / harmonic
        harmonic += 1
    fm = float(rng.uniform(6, 10))
    x *= 0.55 + 0.45 * np.sin(2 * np.pi * fm * t + rng.uniform(0, 2 * np.pi))
    fade = min(int(0.010 * fs), n // 2)
    ramp = np.sin(np.linspace(0, np.pi / 2, fade)) ** 2
    x[:fade] *= ramp
    x[-fade:] *= ramp[::-1]
    return x


def generate_source(class_id: str, rng: np.random.Generator,
                    duration: float = 0.6, fs: int = 100_000) -> np.ndarray:
    if not 0.45 <= duration <= 0.70:
        raise ValueError("A/B/C 母素材时长必须在 0.45–0.70 秒")
    n = int(round(duration * fs))
    if class_id == "A":
        x = _pulse_train(rng, n, fs)
    elif class_id == "B":
        x = _chirp(rng, n, fs)
    elif class_id == "C":
        x = _am_harmonic(rng, n, fs)
    else:
        raise ValueError("class_id 必须为 A/B/C")
    return _normalize(_bandlimit(x, fs)).astype(np.float32)
