# -*- coding: utf-8 -*-
"""16 通道线阵的球面波直达/一阶反射传播，不使用循环 FFT 移位。"""
import numpy as np

from src.data.sessions import EventLabel, Session


def fractional_delay(x: np.ndarray, delay_samples: float, total_n: int) -> np.ndarray:
    """线性插值分数延时；首达之前和有效段之后严格补零。"""
    index = np.arange(total_n, dtype=np.float64) - float(delay_samples)
    return np.interp(index, np.arange(len(x)), x, left=0.0, right=0.0)


def _path_record(kind, image_xy, distances, fs, c, coefficient=1.0):
    return {
        "kind": kind, "image_xy": [float(image_xy[0]), float(image_xy[1])],
        "coefficient": float(coefficient), "distances_m": distances.tolist(),
        "delay_samples": (distances / c * fs).tolist(),
    }


def simulate_session(source: np.ndarray, xy, background: np.ndarray, cfg,
                     snr_db: float, reflections: dict = None,
                     session_id: str = "sim_session", source_id: str = "source",
                     class_id: str = "unknown", seed: int = 0) -> Session:
    source = np.asarray(source, dtype=np.float64)
    bg = np.asarray(background, dtype=np.float64)
    if bg.ndim == 1:
        bg = np.tile(bg[None], (16, 1))
    if bg.ndim != 2 or bg.shape[0] != 16:
        raise ValueError("背景必须为 [N] 或 [16,N]")
    positions = cfg.array.positions()
    source_xy = np.asarray(xy, dtype=np.float64)
    if source_xy[1] <= 0:
        raise ValueError("声源必须位于阵列内侧 y>0")
    distances = np.linalg.norm(positions - source_xy, axis=1)
    fs, c = cfg.array.fs, cfg.array.sound_speed
    paths = [("direct", source_xy, distances, 1.0)]
    reflections = reflections or {}
    x, y = source_xy
    images = {
        "left": np.array([-cfg.field.length_m - x, y]),
        "right": np.array([cfg.field.length_m - x, y]),
        "far": np.array([x, 2 * cfg.field.width_m - y]),
    }
    for kind, coefficient in reflections.items():
        if kind not in images:
            raise ValueError(f"不支持的一阶反射面: {kind}")
        image = images[kind]
        paths.append((kind, image, np.linalg.norm(positions - image, axis=1), float(coefficient)))

    max_delay = max(float(np.max(p[2] / c * fs)) for p in paths)
    total_n = int(max(bg.shape[1], len(source) + np.ceil(max_delay) + 2))
    background_padded = np.zeros((16, total_n))
    background_padded[:, :bg.shape[1]] = bg
    direct = np.zeros((16, total_n))
    reflected = np.zeros((16, total_n))
    records = []
    for kind, image, dist, coefficient in paths:
        target = direct if kind == "direct" else reflected
        for channel in range(16):
            delay = dist[channel] / c * fs
            target[channel] += coefficient / dist[channel] * fractional_delay(source, delay, total_n)
        records.append(_path_record(kind, image, dist, fs, c, coefficient))

    ref = cfg.array.reference_channel
    start = int(np.floor(distances[ref] / c * fs))
    end = min(start + len(source), total_n)
    event_slice = slice(start, end)
    direct_rms = np.sqrt(np.mean(direct[ref, event_slice] ** 2))
    background_rms = np.sqrt(np.mean(background_padded[ref, event_slice] ** 2))
    if background_rms > 1e-12 and direct_rms > 1e-12:
        gain = background_rms * 10 ** (snr_db / 20) / direct_rms
        measured_snr = float(snr_db)
    else:
        gain = 1.0
        measured_snr = None
    audio = background_padded + gain * (direct + reflected)
    theta = float(np.rad2deg(np.arctan2(x, y)))
    label = EventLabel(
        source_id=source_id, class_id=class_id, start_sample=start, end_sample=end,
        x_m=float(x), y_m=float(y), theta_deg=theta, snr_db=float(snr_db),
        seed=seed, mode="simulation", config_id=cfg.config_id,
    )
    return Session(
        session_id=session_id, fs=fs, audio=audio.astype(np.float32), events=[label],
        mode="simulation", config_id=cfg.config_id,
        metadata={
            "paths": records, "common_gain": float(gain),
            "snr_kind": "set_direct_rms" if measured_snr is not None else "undefined_zero_background",
            "snr_measured_db": measured_snr,
            "source_duration_samples": int(len(source)),
        },
    )
