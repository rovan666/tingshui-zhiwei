# -*- coding: utf-8 -*-
"""矩形内侧近场候选网格、球面导向和宽带 CBF/MUSIC。"""
from dataclasses import dataclass
import numpy as np

from src.config import SystemConfig, doa_fmax


@dataclass
class CandidateGrid:
    points: np.ndarray
    angles: np.ndarray
    angle_indices: tuple
    point_angle: np.ndarray


@dataclass
class SteeringTable:
    frequencies: np.ndarray
    values: np.ndarray
    grid: CandidateGrid
    config_id: str


@dataclass
class SpatialSpectra:
    band_spectra: np.ndarray
    valid_bands: np.ndarray
    selected: tuple
    covariances: np.ndarray
    grid: CandidateGrid


def build_grid(cfg: SystemConfig) -> CandidateGrid:
    field = cfg.field
    angles = np.arange(field.angle_min_deg, field.angle_max_deg + 1e-9,
                       field.angle_step_deg)
    points, point_angle, per_angle = [], [], []
    for angle_index, angle in enumerate(angles):
        ray = []
        radius = field.range_min_m
        while True:
            theta = np.deg2rad(angle)
            x, y = radius * np.sin(theta), radius * np.cos(theta)
            if (-field.length_m / 2 <= x <= field.length_m / 2
                    and 0 < y <= field.width_m):
                ray.append(len(points))
                points.append([x, y])
                point_angle.append(angle_index)
            elif radius > np.hypot(field.length_m, field.width_m) * 1.5:
                break
            if radius > field.range_min_m and y > field.width_m:
                break
            radius += field.range_step_m
        per_angle.append(np.asarray(ray, dtype=int))
    return CandidateGrid(
        points=np.asarray(points, dtype=float).reshape(-1, 2), angles=angles,
        angle_indices=tuple(per_angle), point_angle=np.asarray(point_angle, dtype=int),
    )


def build_steering(grid: CandidateGrid, frequencies, cfg: SystemConfig) -> SteeringTable:
    frequencies = np.asarray(frequencies, dtype=float)
    positions = cfg.array.positions()
    ref = cfg.array.reference_channel
    if len(grid.points) == 0:
        values = np.zeros((len(frequencies), 0, 16), dtype=complex)
    else:
        # distances: P×M，参考距离：P。
        distances = np.linalg.norm(grid.points[:, None, :] - positions[None, :, :], axis=2)
        ref_distance = distances[:, ref]
        relative = (distances - ref_distance[:, None]) / cfg.array.sound_speed
        amplitude = ref_distance[:, None] / np.maximum(distances, 1e-12)
        phase = -2j * np.pi * frequencies[:, None, None] * relative[None]
        values = amplitude[None] * np.exp(phase)
        values /= np.maximum(np.linalg.norm(values, axis=2, keepdims=True), 1e-20)
    return SteeringTable(frequencies, values, grid, cfg.config_id)


def select_band_frequencies(X: np.ndarray, freqs: np.ndarray,
                            background_power: np.ndarray, cfg: SystemConfig):
    """每带按阵列平均能量选至多两个高于训练背景中位数的频点。"""
    if X.ndim != 3 or X.shape[0] != 16:
        raise ValueError("频域输入必须为 [16,F,T]")
    power = np.mean(np.abs(X) ** 2, axis=(0, 2))
    background_power = np.broadcast_to(background_power, power.shape)
    edges = np.linspace(cfg.features.fmin, doa_fmax(cfg), cfg.features.n_doa_bands + 1)
    selected, valid = [], []
    for low, high in zip(edges[:-1], edges[1:]):
        candidates = np.flatnonzero((freqs >= low) & (freqs < high)
                                    & (power > background_power))
        if len(candidates):
            candidates = candidates[np.argsort(power[candidates])[-2:][::-1]]
        selected.append(candidates.astype(int))
        valid.append(bool(len(candidates)))
    return tuple(selected), np.asarray(valid, dtype=bool)


def spatial_spectra(X: np.ndarray, freqs: np.ndarray, steering: SteeringTable,
                    background_power: np.ndarray, cfg: SystemConfig,
                    method: str = "cbf") -> SpatialSpectra:
    selected, valid = select_band_frequencies(X, freqs, background_power, cfg)
    n_bands = cfg.features.n_doa_bands
    n_points = len(steering.grid.points)
    band_spectra = np.zeros((n_bands, n_points))
    covariances = np.zeros((n_bands, 16, 16), dtype=complex)
    steering_lookup = {float(f): i for i, f in enumerate(steering.frequencies)}
    for band_index, indices in enumerate(selected):
        if not len(indices):
            continue
        per_frequency = []
        band_covariance = []
        for freq_index in indices:
            key = float(freqs[freq_index])
            if key not in steering_lookup:
                raise ValueError(f"导向表缺少频点 {key} Hz")
            a = steering.values[steering_lookup[key]]
            snapshots = X[:, freq_index, :]
            R = snapshots @ snapshots.conj().T / snapshots.shape[1]
            R = R / (np.trace(R).real + 1e-20)
            band_covariance.append(R)
            if method == "cbf":
                spectrum = np.einsum("pm,mn,pn->p", a.conj(), R, a).real
            elif method == "music":
                _, evecs = np.linalg.eigh(R)
                noise_space = evecs[:, :15]
                projection = a.conj() @ noise_space
                spectrum = 1.0 / (np.sum(np.abs(projection) ** 2, axis=1) + 1e-20)
            else:
                raise ValueError("method 必须为 cbf 或 music")
            spectrum = np.maximum(spectrum.real, 0.0)
            per_frequency.append(spectrum / (spectrum.max() + 1e-20))
        band_spectra[band_index] = np.mean(per_frequency, axis=0)
        covariances[band_index] = np.mean(band_covariance, axis=0)
    return SpatialSpectra(band_spectra, valid, selected, covariances, steering.grid)


def estimate_bearing(spectra: SpatialSpectra, weights=None, threshold: float = None) -> dict:
    valid = spectra.valid_bands.copy()
    result = {
        "valid": False, "reason": "no_valid_frequency", "angle_deg": None,
        "point_index": None, "peak_ratio": 0.0,
        "angle_spectrum": np.full(len(spectra.grid.angles), np.nan),
    }
    if not valid.any() or not len(spectra.grid.points):
        return result
    if weights is None:
        weights = valid.astype(float) / valid.sum()
    else:
        weights = np.asarray(weights, dtype=float) * valid
        if weights.sum() <= 0:
            return result
        weights /= weights.sum()
    fused = np.einsum("b,bp->p", weights, spectra.band_spectra)
    angle_spectrum = np.full(len(spectra.grid.angles), np.nan)
    best_points = np.full(len(spectra.grid.angles), -1, dtype=int)
    for angle_index, indices in enumerate(spectra.grid.angle_indices):
        if len(indices):
            local = int(np.argmax(fused[indices]))
            angle_spectrum[angle_index] = fused[indices[local]]
            best_points[angle_index] = indices[local]
    finite = np.isfinite(angle_spectrum)
    if not finite.any():
        return result
    peak_angle_index = int(np.nanargmax(angle_spectrum))
    peak = float(angle_spectrum[peak_angle_index])
    comparison = finite & (np.abs(spectra.grid.angles - spectra.grid.angles[peak_angle_index]) >= 5.0)
    if not comparison.any():
        result["reason"] = "no_comparison_angle"
        return result
    ratio = peak / (float(np.max(angle_spectrum[comparison])) + 1e-20)
    result.update(
        angle_spectrum=angle_spectrum, peak_ratio=float(ratio),
        angle_deg=float(spectra.grid.angles[peak_angle_index]),
        point_index=int(best_points[peak_angle_index]), reason="ok",
    )
    if threshold is not None and ratio < threshold:
        result["reason"] = "peak_ratio_below_threshold"
        return result
    result["valid"] = True
    return result
