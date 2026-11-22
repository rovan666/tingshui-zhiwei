# -*- coding: utf-8 -*-
"""共享 CNN 学习八个频带的有效掩码 softmax 权重。"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def covariance_bands(covariances, valid_bands) -> torch.Tensor:
    covariances = np.asarray(covariances)
    valid_bands = np.asarray(valid_bands, dtype=bool)
    features = np.stack([
        covariances.real, np.abs(covariances.imag), np.sign(covariances.imag),
    ], axis=2).astype(np.float32)
    features *= valid_bands[:, :, None, None, None]
    return torch.from_numpy(features)


class BandWeightNet(nn.Module):
    def __init__(self, n_bands: int = 8):
        super().__init__()
        self.n_bands = n_bands
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.score = nn.Linear(32, 1)

    def forward(self, features, valid_bands):
        batch, bands = features.shape[:2]
        if bands != self.n_bands:
            raise ValueError("频带数与模型不一致")
        x = features.reshape(batch * bands, *features.shape[2:])
        scores = self.score(self.conv(x).flatten(1)).reshape(batch, bands)
        valid = valid_bands.bool()
        masked = scores.masked_fill(~valid, -1e9)
        weights = F.softmax(masked, dim=1) * valid
        weights = torch.where(valid.any(dim=1, keepdim=True), weights,
                              torch.zeros_like(weights))
        return weights


def soft_angle_targets(true_angles, grid_angles) -> torch.Tensor:
    true_angles = torch.as_tensor(true_angles, dtype=torch.float32)
    grid_angles = torch.as_tensor(grid_angles, dtype=torch.float32)
    targets = torch.zeros((len(true_angles), len(grid_angles)), dtype=torch.float32)
    for row, angle in enumerate(true_angles):
        if angle < grid_angles[0] or angle > grid_angles[-1]:
            raise ValueError("训练事件真值必须位于工作角域")
        right = int(torch.searchsorted(grid_angles, angle, right=False))
        if right < len(grid_angles) and grid_angles[right] == angle:
            targets[row, right] = 1.0
        else:
            left = right - 1
            fraction = (angle - grid_angles[left]) / (grid_angles[right] - grid_angles[left])
            targets[row, left] = 1 - fraction
            targets[row, right] = fraction
    return targets


def focus_cross_entropy(angle_spectrum, targets, temperature: float = 0.1):
    spectrum = torch.nan_to_num(angle_spectrum, nan=-1e9)
    log_probability = F.log_softmax(spectrum / temperature, dim=1)
    targets = targets.to(log_probability.device)
    return -(targets * log_probability).sum(dim=1).mean()


def collapse_angle_spectrum(band_spectra, weights, angle_indices):
    fused = torch.einsum("bi,bip->bp", weights, band_spectra)
    rays = []
    for indices in angle_indices:
        if len(indices):
            rays.append(fused[:, indices].max(dim=1).values)
        else:
            rays.append(torch.full((fused.shape[0],), -1e9, device=fused.device))
    return torch.stack(rays, dim=1)
