# -*- coding: utf-8 -*-
"""学习式宽带空间聚焦训练，验证集按角度 MAE 选权重。"""
import copy
import numpy as np
import torch

from src.doa.learned_focus import (
    collapse_angle_spectrum, focus_cross_entropy, soft_angle_targets,
)


def _forward(model, covariances, spectra, valid, grid_angles, angle_indices):
    weights = model(covariances, valid)
    return collapse_angle_spectrum(spectra, weights, angle_indices), weights


def train_focus(model, covariances, spectra, valid, angles,
                val_covariances, val_spectra, val_valid, val_angles,
                grid_angles, angle_indices, epochs: int = 25, batch_size: int = 8,
                lr: float = 1e-3, weight_decay: float = 1e-4):
    torch.set_num_threads(1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    targets = soft_angle_targets(angles, grid_angles)
    best_mae, best_state, history = float("inf"), None, []
    for epoch in range(min(epochs, 25)):
        model.train()
        losses = []
        for start in range(0, len(angles), batch_size):
            end = start + batch_size
            angle_spectrum, _ = _forward(model, covariances[start:end], spectra[start:end],
                                         valid[start:end], grid_angles, angle_indices)
            loss = focus_cross_entropy(angle_spectrum, targets[start:end])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            val_spectrum, _ = _forward(model, val_covariances, val_spectra, val_valid,
                                       grid_angles, angle_indices)
            predicted = grid_angles[val_spectrum.argmax(dim=1)]
            mae = float((predicted - val_angles).abs().mean())
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "val_mae": mae})
        if mae < best_mae:
            best_mae = mae
            best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"model": model, "history": history, "best_val_mae": best_mae}
