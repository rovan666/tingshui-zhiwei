# -*- coding: utf-8 -*-
"""TinySwin 双视图自监督：30% 谱块掩码重建与 NT-Xent。"""
import torch
from torch import nn
from torch.nn import functional as F

from src.classify.tiny_swin import TinySwinClassifier


def mask_spectrogram(x, mask_ratio: float = 0.30, block_size: int = 8, generator=None):
    if not 0 <= mask_ratio <= 1:
        raise ValueError("遮盖比例必须位于 [0,1]")
    mask = torch.ones_like(x)
    height, width = x.shape[-2:]
    grid_h = (height + block_size - 1) // block_size
    grid_w = (width + block_size - 1) // block_size
    block_area = block_size * block_size
    for sample in range(len(x)):
        target = int(round(height * width * mask_ratio))
        full_blocks, remainder = divmod(target, block_area)
        order = torch.randperm(grid_h * grid_w, generator=generator)
        for block_index in order[:full_blocks]:
            row, col = divmod(int(block_index), grid_w)
            mask[sample, :, row * block_size:min((row + 1) * block_size, height),
                 col * block_size:min((col + 1) * block_size, width)] = 0
        if remainder:
            block_index = int(order[full_blocks])
            row, col = divmod(block_index, grid_w)
            block = mask[sample, :, row * block_size:min((row + 1) * block_size, height),
                         col * block_size:min((col + 1) * block_size, width)]
            block_width = block.shape[-1]
            for linear_index in range(remainder):
                block[..., linear_index // block_width, linear_index % block_width] = 0
    return x * mask, mask


def nt_xent_loss(z1, z2, temperature: float = 0.1):
    z = torch.cat([F.normalize(z1, dim=1), F.normalize(z2, dim=1)], dim=0)
    n = len(z1)
    similarity = z @ z.T / temperature
    similarity.fill_diagonal_(-1e9)
    positives = torch.arange(2 * n, device=z.device)
    positives = (positives + n) % (2 * n)
    return F.cross_entropy(similarity, positives)


class MaskedSwinSSL(nn.Module):
    def __init__(self, projection_dim: int = 64):
        super().__init__()
        self.encoder = TinySwinClassifier(num_classes=3).swin
        channels = self.encoder.head.in_features
        self.projection = nn.Sequential(
            nn.Linear(channels, 256), nn.ReLU(), nn.Linear(256, projection_dim),
        )
        decoder = []
        in_channels = channels
        for out_channels in (256, 128, 64, 32, 16):
            decoder.extend([nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1), nn.ReLU()])
            in_channels = out_channels
        decoder.append(nn.Conv2d(16, 1, 3, padding=1))
        self.decoder = nn.Sequential(*decoder)

    def _encode(self, x):
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        features = self.encoder.features(x)
        features = self.encoder.norm(features)
        feature_map = features.permute(0, 3, 1, 2)
        pooled = features.mean(dim=(1, 2))
        projection = F.normalize(self.projection(pooled), dim=1)
        return self.decoder(feature_map), projection

    def forward(self, view1, view2):
        recon1, projection1 = self._encode(view1)
        recon2, projection2 = self._encode(view2)
        return {"recon1": recon1, "projection1": projection1,
                "recon2": recon2, "projection2": projection2}


def augment_views(x, noise_std: float = 0.03):
    gain1 = torch.empty((len(x), 1, 1, 1), device=x.device).uniform_(0.8, 1.2)
    gain2 = torch.empty((len(x), 1, 1, 1), device=x.device).uniform_(0.8, 1.2)
    return x * gain1 + noise_std * torch.randn_like(x), x * gain2 + noise_std * torch.randn_like(x)


def masked_reconstruction_loss(reconstruction, target, mask):
    target = F.interpolate(target, size=(224, 224), mode="bilinear", align_corners=False)
    mask = F.interpolate(mask, size=(224, 224), mode="nearest")
    return (((reconstruction - target) ** 2) * (1 - mask)).sum() / ((1 - mask).sum() + 1e-8)


def pretrain_ssl(model, features, epochs: int = 25, batch_size: int = 8,
                 lr: float = 1e-3, weight_decay: float = 1e-4):
    torch.set_num_threads(1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
    for epoch in range(min(epochs, 25)):
        model.train()
        losses = []
        for start in range(0, len(features), batch_size):
            batch = features[start:start + batch_size]
            if len(batch) < 2:
                continue
            view1, view2 = augment_views(batch)
            masked1, mask1 = mask_spectrogram(view1)
            masked2, mask2 = mask_spectrogram(view2)
            output = model(masked1, masked2)
            reconstruction = masked_reconstruction_loss(output["recon1"], view1, mask1)
            reconstruction = reconstruction + masked_reconstruction_loss(output["recon2"], view2, mask2)
            loss = reconstruction + nt_xent_loss(output["projection1"], output["projection2"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append({"epoch": epoch + 1, "loss": float(sum(losses) / max(len(losses), 1))})
    return {"model": model, "history": history}
