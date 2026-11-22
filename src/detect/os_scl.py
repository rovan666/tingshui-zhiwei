# -*- coding: utf-8 -*-
"""CNN + FPH + ArcFace + 投影头的 OS-SCL 检测模型与损失。"""
import copy
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class FPH(nn.Module):
    """特征扰动头，为 Noisy-ArcMix 提供扰动嵌入。"""
    def __init__(self, dim: int, scale: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.Tanh())
        self.scale = scale

    def forward(self, x):
        return x + self.scale * self.net(x)


class ArcFaceHead(nn.Module):
    def __init__(self, in_features: int, n_classes: int = 8,
                 scale: float = 16.0, margin: float = 0.2):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.scale, self.margin = scale, margin

    def forward(self, embedding, labels=None):
        cosine = F.linear(F.normalize(embedding), F.normalize(self.weight))
        cosine = cosine.clamp(-1 + 1e-6, 1 - 1e-6)
        if labels is None:
            return self.scale * cosine
        theta = torch.acos(cosine)
        target = torch.cos(theta + self.margin)
        one_hot = F.one_hot(labels, self.weight.shape[0]).bool()
        logits = torch.where(one_hot, target, cosine)
        return self.scale * logits


class OSSCLModel(nn.Module):
    def __init__(self, n_classes: int = 8, embedding_dim: int = 32, projection_dim: int = 16):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, embedding_dim, 3, padding=1), nn.BatchNorm2d(embedding_dim), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.fph = FPH(embedding_dim)
        self.arcface = ArcFaceHead(embedding_dim, n_classes)
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim), nn.ReLU(),
            nn.Linear(embedding_dim, projection_dim),
        )

    def forward(self, x, labels=None, perturb: bool = False):
        raw = self.backbone(x)
        embedding = self.fph(raw) if perturb else raw
        return {
            "raw_embedding": raw, "embedding": embedding,
            "projection": F.normalize(self.projection(raw), dim=1),
            "logits": self.arcface(embedding, labels),
        }


def supervised_contrastive_loss(projections, labels, temperature: float = 0.1):
    projections = F.normalize(projections, dim=1)
    labels = labels.reshape(-1)
    n = len(labels)
    similarity = projections @ projections.T / temperature
    self_mask = torch.eye(n, dtype=torch.bool, device=labels.device)
    positive = labels[:, None].eq(labels[None, :]) & ~self_mask
    losses = []
    for anchor in range(n):
        if not positive[anchor].any():
            continue
        logits = similarity[anchor][~self_mask[anchor]]
        logits = logits - logits.max().detach()
        log_prob = logits - torch.logsumexp(logits, dim=0)
        losses.append(-log_prob[positive[anchor][~self_mask[anchor]]].mean())
    if not losses:
        return projections.sum() * 0.0
    return torch.stack(losses).mean()


def noisy_arcmix_loss(embeddings, labels, arc_head, lam: float = 0.5, permutation=None):
    if permutation is None:
        permutation = torch.arange(len(labels) - 1, -1, -1, device=labels.device)
    mixed = lam * embeddings + (1 - lam) * embeddings[permutation]
    loss_a = F.cross_entropy(arc_head(mixed, labels), labels)
    labels_b = labels[permutation]
    loss_b = F.cross_entropy(arc_head(mixed, labels_b), labels_b)
    return lam * loss_a + (1 - lam) * loss_b


def fit_prototypes(embeddings, labels) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=float)
    labels = np.asarray(labels)
    classes = np.unique(labels)
    prototypes = np.zeros((len(classes), embeddings.shape[1]))
    for index, label in enumerate(classes):
        mean = embeddings[labels == label].mean(axis=0)
        prototypes[index] = mean / max(np.linalg.norm(mean), 1e-20)
    return prototypes


def prototype_score(embeddings, prototypes) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=float)
    norm = np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-20)
    z = embeddings / norm
    return 1.0 - np.max(z @ np.asarray(prototypes).T, axis=1)


def softmax_baseline_score(logits, temperature: float = 5.0) -> np.ndarray:
    logits = np.asarray(logits, dtype=float) / temperature
    probability = np.exp(logits - logits.max(axis=1, keepdims=True))
    probability /= probability.sum(axis=1, keepdims=True)
    return 1.0 - probability.max(axis=1)


def _background_batches(labels, batch_size, rng):
    labels = np.asarray(labels)
    classes = np.unique(labels)
    if batch_size % 2 or batch_size < 8:
        raise ValueError("OS-SCL 批大小至少为 8 且为偶数")
    n_groups = batch_size // 2
    n_steps = int(np.ceil(len(labels) / batch_size))
    for _ in range(n_steps):
        selected = rng.choice(classes, size=min(n_groups, len(classes)), replace=False)
        indices = []
        for label in selected:
            candidates = np.flatnonzero(labels == label)
            if len(candidates) < 2:
                raise ValueError("每个背景伪类至少需要两个窗口")
            indices.extend(rng.choice(candidates, size=2, replace=False))
        yield np.asarray(indices)


def train_os_scl(model, features, labels, val_features, val_labels,
                 epochs: int = 25, batch_size: int = 8, lr: float = 1e-3,
                 weight_decay: float = 1e-4, seed: int = 0, device: str = "cpu"):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_accuracy, best_state, history = -float("inf"), None, []
    for epoch in range(epochs):
        model.train()
        losses = []
        for indices in _background_batches(labels, batch_size, rng):
            x = features[torch.as_tensor(indices)].to(device)
            y = labels[torch.as_tensor(indices)].to(device)
            output = model(x, labels=y, perturb=True)
            loss = supervised_contrastive_loss(output["projection"], y)
            lam = float(rng.beta(2, 2))
            loss = loss + noisy_arcmix_loss(output["embedding"], y, model.arcface, lam=lam)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            correct = total = 0
            for start in range(0, len(val_labels), batch_size):
                out = model(val_features[start:start + batch_size].to(device))
                pred = out["logits"].argmax(dim=1).cpu()
                correct += int((pred == val_labels[start:start + batch_size]).sum())
                total += len(pred)
        accuracy = correct / max(total, 1)
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)),
                        "val_accuracy": float(accuracy)})
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"model": model, "history": history, "best_val_accuracy": best_accuracy}
