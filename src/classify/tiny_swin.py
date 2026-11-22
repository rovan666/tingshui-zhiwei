# -*- coding: utf-8 -*-
"""Torchvision TinySwin 三分类、事件 top-3 平均和未知拒识。"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import swin_t


def _macro_f1(truth, predicted):
    scores = []
    for label in ("A", "B", "C"):
        tp = sum(t == label and p == label for t, p in zip(truth, predicted))
        fp = sum(t != label and p == label for t, p in zip(truth, predicted))
        fn = sum(t == label and p != label for t, p in zip(truth, predicted))
        scores.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(scores))


class TinySwinClassifier(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.swin = swin_t(weights=None, num_classes=num_classes)
        original = self.swin.features[0][0]
        self.swin.features[0][0] = nn.Conv2d(
            1, original.out_channels, original.kernel_size, original.stride,
            original.padding, bias=original.bias is not None,
        )

    def forward(self, x):
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return self.swin(x)

    @torch.no_grad()
    def predict_proba(self, x):
        self.eval()
        return F.softmax(self(x), dim=1)


def select_unknown_threshold(known_probabilities, known_labels, unknown_probabilities) -> dict:
    known = np.asarray(known_probabilities)
    unknown = np.asarray(unknown_probabilities)
    labels = np.asarray(known_labels)
    candidates = []
    for threshold in np.round(np.arange(0.50, 0.951, 0.05), 2):
        predicted = np.array(["ABC"[int(np.argmax(p))] if np.max(p) >= threshold else "unknown"
                              for p in known])
        truth = np.array(["ABC"[int(y)] for y in labels])
        unknown_far = float(np.mean(np.max(unknown, axis=1) >= threshold))
        candidates.append({
            "threshold": float(threshold), "unknown_false_acceptance": unknown_far,
            "macro_f1": _macro_f1(truth, predicted),
        })
    valid = [c for c in candidates if c["unknown_false_acceptance"] <= 0.10]
    if valid:
        best = max(valid, key=lambda c: (c["macro_f1"], -c["unknown_false_acceptance"]))
    else:
        best = min(candidates, key=lambda c: (c["unknown_false_acceptance"], -c["macro_f1"]))
    best["target_met"] = bool(valid)
    best["candidates"] = candidates
    return best


class EventClassifier:
    """predict_fn 接收任意模型窗口并返回长度为 3 的类别概率。"""
    def __init__(self, predict_fn, threshold: float = 0.5):
        self.predict_fn = predict_fn
        self.threshold = threshold
        self.windows = []

    def update(self, model_input, detection_score: float):
        probability = np.asarray(self.predict_fn(model_input), dtype=float)
        if probability.shape != (3,):
            raise ValueError("分类器必须返回 A/B/C 三类概率")
        self.windows.append((float(detection_score), probability))
        self.windows.sort(key=lambda item: item[0], reverse=True)
        self.windows = self.windows[:3]

    def close(self) -> dict:
        if not self.windows:
            return {"label": "unknown", "probability": 0.0, "n_windows": 0}
        probability = np.mean([item[1] for item in self.windows], axis=0)
        index = int(np.argmax(probability))
        confidence = float(probability[index])
        result = {
            "label": "ABC"[index] if confidence >= self.threshold else "unknown",
            "probability": confidence, "n_windows": len(self.windows),
            "class_probabilities": probability.tolist(),
        }
        self.windows.clear()
        return result
