# -*- coding: utf-8 -*-
"""TinySwin 监督微调与按母素材/会话分组的交叉预测接口。"""
import copy
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.model_selection import GroupKFold

from src.classify.tiny_swin import _macro_f1


def train_classifier(model, train_x, train_y, val_x, val_y,
                     epochs: int = 25, batch_size: int = 8,
                     lr: float = 1e-3, weight_decay: float = 1e-4):
    torch.set_num_threads(1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_score, best_state, history = -1.0, None, []
    for epoch in range(min(epochs, 25)):
        model.train()
        losses = []
        for start in range(0, len(train_y), batch_size):
            end = start + batch_size
            logits = model(train_x[start:end])
            loss = F.cross_entropy(logits, train_y[start:end])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            predicted = model(val_x).argmax(dim=1).cpu().numpy()
        truth = np.array(["ABC"[int(y)] for y in val_y])
        pred = np.array(["ABC"[int(y)] for y in predicted])
        score = _macro_f1(truth, pred)
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "val_macro_f1": score})
        if score > best_score:
            best_score, best_state = score, copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"model": model, "history": history, "best_val_macro_f1": best_score}


def grouped_three_fold_predict(items, groups, train_fn, predict_fn):
    """每折训练函数只收到训练索引，确保同组窗口不会跨训练和预测。"""
    items = np.asarray(items)
    groups = np.asarray(groups)
    output = np.empty(len(items), dtype=object)
    for train_indices, held_out in GroupKFold(n_splits=3).split(items, groups=groups):
        if np.intersect1d(groups[train_indices], groups[held_out]).size:
            raise RuntimeError("交叉预测出现组泄漏")
        model = train_fn(train_indices)
        output[held_out] = predict_fn(model, held_out)
    return output
