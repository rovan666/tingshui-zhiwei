# -*- coding: utf-8 -*-
"""轻量神经网络分类器（MLP，log-Mel 特征）。

当前用 sklearn MLPClassifier 实现两层感知机；接口与后续 CNN 版本保持一致，
替换模型文件即可升级（框架文档中 classify 模块）。
"""
import numpy as np
from sklearn.neural_network import MLPClassifier

from src.classify.features import logmel

LABELS = ["ship", "human", "fish", "noise"]


def extract_feature(x_mono: np.ndarray, fs: float = 96_000.0) -> np.ndarray:
    """log-Mel 图拉平成一维特征向量。"""
    return logmel(x_mono, fs).ravel()


def make_dataset(per_class: int = 200, seed: int = 0):
    """用仿真声源生成器构造数据集。返回 (X, y, X_test, y_test)。"""
    from sim.sources import GENERATORS
    rng = np.random.default_rng(seed)
    X, y = [], []
    for label in LABELS:
        gen = GENERATORS[label]
        for _ in range(per_class):
            sig = gen(duration=1.0, rng=rng)
            snr_db = rng.uniform(-3, 12)           # 随机噪声强度，提高鲁棒性
            noise = rng.standard_normal(len(sig))
            scale = np.sqrt(np.mean(sig ** 2) / (10 ** (snr_db / 10.0)))
            x = sig + scale * noise
            X.append(extract_feature(x))
            y.append(LABELS.index(label))
    X, y = np.array(X), np.array(y)
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]
    split = int(0.8 * len(y))
    return X[:split], y[:split], X[split:], y[split:]


def train(seed: int = 0):
    X, y, Xt, yt = make_dataset(seed=seed)
    clf = MLPClassifier(hidden_layer_sizes=(128,), activation="relu",
                        max_iter=300, random_state=seed)
    clf.fit(X, y)
    acc = clf.score(Xt, yt)
    return clf, acc, (Xt, yt)


def classify(x_mono: np.ndarray, clf, fs: float = 96_000.0):
    """返回 (label, prob)。"""
    f = extract_feature(x_mono, fs)[None, :]
    prob = clf.predict_proba(f)[0]
    k = int(np.argmax(prob))
    return LABELS[k], float(prob[k])
