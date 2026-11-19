# -*- coding: utf-8 -*-
"""分类模块 demo：训练轻量神经网络 + 混淆矩阵 + 各类 log-Mel 特征展示。

运行：python run_demo_classify.py
"""
import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.classify.mlp import LABELS, train
from src.classify.features import logmel
from sim.sources import GENERATORS

OUT_DIR = os.path.join("figures", "day3")
MODEL_DIR = "models"


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("=" * 60)
    print("听水·知微 | log-Mel + 轻量神经网络分类（仿真数据）")
    print("=" * 60)

    print("生成数据集并训练中（约 1–2 分钟）……")
    clf, acc, (Xt, yt) = train(seed=0)
    print(f"测试集准确率：{acc * 100:.1f}%")

    # 混淆矩阵
    pred = clf.predict(Xt)
    cm = np.zeros((4, 4), dtype=int)
    for a, b in zip(yt, pred):
        cm[a, b] += 1
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(4), LABELS)
    ax.set_yticks(range(4), LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.suptitle(f"Confusion matrix (test acc {acc*100:.1f}%, simulated)")
    fig.tight_layout()
    p1 = os.path.join(OUT_DIR, "classify_confusion.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    # 四类信号的 log-Mel 图
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6))
    for ax, label in zip(axes.ravel(), LABELS):
        S = logmel(GENERATORS[label](duration=1.0, rng=rng))
        ax.imshow(S, origin="lower", aspect="auto", cmap="magma")
        ax.set_title(label)
        ax.set_ylabel("Mel bin")
        ax.set_xlabel("Frame")
    fig.suptitle("log-Mel spectrograms of four classes (simulated)")
    fig.tight_layout()
    p2 = os.path.join(OUT_DIR, "classify_logmel.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)

    with open(os.path.join(MODEL_DIR, "mlp_classifier.pkl"), "wb") as f:
        pickle.dump(clf, f)
    print(f"已保存：{p1}\n已保存：{p2}\n模型已保存：{MODEL_DIR}/mlp_classifier.pkl")


if __name__ == "__main__":
    main()
