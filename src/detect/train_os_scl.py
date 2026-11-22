# -*- coding: utf-8 -*-
"""命令行：从 NPZ log-Mel 训练 OS-SCL，输入键为 train_x/train_y/val_x/val_y。"""
import argparse
from pathlib import Path
import numpy as np
import torch

from src.detect.os_scl import OSSCLModel, fit_prototypes, train_os_scl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    data = np.load(args.dataset)
    arrays = {k: torch.from_numpy(data[k]) for k in ("train_x", "train_y", "val_x", "val_y")}
    model = OSSCLModel(n_classes=int(arrays["train_y"].max()) + 1)
    result = train_os_scl(model, arrays["train_x"], arrays["train_y"],
                          arrays["val_x"], arrays["val_y"],
                          epochs=min(args.epochs, 25), batch_size=args.batch_size)
    model.eval()
    with torch.no_grad():
        embeddings = model(arrays["train_x"])["raw_embedding"].numpy()
    prototypes = fit_prototypes(embeddings, arrays["train_y"].numpy())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "prototypes": prototypes,
                "history": result["history"], "model_version": "os_scl_v1"}, args.output)


if __name__ == "__main__":
    main()
