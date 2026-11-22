# -*- coding: utf-8 -*-
"""新方案唯一串行训练入口：OS-SCL → 频带加权测向 → SSL＋TinySwin。"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.classify.ssl import MaskedSwinSSL, pretrain_ssl
from src.classify.tiny_swin import TinySwinClassifier, select_unknown_threshold
from src.classify.train_classifier import train_classifier
from src.detect.os_scl import OSSCLModel, fit_prototypes, train_os_scl
from src.doa.learned_focus import BandWeightNet, covariance_bands
from src.doa.train_focus import train_focus

MAX_EPOCHS = 25
BATCH_SIZE = 8


def _load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _required(data, keys, dataset):
    missing = [key for key in keys if key not in data]
    if missing:
        raise ValueError(f"{dataset} 缺少字段：{', '.join(missing)}")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_detection(dataset, output, epochs=MAX_EPOCHS, batch_size=BATCH_SIZE):
    data = _load_npz(dataset)
    keys = ("train_x", "train_y", "val_x", "val_y",
            "feature_mean", "feature_std", "tau_on", "tau_off")
    _required(data, keys, dataset)
    arrays = {key: torch.from_numpy(data[key]) for key in ("train_x", "train_y", "val_x", "val_y")}
    model = OSSCLModel(n_classes=int(arrays["train_y"].max()) + 1)
    result = train_os_scl(model, arrays["train_x"], arrays["train_y"],
                          arrays["val_x"], arrays["val_y"],
                          epochs=epochs, batch_size=batch_size)
    model.eval()
    with torch.no_grad():
        embeddings = model(arrays["train_x"])["raw_embedding"].numpy()
    prototypes = torch.from_numpy(fit_prototypes(embeddings, arrays["train_y"].numpy()))
    payload = {"state_dict": model.state_dict(), "prototypes": prototypes,
               "history": result["history"], "model_version": "os_scl_ula16_v1"}
    for key in ("feature_mean", "feature_std", "tau_on", "tau_off"):
        value = data[key]
        payload[key] = float(value) if key.startswith("tau_") else torch.from_numpy(value)
    torch.save(payload, output)
    return {"model": str(output), "epochs_ran": len(result["history"])}


def train_direction(dataset, output, epochs=MAX_EPOCHS, batch_size=BATCH_SIZE):
    data = _load_npz(dataset)
    keys = ("train_covariances", "train_spectra", "train_valid", "train_angles",
            "val_covariances", "val_spectra", "val_valid", "val_angles",
            "grid_angles", "angle_indices_json", "background_power", "peak_ratio_threshold")
    _required(data, keys, dataset)
    angle_indices = [np.asarray(indices, dtype=int)
                     for indices in json.loads(str(data["angle_indices_json"]))]
    grid_angles = torch.from_numpy(data["grid_angles"].astype(np.float32))
    train_cov = covariance_bands(data["train_covariances"], data["train_valid"])
    val_cov = covariance_bands(data["val_covariances"], data["val_valid"])
    model = BandWeightNet(n_bands=int(data["train_valid"].shape[1]))
    result = train_focus(
        model, train_cov, torch.from_numpy(data["train_spectra"].astype(np.float32)),
        torch.from_numpy(data["train_valid"].astype(bool)),
        torch.from_numpy(data["train_angles"].astype(np.float32)),
        val_cov, torch.from_numpy(data["val_spectra"].astype(np.float32)),
        torch.from_numpy(data["val_valid"].astype(bool)),
        torch.from_numpy(data["val_angles"].astype(np.float32)),
        grid_angles, angle_indices, epochs=epochs, batch_size=batch_size,
    )
    torch.save({"state_dict": model.state_dict(), "history": result["history"],
                "best_val_mae": result["best_val_mae"],
                "background_power": torch.from_numpy(data["background_power"]),
                "peak_ratio_threshold": float(data["peak_ratio_threshold"]),
                "model_version": "learned_focus_ula16_v1"}, output)
    return {"model": str(output), "epochs_ran": len(result["history"]),
            "best_val_mae_deg": result["best_val_mae"]}


def train_classification(dataset, output, epochs=MAX_EPOCHS, batch_size=BATCH_SIZE):
    data = _load_npz(dataset)
    keys = ("train_x", "train_y", "val_x", "val_y", "unknown_x",
            "feature_mean", "feature_std")
    _required(data, keys, dataset)
    train_x = torch.from_numpy(data["train_x"].astype(np.float32))
    train_y = torch.from_numpy(data["train_y"].astype(np.int64))
    val_x = torch.from_numpy(data["val_x"].astype(np.float32))
    val_y = torch.from_numpy(data["val_y"].astype(np.int64))
    unknown_x = torch.from_numpy(data["unknown_x"].astype(np.float32))
    ssl_model = MaskedSwinSSL()
    ssl_result = pretrain_ssl(ssl_model, train_x, epochs=epochs, batch_size=batch_size)
    model = TinySwinClassifier()
    model.swin.load_state_dict(ssl_model.encoder.state_dict())
    result = train_classifier(model, train_x, train_y, val_x, val_y,
                              epochs=epochs, batch_size=batch_size)
    with torch.no_grad():
        known_probability = model.predict_proba(val_x).numpy()
        unknown_probability = model.predict_proba(unknown_x).numpy()
    threshold = select_unknown_threshold(known_probability, val_y.numpy(), unknown_probability)
    torch.save({"state_dict": model.state_dict(), "threshold": threshold["threshold"],
                "threshold_calibration": threshold, "ssl_history": ssl_result["history"],
                "feature_mean": torch.from_numpy(data["feature_mean"]),
                "feature_std": torch.from_numpy(data["feature_std"]),
                "history": result["history"], "model_version": "ssl_tiny_swin_ula16_v1"}, output)
    return {"model": str(output), "ssl_epochs_ran": len(ssl_result["history"]),
            "finetune_epochs_ran": len(result["history"]),
            "unknown_threshold": threshold["threshold"], "threshold_target_met": threshold["target_met"]}


def run_training(args, stage_functions=None):
    if not 1 <= args.epochs <= MAX_EPOCHS:
        raise ValueError("每个训练阶段轮数必须为 1–25")
    if args.batch_size != BATCH_SIZE:
        raise ValueError("新方案统一训练入口固定 batch size = 8")
    torch.set_num_threads(1)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stages = (
        ("detection", args.detection_data, output_dir / "os_scl.pt"),
        ("direction", args.focus_data, output_dir / "learned_focus.pt"),
        ("classification", args.classifier_data, output_dir / "tiny_swin.pt"),
    )
    functions = stage_functions or (train_detection, train_direction, train_classification)
    manifest = {"data_mode": "training_inputs_not_independently_verified",
                "cpu_threads": 1, "batch_size": BATCH_SIZE,
                "max_epochs_per_stage": MAX_EPOCHS, "requested_epochs": args.epochs,
                "stages": []}
    for (name, dataset, output), function in zip(stages, functions):
        print(f"[训练] 开始 {name}", flush=True)
        result = function(Path(dataset), output, args.epochs, args.batch_size)
        if not output.is_file():
            raise RuntimeError(f"{name} 未生成权重文件：{output}")
        manifest["stages"].append({"name": name, "dataset": str(dataset),
                                   "dataset_sha256": _sha256(dataset),
                                   "model_sha256": _sha256(output), **result})
        (output_dir / "training_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[训练] 完成 {name} → {output}", flush=True)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detection-data", required=True)
    parser.add_argument("--focus-data", required=True)
    parser.add_argument("--classifier-data", required=True)
    parser.add_argument("--output-dir", default="models/ula16")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args(argv)
    run_training(args)


if __name__ == "__main__":
    main()
