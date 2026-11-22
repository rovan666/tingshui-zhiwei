# -*- coding: utf-8 -*-
"""听水·知微新方案唯一主入口：16 通道线阵、连续事件和显式模型选择。

基线演示：python run_pipeline.py --demo --method baseline
智能演示：python run_pipeline.py --demo --method intelligent \
  --detector-model models/ula16/os_scl.pt \
  --focus-model models/ula16/learned_focus.pt \
  --classifier-model models/ula16/tiny_swin.pt
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np


def configuration_hash(cfg):
    """对实际生效的完整配置做稳定哈希，而不是假定始终使用水中配置文件。"""
    serialized = json.dumps(asdict(cfg), ensure_ascii=False, sort_keys=True,
                            separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def fixed_analysis_window(x_all, alarm, fs, samples):
    """以告警中心截取定长同步窗口，边界平移，短输入仅在末尾补零。"""
    x_all = np.asarray(x_all)
    if x_all.ndim != 2 or x_all.shape[0] != 16:
        raise ValueError("分析窗输入必须为 [16,N]")
    samples = int(samples)
    if samples <= 0:
        raise ValueError("分析窗长度必须为正")
    if x_all.shape[1] < samples:
        x_all = np.pad(x_all, ((0, 0), (0, samples - x_all.shape[1])))
    center = (float(alarm["t_start"]) + float(alarm["t_end"])) / 2.0
    start = int(round(center * fs)) - samples // 2
    start = min(max(0, start), x_all.shape[1] - samples)
    return x_all[:, start:start + samples]


def load_intelligent_checkpoints(detector_path, focus_path, classifier_path):
    """先检查三类权重，再以受限反序列化加载；任何缺失都禁止退回基线。"""
    import torch

    paths = {"detector": detector_path, "learned_focus": focus_path, "classifier": classifier_path}
    for name, path in paths.items():
        if path is None or not Path(path).is_file():
            raise FileNotFoundError(f"智能模式缺少 {name} 权重：{path or '未提供'}")
    payloads = {}
    for name, path in paths.items():
        payloads[name] = torch.load(path, map_location="cpu", weights_only=True)
    return payloads


def _required(payload, keys, name):
    missing = [key for key in keys if key not in payload]
    if missing:
        raise ValueError(f"{name} 权重缺少推理字段：{', '.join(missing)}")


class FocusedEventClassifier:
    def __init__(self, cfg, model, extractor, threshold):
        from src.classify.tiny_swin import EventClassifier
        self.cfg, self.model, self.extractor = cfg, model, extractor
        self.events = EventClassifier(self._predict, threshold=threshold)

    def _predict(self, wave):
        import torch
        feature = torch.from_numpy(self.extractor.transform(wave))
        with torch.no_grad():
            return self.model.predict_proba(feature[None])[0].numpy()

    def __call__(self, window, bearing, detection_score):
        from src.doa.focus import focus_event
        if bearing.get("valid"):
            wave = focus_event(window, self.cfg.array.fs, self.cfg, bearing["_steering"],
                               bearing["point_index"], bearing["_selected"], bearing["_freqs"])
        else:
            wave = window[self.cfg.array.reference_channel]
        self.events.update(wave, detection_score)
        return {"label": "unknown", "probability": 0.0, "classification_pending": True}

    def close(self):
        return self.events.close()

    def reset(self):
        self.events.windows.clear()


def _feature_extractor(cfg, payload, name):
    from src.features.logmel import LogMelExtractor
    extractor = LogMelExtractor(cfg.array.fs, cfg.features)
    extractor.mean_ = np.asarray(payload["feature_mean"], dtype=np.float32).reshape(cfg.features.n_mels, 1)
    extractor.std_ = np.asarray(payload["feature_std"], dtype=np.float32).reshape(cfg.features.n_mels, 1)
    if np.any(extractor.std_ <= 0):
        raise ValueError(f"{name} 的 log-Mel 归一化标准差必须为正")
    return extractor


def build_intelligent_engine(payloads, cfg=None):
    """由三份真实权重构造连续推理引擎，不创建随机权重替代模型。"""
    import torch
    from src.classify.tiny_swin import TinySwinClassifier
    from src.config import default_water_config
    from src.detect.event_gate import EventGate
    from src.detect.os_scl import OSSCLModel, prototype_score
    from src.doa.learned_focus import BandWeightNet, covariance_bands
    from src.doa.nearfield import (
        build_grid, build_steering, estimate_bearing, select_band_frequencies, spatial_spectra,
    )
    from src.features.logmel import stft16
    from src.pipeline import InferenceEngine

    torch.set_num_threads(1)
    cfg = cfg or default_water_config()
    detector_payload = payloads["detector"]
    focus_payload = payloads["learned_focus"]
    classifier_payload = payloads["classifier"]
    _required(detector_payload, ("state_dict", "prototypes", "feature_mean", "feature_std",
                                 "tau_on", "tau_off", "model_version"), "OS-SCL")
    _required(focus_payload, ("state_dict", "background_power", "peak_ratio_threshold",
                              "model_version"), "学习式测向")
    _required(classifier_payload, ("state_dict", "threshold", "feature_mean", "feature_std",
                                   "model_version"), "TinySwin")

    detector_extractor = _feature_extractor(cfg, detector_payload, "OS-SCL")
    prototypes = detector_payload["prototypes"]
    prototypes = prototypes.numpy() if hasattr(prototypes, "numpy") else np.asarray(prototypes)
    detector_model = OSSCLModel(n_classes=len(prototypes))
    detector_model.load_state_dict(detector_payload["state_dict"])
    detector_model.eval()

    def detector(window):
        feature = torch.from_numpy(detector_extractor.transform(window[cfg.array.reference_channel]))
        with torch.no_grad():
            embedding = detector_model(feature[None])["raw_embedding"].numpy()
        return float(prototype_score(embedding, prototypes)[0])

    gate = EventGate(float(detector_payload["tau_on"]), float(detector_payload["tau_off"]),
                     fs=cfg.array.fs)
    focus_model = BandWeightNet(n_bands=cfg.features.n_doa_bands)
    focus_model.load_state_dict(focus_payload["state_dict"])
    focus_model.eval()
    background_power = np.asarray(focus_payload["background_power"], dtype=float)
    expected_bins = cfg.features.n_fft // 2 + 1
    if background_power.shape != (expected_bins,):
        raise ValueError(f"测向背景功率必须为 [{expected_bins}]")
    grid = build_grid(cfg)

    def doa(window):
        X, freqs, _ = stft16(window, cfg.array.fs, cfg.features)
        selected, _ = select_band_frequencies(X, freqs, background_power, cfg)
        selected_freqs = freqs[np.concatenate(selected)] if any(len(s) for s in selected) else np.array([])
        steering = build_steering(grid, selected_freqs, cfg)
        spectra = spatial_spectra(X, freqs, steering, background_power, cfg)
        features = covariance_bands(spectra.covariances[None], spectra.valid_bands[None])
        with torch.no_grad():
            weights = focus_model(features, torch.from_numpy(spectra.valid_bands[None]))[0].numpy()
        result = estimate_bearing(spectra, weights=weights,
                                  threshold=float(focus_payload["peak_ratio_threshold"]))
        result.update(_steering=steering, _selected=selected, _freqs=freqs)
        return result

    classifier_extractor = _feature_extractor(cfg, classifier_payload, "TinySwin")
    classifier_model = TinySwinClassifier()
    classifier_model.load_state_dict(classifier_payload["state_dict"])
    classifier_model.eval()
    classifier = FocusedEventClassifier(cfg, classifier_model, classifier_extractor,
                                        float(classifier_payload["threshold"]))
    config_hash = configuration_hash(cfg)
    versions = {"detector": detector_payload["model_version"],
                "doa": focus_payload["model_version"],
                "classifier": classifier_payload["model_version"]}
    return InferenceEngine(cfg, detector, gate, doa, classifier,
                           config_hash=config_hash, model_versions=versions)


def run_intelligent_demo(payloads, output_dir="outputs/ula16_demo"):
    """以真实权重回放一个固定仿真会话；没有权重时此函数不会被调用。"""
    from sim.abc_sources import generate_source
    from sim.backgrounds import generate_background
    from sim.ula_sim import simulate_session
    from src.pipeline import replay_session

    engine = build_intelligent_engine(payloads)
    cfg = engine.cfg
    rng = np.random.default_rng(20261122)
    source = generate_source("B", rng, duration=0.60, fs=cfg.array.fs)
    background = np.stack([0.05 * generate_background("pink", rng, 3.0, cfg.array.fs)
                           for _ in range(16)])
    session = simulate_session(source, (0.2, 2.0), background, cfg, snr_db=20.0,
                               session_id="sim_intelligent", source_id="demo_B",
                               class_id="B", seed=20261122)
    results = replay_session(session, engine)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "intelligent_results.json"
    payload = {"data_mode": "simulation", "session_id": session.session_id,
               "model_versions": engine.model_versions, "config_hash": engine.config_hash,
               "results": results,
               "warning": "模型输出不等于指标达标；实录和硬件仍需独立验收"}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for result in results:
        if result["trigger"] or result["close"] or result["status"] == "acquisition_error":
            print(f"seq={result['seq']} event={result['event_id']} status={result['status']} "
                  f"angle={result['angle_deg']} label={result['label']}")
    print(f"智能仿真回放结果：{path.resolve()}")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="运行固定种子的仿真演示")
    parser.add_argument("--method", choices=("baseline", "intelligent"), default="baseline")
    parser.add_argument("--detector-model")
    parser.add_argument("--focus-model")
    parser.add_argument("--classifier-model")
    parser.add_argument("--output-dir", default="outputs/ula16_demo")
    parser.add_argument("--ui", action="store_true", help="演示后启动 PyQt5 事件界面")
    args = parser.parse_args(argv)
    if not args.demo:
        parser.error("请显式使用 --demo；真实采集接入前不会伪装成实时系统")
    if args.method == "baseline":
        from run_demo_ula16 import run_demo
        payload = run_demo(args.output_dir)
    else:
        try:
            checkpoints = load_intelligent_checkpoints(args.detector_model, args.focus_model,
                                                       args.classifier_model)
            payload = run_intelligent_demo(checkpoints, args.output_dir)
        except (FileNotFoundError, ValueError, KeyError, RuntimeError) as exc:
            parser.error(f"智能模式无法启动：{exc}")
    if args.ui:
        from src.ui.pyqt_app import launch
        launch(payload)
    return payload


if __name__ == "__main__":
    main()
