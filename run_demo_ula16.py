# -*- coding: utf-8 -*-
"""16 通道新方案的可复现基线演示；全部输入和输出均为仿真数据。"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sim.abc_sources import generate_source
from sim.backgrounds import generate_background
from sim.ula_sim import simulate_session
from src.classify.baseline import LogMelLogisticBaseline
from src.config import default_water_config
from src.detect.cfar import cfar_detect
from src.doa.nearfield import (
    build_grid, build_steering, estimate_bearing, select_band_frequencies, spatial_spectra,
)
from src.features.logmel import LogMelExtractor, stft16

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _fit_baseline(cfg, seed=20261122):
    """基线训练与测试都使用 ch7 同分布仿真窗，不用单声道干净源冒充阵列特征。"""
    waves, labels = [], []
    for class_index, class_id in enumerate("ABC"):
        for variant in range(12):
            rng = np.random.default_rng(seed + class_index * 100 + variant)
            source = generate_source(class_id, rng, duration=0.60, fs=cfg.array.fs)
            background = np.stack([0.05 * generate_background("pink", rng, 1.0, cfg.array.fs) for _ in range(16)])
            session = simulate_session(source, (0.0, 2.0), background, cfg, snr_db=20.0,
                                       session_id=f"train_{class_id}_{variant}",
                                       source_id=f"train_{class_id}_{variant}", class_id=class_id,
                                       seed=seed + class_index * 100 + variant)
            waves.append(session.audio[cfg.array.reference_channel])
            labels.append(class_id)
    extractor = LogMelExtractor(cfg.array.fs, cfg.features).fit(waves)
    return LogMelLogisticBaseline(extractor).fit(waves, labels)


def _json_bearing(result):
    return {"valid": bool(result["valid"]), "reason": result["reason"],
            "angle_deg": result["angle_deg"], "peak_ratio": result["peak_ratio"]}


def run_demo(output_dir="outputs/ula16_demo"):
    """真实执行 CFAR→近场等权 CBF→ch7 log-Mel 逻辑回归并保存图和 JSON。"""
    from run_pipeline import fixed_analysis_window

    cfg = default_water_config()
    fs = cfg.array.fs
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    classifier = _fit_baseline(cfg)
    grid = build_grid(cfg)

    background_rng = np.random.default_rng(9100)
    background = np.stack([0.05 * generate_background("pink", background_rng, 1.0, fs)
                           for _ in range(16)])
    background_X, freqs, _ = stft16(background, fs, cfg.features)
    background_power = np.full(len(freqs), np.median(np.abs(background_X) ** 2) * 0.1)

    scenarios = (("A", -25.0), ("B", 5.0), ("C", 30.0))
    records = []
    figure, axes = plt.subplots(3, 2, figsize=(13, 10))
    for row, (class_id, angle) in enumerate(scenarios):
        rng = np.random.default_rng(9200 + row)
        source = generate_source(class_id, rng, duration=0.60, fs=fs)
        xy = (2.0 * np.tan(np.deg2rad(angle)), 2.0)
        bg = np.stack([0.05 * generate_background("pink", rng, 1.0, fs) for _ in range(16)])
        session = simulate_session(source, xy, bg, cfg, snr_db=20.0,
                                   session_id=f"sim_{class_id}", source_id=f"demo_{class_id}",
                                   class_id=class_id, seed=9200 + row)
        alarms, _ = cfar_detect(session.audio[cfg.array.reference_channel], fs,
                                f_range=(cfg.features.fmin, cfg.features.fmax))
        alarm_records = []
        first_spectrum = None
        for alarm_index, alarm in enumerate(alarms):
            window = fixed_analysis_window(session.audio, alarm, fs, cfg.features.window_samples)
            X, window_freqs, _ = stft16(window, fs, cfg.features)
            selected, _ = select_band_frequencies(X, window_freqs, background_power, cfg)
            selected_freqs = window_freqs[np.concatenate(selected)] if any(len(s) for s in selected) else np.array([])
            steering = build_steering(grid, selected_freqs, cfg)
            spectra = spatial_spectra(X, window_freqs, steering, background_power, cfg, method="cbf")
            bearing = estimate_bearing(spectra)
            # 逻辑回归是 ch7 对照基线；方向聚焦输入由智能 TinySwin 路径使用。
            probabilities = classifier.predict_proba(window[cfg.array.reference_channel])
            class_index = int(np.argmax(probabilities))
            item = {"alarm_index": alarm_index,
                    "t_start_s": alarm["t_start"], "t_end_s": alarm["t_end"],
                    "detection_stat_db": alarm["detection_stat_db"],
                    "bearing": _json_bearing(bearing),
                    "predicted_class": "ABC"[class_index],
                    "class_probabilities": probabilities.tolist(),
                    "classifier_input": "reference_channel_ch7",
                    "probability_semantics": "uncalibrated_simulation_model_score"}
            alarm_records.append(item)
            if first_spectrum is None:
                first_spectrum = bearing["angle_spectrum"]
        record = {"session_id": session.session_id, "data_mode": "simulation",
                  "true_class": class_id, "true_angle_deg": session.events[0].theta_deg,
                  "set_snr_db": 20.0, "alarms": alarm_records}
        records.append(record)

        time_axis = np.arange(session.audio.shape[1]) / fs
        axes[row, 0].plot(time_axis, session.audio[cfg.array.reference_channel], linewidth=0.6)
        axes[row, 0].set_title(f"{class_id} 类｜参考通道 ch7｜设定方位 {session.events[0].theta_deg:.1f}°")
        axes[row, 0].set_xlabel("时间 / s")
        axes[row, 0].set_ylabel("幅度")
        if first_spectrum is not None and np.isfinite(first_spectrum).any():
            axes[row, 1].plot(grid.angles, first_spectrum, color="tab:blue")
        axes[row, 1].axvline(session.events[0].theta_deg, color="tab:green", linestyle="--", label="设定方位")
        if alarm_records and alarm_records[0]["bearing"]["angle_deg"] is not None:
            axes[row, 1].axvline(alarm_records[0]["bearing"]["angle_deg"], color="tab:red",
                                 linestyle=":", label="CBF 估计")
        axes[row, 1].set_title("近场等权 CBF 方位谱（首个告警）")
        axes[row, 1].set_xlabel("方位角 / °")
        axes[row, 1].set_ylabel("归一化空间谱")
        axes[row, 1].legend(loc="best")
    figure.suptitle("听水·知微｜16 通道线阵基线链路｜仿真数据，非实测性能", fontsize=15)
    figure.tight_layout()
    figure_path = output_dir / "baseline_overview.png"
    figure.savefig(figure_path, dpi=150)
    plt.close(figure)

    payload = {"project": "听水·知微", "data_mode": "simulation",
               "config_id": cfg.config_id, "array_channels": 16,
               "method": "CFAR + nearfield_equal_weight_CBF + ch7_log-Mel_logistic_regression",
               "seed": 20261122, "sessions": records,
               "limitations": ["仅验证固定仿真场景代码链路", "分类概率未经校准",
                               "未进行25轮全量训练", "未进行实录或硬件验收"]}
    result_path = output_dir / "baseline_results.json"
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 68)
    print("听水·知微｜16 通道线阵基线演示（仿真数据）")
    print("CFAR → 近场等权 CBF → ch7 log-Mel＋逻辑回归")
    print("=" * 68)
    for record in records:
        print(f"{record['session_id']}｜设定 {record['true_class']} / {record['true_angle_deg']:.1f}°｜"
              f"告警 {len(record['alarms'])} 段")
        for alarm in record["alarms"]:
            angle = alarm["bearing"]["angle_deg"]
            angle_text = "方位不确定" if angle is None else f"{angle:.1f}°"
            print(f"  {alarm['t_start_s']:.3f}–{alarm['t_end_s']:.3f} s｜{angle_text}｜"
                  f"预测 {alarm['predicted_class']}｜模型得分 {max(alarm['class_probabilities']):.3f}")
    print(f"结果：{result_path.resolve()}")
    print(f"图件：{figure_path.resolve()}")
    print("边界：以上仅为仿真链路输出，不代表实测性能或方案指标达标。")
    return payload


if __name__ == "__main__":
    run_demo()
