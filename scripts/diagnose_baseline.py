# -*- coding: utf-8 -*-
"""输出固定基线演示的训练集与独立场景分类诊断，不产生验收指标。"""
import numpy as np

from run_demo_ula16 import _fit_baseline
from sim.abc_sources import generate_source
from sim.backgrounds import generate_background
from sim.ula_sim import simulate_session
from src.config import default_water_config

cfg = default_water_config()
classifier = _fit_baseline(cfg)
print("optimizer iterations:", classifier.model.n_iter_)
for class_index, class_id in enumerate("ABC"):
    predictions = []
    for variant in range(12):
        rng = np.random.default_rng(20261122 + class_index * 100 + variant)
        source = generate_source(class_id, rng, duration=0.60, fs=cfg.array.fs)
        background = np.stack([0.05 * generate_background("pink", rng, 1.0, cfg.array.fs)
                               for _ in range(16)])
        session = simulate_session(source, (0.0, 2.0), background, cfg, snr_db=20.0)
        probabilities = classifier.predict_proba(session.audio[cfg.array.reference_channel])
        predictions.append("ABC"[int(np.argmax(probabilities))])
    print("train", class_id, "predictions", "".join(predictions))

for row, (class_id, angle) in enumerate((("A", -25.0), ("B", 5.0), ("C", 30.0))):
    rng = np.random.default_rng(9200 + row)
    source = generate_source(class_id, rng, duration=0.60, fs=cfg.array.fs)
    xy = (2.0 * np.tan(np.deg2rad(angle)), 2.0)
    bg = np.stack([0.05 * generate_background("pink", rng, 1.0, cfg.array.fs) for _ in range(16)])
    session = simulate_session(source, xy, bg, cfg, snr_db=20.0)
    probabilities = classifier.predict_proba(session.audio[cfg.array.reference_channel])
    print("held-out scenario", class_id, probabilities)
