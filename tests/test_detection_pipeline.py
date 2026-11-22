# -*- coding: utf-8 -*-
"""新方案保留的 CFAR 基线回归测试；主链路的 ULA16 测试见独立文件。"""
import unittest

import numpy as np

from sim.abc_sources import generate_source
from src.detect.cfar import cfar_detect

FS = 100_000.0


class DetectionPipelineTests(unittest.TestCase):
    def test_cfar_exposes_detection_margin_and_frame_hits(self):
        rng = np.random.default_rng(8)
        n = int(4.0 * FS)
        t = np.arange(n) / FS
        x = 0.5 * rng.standard_normal(n)
        mask = (t >= 1.5) & (t <= 2.5)
        x[mask] += 0.3 * np.sin(2 * np.pi * 5_000.0 * t[mask])
        alarms, dbg = cfar_detect(x, FS, pfa=1e-2)
        self.assertTrue(alarms)
        self.assertIn("detection_stat_db", alarms[0])
        self.assertNotIn("snr_est", alarms[0])
        self.assertIn("frame_hit", dbg)
        self.assertEqual(dbg["th_db"], 0.0)
        self.assertEqual(dbg["pfa_spectral"], 0.005)
        self.assertEqual(dbg["pfa_energy"], 0.005)

    def test_cfar_detects_low_frequency_line_at_band_edge(self):
        rng = np.random.default_rng(21)
        n = int(4.0 * FS)
        t = np.arange(n) / FS
        x = 0.5 * rng.standard_normal(n)
        mask = (t >= 1.0) & (t <= 3.0)
        x[mask] += 0.2 * np.sin(2 * np.pi * 300.0 * t[mask])
        alarms, _ = cfar_detect(x, FS, pfa=1e-2)
        self.assertTrue(any(a["t_start"] < 3.0 and a["t_end"] > 1.0 for a in alarms))

    def test_cfar_detects_sparse_a_class_clicks(self):
        rng = np.random.default_rng(11)
        n = int(2.0 * FS)
        x = 0.05 * rng.standard_normal(n)
        sig = generate_source("A", rng, duration=0.60, fs=FS)
        start = int(0.5 * FS)
        x[start:start + len(sig)] += sig
        alarms, _ = cfar_detect(x, FS, pfa=1e-2)
        self.assertTrue(any(a["t_start"] < 1.1 and a["t_end"] > 0.5 for a in alarms))

    def test_cfar_white_noise_frame_pfa_stays_within_budget(self):
        false_frames, total_frames = 0, 0
        for seed in range(2000, 2040):
            x = np.random.default_rng(seed).standard_normal(int(4.0 * FS))
            _, dbg = cfar_detect(x, FS, pfa=1e-2)
            false_frames += int(np.count_nonzero(dbg["frame_hit"]))
            total_frames += len(dbg["frame_hit"])
        self.assertLess(false_frames / total_frames, 0.01)


if __name__ == "__main__":
    unittest.main()
