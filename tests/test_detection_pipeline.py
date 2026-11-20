# -*- coding: utf-8 -*-
"""检测与全链路场景的关键回归测试。"""
import unittest

import numpy as np

from run_pipeline import FS, add_source
from sim.sources import fish
from src.detect.cfar import cfar_detect


class DetectionPipelineTests(unittest.TestCase):
    def test_add_source_writes_late_event_at_requested_time(self):
        """晚于源信号时长的事件也必须真实写入目标时间窗。"""
        n = int(6.0 * FS)
        x_all = np.zeros((4, n))
        rng = np.random.default_rng(3)
        sig = fish(duration=1.0, rng=rng)
        start = int(4.0 * FS)

        add_source(x_all, sig, bearing_deg=120.0, t0=4.0, snr_db=0.0, rng=rng)

        written = x_all[:, start:start + len(sig)]
        self.assertGreater(float(np.std(written)), 0.5)
        self.assertEqual(int(np.count_nonzero(x_all[:, :start])), 0)
        self.assertEqual(int(np.count_nonzero(x_all[:, start + len(sig):])), 0)

    def test_cfar_detects_low_frequency_line_at_band_edge(self):
        """300 Hz 弱线谱不能因参考窗触及低频边界而漏检。"""
        rng = np.random.default_rng(21)
        n = int(4.0 * FS)
        t = np.arange(n) / FS
        x = 0.5 * rng.standard_normal(n)
        mask = (t >= 1.0) & (t <= 3.0)
        x[mask] += 0.2 * np.sin(2 * np.pi * 300.0 * t[mask])

        alarms, _ = cfar_detect(x, FS, pfa=1e-2)

        self.assertTrue(any(a["t_start"] < 3.0 and a["t_end"] > 1.0 for a in alarms))

    def test_cfar_detects_sparse_fish_clicks(self):
        """短促咔哒串至少应形成一个覆盖事件窗口的告警。"""
        rng = np.random.default_rng(11)
        n = int(4.0 * FS)
        x = 0.5 * rng.standard_normal(n)
        sig = fish(duration=1.5, rng=rng)
        start = int(1.0 * FS)
        x[start:start + len(sig)] += sig

        alarms, _ = cfar_detect(x, FS, pfa=1e-2)

        self.assertTrue(any(a["t_start"] < 2.5 and a["t_end"] > 1.0 for a in alarms))

    def test_doa_band_uses_mean_channel_power(self):
        """选频应按频率轴比较功率，反相通道也不能互相抵消。"""
        try:
            from run_pipeline import select_doa_band
        except ImportError:
            self.fail("缺少统一的 DOA 选频函数")
        freqs = np.array([1000.0, 10_000.0, 20_000.0])
        X = np.zeros((4, 3), dtype=complex)
        X[:, 0] = 0.1
        X[:, 1] = np.array([1.0, -1.0, 1.0, -1.0])
        X[:, 2] = 0.2

        band = select_doa_band(X, freqs, half_width=1500.0)

        self.assertEqual(band, (8500.0, 11_500.0))

    def test_analysis_window_has_fixed_one_second_length(self):
        """任意长度告警都应转换成模型要求的 1 秒定长输入。"""
        try:
            from run_pipeline import analysis_window
        except ImportError:
            self.fail("缺少定长分析窗函数")
        n = int(8.0 * FS)
        x_all = np.tile(np.arange(n), (4, 1))
        alarm = dict(t_start=4.50, t_end=4.52)

        seg = analysis_window(x_all, alarm, FS)

        self.assertEqual(seg.shape, (4, int(FS)))
        expected_start = int(4.51 * FS - FS / 2)
        self.assertEqual(int(seg[0, 0]), expected_start)
        self.assertEqual(int(seg[0, -1]), expected_start + int(FS) - 1)


if __name__ == "__main__":
    unittest.main()
