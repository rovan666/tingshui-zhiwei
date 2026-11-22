# -*- coding: utf-8 -*-
import unittest

import numpy as np

from sim.abc_sources import generate_source
from sim.backgrounds import BACKGROUND_KINDS, generate_background
from sim.ula_sim import simulate_session
from src.config import default_water_config
from src.data.splits import split_master_ids


class ULA16SimulationDataTests(unittest.TestCase):
    def test_source_durations_and_effective_band(self):
        fs = 100_000
        for class_id in "ABC":
            with self.subTest(class_id=class_id):
                sig = generate_source(class_id, np.random.default_rng(10), 0.6, fs)
                self.assertEqual(len(sig), 60_000)
                self.assertAlmostEqual(float(np.sqrt(np.mean(sig ** 2))), 1.0, places=5)
                spectrum = np.abs(np.fft.rfft(sig * np.hanning(len(sig)))) ** 2
                freqs = np.fft.rfftfreq(len(sig), 1 / fs)
                outside = (freqs < 900) | (freqs > 6100)
                self.assertLess(float(spectrum[outside].sum() / spectrum.sum()), 1e-6)

    def test_all_eight_background_generators_are_finite(self):
        self.assertEqual(len(BACKGROUND_KINDS), 8)
        for kind in BACKGROUND_KINDS:
            x = generate_background(kind, np.random.default_rng(2), 0.1, 100_000)
            self.assertEqual(len(x), 10_000)
            self.assertTrue(np.isfinite(x).all())
            self.assertGreater(float(np.std(x)), 0.0)

    def test_reference_snr_uses_common_channel_gain(self):
        cfg = default_water_config()
        fs = cfg.array.fs
        source = generate_source("B", np.random.default_rng(4), 0.5, fs)
        rng = np.random.default_rng(7)
        background = 0.2 * rng.standard_normal((16, fs))
        result = simulate_session(source, (0.0, 2.0), background, cfg, snr_db=10.0)
        event = result.events[0]
        sl = slice(event.start_sample, event.end_sample)
        added = result.audio[7, sl] - background[7, sl]
        measured = 20 * np.log10(np.sqrt(np.mean(added ** 2)) / np.sqrt(np.mean(background[7, sl] ** 2)))
        self.assertAlmostEqual(measured, 10.0, delta=0.05)
        self.assertEqual(len(result.metadata["paths"]), 1)

    def test_fractional_delay_has_no_circular_wraparound(self):
        cfg = default_water_config()
        source = np.zeros(1000)
        source[0] = 1.0
        background = np.zeros(100_000)
        result = simulate_session(source, (0.0, 2.0), background, cfg, snr_db=0.0)
        nonzero = np.flatnonzero(np.abs(result.audio).max(axis=0) > 1e-12)
        self.assertGreaterEqual(int(nonzero[0]), 130)
        self.assertTrue(np.all(result.audio[:, :120] == 0.0))

    def test_angle_sign_and_first_order_reflection_paths(self):
        cfg = default_water_config()
        source = generate_source("A", np.random.default_rng(3), 0.5, cfg.array.fs)
        background = np.ones(100_000) * 0.1
        thetas = []
        for x in (-1.0, 0.0, 1.0):
            result = simulate_session(source, (x, 2.0), background, cfg, 0.0)
            thetas.append(result.events[0].theta_deg)
        self.assertLess(thetas[0], 0.0)
        self.assertAlmostEqual(thetas[1], 0.0)
        self.assertGreater(thetas[2], 0.0)

        reflected = simulate_session(
            source, (0.5, 1.0), background, cfg, 0.0,
            reflections={"left": 0.2, "right": 0.3, "far": 0.4},
        )
        self.assertEqual([p["kind"] for p in reflected.metadata["paths"]],
                         ["direct", "left", "right", "far"])
        self.assertEqual(len(reflected.metadata["paths"]), 4)

    def test_master_split_is_18_6_6_per_class_without_leakage(self):
        ids = {c: [f"{c}_{i:03d}" for i in range(30)] for c in "ABC"}
        split = split_master_ids(ids, seed=11)
        self.assertEqual({k: len(v) for k, v in split.items()},
                         {"train": 54, "val": 18, "test": 18})
        for c in "ABC":
            self.assertEqual([sum(x.startswith(c) for x in split[k]) for k in ("train", "val", "test")],
                             [18, 6, 6])
        all_ids = [x for values in split.values() for x in values]
        self.assertEqual(len(all_ids), len(set(all_ids)))


if __name__ == "__main__":
    unittest.main()
