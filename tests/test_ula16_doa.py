# -*- coding: utf-8 -*-
import unittest

import numpy as np

from sim.abc_sources import generate_source
from sim.ula_sim import simulate_session
from src.config import default_water_config
from src.doa.focus import BearingSmoother, focus_event
from src.doa.nearfield import (
    CandidateGrid, SpatialSpectra, build_grid, build_steering, estimate_bearing,
    select_band_frequencies, spatial_spectra,
)
from src.features.logmel import stft16


class ULA16DOATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = default_water_config()
        cls.grid = build_grid(cls.cfg)

    def test_grid_is_inside_rectangle_and_steering_is_unit_norm(self):
        p = self.grid.points
        self.assertTrue(np.all(np.abs(p[:, 0]) <= self.cfg.field.length_m / 2 + 1e-9))
        self.assertTrue(np.all((p[:, 1] > 0) & (p[:, 1] <= self.cfg.field.width_m + 1e-9)))
        self.assertTrue(np.all(np.linalg.norm(p, axis=1) >= self.cfg.field.range_min_m - 1e-9))
        table = build_steering(self.grid, np.array([2000.0, 3000.0]), self.cfg)
        self.assertEqual(table.values.shape, (2, len(p), 16))
        np.testing.assert_allclose(np.linalg.norm(table.values, axis=2), 1.0, atol=1e-10)

    def test_frequency_selection_has_eight_bands_and_at_most_two_bins(self):
        freqs = np.fft.rfftfreq(2048, 1 / 100_000)
        rng = np.random.default_rng(4)
        X = rng.standard_normal((16, len(freqs), 3)) + 1j * rng.standard_normal((16, len(freqs), 3))
        selected, mask = select_band_frequencies(X, freqs, np.zeros(len(freqs)), self.cfg)
        self.assertEqual(len(selected), 8)
        self.assertEqual(len(mask), 8)
        self.assertTrue(all(len(band) <= 2 for band in selected))
        self.assertTrue(any(mask))

        empty, mask = select_band_frequencies(X, freqs, np.full(len(freqs), np.inf), self.cfg)
        self.assertFalse(any(mask))
        table = build_steering(self.grid, np.array([]), self.cfg)
        spectra = spatial_spectra(X, freqs, table, np.full(len(freqs), np.inf), self.cfg)
        self.assertFalse(estimate_bearing(spectra)["valid"])

    def test_cbf_recovers_left_center_right_and_music_center(self):
        fs = self.cfg.array.fs
        source = generate_source("B", np.random.default_rng(12), 0.6, fs)
        expected = []
        estimates = []
        saved = None
        for x in (-0.6, 0.0, 0.6):
            rng = np.random.default_rng(30)
            bg = 0.02 * rng.standard_normal((16, 100_000))
            session = simulate_session(source, (x, 2.0), bg, self.cfg, snr_db=20.0)
            X, freqs, _ = stft16(session.audio, fs, self.cfg.features)
            background_power = np.full(len(freqs), np.median(np.abs(X) ** 2) * 0.1)
            selected, _ = select_band_frequencies(X, freqs, background_power, self.cfg)
            table = build_steering(self.grid, freqs[np.concatenate(selected)], self.cfg)
            spectra = spatial_spectra(X, freqs, table, background_power, self.cfg)
            result = estimate_bearing(spectra)
            expected.append(session.events[0].theta_deg)
            estimates.append(result["angle_deg"])
            self.assertTrue(result["valid"])
            if x == 0.0:
                saved = (session, X, freqs, background_power, table, spectra, result)
        np.testing.assert_allclose(estimates, expected, atol=2.0)

        session, X, freqs, bg_power, table, _, _ = saved
        music = spatial_spectra(X, freqs, table, bg_power, self.cfg, method="music")
        self.assertAlmostEqual(estimate_bearing(music)["angle_deg"], 0.0, delta=2.0)

    def test_peak_ratio_rejection_and_three_value_median(self):
        grid = CandidateGrid(
            points=np.array([[-1.0, 1.0], [0.0, 1.0], [1.0, 1.0]]),
            angles=np.array([-30.0, 0.0, 30.0]),
            angle_indices=(np.array([0]), np.array([1]), np.array([2])),
            point_angle=np.array([0, 1, 2]),
        )
        spectra = SpatialSpectra(
            band_spectra=np.ones((8, 3)), valid_bands=np.ones(8, dtype=bool),
            selected=([],), covariances=np.zeros((8, 16, 16), complex), grid=grid,
        )
        result = estimate_bearing(spectra, threshold=1.01)
        self.assertFalse(result["valid"])
        self.assertEqual(result["reason"], "peak_ratio_below_threshold")

        smoother = BearingSmoother()
        self.assertEqual(smoother.update({"valid": True, "angle_deg": 10.0}), 10.0)
        self.assertEqual(smoother.update({"valid": True, "angle_deg": 20.0}), 15.0)
        self.assertEqual(smoother.update({"valid": True, "angle_deg": 30.0}), 20.0)
        self.assertEqual(smoother.update({"valid": False}), 20.0)
        self.assertEqual(smoother.update({"valid": True, "angle_deg": 40.0}), 30.0)

    def test_focus_event_returns_same_length_finite_waveform(self):
        fs = self.cfg.array.fs
        source = generate_source("C", np.random.default_rng(9), 0.5, fs)
        bg = 0.01 * np.random.default_rng(8).standard_normal((16, 100_000))
        session = simulate_session(source, (0.0, 2.0), bg, self.cfg, snr_db=20.0)
        X, freqs, _ = stft16(session.audio, fs, self.cfg.features)
        selected, _ = select_band_frequencies(X, freqs, np.zeros(len(freqs)), self.cfg)
        table = build_steering(self.grid, freqs[np.concatenate(selected)], self.cfg)
        spectra = spatial_spectra(X, freqs, table, np.zeros(len(freqs)), self.cfg)
        result = estimate_bearing(spectra)
        focused = focus_event(session.audio, fs, self.cfg, table, result["point_index"], selected, freqs)
        self.assertEqual(focused.shape, (100_000,))
        self.assertTrue(np.isfinite(focused).all())
        self.assertGreater(float(np.std(focused)), 0.0)


if __name__ == "__main__":
    unittest.main()
