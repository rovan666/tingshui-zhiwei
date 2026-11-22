# -*- coding: utf-8 -*-
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.config import default_water_config, doa_fmax, load_config
from src.data.sessions import EventLabel, Session, load_session, save_session
from src.features.logmel import LogMelExtractor, stft16


class ULA16ConfigFeatureTests(unittest.TestCase):
    def test_water_geometry_and_air_alias_limit(self):
        cfg = default_water_config()
        self.assertEqual(cfg.array.n_channels, 16)
        self.assertEqual(cfg.array.reference_channel, 7)
        self.assertEqual(cfg.array.fs, 100_000)
        np.testing.assert_allclose(cfg.array.positions()[0], [-0.375, 0.0])
        np.testing.assert_allclose(cfg.array.positions()[-1], [0.375, 0.0])
        np.testing.assert_allclose(cfg.array.positions()[7], [-0.025, 0.0])
        self.assertAlmostEqual(doa_fmax(cfg), 6000.0)

        air = load_config(Path("configs/air_ula16.json"))
        self.assertAlmostEqual(doa_fmax(air), 3430.0)

    def test_stft_right_padding_produces_193_frames(self):
        cfg = default_water_config().features
        x = np.zeros((16, 100_000), dtype=np.float32)
        x[7, 1000] = 1.0
        X, freqs, times = stft16(x, 100_000, cfg)
        self.assertEqual(X.shape, (16, 1025, 193))
        self.assertEqual(freqs.shape, (1025,))
        self.assertEqual(times.shape, (193,))
        self.assertEqual(times[0], 0.01024)
        self.assertGreater(np.abs(X[7]).max(), 0.0)

    def test_logmel_shape_and_frozen_training_normalization(self):
        cfg = default_water_config().features
        rng = np.random.default_rng(4)
        train = rng.standard_normal(100_000)
        extractor = LogMelExtractor(100_000, cfg).fit(train)
        mean = extractor.mean_.copy()
        std = extractor.std_.copy()
        feat = extractor.transform(train)
        self.assertEqual(feat.shape, (1, 128, 193))
        self.assertLess(abs(float(feat.mean())), 1e-5)
        extractor.transform(rng.standard_normal(100_000))
        np.testing.assert_array_equal(extractor.mean_, mean)
        np.testing.assert_array_equal(extractor.std_, std)

    def test_session_wav_json_round_trip_preserves_labels(self):
        event = EventLabel(
            source_id="A_000", class_id="A", start_sample=1000, end_sample=2000,
            x_m=0.5, y_m=2.0, theta_deg=float(np.rad2deg(np.arctan2(0.5, 2.0))),
            snr_db=10.0, seed=3, mode="simulation", config_id="water_test",
        )
        audio = np.linspace(-2.0, 2.0, 16 * 2000, dtype=np.float32).reshape(16, 2000)
        session = Session(
            session_id="s001", fs=100_000, audio=audio, events=[event],
            mode="simulation", config_id="water_test", metadata={"medium": "water"},
        )
        with tempfile.TemporaryDirectory() as tmp:
            save_session(session, tmp)
            loaded = load_session(Path(tmp) / "s001.json")
            self.assertEqual(loaded.session_id, "s001")
            self.assertEqual(loaded.events[0].class_id, "A")
            self.assertEqual(loaded.events[0].start_sample, 1000)
            self.assertEqual(loaded.events[0].end_sample, 2000)
            self.assertEqual(loaded.metadata["medium"], "water")
            self.assertGreater(loaded.metadata["pcm_scale"], 1.0)
            np.testing.assert_allclose(loaded.audio, audio, atol=2e-4)
            payload = json.loads((Path(tmp) / "s001.json").read_text(encoding="utf-8"))
            self.assertEqual(len(payload["waveform_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
