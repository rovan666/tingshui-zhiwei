# -*- coding: utf-8 -*-
"""16 通道主入口和串行训练编排的回归测试。"""
import argparse
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np


class ULA16EntrypointTests(unittest.TestCase):
    def test_fixed_analysis_window_has_configured_length(self):
        from run_pipeline import fixed_analysis_window
        x = np.tile(np.arange(300, dtype=float), (16, 1))
        alarm = {"t_start": 0.0010, "t_end": 0.0012}
        segment = fixed_analysis_window(x, alarm, fs=100_000, samples=100)
        self.assertEqual(segment.shape, (16, 100))
        self.assertEqual(int(segment[0, 0]), 60)
        self.assertEqual(int(segment[0, -1]), 159)

    def test_intelligent_mode_requires_all_three_real_weight_files(self):
        from run_pipeline import load_intelligent_checkpoints
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            present = root / "detector.pt"
            present.write_bytes(b"not empty")
            with self.assertRaisesRegex(FileNotFoundError, "learned_focus"):
                load_intelligent_checkpoints(present, root / "learned_focus.pt",
                                             root / "tiny_swin.pt")

    def test_intelligent_mode_never_silently_uses_baseline(self):
        import run_pipeline
        with mock.patch("run_demo_ula16.run_demo") as baseline:
            with self.assertRaises(SystemExit):
                run_pipeline.main(["--demo", "--method", "intelligent"])
        baseline.assert_not_called()

    def test_config_hash_describes_the_active_medium_not_only_water_file(self):
        from run_pipeline import configuration_hash
        from src.config import load_config
        water = load_config(Path("configs/water_ula16.json"))
        air = load_config(Path("configs/air_ula16.json"))
        self.assertNotEqual(configuration_hash(water), configuration_hash(air))
        self.assertEqual(configuration_hash(water), configuration_hash(water))
        self.assertEqual(len(configuration_hash(water)), 64)


class TrainingOrchestrationTests(unittest.TestCase):
    def test_training_runs_in_order_with_eight_batch_and_epoch_cap(self):
        from train_all import run_training
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            datasets = []
            for name in ("detection", "focus", "classifier"):
                path = root / f"{name}.npz"
                path.write_bytes(name.encode())
                datasets.append(path)
            calls = []

            def fake_stage(name):
                def run(dataset, output, epochs, batch_size):
                    calls.append((name, epochs, batch_size))
                    output.write_bytes(name.encode())
                    return {"epochs_ran": epochs}
                return run

            args = argparse.Namespace(
                detection_data=datasets[0], focus_data=datasets[1],
                classifier_data=datasets[2], output_dir=root / "models",
                epochs=25, batch_size=8,
            )
            manifest = run_training(args, tuple(fake_stage(name) for name in ("detection", "direction", "classification")))
            self.assertEqual([call[0] for call in calls], ["detection", "direction", "classification"])
            self.assertTrue(all(call[1:] == (25, 8) for call in calls))
            self.assertEqual(len(manifest["stages"]), 3)
            self.assertTrue((root / "models" / "training_manifest.json").is_file())

    def test_training_rejects_more_than_twenty_five_epochs(self):
        from train_all import run_training
        args = argparse.Namespace(epochs=26, batch_size=8)
        with self.assertRaisesRegex(ValueError, "1–25"):
            run_training(args)


if __name__ == "__main__":
    unittest.main()
