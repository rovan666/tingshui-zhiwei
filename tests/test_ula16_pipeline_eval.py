# -*- coding: utf-8 -*-
import unittest

import numpy as np

from src.config import default_water_config
from src.detect.event_gate import EventGate
from src.evaluate import evaluate_sessions, match_events
from src.pipeline import InferenceEngine, replay_session


class PipelineTests(unittest.TestCase):
    def test_acquisition_error_short_circuits_doa_and_classification(self):
        def forbidden(_):
            raise AssertionError("采集异常时不应进入测向")
        engine = InferenceEngine(
            default_water_config(), detector=lambda _: 1.0,
            gate=EventGate(0.8, 0.3, fs=100), doa=forbidden,
            classifier=forbidden, config_hash="abc123",
        )
        result = engine.process_window(np.ones((16, 100)), "s01", 7, acquisition_error=True)
        self.assertEqual(result["status"], "acquisition_error")
        self.assertEqual(result["session_id"], "s01")
        self.assertEqual(result["seq"], 7)
        self.assertIsNone(result["event_id"])
        self.assertIsNone(result["angle_deg"])
        self.assertEqual(result["label"], "unknown")

    def test_trigger_event_carries_ids_and_first_result_log(self):
        engine = InferenceEngine(
            default_water_config(), detector=lambda w: float(w[0, 0]),
            gate=EventGate(0.8, 0.3, fs=100),
            doa=lambda _: {"valid": True, "angle_deg": 12.0, "point_index": 3,
                           "peak_ratio": 2.0, "reason": "ok"},
            classifier=lambda *_: {"label": "B", "probability": 0.8},
            config_hash="cfg_hash", model_versions={"detector": "test", "doa": "cbf", "classifier": "test"},
        )
        result = None
        for seq, score in enumerate((0.9, 0.9, 0.9)):
            window = np.ones((16, 100)) * score
            result = engine.process_window(window, "s02", seq)
        self.assertTrue(result["trigger"])
        self.assertEqual(result["event_id"], 1)
        self.assertEqual(result["session_id"], "s02")
        self.assertEqual(result["seq"], 2)
        self.assertEqual(result["angle_deg"], 12.0)
        self.assertEqual(result["label"], "B")
        self.assertEqual(len(engine.event_log), 3)
        self.assertIn("elapsed_ms", engine.event_log[-1])
        self.assertEqual(engine.event_log[-1]["config_hash"], "cfg_hash")

    def test_replay_uses_one_second_windows_and_250ms_updates(self):
        cfg = default_water_config()
        audio = np.zeros((16, 200))
        engine = InferenceEngine(cfg, detector=lambda _: 0.0,
                                 gate=EventGate(0.8, 0.3, fs=100),
                                 doa=lambda _: {"valid": False}, classifier=lambda *_: None)
        from src.data.sessions import Session
        session = Session("s03", 100, audio, config_id=cfg.config_id)
        cfg.features.window_samples = 100
        cfg.features.update_samples = 25
        results = replay_session(session, engine)
        self.assertEqual([r["seq"] for r in results], [0, 1, 2, 3, 4])
        self.assertTrue(all(r["session_id"] == "s03" for r in results))
    def test_replay_ring_buffer_stores_only_nonoverlapping_new_samples(self):
        cfg = default_water_config()
        cfg.array.fs = 100
        cfg.features.window_samples = 100
        cfg.features.update_samples = 25
        audio = np.tile(np.arange(200, dtype=np.float32), (16, 1))
        engine = InferenceEngine(cfg, detector=lambda _: 0.0,
                                 gate=EventGate(0.8, 0.3, fs=100),
                                 doa=lambda _: {"valid": False}, classifier=lambda *_: None)
        from src.data.sessions import Session
        session = Session("no_duplicate_ring", 100, audio, config_id=cfg.config_id)
        replay_session(session, engine)
        np.testing.assert_array_equal(engine.gate.ring.get(), audio)


class EvaluationTests(unittest.TestCase):
    def test_first_time_match_and_duplicate_trigger(self):
        labels = [{"start_s": 1.0, "end_s": 1.5, "theta_deg": 10.0, "class_id": "A"},
                  {"start_s": 3.0, "end_s": 3.5, "theta_deg": 20.0, "class_id": "B"}]
        predictions = [{"time_s": 1.1, "angle_deg": 9.0, "label": "A"},
                       {"time_s": 1.2, "angle_deg": 9.0, "label": "A"},
                       {"time_s": 3.1, "angle_deg": None, "label": "unknown"}]
        matched, missed, false = match_events(predictions, labels)
        self.assertEqual(len(matched), 2)
        self.assertEqual(len(missed), 0)
        self.assertEqual(len(false), 1)
        self.assertIs(false[0], predictions[1])

        metrics = evaluate_sessions([{
            "mode": "simulation", "labels": labels, "predictions": predictions,
            "background_minutes": 2.0, "background_false_alarms": 1,
        }])["by_mode"]["simulation"]
        self.assertEqual(metrics["event_recall"], 1.0)
        self.assertEqual(metrics["angle_mae_deg"], 1.0)
        self.assertEqual(metrics["angle_p90_deg"], 1.0)
        self.assertEqual(metrics["angle_coverage"], 0.5)
        self.assertEqual(metrics["bearing_rejection_rate"], 0.5)
        self.assertEqual(metrics["extra_false_alarms"], 1)
        self.assertEqual(metrics["background_false_alarms_per_minute"], 0.5)


if __name__ == "__main__":
    unittest.main()
