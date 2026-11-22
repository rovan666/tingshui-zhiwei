# -*- coding: utf-8 -*-
import unittest

import numpy as np

from src.detect.event_gate import EventGate, SyncRingBuffer, calibrate_gate

try:
    import torch
except ImportError:
    torch = None


class EventGateTests(unittest.TestCase):
    def test_two_of_three_opens_three_low_closes_and_middle_holds(self):
        gate = EventGate(tau_on=0.8, tau_off=0.3)
        self.assertFalse(gate.update(0.9)["trigger"])
        self.assertFalse(gate.update(0.1)["trigger"])
        opened = gate.update(0.9)
        self.assertTrue(opened["trigger"])
        self.assertEqual(opened["status"], "event_active")
        self.assertTrue(gate.update(0.5)["active"])
        self.assertFalse(gate.update(0.0)["close"])
        self.assertFalse(gate.update(0.0)["close"])
        self.assertTrue(gate.update(0.0)["close"])
        self.assertEqual(gate.update(0.0)["status"], "listening")

    def test_discontinuity_clears_state_and_ring_buffer(self):
        gate = EventGate(0.8, 0.3, fs=100, prebuffer_s=2.0)
        gate.update(0.9, np.ones((16, 100)))
        gate.update(0.9, np.ones((16, 100)))
        error = gate.update(0.9, np.ones((16, 100)), acquisition_error=True)
        self.assertEqual(error["status"], "acquisition_error")
        self.assertFalse(error["active"])
        self.assertIsNone(error["event_id"])
        self.assertEqual(gate.ring.size, 0)
        self.assertFalse(gate.update(0.9)["trigger"])

    def test_ring_buffer_keeps_exactly_two_seconds(self):
        ring = SyncRingBuffer(fs=100, seconds=2.0)
        ring.append(np.ones((16, 100)))
        ring.append(np.ones((16, 100)) * 2)
        ring.append(np.ones((16, 100)) * 3)
        data = ring.get()
        self.assertEqual(data.shape, (16, 200))
        np.testing.assert_array_equal(data[:, :100], 2)
        np.testing.assert_array_equal(data[:, 100:], 3)

    def test_calibration_prefers_recall_under_false_alarm_budget(self):
        rng = np.random.default_rng(3)
        backgrounds = [rng.uniform(0.0, 0.5, 80) for _ in range(2)]
        events = [np.array([0.1, 0.95, 0.95, 0.2, 0.2, 0.2]) for _ in range(3)]
        result = calibrate_gate(backgrounds, events)
        self.assertTrue(result["target_met"])
        self.assertLessEqual(result["false_alarms_per_minute"], 1.0)
        self.assertEqual(result["event_recall"], 1.0)
        self.assertLess(result["tau_off"], result["tau_on"])


@unittest.skipIf(torch is None, "项目虚拟环境提供 PyTorch 后运行")
class OSSCLModelTests(unittest.TestCase):
    def test_projection_head_receives_supervised_contrastive_gradient(self):
        from src.detect.os_scl import OSSCLModel, supervised_contrastive_loss
        model = OSSCLModel(n_classes=8, embedding_dim=16, projection_dim=8)
        x = torch.randn(4, 1, 128, 193)
        labels = torch.tensor([0, 0, 1, 1])
        output = model(x, labels=labels)
        loss = supervised_contrastive_loss(output["projection"], labels)
        loss.backward()
        grad = model.projection[0].weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_noisy_arcmix_is_lambda_weighted_two_label_loss(self):
        from src.detect.os_scl import ArcFaceHead, noisy_arcmix_loss
        head = ArcFaceHead(6, 8)
        embeddings = torch.randn(4, 6, requires_grad=True)
        labels = torch.tensor([0, 1, 2, 3])
        permutation = torch.tensor([1, 0, 3, 2])
        lam = 0.25
        mixed = lam * embeddings + (1 - lam) * embeddings[permutation]
        expected = lam * torch.nn.functional.cross_entropy(head(mixed, labels), labels)
        expected = expected + (1 - lam) * torch.nn.functional.cross_entropy(
            head(mixed, labels[permutation]), labels[permutation])
        actual = noisy_arcmix_loss(embeddings, labels, head, lam=lam, permutation=permutation)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
        actual.backward()
        self.assertGreater(float(embeddings.grad.abs().sum()), 0.0)

    def test_normalized_prototype_distance_score(self):
        from src.detect.os_scl import fit_prototypes, prototype_score
        embeddings = np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
        labels = np.array([0, 0, 1, 1])
        prototypes = fit_prototypes(embeddings, labels)
        scores = prototype_score(np.array([[1.0, 0.0], [0.0, 1.0]]), prototypes)
        np.testing.assert_allclose(scores, [0.0, 0.0], atol=1e-7)

    def test_one_epoch_training_runs_and_validates(self):
        from src.detect.os_scl import OSSCLModel, train_os_scl
        torch.set_num_threads(1)
        model = OSSCLModel(n_classes=8, embedding_dim=8, projection_dim=4)
        features = torch.randn(16, 1, 128, 193)
        labels = torch.arange(8).repeat_interleave(2)
        result = train_os_scl(model, features, labels, features, labels, epochs=1)
        self.assertEqual(len(result["history"]), 1)
        self.assertTrue(np.isfinite(result["history"][0]["val_accuracy"]))


if __name__ == "__main__":
    unittest.main()
