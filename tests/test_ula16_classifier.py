# -*- coding: utf-8 -*-
import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "项目虚拟环境提供 PyTorch 后运行")
class TinySwinSSLTests(unittest.TestCase):
    def test_mask_spectrogram_masks_thirty_percent_in_blocks(self):
        from src.classify.ssl import mask_spectrogram
        x = torch.ones(2, 1, 128, 192)
        masked, mask = mask_spectrogram(x, mask_ratio=0.30, block_size=8,
                                        generator=torch.Generator().manual_seed(4))
        ratio = 1.0 - mask.float().mean().item()
        self.assertAlmostEqual(ratio, 0.30, delta=1.0 / (128 * 192))
        self.assertTrue(torch.all((masked == 0) | (masked == 1)))
        self.assertEqual(float(masked[mask == 0].sum()), 0.0)

    def test_nt_xent_is_symmetric_and_differentiable(self):
        from src.classify.ssl import nt_xent_loss
        raw1 = torch.randn(4, 8, requires_grad=True)
        raw2 = torch.randn(4, 8, requires_grad=True)
        z1 = torch.nn.functional.normalize(raw1, dim=1)
        z2 = torch.nn.functional.normalize(raw2, dim=1)
        forward = nt_xent_loss(z1, z2)
        reverse = nt_xent_loss(z2, z1)
        self.assertTrue(torch.allclose(forward, reverse, atol=1e-6))
        forward.backward()
        self.assertGreater(float(raw1.grad.abs().sum()), 0.0)

    def test_tiny_swin_real_forward_returns_three_class_logits(self):
        from src.classify.tiny_swin import TinySwinClassifier
        torch.set_num_threads(1)
        model = TinySwinClassifier(num_classes=3)
        model.eval()
        with torch.no_grad():
            logits = model(torch.randn(1, 1, 128, 193))
        self.assertEqual(tuple(logits.shape), (1, 3))
        self.assertTrue(torch.isfinite(logits).all())

    def test_masked_ssl_forward_reconstructs_224_square_spectrogram(self):
        from src.classify.ssl import MaskedSwinSSL, mask_spectrogram
        torch.set_num_threads(1)
        model = MaskedSwinSSL(projection_dim=16)
        model.eval()
        x = torch.randn(1, 1, 128, 193)
        masked, mask = mask_spectrogram(x, 0.3, generator=torch.Generator().manual_seed(2))
        with torch.no_grad():
            output = model(masked, masked)
        self.assertEqual(tuple(output["recon1"].shape), (1, 1, 224, 224))
        self.assertEqual(tuple(output["projection1"].shape), (1, 16))


class UnknownAndEventTests(unittest.TestCase):
    def test_threshold_prioritizes_unknown_false_acceptance_then_macro_f1(self):
        from src.classify.tiny_swin import select_unknown_threshold
        known = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]] * 3)
        labels = np.array([0, 1, 2] * 3)
        unknown = np.array([[0.4, 0.3, 0.3], [0.34, 0.33, 0.33]] * 3)
        result = select_unknown_threshold(known, labels, unknown)
        self.assertTrue(result["target_met"])
        self.assertLessEqual(result["unknown_false_acceptance"], 0.10)
        self.assertEqual(result["macro_f1"], 1.0)
        self.assertEqual(result["threshold"], 0.50)

    def test_event_classifier_averages_top_three_detection_windows_and_freezes(self):
        from src.classify.tiny_swin import EventClassifier
        probabilities = iter([
            np.array([0.60, 0.20, 0.20]), np.array([0.90, 0.05, 0.05]),
            np.array([0.70, 0.20, 0.10]), np.array([0.80, 0.10, 0.10]),
            np.array([0.10, 0.80, 0.10]),
        ])
        classifier = EventClassifier(lambda _: next(probabilities), threshold=0.5)
        for score in (0.1, 0.9, 0.5, 0.7):
            classifier.update(np.zeros(1), score)
        result = classifier.close()
        self.assertEqual(result["label"], "A")
        self.assertAlmostEqual(result["probability"], 0.80)
        self.assertEqual(result["n_windows"], 3)
        classifier.update(np.zeros(1), 1.0)
        self.assertEqual(result["label"], "A")
        self.assertAlmostEqual(result["probability"], 0.80)

    def test_low_confidence_event_is_unknown(self):
        from src.classify.tiny_swin import EventClassifier
        classifier = EventClassifier(lambda _: np.array([0.4, 0.35, 0.25]), threshold=0.5)
        classifier.update(np.zeros(1), 1.0)
        self.assertEqual(classifier.close()["label"], "unknown")


if __name__ == "__main__":
    unittest.main()
