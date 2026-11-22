# -*- coding: utf-8 -*-
import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "项目虚拟环境提供 PyTorch 后运行")
class LearnedFocusTests(unittest.TestCase):
    def test_covariance_three_layers_preserve_real_imag_abs_and_sign(self):
        from src.doa.learned_focus import covariance_bands
        covariance = np.array([[[1.0 + 2.0j, 3.0 - 4.0j], [5.0 + 0.0j, 6.0 - 0.0j]]])
        result = covariance_bands(covariance[None], np.ones((1, 1), dtype=bool))
        self.assertEqual(tuple(result.shape), (1, 1, 3, 2, 2))
        np.testing.assert_allclose(result[0, 0, 0].numpy(), covariance[0].real)
        np.testing.assert_allclose(result[0, 0, 1].numpy(), np.abs(covariance[0].imag))
        np.testing.assert_allclose(result[0, 0, 2].numpy(), np.sign(covariance[0].imag))

    def test_masked_softmax_excludes_invalid_bands(self):
        from src.doa.learned_focus import BandWeightNet
        model = BandWeightNet(n_bands=8)
        features = torch.randn(2, 8, 3, 16, 16)
        valid = torch.tensor([[1, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.bool)
        weights = model(features, valid)
        np.testing.assert_allclose(weights[0].detach().numpy().sum(), 1.0, atol=1e-6)
        self.assertEqual(float(weights[0, 1].detach()), 0.0)
        self.assertEqual(float(weights[1].abs().sum().detach()), 0.0)

    def test_soft_angle_label_interpolates_adjacent_grid_points(self):
        from src.doa.learned_focus import soft_angle_targets
        targets = soft_angle_targets(torch.tensor([-0.5, 0.5]), torch.tensor([-1.0, 0.0, 1.0]))
        np.testing.assert_allclose(targets.numpy(), [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]])

    def test_focus_loss_backpropagates_to_shared_convolution(self):
        from src.doa.learned_focus import BandWeightNet, focus_cross_entropy
        model = BandWeightNet(n_bands=8)
        features = torch.randn(2, 8, 3, 16, 16)
        valid = torch.ones(2, 8, dtype=torch.bool)
        weights = model(features, valid)
        band_spectra = torch.rand(2, 8, 3)
        fused = torch.einsum("bi,bip->bp", weights, band_spectra)
        angle_spectrum = torch.stack([fused.max(dim=1).values, fused.mean(dim=1), fused.min(dim=1).values], dim=1)
        target = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        loss = focus_cross_entropy(angle_spectrum, target)
        loss.backward()
        self.assertGreater(float(model.conv[0].weight.grad.abs().sum()), 0.0)

    def test_one_epoch_focus_training_runs(self):
        from src.doa.learned_focus import BandWeightNet
        from src.doa.train_focus import train_focus
        torch.set_num_threads(1)
        model = BandWeightNet(n_bands=8)
        covariances = torch.randn(8, 8, 3, 16, 16)
        spectra = torch.rand(8, 8, 3)
        valid = torch.ones(8, 8, dtype=torch.bool)
        angles = torch.zeros(8)
        grid_angles = torch.tensor([-1.0, 0.0, 1.0])
        angle_indices = (torch.tensor([0]), torch.tensor([1]), torch.tensor([2]))
        result = train_focus(model, covariances, spectra, valid, angles,
                             covariances, spectra, valid, angles,
                             grid_angles, angle_indices, epochs=1)
        self.assertEqual(len(result["history"]), 1)
        self.assertTrue(np.isfinite(result["history"][0]["val_mae"]))


if __name__ == "__main__":
    unittest.main()
