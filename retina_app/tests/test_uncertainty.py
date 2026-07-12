"""Tests for MC Dropout uncertainty quantification."""

from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch.nn as nn
from django.test import SimpleTestCase


class ComputeEntropyTest(SimpleTestCase):
    """Test entropy computation."""

    def test_certain_distribution_low_entropy(self):
        from retina_app.services.uncertainty import compute_entropy

        probs = np.array([0.95, 0.02, 0.02, 0.01])
        entropy = compute_entropy(probs)
        self.assertLess(entropy, 0.3)

    def test_uniform_distribution_high_entropy(self):
        from retina_app.services.uncertainty import compute_entropy

        probs = np.array([0.25, 0.25, 0.25, 0.25])
        entropy = compute_entropy(probs)
        self.assertGreater(entropy, 1.3)

    def test_zero_entropy_certain(self):
        from retina_app.services.uncertainty import compute_entropy

        probs = np.array([1.0, 0.0, 0.0, 0.0])
        entropy = compute_entropy(probs)
        self.assertAlmostEqual(entropy, 0.0, places=5)

    def test_normalization(self):
        from retina_app.services.uncertainty import compute_entropy

        probs = np.array([2.0, 3.0, 1.0, 0.5])
        entropy = compute_entropy(probs)
        self.assertGreaterEqual(entropy, 0.0)


class ComputePredictionEntropyTest(SimpleTestCase):
    """Test normalized entropy computation."""

    def test_certain_prediction(self):
        from retina_app.services.uncertainty import compute_prediction_entropy

        probs = np.array([0.95, 0.02, 0.02, 0.01])
        norm_entropy = compute_prediction_entropy(probs)
        self.assertLess(norm_entropy, 0.3)

    def test_uncertain_prediction(self):
        from retina_app.services.uncertainty import compute_prediction_entropy

        probs = np.array([0.25, 0.25, 0.25, 0.25])
        norm_entropy = compute_prediction_entropy(probs)
        self.assertGreater(norm_entropy, 0.9)


class EnableDisableDropoutTest(SimpleTestCase):
    """Test dropout enable/disable helpers."""

    def _make_model_with_dropout(self):
        return nn.Sequential(
            nn.Linear(10, 10),
            nn.Dropout(p=0.5),
            nn.Linear(10, 4),
        )

    def test_enable_dropout_sets_train_mode(self):
        from retina_app.services.uncertainty import _enable_dropout

        model = self._make_model_with_dropout()
        model.eval()
        dropout_layers = _enable_dropout(model)
        for layer in dropout_layers:
            self.assertTrue(layer.training)
        self.assertFalse(model[0].training)

    def test_disable_dropout_restores_eval(self):
        from retina_app.services.uncertainty import _disable_dropout, _enable_dropout

        model = self._make_model_with_dropout()
        dropout_layers = _enable_dropout(model)
        _disable_dropout(dropout_layers)
        for layer in dropout_layers:
            self.assertFalse(layer.training)


class MCDropoutForwardPassTest(SimpleTestCase):
    """Test MC Dropout forward pass."""

    def _make_model(self):
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 224 * 224, 128),
            nn.Dropout(p=0.5),
            nn.Linear(128, 4),
        )
        return model

    @patch("retina_app.services.model_manager.DEVICE", torch.device("cpu"))
    def test_returns_correct_shape(self):
        from retina_app.services.uncertainty import mc_dropout_forward_pass

        model = self._make_model()
        model.eval()
        tensor = torch.randn(1, 3, 224, 224)
        mean_probs, entropy, is_uncertain = mc_dropout_forward_pass(model, tensor, n_passes=5)
        self.assertEqual(mean_probs.shape, (4,))
        self.assertIsInstance(entropy, float)
        self.assertIsInstance(is_uncertain, (bool, np.bool_))

    @patch("retina_app.services.model_manager.DEVICE", torch.device("cpu"))
    def test_probs_sum_to_one(self):
        from retina_app.services.uncertainty import mc_dropout_forward_pass

        model = self._make_model()
        model.eval()
        tensor = torch.randn(1, 3, 224, 224)
        mean_probs, _, _ = mc_dropout_forward_pass(model, tensor, n_passes=5)
        self.assertAlmostEqual(float(np.sum(mean_probs)), 1.0, places=3)

    @patch("retina_app.services.model_manager.DEVICE", torch.device("cpu"))
    def test_restores_eval_mode(self):
        from retina_app.services.uncertainty import mc_dropout_forward_pass

        model = self._make_model()
        model.eval()
        tensor = torch.randn(1, 3, 224, 224)
        mc_dropout_forward_pass(model, tensor, n_passes=3)
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                self.assertFalse(module.training)


class IsDropoutEnabledTest(SimpleTestCase):
    """Test ENABLE_MC_DROPOUT flag."""

    def test_default_is_disabled(self):
        from retina_app.services.uncertainty import is_dropout_enabled

        self.assertFalse(is_dropout_enabled())


class MCDropoutEnsembleTest(SimpleTestCase):
    """Test ensemble MC Dropout prediction."""

    @patch("retina_app.services.model_manager.DEVICE", torch.device("cpu"))
    def test_empty_models_returns_uncertain(self):
        from retina_app.services.uncertainty import mc_dropout_ensemble

        result = mc_dropout_ensemble({}, "dummy_path.jpg")
        self.assertTrue(result["is_uncertain"])
        self.assertEqual(result["confidence"], 0.0)
        self.assertEqual(result["n_models"], 0)

    @patch("retina_app.services.model_manager.DEVICE", torch.device("cpu"))
    @patch("retina_app.services.uncertainty.mc_dropout_single_model")
    @patch("retina_app.services.uncertainty.TRANSFORM")
    @patch("PIL.Image.open")
    def test_single_model_result(self, mock_img_open, mock_transform, mock_single):
        from retina_app.services.uncertainty import mc_dropout_ensemble

        # Mock image loading
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = lambda s: s
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_ctx.convert.return_value = mock_ctx
        mock_img_open.return_value = mock_ctx
        # Mock TRANSFORM to return a tensor
        mock_transform.return_value = torch.randn(3, 224, 224)
        mock_single.return_value = {
            "label": "Healthy",
            "confidence": 0.8,
            "probabilities": [0.8, 0.1, 0.05, 0.05],
            "entropy": 0.2,
            "is_uncertain": False,
            "n_passes": 3,
        }
        model = nn.Sequential(nn.Flatten(), nn.Linear(10, 4))
        models = {"efficientnet_v2": model}
        result = mc_dropout_ensemble(models, "test.jpg", n_passes=3)
        self.assertEqual(result["n_models"], 1)
        self.assertIn("individual_results", result)
