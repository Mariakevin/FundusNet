"""Tests for the learned fundus classifier."""

import os
import tempfile

import numpy as np
from django.test import SimpleTestCase

from retina_app.services.fundus_classifier import FundusClassifier


class TestFundusClassifierInit(SimpleTestCase):
    """Test FundusClassifier initialization."""

    def test_default_init(self):
        model = FundusClassifier(num_classes=2, freeze_backbone=True)
        self.assertIsNotNone(model)
        self.assertIsNotNone(model.backbone)
        self.assertIsNotNone(model.transform)

    def test_no_freeze_init(self):
        model = FundusClassifier(num_classes=2, freeze_backbone=False)
        # All parameters should be trainable
        for param in model.backbone.parameters():
            self.assertTrue(param.requires_grad)


class TestFundusClassifierForward(SimpleTestCase):
    """Test FundusClassifier forward pass."""

    def test_forward_shape(self):
        import torch

        model = FundusClassifier(num_classes=2, freeze_backbone=True)
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        self.assertEqual(output.shape, (1, 2))

    def test_batch_forward(self):
        import torch

        model = FundusClassifier(num_classes=2, freeze_backbone=True)
        x = torch.randn(4, 3, 224, 224)
        output = model(x)
        self.assertEqual(output.shape, (4, 2))


class TestFundusClassifierPredict(SimpleTestCase):
    """Test FundusClassifier predict method."""

    def test_predict_returns_expected_keys(self):
        import torch

        model = FundusClassifier(num_classes=2, freeze_backbone=True)
        model.eval()
        tensor = torch.randn(1, 3, 224, 224)
        result = model.predict(tensor)
        self.assertIn("is_fundus", result)
        self.assertIn("confidence", result)
        self.assertIn("probability", result)

    def test_predict_single_image(self):
        import torch

        model = FundusClassifier(num_classes=2, freeze_backbone=True)
        model.eval()
        tensor = torch.randn(3, 224, 224)  # no batch dim
        result = model.predict(tensor)
        self.assertIn("is_fundus", result)
        self.assertIsInstance(result["is_fundus"], bool)

    def test_predict_from_pil(self):
        from PIL import Image

        model = FundusClassifier(num_classes=2, freeze_backbone=True)
        model.eval()
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        result = model.predict_from_pil(img)
        self.assertIn("is_fundus", result)
        self.assertIn("probability", result)
        self.assertGreaterEqual(result["probability"], 0.0)
        self.assertLessEqual(result["probability"], 1.0)


class TestFundusClassifierSaveLoad(SimpleTestCase):
    """Test FundusClassifier save and load."""

    def test_save_and_load(self):
        import torch

        with tempfile.TemporaryDirectory() as tmpdir:
            model = FundusClassifier(num_classes=2, freeze_backbone=True)
            save_path = os.path.join(tmpdir, "test_model.pth")
            model.save(save_path)
            self.assertTrue(os.path.exists(save_path))

            loaded = FundusClassifier.load(save_path)
            self.assertIsNotNone(loaded)

            # Verify predictions match
            model.eval()
            loaded.eval()
            tensor = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                out1 = model(tensor)
                out2 = loaded(tensor)
            np.testing.assert_array_almost_equal(out1.numpy(), out2.numpy(), decimal=5)

    def test_load_with_eval_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = FundusClassifier(num_classes=2, freeze_backbone=True)
            save_path = os.path.join(tmpdir, "test_model.pth")
            model.save(save_path)

            loaded = FundusClassifier.load(save_path, freeze_backbone=False)
            loaded.eval()
            # Should be in eval mode
            self.assertFalse(loaded.backbone.training)


class TestFundusClassifierConstants(SimpleTestCase):
    """Test that constants are properly configured."""

    def test_constants_exist(self):
        from retina_app.constants import (
            FUNDUS_LEARNED_MODEL_PATH,
            FUNDUS_LEARNED_THRESHOLD,
            FUNDUS_LEARNED_VALIDATOR_ENABLED,
        )

        self.assertIsInstance(FUNDUS_LEARNED_VALIDATOR_ENABLED, bool)
        self.assertIsInstance(FUNDUS_LEARNED_MODEL_PATH, str)
        self.assertIsInstance(FUNDUS_LEARNED_THRESHOLD, float)
