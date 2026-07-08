"""Tests for Grad-CAM explainability."""

from unittest.mock import MagicMock, patch

import numpy as np
import torch
from django.test import SimpleTestCase


class GetTargetLayerTest(SimpleTestCase):
    """Test target layer selection for different model architectures."""

    def test_resnet_target_layer(self):
        import torchvision.models as models

        from retina_app.services.gradcam import _get_target_layer

        model = models.resnet50(weights=None)
        layer = _get_target_layer(model, "resnet")
        self.assertEqual(layer, model.layer4)

    def test_efficientnet_target_layer(self):
        import torchvision.models as models

        from retina_app.services.gradcam import _get_target_layer

        model = models.efficientnet_b0(weights=None)
        layer = _get_target_layer(model, "efficientnet")
        self.assertEqual(layer, model.features[-1])

    def test_mobilenet_target_layer(self):
        import torchvision.models as models

        from retina_app.services.gradcam import _get_target_layer

        model = models.mobilenet_v3_small(weights=None)
        layer = _get_target_layer(model, "mobilenet")
        self.assertEqual(layer, model.features[-1])

    def test_squeezenet_target_layer(self):
        import torchvision.models as models

        from retina_app.services.gradcam import _get_target_layer

        model = models.squeezenet1_0(weights=None)
        layer = _get_target_layer(model, "squeezenet")
        self.assertEqual(layer, model.features[12])


class GradCAMInitTest(SimpleTestCase):
    """Test GradCAM initialization."""

    def test_creates_hooks(self):
        import torchvision.models as models

        from retina_app.services.gradcam import GradCAM

        model = models.resnet50(weights=None)
        gradcam = GradCAM(model, "resnet")
        self.assertIsNotNone(gradcam._forward_handle)
        self.assertIsNotNone(gradcam._backward_handle)
        gradcam.cleanup()


class DeprocessImageTest(SimpleTestCase):
    """Test heatmap deprocessing."""

    def test_output_shape(self):
        from retina_app.services.gradcam import _deprocess_image

        cam = np.random.rand(7, 7).astype(np.float32)
        original_size = (224, 224)
        result = _deprocess_image(cam, original_size)
        self.assertEqual(result.shape, (224, 224, 3))

    def test_output_dtype(self):
        from retina_app.services.gradcam import _deprocess_image

        cam = np.random.rand(14, 14).astype(np.float32)
        result = _deprocess_image(cam, (112, 112))
        self.assertEqual(result.dtype, np.uint8)

    def test_output_range(self):
        from retina_app.services.gradcam import _deprocess_image

        cam = np.random.rand(7, 7).astype(np.float32)
        result = _deprocess_image(cam, (100, 100))
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 255))


class BlendHeatmapTest(SimpleTestCase):
    """Test heatmap blending."""

    def test_same_size_blend(self):
        from retina_app.services.gradcam import _blend_heatmap

        original = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        heatmap = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = _blend_heatmap(original, heatmap, alpha=0.5)
        self.assertEqual(result.shape, (100, 100, 3))

    def test_different_size_resizes(self):
        from retina_app.services.gradcam import _blend_heatmap

        original = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        heatmap = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        result = _blend_heatmap(original, heatmap, alpha=0.3)
        self.assertEqual(result.shape, (100, 100, 3))


class GradCAMGenerateTest(SimpleTestCase):
    """Test GradCAM generation with a real model."""

    def test_generate_returns_expected_keys(self):
        import torchvision.models as models

        from retina_app.services.gradcam import generate_gradcam

        model = models.resnet50(weights=None)
        model.eval()

        # Mock the full pipeline instead of running actual inference
        gradcam = MagicMock()
        gradcam.generate.return_value = (
            np.random.rand(7, 7).astype(np.float32),  # cam
            0,  # pred_idx
            0.95,  # confidence
        )

        # Mock model forward to return 4-class output (not 1000)
        mock_output = torch.zeros(1, 4)
        mock_output[0, 0] = 1.0

        with patch("retina_app.services.gradcam.GradCAM", return_value=gradcam):
            with patch("retina_app.services.gradcam.TRANSFORM") as mock_transform:
                mock_transform.return_value = torch.randn(3, 224, 224)
                with patch.object(model, "forward", return_value=mock_output):
                    with patch("retina_app.services.gradcam._deprocess_image") as mock_deproc:
                        mock_deproc.return_value = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                        with patch("retina_app.services.gradcam._blend_heatmap") as mock_blend:
                            mock_blend.return_value = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                            with patch("PIL.Image.open") as mock_img:
                                mock_ctx = MagicMock()
                                mock_ctx.__enter__ = lambda s: s
                                mock_ctx.__exit__ = MagicMock(return_value=False)
                                mock_ctx.convert.return_value = mock_ctx
                                mock_img.return_value = mock_ctx

                                result = generate_gradcam(
                                    model,
                                    "test.jpg",
                                    "resnet",
                                    output_path="/tmp/test_gradcam.png",
                                )
                                self.assertIn("predicted_class", result)
                                self.assertIn("confidence", result)
                                self.assertIn("output_path", result)


class GetGradcamOutputPathTest(SimpleTestCase):
    """Test Grad-CAM output path generation."""

    def test_output_path_format(self):
        import tempfile

        from retina_app.services.gradcam import get_gradcam_output_path

        with tempfile.TemporaryDirectory() as tmpdir:
            path = get_gradcam_output_path(tmpdir, "test_image.jpg")
            self.assertTrue(path.endswith("_gradcam.png"))
            self.assertIn("gradcam", path)

    def test_creates_directory(self):
        import os
        import tempfile

        from retina_app.services.gradcam import get_gradcam_output_path

        with tempfile.TemporaryDirectory() as tmpdir:
            path = get_gradcam_output_path(tmpdir, "image.png")
            self.assertTrue(os.path.isdir(os.path.dirname(path)))
