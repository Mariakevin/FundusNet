"""Tests for contrastive pretraining and distillation modules."""

import torch
from django.test import SimpleTestCase


class NTXentLossTest(SimpleTestCase):
    """Test NT-Xent contrastive loss."""

    def test_same_features_low_loss(self):
        from retina_app.services.contrastive import NTXentLoss

        loss_fn = NTXentLoss(temperature=0.07)
        features_i = torch.randn(8, 256)
        features_j = features_i + 0.01 * torch.randn(8, 256)
        loss = loss_fn(features_i, features_j)
        self.assertGreater(loss.item(), 0)

    def test_different_features_higher_loss(self):
        from retina_app.services.contrastive import NTXentLoss

        loss_fn = NTXentLoss(temperature=0.07)
        features_i = torch.randn(8, 256)
        features_j = torch.randn(8, 256)
        loss = loss_fn(features_i, features_j)
        self.assertGreater(loss.item(), 0)

    def test_loss_scalar(self):
        from retina_app.services.contrastive import NTXentLoss

        loss_fn = NTXentLoss()
        features_i = torch.randn(4, 128)
        features_j = torch.randn(4, 128)
        loss = loss_fn(features_i, features_j)
        self.assertEqual(loss.dim(), 0)


class BarlowTwinsLossTest(SimpleTestCase):
    """Test Barlow Twins loss."""

    def test_loss_scalar(self):
        from retina_app.services.contrastive import BarlowTwinsLoss

        loss_fn = BarlowTwinsLoss()
        features_i = torch.randn(8, 256)
        features_j = torch.randn(8, 256)
        loss = loss_fn(features_i, features_j)
        self.assertEqual(loss.dim(), 0)
        self.assertGreater(loss.item(), 0)


class DistillationLossTest(SimpleTestCase):
    """Test knowledge distillation loss."""

    def test_combined_loss(self):
        from retina_app.services.distillation import DistillationLoss

        loss_fn = DistillationLoss(temperature=4.0, alpha=0.7, beta=0.3)
        student_logits = torch.randn(8, 4)
        teacher_logits = torch.randn(8, 4)
        targets = torch.randint(0, 4, (8,))

        loss = loss_fn(student_logits, teacher_logits, targets)
        self.assertEqual(loss.dim(), 0)
        self.assertGreater(loss.item(), 0)

    def test_alpha_balancing(self):
        from retina_app.services.distillation import DistillationLoss

        loss_fn_alpha_high = DistillationLoss(alpha=0.9, beta=0.1)
        loss_fn_alpha_low = DistillationLoss(alpha=0.1, beta=0.9)

        student = torch.randn(8, 4)
        teacher = torch.randn(8, 4)
        targets = torch.randint(0, 4, (8,))

        loss_high = loss_fn_alpha_high(student, teacher, targets)
        loss_low = loss_fn_alpha_low(student, teacher, targets)
        self.assertNotAlmostEqual(loss_high.item(), loss_low.item(), places=2)


class StudentModelTest(SimpleTestCase):
    """Test lightweight student model creation."""

    def test_creates_efficientnet_b0(self):
        try:
            import timm  # noqa: F401
        except ImportError:
            self.skipTest("timm not installed")

        from retina_app.services.distillation import StudentModel

        model = StudentModel(model_type="efficientnet_b0", num_classes=4)
        self.assertIsNotNone(model)

        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        self.assertEqual(output.shape, (1, 4))
