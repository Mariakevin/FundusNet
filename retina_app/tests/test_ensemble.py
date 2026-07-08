import torch
from django.test import TestCase

from retina_app.constants import CATEGORIES
from retina_app.services.ensemble import (
    apply_temperature_scaling,
    detect_model_disagreement,
    ensemble_predictions,
    selective_ensemble,
)


class ApplyTemperatureScalingTest(TestCase):
    def test_temperature_1_0_returns_softmax(self):
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        result = apply_temperature_scaling(logits, temperature=1.0)
        expected = torch.softmax(logits, dim=1)
        torch.testing.assert_close(result, expected)

    def test_temperature_higher_spreads(self):
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        cool = apply_temperature_scaling(logits, temperature=2.0)
        hot = apply_temperature_scaling(logits, temperature=0.5)
        # Higher temperature produces more uniform (lower std), lower temp produces more peaked (higher std)
        self.assertLess(cool.std().item(), hot.std().item())


class EnsemblePredictionsTest(TestCase):
    def test_empty_predictions_raises(self):
        with self.assertRaises(ValueError):
            ensemble_predictions([])

    def test_single_prediction_passes_through(self):
        preds = [
            (
                "squeezenet",
                {
                    "label": "Healthy",
                    "confidence": 0.9,
                    "probabilities": [0.9, 0.05, 0.03, 0.02],
                },
            )
        ]
        result = ensemble_predictions(preds)
        self.assertEqual(result["label"], "Healthy")
        self.assertEqual(result["n_models"], 1)

    def test_two_models_majority_vote(self):
        preds = [
            (
                "squeezenet",
                {
                    "label": "Healthy",
                    "confidence": 0.8,
                    "probabilities": [0.8, 0.1, 0.05, 0.05],
                },
            ),
            (
                "efficientnet",
                {
                    "label": "Healthy",
                    "confidence": 0.7,
                    "probabilities": [0.7, 0.15, 0.1, 0.05],
                },
            ),
        ]
        result = ensemble_predictions(preds)
        self.assertEqual(result["label"], "Healthy")
        self.assertGreater(result["confidence"], 0.5)
        self.assertEqual(result["n_models"], 2)

    def test_ensemble_returns_all_keys(self):
        preds = [
            (
                "squeezenet",
                {
                    "label": "Cataract",
                    "confidence": 0.6,
                    "probabilities": [0.1, 0.6, 0.2, 0.1],
                },
            ),
            (
                "resnet",
                {
                    "label": "Glaucoma",
                    "confidence": 0.55,
                    "probabilities": [0.1, 0.2, 0.55, 0.15],
                },
            ),
        ]
        result = ensemble_predictions(preds)
        self.assertIn("label", result)
        self.assertIn("confidence", result)
        self.assertIn("avg_model_confidence", result)
        self.assertIn("n_models", result)
        self.assertIn("probabilities", result)
        self.assertIn("uncertainty", result)

    def test_ensemble_probabilities_sum_to_one(self):
        preds = [
            (
                "squeezenet",
                {
                    "label": "Healthy",
                    "confidence": 0.8,
                    "probabilities": [0.8, 0.1, 0.05, 0.05],
                },
            ),
            (
                "efficientnet",
                {
                    "label": "Cataract",
                    "confidence": 0.6,
                    "probabilities": [0.2, 0.5, 0.2, 0.1],
                },
            ),
        ]
        result = ensemble_predictions(preds)
        total = sum(result["probabilities"])
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_ensemble_uncertainty_range(self):
        preds = [
            (
                "squeezenet",
                {
                    "label": "Healthy",
                    "confidence": 0.8,
                    "probabilities": [0.8, 0.1, 0.05, 0.05],
                },
            ),
            (
                "efficientnet",
                {
                    "label": "Healthy",
                    "confidence": 0.7,
                    "probabilities": [0.7, 0.15, 0.1, 0.05],
                },
            ),
        ]
        result = ensemble_predictions(preds)
        self.assertGreaterEqual(result["uncertainty"], 0.0)
        self.assertLessEqual(result["uncertainty"], 1.0)

    def test_ensemble_with_unknown_category_graceful(self):
        preds = [
            (
                "squeezenet",
                {
                    "label": "Unknown",
                    "confidence": 0.5,
                    "probabilities": [0.25, 0.25, 0.25, 0.25],
                },
            ),
            (
                "efficientnet",
                {
                    "label": "Healthy",
                    "confidence": 0.8,
                    "probabilities": [0.8, 0.1, 0.05, 0.05],
                },
            ),
        ]
        result = ensemble_predictions(preds)
        self.assertIn(result["label"], CATEGORIES)


class DetectModelDisagreementTest(TestCase):
    def test_no_disagreement(self):
        preds = [
            ("squeezenet", {"label": "Healthy", "confidence": 0.8, "probabilities": [0.8, 0.1, 0.05, 0.05]}),
            ("efficientnet", {"label": "Healthy", "confidence": 0.7, "probabilities": [0.7, 0.15, 0.1, 0.05]}),
        ]
        result = detect_model_disagreement(preds)
        self.assertFalse(result["disagreement"])

    def test_disagreement_detected(self):
        preds = [
            ("squeezenet", {"label": "Healthy", "confidence": 0.8, "probabilities": [0.8, 0.1, 0.05, 0.05]}),
            ("efficientnet", {"label": "Cataract", "confidence": 0.6, "probabilities": [0.1, 0.6, 0.2, 0.1]}),
            ("resnet", {"label": "Cataract", "confidence": 0.7, "probabilities": [0.05, 0.7, 0.15, 0.1]}),
        ]
        result = detect_model_disagreement(preds)
        self.assertTrue(result["disagreement"])
        self.assertIn("agreement_level", result)
        self.assertIn("dominant_class", result)
        self.assertIn("disagreeing_models", result)

    def test_returns_expected_keys(self):
        preds = [
            ("squeezenet", {"label": "Healthy", "confidence": 0.8, "probabilities": [0.8, 0.1, 0.05, 0.05]}),
        ]
        result = detect_model_disagreement(preds)
        self.assertIn("disagreement", result)
        self.assertIn("agreement_level", result)
        self.assertIn("dominant_class", result)
        self.assertIn("class_votes", result)
        self.assertIn("disagreeing_models", result)
        self.assertIn("model_predictions", result)


class SelectiveEnsembleTest(TestCase):
    def test_agreement_majority_uses_subset(self):
        preds = [
            ("squeezenet", {"label": "Healthy", "confidence": 0.8, "probabilities": [0.8, 0.1, 0.05, 0.05]}),
            ("efficientnet", {"label": "Healthy", "confidence": 0.7, "probabilities": [0.7, 0.15, 0.1, 0.05]}),
            ("resnet", {"label": "Cataract", "confidence": 0.6, "probabilities": [0.1, 0.6, 0.2, 0.1]}),
        ]
        result = selective_ensemble(preds)
        self.assertIn("label", result)
        self.assertIn("selective_ensemble", result)
        self.assertTrue(result["selective_ensemble"])

    def test_no_majority_fallback_to_full(self):
        preds = [
            ("squeezenet", {"label": "Healthy", "confidence": 0.5, "probabilities": [0.5, 0.2, 0.2, 0.1]}),
            ("efficientnet", {"label": "Cataract", "confidence": 0.45, "probabilities": [0.15, 0.45, 0.25, 0.15]}),
        ]
        result = selective_ensemble(preds)
        self.assertFalse(result["selective_ensemble"])

    def test_returns_expected_keys(self):
        preds = [
            ("squeezenet", {"label": "Healthy", "confidence": 0.8, "probabilities": [0.8, 0.1, 0.05, 0.05]}),
        ]
        result = selective_ensemble(preds)
        self.assertIn("label", result)
        self.assertIn("confidence", result)
        self.assertIn("probabilities", result)
        self.assertIn("selective_ensemble", result)
        self.assertIn("agreement_level", result)
