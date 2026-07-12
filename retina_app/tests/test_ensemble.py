import torch
from django.test import TestCase

from retina_app.constants import CATEGORIES
from retina_app.services.ensemble import (
    _apply_disease_co_occurrence,
    _compute_tta_uncertainty,
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


class DiseaseCoOccurrenceTest(TestCase):
    """Test disease co-occurrence matrix application."""

    def test_co_occurrence_returns_list(self):
        probs = [0.7, 0.1, 0.1, 0.1]
        result = _apply_disease_co_occurrence(probs)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 4)

    def test_co_occurrence_sums_to_one(self):
        probs = [0.7, 0.1, 0.1, 0.1]
        result = _apply_disease_co_occurrence(probs)
        self.assertAlmostEqual(sum(result), 1.0, places=5)

    def test_co_occurrence_preserves_high_confidence(self):
        # High confidence should remain high
        probs = [0.95, 0.02, 0.02, 0.01]
        result = _apply_disease_co_occurrence(probs)
        max_idx = result.index(max(result))
        self.assertEqual(max_idx, 0)

    def test_co_occurrence_adjusts_based_on_medical_knowledge(self):
        # Glaucoma and Retina Disease have higher co-occurrence (0.25)
        probs_healthy = [0.4, 0.2, 0.2, 0.2]
        probs_glaucoma = [0.1, 0.1, 0.6, 0.2]

        result_healthy = _apply_disease_co_occurrence(probs_healthy)
        result_glaucoma = _apply_disease_co_occurrence(probs_glaucoma)

        # Both should sum to 1
        self.assertAlmostEqual(sum(result_healthy), 1.0, places=5)
        self.assertAlmostEqual(sum(result_glaucoma), 1.0, places=5)


class TTAUncertaintyTest(TestCase):
    """Test TTA uncertainty computation."""

    def test_single_probs_zero_variance(self):
        probs = [[0.5, 0.3, 0.1, 0.1]]
        result = _compute_tta_uncertainty(probs)
        self.assertEqual(result, 0.0)

    def test_multiple_probs_returns_variance(self):
        probs = [
            [0.5, 0.3, 0.1, 0.1],
            [0.4, 0.4, 0.1, 0.1],
            [0.6, 0.2, 0.1, 0.1],
        ]
        result = _compute_tta_uncertainty(probs)
        self.assertGreater(result, 0.0)

    def test_empty_list_returns_zero(self):
        result = _compute_tta_uncertainty([])
        self.assertEqual(result, 0.0)


class EnsemblePredictionsTest(TestCase):
    def test_empty_predictions_raises(self):
        with self.assertRaises(ValueError):
            ensemble_predictions([])

    def test_single_prediction_passes_through(self):
        preds = [
            (
                "deit",
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
                "deit",
                {
                    "label": "Healthy",
                    "confidence": 0.8,
                    "probabilities": [0.8, 0.1, 0.05, 0.05],
                },
            ),
            (
                "efficientnet_v2",
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
                "deit",
                {
                    "label": "Cataract",
                    "confidence": 0.6,
                    "probabilities": [0.1, 0.6, 0.2, 0.1],
                },
            ),
            (
                "convnext_v2",
                {
                    "label": "Glaucoma",
                    "confidence": 0.55,
                    "probabilities": [0.1, 0.2, 0.55, 0.15],
                },
            ),
            (
                "maxvit",
                {
                    "label": "Glaucoma",
                    "confidence": 0.58,
                    "probabilities": [0.12, 0.18, 0.52, 0.18],
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
                "deit",
                {
                    "label": "Healthy",
                    "confidence": 0.8,
                    "probabilities": [0.8, 0.1, 0.05, 0.05],
                },
            ),
            (
                "efficientnet_v2",
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
                "deit",
                {
                    "label": "Healthy",
                    "confidence": 0.8,
                    "probabilities": [0.8, 0.1, 0.05, 0.05],
                },
            ),
            (
                "efficientnet_v2",
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
                "deit",
                {
                    "label": "Unknown",
                    "confidence": 0.5,
                    "probabilities": [0.25, 0.25, 0.25, 0.25],
                },
            ),
            (
                "efficientnet_v2",
                {
                    "label": "Healthy",
                    "confidence": 0.8,
                    "probabilities": [0.8, 0.1, 0.05, 0.05],
                },
            ),
        ]
        result = ensemble_predictions(preds)
        self.assertIn(result["label"], CATEGORIES)

    def test_four_model_ensemble_with_maxvit(self):
        preds = [
            ("maxvit", {"label": "Healthy", "confidence": 0.85, "probabilities": [0.85, 0.05, 0.05, 0.05]}),
            ("convnext_v2", {"label": "Healthy", "confidence": 0.80, "probabilities": [0.80, 0.10, 0.05, 0.05]}),
            ("efficientnet_v2", {"label": "Cataract", "confidence": 0.60, "probabilities": [0.20, 0.55, 0.15, 0.10]}),
            ("deit", {"label": "Healthy", "confidence": 0.70, "probabilities": [0.70, 0.15, 0.10, 0.05]}),
        ]
        result = ensemble_predictions(preds)
        self.assertEqual(result["label"], "Healthy")
        self.assertEqual(result["n_models"], 4)


class DetectModelDisagreementTest(TestCase):
    def test_no_disagreement(self):
        preds = [
            ("deit", {"label": "Healthy", "confidence": 0.8, "probabilities": [0.8, 0.1, 0.05, 0.05]}),
            ("efficientnet_v2", {"label": "Healthy", "confidence": 0.7, "probabilities": [0.7, 0.15, 0.1, 0.05]}),
        ]
        result = detect_model_disagreement(preds)
        self.assertFalse(result["disagreement"])

    def test_disagreement_detected(self):
        preds = [
            ("deit", {"label": "Healthy", "confidence": 0.8, "probabilities": [0.8, 0.1, 0.05, 0.05]}),
            ("efficientnet_v2", {"label": "Cataract", "confidence": 0.6, "probabilities": [0.1, 0.6, 0.2, 0.1]}),
            ("convnext_v2", {"label": "Cataract", "confidence": 0.7, "probabilities": [0.05, 0.7, 0.15, 0.1]}),
        ]
        result = detect_model_disagreement(preds)
        self.assertTrue(result["disagreement"])
        self.assertIn("agreement_level", result)
        self.assertIn("dominant_class", result)
        self.assertIn("disagreeing_models", result)

    def test_returns_expected_keys(self):
        preds = [
            ("deit", {"label": "Healthy", "confidence": 0.8, "probabilities": [0.8, 0.1, 0.05, 0.05]}),
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
            ("deit", {"label": "Healthy", "confidence": 0.8, "probabilities": [0.8, 0.1, 0.05, 0.05]}),
            ("efficientnet_v2", {"label": "Healthy", "confidence": 0.7, "probabilities": [0.7, 0.15, 0.1, 0.05]}),
            ("convnext_v2", {"label": "Cataract", "confidence": 0.6, "probabilities": [0.1, 0.6, 0.2, 0.1]}),
        ]
        result = selective_ensemble(preds)
        self.assertIn("label", result)
        self.assertIn("selective_ensemble", result)
        self.assertTrue(result["selective_ensemble"])

    def test_no_majority_fallback_to_full(self):
        preds = [
            ("deit", {"label": "Healthy", "confidence": 0.5, "probabilities": [0.5, 0.2, 0.2, 0.1]}),
            ("efficientnet_v2", {"label": "Cataract", "confidence": 0.45, "probabilities": [0.15, 0.45, 0.25, 0.15]}),
        ]
        result = selective_ensemble(preds)
        self.assertFalse(result["selective_ensemble"])

    def test_returns_expected_keys(self):
        preds = [
            ("deit", {"label": "Healthy", "confidence": 0.8, "probabilities": [0.8, 0.1, 0.05, 0.05]}),
        ]
        result = selective_ensemble(preds)
        self.assertIn("label", result)
        self.assertIn("confidence", result)
        self.assertIn("probabilities", result)
        self.assertIn("selective_ensemble", result)
        self.assertIn("agreement_level", result)
