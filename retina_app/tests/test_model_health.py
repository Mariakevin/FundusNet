from django.test import TestCase
from retina_app.services.model_manager import (
    ModelHealthTracker,
    get_health_tracker,
)
from retina_app.constants import MODEL_HEALTH_WINDOW, MODEL_HEALTH_MIN_ACCURACY


class ModelHealthTrackerTest(TestCase):
    def setUp(self):
        self.tracker = ModelHealthTracker()

    def test_record_prediction(self):
        self.tracker.record_prediction("squeezenet", "Healthy", 0.9)
        health = self.tracker.get_model_health("squeezenet")
        self.assertEqual(health["predictions"], 1)
        self.assertAlmostEqual(health["avg_confidence"], 0.9, places=4)

    def test_get_all_health(self):
        self.tracker.record_prediction("squeezenet", "Healthy", 0.9)
        self.tracker.record_prediction("efficientnet", "Cataract", 0.8)
        all_health = self.tracker.get_all_health()
        self.assertIn("squeezenet", all_health)
        self.assertIn("efficientnet", all_health)

    def test_reset(self):
        self.tracker.record_prediction("squeezenet", "Healthy", 0.9)
        self.tracker.reset()
        health = self.tracker.get_model_health("squeezenet")
        self.assertEqual(health["predictions"], 0)

    def test_unknown_model_returns_status(self):
        health = self.tracker.get_model_health("nonexistent")
        self.assertEqual(health["status"], "unknown")
        self.assertEqual(health["predictions"], 0)

    def test_record_ensemble_prediction(self):
        preds = [
            ("squeezenet", {"label": "Healthy", "confidence": 0.9}),
            ("efficientnet", {"label": "Cataract", "confidence": 0.8}),
        ]
        self.tracker.record_ensemble_prediction(preds)
        health_s = self.tracker.get_model_health("squeezenet")
        health_e = self.tracker.get_model_health("efficientnet")
        self.assertEqual(health_s["predictions"], 1)
        self.assertEqual(health_e["predictions"], 1)

    def test_warming_up_status(self):
        for _ in range(5):
            self.tracker.record_prediction("squeezenet", "Healthy", 0.9)
        health = self.tracker.get_model_health("squeezenet")
        self.assertEqual(health["status"], "warming_up")

    def test_healthy_status(self):
        for _ in range(15):
            self.tracker.record_prediction("squeezenet", "Healthy", 0.9)
        health = self.tracker.get_model_health("squeezenet")
        self.assertEqual(health["status"], "healthy")

    def test_degraded_status(self):
        for _ in range(15):
            self.tracker.record_prediction("squeezenet", "Healthy", 0.1)
        health = self.tracker.get_model_health("squeezenet")
        self.assertEqual(health["status"], "degraded")


class GetHealthTrackerSingletonTest(TestCase):
    def test_returns_same_instance(self):
        t1 = get_health_tracker()
        t2 = get_health_tracker()
        self.assertIs(t1, t2)
