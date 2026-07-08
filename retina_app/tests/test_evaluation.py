"""Tests for evaluation framework."""

import numpy as np
from django.test import SimpleTestCase

from evaluation.metrics import (
    compute_ece,
    compute_mce,
    compute_brier,
    per_class_metrics,
    confusion_matrix,
    overall_metrics,
    compute_auroc_per_class,
    compute_reliability_data,
    compute_error_detection_auroc,
)


class TestComputeECE(SimpleTestCase):
    """Test Expected Calibration Error computation."""

    def test_perfect_calibration(self):
        probs = np.array([[1.0, 0.0], [0.0, 1.0]])
        labels = np.array([0, 1])
        ece = compute_ece(probs, labels, n_bins=5)
        self.assertAlmostEqual(ece, 0.0, places=6)

    def test_worst_calibration(self):
        probs = np.array([[0.0, 1.0], [1.0, 0.0]])
        labels = np.array([0, 1])
        ece = compute_ece(probs, labels, n_bins=5)
        self.assertGreater(ece, 0.5)

    def test_empty_bins(self):
        probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]])
        labels = np.array([0, 0, 1])
        ece = compute_ece(probs, labels, n_bins=20)
        self.assertIsInstance(ece, float)
        self.assertGreaterEqual(ece, 0.0)

    def test_single_class(self):
        probs = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        labels = np.array([0, 0, 0])
        ece = compute_ece(probs, labels, n_bins=10)
        self.assertAlmostEqual(ece, 0.0, places=6)


class TestComputeMCE(SimpleTestCase):
    """Test Maximum Calibration Error computation."""

    def test_perfect_calibration(self):
        probs = np.array([[1.0, 0.0], [0.0, 1.0]])
        labels = np.array([0, 1])
        mce = compute_mce(probs, labels, n_bins=5)
        self.assertAlmostEqual(mce, 0.0, places=6)

    def test_mce_ge_ece(self):
        np.random.seed(42)
        probs = np.random.dirichlet([1, 1], size=100)
        labels = np.random.randint(0, 2, size=100)
        mce = compute_mce(probs, labels, n_bins=10)
        ece = compute_ece(probs, labels, n_bins=10)
        self.assertGreaterEqual(mce, ece - 1e-10)


class TestComputeBrier(SimpleTestCase):
    """Test Brier Score computation."""

    def test_perfect_predictions(self):
        probs = np.array([[1.0, 0.0], [0.0, 1.0]])
        labels = np.array([0, 1])
        brier = compute_brier(probs, labels)
        self.assertAlmostEqual(brier, 0.0, places=6)

    def test_worst_predictions(self):
        probs = np.array([[0.0, 1.0], [1.0, 0.0]])
        labels = np.array([0, 1])
        brier = compute_brier(probs, labels)
        self.assertAlmostEqual(brier, 2.0, places=6)

    def test_uniform_predictions(self):
        probs = np.array([[0.5, 0.5], [0.5, 0.5]])
        labels = np.array([0, 1])
        brier = compute_brier(probs, labels)
        self.assertAlmostEqual(brier, 0.5, places=6)


class TestPerClassMetrics(SimpleTestCase):
    """Test per-class metric computation."""

    def test_perfect_classification(self):
        predictions = np.array([0, 1, 2, 0, 1, 2])
        labels = np.array([0, 1, 2, 0, 1, 2])
        categories = ["A", "B", "C"]
        result = per_class_metrics(predictions, labels, categories)
        for cat in categories:
            self.assertAlmostEqual(result[cat]["precision"], 1.0)
            self.assertAlmostEqual(result[cat]["recall"], 1.0)
            self.assertAlmostEqual(result[cat]["f1"], 1.0)

    def test_misclassification(self):
        predictions = np.array([0, 0, 1, 1])
        labels = np.array([0, 1, 0, 1])
        categories = ["A", "B"]
        result = per_class_metrics(predictions, labels, categories)
        self.assertAlmostEqual(result["A"]["precision"], 0.5)
        self.assertAlmostEqual(result["A"]["recall"], 0.5)
        self.assertAlmostEqual(result["B"]["precision"], 0.5)
        self.assertAlmostEqual(result["B"]["recall"], 0.5)

    def test_support(self):
        predictions = np.array([0, 0, 0, 1, 1])
        labels = np.array([0, 1, 0, 1, 1])
        categories = ["A", "B"]
        result = per_class_metrics(predictions, labels, categories)
        self.assertEqual(result["A"]["support"], 2)
        self.assertEqual(result["B"]["support"], 3)


class TestConfusionMatrix(SimpleTestCase):
    """Test confusion matrix computation."""

    def test_perfect(self):
        predictions = np.array([0, 1, 2])
        labels = np.array([0, 1, 2])
        cm = confusion_matrix(predictions, labels, 3)
        np.testing.assert_array_equal(cm, np.diag([1, 1, 1]))

    def test_two_class(self):
        predictions = np.array([0, 0, 1, 1])
        labels = np.array([0, 1, 0, 1])
        cm = confusion_matrix(predictions, labels, 2)
        self.assertEqual(cm[0, 0], 1)  # TN
        self.assertEqual(cm[0, 1], 1)  # FN
        self.assertEqual(cm[1, 0], 1)  # FP
        self.assertEqual(cm[1, 1], 1)  # TP


class TestOverallMetrics(SimpleTestCase):
    """Test overall metric computation."""

    def test_perfect(self):
        predictions = np.array([0, 1, 2])
        labels = np.array([0, 1, 2])
        result = overall_metrics(predictions, labels, ["A", "B", "C"])
        self.assertAlmostEqual(result["accuracy"], 1.0)
        self.assertAlmostEqual(result["macro_f1"], 1.0)

    def test_imbalanced(self):
        predictions = np.array([0, 0, 0, 0, 1])
        labels = np.array([0, 0, 0, 1, 1])
        result = overall_metrics(predictions, labels, ["A", "B"])
        self.assertAlmostEqual(result["accuracy"], 0.8)


class TestAUROCPerClass(SimpleTestCase):
    """Test AUROC computation."""

    def test_perfect_separation(self):
        probs = np.array([[0.9, 0.1], [0.1, 0.9]])
        labels = np.array([0, 1])
        aurocs = compute_auroc_per_class(probs, labels, 2)
        self.assertAlmostEqual(aurocs[0], 1.0)
        self.assertAlmostEqual(aurocs[1], 1.0)

    def test_random(self):
        np.random.seed(42)
        probs = np.random.dirichlet([1, 1], size=100)
        labels = np.random.randint(0, 2, size=100)
        aurocs = compute_auroc_per_class(probs, labels, 2)
        for auroc in aurocs:
            self.assertGreater(auroc, 0.3)
            self.assertLess(auroc, 0.7)


class TestReliabilityData(SimpleTestCase):
    """Test reliability diagram data computation."""

    def test_returns_expected_keys(self):
        probs = np.array([[0.8, 0.2], [0.3, 0.7]])
        labels = np.array([0, 1])
        data = compute_reliability_data(probs, labels, n_bins=5)
        self.assertIn("bin_centers", data)
        self.assertIn("bin_accuracies", data)
        self.assertIn("bin_confidences", data)
        self.assertIn("bin_counts", data)
        self.assertEqual(len(data["bin_centers"]), 5)


class TestErrorDetectionAUROC(SimpleTestCase):
    """Test error detection AUROC."""

    def test_perfect_uncertainty_signal(self):
        is_correct = np.array([True, True, True, False, False, False])
        signal = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9])
        auroc = compute_error_detection_auroc(signal, is_correct)
        self.assertAlmostEqual(auroc, 1.0, places=4)

    def test_random_signal(self):
        np.random.seed(42)
        is_correct = np.random.rand(100) > 0.5
        signal = np.random.rand(100)
        auroc = compute_error_detection_auroc(signal, is_correct)
        self.assertGreater(auroc, 0.3)
        self.assertLess(auroc, 0.7)
