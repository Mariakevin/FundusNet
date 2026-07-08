"""Tests for retina_app.evaluation.figures — scientific plotting functions.

All tests use synthetic data and verify that plotting functions run without error,
produce files when save_path is given, and return matplotlib Figure objects.
"""

import os
import tempfile
import numpy as np
from unittest import TestCase
from evaluation.figures import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_reliability_diagram,
    plot_accuracy_refusal,
    plot_ablation_bars,
    plot_training_curves,
    plot_class_distribution,
    plot_model_comparison,
    plot_gradcam_grid,
    plot_uncertainty_analysis,
)

CATEGORIES = ["Healthy", "Cataract", "Glaucoma", "Retina Disease"]


def _make_probs(n=100, n_classes=4, seed=42):
    rng = np.random.RandomState(seed)
    raw = rng.rand(n, n_classes)
    return raw / raw.sum(axis=1, keepdims=True)


def _make_labels(n=100, n_classes=4, seed=42):
    rng = np.random.RandomState(seed)
    return rng.randint(0, n_classes, size=n)


class TestConfusionMatrix(TestCase):
    def test_runs_with_synthetic_data(self):
        cm = np.array([[80, 5, 3, 2], [4, 75, 6, 5], [3, 7, 70, 10], [2, 8, 5, 75]])
        fig = plot_confusion_matrix(cm, CATEGORIES)
        self.assertIsNotNone(fig)

    def test_saves_file(self):
        cm = np.eye(4, dtype=int) * 25
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "cm.png")
            fig = plot_confusion_matrix(cm, CATEGORIES, save_path=path)
            self.assertTrue(os.path.exists(path))
            self.assertTrue(os.path.exists(path.replace(".png", ".pdf")))

    def test_empty_cm(self):
        cm = np.zeros((4, 4), dtype=int)
        fig = plot_confusion_matrix(cm, CATEGORIES)
        self.assertIsNotNone(fig)


class TestROCCurves(TestCase):
    def test_runs(self):
        probs = _make_probs(200)
        labels = _make_labels(200)
        fig = plot_roc_curves(probs, labels, CATEGORIES)
        self.assertIsNotNone(fig)

    def test_saves(self):
        probs = _make_probs(200)
        labels = _make_labels(200)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "roc.png")
            plot_roc_curves(probs, labels, CATEGORIES, save_path=path)
            self.assertTrue(os.path.exists(path))

    def test_perfect_predictions(self):
        labels = _make_labels(100)
        probs = np.zeros((100, 4))
        probs[np.arange(100), labels] = 0.99
        probs[:, 0] += 0.01 / 4
        fig = plot_roc_curves(probs, labels, CATEGORIES)
        self.assertIsNotNone(fig)


class TestReliabilityDiagram(TestCase):
    def test_runs(self):
        probs = _make_probs(200)
        labels = _make_labels(200)
        fig = plot_reliability_diagram(probs, labels)
        self.assertIsNotNone(fig)

    def test_saves(self):
        probs = _make_probs(200)
        labels = _make_labels(200)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "rel.png")
            plot_reliability_diagram(probs, labels, save_path=path)
            self.assertTrue(os.path.exists(path))


class TestAccuracyRefusal(TestCase):
    def test_runs(self):
        refusal = [0.0, 0.1, 0.2, 0.3, 0.5]
        accs = [0.80, 0.83, 0.87, 0.91, 0.95]
        fig = plot_accuracy_refusal(refusal, accs)
        self.assertIsNotNone(fig)

    def test_saves(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "ar.png")
            plot_accuracy_refusal([0.0, 0.2, 0.5], [0.8, 0.85, 0.92], save_path=path)
            self.assertTrue(os.path.exists(path))


class TestAblationBars(TestCase):
    def test_runs(self):
        summary = {
            "full_system": {"f1_delta": 0, "accuracy_delta": 0},
            "no_fundus_validator": {"f1_delta": -0.023, "accuracy_delta": -0.018},
            "no_selective": {"f1_delta": -0.015, "accuracy_delta": -0.012},
            "no_tta": {"f1_delta": -0.008, "accuracy_delta": -0.006},
        }
        fig = plot_ablation_bars(summary)
        self.assertIsNotNone(fig)

    def test_saves(self):
        summary = {
            "full_system": {"f1_delta": 0},
            "no_mc": {"f1_delta": -0.01},
        }
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "ablation.png")
            plot_ablation_bars(summary, save_path=path)
            self.assertTrue(os.path.exists(path))


class TestTrainingCurves(TestCase):
    def test_runs(self):
        losses = np.linspace(1.5, 0.3, 15).tolist()
        accs = np.linspace(0.4, 0.92, 15).tolist()
        fig = plot_training_curves(losses, [l + 0.1 for l in losses], accs, [a - 0.03 for a in accs])
        self.assertIsNotNone(fig)


class TestClassDistribution(TestCase):
    def test_runs(self):
        labels = np.array([0] * 299 + [1] * 100 + [2] * 99 + [3] * 99)
        fig = plot_class_distribution(labels, CATEGORIES)
        self.assertIsNotNone(fig)


class TestModelComparison(TestCase):
    def test_runs(self):
        metrics = {
            "squeezenet": {"accuracy": 0.82, "macro_f1": 0.78},
            "efficientnet": {"accuracy": 0.88, "macro_f1": 0.85},
            "resnet": {"accuracy": 0.86, "macro_f1": 0.83},
        }
        fig = plot_model_comparison(metrics)
        self.assertIsNotNone(fig)


class TestGradCAMGrid(TestCase):
    def test_runs(self):
        rng = np.random.RandomState(0)
        images = [
            (rng.randint(0, 255, (64, 64, 3), dtype=np.uint8),
             rng.rand(64, 64))
            for _ in range(4)
        ]
        preds = ["Healthy", "Cataract", "Glaucoma", "Retina Disease"]
        fig = plot_gradcam_grid(images, preds)
        self.assertIsNotNone(fig)

    def test_with_ground_truth(self):
        rng = np.random.RandomState(1)
        images = [
            (rng.randint(0, 255, (64, 64, 3), dtype=np.uint8),
             rng.rand(64, 64))
            for _ in range(3)
        ]
        fig = plot_gradcam_grid(images, ["A", "B", "C"], ground_truths=["A", "B", "C"])
        self.assertIsNotNone(fig)


class TestUncertaintyAnalysis(TestCase):
    def test_runs(self):
        rng = np.random.RandomState(42)
        uncertainties = rng.rand(200)
        is_correct = rng.rand(200) > 0.3
        fig = plot_uncertainty_analysis(uncertainties, is_correct)
        self.assertIsNotNone(fig)

    def test_saves(self):
        rng = np.random.RandomState(42)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "unc.png")
            plot_uncertainty_analysis(rng.rand(100), rng.rand(100) > 0.5, save_path=path)
            self.assertTrue(os.path.exists(path))
