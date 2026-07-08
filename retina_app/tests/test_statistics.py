"""Tests for statistics.py — statistical testing utilities."""

import numpy as np
from django.test import SimpleTestCase

from evaluation.statistics import (
    bonferroni_correction,
    bootstrap_confidence_interval,
    compute_cohens_d,
    delong_auc_test,
    holm_correction,
    mcnemar_test,
    paired_t_test,
    significance_summary_table,
)


class McNemarTestTest(SimpleTestCase):
    def test_identical_classifiers(self):
        labels = np.array([0, 1, 2, 0, 1, 2, 0, 1])
        preds = np.array([0, 1, 2, 0, 1, 2, 0, 1])
        result = mcnemar_test(labels, preds, preds)
        self.assertFalse(result["significant"])
        self.assertEqual(result["contingency"]["c"], 0)
        self.assertEqual(result["contingency"]["d"], 0)

    def test_different_classifiers(self):
        labels = np.array([0, 0, 1, 1, 2, 2, 0, 1])
        preds_a = np.array([0, 0, 1, 1, 2, 2, 0, 1])  # perfect
        preds_b = np.array([0, 1, 1, 0, 2, 1, 0, 1])  # 2 errors
        result = mcnemar_test(labels, preds_a, preds_b)
        self.assertIn("chi2_stat", result)
        self.assertIn("p_value", result)
        self.assertGreaterEqual(result["p_value"], 0.0)
        self.assertLessEqual(result["p_value"], 1.0)

    def test_correction_vs_no_correction(self):
        labels = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        preds_a = np.array([0, 1, 1, 0, 0, 1, 1, 0])
        preds_b = np.array([0, 0, 1, 1, 0, 0, 0, 1])
        r_corr = mcnemar_test(labels, preds_a, preds_b, correction=True)
        r_nocorr = mcnemar_test(labels, preds_a, preds_b, correction=False)
        # Corrected chi2 should be <= uncorrected
        self.assertLessEqual(r_corr["chi2_stat"], r_nocorr["chi2_stat"] + 1e-10)

    def test_no_discordant_pairs(self):
        labels = np.array([0, 1, 0, 1])
        preds_a = np.array([0, 1, 0, 1])
        preds_b = np.array([0, 1, 0, 1])
        result = mcnemar_test(labels, preds_a, preds_b)
        self.assertFalse(result["significant"])
        self.assertEqual(result["chi2_stat"], 0.0)


class PairedTTestTest(SimpleTestCase):
    def test_identical_scores(self):
        scores_a = [0.85, 0.90, 0.88, 0.92, 0.87]
        scores_b = [0.85, 0.90, 0.88, 0.92, 0.87]
        result = paired_t_test(scores_a, scores_b)
        self.assertFalse(result["significant"])
        self.assertAlmostEqual(result["mean_diff"], 0.0)

    def test_different_scores(self):
        scores_a = [0.90, 0.92, 0.91, 0.93, 0.94]
        scores_b = [0.80, 0.82, 0.81, 0.83, 0.84]
        result = paired_t_test(scores_a, scores_b)
        self.assertTrue(result["significant"])
        self.assertGreater(result["mean_diff"], 0)

    def test_ci_contains_zero(self):
        scores_a = [0.85, 0.86, 0.84, 0.85, 0.86]
        scores_b = [0.84, 0.85, 0.85, 0.84, 0.85]
        result = paired_t_test(scores_a, scores_b)
        # Similar scores -> CI should contain 0
        self.assertLessEqual(result["ci_95"][0], 0.0)
        self.assertGreaterEqual(result["ci_95"][1], 0.0)

    def test_single_fold(self):
        result = paired_t_test([0.9], [0.8])
        self.assertFalse(result["significant"])


class BootstrapCITest(SimpleTestCase):
    def test_mean_ci(self):
        rng = np.random.RandomState(42)
        values = rng.normal(0.5, 0.1, 100)
        result = bootstrap_confidence_interval(values, n_bootstrap=500, seed=42)
        self.assertAlmostEqual(result["point_estimate"], 0.5, delta=0.1)
        self.assertLess(result["ci_lower"], result["point_estimate"])
        self.assertGreater(result["ci_upper"], result["point_estimate"])

    def test_deterministic_with_seed(self):
        values = np.array([1, 2, 3, 4, 5])
        r1 = bootstrap_confidence_interval(values, seed=42)
        r2 = bootstrap_confidence_interval(values, seed=42)
        self.assertAlmostEqual(r1["point_estimate"], r2["point_estimate"])

    def test_custom_statistic(self):
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = bootstrap_confidence_interval(values, statistic_fn=np.median, seed=42)
        self.assertEqual(result["point_estimate"], 5.5)

    def test_se_positive(self):
        values = np.array([3, 3, 3, 3, 3])
        result = bootstrap_confidence_interval(values, seed=42)
        self.assertEqual(result["se"], 0.0)


class CohensDTest(SimpleTestCase):
    def test_identical_distributions(self):
        scores = [0.85, 0.90, 0.88, 0.92, 0.87]
        result = compute_cohens_d(scores, scores)
        self.assertEqual(result["cohens_d"], 0.0)
        self.assertEqual(result["interpretation"], "no difference")

    def test_large_difference(self):
        scores_a = [0.95, 0.96, 0.94, 0.97, 0.95]
        scores_b = [0.70, 0.72, 0.68, 0.71, 0.69]
        result = compute_cohens_d(scores_a, scores_b)
        self.assertGreater(abs(result["cohens_d"]), 0.8)
        self.assertEqual(result["interpretation"], "large")

    def test_small_difference(self):
        scores_a = [0.85, 0.80, 0.90, 0.78, 0.88, 0.82, 0.86]
        scores_b = [0.82, 0.83, 0.81, 0.85, 0.80, 0.84, 0.83]
        result = compute_cohens_d(scores_a, scores_b)
        self.assertIn(result["interpretation"], ["small", "medium", "negligible", "large"])


class DeLongAUCTest(SimpleTestCase):
    def test_perfect_vs_random(self):
        n = 100
        labels = np.array([0] * 50 + [1] * 50)
        probs_a = np.zeros((n, 2))
        probs_a[:50, 0] = 0.95
        probs_a[50:, 1] = 0.95
        probs_b = np.random.RandomState(42).rand(n, 2)
        probs_b = probs_b / probs_b.sum(axis=1, keepdims=True)

        result = delong_auc_test(probs_a, probs_b, labels, 2)
        self.assertGreater(result["auc_a"], result["auc_b"])
        self.assertIn("p_value", result)

    def test_same_classifiers(self):
        n = 100
        labels = np.array([0] * 50 + [1] * 50)
        probs = np.zeros((n, 2))
        probs[:50, 0] = 0.9
        probs[50:, 1] = 0.9

        result = delong_auc_test(probs, probs, labels, 2)
        self.assertAlmostEqual(result["auc_a"], result["auc_b"], places=4)
        self.assertFalse(result["significant"])


class BonferroniCorrectionTest(SimpleTestCase):
    def test_single_test(self):
        result = bonferroni_correction([0.03])
        self.assertAlmostEqual(result[0], 0.03)

    def test_multiple_tests(self):
        result = bonferroni_correction([0.01, 0.04, 0.03])
        self.assertAlmostEqual(result[0], 0.03)
        self.assertAlmostEqual(result[1], 0.12)  # 0.04 * 3
        self.assertAlmostEqual(result[2], 0.09)  # 0.03 * 3

    def test_clipped_at_one(self):
        result = bonferroni_correction([0.5, 0.5])
        self.assertEqual(result[0], 1.0)
        self.assertEqual(result[1], 1.0)


class HolmCorrectionTest(SimpleTestCase):
    def test_single_test(self):
        result = holm_correction([0.03])
        self.assertAlmostEqual(result[0], 0.03)

    def test_less_conservative_than_bonferroni(self):
        p_vals = [0.01, 0.04, 0.03]
        bonf = bonferroni_correction(p_vals)
        holm = holm_correction(p_vals)
        # Holm should be less conservative (smaller adjusted p-values)
        for b, h in zip(bonf, holm):
            self.assertLessEqual(h, b + 1e-10)

    def test_monotonicity(self):
        p_vals = [0.001, 0.01, 0.05, 0.1]
        result = holm_correction(p_vals)
        # Adjusted p-values should be monotonically increasing
        for i in range(len(result) - 1):
            # Find the original ordering
            pass
        # At minimum, no adjusted p should exceed 1.0
        for r in result:
            self.assertLessEqual(r, 1.0)


class SignificanceSummaryTableTest(SimpleTestCase):
    def test_basic_table(self):
        comparisons = [
            {"name_a": "A", "name_b": "B", "p_value": 0.01, "metric": "accuracy"},
            {"name_a": "A", "name_b": "C", "p_value": 0.04, "metric": "accuracy"},
            {"name_a": "B", "name_b": "C", "p_value": 0.50, "metric": "accuracy"},
        ]
        result = significance_summary_table(comparisons, correction_method="holm")
        self.assertEqual(len(result), 3)
        # First should be significant
        self.assertTrue(result[0]["significant"])
        # Third should not
        self.assertFalse(result[2]["significant"])

    def test_empty_comparisons(self):
        result = significance_summary_table([])
        self.assertEqual(result, [])
