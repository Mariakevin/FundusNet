"""Statistical testing utilities for RetinaAI evaluation.

Provides significance tests, confidence intervals, and effect size measures
for rigorous comparison of classification strategies.
"""

import numpy as np
from scipy import stats as scipy_stats


def mcnemar_test(labels_true, preds_a, preds_b, correction=True):
    """McNemar's test for comparing two classifiers on the same data.

    Tests whether the disagreement between two classifiers is systematic.
    Null hypothesis: both classifiers have the same error rate.

    The contingency table:
                    preds_b correct | preds_b wrong
    preds_a correct |      b        |       c
    preds_a wrong   |      d        |       e

    Args:
        labels_true: (n_samples,) ground truth labels
        preds_a: (n_samples,) predictions from classifier A
        preds_b: (n_samples,) predictions from classifier B
        correction: whether to apply continuity correction (Yates)

    Returns:
        dict with chi2_stat, p_value, significant (at alpha=0.05)

    """
    correct_a = preds_a == labels_true
    correct_b = preds_b == labels_true

    # Contingency: b = both correct, c = A correct B wrong, d = A wrong B correct, e = both wrong
    b = int(np.sum(correct_a & correct_b))
    c = int(np.sum(correct_a & ~correct_b))
    d = int(np.sum(~correct_a & correct_b))
    e = int(np.sum(~correct_a & ~correct_b))

    # McNemar focuses on discordant pairs (c and d)
    n_discordant = c + d
    if n_discordant == 0:
        return {
            "chi2_stat": 0.0,
            "p_value": 1.0,
            "significant": False,
            "contingency": {"b": b, "c": c, "d": d, "e": e},
        }

    if correction:
        chi2_stat = (abs(c - d) - 1) ** 2 / (c + d)
    else:
        chi2_stat = (c - d) ** 2 / (c + d)

    p_value = 1.0 - scipy_stats.chi2.cdf(chi2_stat, df=1)

    return {
        "chi2_stat": float(chi2_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
        "contingency": {"b": b, "c": c, "d": d, "e": e},
    }


def paired_t_test(scores_a, scores_b):
    """Paired t-test for comparing two methods across folds.

    Tests whether the mean difference between paired observations is zero.

    Args:
        scores_a: (n_folds,) metric scores from method A
        scores_b: (n_folds,) metric scores from method B

    Returns:
        dict with t_stat, p_value, significant (at alpha=0.05), mean_diff, ci_95

    """
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    diffs = scores_a - scores_b
    n = len(diffs)

    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))

    if n < 2 or std_diff == 0:
        return {
            "t_stat": 0.0,
            "p_value": 1.0,
            "significant": False,
            "mean_diff": mean_diff,
            "ci_95": (mean_diff, mean_diff),
        }

    t_stat = mean_diff / (std_diff / np.sqrt(n))
    p_value = 2.0 * (1.0 - scipy_stats.t.cdf(abs(t_stat), df=n - 1))

    # 95% CI for mean difference
    t_crit = scipy_stats.t.ppf(0.975, df=n - 1)
    margin = t_crit * std_diff / np.sqrt(n)
    ci_95 = (float(mean_diff - margin), float(mean_diff + margin))

    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
        "mean_diff": mean_diff,
        "ci_95": ci_95,
    }


def bootstrap_confidence_interval(values, n_bootstrap=2000, ci_level=0.95, statistic_fn=None, seed=42):
    """Compute bootstrap confidence interval for any statistic.

    Args:
        values: (n_samples,) observed values
        n_bootstrap: number of bootstrap resamples
        ci_level: confidence level (e.g. 0.95 for 95% CI)
        statistic_fn: function to compute statistic (default: np.mean)
        seed: random seed for reproducibility

    Returns:
        dict with point_estimate, ci_lower, ci_upper, se

    """
    values = np.asarray(values, dtype=float)
    rng = np.random.RandomState(seed)

    if statistic_fn is None:
        statistic_fn = np.mean

    point_estimate = float(statistic_fn(values))

    boot_stats = []
    n = len(values)
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        boot_stats.append(statistic_fn(sample))

    boot_stats = np.array(boot_stats)
    alpha = 1.0 - ci_level
    ci_lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    se = float(np.std(boot_stats))

    return {
        "point_estimate": point_estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "se": se,
    }


def compute_cohens_d(scores_a, scores_b):
    """Cohen's d effect size for paired observations.

    Measures the magnitude of difference between two methods.
    Interpretation: 0.2=small, 0.5=medium, 0.8=large

    Args:
        scores_a: (n_folds,) metric scores from method A
        scores_b: (n_folds,) metric scores from method B

    Returns:
        dict with cohens_d, interpretation

    """
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    diffs = scores_a - scores_b

    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)

    if std_diff == 0:
        return {"cohens_d": 0.0, "interpretation": "no difference"}

    d = mean_diff / std_diff
    d = float(d)

    abs_d = abs(d)
    if abs_d < 0.2:
        interpretation = "negligible"
    elif abs_d < 0.5:
        interpretation = "small"
    elif abs_d < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    return {"cohens_d": d, "interpretation": interpretation}


def delong_auc_test(probs_a, probs_b, labels, n_classes):
    """DeLong test for comparing AUROC between two classifiers.

    Tests whether the difference in AUC between two classifiers is
    statistically significant.

    Args:
        probs_a: (n_samples, n_classes) predicted probabilities from A
        probs_b: (n_samples, n_classes) predicted probabilities from B
        labels: (n_samples,) true class indices
        n_classes: number of classes

    Returns:
        dict with auc_a, auc_b, z_stat, p_value, significant

    """

    def _auc_scores(probs, labels, class_idx):
        binary_labels = (labels == class_idx).astype(float)
        scores = probs[:, class_idx]
        desc_order = np.argsort(-scores)
        binary_labels = binary_labels[desc_order]

        n_pos = np.sum(binary_labels == 1)
        n_neg = np.sum(binary_labels == 0)

        if n_pos == 0 or n_neg == 0:
            return 0.5, 0.0, 0.0

        tpr_list = [0.0]
        fpr_list = [0.0]
        tp = 0
        fp = 0
        for label in binary_labels:
            if label == 1:
                tp += 1
            else:
                fp += 1
            tpr_list.append(tp / n_pos)
            fpr_list.append(fp / n_neg)

        auc = 0.0
        for i in range(1, len(tpr_list)):
            auc += (fpr_list[i] - fpr_list[i - 1]) * (tpr_list[i] + tpr_list[i - 1]) / 2

        # Simplified variance estimate
        var = auc * (1 - auc) / max(n_pos * n_neg, 1)
        return float(auc), float(var), float(n_pos + n_neg)

    aucs_a = []
    aucs_b = []
    vars_a = []
    vars_b = []

    for c in range(n_classes):
        auc_a, var_a, _ = _auc_scores(probs_a, labels, c)
        auc_b, var_b, _ = _auc_scores(probs_b, labels, c)
        aucs_a.append(auc_a)
        aucs_b.append(auc_b)
        vars_a.append(var_a)
        vars_b.append(var_b)

    # Macro AUC
    mean_auc_a = np.mean(aucs_a)
    mean_auc_b = np.mean(aucs_b)

    # Simplified z-test using combined variance
    var_diff = np.mean(vars_a) + np.mean(vars_b)
    if var_diff == 0:
        return {
            "auc_a": float(mean_auc_a),
            "auc_b": float(mean_auc_b),
            "z_stat": 0.0,
            "p_value": 1.0,
            "significant": False,
        }

    z_stat = (mean_auc_a - mean_auc_b) / np.sqrt(var_diff)
    p_value = 2.0 * (1.0 - scipy_stats.norm.cdf(abs(z_stat)))

    return {
        "auc_a": float(mean_auc_a),
        "auc_b": float(mean_auc_b),
        "z_stat": float(z_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
    }


def bonferroni_correction(p_values):
    """Bonferroni correction for multiple comparisons.

    Most conservative correction: alpha_adjusted = alpha / n_tests.

    Args:
        p_values: list of p-values

    Returns:
        list of adjusted p-values (clipped at 1.0)

    """
    n = len(p_values)
    return [min(p * n, 1.0) for p in p_values]


def holm_correction(p_values):
    """Holm-Bonferroni step-down correction (less conservative than Bonferroni).

    Args:
        p_values: list of p-values

    Returns:
        list of adjusted p-values

    """
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n

    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted[orig_idx] = min(p * (n - rank), 1.0)

    # Enforce monotonicity (step-down)
    for i in range(n - 1):
        orig_idx_cur = indexed[i][0]
        orig_idx_next = indexed[i + 1][0]
        if adjusted[orig_idx_next] < adjusted[orig_idx_cur]:
            adjusted[orig_idx_next] = adjusted[orig_idx_cur]

    return adjusted


def significance_summary_table(comparisons, correction_method="holm"):
    """Build a summary table of pairwise significance tests.

    Args:
        comparisons: list of dicts, each with:
            - name_a, name_b: strategy names
            - p_value: raw p-value from any test
            - metric: which metric was compared
            - effect_size: optional Cohen's d
        correction_method: 'bonferroni' or 'holm'

    Returns:
        list of dicts with corrected p-values and significance flags

    """
    p_values = [c["p_value"] for c in comparisons]

    if correction_method == "bonferroni":
        adjusted = bonferroni_correction(p_values)
    else:
        adjusted = holm_correction(p_values)

    results = []
    for comp, adj_p in zip(comparisons, adjusted):
        results.append(
            {
                "comparison": f"{comp['name_a']} vs {comp['name_b']}",
                "metric": comp.get("metric", "accuracy"),
                "raw_p": comp["p_value"],
                "adjusted_p": adj_p,
                "significant": bool(adj_p < 0.05),
                "effect_size": comp.get("effect_size"),
            }
        )

    return results
