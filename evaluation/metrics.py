"""Metric computation utilities for RetinaAI evaluation.

Provides calibration metrics (ECE, MCE, Brier), per-class classification
metrics, and confusion matrix generation.
"""

import numpy as np
from collections import defaultdict


def compute_ece(probs, labels, n_bins=15):
    """Expected Calibration Error.

    Measures how well predicted probabilities match observed frequencies.
    ECE = sum(|accuracy_in_bin| / n * |avg_confidence_in_bin - avg_accuracy_in_bin|)

    Args:
        probs: (n_samples, n_classes) predicted probabilities
        labels: (n_samples,) true class indices
        n_bins: number of confidence bins

    Returns:
        float: ECE value (0 = perfectly calibrated)
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.sum(in_bin) / len(probs)

        if prop_in_bin > 0:
            avg_confidence = np.mean(confidences[in_bin])
            avg_accuracy = np.mean(accuracies[in_bin])
            ece += prop_in_bin * abs(avg_accuracy - avg_confidence)

    return float(ece)


def compute_mce(probs, labels, n_bins=15):
    """Maximum Calibration Error.

    Like ECE but reports the worst-bin calibration gap instead of average.

    Args:
        probs: (n_samples, n_classes) predicted probabilities
        labels: (n_samples,) true class indices
        n_bins: number of confidence bins

    Returns:
        float: MCE value
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    max_gap = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])

        if np.sum(in_bin) > 0:
            avg_confidence = np.mean(confidences[in_bin])
            avg_accuracy = np.mean(accuracies[in_bin])
            max_gap = max(max_gap, abs(avg_accuracy - avg_confidence))

    return float(max_gap)


def compute_brier(probs, labels):
    """Brier Score (lower is better, 0 = perfect).

    Mean squared difference between predicted probabilities and one-hot labels.

    Args:
        probs: (n_samples, n_classes) predicted probabilities
        labels: (n_samples,) true class indices

    Returns:
        float: Brier score
    """
    n_classes = probs.shape[1]
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def compute_reliability_data(probs, labels, n_bins=15):
    """Compute bin-level data for reliability diagrams.

    Returns:
        dict with keys: bin_centers, bin_accuracies, bin_confidences, bin_counts
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        count = np.sum(in_bin)

        if count > 0:
            bin_centers.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
            bin_accuracies.append(float(np.mean(accuracies[in_bin])))
            bin_confidences.append(float(np.mean(confidences[in_bin])))
            bin_counts.append(int(count))
        else:
            bin_centers.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
            bin_accuracies.append(0.0)
            bin_confidences.append(0.0)
            bin_counts.append(0)

    return {
        "bin_centers": bin_centers,
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts,
    }


def per_class_metrics(predictions, labels, categories):
    """Compute per-class precision, recall, F1, and support.

    Args:
        predictions: (n_samples,) predicted class indices
        labels: (n_samples,) true class indices
        categories: list of class names

    Returns:
        dict: {class_name: {precision, recall, f1, support, tp, fp, fn}}
    """
    results = {}
    for i, cat in enumerate(categories):
        tp = int(np.sum((predictions == i) & (labels == i)))
        fp = int(np.sum((predictions == i) & (labels != i)))
        fn = int(np.sum((predictions != i) & (labels == i)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = int(np.sum(labels == i))

        results[cat] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    return results


def confusion_matrix(predictions, labels, n_classes):
    """Compute confusion matrix.

    Args:
        predictions: (n_samples,) predicted class indices
        labels: (n_samples,) true class indices
        n_classes: number of classes

    Returns:
        (n_classes, n_classes) numpy array where [i,j] = true=i, pred=j
    """
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(labels, predictions):
        cm[t, p] += 1
    return cm


def overall_metrics(predictions, labels, categories):
    """Compute overall accuracy, macro F1, weighted F1, and AUC.

    Args:
        predictions: (n_samples,) predicted class indices
        labels: (n_samples,) true class indices
        categories: list of class names

    Returns:
        dict with accuracy, macro_f1, weighted_f1, total_samples
    """
    n_classes = len(categories)
    accuracy = float(np.mean(predictions == labels))

    pcm = per_class_metrics(predictions, labels, categories)
    supports = [pcm[c]["support"] for c in categories]
    f1s = [pcm[c]["f1"] for c in categories]

    macro_f1 = float(np.mean(f1s))
    total = sum(supports)
    weighted_f1 = float(np.sum([f1s[i] * supports[i] for i in range(n_classes)]) / total) if total > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "total_samples": total,
    }


def compute_auroc_per_class(probs, labels, n_classes):
    """Compute one-vs-rest AUROC for each class.

    Uses simple trapezoidal AUC computation without sklearn dependency.

    Args:
        probs: (n_samples, n_classes) predicted probabilities
        labels: (n_samples,) true class indices
        n_classes: number of classes

    Returns:
        list of AUROC values per class
    """
    aurocs = []
    for c in range(n_classes):
        binary_labels = (labels == c).astype(float)
        scores = probs[:, c]

        # Sort by descending score
        desc_order = np.argsort(-scores)
        binary_labels = binary_labels[desc_order]

        n_pos = np.sum(binary_labels == 1)
        n_neg = np.sum(binary_labels == 0)

        if n_pos == 0 or n_neg == 0:
            aurocs.append(0.5)
            continue

        # Trapezoidal AUC
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

        # Trapezoidal rule
        auc = 0.0
        for i in range(1, len(tpr_list)):
            auc += (fpr_list[i] - fpr_list[i - 1]) * (tpr_list[i] + tpr_list[i - 1]) / 2

        aurocs.append(float(auc))

    return aurocs


def compute_error_detection_auroc(uncertainty_signals, is_correct):
    """AUROC for detecting incorrect predictions using uncertainty signals.

    Args:
        uncertainty_signals: (n_samples,) uncertainty values (higher = more uncertain)
        is_correct: (n_samples,) boolean array (True = correct prediction)

    Returns:
        float: AUROC (1.0 = perfect error detection, 0.5 = random)
    """
    return compute_auroc_per_class(
        np.column_stack([1 - uncertainty_signals, uncertainty_signals]),
        (~is_correct).astype(int),
        2,
    )[1]
