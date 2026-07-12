"""Metric computation utilities — upgraded with torchmetrics.

Provides:
- Calibration metrics (ECE, MCE, Brier) via torchmetrics
- Per-class classification metrics
- Confusion matrix generation
- AUROC computation
"""

import numpy as np
import torch


def compute_ece(probs, labels, n_bins=15):
    """Expected Calibration Error using torchmetrics."""
    try:
        from torchmetrics.classification import MulticlassExpectedCalibrationError

        probs_t = torch.tensor(probs, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.long)
        n_classes = probs_t.shape[1]
        metric = MulticlassExpectedCalibrationError(num_classes=n_classes, n_bins=n_bins)
        return float(metric(probs_t, labels_t))
    except ImportError:
        pass

    # Fallback: manual implementation
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
    """Maximum Calibration Error."""
    try:
        from torchmetrics.classification import MulticlassMaximumCalibrationError

        probs_t = torch.tensor(probs, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.long)
        n_classes = probs_t.shape[1]
        metric = MulticlassMaximumCalibrationError(num_classes=n_classes, n_bins=n_bins)
        return float(metric(probs_t, labels_t))
    except ImportError:
        pass

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
    """Brier Score using torchmetrics."""
    try:
        from torchmetrics.classification import MulticlassBrierScore

        probs_t = torch.tensor(probs, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.long)
        n_classes = probs_t.shape[1]
        metric = MulticlassBrierScore(num_classes=n_classes)
        return float(metric(probs_t, labels_t))
    except ImportError:
        pass

    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def compute_reliability_data(probs, labels, n_bins=15):
    """Compute bin-level data for reliability diagrams."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers, bin_accuracies, bin_confidences, bin_counts = [], [], [], []
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        count = int(np.sum(in_bin))
        bin_centers.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
        bin_accuracies.append(float(np.mean(accuracies[in_bin])) if count > 0 else 0.0)
        bin_confidences.append(float(np.mean(confidences[in_bin])) if count > 0 else 0.0)
        bin_counts.append(count)
    return {
        "bin_centers": bin_centers,
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts,
    }


def per_class_metrics(predictions, labels, categories):
    """Compute per-class precision, recall, F1, and support."""
    results = {}
    for i, cat in enumerate(categories):
        tp = int(np.sum((predictions == i) & (labels == i)))
        fp = int(np.sum((predictions == i) & (labels != i)))
        fn = int(np.sum((predictions != i) & (labels == i)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results[cat] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int(np.sum(labels == i)),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return results


def confusion_matrix(predictions, labels, n_classes):
    """Compute confusion matrix."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(labels, predictions):
        cm[t, p] += 1
    return cm


def overall_metrics(predictions, labels, categories):
    """Compute overall accuracy, macro F1, weighted F1 using torchmetrics."""
    try:
        from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

        preds_t = torch.tensor(predictions, dtype=torch.long)
        labels_t = torch.tensor(labels, dtype=torch.long)
        n_classes = len(categories)
        acc = float(MulticlassAccuracy(num_classes=n_classes, average="micro")(preds_t, labels_t))
        macro_f1 = float(MulticlassF1Score(num_classes=n_classes, average="macro")(preds_t, labels_t))
        weighted_f1 = float(MulticlassF1Score(num_classes=n_classes, average="weighted")(preds_t, labels_t))
        return {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "total_samples": len(labels),
        }
    except ImportError:
        pass

    n_classes = len(categories)
    accuracy = float(np.mean(predictions == labels))
    pcm = per_class_metrics(predictions, labels, categories)
    f1s = [pcm[c]["f1"] for c in categories]
    supports = [pcm[c]["support"] for c in categories]
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
    """Compute one-vs-rest AUROC for each class using torchmetrics."""
    try:
        from torchmetrics.classification import MulticlassAUROC

        probs_t = torch.tensor(probs, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.long)
        metric = MulticlassAUROC(num_classes=n_classes, average=None)
        aurocs = metric(probs_t, labels_t)
        return [float(a) for a in aurocs]
    except ImportError:
        pass

    aurocs = []
    for c in range(n_classes):
        binary_labels = (labels == c).astype(float)
        scores = probs[:, c]
        desc_order = np.argsort(-scores)
        binary_labels = binary_labels[desc_order]
        n_pos = np.sum(binary_labels == 1)
        n_neg = np.sum(binary_labels == 0)
        if n_pos == 0 or n_neg == 0:
            aurocs.append(0.5)
            continue
        tpr_list, fpr_list = [0.0], [0.0]
        tp, fp = 0, 0
        for label in binary_labels:
            if label == 1:
                tp += 1
            else:
                fp += 1
            tpr_list.append(tp / n_pos)
            fpr_list.append(fp / n_neg)
        auc = sum(
            (fpr_list[i] - fpr_list[i - 1]) * (tpr_list[i] + tpr_list[i - 1]) / 2 for i in range(1, len(tpr_list))
        )
        aurocs.append(float(auc))
    return aurocs


def compute_error_detection_auroc(uncertainty_signals, is_correct):
    """AUROC for detecting incorrect predictions using uncertainty signals."""
    return compute_auroc_per_class(
        np.column_stack([1 - uncertainty_signals, uncertainty_signals]),
        (~is_correct).astype(int),
        2,
    )[1]
