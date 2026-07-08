"""Scientific figures for RetinaAI research paper.

Generates publication-quality plots (300 DPI, PDF+PNG) for:
confusion matrices, ROC curves, reliability diagrams, accuracy-refusal
tradeoffs, ablation studies, training curves, class distributions,
model comparisons, Grad-CAM grids, and uncertainty analysis.

Usage:
    from evaluation.figures import plot_confusion_matrix
    plot_confusion_matrix(cm, categories, save_path="figures/cm.pdf")
"""

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
from itertools import cycle

import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve

STYLE = {
    "figure.figsize": (8, 6),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
}

CATEGORY_SHORT = {
    "Healthy": "Healthy",
    "Cataract": "Cataract",
    "Glaucoma": "Glaucoma",
    "Retina Disease": "Retina DR",
}


def _apply_style():
    plt.rcParams.update(STYLE)
    plt.style.use("seaborn-v0_8-whitegrid")


def _save_fig(fig, save_path):
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm, categories, save_path=None, title="Confusion Matrix"):
    """Plot confusion matrix as annotated heatmap.

    Args:
        cm: (n_classes, n_classes) numpy array
        categories: list of class names
        save_path: optional path to save figure (.png or .pdf)
        title: plot title

    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(7, 6))

    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    im = ax.imshow(cm_normalized, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    short_names = [CATEGORY_SHORT.get(c, c) for c in categories]
    ax.set(
        xticks=np.arange(len(categories)),
        yticks=np.arange(len(categories)),
        xticklabels=short_names,
        yticklabels=short_names,
        ylabel="True Label",
        xlabel="Predicted Label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm_normalized.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n({cm_normalized[i, j]:.2f})",
                ha="center",
                va="center",
                color="white" if cm_normalized[i, j] > thresh else "black",
                fontsize=9,
            )

    fig.tight_layout()
    _save_fig(fig, save_path)
    return fig


def plot_roc_curves(probs, labels, categories, save_path=None, title="ROC Curves"):
    """Plot per-class one-vs-rest ROC curves with AUROC values.

    Args:
        probs: (n_samples, n_classes) predicted probabilities
        labels: (n_samples,) true class indices
        categories: list of class names
        save_path: optional save path
        title: plot title

    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = cycle(["#2196F3", "#FF9800", "#4CAF50", "#F44336"])
    n_classes = len(categories)

    for i, color in zip(range(n_classes), colors):
        binary_labels = (labels == i).astype(int)
        fpr, tpr, _ = roc_curve(binary_labels, probs[:, i])
        roc_auc = auc(fpr, tpr)
        short = CATEGORY_SHORT.get(categories[i], categories[i])
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{short} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC = 0.500)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title=title)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)
    fig.tight_layout()
    _save_fig(fig, save_path)
    return fig


def plot_reliability_diagram(probs, labels, save_path=None, n_bins=15, title="Reliability Diagram"):
    """Plot reliability diagram (calibration curve).

    Args:
        probs: (n_samples, n_classes) predicted probabilities
        labels: (n_samples,) true class indices
        save_path: optional save path
        n_bins: number of confidence bins
        title: plot title

    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(7, 6))

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        count = np.sum(in_bin)
        if count > 0:
            bin_centers.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
            bin_accs.append(float(np.mean(accuracies[in_bin])))
            bin_confs.append(float(np.mean(confidences[in_bin])))
            bin_counts.append(int(count))

    bin_centers = np.array(bin_centers)
    bin_accs = np.array(bin_accs)
    bin_confs = np.array(bin_confs)
    bin_counts = np.array(bin_counts)

    if len(bin_centers) > 0:
        bar_width = 0.8 * (bin_boundaries[1] - bin_boundaries[0])
        ax.bar(
            bin_centers,
            bin_accs,
            width=bar_width,
            alpha=0.7,
            color="#2196F3",
            label="Accuracy",
            edgecolor="white",
            linewidth=0.5,
        )
        ax.plot(bin_confs, bin_accs, "o-", color="#FF5722", lw=2, markersize=6, label="Calibration curve")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect calibration")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set(xlabel="Confidence", ylabel="Accuracy", title=title)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)

    ece = _quick_ece(confidences, accuracies, n_bins)
    ax.text(
        0.98,
        0.02,
        f"ECE = {ece:.4f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
    )

    fig.tight_layout()
    _save_fig(fig, save_path)
    return fig


def _quick_ece(confidences, accuracies, n_bins):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop = np.sum(in_bin) / len(confidences)
        if prop > 0:
            ece += prop * abs(np.mean(accuracies[in_bin]) - np.mean(confidences[in_bin]))
    return ece


def plot_accuracy_refusal(
    refusal_rates,
    accuracies,
    save_path=None,
    title="Accuracy-Refusal Tradeoff",
    xlabel="Refusal Rate",
    ylabel="Accuracy (Accepted)",
):
    """Plot accuracy vs. refusal rate tradeoff curve.

    Args:
        refusal_rates: list of refusal rates (0 to 1)
        accuracies: list of accuracies at each refusal rate
        save_path: optional save path
        title, xlabel, ylabel: axis labels

    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(refusal_rates, accuracies, "o-", color="#2196F3", lw=2.5, markersize=8, label="Selective Ensemble")
    ax.fill_between(refusal_rates, accuracies, alpha=0.1, color="#2196F3")

    if len(refusal_rates) > 1:
        baseline_acc = accuracies[0]
        ax.axhline(y=baseline_acc, color="#9E9E9E", linestyle="--", lw=1.5, label=f"Full ensemble ({baseline_acc:.3f})")

    ax.set_xlim([-0.02, max(max(refusal_rates) * 1.05, 0.5)])
    ax.set_ylim([min(accuracies) * 0.95, min(max(accuracies) * 1.02, 1.0)])
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)
    fig.tight_layout()
    _save_fig(fig, save_path)
    return fig


def plot_ablation_bars(ablation_summary, save_path=None, title="Ablation Study — Component Contribution"):
    """Plot ablation study as horizontal bar chart of F1 deltas.

    Args:
        ablation_summary: dict from ablation.run_ablation_study()["summary"]
        save_path: optional save path
        title: plot title

    """
    _apply_style()
    configs = []
    deltas = []
    colors = []

    for name, metrics in ablation_summary.items():
        if name == "full_system":
            continue
        delta = metrics.get("f1_delta", metrics.get("accuracy_delta", 0))
        if isinstance(delta, (int, float)):
            configs.append(name.replace("no_", "w/o ").replace("_", " ").title())
            deltas.append(delta)
            colors.append("#F44336" if delta < 0 else "#4CAF50")

    if not configs:
        return None

    fig, ax = plt.subplots(figsize=(9, max(4, len(configs) * 0.8)))
    y_pos = np.arange(len(configs))
    bars = ax.barh(y_pos, deltas, color=colors, edgecolor="white", height=0.6)

    for bar, val in zip(bars, deltas):
        x_pos = val + (0.001 if val >= 0 else -0.001)
        ax.text(
            x_pos,
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.4f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(configs)
    ax.axvline(x=0, color="black", lw=0.8)
    ax.set(xlabel="Macro F1 Delta vs. Full System", title=title)
    ax.text(
        0.02,
        0.98,
        "← Worse | Better →",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        style="italic",
        color="#666",
    )
    fig.tight_layout()
    _save_fig(fig, save_path)
    return fig


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path=None, title="Training Curves"):
    """Plot training and validation loss/accuracy curves.

    Args:
        train_losses, val_losses: lists of per-epoch loss values
        train_accs, val_accs: lists of per-epoch accuracy values
        save_path: optional save path
        title: plot title

    """
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, "o-", color="#2196F3", lw=2, label="Train")
    ax1.plot(epochs, val_losses, "s-", color="#FF9800", lw=2, label="Validation")
    ax1.set(xlabel="Epoch", ylabel="Loss", title="Loss")
    ax1.legend(frameon=True)
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_accs, "o-", color="#2196F3", lw=2, label="Train")
    ax2.plot(epochs, val_accs, "s-", color="#FF9800", lw=2, label="Validation")
    ax2.set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy")
    ax2.legend(frameon=True)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, save_path)
    return fig


def plot_class_distribution(labels, categories, save_path=None, title="Dataset Class Distribution"):
    """Plot bar chart of class distribution.

    Args:
        labels: array of integer class indices
        categories: list of class names
        save_path: optional save path
        title: plot title

    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    unique, counts = np.unique(labels, return_counts=True)
    short_names = [CATEGORY_SHORT.get(categories[i], categories[i]) for i in unique]
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#F44336"][: len(unique)]

    bars = ax.bar(short_names, counts, color=colors, edgecolor="white", width=0.6)

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3,
            str(count),
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    ax.set(xlabel="Class", ylabel="Number of Images", title=title)
    ax.set_ylim([0, max(counts) * 1.15])
    fig.tight_layout()
    _save_fig(fig, save_path)
    return fig


def plot_model_comparison(model_metrics, save_path=None, title="Individual Model Comparison"):
    """Compare individual models on accuracy and macro F1.

    Args:
        model_metrics: dict of {model_name: {accuracy, macro_f1, ...}}
        save_path: optional save path
        title: plot title

    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(9, 5))

    names = list(model_metrics.keys())
    accs = [model_metrics[n].get("accuracy", 0) for n in names]
    f1s = [model_metrics[n].get("macro_f1", 0) for n in names]

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width / 2, accs, width, label="Accuracy", color="#2196F3", edgecolor="white")
    bars2 = ax.bar(x + width / 2, f1s, width, label="Macro F1", color="#FF9800", edgecolor="white")

    for bar in bars1:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for bar in bars2:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim([0, 1.05])
    ax.set(ylabel="Score", title=title)
    ax.legend(frameon=True)
    fig.tight_layout()
    _save_fig(fig, save_path)
    return fig


def plot_gradcam_grid(images, predictions, ground_truths=None, save_path=None, title="Grad-CAM Explanations"):
    """Display a grid of Grad-CAM heatmap overlays.

    Args:
        images: list of (original, heatmap) tuples or (original, heatmap, overlay) tuples
        predictions: list of predicted class names
        ground_truths: optional list of true class names
        save_path: optional save path
        title: plot title

    """
    _apply_style()
    n = len(images)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for idx in range(n):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        img_tuple = images[idx]

        if len(img_tuple) == 3:
            original, heatmap, overlay = img_tuple
            ax.imshow(overlay)
        else:
            original, heatmap = img_tuple
            ax.imshow(heatmap)

        pred_text = f"Pred: {predictions[idx]}"
        if ground_truths:
            pred_text += f"\nTrue: {ground_truths[idx]}"
        ax.set_title(pred_text, fontsize=9)
        ax.axis("off")

    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, save_path)
    return fig


def plot_uncertainty_analysis(
    uncertainties, is_correct, save_path=None, title="Uncertainty vs. Prediction Correctness"
):
    """Analyze how uncertainty signals correlate with prediction errors.

    Args:
        uncertainties: (n_samples,) array of uncertainty values
        is_correct: (n_samples,) boolean array
        save_path: optional save path
        title: plot title

    """
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    correct_unc = uncertainties[is_correct]
    incorrect_unc = uncertainties[~is_correct]

    ax1.hist(
        correct_unc,
        bins=30,
        alpha=0.7,
        color="#4CAF50",
        label=f"Correct (n={len(correct_unc)})",
        density=True,
        edgecolor="white",
    )
    ax1.hist(
        incorrect_unc,
        bins=30,
        alpha=0.7,
        color="#F44336",
        label=f"Incorrect (n={len(incorrect_unc)})",
        density=True,
        edgecolor="white",
    )
    ax1.set(xlabel="Uncertainty (Entropy)", ylabel="Density", title="Uncertainty Distribution by Correctness")
    ax1.legend(frameon=True)

    sorted_idx = np.argsort(uncertainties)
    sorted_unc = uncertainties[sorted_idx]
    sorted_correct = is_correct[sorted_idx].astype(float)

    window = max(1, len(sorted_unc) // 20)
    smoothed_acc = np.convolve(sorted_correct, np.ones(window) / window, mode="valid")
    smoothed_unc = np.convolve(sorted_unc, np.ones(window) / window, mode="valid")

    ax2.plot(smoothed_unc, smoothed_acc, "o-", color="#2196F3", lw=2, markersize=4)
    ax2.set(xlabel="Uncertainty (smoothed)", ylabel="Accuracy (smoothed)", title="Accuracy vs. Uncertainty")
    ax2.set_ylim([0, 1.05])
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, save_path)
    return fig
