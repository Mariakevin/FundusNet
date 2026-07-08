"""Main evaluation script for RetinaAI.

Runs 5-fold stratified cross-validation, computes metrics across folds,
and generates summary statistics with standard deviations and bootstrap CIs.

Usage:
    python -m retina_app.evaluation.evaluate [--models MODEL1,MODEL2] [--folds N]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    cohen_kappa_score,
    matthews_corrcoef,
)
from sklearn.metrics import (
    confusion_matrix as sk_confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "retina_project.settings")

import django

django.setup()

from evaluation.metrics import (
    compute_auroc_per_class,
    compute_brier,
    compute_ece,
    compute_mce,
    confusion_matrix,
    overall_metrics,
    per_class_metrics,
)
from evaluation.statistics import bootstrap_confidence_interval
from retina_app.constants import (
    CATEGORIES,
    MODEL_LIST,
    MODEL_WEIGHTS,
    UNCERTAINTY_THRESHOLD,
)
from retina_app.services.ensemble import (
    _predict_single_model,
    detect_model_disagreement,
    selective_ensemble,
)
from retina_app.services.model_manager import get_model_manager
from retina_app.services.uncertainty import (
    compute_prediction_entropy,
)


def load_dataset(dataset_dir):
    """Load dataset file paths and labels.

    Returns:
        file_paths: list of str (absolute paths)
        labels: list of int (class indices)
        class_names: list of str

    """
    file_paths = []
    labels = []
    class_names = sorted(os.listdir(dataset_dir))

    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                file_paths.append(os.path.join(class_dir, fname))
                labels.append(class_idx)

    return file_paths, np.array(labels), class_names


def evaluate_single_model(models, image_path, model_name, use_tta=False):
    """Run a single model and return prediction dict."""
    if model_name not in models:
        return None
    model = models[model_name]
    try:
        pred = _predict_single_model(model, image_path, use_tta=use_tta)
        return pred
    except Exception:
        return None


def evaluate_ensemble(models, image_path, model_weights=None, use_tta=False, use_selective=True, use_mc_dropout=False):
    """Run ensemble evaluation on a single image.

    Returns:
        dict with label, confidence, probabilities, uncertainty, agreement info

    """
    active_models = {k: v for k, v in models.items() if k in model_weights}
    if not active_models:
        return None

    individual_preds = {}

    for name in active_models:
        try:
            pred = _predict_single_model(active_models[name], image_path, use_tta=use_tta)
            individual_preds[name] = pred
        except Exception:
            continue

    if len(individual_preds) < 2:
        return None

    pred_items = list(individual_preds.items())
    agreement_info = detect_model_disagreement(pred_items)

    if use_selective and agreement_info["agreement_level"] < 0.5:
        result = selective_ensemble(pred_items, min_agreement=0.5)
        if result:
            return {
                "label": result["label"],
                "label_name": result["label"],
                "confidence": result["confidence"],
                "probabilities": result["probabilities"],
                "uncertainty": result.get("uncertainty", 0.0),
                "is_uncertain": result.get("uncertainty", 0.0) > UNCERTAINTY_THRESHOLD,
                "agreement_level": agreement_info["agreement_level"],
                "n_models_used": result.get("filtered_n_models", len(pred_items)),
            }

    probs = np.zeros(len(CATEGORIES))
    weights_used = []
    for pred_name, pred in pred_items:
        if pred_name in model_weights:
            w = model_weights[pred_name]
            probs += w * np.array(pred["probabilities"])
            weights_used.append(pred_name)

    probs = probs / max(probs.sum(), 1e-8)
    label = int(np.argmax(probs))
    confidence = float(probs[label])

    entropy = compute_prediction_entropy(probs)
    is_uncertain = entropy > UNCERTAINTY_THRESHOLD

    return {
        "label": label,
        "label_name": CATEGORIES[label],
        "confidence": confidence,
        "probabilities": probs.tolist(),
        "uncertainty": entropy,
        "is_uncertain": is_uncertain,
        "agreement_level": agreement_info["agreement_level"],
        "n_models_used": len(weights_used),
    }


def run_cv_evaluation(dataset_dir, n_folds=5, model_list=None, use_tta=False, seed=42, output_dir=None):
    """Run full 5-fold stratified CV evaluation.

    Returns:
        dict with per-fold results and aggregated statistics

    """
    file_paths, labels, class_names = load_dataset(dataset_dir)
    n_samples = len(file_paths)
    n_classes = len(class_names)

    print(f"Loaded {n_samples} images across {n_classes} classes: {class_names}")
    print(f"Class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    if model_list is None:
        model_list = MODEL_LIST

    model_weights = {m: MODEL_WEIGHTS.get(m, 1.0 / len(model_list)) for m in model_list}
    print(f"Using models: {model_list}")

    # Load models
    manager = get_model_manager()
    models = {}
    for name in model_list:
        try:
            model = manager.get_model(name)
            if model is not None:
                models[name] = model
                print(f"  Loaded {name}")
            else:
                print(f"  WARNING: Could not load {name}")
        except Exception as e:
            print(f"  WARNING: Failed to load {name}: {e}")

    if len(models) < 2:
        print("ERROR: Need at least 2 models for evaluation")
        return None

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(file_paths, labels)):
        print(f"\n{'=' * 60}")
        print(f"Fold {fold_idx + 1}/{n_folds}")
        print(f"  Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")

        val_paths = [file_paths[i] for i in val_idx]
        val_labels = labels[val_idx]

        preds = []
        probs_list = []
        latencies = []

        for path in val_paths:
            t0 = time.time()
            result = evaluate_ensemble(
                models,
                path,
                model_weights=model_weights,
                use_tta=use_tta,
                use_selective=True,
            )
            elapsed = time.time() - t0
            latencies.append(elapsed)

            if result is not None:
                preds.append(result["label"])
                probs_list.append(result["probabilities"])
            else:
                preds.append(0)
                probs_list.append([1.0 / n_classes] * n_classes)

        preds = np.array(preds)
        probs = np.array(probs_list)

        # Compute fold metrics
        mcc = matthews_corrcoef(val_labels, preds)
        kappa = cohen_kappa_score(val_labels, preds)

        # Specificity (true negative rate) per class, averaged
        cm = sk_confusion_matrix(val_labels, preds, labels=list(range(n_classes)))
        specificities = []
        for i in range(n_classes):
            tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
            fp = cm[:, i].sum() - cm[i, i]
            specificities.append(tn / max(tn + fp, 1))
        macro_specificity = float(np.mean(specificities))

        fold_metrics = {
            "accuracy": float(np.mean(preds == val_labels)),
            "overall": overall_metrics(preds, val_labels, class_names),
            "per_class": per_class_metrics(preds, val_labels, class_names),
            "mcc": float(mcc),
            "kappa": float(kappa),
            "macro_specificity": macro_specificity,
            "per_class_specificity": {class_names[i]: float(specificities[i]) for i in range(n_classes)},
            "ece": compute_ece(probs, val_labels),
            "mce": compute_mce(probs, val_labels),
            "brier": compute_brier(probs, val_labels),
            "confusion_matrix": confusion_matrix(preds, val_labels, n_classes).tolist(),
            "auroc_per_class": compute_auroc_per_class(probs, val_labels, n_classes),
            "mean_latency_ms": float(np.mean(latencies) * 1000),
            "std_latency_ms": float(np.std(latencies) * 1000),
            "n_samples": len(val_labels),
        }

        fold_results.append(fold_metrics)
        print(f"  Accuracy: {fold_metrics['accuracy']:.4f}")
        print(f"  Macro F1: {fold_metrics['overall']['macro_f1']:.4f}")
        print(f"  ECE: {fold_metrics['ece']:.4f}")
        print(f"  Mean latency: {fold_metrics['mean_latency_ms']:.1f}ms")

    # Aggregate across folds
    aggregated = aggregate_fold_results(fold_results, n_classes, class_names)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "evaluation_results.json")
        with open(output_path, "w") as f:
            json.dump(aggregated, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return aggregated


def aggregate_fold_results(fold_results, n_classes, class_names):
    """Aggregate metrics across folds with mean ± std and bootstrap CIs."""
    aggregated = {"per_fold": fold_results, "summary": {}}

    metric_keys = ["accuracy", "ece", "mce", "brier", "mean_latency_ms", "mcc", "kappa", "macro_specificity"]
    for key in metric_keys:
        values = [f[key] for f in fold_results]
        mean_val = float(np.mean(values))
        std_val = float(np.std(values))
        # Bootstrap 95% CI
        ci_result = bootstrap_confidence_interval(np.array(values), ci_level=0.95, n_bootstrap=1000, seed=42)
        aggregated["summary"][key] = {
            "mean": mean_val,
            "std": std_val,
            "ci_95": [ci_result["ci_lower"], ci_result["ci_upper"]],
        }

    # Overall metrics
    overall_keys = ["accuracy", "macro_f1", "weighted_f1"]
    for key in overall_keys:
        values = [f["overall"][key] for f in fold_results]
        aggregated["summary"][f"overall_{key}"] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

    # Per-class metrics
    aggregated["summary"]["per_class"] = {}
    for cat in class_names:
        cat_metrics = {}
        for metric_name in ["precision", "recall", "f1"]:
            values = [f["per_class"][cat][metric_name] for f in fold_results]
            cat_metrics[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
        aggregated["summary"]["per_class"][cat] = cat_metrics

    # AUROC per class
    auroc_per_class = np.array([f["auroc_per_class"] for f in fold_results])
    aggregated["summary"]["auroc_per_class"] = {}
    for i, cat in enumerate(class_names):
        aggregated["summary"]["auroc_per_class"][cat] = {
            "mean": float(np.mean(auroc_per_class[:, i])),
            "std": float(np.std(auroc_per_class[:, i])),
        }

    # Confusion matrix (mean across folds)
    cms = np.array([f["confusion_matrix"] for f in fold_results])
    aggregated["summary"]["mean_confusion_matrix"] = np.mean(cms, axis=0).tolist()

    return aggregated


def print_summary(aggregated):
    """Print formatted summary of evaluation results."""
    s = aggregated["summary"]

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY (5-fold Stratified CV)")
    print("=" * 70)

    for key in ["overall_accuracy", "overall_macro_f1", "overall_weighted_f1"]:
        val = s[key]
        display_key = key.replace("overall_", "").replace("_", " ").title()
        print(f"  {display_key}: {val['mean']:.4f} ± {val['std']:.4f}")

    print("\nCalibration Metrics:")
    for key in ["ece", "mce", "brier"]:
        val = s[key]
        print(
            f"  {key.upper()}: {val['mean']:.4f} ± {val['std']:.4f} [95% CI: {val['ci_95'][0]:.4f}, {val['ci_95'][1]:.4f}]"
        )

    print("\nClassification Quality:")
    for key in ["mcc", "kappa", "macro_specificity"]:
        val = s[key]
        display_key = {"mcc": "MCC", "kappa": "Cohen's Kappa", "macro_specificity": "Macro Specificity"}[key]
        print(
            f"  {display_key}: {val['mean']:.4f} ± {val['std']:.4f} [95% CI: {val['ci_95'][0]:.4f}, {val['ci_95'][1]:.4f}]"
        )

    print("\nPer-Class F1:")
    for cat, metrics in s["per_class"].items():
        print(f"  {cat}: {metrics['f1']['mean']:.4f} ± {metrics['f1']['std']:.4f}")

    print("\nPer-Class AUROC:")
    for cat, metrics in s["auroc_per_class"].items():
        print(f"  {cat}: {metrics['mean']:.4f} ± {metrics['std']:.4f}")

    print(f"\nMean Latency: {s['mean_latency_ms']['mean']:.1f} ± {s['mean_latency_ms']['std']:.1f}ms")


def main():
    parser = argparse.ArgumentParser(description="RetinaAI Evaluation")
    parser.add_argument("--dataset", default="retina_dataset", help="Path to dataset directory")
    parser.add_argument("--models", default=None, help="Comma-separated model names")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--tta", action="store_true", help="Use test-time augmentation")
    parser.add_argument("--output", default="evaluation_results", help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_list = args.models.split(",") if args.models else None

    results = run_cv_evaluation(
        dataset_dir=args.dataset,
        n_folds=args.folds,
        model_list=model_list,
        use_tta=args.tta,
        seed=args.seed,
        output_dir=args.output,
    )

    if results:
        print_summary(results)


if __name__ == "__main__":
    main()
