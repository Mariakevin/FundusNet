"""Uncertainty calibration study for RetinaAI.

Evaluates whether MC Dropout uncertainty correlates with prediction errors,
generates reliability diagrams, accuracy-refusal tradeoff curves, and
compares different uncertainty signals.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "retina_project.settings")

import django

django.setup()

from evaluation.evaluate import evaluate_ensemble, load_dataset
from evaluation.metrics import (
    compute_error_detection_auroc,
    compute_reliability_data,
)
from retina_app.constants import (
    MC_DROPOUT_PASSES,
    MODEL_LIST,
    MODEL_WEIGHTS,
)
from retina_app.services.ensemble import (
    _predict_single_model,
    detect_model_disagreement,
)
from retina_app.services.model_manager import get_model_manager
from retina_app.services.uncertainty import (
    mc_dropout_single_model,
)


def collect_uncertainty_data(models, file_paths, labels, model_weights, n_passes=10):
    """Run inference and collect all uncertainty signals per sample.

    Returns:
        dict with arrays: predictions, labels, softmax_confidence,
        mc_entropy, ensemble_disagreement, probs

    """
    results = {
        "predictions": [],
        "labels": [],
        "softmax_confidence": [],
        "mc_entropy": [],
        "ensemble_disagreement": [],
        "probs": [],
        "is_correct": [],
    }

    for i, (path, label) in enumerate(zip(file_paths, labels)):
        if i % 20 == 0:
            print(f"  Processing {i}/{len(file_paths)}...")

        # Get ensemble prediction
        ensemble_result = evaluate_ensemble(
            models,
            path,
            model_weights=model_weights,
            use_tta=False,
            use_selective=False,
        )

        if ensemble_result is None:
            continue

        probs = np.array(ensemble_result["probabilities"])
        pred = ensemble_result["label"]

        # Softmax confidence (max probability)
        softmax_conf = float(np.max(probs))

        # MC Dropout entropy (use first available model)
        mc_ent = 0.0
        for name in model_weights:
            if name in models:
                try:
                    mc_result = mc_dropout_single_model(models[name], path, n_passes=n_passes)
                    if mc_result is not None:
                        mc_ent = mc_result.get("entropy", 0.0)
                        break
                except Exception:
                    continue

        # Ensemble disagreement
        individual_preds = []
        for name in model_weights:
            if name in models:
                try:
                    pred_i = _predict_single_model(models[name], path, use_tta=False)
                    individual_preds.append(pred_i)
                except Exception:
                    continue

        if len(individual_preds) >= 2:
            agreement_info = detect_model_disagreement(individual_preds)
            disagreement = 1.0 - agreement_info["agreement_level"]
        else:
            disagreement = 0.0

        results["predictions"].append(pred)
        results["labels"].append(label)
        results["softmax_confidence"].append(softmax_conf)
        results["mc_entropy"].append(mc_ent)
        results["ensemble_disagreement"].append(disagreement)
        results["probs"].append(probs.tolist())
        results["is_correct"].append(pred == label)

    for key in results:
        if key == "probs":
            results[key] = np.array(results[key])
        else:
            results[key] = np.array(results[key])

    return results


def evaluate_uncertainty_signals(data):
    """Evaluate each uncertainty signal for error detection.

    Returns:
        dict with AUROC per signal and reliability data

    """
    is_correct = data["is_correct"]
    n_correct = int(np.sum(is_correct))
    n_total = len(is_correct)

    signals = {
        "softmax_confidence": 1 - data["softmax_confidence"],  # invert so higher = more uncertain
        "mc_entropy": data["mc_entropy"],
        "ensemble_disagreement": data["ensemble_disagreement"],
    }

    results = {
        "n_samples": n_total,
        "n_correct": n_correct,
        "accuracy": n_correct / n_total if n_total > 0 else 0,
        "signal_auroc": {},
        "reliability": {},
        "threshold_sweep": {},
    }

    for signal_name, signal_values in signals.items():
        if np.std(signal_values) > 0:
            auroc = compute_error_detection_auroc(signal_values, is_correct)
        else:
            auroc = 0.5
        results["signal_auroc"][signal_name] = float(auroc)

    # Reliability diagram data
    probs = data["probs"]
    labels = data["labels"]
    if len(probs) > 0:
        results["reliability"] = compute_reliability_data(probs, labels, n_bins=10)

    # Threshold sweep for MC Dropout entropy
    if np.std(data["mc_entropy"]) > 0:
        results["threshold_sweep"]["mc_entropy"] = sweep_thresholds(
            data["mc_entropy"],
            is_correct,
            data["predictions"],
            data["labels"],
            lower_is_uncertain=False,
        )

    if np.std(data["ensemble_disagreement"]) > 0:
        results["threshold_sweep"]["ensemble_disagreement"] = sweep_thresholds(
            data["ensemble_disagreement"],
            is_correct,
            data["predictions"],
            data["labels"],
            lower_is_uncertain=False,
        )

    return results


def sweep_thresholds(signal_values, is_correct, predictions, labels, lower_is_uncertain=False, n_steps=30):
    """Sweep uncertainty thresholds and compute accuracy-refusal tradeoff.

    Returns:
        list of {threshold, accuracy, refusal_rate, n_kept}

    """
    min_val = float(np.min(signal_values))
    max_val = float(np.max(signal_values))

    if min_val >= max_val:
        return []

    thresholds = np.linspace(min_val, max_val, n_steps)
    results = []

    for threshold in thresholds:
        if lower_is_uncertain:
            keep_mask = signal_values >= threshold
        else:
            keep_mask = signal_values <= threshold

        n_kept = int(np.sum(keep_mask))
        refusal_rate = 1.0 - n_kept / len(signal_values) if len(signal_values) > 0 else 0

        if n_kept > 0:
            acc = float(np.mean(predictions[keep_mask] == labels[keep_mask]))
        else:
            acc = 0.0

        results.append(
            {
                "threshold": float(threshold),
                "accuracy": acc,
                "refusal_rate": refusal_rate,
                "n_kept": n_kept,
            }
        )

    return results


def find_optimal_threshold(sweep_data, min_refusal=0.0, max_refusal=0.5):
    """Find threshold that maximizes accuracy within refusal constraints.

    Returns:
        dict with optimal threshold, accuracy, refusal_rate

    """
    valid = [d for d in sweep_data if min_refusal <= d["refusal_rate"] <= max_refusal and d["n_kept"] > 0]

    if not valid:
        return None

    best = max(valid, key=lambda x: x["accuracy"])
    return best


def run_uncertainty_study(dataset_dir, n_folds=5, model_list=None, seed=42, output_dir=None):
    """Run full uncertainty calibration study."""
    file_paths, labels, class_names = load_dataset(dataset_dir)

    if model_list is None:
        model_list = MODEL_LIST

    model_weights = {m: MODEL_WEIGHTS.get(m, 1.0 / len(model_list)) for m in model_list}

    manager = get_model_manager()
    models = {}
    for name in model_list:
        try:
            model = manager.get_model(name)
            if model is not None:
                models[name] = model
        except Exception:
            continue

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_results = []

    for fold_idx, (_, val_idx) in enumerate(skf.split(file_paths, labels)):
        print(f"\nUncertainty Study — Fold {fold_idx + 1}/{n_folds}")

        val_paths = [file_paths[i] for i in val_idx]
        val_labels = labels[val_idx]

        data = collect_uncertainty_data(models, val_paths, val_labels, model_weights, n_passes=MC_DROPOUT_PASSES)

        if len(data["predictions"]) < 10:
            print("  Too few valid predictions, skipping fold")
            continue

        eval_result = evaluate_uncertainty_signals(data)
        fold_results.append(eval_result)

        print(f"  Accuracy: {eval_result['accuracy']:.4f}")
        for signal, auroc in eval_result["signal_auroc"].items():
            print(f"  AUROC ({signal}): {auroc:.4f}")

    # Aggregate across folds
    summary = aggregate_uncertainty_results(fold_results)

    results = {"per_fold": fold_results, "summary": summary}

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "uncertainty_study.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)

    return results


def aggregate_uncertainty_results(fold_results):
    """Aggregate uncertainty study results across folds."""
    if not fold_results:
        return {}

    summary = {"signal_auroc": {}, "optimal_thresholds": {}}

    # Aggregate AUROC per signal
    all_signals = set()
    for fr in fold_results:
        all_signals.update(fr.get("signal_auroc", {}).keys())

    for signal in all_signals:
        values = [fr["signal_auroc"][signal] for fr in fold_results if signal in fr.get("signal_auroc", {})]
        if values:
            summary["signal_auroc"][signal] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }

    # Find optimal thresholds across folds
    for signal in ["mc_entropy", "ensemble_disagreement"]:
        all_sweep = []
        for fr in fold_results:
            sweep = fr.get("threshold_sweep", {}).get(signal, [])
            all_sweep.extend(sweep)

        if all_sweep:
            optimal = find_optimal_threshold(all_sweep)
            if optimal:
                summary["optimal_thresholds"][signal] = optimal

    return summary


def print_uncertainty_summary(results):
    """Print formatted uncertainty study summary."""
    s = results.get("summary", {})

    print("\n" + "=" * 70)
    print("UNCERTAINTY CALIBRATION STUDY")
    print("=" * 70)

    print("\nError Detection AUROC (higher = better uncertainty signal):")
    for signal, metrics in s.get("signal_auroc", {}).items():
        print(f"  {signal}: {metrics['mean']:.4f} ± {metrics['std']:.4f}")

    print("\nOptimal Thresholds:")
    for signal, opt in s.get("optimal_thresholds", {}).items():
        print(
            f"  {signal}: threshold={opt['threshold']:.4f}, "
            f"accuracy={opt['accuracy']:.4f}, "
            f"refusal_rate={opt['refusal_rate']:.2%}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="retina_dataset")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", default="evaluation_results")
    args = parser.parse_args()

    results = run_uncertainty_study(args.dataset, args.folds, output_dir=args.output)
    if results:
        print_uncertainty_summary(results)
