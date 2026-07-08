"""Selective ensemble formalization and evaluation.

Transforms the heuristic selective ensemble into a principled method by:
1. Computing continuous agreement scores (not binary)
2. Sweeping agreement thresholds on validation data
3. Finding Pareto-optimal operating points
4. Reporting accuracy-refusal tradeoff curves
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "retina_project.settings")

import django
django.setup()

from evaluation.metrics import overall_metrics, per_class_metrics
from evaluation.evaluate import load_dataset, evaluate_ensemble
from retina_app.constants import CATEGORIES, MODEL_LIST, MODEL_WEIGHTS
from retina_app.services.model_manager import get_model_manager
from retina_app.services.ensemble import (
    _predict_single_model,
    detect_model_disagreement,
    ensemble_predictions,
    selective_ensemble,
)


def compute_agreement_scores(models, file_paths, model_weights):
    """Compute per-sample agreement scores from model predictions.

    Returns:
        list of dicts with: ensemble_pred, individual_preds, agreement_score, n_models
    """
    results = []

    for path in file_paths:
        individual_preds = {}
        for name in model_weights:
            if name in models:
                try:
                    pred = _predict_single_model(models[name], path, use_tta=False)
                    individual_preds[name] = pred
                except Exception:
                    continue

        if len(individual_preds) < 2:
            continue

        pred_list = list(individual_preds.values())
        agreement_info = detect_model_disagreement(pred_list)

        # Compute continuous agreement score
        votes = [p["label"] for p in pred_list]
        if len(votes) > 0:
            majority_label = max(set(votes), key=votes.count)
            agreement_score = votes.count(majority_label) / len(votes)
        else:
            agreement_score = 0.0

        # Ensemble prediction
        ensemble_pred = ensemble_predictions(pred_list)

        results.append({
            "path": path,
            "ensemble_label": ensemble_pred["label"],
            "ensemble_confidence": ensemble_pred["confidence"],
            "ensemble_probabilities": ensemble_pred["probabilities"],
            "agreement_score": agreement_score,
            "n_models": len(individual_preds),
            "individual_labels": [p["label"] for p in pred_list],
        })

    return results


def sweep_selective_ensemble(results, true_labels, thresholds=None):
    """Sweep agreement thresholds for selective ensemble.

    Returns:
        list of {threshold, accuracy, refusal_rate, n_kept, macro_f1}
    """
    if thresholds is None:
        thresholds = np.arange(0.3, 1.05, 0.05)

    ensemble_labels = np.array([r["ensemble_label"] for r in results])
    agreement_scores = np.array([r["agreement_score"] for r in results])
    true_labels = np.array(true_labels[:len(results)])

    sweep_results = []

    for threshold in thresholds:
        keep_mask = agreement_scores >= threshold
        n_kept = int(np.sum(keep_mask))
        refusal_rate = 1.0 - n_kept / len(results) if len(results) > 0 else 0

        if n_kept > 0:
            kept_preds = ensemble_labels[keep_mask]
            kept_labels = true_labels[keep_mask]
            acc = float(np.mean(kept_preds == kept_labels))
            metrics = overall_metrics(kept_preds, kept_labels, CATEGORIES)
            macro_f1 = metrics["macro_f1"]
        else:
            acc = 0.0
            macro_f1 = 0.0

        sweep_results.append({
            "threshold": float(threshold),
            "accuracy": acc,
            "macro_f1": macro_f1,
            "refusal_rate": refusal_rate,
            "n_kept": n_kept,
        })

    return sweep_results


def find_pareto_optimal(sweep_results):
    """Find Pareto-optimal points on accuracy-refusal curve.

    A point is Pareto-optimal if no other point has both higher accuracy
    AND lower refusal rate.

    Returns:
        list of Pareto-optimal sweep results
    """
    pareto = []
    for i, point in enumerate(sweep_results):
        dominated = False
        for j, other in enumerate(sweep_results):
            if i != j:
                if (other["accuracy"] >= point["accuracy"] and
                    other["refusal_rate"] <= point["refusal_rate"] and
                    (other["accuracy"] > point["accuracy"] or
                     other["refusal_rate"] < point["refusal_rate"])):
                    dominated = True
                    break
        if not dominated:
            pareto.append(point)

    return pareto


def run_ensemble_evaluation(dataset_dir, n_folds=5, model_list=None, seed=42,
                             output_dir=None):
    """Run selective ensemble formalization study."""
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
        print(f"\nEnsemble Evaluation — Fold {fold_idx + 1}/{n_folds}")

        val_paths = [file_paths[i] for i in val_idx]
        val_labels = labels[val_idx]

        # Compute agreement scores
        results = compute_agreement_scores(models, val_paths, model_weights)
        if not results:
            continue

        # Sweep thresholds
        sweep = sweep_selective_ensemble(results, val_labels)

        # Find Pareto optimal
        pareto = find_pareto_optimal(sweep)

        # Find best by accuracy at < 30% refusal
        low_refusal = [s for s in sweep if s["refusal_rate"] <= 0.3]
        best_low_refusal = max(low_refusal, key=lambda x: x["accuracy"]) if low_refusal else None

        fold_result = {
            "sweep": sweep,
            "pareto": pareto,
            "best_low_refusal": best_low_refusal,
            "n_samples": len(val_labels),
            "mean_agreement": float(np.mean([r["agreement_score"] for r in results])),
        }

        fold_results.append(fold_result)

        if best_low_refusal:
            print(f"  Best @ <30% refusal: threshold={best_low_refusal['threshold']:.2f}, "
                  f"acc={best_low_refusal['accuracy']:.4f}, "
                  f"refusal={best_low_refusal['refusal_rate']:.2%}")

    # Aggregate
    summary = aggregate_ensemble_results(fold_results)
    results_out = {"per_fold": fold_results, "summary": summary}

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "ensemble_eval.json"), "w") as f:
            json.dump(results_out, f, indent=2, default=str)

    return results_out


def aggregate_ensemble_results(fold_results):
    """Aggregate ensemble evaluation across folds."""
    if not fold_results:
        return {}

    # Aggregate sweep curves
    all_thresholds = set()
    for fr in fold_results:
        for s in fr["sweep"]:
            all_thresholds.add(round(s["threshold"], 2))

    threshold_agg = {}
    for t in sorted(all_thresholds):
        accs = []
        refusals = []
        for fr in fold_results:
            for s in fr["sweep"]:
                if abs(s["threshold"] - t) < 0.01:
                    accs.append(s["accuracy"])
                    refusals.append(s["refusal_rate"])
        if accs:
            threshold_agg[str(t)] = {
                "accuracy_mean": float(np.mean(accs)),
                "accuracy_std": float(np.std(accs)),
                "refusal_mean": float(np.mean(refusals)),
                "refusal_std": float(np.std(refusals)),
            }

    # Mean agreement across folds
    mean_agreement = float(np.mean([fr["mean_agreement"] for fr in fold_results]))

    return {
        "threshold_sweep": threshold_agg,
        "mean_agreement": mean_agreement,
    }


def print_ensemble_summary(results):
    """Print ensemble evaluation summary."""
    s = results.get("summary", {})

    print("\n" + "=" * 70)
    print("SELECTIVE ENSEMBLE EVALUATION")
    print("=" * 70)
    print(f"Mean agreement across models: {s.get('mean_agreement', 0):.4f}")

    print("\nAccuracy-Refusal Tradeoff:")
    print(f"  {'Threshold':>10} {'Accuracy':>12} {'Refusal%':>12}")
    print("  " + "-" * 36)

    for t, metrics in sorted(s.get("threshold_sweep", {}).items()):
        print(f"  {float(t):>10.2f} "
              f"{metrics['accuracy_mean']:.4f}±{metrics['accuracy_std']:.4f} "
              f"{metrics['refusal_mean']:.2%}±{metrics['refusal_std']:.2%}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="retina_dataset")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", default="evaluation_results")
    args = parser.parse_args()

    results = run_ensemble_evaluation(args.dataset, args.folds, output_dir=args.output)
    if results:
        print_ensemble_summary(results)
