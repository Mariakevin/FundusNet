"""Dynamic weight learning for ensemble models.

Replaces hand-tuned CLASS_PERFORMANCE_WEIGHTS with optimization-based
per-class per-model weights learned from validation data.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "retina_project.settings")

import django

django.setup()

from evaluation.evaluate import load_dataset
from evaluation.metrics import overall_metrics
from retina_app.constants import CATEGORIES, MODEL_LIST
from retina_app.services.ensemble import _predict_single_model
from retina_app.services.model_manager import get_model_manager


def collect_model_predictions(models, file_paths, model_names):
    """Collect per-model prediction probabilities for each sample.

    Returns:
        preds_per_model: dict {model_name: (n_samples, n_classes) array}
        valid_mask: boolean array of samples where all models succeeded

    """
    all_preds = {name: [] for name in model_names}
    valid_indices = []

    for i, path in enumerate(file_paths):
        all_succeeded = True
        for name in model_names:
            if name not in models:
                all_succeeded = False
                break
            try:
                pred = _predict_single_model(models[name], path, use_tta=False)
                all_preds[name].append(pred["probabilities"])
            except Exception:
                all_succeeded = False
                break

        if all_succeeded:
            valid_indices.append(i)

    for name in model_names:
        if all_preds[name]:
            all_preds[name] = np.array(all_preds[name])
        else:
            all_preds[name] = np.array([])

    return all_preds, valid_indices


def learn_class_weights(preds_per_model, labels, model_names, n_classes):
    """Learn optimal per-class model weights via convex optimization.

    For each class c, solves:
        max_w  sum_i log(sum_m w_m * p_m,c(x_i))
        s.t.   sum_m w_m = 1, w_m >= 0

    This maximizes the log-likelihood of correct class probabilities.

    Args:
        preds_per_model: dict {model_name: (n_samples, n_classes)}
        labels: (n_samples,) true class indices
        model_names: list of model names
        n_classes: number of classes

    Returns:
        dict {model_name: weight} per-class weights, or None if optimization fails

    """
    n_models = len(model_names)
    n_samples = len(labels)

    if n_models == 0 or n_samples == 0:
        return None

    # Stack predictions: (n_models, n_samples, n_classes)
    model_preds = np.array([preds_per_model[name] for name in model_names])

    learned_weights = {}

    for c in range(n_classes):
        # Binary labels: is this sample class c?
        binary_labels = (labels == c).astype(float)
        n_pos = np.sum(binary_labels)

        if n_pos == 0:
            # No samples for this class, use equal weights
            learned_weights[c] = {name: 1.0 / n_models for name in model_names}
            continue

        # For class c, get each model's probability of class c
        # model_probs[m, i] = model m's probability of class c for sample i
        model_probs_c = model_preds[:, :, c]  # (n_models, n_samples)

        def neg_log_likelihood(w):
            """Negative log-likelihood to minimize."""
            # w is (n_models,), normalize
            w_norm = w / max(w.sum(), 1e-8)
            # For each sample, weighted sum of model probs for class c
            weighted_sum = w_norm @ model_probs_c  # (n_samples,)
            # Only optimize on positive samples (class c)
            weighted_sum_pos = weighted_sum[labels == c]
            # Avoid log(0)
            eps = 1e-8
            nll = -np.sum(np.log(weighted_sum_pos + eps))
            return nll

        # Constraints: weights sum to 1
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        # Bounds: each weight in [0, 1]
        bounds = [(0.0, 1.0)] * n_models
        # Initial guess: equal weights
        w0 = np.ones(n_models) / n_models

        try:
            result = minimize(
                neg_log_likelihood,
                w0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 100, "ftol": 1e-10},
            )

            if result.success:
                w_opt = result.x / result.x.sum()
                learned_weights[c] = {name: float(w_opt[i]) for i, name in enumerate(model_names)}
            else:
                learned_weights[c] = {name: 1.0 / n_models for name in model_names}
        except Exception:
            learned_weights[c] = {name: 1.0 / n_models for name in model_names}

    return learned_weights


def learn_overall_weights(preds_per_model, labels, model_names):
    """Learn a single set of weights (not per-class) via optimization.

    Solves:
        max_w  sum_i log(sum_m w_m * p_m,y_i(x_i))
        s.t.   sum_m w_m = 1, w_m >= 0

    Returns:
        dict {model_name: weight}

    """
    n_models = len(model_names)
    model_preds = np.array([preds_per_model[name] for name in model_names])

    def neg_log_likelihood(w):
        w_norm = w / max(w.sum(), 1e-8)
        # For each sample, weighted sum of probs for true class
        total = 0.0
        for i, label in enumerate(labels):
            weighted_prob = sum(w_norm[m] * model_preds[m, i, label] for m in range(n_models))
            total += np.log(weighted_prob + 1e-8)
        return -total

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, 1.0)] * n_models
    w0 = np.ones(n_models) / n_models

    try:
        result = minimize(
            neg_log_likelihood,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-12},
        )
        if result.success:
            w_opt = result.x / result.x.sum()
            return {name: float(w_opt[i]) for i, name in enumerate(model_names)}
    except Exception:
        pass

    return {name: 1.0 / n_models for name in model_names}


def apply_learned_weights(predictions, learned_weights_per_class, model_names):
    """Apply learned per-class weights to prediction list.

    Args:
        predictions: list of prediction dicts from individual models
        learned_weights_per_class: dict {class_idx: {model_name: weight}}
        model_names: list of model names matching predictions order

    Returns:
        dict with label, confidence, probabilities

    """
    probs = np.zeros(len(CATEGORIES))

    for pred, name in zip(predictions, model_names):
        if name in learned_weights_per_class.get(pred["label"], {}):
            w = learned_weights_per_class[pred["label"]][name]
        else:
            w = 1.0 / len(model_names)
        probs += w * np.array(pred["probabilities"])

    probs = probs / max(probs.sum(), 1e-8)
    label = int(np.argmax(probs))

    return {
        "label": label,
        "confidence": float(probs[label]),
        "probabilities": probs.tolist(),
    }


def run_weight_learning(dataset_dir, n_folds=5, model_list=None, seed=42, output_dir=None):
    """Run weight learning study: compare hand-tuned vs learned weights."""
    file_paths, labels, class_names = load_dataset(dataset_dir)

    if model_list is None:
        model_list = MODEL_LIST

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

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(file_paths, labels)):
        print(f"\nWeight Learning — Fold {fold_idx + 1}/{n_folds}")

        train_paths = [file_paths[i] for i in train_idx]
        val_paths = [file_paths[i] for i in val_idx]
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]

        # Collect predictions on training set
        train_preds, train_valid = collect_model_predictions(models, train_paths, model_list)
        train_labels_valid = train_labels[train_valid]

        if len(train_labels_valid) < 10:
            print("  Too few valid training samples, skipping")
            continue

        # Learn weights on training set
        learned_class_weights = learn_class_weights(train_preds, train_labels_valid, model_list, len(CATEGORIES))
        learned_overall = learn_overall_weights(train_preds, train_labels_valid, model_list)

        # Evaluate on validation set
        val_preds_raw = []
        val_labels_list = []
        for path in val_paths:
            preds = {}
            for name in model_list:
                if name in models:
                    try:
                        pred = _predict_single_model(models[name], path, use_tta=False)
                        preds[name] = pred
                    except Exception:
                        continue
            if len(preds) == len(model_list):
                val_preds_raw.append(preds)
                val_labels_list.append(0)  # placeholder

        # Evaluate three strategies
        strategies = {
            "hand_tuned": {},
            "learned_overall": learned_overall,
            "learned_per_class": None,  # handled separately
        }

        strategy_results = {}
        for strat_name, weights in strategies.items():
            preds = []
            for vp in val_preds_raw:
                pred_list = list(vp.values())
                if strat_name == "learned_per_class":
                    result = apply_learned_weights(pred_list, learned_class_weights, model_list)
                else:
                    # Use simple weighted average
                    probs = np.zeros(len(CATEGORIES))
                    for name, pred in vp.items():
                        w = weights.get(name, 1.0 / len(model_list))
                        probs += w * np.array(pred["probabilities"])
                    probs = probs / max(probs.sum(), 1e-8)
                    result = {"label": int(np.argmax(probs)), "probabilities": probs.tolist()}

                preds.append(result["label"])

            preds = np.array(preds)
            val_labs = np.array(val_labels[: len(preds)])
            acc = float(np.mean(preds == val_labs))
            metrics = overall_metrics(preds, val_labs, class_names)

            strategy_results[strat_name] = {
                "accuracy": acc,
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
            }

            print(f"  {strat_name}: acc={acc:.4f}, macro_f1={metrics['macro_f1']:.4f}")

        fold_results.append(
            {
                "strategies": strategy_results,
                "learned_class_weights": learned_class_weights,
                "learned_overall_weights": learned_overall,
                "n_train": len(train_labels_valid),
                "n_val": len(val_labels),
            }
        )

    # Aggregate
    summary = {}
    for strat_name in ["hand_tuned", "learned_overall", "learned_per_class"]:
        accs = [fr["strategies"][strat_name]["accuracy"] for fr in fold_results]
        f1s = [fr["strategies"][strat_name]["macro_f1"] for fr in fold_results]
        summary[strat_name] = {
            "accuracy": {"mean": float(np.mean(accs)), "std": float(np.std(accs))},
            "macro_f1": {"mean": float(np.mean(f1s)), "std": float(np.std(f1s))},
        }

    results = {"per_fold": fold_results, "summary": summary}

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "weight_learning.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)

    return results


def print_weight_summary(results):
    """Print weight learning summary."""
    s = results.get("summary", {})

    print("\n" + "=" * 70)
    print("WEIGHT LEARNING COMPARISON")
    print("=" * 70)
    print(f"{'Strategy':<25} {'Accuracy':>20} {'Macro F1':>20}")
    print("-" * 70)

    for strat, metrics in s.items():
        acc = metrics["accuracy"]
        f1 = metrics["macro_f1"]
        print(f"{strat:<25} {acc['mean']:.4f}±{acc['std']:.4f} {f1['mean']:.4f}±{f1['std']:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="retina_dataset")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", default="evaluation_results")
    args = parser.parse_args()

    results = run_weight_learning(args.dataset, args.folds, output_dir=args.output)
    if results:
        print_weight_summary(results)
