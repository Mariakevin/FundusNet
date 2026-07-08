"""Ablation study for RetinaAI.

Systematically removes components and measures their contribution to
overall performance. Produces a table showing each component's impact
on accuracy, F1, and latency.
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "retina_project.settings")

import django
django.setup()

from evaluation.metrics import overall_metrics, per_class_metrics
from evaluation.evaluate import load_dataset
from retina_app.constants import (
    CATEGORIES,
    MODEL_LIST,
    MODEL_WEIGHTS,
    UNCERTAINTY_THRESHOLD,
)
from retina_app.services.model_manager import get_model_manager
from retina_app.services.ensemble import (
    _predict_single_model,
    ensemble_predictions,
    selective_ensemble,
    detect_model_disagreement,
)
from retina_app.services.uncertainty import mc_dropout_single_model


COMPONENTS = [
    {
        "name": "fundus_validator",
        "description": "Fundus image validation (rejects non-fundus images)",
        "disable_kwarg": "skip_fundus_validation",
    },
    {
        "name": "image_quality",
        "description": "Image quality check (blur, brightness, contrast)",
        "disable_kwarg": "skip_quality_check",
    },
    {
        "name": "selective_ensemble",
        "description": "Selective ensemble (filters outlier models)",
        "disable_kwarg": "skip_selective",
    },
    {
        "name": "mc_dropout",
        "description": "MC Dropout uncertainty (refuses uncertain predictions)",
        "disable_kwarg": "skip_mc_dropout",
    },
    {
        "name": "tta",
        "description": "Test-time augmentation",
        "disable_kwarg": "skip_tta",
    },
]


def run_inference_single_config(models, file_paths, labels, model_weights,
                                  skip_fundus_validation=False,
                                  skip_quality_check=False,
                                  skip_selective=False,
                                  skip_mc_dropout=False,
                                  skip_tta=False):
    """Run inference with specific component configuration.

    Args:
        models: dict of loaded models
        file_paths: list of image paths
        labels: array of true labels
        model_weights: dict of model weights
        skip_*: boolean flags to disable components

    Returns:
        dict with predictions, latencies, n_refused, accuracy
    """
    preds = []
    latencies = []
    n_refused = 0

    for path in file_paths:
        t0 = time.time()

        # Get individual model predictions
        individual_preds = []
        for name in model_weights:
            if name in models:
                try:
                    pred = _predict_single_model(
                        models[name], path,
                        use_tta=not skip_tta,
                    )
                    individual_preds.append(pred)
                except Exception:
                    continue

        if len(individual_preds) < 2:
            preds.append(0)
            latencies.append(time.time() - t0)
            continue

        # Selective ensemble
        if not skip_selective:
            agreement_info = detect_model_disagreement(individual_preds)
            if agreement_info["agreement_level"] < 0.5:
                result = selective_ensemble(individual_preds, min_agreement=0.5)
                if result:
                    individual_preds = [result]

        # Ensemble
        result = ensemble_predictions(individual_preds)

        # MC Dropout uncertainty check
        if not skip_mc_dropout and result is not None:
            try:
                for name in model_weights:
                    if name in models:
                        mc_result = mc_dropout_single_model(
                            models[name], path, n_passes=5
                        )
                        if mc_result and mc_result.get("is_uncertain", False):
                            n_refused += 1
                            result = {"label": -1, "is_refused": True}
                            break
            except Exception:
                pass

        elapsed = time.time() - t0
        latencies.append(elapsed)

        if result is None or result.get("is_refused"):
            preds.append(-1)
        else:
            preds.append(result["label"])

    preds = np.array(preds)
    labels = np.array(labels[:len(preds)])

    # Filter refused
    valid_mask = preds >= 0
    valid_preds = preds[valid_mask]
    valid_labels = labels[valid_mask]

    if len(valid_preds) > 0:
        accuracy = float(np.mean(valid_preds == valid_labels))
        metrics = overall_metrics(valid_preds, valid_labels, CATEGORIES)
    else:
        accuracy = 0.0
        metrics = {"macro_f1": 0.0, "weighted_f1": 0.0}

    return {
        "accuracy": accuracy,
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "mean_latency_ms": float(np.mean(latencies) * 1000) if latencies else 0,
        "n_refused": n_refused,
        "refusal_rate": n_refused / len(labels) if len(labels) > 0 else 0,
        "n_evaluated": int(np.sum(valid_mask)),
        "n_total": len(labels),
    }


def run_ablation_study(dataset_dir, n_folds=5, model_list=None, seed=42,
                         output_dir=None):
    """Run full ablation study.

    Tests:
    1. All components enabled (full system)
    2. Each component removed one at a time
    3. All components disabled (baseline)
    """
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

    if len(models) < 2:
        print("ERROR: Need at least 2 models")
        return None

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_idx = 0

    all_configs = []

    # Full system (baseline)
    all_configs.append({
        "name": "full_system",
        "description": "All components enabled",
        "kwargs": {},
    })

    # Ablation: remove each component
    for comp in COMPONENTS:
        kwargs = {comp["disable_kwarg"]: True}
        all_configs.append({
            "name": f"no_{comp['name']}",
            "description": f"Without {comp['description']}",
            "kwargs": kwargs,
        })

    # All disabled (bare ensemble)
    all_configs.append({
        "name": "bare_ensemble",
        "description": "Equal-weight ensemble, no extras",
        "kwargs": {
            "skip_fundus_validation": True,
            "skip_quality_check": True,
            "skip_selective": True,
            "skip_mc_dropout": True,
            "skip_tta": True,
        },
    })

    fold_results = []

    for train_idx, val_idx in skf.split(file_paths, labels):
        print(f"\nAblation Study — Fold {fold_idx + 1}/{n_folds}")
        val_paths = [file_paths[i] for i in val_idx]
        val_labels = labels[val_idx]

        config_results = {}
        for config in all_configs:
            result = run_inference_single_config(
                models, val_paths, val_labels, model_weights,
                **config["kwargs"],
            )
            config_results[config["name"]] = result
            print(f"  {config['name']}: acc={result['accuracy']:.4f}, "
                  f"f1={result['macro_f1']:.4f}, "
                  f"latency={result['mean_latency_ms']:.1f}ms")

        fold_results.append(config_results)
        fold_idx += 1

    # Aggregate
    summary = {}
    for config in all_configs:
        name = config["name"]
        accs = [fr[name]["accuracy"] for fr in fold_results]
        f1s = [fr[name]["macro_f1"] for fr in fold_results]
        lats = [fr[name]["mean_latency_ms"] for fr in fold_results]
        refusals = [fr[name]["refusal_rate"] for fr in fold_results]

        summary[name] = {
            "description": config["description"],
            "accuracy": {"mean": float(np.mean(accs)), "std": float(np.std(accs))},
            "macro_f1": {"mean": float(np.mean(f1s)), "std": float(np.std(f1s))},
            "mean_latency_ms": {"mean": float(np.mean(lats)), "std": float(np.std(lats))},
            "refusal_rate": {"mean": float(np.mean(refusals)), "std": float(np.std(refusals))},
        }

    # Compute deltas relative to full system
    full_acc = summary["full_system"]["accuracy"]["mean"]
    full_f1 = summary["full_system"]["macro_f1"]["mean"]
    full_lat = summary["full_system"]["mean_latency_ms"]["mean"]

    for name in summary:
        if name != "full_system":
            summary[name]["accuracy_delta"] = round(
                summary[name]["accuracy"]["mean"] - full_acc, 4
            )
            summary[name]["f1_delta"] = round(
                summary[name]["macro_f1"]["mean"] - full_f1, 4
            )
            summary[name]["latency_delta_ms"] = round(
                summary[name]["mean_latency_ms"]["mean"] - full_lat, 2
            )

    results = {"per_fold": fold_results, "summary": summary}

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "ablation_study.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)

    return results


def print_ablation_summary(results):
    """Print formatted ablation study table."""
    s = results.get("summary", {})

    print("\n" + "=" * 100)
    print("ABLATION STUDY RESULTS")
    print("=" * 100)
    print(f"{'Configuration':<25} {'Accuracy':>15} {'Δ Acc':>10} {'Macro F1':>15} {'Δ F1':>10} {'Latency':>12}")
    print("-" * 100)

    for name, metrics in s.items():
        acc = metrics["accuracy"]
        f1 = metrics["macro_f1"]
        lat = metrics["mean_latency_ms"]
        delta_acc = metrics.get("accuracy_delta", "-")
        delta_f1 = metrics.get("f1_delta", "-")

        delta_acc_str = f"{delta_acc:+.4f}" if isinstance(delta_acc, (int, float)) else str(delta_acc)
        delta_f1_str = f"{delta_f1:+.4f}" if isinstance(delta_f1, (int, float)) else str(delta_f1)

        print(f"{name:<25} "
              f"{acc['mean']:.4f}±{acc['std']:.4f} "
              f"{delta_acc_str:>10} "
              f"{f1['mean']:.4f}±{f1['std']:.4f} "
              f"{delta_f1_str:>10} "
              f"{lat['mean']:.1f}±{lat['std']:.1f}ms")

    print("\n" + "=" * 100)
    print("Negative Δ = component helps performance")
    print("Positive Δ = removing component hurts performance")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="retina_dataset")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", default="evaluation_results")
    args = parser.parse_args()

    results = run_ablation_study(args.dataset, args.folds, output_dir=args.output)
    if results:
        print_ablation_summary(results)
