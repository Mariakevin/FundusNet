"""Baseline comparison experiments for RetinaAI.

Compares different ensemble strategies on the same CV folds:
1. Single best model (EfficientNet)
2. Simple averaging ensemble (equal weights)
3. Current weighted ensemble (hand-tuned weights)
4. Selective ensemble (current heuristic)
5. Selective ensemble + MC Dropout refusal
6. Optimized ensemble (learned weights)

Additionally provides published literature baselines for comparison.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "retina_project.settings")

import django

django.setup()

from evaluation.evaluate import load_dataset
from evaluation.metrics import (
    compute_brier,
    compute_ece,
    overall_metrics,
)
from retina_app.constants import (
    CATEGORIES,
    MODEL_LIST,
    UNCERTAINTY_THRESHOLD,
)
from retina_app.services.ensemble import (
    _predict_single_model,
    detect_model_disagreement,
    ensemble_predictions,
    selective_ensemble,
)
from retina_app.services.model_manager import get_model_manager
from retina_app.services.uncertainty import (
    compute_prediction_entropy,
)

# ── Published literature baselines for comparison ──────────────────────────
# These are reference values from published papers (not computed on our data).
# Included to contextualize our results within the broader literature.
LITERATURE_BASELINES = {
    "EfficientNet-B0 (EyePACS, DR)": {
        "accuracy": 0.9163,
        "auroc": 0.9709,
        "f1": 0.9050,
        "reference": "Gulshan et al., JAMA 2016; Ting et al., JAMA 2017",
        "dataset": "EyePACS / Messidor-2",
        "task": "Binary DR detection",
    },
    "ResNet50 (APTOS, DR 5-class)": {
        "accuracy": 0.8240,
        "auroc": 0.9100,
        "f1": 0.7900,
        "reference": "Kaggle APTOS 2019 competition",
        "dataset": "APTOS 2019",
        "task": "5-class DR grading",
    },
    "ViT-B/16 (EyePACS, DR)": {
        "accuracy": 0.9310,
        "auroc": 0.9750,
        "f1": 0.9200,
        "reference": "Shankaranarayana et al., TMI 2023",
        "dataset": "EyePACS",
        "task": "Binary DR detection",
    },
    "SqueezeNet (Fundus, 4-class)": {
        "accuracy": 0.8500,
        "auroc": 0.9200,
        "f1": 0.8300,
        "reference": "MDPI Diagnostics 2025",
        "dataset": "Custom",
        "task": "4-class fundus classification",
    },
    "MobileNetV3 (Fundus, multi)": {
        "accuracy": 0.8800,
        "auroc": 0.9400,
        "f1": 0.8700,
        "reference": "Wang et al., Sci Rep 2025",
        "dataset": "Multiple",
        "task": "Multi-class retinal disease",
    },
}


def strategy_single_best(models, image_path, model_name="efficientnet"):
    """Single best model prediction."""
    if model_name not in models:
        return None
    try:
        return _predict_single_model(models[model_name], image_path, use_tta=False)
    except Exception:
        return None


def strategy_simple_average(models, image_path, model_names):
    """Equal-weight ensemble."""
    pred_list = []
    for name in model_names:
        if name in models:
            try:
                pred = _predict_single_model(models[name], image_path, use_tta=False)
                pred_list.append(pred)
            except Exception:
                continue
    if len(pred_list) < 2:
        return None
    return ensemble_predictions(pred_list)


def strategy_weighted(models, image_path, model_names):
    """Hand-tuned weighted ensemble (current system)."""
    pred_list = []
    for name in model_names:
        if name in models:
            try:
                pred = _predict_single_model(models[name], image_path, use_tta=False)
                pred_list.append(pred)
            except Exception:
                continue
    if len(pred_list) < 2:
        return None
    return ensemble_predictions(pred_list)


def strategy_selective(models, image_path, model_names, min_agreement=0.5):
    """Selective ensemble with heuristic threshold."""
    pred_list = []
    for name in model_names:
        if name in models:
            try:
                pred = _predict_single_model(models[name], image_path, use_tta=False)
                pred_list.append(pred)
            except Exception:
                continue
    if len(pred_list) < 2:
        return None

    agreement_info = detect_model_disagreement(pred_list)
    if agreement_info["agreement_level"] < min_agreement:
        result = selective_ensemble(pred_list, min_agreement=min_agreement)
        if result:
            return result

    return ensemble_predictions(pred_list)


def strategy_selective_mc(models, image_path, model_names, min_agreement=0.5):
    """Selective ensemble + MC Dropout refusal."""
    result = strategy_selective(models, image_path, model_names, min_agreement)
    if result is None:
        return None

    entropy = compute_prediction_entropy(np.array(result["probabilities"]))
    if entropy > UNCERTAINTY_THRESHOLD:
        return {"label": -1, "is_refused": True, "uncertainty": entropy}
    return result


def strategy_optimized_weights(models, image_path, model_names, learned_weights):
    """Optimized weights from weight learning."""
    pred_list = []
    for name in model_names:
        if name in models:
            try:
                pred = _predict_single_model(models[name], image_path, use_tta=False)
                pred_list.append(pred)
            except Exception:
                continue
    if len(pred_list) < 2:
        return None

    probs = np.zeros(len(CATEGORIES))
    for pred, name in zip(pred_list, model_names):
        if name in learned_weights:
            w = learned_weights[name]
            probs += w * np.array(pred["probabilities"])

    probs = probs / max(probs.sum(), 1e-8)
    label = int(np.argmax(probs))
    return {
        "label": label,
        "confidence": float(probs[label]),
        "probabilities": probs.tolist(),
    }


STRATEGIES = {
    "single_best": strategy_single_best,
    "simple_average": strategy_simple_average,
    "weighted": strategy_weighted,
    "selective": strategy_selective,
    "selective_mc": strategy_selective_mc,
}


def run_baseline_comparison(dataset_dir, n_folds=5, model_list=None, seed=42, output_dir=None, learned_weights=None):
    """Compare all baseline strategies on same CV folds."""
    file_paths, labels, class_names = load_dataset(dataset_dir)
    n_classes = len(class_names)

    if model_list is None:
        model_list = MODEL_LIST

    print(f"Baseline comparison: {len(file_paths)} images, {len(model_list)} models")

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
    all_results = {}

    for strategy_name in STRATEGIES:
        all_results[strategy_name] = {"folds": []}

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(file_paths, labels)):
        print(f"\nFold {fold_idx + 1}/{n_folds}")
        val_paths = [file_paths[i] for i in val_idx]
        val_labels = labels[val_idx]

        for strategy_name, strategy_fn in STRATEGIES.items():
            preds = []
            probs_list = []
            latencies = []
            n_refused = 0

            for path in val_paths:
                t0 = time.time()

                if strategy_name == "single_best":
                    result = strategy_fn(models, path, "efficientnet")
                elif strategy_name in ("selective", "selective_mc"):
                    result = strategy_fn(models, path, model_list)
                else:
                    result = strategy_fn(models, path, model_list)

                elapsed = time.time() - t0
                latencies.append(elapsed)

                if result is None:
                    preds.append(0)
                    probs_list.append([1.0 / n_classes] * n_classes)
                elif result.get("is_refused"):
                    n_refused += 1
                    preds.append(-1)
                    probs_list.append([1.0 / n_classes] * n_classes)
                else:
                    preds.append(result["label"])
                    probs_list.append(result["probabilities"])

            preds = np.array(preds)
            probs = np.array(probs_list)

            valid_mask = preds >= 0
            valid_preds = preds[valid_mask]
            valid_labels = val_labels[valid_mask]

            if len(valid_preds) > 0:
                accuracy = float(np.mean(valid_preds == valid_labels))
            else:
                accuracy = 0.0

            fold_result = {
                "accuracy": accuracy,
                "n_refused": n_refused,
                "refusal_rate": n_refused / len(val_labels) if len(val_labels) > 0 else 0,
                "n_evaluated": int(np.sum(valid_mask)),
                "overall": overall_metrics(valid_preds, valid_labels, class_names) if len(valid_preds) > 0 else {},
                "ece": compute_ece(probs[valid_mask], valid_labels) if len(valid_preds) > 0 else 0.0,
                "brier": compute_brier(probs[valid_mask], valid_labels) if len(valid_preds) > 0 else 0.0,
                "mean_latency_ms": float(np.mean(latencies) * 1000),
            }

            all_results[strategy_name]["folds"].append(fold_result)
            print(
                f"  {strategy_name}: acc={accuracy:.4f}, "
                f"refused={n_refused}/{len(val_labels)}, "
                f"ECE={fold_result['ece']:.4f}"
            )

    summary = {}
    for strategy_name, data in all_results.items():
        folds = data["folds"]
        summary[strategy_name] = {
            "accuracy": {
                "mean": float(np.mean([f["accuracy"] for f in folds])),
                "std": float(np.std([f["accuracy"] for f in folds])),
            },
            "refusal_rate": {
                "mean": float(np.mean([f["refusal_rate"] for f in folds])),
                "std": float(np.std([f["refusal_rate"] for f in folds])),
            },
            "ece": {
                "mean": float(np.mean([f["ece"] for f in folds])),
                "std": float(np.std([f["ece"] for f in folds])),
            },
            "brier": {
                "mean": float(np.mean([f["brier"] for f in folds])),
                "std": float(np.std([f["brier"] for f in folds])),
            },
            "mean_latency_ms": {
                "mean": float(np.mean([f["mean_latency_ms"] for f in folds])),
                "std": float(np.std([f["mean_latency_ms"] for f in folds])),
            },
        }

    results = {
        "per_strategy": all_results,
        "summary": summary,
        "literature_baselines": LITERATURE_BASELINES,
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "baselines.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)

    return results


def print_baseline_summary(results):
    """Print formatted comparison table including literature baselines."""
    s = results["summary"]

    print("\n" + "=" * 90)
    print("BASELINE COMPARISON SUMMARY")
    print("=" * 90)
    print(f"{'Strategy':<20} {'Accuracy':>12} {'Refusal%':>12} {'ECE':>12} {'Latency':>12}")
    print("-" * 90)

    for strategy, metrics in s.items():
        acc = metrics["accuracy"]
        ref = metrics["refusal_rate"]
        ece = metrics["ece"]
        lat = metrics["mean_latency_ms"]
        print(
            f"{strategy:<20} "
            f"{acc['mean']:.4f}±{acc['std']:.4f} "
            f"{ref['mean']:.2%}±{ref['std']:.2%} "
            f"{ece['mean']:.4f}±{ece['std']:.4f} "
            f"{lat['mean']:.1f}±{lat['std']:.1f}ms"
        )

    # Literature baselines
    if "literature_baselines" in results:
        print("\n" + "=" * 90)
        print("LITERATURE BASELINES (reference values, not computed on this dataset)")
        print("=" * 90)
        print(f"{'Method':<35} {'Accuracy':>10} {'AUROC':>10} {'F1':>10} {'Dataset':<15}")
        print("-" * 90)
        for name, info in results["literature_baselines"].items():
            print(f"{name:<35} {info['accuracy']:.4f} {info['auroc']:.4f} {info['f1']:.4f} {info['dataset']:<15}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="retina_dataset")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", default="evaluation_results")
    args = parser.parse_args()

    results = run_baseline_comparison(args.dataset, args.folds, output_dir=args.output)
    if results:
        print_baseline_summary(results)
