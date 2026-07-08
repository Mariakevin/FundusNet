"""
MC Dropout Uncertainty Quantification.

Run T stochastic forward passes with dropout enabled at inference time.
Compute variance/entropy across predictions to estimate model uncertainty.
Based on Gal & Ghahramani (ICML 2016).
"""

import logging
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from retina_app.constants import (
    CATEGORIES,
    MC_DROPOUT_PASSES,
    UNCERTAINTY_THRESHOLD,
    ENABLE_MC_DROPOUT,
)
from retina_app.services.transforms import TRANSFORM

logger = logging.getLogger("retina_app")


def _enable_dropout(model: nn.Module) -> List[nn.Module]:
    """Set all dropout layers to train mode while keeping batchnorm in eval mode.
    Returns list of dropout modules that were modified."""
    dropout_layers = []
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
            dropout_layers.append(module)
    return dropout_layers


def _disable_dropout(dropout_layers: List[nn.Module]) -> None:
    """Restore all dropout layers to eval mode."""
    for module in dropout_layers:
        module.eval()


def compute_entropy(probs: np.ndarray) -> float:
    """Compute Shannon entropy of a probability distribution.
    Higher entropy = more uncertain."""
    probs = np.clip(probs, 1e-10, 1.0)
    probs = probs / probs.sum()
    entropy = -np.sum(probs * np.log(probs))
    return float(entropy)


def compute_prediction_entropy(probs: np.ndarray) -> float:
    """Compute normalized entropy (0 = certain, 1 = maximally uncertain).
    Normalized by max entropy for n classes."""
    raw_entropy = compute_entropy(probs)
    max_entropy = np.log(len(probs))
    if max_entropy == 0:
        return 0.0
    return raw_entropy / max_entropy


def mc_dropout_forward_pass(
    model: nn.Module,
    image_tensor: torch.Tensor,
    n_passes: int = MC_DROPOUT_PASSES,
) -> Tuple[np.ndarray, float, bool]:
    """Run T stochastic forward passes with dropout enabled.

    Args:
        model: PyTorch model with dropout layers
        image_tensor: Preprocessed input tensor [1, C, H, W]
        n_passes: Number of MC Dropout forward passes

    Returns:
        Tuple of (mean_probabilities, entropy, is_uncertain)
    """
    from retina_app.services.model_manager import DEVICE

    all_probs = []

    # Save original dropout states
    dropout_layers = _enable_dropout(model)

    try:
        for _ in range(n_passes):
            with torch.no_grad():
                output = model(image_tensor.to(DEVICE))

                if isinstance(output, tuple):
                    output = output[0]

                if len(output.shape) > 1:
                    probs = F.softmax(output, dim=1)
                else:
                    probs = F.softmax(output, dim=0).unsqueeze(0)

                all_probs.append(probs.cpu().numpy().flatten())
    finally:
        # Always restore dropout to eval mode
        model.eval()
        _disable_dropout(dropout_layers)

    if not all_probs:
        return np.zeros(len(CATEGORIES)), 1.0, True

    # Stack predictions and compute statistics
    stacked = np.stack(all_probs, axis=0)  # [T, n_classes]
    mean_probs = np.mean(stacked, axis=0)  # [n_classes]
    mean_probs = mean_probs / mean_probs.sum()  # Re-normalize

    # Compute uncertainty metrics
    entropy = compute_prediction_entropy(mean_probs)

    # Compute variance across passes (model uncertainty)
    variance = np.var(stacked, axis=0)
    avg_variance = float(np.mean(variance))

    # Classification uncertainty: 1 - max probability
    max_confidence = float(np.max(mean_probs))

    # Combined uncertainty score
    is_uncertain = entropy > UNCERTAINTY_THRESHOLD

    logger.debug(
        f"MC Dropout: entropy={entropy:.4f}, avg_variance={avg_variance:.4f}, "
        f"max_confidence={max_confidence:.4f}, uncertain={is_uncertain}"
    )

    return mean_probs, entropy, is_uncertain


def mc_dropout_single_model(
    model: nn.Module,
    image_path: str = None,
    n_passes: int = MC_DROPOUT_PASSES,
    input_tensor: torch.Tensor = None,
) -> Dict[str, Any]:
    """Run MC Dropout uncertainty estimation for a single model.

    Accepts either image_path (loads and transforms internally) or
    pre-computed input_tensor (avoids redundant image loading in ensemble mode).

    Returns dict with mean_probs, entropy, is_uncertain, individual_predictions.
    """
    from retina_app.services.transforms import TRANSFORM
    from PIL import Image

    if input_tensor is None:
        if image_path is None:
            raise ValueError("Either image_path or input_tensor must be provided")
        with Image.open(image_path) as pil_img:
            image = pil_img.convert("RGB")
            input_tensor = TRANSFORM(image).unsqueeze(0)

    mean_probs, entropy, is_uncertain = mc_dropout_forward_pass(
        model, input_tensor, n_passes
    )

    max_idx = int(np.argmax(mean_probs))
    confidence = float(mean_probs[max_idx])

    return {
        "label": CATEGORIES[max_idx],
        "confidence": confidence,
        "probabilities": mean_probs.tolist(),
        "entropy": entropy,
        "is_uncertain": is_uncertain,
        "n_passes": n_passes,
    }


def mc_dropout_ensemble(
    models: Dict[str, nn.Module],
    image_path: str,
    model_weights: Dict[str, float] = None,
    n_passes: int = MC_DROPOUT_PASSES,
) -> Dict[str, Any]:
    """Run MC Dropout across an ensemble of models.

    Loads and transforms image ONCE, then passes the tensor to each model.
    Each model runs T forward passes independently.
    Results are combined using weighted averaging.

    Returns dict with aggregated predictions and uncertainty metrics.
    """
    from retina_app.constants import MODEL_WEIGHTS
    from retina_app.services.transforms import TRANSFORM
    from PIL import Image

    if model_weights is None:
        model_weights = MODEL_WEIGHTS

    if not models:
        return {
            "label": "Unknown",
            "confidence": 0.0,
            "probabilities": [0.0] * len(CATEGORIES),
            "entropy": 1.0,
            "is_uncertain": True,
            "n_models": 0,
            "individual_results": [],
        }

    # Load and transform image once (avoids redundant decoding per model)
    with Image.open(image_path) as pil_img:
        image = pil_img.convert("RGB")
        shared_tensor = TRANSFORM(image).unsqueeze(0)

    all_model_results = []

    for model_type, model in models.items():
        try:
            result = mc_dropout_single_model(
                model, n_passes=n_passes, input_tensor=shared_tensor
            )
            weight = model_weights.get(model_type, 1.0 / len(models))
            result["weight"] = weight
            result["model_type"] = model_type
            all_model_results.append(result)
            logger.debug(
                f"MC Dropout {model_type}: {result['label']} "
                f"({result['confidence']:.3f}), entropy={result['entropy']:.4f}"
            )
        except Exception as exc:
            logger.warning(f"MC Dropout failed for {model_type}: {exc}")

    if not all_model_results:
        return {
            "label": "Unknown",
            "confidence": 0.0,
            "probabilities": [0.0] * len(CATEGORIES),
            "entropy": 1.0,
            "is_uncertain": True,
            "n_models": 0,
            "individual_results": [],
        }

    # Weighted average of mean probabilities across models
    n_classes = len(CATEGORIES)
    weighted_probs = np.zeros(n_classes)
    total_weight = 0.0

    for result in all_model_results:
        probs = np.array(result["probabilities"])
        weight = result["weight"]
        weighted_probs += probs * weight
        total_weight += weight

    if total_weight > 0:
        weighted_probs = weighted_probs / total_weight

    # Re-normalize
    if weighted_probs.sum() > 0:
        weighted_probs = weighted_probs / weighted_probs.sum()

    max_idx = int(np.argmax(weighted_probs))
    confidence = float(weighted_probs[max_idx])
    entropy = compute_prediction_entropy(weighted_probs)
    is_uncertain = entropy > UNCERTAINTY_THRESHOLD

    return {
        "label": CATEGORIES[max_idx],
        "confidence": confidence,
        "probabilities": weighted_probs.tolist(),
        "entropy": entropy,
        "is_uncertain": is_uncertain,
        "n_models": len(all_model_results),
        "n_passes": n_passes,
        "individual_results": all_model_results,
    }


def is_dropout_enabled() -> bool:
    """Check if MC Dropout uncertainty is enabled."""
    return ENABLE_MC_DROPOUT


def compute_ensemble_disagreement(predictions):
    """Compute ensemble disagreement as an uncertainty signal.

    Uses the fraction of models that disagree with the majority vote.

    Args:
        predictions: list of prediction dicts from individual models

    Returns:
        float: disagreement score (0 = full agreement, 1 = full disagreement)
    """
    if len(predictions) < 2:
        return 0.0

    votes = [p["label"] for p in predictions]
    majority_label = max(set(votes), key=votes.count)
    n_disagree = sum(1 for v in votes if v != majority_label)

    return n_disagree / len(votes)


def calibrate_threshold(signal_values, true_labels, predictions,
                          metric="f1", n_steps=50):
    """Find optimal uncertainty threshold via validation sweep.

    Args:
        signal_values: (n_samples,) uncertainty signal values
        true_labels: (n_samples,) ground truth labels
        predictions: (n_samples,) predicted labels
        metric: metric to optimize ('f1', 'accuracy', 'precision')
        n_steps: number of thresholds to test

    Returns:
        dict with optimal_threshold, best_score, and all_results
    """
    signal_values = np.array(signal_values)
    true_labels = np.array(true_labels)
    predictions = np.array(predictions)

    min_val = float(np.min(signal_values))
    max_val = float(np.max(signal_values))

    if min_val >= max_val:
        return {"optimal_threshold": min_val, "best_score": 0.0, "all_results": []}

    thresholds = np.linspace(min_val, max_val, n_steps)
    all_results = []

    for threshold in thresholds:
        # Keep predictions where uncertainty is below threshold
        keep_mask = signal_values <= threshold
        n_kept = int(np.sum(keep_mask))
        refusal_rate = 1.0 - n_kept / len(signal_values) if len(signal_values) > 0 else 0

        if n_kept < 5:
            continue

        kept_preds = predictions[keep_mask]
        kept_labels = true_labels[keep_mask]

        # Compute metric
        if metric == "accuracy":
            score = float(np.mean(kept_preds == kept_labels))
        elif metric == "f1":
            # Macro F1
            classes = np.unique(kept_labels)
            f1s = []
            for c in classes:
                tp = np.sum((kept_preds == c) & (kept_labels == c))
                fp = np.sum((kept_preds == c) & (kept_labels != c))
                fn = np.sum((kept_preds != c) & (kept_labels == c))
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
                f1s.append(f1)
            score = float(np.mean(f1s))
        elif metric == "precision":
            classes = np.unique(kept_labels)
            precs = []
            for c in classes:
                tp = np.sum((kept_preds == c) & (kept_labels == c))
                fp = np.sum((kept_preds == c) & (kept_labels != c))
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                precs.append(prec)
            score = float(np.mean(precs))
        else:
            score = float(np.mean(kept_preds == kept_labels))

        all_results.append({
            "threshold": float(threshold),
            "score": score,
            "refusal_rate": refusal_rate,
            "n_kept": n_kept,
        })

    if not all_results:
        return {"optimal_threshold": min_val, "best_score": 0.0, "all_results": []}

    best = max(all_results, key=lambda x: x["score"])
    return {
        "optimal_threshold": best["threshold"],
        "best_score": best["score"],
        "best_refusal_rate": best["refusal_rate"],
        "all_results": all_results,
    }
