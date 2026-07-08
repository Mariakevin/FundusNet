"""
Ensemble prediction logic — single model inference, TTA, weighted averaging.
"""

import logging
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from retina_app.constants import (
    CATEGORIES,
    MODEL_WEIGHTS,
    CLASS_PERFORMANCE_WEIGHTS,
    TTA_AGGREGATION_METHOD,
    TEMPERATURE_SCALING,
    ENABLE_MC_DROPOUT,
    MC_DROPOUT_PASSES,
)
from retina_app.services.exceptions import InferenceError
from retina_app.services.model_manager import DEVICE
from retina_app.services.transforms import TRANSFORMS, TRANSFORM

logger = logging.getLogger("retina_app")


def apply_temperature_scaling(logits: torch.Tensor, temperature: float = TEMPERATURE_SCALING) -> torch.Tensor:
    """Apply temperature scaling for confidence calibration."""
    if temperature == 1.0:
        return torch.softmax(logits, dim=1)
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=1)


def _predict_single_model(model: nn.Module, image_path: str, use_tta: bool = False) -> Dict[str, Any]:
    """Run inference on a single model with optional test-time augmentation."""
    with Image.open(image_path) as pil_img:
        image = pil_img.convert("RGB")
        if use_tta:
            all_probs = []
            failed_transforms = []

            for transform_name, transform in TRANSFORMS.items():
                try:
                    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        output = model(input_tensor)

                        if isinstance(output, tuple):
                            output = output[0]

                        if len(output.shape) == 4:
                            output = output.view(output.size(0), output.size(1), -1).squeeze(-1).squeeze(0)
                        elif len(output.shape) == 3:
                            output = output.view(output.size(0), output.size(1), -1).squeeze(-1).squeeze(0)
                        elif len(output.shape) == 2:
                            output = output.squeeze(0)
                        elif len(output.shape) == 1:
                            pass

                        if output.numel() != len(CATEGORIES):
                            if output.numel() % len(CATEGORIES) == 0:
                                spatial_size = output.numel() // len(CATEGORIES)
                                h = w = int(spatial_size ** 0.5)
                                if h * w == spatial_size:
                                    output = output.view(len(CATEGORIES), h, w).mean(dim=(1, 2))
                                else:
                                    logger.warning(f"Cannot reshape TTA output {output.numel()} to {len(CATEGORIES)}")
                                    continue
                            else:
                                logger.warning(f"Unexpected TTA output size: {output.numel()}")
                                continue

                        probs = F.softmax(output, dim=0)
                        all_probs.append(probs.cpu().numpy())
                except Exception as exc:
                    logger.warning(f"Transform {transform_name} failed: {exc}")
                    failed_transforms.append(transform_name)
                    continue

            if not all_probs:
                raise InferenceError(f"All TTA transforms failed for {image_path}")

            if TTA_AGGREGATION_METHOD == "geometric":
                stacked = np.stack(all_probs)
                log_probs = np.log(stacked + 1e-10)
                avg_log = np.mean(log_probs, axis=0)
                avg_probs = np.exp(avg_log)
                avg_probs = avg_probs / np.sum(avg_probs)
            else:
                avg_probs = sum(all_probs) / len(all_probs)

            max_idx = max(range(len(avg_probs)), key=lambda i: avg_probs[i])
            confidence = avg_probs[max_idx]

            result = {
                "label": CATEGORIES[max_idx],
                "confidence": float(confidence),
                "probabilities": avg_probs.tolist(),
            }

            if failed_transforms:
                result["warnings"] = [f"Transform {t} failed" for t in failed_transforms]

            return result
        else:
            input_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = model(input_tensor)

                if isinstance(output, tuple):
                    output = output[0]

                if len(output.shape) == 4:
                    output = output.view(output.size(0), output.size(1), -1)
                    output = output.squeeze(-1).squeeze(0)
                elif len(output.shape) == 3:
                    output = output.view(output.size(0), output.size(1), -1).squeeze(-1).squeeze(0)
                elif len(output.shape) == 2:
                    output = output.squeeze(0)
                elif len(output.shape) == 1:
                    pass

                if output.numel() != len(CATEGORIES):
                    if output.numel() % len(CATEGORIES) == 0:
                        spatial_size = output.numel() // len(CATEGORIES)
                        h = w = int(spatial_size ** 0.5)
                        if h * w == spatial_size:
                            output = output.view(len(CATEGORIES), h, w)
                            output = output.mean(dim=(1, 2))
                        else:
                            raise InferenceError(f"Cannot reshape output {output.numel()} to {len(CATEGORIES)} classes")
                    else:
                        raise InferenceError(f"Unexpected output size: {output.numel()}, expected {len(CATEGORIES)}")

                probabilities = F.softmax(output, dim=0)
                confidence, predicted_idx = torch.max(probabilities, dim=0)

            return {
                "label": CATEGORIES[predicted_idx.item()],
                "confidence": float(confidence.item()),
                "probabilities": probabilities.cpu().numpy().tolist(),
            }


def predict_models_parallel(models: Dict[str, nn.Module], image_path: str, use_tta: bool, executor) -> List[tuple]:
    """Run parallel inference on multiple models using ThreadPoolExecutor."""
    predictions = []

    def predict_wrapper(model_type_and_model):
        model_type, model = model_type_and_model
        try:
            pred = _predict_single_model(model, image_path, use_tta=use_tta)
            return (model_type, pred, None)
        except Exception as exc:
            logger.error(f"Model {model_type} failed: {exc}")
            return (model_type, None, exc)

    results = list(executor.map(predict_wrapper, list(models.items())))

    for model_type, pred, error in results:
        if pred is not None:
            predictions.append((model_type, pred))
        else:
            logger.warning(f"Model {model_type} prediction failed: {error}")

    if not predictions:
        raise InferenceError("All models failed to make predictions")

    return predictions


def ensemble_predictions(predictions: List[tuple]) -> Dict[str, Any]:
    """Combine predictions from multiple models using per-class dynamic weighted averaging."""
    if not predictions:
        raise ValueError("No predictions to ensemble")

    n_models = len(predictions)
    n_classes = len(CATEGORIES)

    weighted_probs = [0.0] * n_classes
    class_weights = [0.0] * n_classes
    model_details = []

    # Collect raw weights first, then normalize to sum to 1.0
    raw_weights = []
    valid_predictions = []
    for model_type, pred in predictions:
        probs = pred["probabilities"]
        confidence = pred["confidence"]
        predicted_class = pred["label"]

        base_weight = MODEL_WEIGHTS.get(model_type, 1.0 / n_models)

        if predicted_class in CLASS_PERFORMANCE_WEIGHTS:
            class_weight = CLASS_PERFORMANCE_WEIGHTS[predicted_class].get(model_type, base_weight)
        else:
            class_weight = base_weight

        confidence_boost = 1.0 + (confidence - 0.5) * 0.2
        final_weight = class_weight * confidence_boost

        raw_weights.append(final_weight)
        valid_predictions.append((model_type, pred, final_weight, predicted_class, confidence))

    # Normalize weights so they sum to 1.0 — prevents random/untrained models from dominating
    total_weight = sum(raw_weights)
    if total_weight > 0:
        raw_weights = [w / total_weight for w in raw_weights]

    for i, (model_type, pred, _, predicted_class, confidence) in enumerate(valid_predictions):
        probs = pred["probabilities"]
        final_weight = raw_weights[i]

        for j, p in enumerate(probs):
            weighted_probs[j] += p * final_weight
            class_weights[j] += final_weight

        model_details.append({
            "model": model_type,
            "label": predicted_class,
            "confidence": confidence,
            "weight": final_weight,
        })

    normalized_probs = [
        weighted_probs[i] / class_weights[i] if class_weights[i] > 0 else 0
        for i in range(n_classes)
    ]

    total = sum(normalized_probs)
    if total > 0:
        normalized_probs = [p / total for p in normalized_probs]

    max_idx = int(np.argmax(normalized_probs))

    avg_confidence = sum(pred["confidence"] for _, pred in predictions) / n_models

    ensemble_uncertainty = 1.0 - max(normalized_probs)

    logger.info(f"Ensemble used {n_models} models: {model_details}")
    logger.debug(f"Ensemble probabilities: {normalized_probs}")

    return {
        "label": CATEGORIES[max_idx],
        "confidence": normalized_probs[max_idx],
        "avg_model_confidence": avg_confidence,
        "n_models": n_models,
        "probabilities": normalized_probs,
        "uncertainty": ensemble_uncertainty,
    }


def predict_with_uncertainty_ensemble(
    models: Dict[str, nn.Module],
    image_path: str,
    model_weights: Dict[str, float] = None,
    n_passes: int = MC_DROPOUT_PASSES,
) -> Dict[str, Any]:
    """Run MC Dropout uncertainty quantification across an ensemble.

    Each model runs T stochastic forward passes with dropout enabled.
    Results are aggregated using weighted averaging.

    Returns dict with prediction, entropy, is_uncertain, confidence_interval.
    """
    from retina_app.services.uncertainty import mc_dropout_ensemble

    return mc_dropout_ensemble(
        models, image_path,
        model_weights=model_weights,
        n_passes=n_passes,
    )


def detect_model_disagreement(predictions: List[tuple]) -> Dict[str, Any]:
    """Detect when models disagree on the predicted class.

    Returns disagreement analysis including which models disagree,
    the level of agreement, and the dominant prediction.
    """
    if len(predictions) < 2:
        label = predictions[0][1]["label"] if predictions else None
        return {
            "disagreement": False,
            "agreement_level": 1.0,
            "dominant_class": label,
            "class_votes": {label: 1} if label else {},
            "disagreeing_models": [],
            "model_predictions": {},
        }

    class_votes = {}
    model_predictions = {}
    for model_type, pred in predictions:
        label = pred["label"]
        class_votes[label] = class_votes.get(label, 0) + 1
        model_predictions[model_type] = label

    n_models = len(predictions)
    dominant_class = max(class_votes, key=class_votes.get)
    dominant_count = class_votes[dominant_class]
    agreement_level = dominant_count / n_models

    disagreeing_models = [
        mt for mt, label in model_predictions.items()
        if label != dominant_class
    ]

    return {
        "disagreement": len(disagreeing_models) > 0,
        "agreement_level": round(agreement_level, 3),
        "dominant_class": dominant_class,
        "class_votes": class_votes,
        "disagreeing_models": disagreeing_models,
        "model_predictions": model_predictions,
    }


def selective_ensemble(
    predictions: List[tuple],
    min_agreement: float = 0.5,
) -> Dict[str, Any]:
    """Selective ensemble that filters out outlier predictions.

    If models disagree strongly, keep only the models that agree
    with the majority and re-ensemble. Falls back to full ensemble
    if no majority exists.
    """
    if len(predictions) <= 2:
        result = ensemble_predictions(predictions)
        result["selective_ensemble"] = False
        result["agreement_level"] = 1.0
        return result

    analysis = detect_model_disagreement(predictions)

    if not analysis["disagreement"]:
        return ensemble_predictions(predictions)

    if analysis["agreement_level"] >= min_agreement:
        majority_class = analysis["dominant_class"]
        filtered = [
            (mt, pred) for mt, pred in predictions
            if pred["label"] == majority_class
        ]

        if len(filtered) >= 2:
            logger.info(
                f"Selective ensemble: {len(filtered)}/{len(predictions)} models agree on "
                f"'{majority_class}' (agreement={analysis['agreement_level']:.2f})"
            )
            result = ensemble_predictions(filtered)
            result["selective_ensemble"] = True
            result["original_n_models"] = len(predictions)
            result["filtered_n_models"] = len(filtered)
            result["agreement_level"] = analysis["agreement_level"]
            return result

    logger.warning(
        f"Low agreement ({analysis['agreement_level']:.2f}) — using full ensemble. "
        f"Votes: {analysis['class_votes']}"
    )
    result = ensemble_predictions(predictions)
    result["selective_ensemble"] = False
    result["agreement_level"] = analysis["agreement_level"]
    return result


def compute_agreement_scores(predictions):
    """Compute continuous agreement scores for each prediction.

    Unlike detect_model_disagreement which returns a single aggregate,
    this computes a per-prediction agreement score.

    Args:
        predictions: list of prediction dicts from individual models

    Returns:
        dict with agreement_score (float 0-1), class_votes (dict),
        majority_class, n_models
    """
    if not predictions:
        return {
            "agreement_score": 0.0,
            "class_votes": {},
            "majority_class": None,
            "n_models": 0,
        }

    votes = [p["label"] for p in predictions]
    class_counts = {}
    for v in votes:
        class_counts[v] = class_counts.get(v, 0) + 1

    majority_class = max(class_counts, key=class_counts.get)
    agreement_score = class_counts[majority_class] / len(votes)

    return {
        "agreement_score": agreement_score,
        "class_votes": class_counts,
        "majority_class": majority_class,
        "n_models": len(predictions),
    }


def selective_ensemble_adaptive(predictions, target_metric="accuracy"):
    """Adaptive selective ensemble that learns threshold from data.

    Instead of a fixed agreement threshold, this computes the agreement
    score and returns it alongside the prediction for downstream thresholding.

    Args:
        predictions: list of prediction dicts from individual models
        target_metric: unused, kept for API consistency

    Returns:
        dict with ensemble result, agreement_score, and individual scores
    """
    if not predictions:
        return None

    agreement_info = compute_agreement_scores(predictions)
    agreement_score = agreement_info["agreement_score"]

    # Filter models that disagree with majority
    majority_class = agreement_info["majority_class"]
    filtered = [p for p in predictions if p["label"] == majority_class]

    if len(filtered) >= 2:
        result = ensemble_predictions(filtered)
        result["agreement_score"] = agreement_score
        result["n_filtered"] = len(filtered)
        result["n_original"] = len(predictions)
        return result

    result = ensemble_predictions(predictions)
    result["agreement_score"] = agreement_score
    result["n_filtered"] = len(predictions)
    result["n_original"] = len(predictions)
    return result
