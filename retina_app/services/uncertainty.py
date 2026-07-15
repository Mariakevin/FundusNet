"""Uncertainty quantification — MC Dropout for prediction confidence.

MC Dropout (Gal & Ghahramani, ICML 2016): preserved for uncertainty estimation.
"""

import logging

import numpy as np
import torch
import torch.nn.functional as F

from retina_app.constants import (
    CATEGORIES,
    ENABLE_MC_DROPOUT,
    MC_DROPOUT_PASSES,
    UNCERTAINTY_THRESHOLD,
)
from retina_app.services.transforms import TRANSFORM

logger = logging.getLogger("retina_app")


def _enable_dropout(model):
    dropout_layers = []
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()
            dropout_layers.append(module)
    return dropout_layers


def _disable_dropout(dropout_layers):
    for module in dropout_layers:
        module.eval()


def compute_entropy(probs):
    probs = np.clip(probs, 1e-10, 1.0)
    probs = probs / probs.sum()
    return float(-np.sum(probs * np.log(probs)))


def compute_prediction_entropy(probs):
    raw_entropy = compute_entropy(probs)
    max_entropy = np.log(len(probs))
    if max_entropy == 0:
        return 0.0
    return raw_entropy / max_entropy


def mc_dropout_forward_pass(model, image_tensor, n_passes=MC_DROPOUT_PASSES):
    from retina_app.services.model_manager import DEVICE

    all_probs = []
    dropout_layers = _enable_dropout(model)
    try:
        for _ in range(n_passes):
            with torch.no_grad():
                output = model(image_tensor.to(DEVICE))
                if isinstance(output, tuple):
                    output = output[0]
                probs = F.softmax(output, dim=1) if len(output.shape) > 1 else F.softmax(output.unsqueeze(0), dim=1)
                all_probs.append(probs.cpu().numpy().flatten())
    finally:
        model.eval()
        _disable_dropout(dropout_layers)
    if not all_probs:
        return np.zeros(len(CATEGORIES)), 1.0, True
    stacked = np.stack(all_probs, axis=0)
    mean_probs = np.mean(stacked, axis=0)
    mean_probs = mean_probs / mean_probs.sum()
    entropy = compute_prediction_entropy(mean_probs)
    is_uncertain = entropy > UNCERTAINTY_THRESHOLD
    return mean_probs, entropy, is_uncertain


def mc_dropout_single_model(model, image_path=None, n_passes=MC_DROPOUT_PASSES, input_tensor=None):
    from PIL import Image

    if input_tensor is None:
        if image_path is None:
            raise ValueError("Either image_path or input_tensor must be provided")
        with Image.open(image_path) as pil_img:
            input_tensor = TRANSFORM(pil_img.convert("RGB")).unsqueeze(0)
    mean_probs, entropy, is_uncertain = mc_dropout_forward_pass(model, input_tensor, n_passes)
    max_idx = int(np.argmax(mean_probs))
    return {
        "label": CATEGORIES[max_idx],
        "confidence": float(mean_probs[max_idx]),
        "probabilities": mean_probs.tolist(),
        "entropy": entropy,
        "is_uncertain": is_uncertain,
        "n_passes": n_passes,
    }


def mc_dropout_ensemble(models, image_path, model_weights=None, n_passes=MC_DROPOUT_PASSES):
    from PIL import Image

    from retina_app.constants import MODEL_WEIGHTS

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
    with Image.open(image_path) as pil_img:
        shared_tensor = TRANSFORM(pil_img.convert("RGB")).unsqueeze(0)
    all_model_results = []
    for model_type, model in models.items():
        try:
            result = mc_dropout_single_model(model, n_passes=n_passes, input_tensor=shared_tensor)
            result["weight"] = model_weights.get(model_type, 1.0 / len(models))
            result["model_type"] = model_type
            all_model_results.append(result)
        except Exception as exc:
            logger.warning("MC Dropout failed for %s: %s", model_type, exc)
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
    n_classes = len(CATEGORIES)
    weighted_probs = np.zeros(n_classes)
    total_weight = 0.0
    for result in all_model_results:
        weighted_probs += np.array(result["probabilities"]) * result["weight"]
        total_weight += result["weight"]
    if total_weight > 0:
        weighted_probs /= total_weight
    if weighted_probs.sum() > 0:
        weighted_probs /= weighted_probs.sum()
    max_idx = int(np.argmax(weighted_probs))
    entropy = compute_prediction_entropy(weighted_probs)
    return {
        "label": CATEGORIES[max_idx],
        "confidence": float(weighted_probs[max_idx]),
        "probabilities": weighted_probs.tolist(),
        "entropy": entropy,
        "is_uncertain": entropy > UNCERTAINTY_THRESHOLD,
        "n_models": len(all_model_results),
        "n_passes": n_passes,
        "individual_results": all_model_results,
    }


def is_dropout_enabled():
    return ENABLE_MC_DROPOUT
